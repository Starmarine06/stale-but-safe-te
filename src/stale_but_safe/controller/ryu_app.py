"""Ryu app: stats poller -> observation builder -> policy (+guard) -> select-group installer.

Env vars:
  SBS_SCENARIO  path to scenario.json         SBS_SHARED  shared dir (ready/start/out)
  SBS_POLICY    ecmp | spf | tuned:<profile> | model:<path>      (optionally prefixed guard<eps>@)
  SBS_PROFILE   ideal | rich | snmp | degraded   (telemetry impairment layered on real counters)
  SBS_TAG       run label, SBS_HYST  min per-demand split change (L1) to re-program a group

Measurement model: every `delta` seconds the controller reads port/flow counters (real OpenFlow
stats replies, with real jitter) giving true per-interval link utilisation and demand rates. Those
streams pass through the same `Channel` impairment model as the simulator (P, d, sigma, p), and the
policy only ever sees the impaired view.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3

from stale_but_safe.emu import scenario as S
from stale_but_safe.env.telemetry import PROFILES, Channel
from stale_but_safe.env.traffic import ecmp_splits

GROUP_BASE = 1000


class _Shim:
    """Minimal stand-in for TEEnv (policies only need .topo and link state)."""

    def __init__(self, topo):
        self.topo = topo
        self.link_up = np.ones(topo.E, bool)


def build_policy(spec: str, topo, shim):
    from stale_but_safe.baselines.heuristics import ECMP, SPF, TunedHeuristic
    guard_eps, guard_layers = None, (True, True, True)
    if spec.startswith("guard"):
        head, spec = spec.split("@", 1)
        if head.startswith("guardL2"):                      # e.g. guardL2-0.06@model:...  (pre-action check only)
            guard_eps, guard_layers = float(head[7:] or -0.06), (True, True, False)
        else:
            guard_eps = float(head[5:] or 0.05)
    if spec == "ecmp":
        pol = ECMP()
    elif spec == "spf":
        pol = SPF()
    elif spec.startswith("tuned:"):
        import yaml
        prof = spec.split(":", 1)[1]
        params = yaml.safe_load(open("/app/configs/tuned_heuristic.yaml"))[prof]["params"]
        pol = TunedHeuristic(**params)
    elif spec.startswith("model:"):
        from stale_but_safe.models.learned import LearnedPolicy, load
        m, c = load(spec.split(":", 1)[1])
        pol = LearnedPolicy(m, c, "gnn")
    else:
        raise ValueError(spec)
    if guard_eps is not None:
        from stale_but_safe.guard.guard import Guard, GuardCfg
        pol = Guard(pol, GuardCfg(eps=guard_eps, layers=guard_layers))
    pol.reset(shim)
    return pol


class TEApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        e = os.environ
        self.sc = S.load(e.get("SBS_SCENARIO", "/shared/scenario.json"))
        self.shared = e.get("SBS_SHARED", "/shared")
        self.tag = e.get("SBS_TAG", "run")
        self.hyst = float(e.get("SBS_HYST", "0.05"))
        self.topo = S.topo_from_dict(self.sc["topo"])
        self.out_port = {int(k): v for k, v in self.sc["ports"]["out_port"].items()}
        self.delta, self.steps = self.sc["delta"], self.sc["steps"]
        self.cap_mbps = self.topo.cap * self.sc["mbps_per_unit"]
        self.active = self.sc["active"]
        self.dps: dict[int, object] = {}
        self.port_tx: dict[tuple, int] = {}      # (dpid, port) -> (bytes, t)
        self.flow_tx: dict[int, tuple] = {}      # demand -> (bytes, t)
        self.cur_tx = {}
        self.cur_flow = {}
        self.replies = 0
        tele = PROFILES[e.get("SBS_PROFILE", "ideal")]
        rng = np.random.default_rng(self.sc["seed"] + 17)
        self.ch_util = Channel(self.topo.E, tele, rng)
        self.ch_dem = Channel(self.topo.D, tele, rng)
        self.shim = _Shim(self.topo)
        self.policy = build_policy(e.get("SBS_POLICY", "ecmp"), self.topo, self.shim)
        self.installed = ecmp_splits(self.topo)
        self.last_util = np.zeros(self.topo.E)
        self.missing = 0
        self.errors = 0
        self.prev_fb = 0
        self.log = []
        self.started = False
        self.loop_thread = hub.spawn(self._main)

    # ------------------------------------------------------------------ OpenFlow
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features(self, ev):
        dp = ev.msg.datapath
        self.dps[dp.id] = dp
        dp.send_msg(dp.ofproto_parser.OFPFlowMod(
            datapath=dp, command=dp.ofproto.OFPFC_DELETE, out_port=dp.ofproto.OFPP_ANY,
            out_group=dp.ofproto.OFPG_ANY, match=dp.ofproto_parser.OFPMatch()))
        dp.send_msg(dp.ofproto_parser.OFPGroupMod(
            datapath=dp, command=dp.ofproto.OFPGC_DELETE, group_id=dp.ofproto.OFPG_ALL))

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats(self, ev):
        dp, now = ev.msg.datapath, time.time()
        for st in ev.msg.body:
            self.cur_tx[(dp.id, st.port_no)] = (st.tx_bytes, now)
        self.replies += 1

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats(self, ev):
        now = time.time()
        for st in ev.msg.body:
            if st.priority != 10:
                continue
            m = st.match
            try:
                src, dst = m["ipv4_src"], m["ipv4_dst"]
            except KeyError:
                continue
            s, t = int(src.split(".")[-1]) - 1, int(dst.split(".")[-1]) - 1
            d = self.topo.demands.index((s, t)) if (s, t) in self.topo.demands else None
            if d is not None:
                self.cur_flow[d] = (st.byte_count, now)
        self.replies += 1

    # ------------------------------------------------------------------ install
    def _vid(self, d, p):
        return 1 + d * self.topo.k + p

    def _buckets(self, dp, d, splits_row):
        ofp, ps = dp.ofproto, dp.ofproto_parser
        out = []
        for p, w in enumerate(splits_row):
            wi = int(round(w * 100))
            if wi <= 0 or p >= len(self.topo.path_links[d]):
                continue
            first = self.topo.path_links[d][p][0]
            acts = [ps.OFPActionPushVlan(0x8100),
                    ps.OFPActionSetField(vlan_vid=(ofp.OFPVID_PRESENT | self._vid(d, p))),
                    ps.OFPActionOutput(self.out_port[first])]
            out.append(ps.OFPBucket(weight=wi, watch_port=ofp.OFPP_ANY, watch_group=ofp.OFPG_ANY, actions=acts))
        return out

    def _install_static(self):
        topo = self.topo
        for d in self.active:
            s, t = topo.demands[d]
            for p, ls in enumerate(topo.path_links[d]):
                nodes = [topo.links[ls[0]][0]] + [topo.links[l][1] for l in ls]
                vid = self._vid(d, p)
                for idx in range(1, len(nodes)):
                    sw = nodes[idx]
                    dp = self.dps[sw + 1]
                    ofp, ps = dp.ofproto, dp.ofproto_parser
                    match = ps.OFPMatch(vlan_vid=(ofp.OFPVID_PRESENT | vid))
                    if idx == len(nodes) - 1:
                        acts = [ps.OFPActionPopVlan(), ps.OFPActionOutput(1)]
                    else:
                        acts = [ps.OFPActionOutput(self.out_port[ls[idx]])]
                    dp.send_msg(ps.OFPFlowMod(datapath=dp, priority=20, match=match,
                                              instructions=[ps.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, acts)]))
        # ingress: group + flow per demand
        for d in self.active:
            s, t = topo.demands[d]
            dp = self.dps[s + 1]
            ofp, ps = dp.ofproto, dp.ofproto_parser
            dp.send_msg(ps.OFPGroupMod(datapath=dp, command=ofp.OFPGC_ADD, type_=ofp.OFPGT_SELECT,
                                       group_id=GROUP_BASE + d, buckets=self._buckets(dp, d, self.installed[d])))
        hub.sleep(1.0)
        for d in self.active:
            s, t = topo.demands[d]
            dp = self.dps[s + 1]
            ofp, ps = dp.ofproto, dp.ofproto_parser
            match = ps.OFPMatch(eth_type=0x0800, ipv4_src=S.host_ip(s), ipv4_dst=S.host_ip(t))
            dp.send_msg(ps.OFPFlowMod(datapath=dp, priority=10, match=match, instructions=[
                ps.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [ps.OFPActionGroup(GROUP_BASE + d)])]))

    def _apply(self, splits):
        """Re-program only demands whose split changed by more than `hyst` (L1)."""
        changed = 0
        for d in self.active:
            new = splits[d]
            if np.abs(new - self.installed[d]).sum() / 2 < self.hyst:
                continue
            s, _ = self.topo.demands[d]
            dp = self.dps[s + 1]
            ofp, ps = dp.ofproto, dp.ofproto_parser
            b = self._buckets(dp, d, new)
            if not b:
                continue
            dp.send_msg(ps.OFPGroupMod(datapath=dp, command=ofp.OFPGC_MODIFY, type_=ofp.OFPGT_SELECT,
                                       group_id=GROUP_BASE + d, buckets=b))
            q = np.round(new * 100) / 100
            self.installed[d] = q / q.sum() if q.sum() > 0 else new
            changed += 1
        return changed

    # ------------------------------------------------------------------ measurement
    def _poll(self):
        self.cur_tx, self.cur_flow, self.replies = {}, {}, 0
        n_sw = len(self.dps)
        ingress = {self.topo.demands[d][0] + 1 for d in self.active}
        want = n_sw + len(ingress)
        for dp in self.dps.values():
            ofp, ps = dp.ofproto, dp.ofproto_parser
            dp.send_msg(ps.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY))
            if dp.id in ingress:
                dp.send_msg(ps.OFPFlowStatsRequest(dp, 0, ofp.OFPTT_ALL, ofp.OFPP_ANY, ofp.OFPG_ANY, 0, 0,
                                                   ps.OFPMatch()))
        t0 = time.time()
        while self.replies < want and time.time() - t0 < 0.5 * self.delta:
            hub.sleep(0.005)

    def _rates(self):
        topo = self.topo
        util = self.last_util.copy()
        self.missing = 0
        for l in range(topo.E):
            a, _ = topo.links[l]
            key = (a + 1, self.out_port[l])
            if key in self.cur_tx and key in self.port_tx:
                (b1, t1), (b0, t0) = self.cur_tx[key], self.port_tx[key]
                if t1 > t0:
                    util[l] = (b1 - b0) * 8 / (t1 - t0) / 1e6 / self.cap_mbps[l]
                    continue
            self.missing += 1
        dem = np.zeros(topo.D)
        for d, (b1, t1) in self.cur_flow.items():
            if d in self.flow_tx and t1 > self.flow_tx[d][1]:
                b0, t0 = self.flow_tx[d]
                dem[d] = (b1 - b0) * 8 / (t1 - t0) / 1e6 / self.sc["mbps_per_unit"]
        have_base = bool(self.port_tx)
        self.last_util = util.copy()
        self.port_tx.update(self.cur_tx)
        self.flow_tx.update(self.cur_flow)
        return util, dem, have_base

    def _safe_act(self, obs):
        """Policy call that can never take the controller down: any exception, NaN or malformed
        proposal becomes ECMP and is counted (this is the L1 'malformed action' rule at the edge)."""
        ecmp = ecmp_splits(self.topo, self.topo.valid_paths(obs["link_up"]))
        try:
            out = np.asarray(self.policy.act(obs, self.shim), float)
            if out.shape != ecmp.shape or not np.isfinite(out).all() or (out < -1e-9).any():
                raise ValueError("malformed action")
            return out, 0
        except Exception as exc:                      # noqa: BLE001
            self.errors += 1
            self.get_logger().warning("policy error -> ECMP: %r", exc)
            return ecmp, 1

    # ------------------------------------------------------------------ main loop
    def _main(self):
        n = self.topo.n
        while len(self.dps) < n:
            hub.sleep(0.5)
        hub.sleep(2.0)
        self._install_static()
        hub.sleep(1.0)
        open(os.path.join(self.shared, "ready.flag"), "w").write("1")
        sp = os.path.join(self.shared, "start.json")
        while not os.path.exists(sp):
            hub.sleep(0.2)
        hub.sleep(0.1)
        start = json.load(open(sp))["start"]
        rows, primed = [], False
        for t in range(self.steps + 1):
            hub.sleep(max(0.0, start + t * self.delta - time.time()))
            tick0 = time.time()
            self._poll()
            util, dem, have_base = self._rates()
            row = {"tick": t, "interval": t - 1, "true_max_util": float("nan"), "true_mean_util": float("nan"),
                   "obs_max_util": float("nan"), "changed": 0, "infer_ms": 0.0, "latency_ms": 0.0,
                   "fallback": 0, "mean_age": float("nan")}
            if have_base:
                self.ch_util.push(util)
                self.ch_dem.push(dem)
                row["true_max_util"] = float(util.max())
                row["true_mean_util"] = float(util.mean())
                row["util_vec"] = np.round(util, 4).tolist()
                primed = True
            if t >= self.steps:
                rows.append(row)
                break
            if primed:
                u, ua = self.ch_util.observe()
                d, da = self.ch_dem.observe()
                obs = {"util": u, "util_age": ua, "demand": d, "demand_age": da,
                       "link_up": self.shim.link_up.copy(), "t": t}
                t1 = time.time()
                splits, err = self._safe_act(obs)
                row["infer_ms"] = (time.time() - t1) * 1000
                row["policy_error"] = err
                row["obs_max_util"] = float(u.max())
                row["mean_age"] = float(ua.mean())
                fb = int(getattr(self.policy, "stats", {}).get("fallback", 0))
                row["fallback"] = int(fb > self.prev_fb or err)     # 1 when this tick ran ECMP
                self.prev_fb = fb
                row["changed"] = self._apply(splits)
                row["missing_links"] = self.missing
                inner = getattr(self.policy, "policy", self.policy)
                if hasattr(inner, "feat"):
                    inner.feat.note_action(self.installed.copy())    # what is really in the dataplane
                row["latency_ms"] = (time.time() - tick0) * 1000
            rows.append(row)
        stats = getattr(self.policy, "stats", {})
        out = {"tag": self.tag, "rows": rows, "guard": stats, "policy_errors": self.errors, "policy": os.environ.get("SBS_POLICY"),
               "profile": os.environ.get("SBS_PROFILE")}
        json.dump(out, open(os.path.join(self.shared, f"ctrl_{self.tag}.json"), "w"))
        hub.sleep(1.0)
        os._exit(0)
