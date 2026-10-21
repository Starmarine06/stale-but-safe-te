"""NOTE: the 'raw mu' column evaluates the raw proposal on the guard's own trajectory, so guarded-vs-raw here understates the guard's cost; use experiments/guard_regret_sim_vs_emu.py for guarded vs unguarded trajectories.
Does the emulation-style (carried, capped) telemetry bias the guard's pre-action check? Replays each TCP scenario in the
simulator with carried_obs on/off and audits guard decisions against the true-demand counterfactual."""
import sys
from pathlib import Path
import numpy as np
from stale_but_safe.baselines.heuristics import Policy
from stale_but_safe.emu import scenario as S
from stale_but_safe.env.te_env import TEEnv
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

NAMES = ["s1", "s2", "s3", "n14", "n15", "n16", "n17", "n18", "n19"]
GUARDS = {"guard0.05": GuardCfg(eps=0.05), "guardL2-0.06": GuardCfg(eps=-0.06, layers=(True, True, False))}


class Spy(Policy):
    name = "spy"
    def __init__(self, inner): self.inner, self.last = inner, None
    def reset(self, env): super().reset(env); self.inner.reset(env)
    def act(self, obs, env):
        self.last = np.asarray(self.inner.act(obs, env), float); return self.last


def mu(topo, dem, sp, up):
    return float((topo.inc @ (dem[:, None] * sp).ravel() / topo.cap)[up].max())


def run(n, carried, gcfg):
    d = Path("runs/emu_tcp") / n
    sc = S.load(d / "scenario.json")
    mp = (d / "model.txt").read_text().strip() if (d / "model.txt").exists() else f"runs/ours_dfm_final_{n}.pt"
    m, c = load(mp)
    topo = S.topo_from_dict(sc["topo"])
    env = TEEnv(topo, S.sim_cfg(sc, PROFILES["degraded"], carried=carried), sc["seed"])
    obs = env.reset(sc["seed"])
    spy = Spy(LearnedPolicy(m, c, "gnn")); g = Guard(spy, gcfg); g.reset(env)
    rec, done = [], False
    while not done:
        dem, up = env.peek_demand(), env.link_up.copy()
        act = g.act(obs, env)
        ec = ecmp_splits(topo, topo.valid_paths(up))
        prop = spy.last if spy.last is not None else act
        fb = spy.last is None or not np.allclose(act, spy.last, atol=1e-9)
        rec.append((mu(topo, dem, prop, up), mu(topo, dem, act, up), mu(topo, dem, ec, up), fb))
        spy.last = None
        obs, _, done, _ = env.step(act)
    return np.array(rec, float)[3:]


print("| guard | counters | fallback | false rejection | harm caught | guarded mu | raw mu | ecmp mu |\n|---|---|---|---|---|---|---|---|")
for gn, gcfg in GUARDS.items():
    for carried in (False, True):
        R = np.concatenate([run(n, carried, gcfg) for n in NAMES])
        mp_, ma, me, fb = R[:, 0], R[:, 1], R[:, 2], R[:, 3].astype(bool)
        ben, harm = mp_ < 0.99 * me, mp_ > 1.01 * me
        fr = lambda a, b: a.sum() / b.sum() if b.sum() else float("nan")
        print(f"| {gn} | {'carried (emu-like)' if carried else 'offered'} | {100*fb.mean():.0f}% | {100*fr(fb&ben, fb):.0f}% "
              f"| {100*fr(fb&harm, harm):.0f}% | {np.minimum(ma,1).mean():.3f} | {np.minimum(mp_,1).mean():.3f} | {np.minimum(me,1).mean():.3f} |", flush=True)
