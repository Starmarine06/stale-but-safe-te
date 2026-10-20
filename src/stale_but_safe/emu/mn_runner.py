"""Mininet side of the emulation: builds the topology from the scenario, attaches
to the remote Ryu controller, replays the demand schedule and records goodput.

Run inside the mininet container (needs OVS + root):
  python3 -m stale_but_safe.emu.mn_runner --scenario /shared/scenario.json --ctrl ryu --shared /shared
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.log import setLogLevel

from . import scenario as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--ctrl", default="ryu")
    ap.add_argument("--shared", default="/shared")
    ap.add_argument("--nflows", type=int, default=16)
    ap.add_argument("--proto", choices=["udp", "tcp"], default="udp")
    a = ap.parse_args()
    setLogLevel("warning")
    sc = S.load(a.scenario)
    topo = S.topo_from_dict(sc["topo"])
    n, steps, delta, scale = topo.n, sc["steps"], sc["delta"], sc["mbps_per_unit"]
    ports = sc["ports"]["link_ports"]

    net = Mininet(controller=None, switch=lambda name, **kw: OVSSwitch(name, protocols="OpenFlow13", **kw),
                  link=TCLink, autoSetMacs=False, waitConnected=False)
    sw = [net.addSwitch(f"s{i}", dpid="%016x" % (i + 1)) for i in range(n)]
    hosts = [net.addHost(f"h{i}", ip=S.host_ip(i) + "/24", mac=S.host_mac(i)) for i in range(n)]
    for i in range(n):
        net.addLink(hosts[i], sw[i], port1=1, port2=1)
    for m, (a_, pa, b_, pb) in enumerate(ports):
        cap_mbps = float(topo.cap[2 * m]) * scale
        net.addLink(sw[a_], sw[b_], port1=pa, port2=pb, bw=cap_mbps, max_queue_size=100,
                    use_htb=True)
    ip = socket.gethostbyname(a.ctrl)
    c = net.addController("c0", controller=RemoteController, ip=ip, port=6653)
    net.start()
    for i in range(n):
        for j in range(n):
            if i != j:
                hosts[i].cmd(f"arp -s {S.host_ip(j)} {S.host_mac(j)}")
    # wait for controller to finish installing rules
    ready = os.path.join(a.shared, "ready.flag")
    t0 = time.time()
    while not os.path.exists(ready) and time.time() - t0 < 120:
        time.sleep(0.5)
    if not os.path.exists(ready):
        print("controller never became ready")
        net.stop()
        raise SystemExit(2)
    start = time.time() + 3.0
    json.dump({"start": start}, open(os.path.join(a.shared, "start.json"), "w"))

    procs, rxfiles = [], {}
    sched = sc["schedule"]
    for d in sc["active"]:
        s_, t_ = topo.demands[d]
        rates = [sched[i][d] * scale / S.WIRE_FACTOR for i in range(steps)]
        sp = os.path.join(a.shared, f"sched_{d}.json")
        json.dump(rates, open(sp, "w"))
        rx = os.path.join(a.shared, f"rx_{d}.json")
        rxfiles[d] = rx
        procs.append(hosts[t_].popen(["python3", "-m", f"stale_but_safe.emu.{a.proto}_tools", "recv",
                                      str(5000 + d), rx, str(delta), str(start), str(steps)]))
    time.sleep(1.0)
    for d in sc["active"]:
        s_, t_ = topo.demands[d]
        procs.append(hosts[s_].popen(["python3", "-m", f"stale_but_safe.emu.{a.proto}_tools", "send",
                                      S.host_ip(t_), str(5000 + d), os.path.join(a.shared, f"sched_{d}.json"),
                                      str(delta), str(start), str(a.nflows)]))
    time.sleep(max(0.0, start + steps * delta + 5 - time.time()))
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
    net.stop()
    json.dump({"done": True}, open(os.path.join(a.shared, "done.json"), "w"))


if __name__ == "__main__":
    main()
