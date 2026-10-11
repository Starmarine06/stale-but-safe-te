"""Stdlib-only paced UDP sender / counting receiver run on Mininet hosts.

send: python3 -m stale_but_safe.emu.udp_tools send DST_IP PORT SCHEDULE.json DELTA START_EPOCH [NFLOWS]
recv: python3 -m stale_but_safe.emu.udp_tools recv PORT OUT.json DELTA START_EPOCH STEPS
The sender splits its rate over NFLOWS UDP sockets (distinct source ports) so that
OpenFlow select groups, which hash per flow, can realise fractional splits.
"""
from __future__ import annotations

import json
import socket
import sys
import time

PAYLOAD = 1000


def send(dst, port, sched_path, delta, start, nflows=16):
    rates = json.load(open(sched_path))          # payload Mbit/s per interval
    socks = []
    for _ in range(nflows):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 0))
        socks.append(s)
    buf = b"x" * PAYLOAD
    tick, credit, rr = 0.02, 0.0, 0
    t_end = start + len(rates) * delta
    nxt = max(start, time.time())
    while True:
        now = time.time()
        if now >= t_end:
            break
        if now < nxt:
            time.sleep(min(nxt - now, tick))
            continue
        i = min(int((now - start) / delta), len(rates) - 1) if now >= start else 0
        rate = rates[i] if now >= start else 0.0
        credit += rate * 1e6 / 8 * tick
        while credit >= PAYLOAD:
            try:
                socks[rr % nflows].sendto(buf, (dst, port))
            except OSError:
                pass
            rr += 1
            credit -= PAYLOAD
        nxt += tick


def recv(port, out, delta, start, steps):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    s.bind(("", port))
    s.settimeout(0.2)
    got = [0] * (steps + 2)
    t_end = start + steps * delta + 3
    while time.time() < t_end:
        try:
            data = s.recv(65535)
        except socket.timeout:
            continue
        i = int((time.time() - start) / delta)
        if 0 <= i < len(got):
            got[i] += len(data)
    json.dump(got, open(out, "w"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "send":
        send(sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), float(sys.argv[6]),
             int(sys.argv[7]) if len(sys.argv) > 7 else 16)
    else:
        recv(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
