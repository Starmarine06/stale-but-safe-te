"""Stdlib-only paced TCP sender / counting receiver run on Mininet hosts (packet-level validation).

send: python3 -m stale_but_safe.emu.tcp_tools send DST_IP PORT SCHEDULE.json DELTA START_EPOCH [NFLOWS]
recv: python3 -m stale_but_safe.emu.tcp_tools recv PORT OUT.json DELTA START_EPOCH STEPS
The sender opens NFLOWS long-lived TCP connections (distinct source ports, so OpenFlow select
groups can split them) and paces the schedule's rate over them as an application-limited
source; the kernel's congestion control decides what is delivered. Bytes the socket cannot
accept are not queued beyond a small credit cap, so a congested path shows up as a goodput
shortfall against the offered schedule.
"""
from __future__ import annotations

import json
import select
import socket
import sys
import time

CHUNK = 1448
CREDIT_CAP = 64 * CHUNK


def send(dst, port, sched_path, delta, start, nflows=16):
    rates = json.load(open(sched_path))          # payload Mbit/s per interval
    socks = []
    for _ in range(nflows):
        for attempt in range(40):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((dst, port))
                s.setblocking(False)
                socks.append(s)
                break
            except OSError:
                s.close()
                time.sleep(0.25)
    if not socks:
        return
    buf = b"x" * CHUNK
    tick, rr = 0.02, 0
    credit = [0.0] * len(socks)
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
        per = rate * 1e6 / 8 * tick / len(socks)
        for k, s in enumerate(socks):
            credit[k] = min(credit[k] + per, CREDIT_CAP)
            while credit[k] >= CHUNK:
                try:
                    n = s.send(buf)
                except (BlockingIOError, InterruptedError):
                    break
                except OSError:
                    credit[k] = 0.0
                    break
                credit[k] -= n
        nxt += tick
    for s in socks:
        s.close()


def recv(port, out, delta, start, steps):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", port))
    srv.listen(256)
    srv.setblocking(False)
    conns = []
    got = [0] * (steps + 2)
    t_end = start + steps * delta + 3
    while time.time() < t_end:
        r, _, _ = select.select([srv] + conns, [], [], 0.1)
        for s in r:
            if s is srv:
                c, _ = srv.accept()
                c.setblocking(False)
                conns.append(c)
                continue
            try:
                data = s.recv(65535)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                data = b""
            if not data:
                conns.remove(s)
                s.close()
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
