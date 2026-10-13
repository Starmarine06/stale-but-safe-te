"""Host-side orchestrator: one Ryu container + one Mininet container per run.

  python -m stale_but_safe.emu.run_emulation --scenario runs/emu/scenario.json \
      --run ecmp:ideal --run tuned:snmp --run model:runs/ours_v2_s0.pt:snmp ...

Run spec = <policy spec>:<profile>, where policy spec is as in controller/ryu_app.py
(e.g. ecmp, tuned:snmp, model:<path>, guard0.05@model:<path>). The profile is the last ':' field.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sh(*args, check=True, **kw):
    return subprocess.run(list(args), check=check, capture_output=True, text=True, **kw)


def tag_of(spec: str, profile: str) -> str:
    return (spec.replace(":", "-").replace("@", "_").replace("/", "_").replace(".pt", "")
            + "__" + profile)


def run_one(root: Path, scenario: Path, spec: str, profile: str, outdir: Path, net="sbsnet", timeout=400, hyst=0.05):
    tag = tag_of(spec, profile)
    shared = outdir / tag
    if shared.exists():
        shutil.rmtree(shared)
    shared.mkdir(parents=True)
    shutil.copy(scenario, shared / "scenario.json")
    ryu_name = f"sbs-ryu-{os.getpid()}"
    models = (root / "runs").resolve()
    model_spec = spec
    # map host model path -> /models/<file> inside the container
    if "model:" in spec:
        head, path = spec.split("model:", 1)
        model_spec = head + "model:/models/" + Path(path).name
    env = {"SBS_SCENARIO": "/shared/scenario.json", "SBS_SHARED": "/shared", "SBS_POLICY": model_spec,
           "SBS_PROFILE": profile, "SBS_TAG": tag, "SBS_HYST": str(hyst), "PYTHONPATH": "/app/src"}
    ea = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
    sh("docker", "rm", "-f", ryu_name, check=False)
    sh("docker", "run", "-d", "--rm", "--name", ryu_name, "--network", net, "-v", f"{shared}:/shared",
       "-v", f"{models}:/models:ro", *ea, "sbs-ryu", "ryu-manager", "--ofp-tcp-listen-port", "6653",
       "/app/src/stale_but_safe/controller/ryu_app.py")
    t0 = time.time()
    try:
        r = sh("docker", "run", "--rm", "--privileged", "--network", net, "-v", f"{shared}:/shared",
               "-e", "PYTHONPATH=/app/src", "sbs-mininet", "python3", "-m", "stale_but_safe.emu.mn_runner",
               "--scenario", "/shared/scenario.json", "--ctrl", ryu_name, "--shared", "/shared",
               check=False, timeout=timeout)
        (shared / "mininet.log").write_text(r.stdout + "\n" + r.stderr)
        for _ in range(20):
            if (shared / f"ctrl_{tag}.json").exists():
                break
            time.sleep(1)
    finally:
        logs = sh("docker", "logs", ryu_name, check=False)
        (shared / "ryu.log").write_text(logs.stdout + logs.stderr)
        sh("docker", "rm", "-f", ryu_name, check=False)
    ok = (shared / f"ctrl_{tag}.json").exists()
    print(f"[{tag}] ok={ok} {time.time() - t0:.0f}s", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="runs/emu")
    ap.add_argument("--hyst", type=float, default=0.05, help="controller per-demand update threshold")
    a = ap.parse_args()
    root = Path.cwd()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sh("docker", "network", "create", "sbsnet", check=False)
    for r in a.run:
        spec, profile = r.rsplit(":", 1)
        run_one(root, Path(a.scenario).resolve(), spec, profile, out, hyst=a.hyst)


if __name__ == "__main__":
    main()
