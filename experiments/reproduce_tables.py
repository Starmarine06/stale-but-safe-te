"""Regenerate the tables that are pure post-processing of committed raw result files and compare them with the committed copies.
Needs no real-trace data and no Topology Zoo download; takes about a minute.  Exit code 1 if any table differs.
  python experiments/reproduce_tables.py
Tables that need a re-run of simulations or emulation (final_eval.py, delay grid, guard audit, emulation analyses) are not covered here;
see docs/reproduction_check.md for what has and has not been re-run from scratch."""
import subprocess
import sys

PAIRS = [("analyze_retune.py", "table_retune.md"), ("analyze_v2_guard_stress.py", "table_v2_guard_stress.md"),
         ("ranking_stability.py", "table_ranking_stability.md"), ("analyze_gnn_vs_mlp.py", "table_gnn_vs_mlp.md"),
         ("analyze_ablations.py", "table_ablations.md"), ("analyze_seeds.py", "table_seeds.md"),
         ("analyze_realci_v2.py", "table_realci_v2.md"), ("analyze_dote.py", "table_dote.md"), ("analyze_ft.py", "table_ft.md")]


def dec(b):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("cp1252")          # some committed tables were written with the Windows code page


def norm(t):
    return [ln.rstrip() for ln in t.replace("\r", "").strip().split("\n")]


bad = 0
for script, table in PAIRS:
    out = subprocess.run([sys.executable, "experiments/" + script], capture_output=True).stdout
    ok = norm(dec(out)) == norm(dec(open("experiments/final/" + table, "rb").read()))
    bad += not ok
    print(f"{'IDENTICAL' if ok else 'DIFFERS  '}  {script:32s} -> {table}")
sys.exit(1 if bad else 0)
