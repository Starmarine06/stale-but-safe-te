"""Scale factor applied to each real trace so that the median ECMP max-util equals the target load."""
import numpy as np
from stale_but_safe.env import real
from stale_but_safe.env.traffic import TraceTraffic
rng = np.random.default_rng(0)
print("| network | nodes | intervals | native peak ECMP max-util (x scale = 1) median / p95 / max | scale factor for load 0.6 / 0.8 / 1.0 | max ECMP util after scaling to 0.8 |")
print("|---|---|---|---|---|---|")
for n in ("abilene", "geant", "germany17", "germany50", "brain"):
    topo, tr = real.load(n)
    from stale_but_safe.env.traffic import ecmp_splits
    x = tr[:, :, None] * ecmp_splits(topo)[None]
    u = (x.reshape(len(x), -1) @ topo.inc.T / topo.cap).max(1)
    sc = [TraceTraffic(topo, rng, tr, load=l).scale for l in (0.6, 0.8, 1.0)]
    print(f"| {n} | {topo.n} | {len(tr)} | {np.median(u):.3g} / {np.quantile(u, .95):.3g} / {u.max():.3g} | {sc[0]:.3g} / {sc[1]:.3g} / {sc[2]:.3g} | {(u * sc[1]).max():.2f} |")
