import numpy as np

from stale_but_safe.env.telemetry import Channel, TelemetryCfg


def stream(cfg, steps=60, seed=0):
    ch = Channel(5, cfg, np.random.default_rng(seed))
    out = []
    for t in range(steps):
        ch.push(np.full(5, 10.0 + t))
        out.append(ch.observe()[0].copy())
    return np.array(out)


def test_extensions_off_by_default():
    base = TelemetryCfg(5, 3, 0.2, 0.1)
    assert (base.outlier, base.round_drop, base.jitter) == (0.0, 0.0, 0.0)
    assert np.array_equal(stream(base), stream(TelemetryCfg(5, 3, 0.2, 0.1, 0.0, 0.0, 0.0)))


def test_outliers_stay_within_zero_to_two_times_the_reading():
    cfg = TelemetryCfg(1, 0, 0.0, 0.0, outlier=1.0)
    v = stream(cfg)
    truth = 10.0 + np.arange(60)[:, None]
    assert (v >= 0).all() and (v <= 2.0 * truth + 1e-9).all()
    assert not np.allclose(v, truth)


def test_lost_rounds_keep_stale_values():
    cfg = TelemetryCfg(1, 0, 0.0, 0.0, round_drop=1.0)
    v = stream(cfg)
    assert np.allclose(v[1:], v[0])          # after the first poll nothing ever arrives


def test_jitter_keeps_poll_gaps_within_one_interval():
    cfg = TelemetryCfg(5, 0, 0.0, 0.0, jitter=1.0)
    ch = Channel(1, cfg, np.random.default_rng(1))
    polls = []
    for t in range(200):
        before = ch.next_poll
        ch.push(np.array([float(t)]))
        if ch.age[0] == 0:
            polls.append(t)
    gaps = np.diff(polls)
    assert gaps.min() >= 4 and gaps.max() <= 6 and len(set(gaps)) > 1
