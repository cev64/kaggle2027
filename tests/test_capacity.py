import numpy as np
import pandas as pd
import pytest

from bdb27 import capacity as K


def synthetic_forty(vmax=10.0, tau=1.1, t0=-0.15, angle=20.0, noise=0.0, seed=0):
    """A 40 attempt along a rotated axis; the clip starts 0.15 s after first movement."""
    t = np.arange(0, 6.0, 0.1)
    s = K.sprint_speed(t, vmax, tau, t0)
    d = K.sprint_distance(np.clip(t - t0, 0, None), vmax, tau) - K.sprint_distance(-t0, vmax, tau)
    keep = d <= 40.5
    th = np.radians(angle)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"eventId": "e1", "nflId": 1, "frameId": np.arange(keep.sum()) + 1,
                         "x": 30 + d[keep] * np.cos(th), "y": 20 + d[keep] * np.sin(th),
                         "s": s[keep] + rng.normal(0, noise, keep.sum())})


def test_fit_recovers_parameters():
    out = K.attempt_profile(synthetic_forty())
    assert out["vmax"] == pytest.approx(10.0, rel=1e-3) and out["tau"] == pytest.approx(1.1, rel=1e-3)
    assert out["t0"] == pytest.approx(-0.15, abs=1e-3) and out["r2"] > 0.999
    assert out["a0"] == pytest.approx(10 / 1.1, rel=1e-3)
    assert out["d90"] == pytest.approx(10 * 1.1 * (np.log(10) - 0.9), rel=1e-3)
    assert 39.0 < out["runLength"] <= 40.5
    # splits measured from first movement despite the late clip start
    t2, t10 = (K.time_to_distance(m, 10.0, 1.1) for m in (2, 10))
    assert out["dPreClip"] == pytest.approx(K.sprint_distance(0.15, 10.0, 1.1), rel=1e-3)
    assert out["split2_10"] == pytest.approx(t10 - t2, abs=0.01)
    assert out["v10rel"] == pytest.approx(1 - np.exp(-t10 / 1.1), abs=0.01)


def test_model_times_are_consistent():
    t40 = K.time_to_distance(40, 10.0, 1.1)
    assert K.sprint_distance(t40, 10.0, 1.1) == pytest.approx(40)
    assert K.time_to_distance(10, 10.0, 1.1) < t40


def test_noisy_fit_and_player_aggregation():
    a = K.profile_attempts(pd.concat([synthetic_forty(noise=0.1, seed=s).assign(eventId=f"e{s}") for s in range(3)]))
    a["usable"] = K.flag_fits(a, min_run=30)
    assert a["usable"].all() and np.allclose(a["vmax"], 10, atol=0.3)
    cap = K.add_si(K.player_capacity(a).assign(combineWeight=200.0))
    assert cap["nReps"].iloc[0] == 3
    assert cap["f0"].iloc[0] == pytest.approx(cap["f0Rel"].iloc[0] * 200 * K.LB)
