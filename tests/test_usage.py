import numpy as np
import polars as pl
import pytest

from bdb27 import usage as U


def route_play(speed=lambda t: np.minimum(8.0 * t, 9.0), n_pre=10, n_post=40, pass_at=30, gid=1, pid=1, nfl=7):
    """Straight-line route along +x; reported s = speed(t); pass_forward at frame pass_at after the snap."""
    f = np.arange(-n_pre, n_post + 1)
    t = np.clip(f, 0, None) * 0.1
    s = speed(t) * (f >= 0)
    x = 30 + np.concatenate([[0], np.cumsum((s[1:] + s[:-1]) / 2 * 0.1)])
    ev = pl.Series([("ball_snap" if k == 0 else "pass_forward" if k == pass_at else None) for k in f], dtype=pl.String)
    return pl.DataFrame({"gameId": gid, "playId": pid, "nflId": nfl, "frameId": f + n_pre + 1, "framesFromSnap": f,
                         "x": x, "y": 20.0, "s": s, "event": ev})


def test_window_ends_at_pass_and_measures_speed():
    out = U.route_windows(route_play())
    r = out.row(0, named=True)
    assert r["windowEnd"] == "pass" and r["tWindow"] == pytest.approx(3.0)
    assert r["peakSpeed"] == pytest.approx(9.0) and r["tPeak"] <= 3.0
    # constant 8 yd/s^2 then 9 yd/s: distance 2 yd at t=sqrt(0.5); 10 yd at 1.125 + (10-5.0625)/9
    t2, t10 = np.sqrt(0.5), 1.125 + (10 - 5.0625) / 9
    assert r["tPath2"] == pytest.approx(t2, abs=0.03) and r["tPath10"] == pytest.approx(t10, abs=0.03)
    assert r["burst"] == pytest.approx(8 / (t10 - t2), rel=0.03)


def test_no_pass_window_capped_and_capacity_fraction():
    play = route_play(pass_at=999, n_post=80)
    out = U.route_windows(play, max_window=4.0)
    assert out["windowEnd"][0] == "cap" and out["tWindow"][0] == pytest.approx(4.0)
    cap = pl.DataFrame({"nflId": [7], "vmax": [10.0], "burst2_10": [7.0]})
    u = U.attach_capacity(out, cap)
    assert u["speedFrac"][0] == pytest.approx(0.9)
