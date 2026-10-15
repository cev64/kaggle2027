import numpy as np
import pandas as pd
import pytest
from conftest import cut_drill_truth, make_combine, make_mirror

from bdb27 import drills as D
from bdb27 import features as F
from bdb27 import tracking as T


def frame(rows):
    df = pd.DataFrame(rows, columns=["nflId", "side", "x", "y", "vx", "vy"])
    return df.assign(gameId=1, playId=1, frameId=1, isBall=df["side"] == "ball")


def test_nearest_players():
    f = frame([(1, "offense", 10, 10, 0, 0), (2, "offense", 10, 14, 0, 0),
               (3, "defense", 13, 10, 0, 0), (4, "defense", 30, 30, 0, 0), (None, "ball", 10, 11, 0, 0)])
    out = F.nearest_players(f).set_index("nflId")
    assert out.loc[1, "distNearestOpp"] == pytest.approx(3) and out.loc[1, "nearestOppId"] == 3
    assert out.loc[1, "distNearestMate"] == pytest.approx(4) and out.loc[1, "nearestMateId"] == 2
    assert out.loc[3, "nearestOppId"] == 1 and len(out) == 4  # ball excluded
    sep = F.receiver_separation(f)
    assert set(sep["nflId"]) == {1, 2} and sep.set_index("nflId").loc[2, "separation"] == pytest.approx(5)


def test_leverage():
    f = frame([(1, "offense", 50, 10, 0, 0), (3, "defense", 52, 13, 0, 0),
               (2, "offense", 50, 45, 0, 0), (4, "defense", 51, 47, 0, 0)])
    lev = F.leverage(f).set_index("nflId")
    assert lev.loc[1, "leverageSide"] == "inside" and lev.loc[1, "leverageAngle"] == pytest.approx(np.degrees(np.arctan2(3, 2)))
    assert lev.loc[2, "leverageSide"] == "outside" and lev.loc[2, "leverageAngle"] < 0


def test_voronoi_partitions_field():
    f = frame([(1, "offense", 30, 26.67, 0, 0), (2, "defense", 90, 26.67, 0, 0)])
    a = F.voronoi_areas(f)
    assert a.sum() == pytest.approx(120 * 160 / 3, rel=1e-4) and a.iloc[0] == pytest.approx(a.iloc[1], rel=1e-4)
    rng = np.random.default_rng(1)
    many = frame([(i, "offense" if i < 11 else "defense", *rng.uniform([0, 0], [120, 53.3]), 0, 0) for i in range(22)])
    assert F.voronoi_areas(many).sum() == pytest.approx(120 * 160 / 3, rel=1e-4)


def test_pitch_control():
    f = frame([(1, "offense", 30, 20, 3, 0), (2, "defense", 60, 35, -3, 0), (None, "ball", 30, 20, 0, 0)])
    X, Y, c = F.pitch_control(f, step=1.0)
    assert c.shape == X.shape and ((c > 0) & (c < 1)).all()
    at = lambda x, y: c[np.argmin(np.abs(Y[:, 0] - y)), np.argmin(np.abs(X[0] - x))]  # noqa: E731
    assert at(31, 20) > 0.6 and at(59, 35) < 0.4
    infl = F.player_influence(np.array([[0.0, 0.0]]), np.array([[8.0, 0.0]]), None, np.array([[4.0, 0.0], [0.0, 4.0]]))
    assert infl[0, 0] > infl[0, 1]  # stretched along direction of travel


def test_closing_speed():
    assert F.closing_speed([0, 0], [5, 0], [10, 0], [-5, 0]) == pytest.approx(10)
    assert F.closing_speed([0, 0], [-5, 0], [10, 0], [5, 0]) == pytest.approx(-10)
    assert F.closing_speed([0, 0], [0, 5], [10, 0], [0, 5]) == pytest.approx(0)
    play = pd.DataFrame({"nflId": [1, 1, 2, 2], "frameId": [1, 2, 1, 2], "x": [0, 0.5, 10, 9.5], "y": 0.0,
                         "vx": [5, 5, -5, -5], "vy": 0.0})
    pc = F.pair_closing(play, 1, 2)
    assert pc["distance"].tolist() == pytest.approx([10, 9]) and pc["closingSpeed"].tolist() == pytest.approx([10, 10])


# ---------------------------------------------------------------- drill features
def test_speed_profile_and_cut():
    t, s, head, pos = cut_drill_truth()
    traj = pd.DataFrame({"frameId": np.arange(1, len(t) + 1), "xd": pos[:, 0], "yd": pos[:, 1], "s": s})
    prof = D.speed_profile(traj)
    assert prof["peakSpeed"] == pytest.approx(7, abs=0.1)
    assert prof["tTo5"] == pytest.approx(5 / 4.5 - 0.2, abs=0.15)  # from motion onset (s > 0.5)
    assert prof["peakDecel"] == pytest.approx(8.3, abs=1.0) and prof["peakAccel"] == pytest.approx(5, abs=0.5)
    cuts = D.detect_cuts(traj)
    assert len(cuts) == 1
    c = cuts.iloc[0]
    assert c["turnAngle"] == pytest.approx(90, abs=5) and c["minSpeed"] == pytest.approx(1.75, abs=0.15)
    assert c["entrySpeed"] == pytest.approx(6.75, abs=0.15) and c["tCut"] == pytest.approx(2.75, abs=0.2)
    assert c["meanDecel"] == pytest.approx(8.3, abs=2.0) and c["decelDist"] > 1
    assert c["timeToRegain"] == pytest.approx((0.9 * 6.75 - 1.75) / 5 + 0.15, abs=0.25)


def test_drill_summary_on_combine_file():
    df = T.prepare_tracking(T.normalize_columns(__import__("polars").from_pandas(make_combine())))
    summ = D.drill_summary(df)
    assert len(summ) == 1 and summ["nCuts"][0] == 1 and summ["drillName"][0] == "CUT_TEST"
    assert summ["cut1_turnAngle"][0] == pytest.approx(90, abs=5)


def test_mirror_reaction_lag():
    m = make_mirror(lag_s=0.3)
    lead, follow = m[m.nflId == 1], m[m.nflId == 2]
    r = D.reaction_lag(lead, follow, axis="y")
    assert r["lag"] == pytest.approx(0.3) and r["corr"] > 0.95
    fl = D.flip_latencies(lead, follow, axis="y")
    assert len(fl) >= 5 and np.allclose(fl["latency"], 0.3)


def test_response_after_event():
    n = 40
    s = np.r_[np.full(20, 1.0), 1.0 + 4.0 * np.arange(1, 21) * 0.1]  # accelerates from frame 21
    traj = pd.DataFrame({"frameId": np.arange(1, n + 1), "x": np.cumsum(s) * 0.1, "y": 0.0, "s": s})
    r = D.response_after_event(traj, event_frame=20, window_s=1.5)
    assert r["tSpeedGain"] == pytest.approx(0.3, abs=0.15) and r["peakAccel"] == pytest.approx(4, abs=0.6)
