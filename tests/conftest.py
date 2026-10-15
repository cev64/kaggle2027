"""Synthetic tracking data for tests: two plays (one per direction) with known truth."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdb27.tracking import FIELD_LENGTH, FIELD_WIDTH  # noqa: E402

N_FRAMES, SNAP, PASS = 30, 11, 20
LOS = {1: 40.0, 2: 65.0}  # standardized LOS per playId
DIRECTION = {1: "left", 2: "right"}


def make_play(game_id: int, play_id: int, direction: str, los: float, seed: int = 0) -> pd.DataFrame:
    """Constant-velocity play. Truth columns (x_true, y_true, dir_true) are in standardized coords."""
    rng = np.random.default_rng(seed + play_id)
    rows = []
    teams = [("OFF", 1000 + 100 * play_id, -1), ("DEF", 2000 + 100 * play_id, 1)]
    for club, base, sign in teams:
        for k in range(11):
            nfl = base + k
            x0 = los + sign * rng.uniform(1, 8)
            y0 = 4 + k * 4.4 + rng.uniform(-0.5, 0.5)
            vx = rng.uniform(2, 6) if club == "OFF" else rng.uniform(-3, 1)
            vx = 5.5 if (club, k) == ("OFF", 5) else vx  # guaranteed runner (orientation tests)
            vy = rng.uniform(-1.5, 1.5)
            s = float(np.hypot(vx, vy))
            d = float(np.degrees(np.arctan2(vx, vy)) % 360)  # clockwise from +y
            for f in range(1, N_FRAMES + 1):
                t = (f - SNAP) * 0.1  # x0/y0 are snap positions
                rows.append(dict(gameId=game_id, playId=play_id, nflId=nfl, displayName=f"P{nfl}", frameId=f,
                                 jerseyNumber=k + 1, club=club, x_true=x0 + vx * t, y_true=y0 + vy * t,
                                 s=s, a=0.0, dis=s * 0.1, dir_true=d, vx_true=vx, vy_true=vy))
    for f in range(1, N_FRAMES + 1):  # ball: still until snap, then moves +x at 8 yd/s
        bx = los + max(0, f - SNAP) * 0.8
        rows.append(dict(gameId=game_id, playId=play_id, nflId=None, displayName="football", frameId=f,
                         jerseyNumber=None, club="football", x_true=bx, y_true=FIELD_WIDTH / 2,
                         s=np.nan, a=np.nan, dis=np.nan, dir_true=np.nan, vx_true=np.nan, vy_true=np.nan))
    df = pd.DataFrame(rows)
    left = direction == "left"
    df["playDirection"] = direction
    df["x"] = FIELD_LENGTH - df["x_true"] if left else df["x_true"]
    df["y"] = FIELD_WIDTH - df["y_true"] if left else df["y_true"]
    df["dir"] = (df["dir_true"] + 180) % 360 if left else df["dir_true"]
    df["o"] = df["dir"]
    df["frameType"] = np.select([df.frameId < SNAP, df.frameId == SNAP], ["BEFORE_SNAP", "SNAP"], "AFTER_SNAP")
    df["event"] = np.select([df.frameId == SNAP, df.frameId == PASS], ["ball_snap", "pass_forward"], None)
    df["time"] = (pd.Timestamp("2024-09-08 13:00:00") + pd.to_timedelta((df.frameId + 100 * play_id) * 0.1, "s")).astype(str)
    return df


@pytest.fixture
def clean_pd() -> pd.DataFrame:
    return pd.concat([make_play(1, p, DIRECTION[p], LOS[p]) for p in (1, 2)], ignore_index=True)


@pytest.fixture
def plays() -> pl.DataFrame:
    return pl.DataFrame({"gameId": [1, 1], "playId": [1, 2], "possessionTeam": ["OFF", "OFF"],
                         "defensiveTeam": ["DEF", "DEF"],
                         "absoluteYardlineNumber": [FIELD_LENGTH - LOS[1], LOS[2]]})


def to_pl(df: pd.DataFrame) -> pl.DataFrame:
    from bdb27.tracking import normalize_columns
    return normalize_columns(pl.from_pandas(df.drop(columns=[c for c in df.columns if c.endswith("_true")])))


@pytest.fixture
def clean(clean_pd) -> pl.DataFrame:
    return to_pl(clean_pd)


@pytest.fixture
def dirty_pd(clean_pd) -> pd.DataFrame:
    """Clean data plus known anomalies (see ANOMALIES)."""
    df = clean_pd.copy()
    m = lambda nfl, f, p=1: (df.nflId == nfl) & (df.frameId == f) & (df.playId == p)  # noqa: E731
    df.loc[m(1101, 15), "x"] += 6.0                     # teleport spike
    df = df[~m(1102, 8)]                                # missing frame
    df = pd.concat([df, df[m(1103, 5)]])                # duplicate row
    df.loc[m(2105, 20), "s"] = 14.0                     # impossible speed
    df.loc[m(1104, 25), "y"] = 60.0                     # out of bounds (and a jump)
    sel = (df.nflId == 1205) & (df.playId == 2)
    df.loc[sel, "o"] = (df.loc[sel, "dir"] + 180) % 360  # flipped orientation
    df.loc[(df.playId == 2) & (df.frameId == 13), "event"] = "ball_snap"  # second snap
    df = df[~((df.playId == 2) & (df.nflId == 2210))]   # only 10 defenders
    return df.reset_index(drop=True)


@pytest.fixture
def dirty(dirty_pd) -> pl.DataFrame:
    return to_pl(dirty_pd)


# ----------------------------------------------------------------- BDB 2027 style
def cut_drill_truth(n: int = 50):
    """Accelerate along +x, brake into a 90-degree left cut, re-accelerate along +y (drill frame)."""
    t = np.arange(n) * 0.1
    s = np.select([t < 1.5, t < 2.0, t < 2.6, t < 2.9],
                  [4.5 * t, 6.75, 6.75 - (6.75 - 1.75) * (t - 2.0) / 0.6, 1.75],
                  np.minimum(1.75 + 5 * (t - 2.9), 7.0))
    head = np.clip((t - 2.6) / 0.3, 0, 1) * 90.0  # math degrees, CCW from +x
    v = np.c_[s * np.cos(np.radians(head)), s * np.sin(np.radians(head))]
    pos = np.vstack([[0, 0], np.cumsum(v[:-1] * 0.1, axis=0)])
    return t, s, head, pos


def make_combine(rot_deg: float = 30.0, origin=(30.0, 20.0), with_ball: bool = True) -> pd.DataFrame:
    """snake_case combine_tracking-like rows: one cut-drill attempt rotated/translated on the field."""
    t, s, head, pos = cut_drill_truth()
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = pos @ R.T + np.asarray(origin)
    d = (90 - (head + rot_deg)) % 360  # NGS: clockwise from +y
    times = pd.Timestamp("2024-03-01T15:00:00") + pd.to_timedelta(t, "s")
    iso = [ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" for ts in times]
    rows = pd.DataFrame({"draft_year": 2024, "event_id": 501, "nfl_id": 777, "entity_type": "PLAYER",
                         "time": iso, "drill_type": "SKILL_DRILLS_WR", "drill_name": "CUT_TEST", "attempt": 1,
                         "x": xy[:, 0], "y": xy[:, 1], "s": s, "a": np.abs(np.gradient(s, 0.1)),
                         "dis": np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)], "dir": d})
    if with_ball:
        ball = rows.assign(nfl_id=None, entity_type="BALL", x=origin[0] + 15, y=origin[1], s=0.0, a=0.0, dis=0.0, dir=np.nan)
        rows = pd.concat([rows, ball], ignore_index=True)
    return rows


def make_mirror(lag_s: float = 0.3, n: int = 80) -> pd.DataFrame:
    """Leader (1) shuffles laterally; follower (2) mirrors with a fixed lag."""
    t = np.arange(n) * 0.1
    out = []
    for nfl, shift in ((1, 0.0), (2, lag_s)):
        y = 2 * np.sin(2 * np.pi * 0.5 * (t - shift))
        vy = 2 * np.pi * 0.5 * 2 * np.cos(2 * np.pi * 0.5 * (t - shift))
        out.append(pd.DataFrame({"frameId": np.arange(1, n + 1), "nflId": nfl, "xd": 0.0, "yd": y,
                                 "s": np.abs(vy)}))
    return pd.concat(out, ignore_index=True)


@pytest.fixture
def combine_pd() -> pd.DataFrame:
    return make_combine()


@pytest.fixture
def game_sparse_pd(clean_pd) -> tuple[pd.DataFrame, pd.DataFrame]:
    """game_tracking-like (2 cohort players per play, no ball, no frame id / direction) + player_play."""
    keep = clean_pd[clean_pd.nflId.isin([1100, 2105, 1200, 2203])]
    trk = keep[["gameId", "playId", "nflId", "time", "x", "y", "s", "a", "dis", "o", "dir", "event"]].rename(
        columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id"}).copy()
    trk["nfl_id"] = trk["nfl_id"].astype(int)
    pp = (keep.groupby(["gameId", "playId", "nflId"], as_index=False)[["playDirection", "club"]].first()
          .rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
                           "playDirection": "play_direction", "club": "team_abbr"}))
    pp["possession_team"] = "OFF"
    pp["absolute_yardline_number"] = pp["play_id"].map({1: FIELD_LENGTH - LOS[1], 2: LOS[2]})
    return trk, pp
