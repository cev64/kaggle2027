import numpy as np
import pandas as pd
import polars as pl
import pytest
from conftest import LOS, SNAP, make_combine, to_pl

from bdb27 import tracking as T


def truth(clean_pd):
    return clean_pd[["playId", "nflId", "frameId", "x_true", "y_true", "dir_true", "vx_true", "vy_true"]]


def prep(clean, plays):
    return T.prepare_tracking(clean, plays).to_pandas()


def test_flip_math_unit():
    df = pl.DataFrame({"gameId": [1, 1], "playId": [1, 1], "nflId": [1, 2], "frameId": [1, 1],
                       "playDirection": ["left", "right"], "x": [30.0, 30.0], "y": [10.0, 10.0],
                       "o": [90.0, 90.0], "dir": [350.0, 350.0]})
    out = T.standardize_direction(df).sort("nflId")
    assert out["x"].to_list() == pytest.approx([90.0, 30.0])
    assert out["y"].to_list() == pytest.approx([T.FIELD_WIDTH - 10, 10.0], abs=1e-4)
    assert out["o"].to_list() == pytest.approx([270.0, 90.0])
    assert out["dir"].to_list() == pytest.approx([170.0, 350.0])  # wraps mod 360
    # A raw left-play heading of 270 (moving -x, i.e. toward the offense's goal) becomes +x.
    one = T.standardize_direction(df.with_columns(dir=pl.lit(270.0))).filter(pl.col("nflId") == 1)
    assert np.sin(np.radians(one["dir"][0])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="already standardized"):
        T.standardize_direction(out)


def test_standardized_matches_truth(clean, plays, clean_pd):
    m = prep(clean, plays).merge(truth(clean_pd), on=["playId", "nflId", "frameId"])
    assert np.allclose(m["x"], m["x_true"], atol=1e-3) and np.allclose(m["y"], m["y_true"], atol=1e-3)
    pl_ = m[m.nflId.notna()]
    assert np.allclose((pl_["dir"] - pl_["dir_true"] + 180) % 360 - 180, 0, atol=1e-3)
    assert m.groupby("playId")["flipped"].first().to_dict() == {1: True, 2: False}


def test_velocity_consistent_with_positions(clean, plays, clean_pd):
    """vx/vy from (s, dir) must agree with finite differences of standardized x/y, in both directions."""
    df = prep(clean, plays)
    df = df[~df.isBall].sort_values(["playId", "nflId", "frameId"])
    fd = df.groupby(["playId", "nflId"])[["x", "y"]].diff() / 0.1
    ok = fd["x"].notna()
    assert np.allclose(df.loc[ok, "vx"], fd.loc[ok, "x"], atol=1e-2)
    assert np.allclose(df.loc[ok, "vy"], fd.loc[ok, "y"], atol=1e-2)
    off = df[df.side == "offense"]
    assert (off["vx"] > 0).all()  # offense moves left->right after standardization
    assert np.allclose(df[["ax", "ay"]].fillna(0), 0, atol=1e-2)  # constant velocity


def test_ball_snap_los(clean, plays):
    df = prep(clean, plays)
    ball = df[df.isBall]
    assert len(ball) == 60 and ball["nflId"].isna().all() and (ball["side"] == "ball").all()
    assert (ball["vx"].dropna() >= 0).all()  # ball velocity from positions (no s/dir)
    assert df.groupby("playId")["losX"].first().to_dict() == pytest.approx(LOS)
    assert (df.groupby("playId")["losSource"].first() == "ball").all()
    snap = df[df.frameId == SNAP]
    assert (snap["framesFromSnap"] == 0).all()
    assert (snap.loc[snap.side == "offense", "xRelLos"] < 0).all()
    assert (snap.loc[snap.side == "defense", "xRelLos"] > 0).all()


def test_los_from_yardline_without_ball(clean, plays):
    df = T.prepare_tracking(clean.filter(~pl.col("isBall")), plays).to_pandas()
    assert df.groupby("playId")["losX"].first().to_dict() == pytest.approx(LOS)
    assert (df["losSource"] == "yardline").all()


def test_load_tracking_csv_cache_and_errors(tmp_path, clean_pd):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="download the competition data"):
        T.load_tracking(raw, tmp_path / "cache")
    cols = [c for c in clean_pd.columns if not c.endswith("_true")]
    clean_pd[cols].to_csv(raw / "tracking_week_1.csv", index=False, na_rep="NA")
    df = T.load_tracking(raw, tmp_path / "cache")
    assert (tmp_path / "cache" / "tracking_week_1.parquet").exists()
    assert df["week"].unique().to_list() == [1] and df.schema["x"] == pl.Float32
    assert df["isBall"].sum() == 60


def test_load_2026_snake_case(tmp_path, clean_pd):
    snake = {"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id", "frameId": "frame_id",
             "playDirection": "play_direction", "displayName": "player_name"}
    d = clean_pd[clean_pd.nflId.notna()][list(snake) + ["x", "y", "s", "a", "o", "dir"]].rename(columns=snake)
    (tmp_path / "train").mkdir()
    d.to_csv(tmp_path / "train" / "input_2023_w01.csv", index=False)
    df = T.load_tracking(tmp_path, None)
    assert {"gameId", "playId", "nflId", "frameId", "playDirection", "displayName", "week"} <= set(df.columns)


def test_stitch_2026():
    inp = pl.DataFrame({"gameId": [1, 1], "playId": [1, 1], "nflId": [5, 5], "frameId": [1, 2],
                        "x": [1.0, 2.0], "y": [1.0, 1.0], "playDirection": ["right"] * 2})
    out = pl.DataFrame({"gameId": [1], "playId": [1], "nflId": [5], "frameId": [1], "x": [3.0], "y": [1.0]})
    s = T.stitch_2026(inp, out)
    assert s["frameId"].to_list() == [1, 2, 3] and s["playDirection"].to_list() == ["right"] * 3


def test_game_tracking_2027_sparse(tmp_path, game_sparse_pd, clean_pd):
    """Cohort-only game tracking: no frame id, ball or direction; direction comes from player_play."""
    trk, pp = game_sparse_pd
    trk.to_csv(tmp_path / "game_tracking_2024.csv", index=False)
    pp.to_csv(tmp_path / "player_play.csv", index=False)
    df = T.load_tracking(tmp_path, None)
    assert df["season"].unique().to_list() == [2024]
    assert df.group_by("playId").agg(pl.col("frameId").min())["frameId"].to_list() == [1, 1]
    out = T.prepare_tracking(df, T.load_table("player_play", tmp_path)).to_pandas()
    m = out.merge(truth(clean_pd), on=["playId", "nflId", "frameId"])
    assert len(m) == len(out) and np.allclose(m["x"], m["x_true"], atol=1e-3)
    assert set(out["side"]) == {"offense", "defense"}
    assert out.groupby("playId")["losX"].first().to_dict() == pytest.approx(LOS)


def test_combine_load_and_drill_normalization(tmp_path):
    make_combine().to_csv(tmp_path / "combine_tracking.csv", index=False)
    df = T.load_tracking(tmp_path, tmp_path / "cache")
    assert T.play_key(df) == ["draftYear", "eventId"]
    assert df["frameId"].min() == 1 and df["frameId"].max() == 50 and df["isBall"].sum() == 50
    out = T.prepare_tracking(df).filter(~pl.col("isBall")).sort("frameId").to_pandas()
    assert out[["xd", "yd"]].iloc[0].abs().max() < 1e-4
    assert abs(out["yd"].iloc[15]) < 1e-3 and out["xd"].iloc[15] > 3  # first leg along +x
    assert out["yd"].iloc[-1] > 5  # left cut stays left after rotation
    assert out["dird"].iloc[5] == pytest.approx(90, abs=1e-3)
    assert out["dird"].iloc[-1] == pytest.approx(0, abs=1e-3)  # final leg = +y = 0 deg (NGS)
    assert out["drillRotation"].iloc[0] == pytest.approx(-30, abs=1e-3)


def test_drill_mirror():
    df = T.normalize_columns(pl.from_pandas(make_combine(with_ball=False)))
    df = T.add_frame_ids(df).with_columns(y=60 - pl.col("y"), dir=(180 - pl.col("dir")) % 360)  # reflected cut
    out = T.normalize_drill(df, mirror=True).sort("frameId")
    assert out["drillMirrored"].all() and out["yd"][-1] > 5
    assert out["dird"][-1] == pytest.approx(0, abs=1e-3)
