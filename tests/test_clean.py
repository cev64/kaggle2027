import numpy as np
import polars as pl
import pytest

from bdb27 import clean as C
from bdb27 import tracking as T


def flagged(d, col="nflId"):
    return set(d[col].to_list()) if d.height else set()


def test_clean_data_has_no_flags(clean):
    rep = C.run_qa(clean)
    assert {k: v.height for k, v in rep.items() if v.height} == {}


def test_detectors_find_injected_anomalies(dirty):
    rep = C.run_qa(dirty)
    assert flagged(rep["duplicates"]) == {1103}
    gaps = rep["frame_gaps"]
    assert gaps.height == 1 and gaps["nflId"][0] == 1102 and gaps["nMissing"][0] == 1
    assert {1101, 1104} <= flagged(rep["teleports"]) and flagged(rep["teleports"]) <= {1101, 1104}
    assert flagged(rep["impossible_kinematics"]) == {2105}
    assert flagged(rep["out_of_bounds"]) == {1104}
    pc = rep["player_counts"]
    # play 2 lacks a defender throughout; play 1's missing frame leaves 10 offense in one frame
    assert pc["playId"].to_list() == [1, 2] and pc["nBadFrames"].to_list() == [1, 30]
    assert pc["minPerTeam"].to_list() == [10, 10]
    ev = rep["event_issues"]
    assert ev["playId"].to_list() == [2] and "multiple_snaps" in ev["problem"][0]
    assert flagged(rep["orientation"]) == {1205}
    assert rep["orientation_by_game"].height == 0  # one bad player is not a systematic game issue


def test_orientation_game_offset_and_fix(clean):
    shifted = clean.with_columns(o=(pl.col("o") + 90) % 360)
    _, games = C.check_orientation(shifted)
    assert games.height == 1 and games["medOffset"][0] == pytest.approx(90, abs=1)
    fixed = C.correct_orientation(shifted, games)
    assert C.check_orientation(fixed)[1].height == 0


def test_orientation_optional(clean):
    assert C.check_orientation(clean.drop("o"))[0].height == 0


def test_counts_skipped_for_sparse(clean):
    sparse = clean.filter(pl.col("nflId").is_in([1100, 2100, 1200, 2200]))
    assert not C.is_full_field(sparse) and C.run_qa(sparse)["player_counts"].height == 0
    assert C.run_qa(sparse, counts=True)["player_counts"].height == 2


def test_clean_tracking_fixes(dirty, clean):
    out, rep = C.clean_tracking(dirty)
    assert rep["duplicates"].height == 2
    after = C.run_qa(out)
    assert after["duplicates"].height == 0 and after["frame_gaps"].height == 0
    assert 1101 not in flagged(after["teleports"])
    key = ["playId", "nflId", "frameId"]
    ref = clean.select(key + ["x", "y", "dir"])
    m = out.join(ref, on=key, suffix="_ref")
    filled = m.filter((pl.col("nflId") == 1102) & (pl.col("frameId") == 8))
    assert filled["imputed"][0] and filled["x"][0] == pytest.approx(filled["x_ref"][0], abs=1e-3)
    assert filled["dir"][0] == pytest.approx(filled["dir_ref"][0], abs=1e-2)
    spike = m.filter((pl.col("nflId") == 1101) & (pl.col("frameId") == 15))
    assert spike["spikeFixed"][0] and spike["x"][0] == pytest.approx(spike["x_ref"][0], abs=1e-3)


def test_fill_gaps_respects_max_gap(clean):
    holed = clean.filter(~((pl.col("nflId") == 1100) & pl.col("frameId").is_between(5, 9)))
    out = C.fill_gaps(holed, max_gap=3)
    assert out.filter(pl.col("nflId") == 1100).height == 25


def test_smoothing_recovers_velocity(clean, plays):
    df = T.prepare_tracking(clean, plays)
    sm = C.smooth_kinematics(df).filter(~pl.col("isBall") & pl.col("vxSmooth").is_not_null())
    assert sm.height > 0
    assert np.allclose(sm["vxSmooth"], sm["vx"], atol=1e-2) and np.allclose(sm["axSmooth"], 0, atol=1e-2)


def test_combine_qa(combine_pd):
    df = T.add_frame_ids(T.normalize_columns(pl.from_pandas(combine_pd)))
    rep = C.run_qa(df)
    assert rep["player_counts"].height == 0 and rep["event_issues"].height == 0
    assert rep["teleports"].height == 0 and rep["duplicates"].height == 0
    dup = pl.concat([df, df.head(1)])
    assert C.run_qa(dup)["duplicates"].height == 2


def test_markdown_report(tmp_path, dirty):
    p = C.write_qa_markdown({"game": C.run_qa(dirty)}, tmp_path / "qa.md", stats={"rows": dirty.height})
    text = p.read_text()
    assert "### Summary" in text and "teleports" in text and "| issue |" in text


def test_negative_accel_flagged_and_fixed(clean):
    bad = clean.with_columns(a=pl.when((pl.col("nflId") == 1101) & (pl.col("frameId") == 3)).then(-2.0)
                             .otherwise(pl.col("a")))
    assert flagged(C.check_kinematics(bad)) == {1101}
    fixed = C.fix_negative_accel(bad)
    assert fixed["a"].min() >= 0 and fixed["aSignFixed"].sum() == 1


def test_out_of_bounds_skipped_for_drills():
    drill = pl.DataFrame({"eventId": ["e"], "nflId": [1], "frameId": [1], "x": [5.0], "y": [-11.0]})
    assert C.check_out_of_bounds(drill).height == 0
