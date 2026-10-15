"""Data-quality checks and fixes for tracking data (polars).

Each ``check_*`` returns a DataFrame of flagged rows/groups (empty = clean).
:func:`run_qa` bundles them into a report dict, :func:`qa_summary` tabulates it,
and :func:`write_qa_markdown` renders it. ``fix``-style functions return a new
DataFrame. Works on raw or standardized frames (checks are direction-agnostic).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from .tracking import DT, FIELD_LENGTH, FIELD_WIDTH, PID, PLAY_KEY, SNAP_EVENTS

GRP = ["gameId", "playId", "_pid"]
KEY_COLS = ["gameId", "playId", "nflId", "frameId"]
ANGLE_COLS = ("o", "dir")
INTERP_COLS = ("x", "y", "s", "a", "dis", "ballLandX", "ballLandY")
SINGLE_EVENTS = ("ball_snap", "snap_direct", "pass_forward", "handoff", "pass_arrived", "first_contact")

DESCRIPTIONS = {
    "duplicates": "Rows repeated on (gameId, playId, nflId, frameId)",
    "missing_xy": "Rows with null x or y",
    "frame_gaps": "Missing frames inside a player's play trajectory",
    "partial_presence": "Player present for fewer frames than the play spans",
    "impossible_kinematics": "Player s or a above physical limits",
    "teleports": "Frame-to-frame jump too large or inconsistent with `dis`",
    "out_of_bounds": "Position beyond the field by more than the margin",
    "player_counts": "Plays with frames not having 11 per team (+1 ball)",
    "event_issues": "Plays with missing/multiple snaps, repeated one-off events or mis-ordered events",
    "event_inconsistent": "Frames whose rows disagree on the event tag",
    "orientation": "Player-plays where o deviates from dir by >90 deg while running",
    "orientation_by_game": "Games with systematic orientation offset (historic NGS issue)",
}


def _with_pid(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(PID.alias("_pid")) if "_pid" not in df.columns else df


def _players(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(~pl.col("isBall")) if "isBall" in df.columns else df


def _angdiff(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Signed smallest difference a - b in degrees, in [-180, 180)."""
    return ((a - b + 180) % 360) - 180


# ------------------------------------------------------------------------ checks
def check_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with duplicate keys (ball keyed by nflId=null)."""
    keys = [c for c in KEY_COLS if c in df.columns]
    return df.filter(df.select(keys).is_duplicated()).select(keys + [c for c in ("x", "y") if c in df.columns])


def check_missing_xy(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("x").is_null() | pl.col("y").is_null()).select(KEY_COLS)


def check_frame_gaps(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(gaps, partial): interior missing frames per entity; entities spanning fewer frames than the play."""
    d = _with_pid(df).select(["gameId", "playId", "nflId", "_pid", "frameId"]).unique().sort(KEY_COLS[:2] + ["_pid", "frameId"])
    gaps = (d.with_columns(nMissing=(pl.col("frameId").diff().over(GRP) - 1))
            .filter(pl.col("nMissing") > 0)
            .select(["gameId", "playId", "nflId", pl.col("frameId").alias("frameAfterGap"), "nMissing"]))
    span = d.group_by(GRP + ["nflId"]).agg(n=pl.len(), lo=pl.col("frameId").min(), hi=pl.col("frameId").max())
    play = d.group_by(PLAY_KEY).agg(playLo=pl.col("frameId").min(), playHi=pl.col("frameId").max())
    partial = (span.join(play, on=PLAY_KEY)
               .filter((pl.col("lo") > pl.col("playLo")) | (pl.col("hi") < pl.col("playHi")))
               .select(["gameId", "playId", "nflId", "lo", "hi", "playLo", "playHi"]))
    return gaps, partial


def check_kinematics(df: pl.DataFrame, max_s: float = 12.0, max_a: float = 15.0) -> pl.DataFrame:
    """Player rows with speed > ``max_s`` yd/s or acceleration > ``max_a`` yd/s^2."""
    conds = [pl.col(c) > m for c, m in (("s", max_s), ("a", max_a)) if c in df.columns]
    if not conds:
        return pl.DataFrame()
    return _players(df).filter(pl.any_horizontal(conds)).select(KEY_COLS + [c for c in ("s", "a") if c in df.columns])


def _steps(df: pl.DataFrame) -> pl.DataFrame:
    d = _with_pid(df).sort(KEY_COLS[:2] + ["_pid", "frameId"])
    consec = pl.col("frameId").diff().over(GRP) == 1
    step = ((pl.col("x").diff().over(GRP)) ** 2 + (pl.col("y").diff().over(GRP)) ** 2).sqrt()
    return d.with_columns(step=pl.when(consec).then(step))


def check_teleports(df: pl.DataFrame, max_step: float = 1.5, dis_tol: float = 0.6,
                    include_ball: bool = False) -> pl.DataFrame:
    """Frames whose x/y jump from the previous frame exceeds ``max_step`` yd (15 yd/s at 10 Hz)
    or disagrees with ``dis`` (or ``s*dt`` if no ``dis``) by more than ``dis_tol`` yd."""
    d = _steps(df if include_ball else _players(df))
    ref = pl.col("dis") if "dis" in d.columns else (pl.col("s") * DT if "s" in d.columns else None)
    cond = pl.col("step") > max_step
    if ref is not None:
        cond = cond | ((pl.col("step") - ref).abs() > dis_tol)
    out = [c for c in ("dis", "s") if c in d.columns]
    return d.filter(cond.fill_null(False)).select(KEY_COLS + ["x", "y", "step"] + out)


def check_out_of_bounds(df: pl.DataFrame, margin: float = 3.0) -> pl.DataFrame:
    """Rows more than ``margin`` yd outside the 120 x 53.3 field."""
    oob = ((pl.col("x") < -margin) | (pl.col("x") > FIELD_LENGTH + margin)
           | (pl.col("y") < -margin) | (pl.col("y") > FIELD_WIDTH + margin))
    return df.filter(oob).select(KEY_COLS + ["x", "y"])


def check_player_counts(df: pl.DataFrame, per_team: int = 11) -> pl.DataFrame:
    """Plays where any frame lacks ``per_team`` players per team, 2 teams, or exactly one ball.

    The ball requirement only applies if the dataset contains ball rows at all.
    """
    team = next((c for c in ("club", "playerSide", "side") if c in df.columns), None)
    if team is None:
        return pl.DataFrame()
    has_ball = "isBall" in df.columns and df["isBall"].any()
    players = _players(df)
    per = players.group_by(PLAY_KEY + ["frameId", team]).agg(n=pl.col("frameId").len())
    frame = per.group_by(PLAY_KEY + ["frameId"]).agg(
        nTeams=pl.len(), minPerTeam=pl.col("n").min(), maxPerTeam=pl.col("n").max())
    if has_ball:
        balls = df.filter(pl.col("isBall")).group_by(PLAY_KEY + ["frameId"]).agg(nBall=pl.len())
        frame = frame.join(balls, on=PLAY_KEY + ["frameId"], how="left").with_columns(pl.col("nBall").fill_null(0))
    bad = (pl.col("nTeams") != 2) | (pl.col("minPerTeam") != per_team) | (pl.col("maxPerTeam") != per_team)
    if has_ball:
        bad = bad | (pl.col("nBall") != 1)
    agg = [pl.len().alias("nBadFrames"), pl.col("minPerTeam").min(), pl.col("maxPerTeam").max(), pl.col("nTeams").max()]
    if has_ball:
        agg.append(pl.col("nBall").max())
    return frame.filter(bad).group_by(PLAY_KEY).agg(agg).sort(PLAY_KEY)


def check_events(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(play_issues, inconsistent_frames) for event tagging.

    play_issues: snap count != 1, one-off events tagged on >1 frame, pass_forward before snap.
    inconsistent_frames: frames where rows carry different (or partly missing) events.
    """
    if "event" not in df.columns:
        return pl.DataFrame(), pl.DataFrame()
    ev = df.select(PLAY_KEY + ["frameId", pl.col("event").cast(pl.String)])
    per_frame = ev.group_by(PLAY_KEY + ["frameId"]).agg(
        nEvents=pl.col("event").drop_nulls().n_unique(), nTagged=pl.col("event").is_not_null().sum(), nRows=pl.len(),
        event=pl.col("event").drop_nulls().first())
    inconsistent = per_frame.filter((pl.col("nEvents") > 1) | ((pl.col("nTagged") > 0) & (pl.col("nTagged") < pl.col("nRows"))))
    tagged = per_frame.filter(pl.col("event").is_not_null())
    plays = df.select(PLAY_KEY).unique()
    counts = tagged.group_by(PLAY_KEY).agg(
        nSnap=pl.col("event").is_in(SNAP_EVENTS).sum(),
        snapFrame=pl.col("frameId").filter(pl.col("event").is_in(SNAP_EVENTS)).min(),
        passFrame=pl.col("frameId").filter(pl.col("event") == "pass_forward").min(),
        repeated=pl.col("event").filter(pl.col("event").is_in(SINGLE_EVENTS)).is_duplicated().any())
    issues = (plays.join(counts, on=PLAY_KEY, how="left")
              .with_columns(pl.col("nSnap").fill_null(0), pl.col("repeated").fill_null(False))
              .with_columns(problem=pl.concat_str([
                  pl.when(pl.col("nSnap") == 0).then(pl.lit("no_snap;")),
                  pl.when(pl.col("nSnap") > 1).then(pl.lit("multiple_snaps;")),
                  pl.when(pl.col("repeated")).then(pl.lit("repeated_event;")),
                  pl.when(pl.col("passFrame") < pl.col("snapFrame")).then(pl.lit("pass_before_snap;")),
              ], ignore_nulls=True))
              .filter(pl.col("problem") != "").sort(PLAY_KEY))
    return issues, inconsistent.sort(PLAY_KEY + ["frameId"])


def check_orientation(df: pl.DataFrame, min_speed: float = 3.0, max_dev: float = 90.0,
                      min_frames: int = 5, game_frac: float = 0.25) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(player_plays, games): orientation vs direction of travel while running.

    Running players mostly face where they move, so a player-play whose median
    |o - dir| exceeds ``max_dev`` is suspicious (backpedals excepted). Games with
    more than ``game_frac`` flagged player-plays, or a large median offset, point to
    the systematic NGS orientation errors seen in early seasons.
    """
    if not {"o", "dir", "s"} <= set(df.columns):
        return pl.DataFrame(), pl.DataFrame()
    run = _players(df).filter(pl.col("s") > min_speed).with_columns(dev=_angdiff(pl.col("o"), pl.col("dir")))
    pp = (run.group_by(["gameId", "playId", "nflId"]).agg(
        n=pl.len(), medAbsDev=pl.col("dev").abs().median(), medDev=pl.col("dev").median())
          .filter(pl.col("n") >= min_frames))
    games = (pp.group_by("gameId").agg(
        nPlayerPlays=pl.len(), fracFlagged=(pl.col("medAbsDev") > max_dev).mean(), medOffset=pl.col("medDev").median())
             .filter((pl.col("fracFlagged") > game_frac) | (pl.col("medOffset").abs() > 45)).sort("gameId"))
    return pp.filter(pl.col("medAbsDev") > max_dev).sort(["gameId", "playId", "nflId"]), games


def run_qa(df: pl.DataFrame, max_s: float = 12.0, max_a: float = 15.0, oob_margin: float = 3.0,
           max_step: float = 1.5, per_team: int = 11) -> dict[str, pl.DataFrame]:
    """Run every check; returns {issue_name: flagged DataFrame}."""
    gaps, partial = check_frame_gaps(df)
    ev, ev_frames = check_events(df)
    ori, ori_games = check_orientation(df)
    return {
        "duplicates": check_duplicates(df), "missing_xy": check_missing_xy(df),
        "frame_gaps": gaps, "partial_presence": partial,
        "impossible_kinematics": check_kinematics(df, max_s, max_a),
        "teleports": check_teleports(df, max_step), "out_of_bounds": check_out_of_bounds(df, oob_margin),
        "player_counts": check_player_counts(df, per_team),
        "event_issues": ev, "event_inconsistent": ev_frames,
        "orientation": ori, "orientation_by_game": ori_games,
    }


def qa_summary(report: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per issue: number of flags and distinct plays affected."""
    rows = []
    for name, d in report.items():
        n_plays = d.select(PLAY_KEY).n_unique() if d.height and set(PLAY_KEY) <= set(d.columns) else None
        rows.append({"issue": name, "nFlags": d.height, "nPlays": n_plays, "description": DESCRIPTIONS.get(name, "")})
    return pl.DataFrame(rows, schema={"issue": pl.String, "nFlags": pl.Int64, "nPlays": pl.Int64, "description": pl.String})


def _md_table(d: pl.DataFrame, n: int = 10) -> str:
    if d.height == 0:
        return "_none_\n"
    d = d.head(n)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.iter_rows()]
    return "\n".join(lines) + "\n"


def write_qa_markdown(report: dict[str, pl.DataFrame], path: str | Path, stats: dict | None = None,
                      examples: int = 10, title: str = "Tracking data QA report") -> Path:
    """Write a markdown QA report (summary table + first ``examples`` rows per issue)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}\n", f"_Generated {date.today().isoformat()} by `scripts/build_processed.py`._\n"]
    if stats:
        parts.append("\n".join(f"- **{k}**: {v}" for k, v in stats.items()) + "\n")
    parts += ["## Summary\n", _md_table(qa_summary(report), n=len(report))]
    for name, d in report.items():
        parts += [f"\n## {name} ({d.height})\n", f"{DESCRIPTIONS.get(name, '')}\n\n", _md_table(d, examples)]
    path.write_text("\n".join(parts))
    return path


# -------------------------------------------------------------------------- fixes
def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the first row per (gameId, playId, nflId, frameId)."""
    return df.unique(subset=[c for c in KEY_COLS if c in df.columns], keep="first", maintain_order=True)


def fix_spikes(df: pl.DataFrame, max_step: float = 1.5) -> pl.DataFrame:
    """Null-and-interpolate single-frame position spikes (jump out and straight back).

    Persistent jumps (no return) are left in place; they remain visible to
    :func:`check_teleports`. Adds boolean ``spikeFixed``.
    """
    d = _with_pid(df).sort(["gameId", "playId", "_pid", "frameId"])
    dx_in, dy_in = pl.col("x").diff().over(GRP), pl.col("y").diff().over(GRP)
    dx_out = (pl.col("x").shift(-1) - pl.col("x")).over(GRP)
    dy_out = (pl.col("y").shift(-1) - pl.col("y")).over(GRP)
    dx_skip = (pl.col("x").shift(-1) - pl.col("x").shift(1)).over(GRP)
    dy_skip = (pl.col("y").shift(-1) - pl.col("y").shift(1)).over(GRP)
    hyp = lambda a, b: (a ** 2 + b ** 2).sqrt()  # noqa: E731
    spike = ((hyp(dx_in, dy_in) > max_step) & (hyp(dx_out, dy_out) > max_step)
             & (hyp(dx_skip, dy_skip) < max_step)).fill_null(False)
    if "isBall" in d.columns:
        spike = spike & ~pl.col("isBall")
    d = d.with_columns(spikeFixed=spike)
    d = d.with_columns([pl.when(pl.col("spikeFixed")).then(None).otherwise(pl.col(c)).alias(c) for c in ("x", "y")])
    return d.with_columns([pl.col(c).interpolate().over(GRP) for c in ("x", "y")]).drop("_pid")


def fill_gaps(df: pl.DataFrame, max_gap: int = 3) -> pl.DataFrame:
    """Insert missing interior frames (runs of <= ``max_gap``) per player-play and interpolate.

    Positions/speeds are linearly interpolated, angles via sin/cos, events/time are
    left null and other columns forward-filled. Adds boolean ``imputed``.
    """
    d = _with_pid(df)
    grid = (d.group_by(GRP).agg(lo=pl.col("frameId").min(), hi=pl.col("frameId").max())
            .with_columns(frameId=pl.int_ranges("lo", pl.col("hi") + 1)).explode("frameId")
            .drop("lo", "hi").with_columns(pl.col("frameId").cast(d.schema["frameId"])))
    g = grid.join(d.with_columns(imputed=pl.lit(False)), on=GRP + ["frameId"], how="left")
    g = g.with_columns(pl.col("imputed").fill_null(True)).sort(GRP + ["frameId"])
    run = pl.col("imputed").rle_id().over(GRP)
    g = g.with_columns(_run=run).with_columns(_runLen=pl.len().over(GRP + ["_run"]))
    g = g.filter(~pl.col("imputed") | (pl.col("_runLen") <= max_gap))
    interp = [c for c in INTERP_COLS if c in g.columns]
    angles = [c for c in ANGLE_COLS if c in g.columns]
    leave_null = {"event", "time", "imputed", "_run", "_runLen", "frameId", *GRP, *interp, *angles}
    ffill = [c for c in g.columns if c not in leave_null]
    g = g.with_columns([pl.col(c).interpolate().over(GRP) for c in interp]
                       + [pl.col(c).forward_fill().over(GRP) for c in ffill])
    if angles:
        rad = lambda c: pl.col(c).radians()  # noqa: E731
        g = g.with_columns([pl.arctan2(rad(c).sin().interpolate().over(GRP), rad(c).cos().interpolate().over(GRP))
                            .degrees().mod(360).cast(pl.Float32).alias(c) for c in angles])
    return g.drop("_run", "_runLen", "_pid").select([c for c in df.columns if c != "_pid"] + ["imputed"])


def correct_orientation(df: pl.DataFrame, game_offsets: pl.DataFrame) -> pl.DataFrame:
    """Subtract a per-game orientation offset (``gameId``, ``medOffset`` from check_orientation)."""
    d = df.join(game_offsets.select("gameId", "medOffset"), on="gameId", how="left")
    return d.with_columns(o=((pl.col("o") - pl.col("medOffset").fill_null(0)) % 360).cast(pl.Float32)).drop("medOffset")


def smooth_kinematics(df: pl.DataFrame, window: int = 7, poly: int = 2) -> pl.DataFrame:
    """Savitzky-Golay smoothed positions and derivatives per contiguous player trajectory.

    Adds xSmooth, ySmooth, vxSmooth, vySmooth, axSmooth, aySmooth, sSmooth. Rows
    within ``window//2`` of a segment edge (play start/end or frame gap) or in
    segments shorter than ``window`` keep raw x/y and null derivatives.
    """
    d = _with_pid(df).sort(["gameId", "playId", "_pid", "frameId"])
    new_seg = ((pl.col("frameId").diff().over(GRP) != 1).fill_null(True))
    seg = d.select(new_seg.cum_sum().alias("seg"))["seg"].to_numpy()
    half = window // 2
    idx = np.arange(len(seg))
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
    ends = np.r_[starts[1:], len(seg)]
    pos_in = idx - np.repeat(starts, ends - starts)
    pos_to_end = np.repeat(ends, ends - starts) - 1 - idx
    valid = (pos_in >= half) & (pos_to_end >= half)
    for c in ("x", "y"):  # also exclude rows whose window touches a null
        bad = np.isnan(d[c].cast(pl.Float64).to_numpy()).astype(float)
        valid &= np.convolve(bad, np.ones(window), mode="same") == 0
    out = {}
    for c in ("x", "y"):
        v = d[c].cast(pl.Float64).to_numpy()
        filled = np.where(np.isnan(v), 0.0, v)
        for k, name in ((0, f"{c}Smooth"), (1, f"v{c}Smooth"), (2, f"a{c}Smooth")):
            sm = savgol_filter(filled, window, poly, deriv=k, delta=DT, mode="interp") if len(v) >= window else v * np.nan
            ok = valid & ~np.isnan(v)
            out[name] = np.where(ok, sm, v if k == 0 else np.nan)
    d = d.with_columns([pl.Series(k, v, dtype=pl.Float32) for k, v in out.items()])
    return d.with_columns(sSmooth=(pl.col("vxSmooth") ** 2 + pl.col("vySmooth") ** 2).sqrt()).drop("_pid")


def clean_tracking(df: pl.DataFrame, max_gap: int = 3, max_step: float = 1.5, drop_bad_count_plays: bool = False,
                   orientation_fix: bool = False) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """QA on raw data, then dedupe -> fix spikes -> fill small gaps (-> optional drops/fixes).

    Returns ``(cleaned, report)`` where ``report`` describes the *raw* input.
    """
    report = run_qa(df, max_step=max_step)
    out = fill_gaps(fix_spikes(drop_duplicates(df), max_step), max_gap)
    if drop_bad_count_plays and report["player_counts"].height:
        out = out.join(report["player_counts"].select(PLAY_KEY), on=PLAY_KEY, how="anti")
    if orientation_fix and report["orientation_by_game"].height:
        out = correct_orientation(out, report["orientation_by_game"])
    return out, report
