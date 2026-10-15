"""Data-quality checks and fixes for tracking data (polars).

Each ``check_*`` returns a DataFrame of flagged rows/groups (empty = clean).
:func:`run_qa` bundles them into a report dict, :func:`qa_summary` tabulates it,
and :func:`write_qa_markdown` renders it. Fixes return new DataFrames. Works on
game plays (``gameId, playId``) and Combine drill attempts (``eventId``) alike;
checks whose columns are absent (event, o, team) return empty frames.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from .tracking import DT, FIELD_LENGTH, FIELD_WIDTH, SNAP_EVENTS, entity_key, play_key, with_pid

ANGLE_COLS = ("o", "dir")
INTERP_COLS = ("x", "y", "s", "a", "dis", "tSec", "ballLandX", "ballLandY")
SINGLE_EVENTS = ("ball_snap", "snap_direct", "pass_forward", "handoff", "pass_arrived", "first_contact")
EMPTY = pl.DataFrame()

DESCRIPTIONS = {
    "duplicates": "Rows repeated on (play, entity, frameId)",
    "missing_xy": "Rows with null x or y",
    "frame_gaps": "Missing frames inside an entity's trajectory",
    "partial_presence": "Entity present for fewer frames than the play/attempt spans",
    "impossible_kinematics": "Player s or a above physical limits, or negative (s, a are unsigned)",
    "teleports": "Frame-to-frame jump too large or inconsistent with `dis`",
    "out_of_bounds": "Position beyond the field by more than the margin (game data only)",
    "player_counts": "Plays with frames not having 11 per team (+1 ball); skipped for sparse data",
    "event_issues": "Plays with missing/multiple snaps, repeated one-off events or mis-ordered events",
    "event_inconsistent": "Frames whose rows disagree on the event tag",
    "orientation": "Player-plays where o deviates from dir by >90 deg while running",
    "orientation_by_game": "Games with systematic orientation offset (historic NGS issue)",
}


def _keys(df) -> list[str]:
    return play_key(df) + ["nflId", "frameId"]


def _players(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(~pl.col("isBall")) if "isBall" in df.columns else df


def _angdiff(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Signed smallest difference a - b in degrees, in [-180, 180)."""
    return ((a - b + 180) % 360) - 180


# ------------------------------------------------------------------------ checks
def check_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with duplicate (play, entity, frame) keys."""
    d = with_pid(df)
    return d.filter(d.select(entity_key(d) + ["frameId"]).is_duplicated()).select(_keys(df) + ["x", "y"])


def check_missing_xy(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("x").is_null() | pl.col("y").is_null()).select(_keys(df))


def check_frame_gaps(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(gaps, partial): interior missing frames per entity; entities spanning fewer frames than the play."""
    key = play_key(df)
    d = with_pid(_players(df))
    grp = entity_key(d)
    d = d.select(grp + ["nflId", "frameId"]).unique().sort(grp + ["frameId"])
    gaps = (d.with_columns(nMissing=(pl.col("frameId").diff().over(grp) - 1))
            .filter(pl.col("nMissing") > 0)
            .select(key + ["nflId", pl.col("frameId").alias("frameAfterGap"), "nMissing"]))
    span = d.group_by(grp + ["nflId"]).agg(lo=pl.col("frameId").min(), hi=pl.col("frameId").max())
    play = d.group_by(key).agg(playLo=pl.col("frameId").min(), playHi=pl.col("frameId").max())
    partial = (span.join(play, on=key)
               .filter((pl.col("lo") > pl.col("playLo")) | (pl.col("hi") < pl.col("playHi")))
               .select(key + ["nflId", "lo", "hi", "playLo", "playHi"]).sort(key))
    return gaps, partial


def check_kinematics(df: pl.DataFrame, max_s: float = 12.0, max_a: float = 15.0) -> pl.DataFrame:
    """Player rows with speed > ``max_s`` yd/s, acceleration > ``max_a`` yd/s^2, or either negative
    (NGS ``a`` is a magnitude; one 2025 game carries signed values)."""
    conds = [(pl.col(c) > m) | (pl.col(c) < 0) for c, m in (("s", max_s), ("a", max_a)) if c in df.columns]
    if not conds:
        return EMPTY
    return _players(df).filter(pl.any_horizontal(conds)).select(_keys(df) + [c for c in ("s", "a") if c in df.columns])


def _steps(df: pl.DataFrame) -> pl.DataFrame:
    d = with_pid(df)
    grp = entity_key(d)
    d = d.sort(grp + ["frameId"])
    consec = pl.col("frameId").diff().over(grp) == 1
    step = ((pl.col("x").diff().over(grp)) ** 2 + (pl.col("y").diff().over(grp)) ** 2).sqrt()
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
    return d.filter(cond.fill_null(False)).select(_keys(df) + ["x", "y", "step"] + out)


def check_out_of_bounds(df: pl.DataFrame, margin: float = 3.0) -> pl.DataFrame:
    """Rows more than ``margin`` yd outside the 120 x 53.3 field. Combine drills are not run on
    game-field coordinates (y spans about -11 to 66), so drill data returns no flags."""
    if "gameId" not in df.columns:
        return EMPTY
    oob = ((pl.col("x") < -margin) | (pl.col("x") > FIELD_LENGTH + margin)
           | (pl.col("y") < -margin) | (pl.col("y") > FIELD_WIDTH + margin))
    return df.filter(oob).select(_keys(df) + ["x", "y"])


def is_full_field(df: pl.DataFrame, min_players: int = 20) -> bool:
    """True if frames typically contain both full teams (median >= ``min_players`` players)."""
    n = _players(df).group_by(play_key(df) + ["frameId"]).len()["len"]
    return n.len() > 0 and n.median() >= min_players


def check_player_counts(df: pl.DataFrame, per_team: int = 11) -> pl.DataFrame:
    """Plays where any frame lacks ``per_team`` players per team, 2 teams, or exactly one ball.

    The ball requirement only applies if the dataset contains ball rows at all.
    """
    team = next((c for c in ("club", "playerSide", "side") if c in df.columns), None)
    if team is None:
        return EMPTY
    key = play_key(df)
    fk = key + ["frameId"]
    has_ball = "isBall" in df.columns and df["isBall"].any()
    per = _players(df).group_by(fk + [team]).agg(n=pl.col("nflId").n_unique())
    frame = per.group_by(fk).agg(nTeams=pl.len(), minPerTeam=pl.col("n").min(), maxPerTeam=pl.col("n").max())
    bad = (pl.col("nTeams") != 2) | (pl.col("minPerTeam") != per_team) | (pl.col("maxPerTeam") != per_team)
    agg = [pl.len().alias("nBadFrames"), pl.col("minPerTeam").min(), pl.col("maxPerTeam").max(), pl.col("nTeams").max()]
    if has_ball:
        balls = df.filter(pl.col("isBall")).group_by(fk).agg(nBall=pl.len())
        frame = frame.join(balls, on=fk, how="left").with_columns(pl.col("nBall").fill_null(0))
        bad = bad | (pl.col("nBall") != 1)
        agg.append(pl.col("nBall").max())
    return frame.filter(bad).group_by(key).agg(agg).sort(key)


def check_events(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(play_issues, inconsistent_frames) for event tagging.

    play_issues: snap count != 1, one-off events on >1 frame, pass_forward before snap.
    inconsistent_frames: frames whose rows carry different (or partly missing) events.
    """
    if "event" not in df.columns:
        return EMPTY, EMPTY
    key = play_key(df)
    fk = key + ["frameId"]
    ev = df.select(fk + [pl.col("event").cast(pl.String)])
    per_frame = ev.group_by(fk).agg(
        nEvents=pl.col("event").drop_nulls().n_unique(), nTagged=pl.col("event").is_not_null().sum(), nRows=pl.len(),
        event=pl.col("event").drop_nulls().first())
    inconsistent = per_frame.filter((pl.col("nEvents") > 1)
                                    | ((pl.col("nTagged") > 0) & (pl.col("nTagged") < pl.col("nRows"))))
    counts = per_frame.filter(pl.col("event").is_not_null()).group_by(key).agg(
        nSnap=pl.col("event").is_in(SNAP_EVENTS).sum(),
        snapFrame=pl.col("frameId").filter(pl.col("event").is_in(SNAP_EVENTS)).min(),
        passFrame=pl.col("frameId").filter(pl.col("event") == "pass_forward").min(),
        repeated=pl.col("event").filter(pl.col("event").is_in(SINGLE_EVENTS)).is_duplicated().any())
    issues = (df.select(key).unique().join(counts, on=key, how="left")
              .with_columns(pl.col("nSnap").fill_null(0), pl.col("repeated").fill_null(False))
              .with_columns(problem=pl.concat_str([
                  pl.when(pl.col("nSnap") == 0).then(pl.lit("no_snap;")),
                  pl.when(pl.col("nSnap") > 1).then(pl.lit("multiple_snaps;")),
                  pl.when(pl.col("repeated")).then(pl.lit("repeated_event;")),
                  pl.when(pl.col("passFrame") < pl.col("snapFrame")).then(pl.lit("pass_before_snap;")),
              ], ignore_nulls=True))
              .filter(pl.col("problem") != "").sort(key))
    return issues, inconsistent.sort(fk)


def check_orientation(df: pl.DataFrame, min_speed: float = 3.0, max_dev: float = 90.0,
                      min_frames: int = 5, game_frac: float = 0.25) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(player_plays, games): orientation vs direction of travel while running (needs ``o``).

    Running players mostly face where they move, so a player-play whose median
    |o - dir| exceeds ``max_dev`` is suspicious (backpedals excepted). Games with
    more than ``game_frac`` flagged player-plays, or a median offset > 45 deg, point
    to the systematic NGS orientation errors seen in early seasons.
    """
    if not {"o", "dir", "s"} <= set(df.columns):
        return EMPTY, EMPTY
    key = play_key(df)
    group = key[:1] if key == ["gameId", "playId"] else key
    run = _players(df).filter(pl.col("s") > min_speed).with_columns(dev=_angdiff(pl.col("o"), pl.col("dir")))
    pp = (run.group_by(key + ["nflId"]).agg(n=pl.len(), medAbsDev=pl.col("dev").abs().median(),
                                            medDev=pl.col("dev").median())
          .filter(pl.col("n") >= min_frames))
    games = (pp.group_by(group).agg(nPlayerPlays=pl.len(), fracFlagged=(pl.col("medAbsDev") > max_dev).mean(),
                                    medOffset=pl.col("medDev").median())
             .filter((pl.col("fracFlagged") > game_frac) | (pl.col("medOffset").abs() > 45)).sort(group))
    return pp.filter(pl.col("medAbsDev") > max_dev).sort(key + ["nflId"]), games


def run_qa(df: pl.DataFrame, max_s: float = 12.0, max_a: float = 15.0, oob_margin: float = 3.0,
           max_step: float = 1.5, per_team: int = 11, counts: bool | None = None) -> dict[str, pl.DataFrame]:
    """Run every applicable check; returns {issue_name: flagged DataFrame}.

    ``counts=None`` runs the 11v11 check only for full-field data (see :func:`is_full_field`).
    """
    gaps, partial = check_frame_gaps(df)
    ev, ev_frames = check_events(df)
    ori, ori_games = check_orientation(df)
    do_counts = is_full_field(df) if counts is None else counts
    return {
        "duplicates": check_duplicates(df), "missing_xy": check_missing_xy(df),
        "frame_gaps": gaps, "partial_presence": partial,
        "impossible_kinematics": check_kinematics(df, max_s, max_a),
        "teleports": check_teleports(df, max_step), "out_of_bounds": check_out_of_bounds(df, oob_margin),
        "player_counts": check_player_counts(df, per_team) if do_counts else EMPTY,
        "event_issues": ev, "event_inconsistent": ev_frames,
        "orientation": ori, "orientation_by_game": ori_games,
    }


def qa_summary(report: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per issue: number of flags and distinct plays/attempts affected."""
    rows = []
    for name, d in report.items():
        key = next((k for k in (["gameId", "playId"], ["eventId"], ["gameId"]) if set(k) <= set(d.columns)), None)
        n_plays = d.select(key).n_unique() if d.height and key else None
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


def write_qa_markdown(reports: dict[str, dict[str, pl.DataFrame]] | dict[str, pl.DataFrame], path,
                      stats: dict | None = None, examples: int = 10, title: str = "Tracking data QA report",
                      preamble: str | None = None) -> Path:
    """Write a markdown QA report: summary + first ``examples`` rows per issue.

    ``reports`` is one :func:`run_qa` result or ``{section_name: run_qa_result}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reports and all(isinstance(v, pl.DataFrame) for v in reports.values()):
        reports = {"": reports}
    parts = [f"# {title}\n", f"_Generated {date.today().isoformat()} by `scripts/build_processed.py`._\n"]
    if preamble:
        parts.append(preamble)
    if stats:
        parts.append("## Processing\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in stats.items()) + "\n")
    for section, report in reports.items():
        h = "##" if not section else "###"
        if section:
            parts.append(f"\n## {section}\n")
        parts += [f"{h} Summary\n", _md_table(qa_summary(report), n=len(report))]
        for name, d in report.items():
            if d.height:
                parts += [f"\n{h} {name} ({d.height})\n", f"{DESCRIPTIONS.get(name, '')}\n\n", _md_table(d, examples)]
    path.write_text("\n".join(parts))
    return path


# -------------------------------------------------------------------------- fixes
def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the first row per (play, entity, frame)."""
    d = with_pid(df)
    return d.unique(subset=entity_key(d) + ["frameId"], keep="first", maintain_order=True).drop("_pid")


def fix_negative_accel(df: pl.DataFrame) -> pl.DataFrame:
    """Replace negative ``a`` with its magnitude (NGS ``a`` is unsigned); adds ``aSignFixed``."""
    if "a" not in df.columns:
        return df
    neg = (pl.col("a") < 0).fill_null(False)
    return df.with_columns(aSignFixed=neg, a=pl.col("a").abs())


def fix_spikes(df: pl.DataFrame, max_step: float = 1.5) -> pl.DataFrame:
    """Null-and-interpolate single-frame position spikes (jump out and straight back).

    Persistent jumps (no return) are left in place and stay visible to
    :func:`check_teleports`. Adds boolean ``spikeFixed``.
    """
    d = with_pid(df)
    grp = entity_key(d)
    d = d.sort(grp + ["frameId"])
    hyp = lambda a, b: (a ** 2 + b ** 2).sqrt()  # noqa: E731
    lag = lambda c, a, b: (pl.col(c).shift(a) - pl.col(c).shift(b)).over(grp)  # noqa: E731
    spike = ((hyp(lag("x", 0, 1), lag("y", 0, 1)) > max_step) & (hyp(lag("x", -1, 0), lag("y", -1, 0)) > max_step)
             & (hyp(lag("x", -1, 1), lag("y", -1, 1)) < max_step)).fill_null(False)
    if "isBall" in d.columns:
        spike = spike & ~pl.col("isBall")
    d = d.with_columns(spikeFixed=spike)
    d = d.with_columns([pl.when(pl.col("spikeFixed")).then(None).otherwise(pl.col(c)).alias(c) for c in ("x", "y")])
    return d.with_columns([pl.col(c).interpolate().over(grp) for c in ("x", "y")]).drop("_pid")


def fill_gaps(df: pl.DataFrame, max_gap: int = 3) -> pl.DataFrame:
    """Insert missing interior frames (runs of <= ``max_gap``) per entity and interpolate.

    Positions/speeds/tSec are linearly interpolated, angles via sin/cos, events and
    time left null, other columns forward-filled. Adds boolean ``imputed``.
    """
    d = with_pid(df)
    grp = entity_key(d)
    grid = (d.group_by(grp).agg(lo=pl.col("frameId").min(), hi=pl.col("frameId").max())
            .with_columns(frameId=pl.int_ranges("lo", pl.col("hi") + 1)).explode("frameId")
            .drop("lo", "hi").with_columns(pl.col("frameId").cast(d.schema["frameId"])))
    g = grid.join(d.with_columns(imputed=pl.lit(False)), on=grp + ["frameId"], how="left")
    g = g.with_columns(pl.col("imputed").fill_null(True)).sort(grp + ["frameId"])
    g = g.with_columns(_run=pl.col("imputed").rle_id().over(grp)).with_columns(_runLen=pl.len().over(grp + ["_run"]))
    g = g.filter(~pl.col("imputed") | (pl.col("_runLen") <= max_gap))
    interp = [c for c in INTERP_COLS if c in g.columns]
    angles = [c for c in ANGLE_COLS if c in g.columns]
    skip = {"event", "time", "imputed", "_run", "_runLen", "frameId", *grp, *interp, *angles}
    g = g.with_columns([pl.col(c).interpolate().over(grp) for c in interp]
                       + [pl.col(c).forward_fill().over(grp) for c in g.columns if c not in skip])
    if angles:
        rad = lambda c: pl.col(c).radians()  # noqa: E731
        g = g.with_columns([pl.arctan2(rad(c).sin().interpolate().over(grp), rad(c).cos().interpolate().over(grp))
                            .degrees().mod(360).cast(pl.Float32).alias(c) for c in angles])
    return g.select([c for c in df.columns if c not in ("_pid", "imputed")] + ["imputed"])


def correct_orientation(df: pl.DataFrame, game_offsets: pl.DataFrame) -> pl.DataFrame:
    """Subtract a per-game orientation offset (``medOffset`` from :func:`check_orientation`)."""
    on = [c for c in game_offsets.columns if c in ("gameId", "eventId", "draftYear")]
    d = df.join(game_offsets.select(on + ["medOffset"]), on=on, how="left")
    return d.with_columns(o=((pl.col("o") - pl.col("medOffset").fill_null(0)) % 360).cast(pl.Float32)).drop("medOffset")


def smooth_kinematics(df: pl.DataFrame, window: int = 7, poly: int = 2, cols=("x", "y")) -> pl.DataFrame:
    """Savitzky-Golay smoothed positions and derivatives per contiguous trajectory.

    For each of ``cols`` (e.g. ``("xd", "yd")`` for drills) adds ``<c>Smooth``,
    ``v<c>Smooth`` and ``a<c>Smooth``, plus ``sSmooth`` from the first two. Rows
    within ``window//2`` of a segment edge (start/end, frame gap or null) keep the
    raw position and get null derivatives.
    """
    d = with_pid(df)
    grp = entity_key(d)
    d = d.sort(grp + ["frameId"])
    seg = d.select(((pl.col("frameId").diff().over(grp) != 1).fill_null(True)).cum_sum().alias("s"))["s"].to_numpy()
    half, n = window // 2, len(seg)
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1] if n else np.array([], int)
    ends = np.r_[starts[1:], n]
    idx = np.arange(n)
    valid = (idx - np.repeat(starts, ends - starts) >= half) & (np.repeat(ends, ends - starts) - 1 - idx >= half)
    raw = {c: d[c].cast(pl.Float64).to_numpy() for c in cols}
    for v in raw.values():  # exclude rows whose window touches a null
        valid &= np.convolve(np.isnan(v).astype(float), np.ones(window), mode="same") == 0
    out = {}
    for c, v in raw.items():
        filled = np.nan_to_num(v)
        for k, name in ((0, f"{c}Smooth"), (1, f"v{c}Smooth"), (2, f"a{c}Smooth")):
            sm = savgol_filter(filled, window, poly, deriv=k, delta=DT, mode="interp") if n >= window else np.full(n, np.nan)
            out[name] = np.where(valid, sm, v if k == 0 else np.nan)
    d = d.with_columns([pl.Series(k, v, dtype=pl.Float32, nan_to_null=True) for k, v in out.items()])
    c0, c1 = cols[:2]
    return d.with_columns(sSmooth=(pl.col(f"v{c0}Smooth") ** 2 + pl.col(f"v{c1}Smooth") ** 2).sqrt()).drop("_pid")


def clean_tracking(df: pl.DataFrame, max_gap: int = 3, max_step: float = 1.5, drop_bad_count_plays: bool = False,
                   orientation_fix: bool = False, counts: bool | None = None) -> tuple[pl.DataFrame, dict]:
    """QA on raw data, then dedupe -> fix spikes -> fill small gaps (-> optional drops/fixes).

    Returns ``(cleaned, report)``; ``report`` describes the *raw* input.
    """
    report = run_qa(df, max_step=max_step, counts=counts)
    out = fill_gaps(fix_spikes(fix_negative_accel(drop_duplicates(df)), max_step), max_gap)
    if drop_bad_count_plays and report["player_counts"].height:
        key = play_key(df)
        out = out.join(report["player_counts"].select(key), on=key, how="anti")
    if orientation_fix and report["orientation_by_game"].height:
        out = correct_orientation(out, report["orientation_by_game"])
    return out, report
