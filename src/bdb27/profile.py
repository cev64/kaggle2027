"""Dataset-level profile of the BDB 2027 raw files (row counts, coverage, season types).

Used by ``scripts/build_processed.py`` to prepend a data summary to ``docs/data_qa.md``.
Reads the small context tables directly and tracking through the normalized parquet
cache (lazy), so it is cheap to rerun.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from . import tracking as T

GROUPS = ("WR", "TE", "DB", "OL", "DL")


def _md(d: pl.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.iter_rows()]
    return "\n".join(lines) + "\n"


def table_counts(raw: Path) -> pl.DataFrame:
    rows = []
    for name in ("players", "combine_results", "player_career_successes", "games", "player_play"):
        p = raw / f"{name}.csv"
        if p.exists():
            d = pl.read_csv(p, null_values=["NA", ""], infer_schema_length=None)
            rows.append({"file": p.name, "rows": d.height, "columns": d.width})
    return pl.DataFrame(rows)


def combine_coverage(ct: pl.LazyFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """Players with tracked attempts per drill type x position group."""
    ev = (ct.filter(~pl.col("isBall")).select("eventId", "nflId", pl.col("drillType").cast(pl.String)).unique()
          .collect().join(positions, on="nflId", how="left"))
    out = (ev.group_by("drillType", "combinePosition").agg(pl.col("nflId").n_unique())
           .pivot(on="combinePosition", index="drillType", values="nflId").sort("drillType"))
    totals = ev.group_by("drillType").agg(attempts=pl.len())
    return out.join(totals, on="drillType").select(["drillType", *[g for g in GROUPS if g in out.columns], "attempts"])


def game_coverage(raw: Path, cache: Path) -> pl.DataFrame:
    """Player-plays, players and games in game tracking per season and season type."""
    games = T.load_table("games", raw).select("gameId", pl.col("seasonType").cast(pl.String))
    frames = []
    for p in T.discover_files(raw)["game_tracking"]:
        lf = T.cached_frame(p, cache)
        pp = lf.select("gameId", "playId", "nflId", "season").unique().collect().join(games, on="gameId", how="left")
        frames.append(pp)
    d = pl.concat(frames)
    return (d.group_by("season", "seasonType")
            .agg(playerPlays=pl.len(), players=pl.col("nflId").n_unique(), games=pl.col("gameId").n_unique())
            .sort("season", "seasonType"))


def outcome_coverage(raw: Path, positions: pl.DataFrame) -> pl.DataFrame:
    """Regular-season non-null counts of the main NGS outcome fields by position group."""
    pp = T.load_table("player_play", raw)
    games = T.load_table("games", raw).select("gameId", pl.col("seasonType").cast(pl.String))
    pp = pp.join(games, on="gameId").filter(pl.col("seasonType") == "REG").join(positions, on="nflId", how="left")
    fields = ["separationAtPassForward", "cushion", "playerGetOff", "peakPressureProbabilityAllowed",
              "pressureAllowed", "timeToPressure", "coverageAssignment"]
    aggs = [pl.len().alias("regPlayerPlays")] + [pl.col(f).is_not_null().sum().alias(f) for f in fields if f in pp.columns]
    return pp.group_by("combinePosition").agg(aggs).sort("combinePosition")


def profile_markdown(raw: Path, cache: Path) -> str:
    positions = T.load_table("combine_results", raw).select("nflId", pl.col("combinePosition").cast(pl.String))
    parts = ["## Dataset profile\n", "### Files\n", _md(table_counts(raw))]
    ct_paths = T.discover_files(raw)["combine_tracking"]
    if ct_paths:
        ct = T.cached_frame(ct_paths[0], cache)
        parts += ["\n### Combine tracking: players with tracked attempts, by drill type and position group\n",
                  _md(combine_coverage(ct, positions))]
    if T.discover_files(raw)["game_tracking"]:
        parts += ["\n### Game tracking coverage (cohort players only; PRE/POST included in the raw files)\n",
                  _md(game_coverage(raw, cache))]
    parts += ["\n### Regular-season NGS outcome coverage in player_play (non-null player-plays)\n",
              _md(outcome_coverage(raw, positions))]
    return "\n".join(parts)
