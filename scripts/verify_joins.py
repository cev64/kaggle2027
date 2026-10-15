#!/usr/bin/env python
"""Measure join rates from the BDB 2027 files to nflverse (games, plays, players).

    python scripts/verify_joins.py [--raw data/raw] [--out docs/join_rates.md]

Checks, split by season and season type (PRE/REG/POST):
* gameId -> schedules.old_game_id
* (gameId, playId) of player_play -> pbp, participation, FTN charting
* nflId (players.csv) -> nflverse players.nfl_id, with draft pick/year agreement
* (gameId, playId, nflId) of player_play -> that player listed on the field in participation
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from bdb27 import external, keys  # noqa: E402
from bdb27 import tracking as T  # noqa: E402
from bdb27.profile import _md  # noqa: E402


def rate(df: pl.DataFrame, flag: str, by=("season", "seasonType")) -> pl.DataFrame:
    return (df.group_by(list(by)).agg(n=pl.len(), matched=pl.col(flag).sum(), rate=pl.col(flag).mean())
            .sort(list(by)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/join_rates.md")
    args = ap.parse_args(argv)

    games = T.load_table("games", args.raw).with_columns(pl.col("seasonType").cast(pl.String))
    pp = T.load_table("player_play", args.raw)
    players = T.load_table("players", args.raw)
    seasons = sorted(games["season"].unique().to_list())
    parts = [f"# BDB 2027 -> nflverse join rates\n\n_Generated {date.today().isoformat()} by "
             "`scripts/verify_joins.py`._\n"]

    # games
    sched = keys.game_keys(seasons)
    g = games.join(sched.select("gameId", pl.lit(True).alias("inSched")), on="gameId", how="left") \
             .with_columns(pl.col("inSched").fill_null(False))
    parts += ["\n## gameId -> nflverse schedules (`old_game_id`)\n", _md(rate(g, "inSched"))]

    # plays
    plays = pp.select("gameId", "playId").unique().join(games.select("gameId", "season", "seasonType"), on="gameId")
    for name, ds in (("pbp", "pbp"), ("participation", "participation")):
        src = external.load(ds, seasons).select(pl.col("old_game_id").cast(pl.Int64).alias("gameId"),
                                                pl.col("play_id").cast(pl.Int64).alias("playId")).unique()
        plays = plays.join(src.with_columns(pl.lit(True).alias(f"in_{name}")), on=["gameId", "playId"], how="left")
    ftn = external.load("ftn_charting", seasons).select(pl.col("nflverse_game_id").alias("game_id"),
                                                         pl.col("play_id").cast(pl.Int64).alias("playId")).unique()
    ftn = ftn.join(sched.select("gameId", "game_id"), on="game_id").select("gameId", "playId")
    plays = plays.join(ftn.with_columns(in_ftn=pl.lit(True)), on=["gameId", "playId"], how="left")
    plays = plays.with_columns([pl.col(c).fill_null(False) for c in ("in_pbp", "in_participation", "in_ftn")])
    tab = (plays.group_by("season", "seasonType")
           .agg(plays=pl.len(), pbp=pl.col("in_pbp").mean(), participation=pl.col("in_participation").mean(),
                ftn=pl.col("in_ftn").mean()).sort("season", "seasonType"))
    parts += ["\n## (gameId, playId) of player_play -> pbp / participation / FTN (share of plays)\n", _md(tab)]

    # players
    ref = external.load("players").filter(pl.col("nfl_id_int").is_not_null()).select(
        pl.col("nfl_id_int").alias("nflId"), "gsis_id", "display_name", "draft_year", "draft_pick", "pfr_id")
    pm = players.join(ref, on="nflId", how="left", suffix="_nv")
    has = pm["gsis_id"].is_not_null()
    drafted = pm.filter(pl.col("draftOverallPick").is_not_null() & pl.col("draft_pick").is_not_null())
    pick_ok = (drafted["draftOverallPick"] == drafted["draft_pick"]).mean()
    year_ok = (pm.filter(pl.col("draft_year").is_not_null())
               .select((pl.col("draftYear") == pl.col("draft_year")).mean()).item())
    name_ok = pm.filter(has).select((pl.col("displayName") == pl.col("display_name")).mean()).item()
    combine = external.load("combine").select("pfr_id").drop_nulls().unique()
    in_combine = pm.filter(pl.col("pfr_id").is_not_null()).join(combine, on="pfr_id", how="semi").height
    parts += ["\n## nflId -> nflverse players\n",
              f"- players.csv rows: {players.height}; matched to nflverse `nfl_id`: {has.sum()} ({has.mean():.1%})\n"
              f"- draft overall pick agrees (both drafted): {pick_ok:.1%} of {drafted.height}\n"
              f"- draft year agrees (where nflverse has one): {year_ok:.1%}\n"
              f"- display name identical: {name_ok:.1%} (rest are suffix/nickname variants)\n"
              f"- with a `pfr_id`: {pm['pfr_id'].is_not_null().sum()}; of those in nflverse combine: {in_combine}\n"]
    unmatched = pm.filter(~has).select("nflId", "displayName", "draftYear", "nflPosition")
    if unmatched.height:
        parts += ["\nUnmatched nflIds:\n", _md(unmatched.head(20))]

    # player on field
    onfield = keys.participation_long(seasons).select("gameId", "playId", "nflId").drop_nulls().unique()
    ppl = (pp.select("gameId", "playId", "nflId").join(games.select("gameId", "season", "seasonType"), on="gameId")
           .join(plays.select("gameId", "playId", "in_participation"), on=["gameId", "playId"], how="left")
           .filter(pl.col("in_participation"))
           .join(onfield.with_columns(onField=pl.lit(True)), on=["gameId", "playId", "nflId"], how="left")
           .with_columns(pl.col("onField").fill_null(False)))
    parts += ["\n## (gameId, playId, nflId) -> player listed on field in participation "
              "(plays present in participation only)\n", _md(rate(ppl, "onField"))]

    args.out.write_text("\n".join(parts))
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
