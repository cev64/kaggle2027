"""Join keys between Big Data Bowl (NGS tracking) data and nflverse.

Verified key mapping (see ``docs/external_data.md`` for the evidence):

=================  ===========================================  =====================================
BDB column         nflverse column                              Notes
=================  ===========================================  =====================================
``gameId`` (int)   ``old_game_id`` (str) in schedules / pbp /   ``YYYYMMDDNN``; cast to str (or cast
                   participation; ``game_id`` = ``2023_01_AR..``  nflverse to int). Unique per game.
``playId`` (int)   ``play_id`` in pbp / participation / FTN     GSIS play id; unique within a game.
                   (FTN: ``nflverse_play_id``)                  FTN keys on ``game_id`` not
                                                                ``old_game_id``.
``nflId`` (int)    ``players.nfl_id`` (str) ==                  1:1 with ``gsis_id``; verified 100%
                   ``rosters.gsis_it_id`` (str)                 against NGS ids embedded in 2022
                                                                participation ``players_on_play``.
=================  ===========================================  =====================================

Every other nflverse player table (pbp ``*_player_id``, participation ``offense_players``,
rosters, depth charts, injuries, NGS ``player_gsis_id``) keys on ``gsis_id`` (``00-00xxxxx``);
use :func:`player_crosswalk` to go between ``nflId`` and ``gsis_id``.

Helpers accept pandas or polars frames and return the same type they were given.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

import pandas as pd
import polars as pl

from . import external

Frame = TypeVar("Frame", pd.DataFrame, pl.DataFrame)

PBP_COLS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "yardline_100",
    "down",
    "ydstogo",
    "play_type",
    "shotgun",
    "no_huddle",
    "qb_dropback",
    "qb_scramble",
    "pass_attempt",
    "complete_pass",
    "sack",
    "interception",
    "pass_location",
    "pass_length",
    "air_yards",
    "yards_after_catch",
    "yards_gained",
    "penalty",
    "success",
    "epa",
    "wpa",
    "wp",
    "vegas_wp",
    "xpass",
    "pass_oe",
    "cp",
    "cpoe",
    "xyac_epa",
    "xyac_mean_yardage",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
)
PARTICIPATION_COLS: tuple[str, ...] = (
    "offense_formation",
    "offense_personnel",
    "defense_personnel",
    "defenders_in_box",
    "number_of_pass_rushers",
    "defense_man_zone_type",
    "defense_coverage_type",
    "route",
    "was_pressure",
    "time_to_throw",
)
FTN_COLS: tuple[str, ...] = (
    "starting_hash",
    "qb_location",
    "n_offense_backfield",
    "n_defense_box",
    "is_motion",
    "is_play_action",
    "is_screen_pass",
    "is_rpo",
    "is_qb_out_of_pocket",
    "is_throw_away",
    "read_thrown",
    "is_catchable_ball",
    "is_contested_ball",
    "is_drop",
    "is_interception_worthy",
    "is_created_reception",
    "n_blitzers",
    "n_pass_rushers",
)
SCHEDULE_COLS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
)
PLAYER_COLS: tuple[str, ...] = (
    "gsis_id",
    "esb_id",
    "pff_id",
    "pfr_id",
    "display_name",
    "position",
    "position_group",
    "ngs_position",
    "height",
    "weight",
    "birth_date",
    "college_name",
    "rookie_season",
    "years_of_experience",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_team",
)

SUFFIX = "_nflverse"


# --------------------------------------------------------------------------- utilities


def _to_pl(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df


def _like(template: Frame, out: pl.DataFrame) -> Frame:
    return out.to_pandas() if isinstance(template, pd.DataFrame) else out  # type: ignore[return-value]


def season_from_game_id(game_id: int) -> int:
    """NFL season of a BDB ``gameId`` (``YYYYMMDDNN``); Jan-Mar games belong to the prior season."""
    year, month = game_id // 1_000_000, (game_id // 10_000) % 100
    return year - 1 if month <= 3 else year


def seasons_in(df: pd.DataFrame | pl.DataFrame, col: str = "gameId") -> list[int]:
    """Distinct seasons covered by the ``gameId`` column of ``df``."""
    ids = _to_pl(df).get_column(col).cast(pl.Int64).unique().to_list()
    return sorted({season_from_game_id(g) for g in ids})


def _select(df: pl.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# --------------------------------------------------------------------------- games / plays


def game_keys(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    """``gameId`` (Int64) <-> nflverse ``game_id`` crosswalk from schedules."""
    return external.load("schedules", seasons).select(
        pl.col("old_game_id").cast(pl.Int64).alias("gameId"), "game_id", "season", "week", "game_type"
    )


def merge_games(games: Frame, cols: Sequence[str] = SCHEDULE_COLS) -> Frame:
    """Left-join schedule info (roof, surface, weather, lines...) onto a frame with ``gameId``."""
    left = _to_pl(games)
    sched = external.load("schedules", seasons_in(left))
    right = sched.select(pl.col("old_game_id").cast(pl.Int64).alias("gameId"), *_select(sched, cols))
    out = left.with_columns(pl.col("gameId").cast(pl.Int64)).join(right, on="gameId", how="left", suffix=SUFFIX)
    return _like(games, out)


def merge_plays(
    plays: Frame,
    pbp_cols: Sequence[str] = PBP_COLS,
    participation_cols: Sequence[str] = PARTICIPATION_COLS,
    ftn_cols: Sequence[str] = FTN_COLS,
) -> Frame:
    """Left-join nflfastR pbp, participation and FTN columns onto a BDB plays frame.

    ``plays`` needs ``gameId`` and ``playId``. Columns that already exist in ``plays``
    get the ``_nflverse`` suffix. Pass an empty sequence to skip a source.
    """
    left = _to_pl(plays).with_columns(pl.col("gameId").cast(pl.Int64), pl.col("playId").cast(pl.Int64))
    seasons = seasons_in(left)
    key = [pl.col("old_game_id").cast(pl.Int64).alias("gameId"), pl.col("play_id").alias("playId")]

    pbp = external.load("pbp", seasons)
    out = left.join(
        pbp.select(*key, pl.col("game_id").alias("_gid"), *_select(pbp, pbp_cols)),
        on=["gameId", "playId"],
        how="left",
        suffix=SUFFIX,
    )
    if participation_cols:
        part = external.load("participation", seasons)
        out = out.join(
            part.select(*key, *_select(part, participation_cols)), on=["gameId", "playId"], how="left", suffix=SUFFIX
        )
    if ftn_cols:
        ftn = external.load("ftn_charting", seasons)
        right = ftn.select(
            pl.col("nflverse_game_id").alias("_gid"), pl.col("play_id").alias("playId"), *_select(ftn, ftn_cols)
        )
        out = out.join(right, on=["_gid", "playId"], how="left", suffix=SUFFIX)
    out = out.drop("_gid")
    return _like(plays, out)


# --------------------------------------------------------------------------- players


def player_crosswalk() -> pl.DataFrame:
    """``nflId`` (Int64) <-> ``gsis_id`` / ``esb_id`` / ``pff_id`` / ``pfr_id`` / ``smart_id``."""
    return (
        external.load("players")
        .filter(pl.col("nfl_id_int").is_not_null())
        .select(pl.col("nfl_id_int").alias("nflId"), "gsis_id", "esb_id", "smart_id", "pff_id", "pfr_id", "espn_id")
    )


def merge_players(players: Frame, cols: Sequence[str] = PLAYER_COLS) -> Frame:
    """Left-join nflverse bio / draft / ID columns onto a frame with ``nflId``."""
    left = _to_pl(players).with_columns(pl.col("nflId").cast(pl.Int64))
    ref = external.load("players")
    right = ref.filter(pl.col("nfl_id_int").is_not_null()).select(
        pl.col("nfl_id_int").alias("nflId"), *_select(ref, cols)
    )
    return _like(players, left.join(right, on="nflId", how="left", suffix=SUFFIX))


def participation_long(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    """One row per (gameId, playId, player) on the field, with ``nflId`` and side.

    ``offense_players``/``defense_players`` (gsis ids) are positionally aligned with the
    ``*_positions``/``*_names``/``*_numbers`` lists, which exist from 2023 on (null in 2022).
    """
    part = external.load("participation", seasons)
    sides = []
    for side in ("offense", "defense"):
        extras = {f"{side}_positions": "position", f"{side}_names": "name", f"{side}_numbers": "jersey_number"}
        extras = {k: v for k, v in extras.items() if k in part.columns}
        f = (
            part.filter(pl.col(f"{side}_players").is_not_null())
            .select(
                pl.col("old_game_id").cast(pl.Int64).alias("gameId"),
                pl.col("play_id").alias("playId"),
                pl.lit(side).alias("side"),
                pl.col(f"{side}_players").str.split(";").alias("gsis_id"),
                *[pl.col(k).str.split(";").alias(v) for k, v in extras.items()],
            )
            .with_columns(pl.int_ranges(pl.col("gsis_id").list.len()).alias("_slot"))
            .explode("gsis_id", "_slot")
            .with_columns([pl.col(v).list.get(pl.col("_slot"), null_on_oob=True) for v in extras.values()])
            .drop("_slot")
        )
        sides.append(f)
    long = pl.concat(sides, how="diagonal_relaxed")
    return long.join(player_crosswalk().select("nflId", "gsis_id"), on="gsis_id", how="left")
