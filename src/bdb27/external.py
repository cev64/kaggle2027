"""Fetch and cache nflverse external data (via nflreadpy) to ``data/external``.

Every dataset is registered in :data:`DATASETS`. Season-scoped datasets are cached
one parquet per season (``data/external/<name>/<season>.parquet``) so that adding a
season never re-downloads the others; season-less datasets (players, combine,
draft picks) are cached as a single ``data/external/<name>.parquet``.

Typical use::

    from bdb27 import external
    pbp = external.load("pbp", [2023, 2024])        # cached after the first call
    players = external.load("players")
    external.fetch_all([2022, 2023, 2024, 2025])     # warm the whole cache
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
DEFAULT_SEASONS: tuple[int, ...] = (2022, 2023, 2024, 2025)

Loader = Callable[[int], pl.DataFrame]


@dataclass(frozen=True)
class Dataset:
    """A cached nflverse dataset.

    ``loader`` takes a season (ignored when ``seasonal`` is False) and returns a frame.
    """

    name: str
    loader: Loader
    seasonal: bool = True
    description: str = ""


# --------------------------------------------------------------------------- cleaning


def _clean_participation(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise participation: integer ``play_id`` (2025 ships it as float), empty strings -> null."""
    str_cols = [c for c, t in df.schema.items() if t == pl.String]
    return df.with_columns(
        pl.col("play_id").cast(pl.Int64),
        *[pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c) for c in str_cols],
    )


def _clean_pbp(df: pl.DataFrame) -> pl.DataFrame:
    """Cast ``play_id`` to Int64 so it joins directly to BDB ``playId``."""
    return df.with_columns(pl.col("play_id").cast(pl.Int64))


def _clean_ftn(df: pl.DataFrame) -> pl.DataFrame:
    """Cast nflverse play id to Int64 and expose it as ``play_id`` for consistency."""
    return df.with_columns(pl.col("nflverse_play_id").cast(pl.Int64).alias("play_id"))


def _clean_players(df: pl.DataFrame) -> pl.DataFrame:
    """Add integer ``nfl_id_int`` (matches BDB ``nflId``) next to the string ``nfl_id``."""
    return df.with_columns(pl.col("nfl_id").cast(pl.Int64, strict=False).alias("nfl_id_int"))


# --------------------------------------------------------------------------- registry


def _seasonal(
    fn: Callable[..., pl.DataFrame], clean: Callable[[pl.DataFrame], pl.DataFrame] | None = None, **kw: str
) -> Loader:
    def load_one(season: int) -> pl.DataFrame:
        df = fn(season, **kw)
        return clean(df) if clean else df

    return load_one


def _static(
    fn: Callable[..., pl.DataFrame], clean: Callable[[pl.DataFrame], pl.DataFrame] | None = None, **kw: object
) -> Loader:
    def load_all(_season: int = 0) -> pl.DataFrame:
        df = fn(**kw)
        return clean(df) if clean else df

    return load_all


def _registry() -> dict[str, Dataset]:
    ds = [
        Dataset(
            "pbp", _seasonal(nfl.load_pbp, _clean_pbp), description="nflfastR play-by-play incl. EPA/WPA/xpass/cp/cpoe"
        ),
        Dataset(
            "participation",
            _seasonal(nfl.load_participation, _clean_participation),
            description="NGS/FTN on-field players, personnel, box, coverage",
        ),
        Dataset(
            "ftn_charting",
            _seasonal(nfl.load_ftn_charting, _clean_ftn),
            description="FTN charting (play action, RPO, blitzers, drops...)",
        ),
        Dataset("rosters", _seasonal(nfl.load_rosters), description="Season rosters"),
        Dataset("rosters_weekly", _seasonal(nfl.load_rosters_weekly), description="Weekly rosters"),
        Dataset(
            "schedules", _seasonal(nfl.load_schedules), description="Games: old_game_id, roof, surface, weather, lines"
        ),
        Dataset("snap_counts", _seasonal(nfl.load_snap_counts), description="PFR snap counts by game"),
        Dataset("injuries", _seasonal(nfl.load_injuries), description="Weekly injury reports"),
        Dataset("depth_charts", _seasonal(nfl.load_depth_charts), description="Depth charts"),
        Dataset(
            "players",
            _static(nfl.load_players, _clean_players),
            seasonal=False,
            description="Player master table with ID crosswalk + bio + draft",
        ),
        Dataset(
            "combine",
            _static(nfl.load_combine, seasons=True),
            seasonal=False,
            description="NFL combine results, all years",
        ),
        Dataset(
            "draft_picks",
            _static(nfl.load_draft_picks, seasons=True),
            seasonal=False,
            description="Draft picks, all years",
        ),
    ]
    for st in ("passing", "rushing", "receiving"):
        ds.append(
            Dataset(
                f"ngs_{st}", _seasonal(nfl.load_nextgen_stats, stat_type=st), description=f"NGS {st} (week 0 = season)"
            )
        )
    for st in ("pass", "rush", "rec", "def"):
        ds.append(
            Dataset(
                f"pfr_{st}",
                _seasonal(nfl.load_pfr_advstats, stat_type=st, summary_level="week"),
                description=f"PFR advanced {st}, weekly",
            )
        )
    return {d.name: d for d in ds}


DATASETS: dict[str, Dataset] = _registry()


# --------------------------------------------------------------------------- cache API


def cache_path(name: str, season: int | None = None) -> Path:
    """Parquet path for ``name`` (and ``season`` for seasonal datasets)."""
    if DATASETS[name].seasonal:
        if season is None:
            raise ValueError(f"{name} is seasonal; pass a season")
        return EXTERNAL_DIR / name / f"{season}.parquet"
    return EXTERNAL_DIR / f"{name}.parquet"


def fetch(name: str, season: int | None = None, refresh: bool = False) -> pl.DataFrame:
    """Return one dataset (one season if seasonal), downloading and caching it if needed."""
    path = cache_path(name, season)
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = DATASETS[name].loader(season or 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def load(name: str, seasons: int | Iterable[int] | None = None, refresh: bool = False) -> pl.DataFrame:
    """Load a dataset for ``seasons`` (default :data:`DEFAULT_SEASONS`) from cache, fetching missing parts.

    Seasons with differing schemas are concatenated with ``diagonal_relaxed``
    (missing columns become null, numeric types are widened).
    """
    if not DATASETS[name].seasonal:
        return fetch(name, refresh=refresh)
    if seasons is None:
        seasons = DEFAULT_SEASONS
    elif isinstance(seasons, int):
        seasons = [seasons]
    frames = [fetch(name, s, refresh) for s in seasons]
    return pl.concat(frames, how="diagonal_relaxed")


def fetch_all(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    names: Iterable[str] | None = None,
    refresh: bool = False,
) -> dict[tuple[str, int | None], int | str]:
    """Download every dataset/season into the cache.

    Returns ``{(name, season): n_rows}``; a failed download maps to its error message
    instead of raising, so one missing release does not abort the batch.
    """
    seasons = list(seasons)
    out: dict[tuple[str, int | None], int | str] = {}
    for name in names or DATASETS:
        targets: list[int | None] = list(seasons) if DATASETS[name].seasonal else [None]
        for s in targets:
            try:
                out[(name, s)] = fetch(name, s, refresh).height
            except Exception as exc:  # noqa: BLE001 - report and continue
                out[(name, s)] = f"ERROR: {type(exc).__name__}: {exc}"[:200]
    return out
