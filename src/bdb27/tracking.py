"""Load, cache and standardize Big Data Bowl tracking data (polars-based).

Schema-tolerant across layouts; all columns are normalized to camelCase
(``game_id`` -> ``gameId``):

* BDB 2027: ``combine_tracking.csv`` (one *drill attempt* per ``eventId``; PLAYER/BALL
  entities; no orientation, no frame id) and ``game_tracking_<season>.csv``
  (cohort players only, no ball/opponents, no frame id or play direction; the
  direction comes from ``player_play.csv``).
* BDB 2025: ``tracking_week_N.csv`` (22 players + football, ``frameId``, ``frameType``).
* BDB 2026: ``train/input_2023_wNN.csv`` / ``output_2023_wNN.csv``.

Grouping is generic: a "play" is ``(gameId, playId)`` or, for Combine data,
``(draftYear, eventId)`` (see :func:`play_key`); an entity within it is
``nflId`` (ball = -1). Missing ``frameId`` is derived from ``time`` at 10 Hz,
preserving gaps.

NGS conventions: x in [0, 120] incl. end zones, y in [0, 53.33]; ``o``/``dir``
are degrees clockwise from +y, so ``vx = s*sin(dir)``, ``vy = s*cos(dir)``.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import polars as pl

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160 / 3  # 53.33 yd
FPS = 10
DT = 1 / FPS
SNAP_EVENTS = ("ball_snap", "snap_direct")
GAME_PLAY_KEY = ["gameId", "playId"]
PLAY_KEY = GAME_PLAY_KEY  # default for game data; prefer play_key(df)

_ALIASES = {"playerName": "displayName", "team": "club", "teamAbbr": "club"}
_DTYPES = {
    "gameId": pl.Int64, "playId": pl.Int32, "nflId": pl.Int32, "frameId": pl.Int16,
    "jerseyNumber": pl.Int16, "week": pl.Int8, "season": pl.Int16, "draftYear": pl.Int16, "attempt": pl.Int16,
    **{c: pl.Float32 for c in ("x", "y", "s", "a", "dis", "o", "dir",
                               "ballLandX", "ballLandY", "absoluteYardlineNumber")},
    **{c: pl.Categorical for c in ("club", "playDirection", "frameType", "event", "displayName",
                                   "playerRole", "playerSide", "playerPosition", "entityType",
                                   "drillType", "drillName")},
}
TRACKING_KINDS = ("combine_tracking", "game_tracking", "tracking", "input", "output")
_TABLE_STEMS = {
    "games": ["games"], "plays": ["plays"], "players": ["players"], "player_play": ["player_play"],
    "supplementary": ["supplementary_data", "supplementary"], "combine_results": ["combine_results"],
    "career": ["player_career_successes"],
}


# ------------------------------------------------------------------------- keys
def _cols(df) -> list[str]:
    return df.collect_schema().names() if hasattr(df, "collect_schema") else list(df.columns)


def play_key(df) -> list[str]:
    """Columns identifying one play / drill attempt in ``df``."""
    cols = _cols(df)
    if {"gameId", "playId"} <= set(cols):
        return list(GAME_PLAY_KEY)
    if "eventId" in cols:
        return ["draftYear", "eventId"] if "draftYear" in cols else ["eventId"]
    raise ValueError(f"Cannot identify plays: need (gameId, playId) or eventId. Columns: {cols}")


def frame_key(df) -> list[str]:
    return play_key(df) + ["frameId"]


def pid_expr() -> pl.Expr:
    """Entity id within a play: nflId, with the ball (or unknown) as -1."""
    return pl.when(pl.col("isBall")).then(-1).otherwise(pl.col("nflId").fill_null(-1)).cast(pl.Int32).alias("_pid")


def entity_key(df) -> list[str]:
    """Group columns for one entity trajectory (requires the ``_pid`` helper column)."""
    return play_key(df) + ["_pid"]


def with_pid(df: pl.DataFrame) -> pl.DataFrame:
    return df if "_pid" in df.columns else df.with_columns(pid_expr())


# ----------------------------------------------------------------------------- io
def camelize(name: str) -> str:
    """``absolute_yardline_number`` -> ``absoluteYardlineNumber``; camelCase passes through."""
    out = re.sub(r"_([a-zA-Z0-9])", lambda m: m.group(1).upper(), name.strip())
    return _ALIASES.get(out, out)


def normalize_columns(df):
    """Camel-case names, cast known columns to compact dtypes, parse ``time``, add ``isBall``."""
    df = df.rename({c: camelize(c) for c in _cols(df) if camelize(c) != c})
    cols = _cols(df)
    casts = [pl.col(c).cast(t, strict=False) for c, t in _DTYPES.items() if c in cols]
    if "time" in cols and df.collect_schema()["time"] == pl.String:
        casts.append(pl.col("time").str.replace(r"(Z|[+-]\d{2}:?\d{2})$", "").str.to_datetime(strict=False))
    df = df.with_columns(casts)
    if "isBall" not in cols and ({"nflId", "entityType", "displayName", "club"} & set(cols)):
        ball = pl.col("nflId").is_null() if "nflId" in cols else pl.lit(False)
        if "entityType" in cols:  # Combine: explicit PLAYER/BALL
            ball = pl.col("entityType").cast(pl.String).str.to_uppercase() == "BALL"
        for c in ("displayName", "club"):
            if c in cols:
                ball = ball | (pl.col(c).cast(pl.String).str.to_lowercase() == "football").fill_null(False)
        df = df.with_columns(isBall=ball.fill_null(False))
    return df


def add_frame_ids(df):
    """Derive ``frameId`` (1-based, 10 Hz, gaps preserved) and ``tSec`` from ``time`` per play."""
    cols = _cols(df)
    if "time" not in cols:
        if "frameId" not in cols:
            raise ValueError("Tracking has neither 'frameId' nor 'time'; cannot order frames")
        return df.with_columns(tSec=((pl.col("frameId") - pl.col("frameId").min().over(play_key(df))) * DT)
                               .cast(pl.Float32))
    t = (pl.col("time") - pl.col("time").min().over(play_key(df))).dt.total_milliseconds()
    out = df.with_columns(tSec=(t / 1000).cast(pl.Float32))
    if "frameId" not in cols:
        out = out.with_columns(frameId=((t / (1000 * DT)).round() + 1).cast(pl.Int16))
    return out


def require_columns(df, required, what: str = "data") -> None:
    """Raise a clear error if any required column is missing."""
    missing = [c for c in required if c not in _cols(df)]
    if missing:
        raise ValueError(f"{what} is missing required columns {missing}. Columns present: {_cols(df)}")


def discover_files(raw_dir: str | Path = "data/raw") -> dict[str, list[Path]]:
    """Find tracking files (by kind) and context tables (CSV or parquet) under ``raw_dir``."""
    files = sorted(p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in (".csv", ".parquet"))
    out: dict[str, list[Path]] = {k: [] for k in TRACKING_KINDS}
    for p in files:
        stem = p.stem.lower()
        kind = next((k for k in TRACKING_KINDS if stem.startswith(k)), None)
        if kind:
            out[kind].append(p)
        for name, stems in _TABLE_STEMS.items():
            if stem in stems:
                out.setdefault(name, []).append(p)
    out["all"] = files
    return out


def week_from_path(path) -> int | None:
    """Week from names like ``tracking_week_3`` / ``input_2023_w03``."""
    m = re.search(r"(?:week_?|_w)(\d+)", Path(path).stem.lower())
    return int(m.group(1)) if m else None


def season_from_path(path) -> int | None:
    """Season from names like ``game_tracking_2024``."""
    m = re.search(r"(20\d{2})", Path(path).stem)
    return int(m.group(1)) if m else None


def _read_any(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, null_values=["NA", "", "NaN", "nan", "NULL"], infer_schema_length=100_000)


def _tag_file(df, path: Path):
    cols = _cols(df)
    if "week" not in cols and (wk := week_from_path(path)) is not None:
        df = df.with_columns(week=pl.lit(wk, pl.Int8))
    if path.stem.lower().startswith("game_tracking") and "season" not in cols and (s := season_from_path(path)):
        df = df.with_columns(season=pl.lit(s, pl.Int16))
    return df


def _prep_file(df: pl.DataFrame, path: Path) -> pl.DataFrame:
    df = _tag_file(normalize_columns(df), path)
    if "x" in df.columns and ("frameId" not in df.columns or "tSec" not in df.columns):
        try:
            df = add_frame_ids(df)
        except ValueError:
            pass
    return df


def cached_frame(path: Path, cache_dir, refresh: bool = False) -> pl.LazyFrame:
    """Lazy frame for ``path`` via a normalized parquet cache (rebuilt when stale)."""
    if cache_dir is None:
        return _prep_file(_read_any(path), path).lazy()
    cache = Path(cache_dir) / f"{path.stem}.parquet"
    if refresh or not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _prep_file(_read_any(path), path).write_parquet(cache, compression="zstd")
    return pl.scan_parquet(cache)


def load_tracking(raw_dir="data/raw", cache_dir="data/processed/cache", kind: str = "auto",
                  weeks=None, seasons=None, lazy: bool = False, refresh: bool = False):
    """Load tracking frames of one ``kind`` (all files, or selected weeks/seasons).

    ``kind``: ``combine_tracking``, ``game_tracking``, ``tracking`` (BDB 2025),
    ``input``/``output`` (BDB 2026) or ``auto`` (first kind present in that order).
    Returns a polars DataFrame (LazyFrame if ``lazy``) with normalized columns,
    ``frameId``/``tSec`` and ``week``/``season`` where derivable.
    """
    found = discover_files(raw_dir)
    if kind == "auto":
        kind = next((k for k in TRACKING_KINDS if found[k]), TRACKING_KINDS[0])
    paths = [p for p in found.get(kind, [])
             if (weeks is None or week_from_path(p) in set(weeks))
             and (seasons is None or season_from_path(p) in set(seasons))]
    if not paths:
        seen = [str(p.relative_to(raw_dir)) for p in found["all"]][:30]
        raise FileNotFoundError(
            f"No '{kind}' tracking files under {Path(raw_dir).resolve()} (weeks={weeks}, seasons={seasons}). "
            f"Expected combine_tracking.csv / game_tracking_YYYY.csv (BDB 2027), tracking_week_*.csv (2025) "
            f"or train/input_*_w*.csv (2026). Data files found: {seen or 'none - download the competition data first'}")
    lf = pl.concat([cached_frame(p, cache_dir, refresh) for p in paths], how="diagonal_relaxed")
    require_columns(lf, ["frameId", "x", "y"], f"{kind} tracking")
    play_key(lf)  # validates key columns
    return lf if lazy else lf.collect()


def load_table(name: str, raw_dir="data/raw") -> pl.DataFrame:
    """Load a context table (games, plays, players, player_play, supplementary, combine_results, career)."""
    paths = discover_files(raw_dir).get(name, [])
    if not paths:
        raise FileNotFoundError(f"Table '{name}' not found under {raw_dir} (looked for {_TABLE_STEMS.get(name)})")
    return normalize_columns(_read_any(paths[0]))


def stitch_2026(inp: pl.DataFrame, out: pl.DataFrame) -> pl.DataFrame:
    """Append BDB 2026 output frames (frameId restarts at 1 after the throw) to input frames."""
    key = GAME_PLAY_KEY
    last = inp.group_by(key).agg(lastInput=pl.col("frameId").max())
    static = [c for c in inp.columns if c not in (*key, "nflId", "frameId", "x", "y", "s", "a", "o", "dir", "dis",
                                                  "week", "tSec", "time")]
    meta = inp.sort("frameId").group_by(key + ["nflId"]).agg([pl.col(c).last() for c in static])
    out = (out.join(last, on=key).with_columns(frameId=(pl.col("frameId") + pl.col("lastInput")).cast(pl.Int16))
           .drop("lastInput").join(meta, on=key + ["nflId"], how="left"))
    return pl.concat([inp.with_columns(segment=pl.lit("input")), out.with_columns(segment=pl.lit("output"))],
                     how="diagonal_relaxed").sort(key + ["frameId", "nflId"])


# ---------------------------------------------------------------- game standardization
def attach_play_context(df: pl.DataFrame, plays: pl.DataFrame, cols=None) -> pl.DataFrame:
    """Left-join play context (direction, possession, LOS, down, ...) onto tracking.

    ``plays`` may be play-level (BDB 2025 plays.csv) or player-play level (BDB 2027
    player_play.csv). For the latter, player-level ``club``/``linedUpPosition`` are
    joined on nflId and play-level columns reduced to one row per play.
    """
    plays = normalize_columns(plays)
    if "nflId" in plays.columns and "nflId" in df.columns:
        pcols = [c for c in ("club", "linedUpPosition") if c in plays.columns and c not in df.columns]
        if pcols:
            df = df.join(plays.select(GAME_PLAY_KEY + ["nflId"] + pcols).unique(GAME_PLAY_KEY + ["nflId"]),
                         on=GAME_PLAY_KEY + ["nflId"], how="left")
    cols = cols or [c for c in ("playDirection", "possessionTeam", "defensiveTeam", "absoluteYardlineNumber",
                                "yardlineNumber", "yardlineSide", "yardsToGo", "down", "quarter") if c in plays.columns]
    cols = [c for c in cols if c not in df.columns]
    if not cols:
        return df
    ctx = plays.select(GAME_PLAY_KEY + cols).unique(subset=GAME_PLAY_KEY, keep="first")
    return df.join(ctx, on=GAME_PLAY_KEY, how="left")


def add_side(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``side`` in {offense, defense, ball} from ``playerSide`` or team vs ``possessionTeam``."""
    cols = df.columns
    team = next((c for c in ("club", "teamAbbr") if c in cols), None)
    if "playerSide" in cols:
        side = pl.col("playerSide").cast(pl.String).str.to_lowercase()
    elif team and "possessionTeam" in cols:
        side = pl.when(pl.col(team).cast(pl.String) == pl.col("possessionTeam").cast(pl.String)) \
                 .then(pl.lit("offense")).otherwise(pl.lit("defense"))
    else:
        raise ValueError("add_side needs 'playerSide' or ('club'/'teamAbbr' and 'possessionTeam')")
    if "isBall" in cols:
        side = pl.when(pl.col("isBall")).then(pl.lit("ball")).otherwise(side)
    return df.with_columns(side=side)


def standardize_direction(df: pl.DataFrame) -> pl.DataFrame:
    """Rotate plays with ``playDirection == 'left'`` by 180 deg so offense moves toward +x.

    x -> 120 - x, y -> 53.33 - y, o/dir -> (angle + 180) mod 360 (a point reflection
    reverses every heading). Also flips ``ballLandX/Y`` and ``absoluteYardlineNumber``
    if present. Adds ``flipped``; refuses to run twice.
    """
    require_columns(df, ["playDirection", "x", "y"], "standardize_direction input")
    if "flipped" in df.columns:
        raise ValueError("Data already standardized (column 'flipped' exists)")
    left = (pl.col("playDirection").cast(pl.String).str.to_lowercase() == "left").fill_null(False)
    flip = {"x": FIELD_LENGTH, "ballLandX": FIELD_LENGTH, "absoluteYardlineNumber": FIELD_LENGTH,
            "y": FIELD_WIDTH, "ballLandY": FIELD_WIDTH}
    exprs = [pl.when(left).then(L - pl.col(c)).otherwise(pl.col(c)).cast(pl.Float32).alias(c)
             for c, L in flip.items() if c in df.columns]
    exprs += [pl.when(left).then((pl.col(c) + 180) % 360).otherwise(pl.col(c)).alias(c)
              for c in ("o", "dir") if c in df.columns]
    return df.with_columns(exprs).with_columns(flipped=left)


def add_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Add velocity (vx, vy) and acceleration (ax, ay, aLong, aLat) components.

    vx/vy come from ``s`` and ``dir`` (falling back to position differences, e.g.
    for the ball). ax/ay are gap-aware central differences of vx/vy. aLong is
    along the direction of travel (negative = braking); aLat > 0 = turning
    counter-clockwise.
    """
    df = with_pid(df).sort(frame_key(df))
    grp = entity_key(df)
    fid = pl.col("frameId").cast(pl.Float32)
    dt_c = (fid.shift(-1) - fid.shift(1)).over(grp) * DT
    dt_f = (fid.shift(-1) - fid).over(grp) * DT
    dt_b = (fid - fid.shift(1)).over(grp) * DT

    def deriv(c):  # central difference, one-sided at the ends
        c_ = (pl.col(c).shift(-1) - pl.col(c).shift(1)).over(grp) / dt_c
        f_ = (pl.col(c).shift(-1) - pl.col(c)).over(grp) / dt_f
        b_ = (pl.col(c) - pl.col(c).shift(1)).over(grp) / dt_b
        return pl.coalesce(c_, f_, b_)

    if {"s", "dir"} <= set(df.columns):
        rad = pl.col("dir").radians()
        df = df.with_columns(vx=pl.coalesce(pl.col("s") * rad.sin(), deriv("x")),
                             vy=pl.coalesce(pl.col("s") * rad.cos(), deriv("y")))
    else:
        df = df.with_columns(vx=deriv("x"), vy=deriv("y"))
    df = df.with_columns(ax=deriv("vx"), ay=deriv("vy"))
    spd = (pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt()
    df = df.with_columns(
        aLong=pl.when(spd > 0.05).then((pl.col("vx") * pl.col("ax") + pl.col("vy") * pl.col("ay")) / spd),
        aLat=pl.when(spd > 0.05).then((pl.col("vx") * pl.col("ay") - pl.col("vy") * pl.col("ax")) / spd),
    )
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in ("vx", "vy", "ax", "ay", "aLong", "aLat")]).drop("_pid")


def add_snap_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-play ``snapFrame`` and ``framesFromSnap`` (null when no snap is tagged)."""
    cols, key = df.columns, play_key(df)
    is_snap = pl.lit(False)
    if "event" in cols:
        is_snap = is_snap | pl.col("event").cast(pl.String).is_in(SNAP_EVENTS).fill_null(False)
    if "frameType" in cols:
        is_snap = is_snap | (pl.col("frameType").cast(pl.String) == "SNAP").fill_null(False)
    snap = df.filter(is_snap).group_by(key).agg(snapFrame=pl.col("frameId").min())
    df = df.drop("snapFrame", strict=False).join(snap, on=key, how="left")
    return df.with_columns(framesFromSnap=(pl.col("frameId").cast(pl.Int32) - pl.col("snapFrame")).cast(pl.Int16))


def frames_from_event(df: pl.DataFrame, event: str, name: str | None = None) -> pl.DataFrame:
    """Add frame index relative to the first frame tagged ``event`` in each play."""
    key = play_key(df)
    first = df.filter(pl.col("event").cast(pl.String) == event).group_by(key).agg(_ef=pl.col("frameId").min())
    name = name or f"framesFrom_{event}"
    return (df.join(first, on=key, how="left")
            .with_columns((pl.col("frameId").cast(pl.Int32) - pl.col("_ef")).cast(pl.Int16).alias(name)).drop("_ef"))


def add_los(df: pl.DataFrame) -> pl.DataFrame:
    """Add line of scrimmage ``losX`` plus ``xRelLos`` (downfield +) and ``yRelCenter``.

    Uses the ball's x at the snap frame when ball rows exist, else the (already
    standardized) ``absoluteYardlineNumber``; ``losSource`` records which. Run after
    :func:`standardize_direction` and :func:`add_snap_frame`. No-op (with nulls)
    when neither source exists, e.g. cohort-only game tracking without context.
    """
    key = play_key(df)
    if "absoluteYardlineNumber" in df.columns and "flipped" not in df.columns:
        raise ValueError("add_los: run standardize_direction first so absoluteYardlineNumber is flipped")
    if "isBall" in df.columns and "snapFrame" in df.columns:
        los = (df.filter(pl.col("isBall") & (pl.col("frameId") == pl.col("snapFrame")))
               .group_by(key).agg(_ballLos=pl.col("x").first()))
        df = df.join(los, on=key, how="left")
    else:
        df = df.with_columns(_ballLos=pl.lit(None, pl.Float32))
    ayn = pl.col("absoluteYardlineNumber") if "absoluteYardlineNumber" in df.columns else pl.lit(None, pl.Float32)
    return df.with_columns(
        losX=pl.coalesce(pl.col("_ballLos"), ayn).cast(pl.Float32),
        losSource=pl.when(pl.col("_ballLos").is_not_null()).then(pl.lit("ball"))
                    .when(ayn.is_not_null()).then(pl.lit("yardline")),
    ).with_columns(
        xRelLos=(pl.col("x") - pl.col("losX")).cast(pl.Float32),
        yRelCenter=(pl.col("y") - FIELD_WIDTH / 2).cast(pl.Float32),
    ).drop("_ballLos")


# ---------------------------------------------------------------- drill normalization
def normalize_drill(df: pl.DataFrame, min_disp: float = 2.0, mirror: bool = False) -> pl.DataFrame:
    """Drill-relative coordinates for Combine attempts (no play direction / LOS exists).

    Per attempt, translates so the primary player (first non-ball entity) starts at
    the origin and rotates so his initial direction of travel (start -> first point
    >= ``min_disp`` yd away) is +x. Adds ``xd``, ``yd``, ``dird`` (heading in the same
    clockwise-from-+y convention) and ``drillRotation`` (deg, counter-clockwise).
    With ``mirror``, attempts whose first lateral move is to -y are reflected so it
    is +y (heading mirrored accordingly); ``drillMirrored`` records it.
    """
    key = play_key(df)
    df = with_pid(df).sort(frame_key(df))
    prim = df.filter(~pl.col("isBall")).group_by(key).agg(_prim=pl.col("_pid").first())
    p = df.join(prim, on=key).filter(pl.col("_pid") == pl.col("_prim"))
    d = ((pl.col("x") - pl.col("x").first()) ** 2 + (pl.col("y") - pl.col("y").first()) ** 2).sqrt()
    ref = p.group_by(key, maintain_order=True).agg(
        _x0=pl.col("x").first(), _y0=pl.col("y").first(),
        _hx=pl.col("x").filter(d >= min_disp).first().fill_null(pl.col("x").last()),
        _hy=pl.col("y").filter(d >= min_disp).first().fill_null(pl.col("y").last()))
    ref = ref.with_columns(_th=-pl.arctan2(pl.col("_hy") - pl.col("_y0"), pl.col("_hx") - pl.col("_x0")))
    out = df.join(ref, on=key, how="left")
    c, s = pl.col("_th").cos(), pl.col("_th").sin()
    dx, dy = pl.col("x") - pl.col("_x0"), pl.col("y") - pl.col("_y0")
    out = out.with_columns(xd=(dx * c - dy * s).cast(pl.Float32), yd=(dx * s + dy * c).cast(pl.Float32),
                           drillRotation=pl.col("_th").degrees().cast(pl.Float32))
    if "dir" in out.columns:  # rotating the frame CCW by th turns clockwise headings by -th
        out = out.with_columns(dird=((pl.col("dir") - pl.col("drillRotation")) % 360).cast(pl.Float32))
    if mirror:
        prim_lat = (out.join(prim, on=key).filter((pl.col("_pid") == pl.col("_prim")) & (pl.col("yd").abs() > 0.5))
                    .sort(frame_key(out)).group_by(key).agg(_sgn=pl.col("yd").first().sign()))
        out = out.join(prim_lat, on=key, how="left").with_columns(drillMirrored=(pl.col("_sgn") < 0).fill_null(False))
        flip = pl.col("drillMirrored")
        out = out.with_columns(yd=pl.when(flip).then(-pl.col("yd")).otherwise(pl.col("yd")))
        if "dird" in out.columns:
            out = out.with_columns(dird=pl.when(flip).then((180 - pl.col("dird")) % 360).otherwise(pl.col("dird")))
        out = out.drop("_sgn")
    return out.drop("_x0", "_y0", "_hx", "_hy", "_th", "_pid")


# ---------------------------------------------------------------------- pipeline
def prepare_tracking(df: pl.DataFrame, plays: pl.DataFrame | None = None) -> pl.DataFrame:
    """Normalize -> context -> side -> standardize -> kinematics -> snap -> LOS (game data),
    or normalize -> kinematics -> drill coordinates (Combine data). Steps whose inputs
    are missing are skipped with a warning."""
    df = normalize_columns(df)
    if "frameId" not in df.columns or "tSec" not in df.columns:
        df = add_frame_ids(df)
    if "eventId" in df.columns and "gameId" not in df.columns:
        return normalize_drill(add_kinematics(df))
    if plays is not None:
        df = attach_play_context(df, plays)
    try:
        df = add_side(df)
    except ValueError:
        pass
    if "playDirection" in df.columns:
        df = standardize_direction(df)
    else:
        warnings.warn("No playDirection (attach player_play/plays context); coordinates left unstandardized")
    df = add_kinematics(df)
    df = add_snap_frame(df)
    return add_los(df)

