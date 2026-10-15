"""Load, cache and standardize Big Data Bowl tracking data (polars-based).

Schema-tolerant: handles the BDB 2025 layout (``tracking_week_N.csv``, camelCase)
and the BDB 2026 layout (``train/input_2023_wNN.csv`` / ``output_2023_wNN.csv``,
snake_case). All columns are normalized to camelCase (``game_id`` -> ``gameId``).

Coordinate conventions (NGS): x in [0, 120] along the field incl. end zones,
y in [0, 53.33] across it; ``o`` and ``dir`` are degrees clockwise from the +y
axis, so ``vx = s*sin(dir)``, ``vy = s*cos(dir)``. Frames are 10 Hz.

Typical use::

    trk = load_tracking("data/raw")                 # all weeks, parquet-cached
    plays = load_table("plays", "data/raw")
    trk = prepare_tracking(trk, plays)              # standardized + derived cols
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160 / 3  # 53.33 yd
FPS = 10
DT = 1 / FPS
SNAP_EVENTS = ("ball_snap", "snap_direct")
PLAY_KEY = ["gameId", "playId"]
FRAME_KEY = ["gameId", "playId", "frameId"]
# Ball rows have a null nflId; use -1 as the per-entity group key.
PID = pl.col("nflId").fill_null(-1)
ENTITY_KEY = [pl.col("gameId"), pl.col("playId"), PID]

_ALIASES = {"playerName": "displayName", "team": "club"}
_DTYPES = {
    "gameId": pl.Int64, "playId": pl.Int32, "nflId": pl.Int32, "frameId": pl.Int16,
    "jerseyNumber": pl.Int16, "week": pl.Int8,
    **{c: pl.Float32 for c in ("x", "y", "s", "a", "dis", "o", "dir",
                               "ballLandX", "ballLandY", "absoluteYardlineNumber")},
    **{c: pl.Categorical for c in ("club", "playDirection", "frameType", "event",
                                   "displayName", "playerRole", "playerSide",
                                   "playerPosition")},
}
_TABLE_STEMS = {
    "games": ["games"], "plays": ["plays"], "players": ["players"],
    "player_play": ["player_play"], "supplementary": ["supplementary_data", "supplementary"],
}


# ----------------------------------------------------------------------------- io
def camelize(name: str) -> str:
    """``absolute_yardline_number`` -> ``absoluteYardlineNumber``; camelCase passes through."""
    out = re.sub(r"_([a-zA-Z0-9])", lambda m: m.group(1).upper(), name.strip())
    return _ALIASES.get(out, out)


def normalize_columns(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Camel-case column names, cast known columns to compact dtypes, add ``isBall``."""
    cols = df.collect_schema().names()
    df = df.rename({c: camelize(c) for c in cols if camelize(c) != c})
    cols = df.collect_schema().names()
    casts = [pl.col(c).cast(t, strict=False) for c, t in _DTYPES.items() if c in cols]
    if "time" in cols and df.collect_schema()["time"] == pl.String:
        casts.append(pl.col("time").str.to_datetime(strict=False))
    df = df.with_columns(casts)
    if "nflId" in cols and "isBall" not in cols:
        ball = pl.col("nflId").is_null()
        for c in ("displayName", "club"):
            if c in cols:
                ball = ball | (pl.col(c).cast(pl.String).str.to_lowercase() == "football").fill_null(False)
        df = df.with_columns(isBall=ball)
    return df


def require_columns(df: pl.DataFrame | pl.LazyFrame, required, what: str = "data") -> None:
    """Raise a clear error if any required column is missing."""
    cols = df.collect_schema().names()
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{what} is missing required columns {missing}. Columns present: {cols}")


def discover_files(raw_dir: str | Path = "data/raw") -> dict[str, list[Path]]:
    """Find tracking / input / output / table CSVs (or parquet) anywhere under ``raw_dir``."""
    raw = Path(raw_dir)
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in (".csv", ".parquet"))
    out: dict[str, list[Path]] = {"tracking": [], "input": [], "output": []}
    for p in files:
        stem = p.stem.lower()
        if stem.startswith("tracking"):
            out["tracking"].append(p)
        elif stem.startswith("input"):
            out["input"].append(p)
        elif stem.startswith("output"):
            out["output"].append(p)
        else:
            for name, stems in _TABLE_STEMS.items():
                if stem in stems:
                    out.setdefault(name, []).append(p)
    out["all"] = files
    return out


def week_from_path(path: str | Path) -> int | None:
    """Parse a week number from names like ``tracking_week_3`` or ``input_2023_w03``."""
    m = re.search(r"(?:week_?|_w)(\d+)", Path(path).stem.lower())
    return int(m.group(1)) if m else None


def _read_any(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, null_values=["NA", "", "NaN", "nan"], infer_schema_length=50_000)


def cached_frame(path: Path, cache_dir: str | Path | None, refresh: bool = False) -> pl.LazyFrame:
    """Return a lazy frame for ``path`` via a normalized parquet cache (rebuilt if stale)."""
    if cache_dir is None or path.suffix.lower() == ".parquet":
        return normalize_columns(_read_any(path)).lazy()
    cache = Path(cache_dir) / f"{path.stem}.parquet"
    if refresh or not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df = normalize_columns(_read_any(path))
        if "week" not in df.columns and (wk := week_from_path(path)) is not None:
            df = df.with_columns(week=pl.lit(wk, pl.Int8))
        df.write_parquet(cache, compression="zstd")
    return pl.scan_parquet(cache)


def load_tracking(raw_dir: str | Path = "data/raw", cache_dir: str | Path | None = "data/processed/cache",
                  kind: str = "auto", weeks=None, lazy: bool = False, refresh: bool = False):
    """Load tracking frames for all (or selected) weeks.

    ``kind``: ``"tracking"`` (BDB<=2025 files), ``"input"`` / ``"output"`` (BDB 2026),
    or ``"auto"`` (tracking if present, else input). Returns a polars DataFrame
    (or LazyFrame if ``lazy``) with normalized columns plus ``week``.
    """
    found = discover_files(raw_dir)
    if kind == "auto":
        kind = "tracking" if found["tracking"] else "input"
    paths = [p for p in found.get(kind, []) if weeks is None or week_from_path(p) in set(weeks)]
    if not paths:
        seen = [str(p.relative_to(raw_dir)) for p in found["all"]][:30]
        raise FileNotFoundError(
            f"No '{kind}' tracking files under {Path(raw_dir).resolve()} (weeks={weeks}). "
            f"Expected tracking_week_*.csv (BDB 2025) or train/input_*_w*.csv (BDB 2026). "
            f"Data files found: {seen or 'none - download the competition data first'}")
    frames = []
    for p in paths:
        lf = cached_frame(p, cache_dir, refresh)
        if "week" not in lf.collect_schema().names() and (wk := week_from_path(p)) is not None:
            lf = lf.with_columns(week=pl.lit(wk, pl.Int8))
        frames.append(lf)
    lf = pl.concat(frames, how="diagonal_relaxed")
    require_columns(lf, ["gameId", "playId", "frameId", "x", "y"], f"{kind} tracking")
    return lf if lazy else lf.collect()


def load_table(name: str, raw_dir: str | Path = "data/raw") -> pl.DataFrame:
    """Load a context table (games, plays, players, player_play, supplementary), normalized."""
    paths = discover_files(raw_dir).get(name, [])
    if not paths:
        raise FileNotFoundError(f"Table '{name}' not found under {raw_dir} (looked for {_TABLE_STEMS[name]})")
    return normalize_columns(_read_any(paths[0]))


def stitch_2026(inp: pl.DataFrame, out: pl.DataFrame) -> pl.DataFrame:
    """Append BDB 2026 output frames (frameId restarts at 1 after the throw) to input frames.

    Output frames are renumbered to continue the play's input frames and tagged
    ``segment='output'``; static per-player columns are carried over from input.
    """
    last = inp.group_by(PLAY_KEY).agg(lastInput=pl.col("frameId").max())
    static = [c for c in inp.columns if c not in ("frameId", "x", "y", "s", "a", "o", "dir", "dis", "week")]
    meta = inp.sort("frameId").group_by(PLAY_KEY + ["nflId"]).agg([pl.col(c).last() for c in static
                                                                   if c not in PLAY_KEY + ["nflId"]])
    out = (out.join(last, on=PLAY_KEY).with_columns(frameId=(pl.col("frameId") + pl.col("lastInput")).cast(pl.Int16))
           .drop("lastInput").join(meta, on=PLAY_KEY + ["nflId"], how="left"))
    return pl.concat([inp.with_columns(segment=pl.lit("input")), out.with_columns(segment=pl.lit("output"))],
                     how="diagonal_relaxed").sort(FRAME_KEY + ["nflId"])


# ---------------------------------------------------------------- standardization
def attach_play_context(df: pl.DataFrame, plays: pl.DataFrame, cols=None) -> pl.DataFrame:
    """Left-join play-level context (possession team, LOS, down, ...) onto tracking rows."""
    plays = normalize_columns(plays)
    cols = cols or [c for c in ("possessionTeam", "defensiveTeam", "absoluteYardlineNumber", "yardlineNumber",
                                "yardlineSide", "yardsToGo", "down", "quarter") if c in plays.columns]
    cols = [c for c in cols if c not in df.columns]
    return df.join(plays.select(PLAY_KEY + cols), on=PLAY_KEY, how="left") if cols else df


def add_side(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``side`` in {offense, defense, ball} from ``playerSide`` or ``club`` vs ``possessionTeam``."""
    cols = df.columns
    if "playerSide" in cols:
        side = pl.col("playerSide").cast(pl.String).str.to_lowercase()
    elif "club" in cols and "possessionTeam" in cols:
        side = pl.when(pl.col("club").cast(pl.String) == pl.col("possessionTeam").cast(pl.String)) \
                 .then(pl.lit("offense")).otherwise(pl.lit("defense"))
    else:
        raise ValueError("add_side needs 'playerSide' or ('club' and 'possessionTeam'; see attach_play_context)")
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

    vx/vy come from ``s`` and ``dir`` (or position differences when absent, e.g.
    the ball). ax/ay are central differences of vx/vy over time (gap-aware).
    aLong is along the direction of travel; aLat > 0 means turning counter-clockwise.
    """
    df = df.sort(FRAME_KEY).with_columns(PID.alias("_pid"))
    grp = ["gameId", "playId", "_pid"]
    d = lambda c, k: (pl.col(c).shift(-k) - pl.col(c).shift(k)).over(grp)  # noqa: E731
    dt = (d("frameId", 1).cast(pl.Float32) * DT)
    dt_f = ((pl.col("frameId").shift(-1) - pl.col("frameId")).over(grp).cast(pl.Float32) * DT)
    dt_b = ((pl.col("frameId") - pl.col("frameId").shift(1)).over(grp).cast(pl.Float32) * DT)

    def deriv(c):  # central difference, one-sided at the ends
        fwd = (pl.col(c).shift(-1) - pl.col(c)).over(grp) / dt_f
        bwd = (pl.col(c) - pl.col(c).shift(1)).over(grp) / dt_b
        return pl.coalesce(d(c, 1) / dt, fwd, bwd)

    if {"s", "dir"} <= set(df.columns):
        rad = pl.col("dir").radians()
        vx, vy = pl.col("s") * rad.sin(), pl.col("s") * rad.cos()
        df = df.with_columns(vx=pl.coalesce(vx, deriv("x")), vy=pl.coalesce(vy, deriv("y")))
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
    cols = df.columns
    is_snap = pl.lit(False)
    if "event" in cols:
        is_snap = is_snap | pl.col("event").cast(pl.String).is_in(SNAP_EVENTS).fill_null(False)
    if "frameType" in cols:
        is_snap = is_snap | (pl.col("frameType").cast(pl.String) == "SNAP").fill_null(False)
    snap = df.filter(is_snap).group_by(PLAY_KEY).agg(snapFrame=pl.col("frameId").min())
    df = df.drop("snapFrame", strict=False).join(snap, on=PLAY_KEY, how="left")
    return df.with_columns(framesFromSnap=(pl.col("frameId").cast(pl.Int32) - pl.col("snapFrame")).cast(pl.Int16))


def frames_from_event(df: pl.DataFrame, event: str, name: str | None = None) -> pl.DataFrame:
    """Add frame index relative to the first frame tagged ``event`` in each play."""
    first = (df.filter(pl.col("event").cast(pl.String) == event)
             .group_by(PLAY_KEY).agg(_ef=pl.col("frameId").min()))
    name = name or f"framesFrom_{event}"
    return (df.join(first, on=PLAY_KEY, how="left")
            .with_columns((pl.col("frameId").cast(pl.Int32) - pl.col("_ef")).cast(pl.Int16).alias(name)).drop("_ef"))


def add_los(df: pl.DataFrame) -> pl.DataFrame:
    """Add line of scrimmage ``losX`` plus ``xRelLos`` (downfield +) and ``yRelCenter``.

    Uses the ball's x at the snap frame when ball rows exist, else the (already
    standardized) ``absoluteYardlineNumber``. ``losSource`` records which. Run
    after :func:`standardize_direction` and :func:`add_snap_frame`.
    """
    los = None
    if "isBall" in df.columns and "snapFrame" in df.columns:
        los = (df.filter(pl.col("isBall") & (pl.col("frameId") == pl.col("snapFrame")))
               .group_by(PLAY_KEY).agg(_ballLos=pl.col("x").first()))
    df = df.join(los, on=PLAY_KEY, how="left") if los is not None else df.with_columns(_ballLos=pl.lit(None, pl.Float32))
    ayn = pl.col("absoluteYardlineNumber") if "absoluteYardlineNumber" in df.columns else pl.lit(None, pl.Float32)
    if "absoluteYardlineNumber" in df.columns and "flipped" not in df.columns:
        raise ValueError("add_los: run standardize_direction first so absoluteYardlineNumber is flipped")
    return df.with_columns(
        losX=pl.coalesce(pl.col("_ballLos"), ayn).cast(pl.Float32),
        losSource=pl.when(pl.col("_ballLos").is_not_null()).then(pl.lit("ball"))
                    .when(ayn.is_not_null()).then(pl.lit("yardline")),
    ).with_columns(
        xRelLos=(pl.col("x") - pl.col("losX")).cast(pl.Float32),
        yRelCenter=(pl.col("y") - FIELD_WIDTH / 2).cast(pl.Float32),
    ).drop("_ballLos")


def prepare_tracking(df: pl.DataFrame, plays: pl.DataFrame | None = None) -> pl.DataFrame:
    """Full pipeline: normalize -> play context -> side -> standardize -> kinematics -> snap -> LOS."""
    df = normalize_columns(df)
    if plays is not None:
        df = attach_play_context(df, plays)
    try:
        df = add_side(df)
    except ValueError:
        pass
    df = standardize_direction(df)
    df = add_kinematics(df)
    df = add_snap_frame(df)
    return add_los(df)
