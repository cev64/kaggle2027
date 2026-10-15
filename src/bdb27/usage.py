"""Capacity usage on game routes: how much of his Combine sprint capacity a receiver uses.

For each regular-season route (WR/TE player-play with ``routeRan``), the *route window*
runs from the snap to the throw (``pass_forward``/``pass_shovel``). Without a throw it
ends at a sack event or ``max_window`` seconds. Per window:

* ``peakSpeed``, ``tPeak``: max reported speed ``s`` and its time after the snap
* ``pathDist``: distance run along the path (10 Hz position steps)
* ``tPath2`` / ``tPath10`` and ``burst``: mean path speed between 2 and 10 yd of path,
  the in-game analogue of the Combine ``burst2_10`` (NaN if 10 yd are not reached)
* ``sAtSnap``: speed at the snap (motion), ``windowEnd``: pass / sack / cap

:func:`attach_capacity` adds ``speedFrac = peakSpeed / vmax`` and
``burstFrac = burst / burst2_10``: the share of tested capacity used on that rep.
"""
from __future__ import annotations

import polars as pl

from .tracking import DT

PASS_EVENTS = ("pass_forward", "pass_shovel")
SACK_EVENTS = ("qb_sack", "qb_strip_sack")
KEY = ["gameId", "playId", "nflId"]


def route_windows(frames: pl.LazyFrame | pl.DataFrame, max_window: float = 6.0) -> pl.DataFrame:
    """Per player-play route-window summary from standardized game tracking.

    ``frames`` needs gameId, playId, nflId, frameId, framesFromSnap, x, y, s, event.
    Plays without a tagged snap are dropped.
    """
    lf = frames.lazy() if isinstance(frames, pl.DataFrame) else frames
    lf = lf.filter(pl.col("framesFromSnap").is_not_null())
    ev = pl.col("event").cast(pl.String)
    ends = (lf.filter(ev.is_in(PASS_EVENTS + SACK_EVENTS) & (pl.col("framesFromSnap") > 0))
            .group_by(KEY).agg(
                passF=pl.col("framesFromSnap").filter(ev.is_in(PASS_EVENTS)).min(),
                sackF=pl.col("framesFromSnap").filter(ev.is_in(SACK_EVENTS)).min()))
    cap = int(round(max_window / DT))
    w = (lf.join(ends, on=KEY, how="left")
         .with_columns(endF=pl.coalesce("passF", "sackF", pl.lit(cap)).clip(upper_bound=cap),
                       windowEnd=pl.when(pl.col("passF").is_not_null()).then(pl.lit("pass"))
                       .when(pl.col("sackF").is_not_null()).then(pl.lit("sack")).otherwise(pl.lit("cap")))
         .filter((pl.col("framesFromSnap") >= 0) & (pl.col("framesFromSnap") <= pl.col("endF")))
         .sort(KEY + ["frameId"]))
    step = ((pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2).sqrt()).over(KEY).fill_null(0)
    w = w.with_columns(cum=step.cum_sum().over(KEY), t=pl.col("framesFromSnap") * DT)

    def t_at(dist):  # first time the path reaches ``dist`` yd (frame resolution, linearly interpolated)
        return pl.col("t").filter(pl.col("cum") >= dist).first() - DT * (
            (pl.col("cum").filter(pl.col("cum") >= dist).first() - dist)
            / (pl.col("cum").filter(pl.col("cum") >= dist).first()
               - pl.col("cum").filter(pl.col("cum") < dist).last()).clip(lower_bound=1e-6)).clip(0, 1)

    extra = [pl.col("losX").first()] if "losX" in w.collect_schema().names() else []
    out = w.group_by(KEY).agg(
        *extra,
        windowEnd=pl.col("windowEnd").first(),
        tWindow=pl.col("t").max(),
        nFrames=pl.len(),
        peakSpeed=pl.col("s").max(),
        tPeak=pl.col("t").sort_by("s").last(),
        pathDist=pl.col("cum").max(),
        sAtSnap=pl.col("s").sort_by("t").first(),
        tPath2=t_at(2.0),
        tPath10=t_at(10.0),
    ).with_columns(burst=(8.0 / (pl.col("tPath10") - pl.col("tPath2"))))
    return out.collect()


def attach_capacity(routes: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Join player capacity (``vmax``, ``burst2_10``) and add the usage fractions."""
    cap = capacity.select("nflId", "vmax", "burst2_10", *[c for c in ("vmaxSource",) if c in capacity.columns])
    return routes.join(cap, on="nflId", how="left").with_columns(
        speedFrac=pl.col("peakSpeed") / pl.col("vmax"), burstFrac=pl.col("burst") / pl.col("burst2_10"))
