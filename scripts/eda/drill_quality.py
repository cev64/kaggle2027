"""Drill coverage and kinematic-quality EDA for the Combine tracking data.

Run everything (about 2 min, peak RAM < 1 GB):

    PYTHONPATH=src python scripts/eda/drill_quality.py

Outputs
* ``data/processed/eda/attempt_features.parquet``: one row per player attempt with speed,
  main-cut, forty/agility timing and lateral-reversal features (reusable downstream).
* ``data/processed/eda/*.csv``: the tables quoted in ``docs/eda/drill_quality.md``.
* ``docs/eda/fig/dq_*.png``: diagnostic figures.

The functions are importable (``sys.path`` + ``import drill_quality``) for later feature work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdb27.drills import detect_cuts, kinematics  # noqa: E402
from bdb27.tracking import DT  # noqa: E402

TRACK = ROOT / "data/processed/combine_tracking_clean.parquet"
RESULTS = ROOT / "data/raw/combine_results.csv"
OUT = ROOT / "data/processed/eda"
FIG = ROOT / "docs/eda/fig"
GROUPS = ["WR", "TE", "DB", "OL", "DL"]
ONSET = 0.5  # yd/s

# heading-based break (target turn in deg, reference-heading window after onset in s): routes and
# DB breaks where the speed loss is too small for a speed-minimum detector
HEADING_BREAKS = {
    "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION": (40.0, (0.5, 1.5)),
    "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL": (80.0, (0.5, 1.5)),
    "SLANT_ROUTE_LEFT": (40.0, (0.2, 0.6)), "SPEED_OUT_ROUTE_LEFT": (80.0, (0.3, 0.8)),
    "DAGGER_ROUTE_LEFT": (80.0, (0.3, 0.8)), "POST_CORNER_ROUTE_RIGHT": (30.0, (0.3, 0.8)),
    "POST_CORNER_ROUTE_LEFT": (30.0, (0.3, 0.8)), "CURL_ROUTE_RIGHT": (120.0, (0.3, 0.8)),
    "COMEBACK_ROUTE_RIGHT": (120.0, (0.3, 0.8)), "IN_ROUTE_LEFT": (80.0, (0.3, 0.8)),
}
# drills that matter for the candidate ideas
LATERAL_DRILLS = ["PASS_PRO_MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
                  "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION"]
IDEA_DRILLS = {
    "Brake Check (WR)": ["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT",
                         "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT"],
    "Brake Check (DB)": ["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "W_DRILL"],
    "Mirror Image (OL)": ["PASS_PRO_MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"],
    "Trigger Time (DB)": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"],
    "DL burst": ["PASS_RUSH_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "FORTY_YARD_DASH"],
    "Speed Tax": ["FORTY_YARD_DASH"],
}


# ----------------------------------------------------------------------------- loading
def load_frames(drills=None, cols=("draftYear", "eventId", "nflId", "drillCanon", "attempt", "frameId",
                                   "x", "y", "xd", "yd", "s", "a", "dir")) -> pl.DataFrame:
    """Non-ball Combine frames, column-pruned, sorted by attempt and frame."""
    lf = pl.scan_parquet(TRACK).filter(~pl.col("isBall"))
    if drills is not None:
        lf = lf.filter(pl.col("drillCanon").cast(pl.String).is_in(list(drills)))
    return (lf.select(list(cols)).with_columns(pl.col("drillCanon").cast(pl.String))
            .sort("eventId", "frameId").collect())


def load_results() -> pl.DataFrame:
    r = pl.read_csv(RESULTS, null_values="NA")
    return r.rename({"nfl_id": "nflId", "draft_year": "draftYear", "combine_position": "group"})


# ----------------------------------------------------------------------------- 1. coverage
def coverage_table(df: pl.DataFrame, res: pl.DataFrame) -> pd.DataFrame:
    """Per group x drill: players, attempts, players with 2+ reps, years present, usable flag."""
    grp = res.select("nflId", "group")
    n_group = grp.group_by("group").len().rename({"len": "groupN"})
    cell = (df.group_by("eventId", "nflId", "drillCanon", "draftYear").len().join(grp, on="nflId")
            .group_by("group", "drillCanon", "nflId").agg(reps=pl.len(), year=pl.col("draftYear").first()))
    cov = (cell.group_by("group", "drillCanon").agg(
        players=pl.len(), attempts=pl.col("reps").sum(), players2plus=(pl.col("reps") >= 2).sum(),
        years=pl.col("year").unique().sort().cast(pl.String).str.slice(2).str.join("/"),
        nYears=pl.col("year").n_unique())
        .join(n_group, on="group").with_columns(share=(pl.col("players") / pl.col("groupN")).round(2)))
    cov = cov.with_columns(usable=((pl.col("players") >= 80) | (pl.col("share") >= 0.85)) & (pl.col("nYears") == 3))
    out = cov.to_pandas()
    out["group"] = pd.Categorical(out["group"], GROUPS)
    return out.sort_values(["group", "players"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- 2. kinematic QA
def _per_attempt_diffs(df: pl.DataFrame) -> pl.DataFrame:
    k = "eventId"
    return df.with_columns(
        vxc=((pl.col("x").shift(-1) - pl.col("x").shift(1)) / (2 * DT)).over(k),
        vyc=((pl.col("y").shift(-1) - pl.col("y").shift(1)) / (2 * DT)).over(k),
        ddir=((pl.col("dir") - pl.col("dir").shift(1) + 180) % 360 - 180).abs().over(k),
    ).with_columns(sPos=(pl.col("vxc") ** 2 + pl.col("vyc") ** 2).sqrt(),
                   hdg=pl.arctan2(pl.col("vxc"), pl.col("vyc")).degrees() % 360)


def speed_check(df: pl.DataFrame) -> dict:
    """Reported s vs speed from central differences of x, y (lags -2..2 frames)."""
    d = _per_attempt_diffs(df)
    out = {}
    for lag in (-2, -1, 0, 1, 2):
        dd = d.with_columns(sl=pl.col("s").shift(lag).over("eventId")).drop_nulls(["sl", "sPos"])
        out[f"r_lag{lag}"] = float(np.corrcoef(dd["sl"], dd["sPos"])[0, 1])
    dd = d.drop_nulls("sPos")
    diff = (dd["s"] - dd["sPos"]).to_numpy()
    out.update(n=len(dd), bias=float(np.median(diff)), mae=float(np.median(np.abs(diff))),
               p95=float(np.quantile(np.abs(diff), .95)), p999=float(np.quantile(np.abs(diff), .999)))
    by_drill = (dd.group_by("drillCanon").agg(r=pl.corr("s", "sPos"), mae=(pl.col("s") - pl.col("sPos")).abs().median(),
                                               p99=(pl.col("s") - pl.col("sPos")).abs().quantile(.99)).sort("r"))
    return out, by_drill.to_pandas()


def dir_noise(df: pl.DataFrame) -> pd.DataFrame:
    """|dir - heading from positions| and frame-to-frame |d dir| by speed bin."""
    d = _per_attempt_diffs(df).drop_nulls(["hdg", "ddir"])
    d = d.with_columns(err=((pl.col("dir") - pl.col("hdg") + 180) % 360 - 180).abs())
    edges = [0, .25, .5, 1, 2, 4, 6, 99]
    d = d.with_columns(bin=pl.col("s").cut(edges[1:-1], labels=[f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])]))
    return (d.group_by("bin").agg(frames=pl.len(), errMedian=pl.col("err").median(), errP90=pl.col("err").quantile(.9),
                                  jitterMedian=pl.col("ddir").median(), jitterP90=pl.col("ddir").quantile(.9))
            .sort("bin").to_pandas())


def trim_stats(df: pl.DataFrame) -> pd.DataFrame:
    """Start/end trimming: standing frames before onset/after stop, start/end speed, end at speed."""
    k = "eventId"
    d = df.with_columns(dx=(pl.col("x") - pl.col("x").shift(1)).over(k), dy=(pl.col("y") - pl.col("y").shift(1)).over(k))
    d = d.with_columns(jerkPos=((pl.col("dx") - pl.col("dx").shift(1)).abs() + (pl.col("dy") - pl.col("dy").shift(1)).abs()).over(k))
    g = d.group_by(k).agg(
        drillCanon=pl.col("drillCanon").first(), draftYear=pl.col("draftYear").first(), n=pl.len(),
        s0=pl.col("s").first(), s1=pl.col("s").last(), smax=pl.col("s").max(),
        preFrames=(pl.col("s") > ONSET).arg_max(),
        postFrames=pl.len() - 1 - (pl.col("s") > ONSET).cum_sum().arg_max(),
        tailConst=pl.col("jerkPos").reverse().lt(2e-3).cast(pl.Int32).cum_prod().sum(),
        peakAtEnd=pl.col("s").arg_max() >= pl.len() - 3,
    )
    return g.with_columns(startMoving=pl.col("s0") > ONSET, endFast=pl.col("s1") > 0.5 * pl.col("smax"),
                          s1rel=pl.col("s1") / pl.col("smax"), padded=pl.col("tailConst") >= 2).to_pandas()


def trim_summary(tr: pd.DataFrame) -> pd.DataFrame:
    return tr.groupby("drillCanon").agg(attempts=("n", "size"), framesMed=("n", "median"),
                                        startMoving=("startMoving", "mean"), s0Med=("s0", "median"),
                                        preFramesMean=("preFrames", "mean"), endFast=("endFast", "mean"),
                                        s1relMed=("s1rel", "median"), peakInLast3=("peakAtEnd", "mean"),
                                        paddedTail=("padded", "mean")).round(2)


# ----------------------------------------------------------------------------- per-attempt features
def _onset(t, s):
    """Time when speed first reaches ONSET; back-extrapolated (linear) when the clip starts already moving."""
    if s[0] < ONSET:
        i = int(np.argmax(s >= ONSET))
        return float(np.interp(ONSET, s[i - 1:i + 1], t[i - 1:i + 1])) if i > 0 else float(t[0])
    a0 = max((s[min(2, len(s) - 1)] - s[0]) / (t[min(2, len(s) - 1)] - t[0] + 1e-9), 1.0)
    return float(t[0] - (s[0] - ONSET) / a0)


def _onset_zero(t, s):
    """Back-extrapolated time of zero speed (first movement)."""
    a0 = max((s[min(2, len(s) - 1)] - s[0]) / (t[min(2, len(s) - 1)] - t[0] + 1e-9), 1.0)
    return float(t[0] - s[0] / a0) if s[0] > 0.2 else float(t[int(np.argmax(s > 0.2))] - 0.2 / a0)


def _cross_time(t, d, target):
    hit = np.flatnonzero(d >= target)
    if not len(hit) or hit[0] == 0:
        return np.nan
    i = hit[0]
    return float(np.interp(target, d[i - 1:i + 1], t[i - 1:i + 1]))


def forty_features(t, pos, s) -> dict:
    """40-yd dash: split times along the run (projection on the start->end chord) and the shape
    of the acceleration curve. The clip ends at the finish (median run length 40.2 yd), so the clip
    duration is also a 40 time. Distance-anchored splits (from 2 yd) avoid the onset problem."""
    chord = pos[-1] - pos[0]
    u = chord / (np.linalg.norm(chord) + 1e-9)
    d = (pos - pos[0]) @ u
    t0, tz = _onset(t, s), _onset_zero(t, s)
    out = {"runLength": float(d.max()), "tClip": float(t[-1] - t[0])}
    # extrapolate the last few frames at constant speed when the clip stops just short of 40 yd
    if 37.5 <= d[-1] < 40:
        t = np.r_[t, t[-1] + (40.5 - d[-1]) / max(s[-1], 1.0)]
        d = np.r_[d, 40.5]
    for yd in (2, 5, 10, 20, 40):
        tc = _cross_time(t, d, yd)
        out[f"t{yd}"] = tc - t[0]          # from first tracked frame
        out[f"t{yd}_onset"] = tc - t0      # from s = 0.5 (extrapolated)
        out[f"t{yd}_zero"] = tc - tz       # from s = 0 (extrapolated first movement)
    out["split2_10"] = out["t10"] - out["t2"]
    out["split10_20"] = out["t20"] - out["t10"]
    out["split20_40"] = out["t40"] - out["t20"]
    out["fly10_40"] = out["t40"] - out["t10"]
    sp = s[: len(pos)]
    for yd in (5, 10, 20):
        i = np.flatnonzero(d[: len(sp)] >= yd)
        out[f"v{yd}"] = float(sp[i[0]]) if len(i) else np.nan
    out["v10rel"] = out["v10"] / sp.max()
    i90 = np.flatnonzero(sp >= 0.9 * sp.max())
    out["d90"] = float(d[i90[0]]) if len(i90) else np.nan  # distance at which 90% of top speed is reached
    return out


def agility_features(t, pos, s) -> dict:
    """3-cone / shuttle total time. The start is anchored on the last standstill (s < 0.5) before the
    first turn, which removes pre-drill shuffling; clips with fewer turns than the drill needs are
    marked incomplete."""
    t0, tz = _onset(t, s), _onset_zero(t, s)
    mins, _ = find_peaks(-s, prominence=1.0)
    first = mins[0] if len(mins) else len(s) - 1
    still = np.flatnonzero(s[:first] < ONSET)
    ta = t[still[-1]] if len(still) else t0
    return {"agTime_first": float(t[-1] - t[0]), "agTime_onset": float(t[-1] - t0), "agTime_zero": float(t[-1] - tz),
            "agTime_anchor": float(t[-1] - ta), "agTurns": len(mins), "agFirstTurn": float(t[first] - ta),
            "agAfterFirstTurn": float(t[-1] - t[first]),
            "endDistFromStart": float(np.linalg.norm(pos[-1] - pos[0])),
            "maxDistFromStart": float(np.linalg.norm(pos - pos[0], axis=1).max())}


def detect_breaks(t, s, dsdt, vel, pos, min_drop: float = 0.75, min_turn: float = 30.0, look: int = 5) -> pd.DataFrame:
    """Change-of-direction events (variant of ``bdb27.drills.detect_cuts``).

    ``detect_cuts`` measures the turn between the entry and the *exit speed peak*; on routes the
    exit peak is often after the receiver has turned upfield again, so most curls/digs/outs are
    rejected (26% of curls detected). Here the exit heading is taken ``look`` frames (0.5 s) after
    the speed minimum, and re-acceleration is measured over 1 s after it (routes end soon after
    the break, so 90%-of-entry regain is often never reached). ``tGain2`` = time from the minimum to
    +2 yd/s; ``codTime`` = decelTime + tGain2.
    """
    n = len(s)
    mins, _ = find_peaks(-s, prominence=min_drop)
    bounds = np.r_[0, mins, n - 1]
    rows = []
    for j, m in enumerate(mins):
        seg = s[bounds[j]:m + 1]
        i0 = bounds[j] + int(np.flatnonzero(seg >= 0.98 * seg.max())[-1])
        ie = min(m + look, n - 1)
        if i0 == m or ie == m:
            continue
        a, b = vel[i0], vel[ie]
        turn = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))
        if abs(turn) < min_turn:
            continue
        w1 = min(m + 10, n - 1)
        gain = np.flatnonzero(s[m:] >= s[m] + 2.0)
        tg = t[m + gain[0]] - t[m] if len(gain) else np.nan
        dec_t = t[m] - t[i0]
        rows.append({"tCut": t[m], "iCut": m, "turnAngle": turn, "entrySpeed": s[i0], "minSpeed": s[m],
                     "exit05Speed": s[ie], "speedRetained": s[m] / s[i0], "decelTime": dec_t,
                     "decelDist": float(np.linalg.norm(np.diff(pos[i0:m + 1], axis=0), axis=1).sum()),
                     "meanDecel": (s[i0] - s[m]) / dec_t, "peakDecel": -np.min(dsdt[i0:m + 1]),
                     "peakReaccel": np.max(dsdt[m:w1 + 1]), "tGain2": tg, "codTime": dec_t + tg})
    return pd.DataFrame(rows)


def heading_break(t, s, dsdt, vel, target: float, ref=(0.5, 1.5)) -> dict:
    """First heading change of about ``target`` deg away from the early-pedal heading (DB 45/90
    drills, which show little speed loss so speed-minimum cuts miss them). Transition = the heading
    moving from 20% to 80% of ``target``."""
    t0 = _onset(t, s)
    h = np.degrees(np.unwrap(np.arctan2(vel[:, 1], vel[:, 0])))
    ref_i = np.flatnonzero((t >= t0 + ref[0]) & (t <= t0 + ref[1]) & (s > 1.0))
    if len(ref_i) < 3:
        return {}
    dh = np.abs(h - np.median(h[ref_i]))
    after = ref_i[-1]
    half = after + np.flatnonzero(dh[after:] >= 0.5 * target)
    if not len(half):
        return {}
    ih = half[0]
    ia = ih - int(np.argmax(dh[:ih + 1][::-1] < 0.2 * target))
    ib = ih + np.flatnonzero(dh[ih:] >= 0.8 * target)
    if not len(ib):
        return {}
    ib = ib[0]
    yaw = np.abs(savgol_filter(h, 5, 2, deriv=1, delta=DT)) if len(h) >= 5 else np.abs(np.gradient(h, DT))
    lo, hi = max(ia - 10, 0), min(ib + 10, len(s) - 1)
    entry = s[lo:ia + 1].max()
    mn = s[ia:min(ib + 5, len(s) - 1) + 1].min()
    return {"hbStart": float(t[ia] - t0), "hbTurnTime": float(t[ib] - t[ia]),
            "hbPeakYaw": float(yaw[max(ia - 3, 0):ib + 4].max()), "hbEntrySpeed": float(entry),
            "hbMinSpeed": float(mn), "hbSpeedRetained": float(mn / max(entry, 1e-6)),
            "hbPeakReaccel": float(np.max(dsdt[ib:hi + 1])), "hbSpeed1s": float(s[hi])}


def mirror_phase_end(s: np.ndarray) -> int:
    """Index where the final sprint-out of a lateral drill begins: the last clear speed minimum,
    provided the clip ends faster than anything in the shuffle phase; else the whole clip."""
    mins, _ = find_peaks(-s, prominence=0.5)
    if not len(mins):
        return len(s)
    m = int(mins[-1])
    return m + 1 if s[-1] > 1.1 * s[:m + 1].max() or s[m:].max() > 4.5 else len(s)


def reversals(t, pos, s, min_v: float = 0.5) -> tuple[pd.DataFrame, float]:
    """Per-reversal lateral mechanics from the player's own velocity (no partner is tracked).

    The shuffle phase is the clip before the final sprint-out; its principal axis is the lateral
    axis. For each reversal of lateral velocity (|v| > ``min_v`` on both sides): ``latPeak`` (prior
    peak |v|), ``brakeTime`` (|v| falls from 90% of that peak to the zero crossing), ``overshoot``
    (lateral distance from the prior peak to the crossing), ``deadTime`` (contiguous |v| < 1 yd/s
    around the crossing), ``reaccelTime`` (crossing -> 2 yd/s the other way), ``peakLatDecel``.
    """
    e = mirror_phase_end(s)
    if e < 15:
        return pd.DataFrame(), e * DT
    P = pos[:e] - pos[:e].mean(0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    lat = P @ vt[0]
    v = savgol_filter(lat, 7, 2, deriv=1, delta=DT)
    acc = savgol_filter(lat, 7, 2, deriv=2, delta=DT)
    sgn = np.where(v > min_v, 1, np.where(v < -min_v, -1, 0))
    idx = np.flatnonzero(sgn)
    slow = np.abs(v) < 1.0
    rows, last_z = [], 0
    for a, b in zip(idx[:-1], idx[1:]):
        if sgn[a] == sgn[b]:
            continue
        ipk = last_z + int(np.argmax(np.abs(v[last_z:a + 1])))
        vpk = abs(v[ipk])
        z = a + int(np.argmin(np.abs(v[a:b + 1])))
        i90 = ipk + int(np.argmax(np.abs(v[ipk:z + 1]) < 0.9 * vpk))
        after = np.flatnonzero(-sgn[a] * v[z:] >= 2.0)
        dead = np.nan
        if slow[z]:
            k0, k1 = z, z
            while k0 > 0 and slow[k0 - 1]:
                k0 -= 1
            while k1 < len(v) - 1 and slow[k1 + 1]:
                k1 += 1
            dead = (k1 - k0 + 1) * DT
        rows.append({"latPeak": vpk, "brakeTime": (z - i90) * DT, "overshoot": abs(lat[z] - lat[ipk]),
                     "deadTime": dead, "reaccelTime": after[0] * DT if len(after) else np.nan,
                     "peakLatDecel": float(np.max(-np.sign(v[ipk]) * acc[ipk:z + 1])) if z > ipk else np.nan})
        last_z = z
    return pd.DataFrame(rows), e * DT


REV_COLS = ["latPeak", "brakeTime", "overshoot", "deadTime", "reaccelTime", "peakLatDecel"]


def reversal_features(t, pos, s) -> dict:
    r, dur = reversals(t, pos, s)
    out = {"nRev": len(r), "shuffleTime": dur, "revRate": len(r) / dur if dur > 0 else np.nan}
    if len(r):
        for c in REV_COLS:
            out[f"rev_{c}"] = float(r[c].median())
            out[f"rev_{c}_odd"] = float(r[c].iloc[0::2].median())
            out[f"rev_{c}_even"] = float(r[c].iloc[1::2].median()) if len(r) > 1 else np.nan
    return out


def attempt_features(df: pl.DataFrame) -> pd.DataFrame:
    """One row per attempt: speed profile, main cut (largest speed drop), drill-specific timing."""
    pdf = df.select("draftYear", "eventId", "nflId", "drillCanon", "attempt", "frameId", "xd", "yd", "s").to_pandas()
    rows = []
    for (ev, nfl, drill), g in pdf.groupby(["eventId", "nflId", "drillCanon"], sort=False):
        if len(g) < 10:
            continue
        k = kinematics(g, ("xd", "yd"))
        t, s, dsdt, pos, vel = k["t"], k["speed"], k["dsdt"], k["pos"], k["vel"]
        t0 = _onset(t, s)
        row = {"eventId": ev, "nflId": nfl, "drillCanon": drill, "draftYear": g["draftYear"].iat[0],
               "attempt": g["attempt"].iat[0], "frames": len(g), "duration": t[-1] - t0,
               "peakSpeed": s.max(), "peakAccel": np.nanmax(dsdt), "peakDecel": max(0.0, -np.nanmin(dsdt)),
               "tTo5": _cross_time(t, s, 5.0) - t0, "tTo3": _cross_time(t, s, 3.0) - t0,
               "pathLen": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
               "endX": pos[-1, 0], "endY": pos[-1, 1], "startSpeed": s[0], "endSpeedRel": s[-1] / s.max()}
        row["nCutsLib"] = len(detect_cuts(g, ("xd", "yd")))  # bdb27.drills detector, for comparison
        cuts = detect_breaks(t, s, dsdt, vel, pos)
        row["nCuts"] = len(cuts)
        if len(cuts):
            c = cuts.loc[(cuts["entrySpeed"] - cuts["minSpeed"]).idxmax()]
            i = int(c["iCut"])
            row.update({f"cut_{n}": c[n] for n in cuts.columns if n != "iCut"})
            row.update(cut_x=pos[i, 0], cut_y=pos[i, 1], cut_dist=float(np.linalg.norm(pos[i] - pos[0])),
                       cut_turnAbs=abs(c["turnAngle"]))
        if drill == "FORTY_YARD_DASH":
            row.update(forty_features(t, pos, s))
        elif drill in ("THREE_CONE_DRILL", "SHORT_SHUTTLE"):
            row.update(agility_features(t, pos, s))
        if drill in LATERAL_DRILLS:
            row.update(reversal_features(t, pos, s))
        if drill in HEADING_BREAKS:
            target, ref = HEADING_BREAKS[drill]
            row.update(heading_break(t, s, dsdt, vel, target, ref))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- geometry & outliers
def geometry(feat: pd.DataFrame) -> pd.DataFrame:
    """Spread of endpoints and of main-cut location within a drill (drill frame from normalize_drill,
    lateral sign mirrored so the endpoint is +y). Robust spread = median distance to the drill median."""
    out = []
    for d, g in feat.groupby("drillCanon"):
        if len(g) < 30:
            continue
        ey = g["endY"].abs()
        end = np.c_[g["endX"], ey]
        endspread = np.median(np.linalg.norm(end - np.median(end, 0), axis=1))
        sign = np.sign(g["endY"]).replace(0, 1)
        cut = np.c_[g["cut_x"], g["cut_y"] * sign]
        ok = np.isfinite(cut).all(1)
        cutspread = np.median(np.linalg.norm(cut[ok] - np.median(cut[ok], 0), axis=1)) if ok.sum() > 10 else np.nan
        cd = g["cut_dist"].dropna()
        ang = np.degrees(np.arctan2(ey, g["endX"]))
        out.append({"drillCanon": d, "attempts": len(g), "endDistMed": float(np.median(np.linalg.norm(end, axis=1))),
                    "endSpread": endspread, "endAngleIQR": float(np.subtract(*np.percentile(ang, [75, 25]))),
                    "cutDistMed": cd.median(), "cutDistIQR": float(np.subtract(*np.percentile(cd, [75, 25]))) if len(cd) else np.nan,
                    "cutSpread": cutspread, "turnAbsMed": g["cut_turnAbs"].median(),
                    "turnAbsIQR": float(np.subtract(*np.nanpercentile(g["cut_turnAbs"], [75, 25]))),
                    "pctLeftTurn": float((g["cut_turnAngle"] > 0).mean())})
    return pd.DataFrame(out).round(2)


def flag_outliers(feat: pd.DataFrame, z: float = 4.0) -> pd.DataFrame:
    """Robust-z flags within drill: short/long duration, short path (aborted/truncated), low peak speed."""
    f = feat.copy()
    for c in ("duration", "pathLen", "peakSpeed"):
        med = f.groupby("drillCanon")[c].transform("median")
        mad = f.groupby("drillCanon")[c].transform(lambda v: (v - v.median()).abs().median()) * 1.4826
        f[f"z_{c}"] = (f[c] - med) / mad.replace(0, np.nan)
    f["truncated"] = (f["z_pathLen"] < -z) | (f["z_duration"] < -z)
    f["slowRep"] = f["z_peakSpeed"] < -z
    f["longRep"] = f["z_duration"] > z
    f["anyFlag"] = f["truncated"] | f["slowRep"] | f["longRep"]
    return f


# ----------------------------------------------------------------------------- reliability
def icc(x1, x2) -> tuple[float, float]:
    """ICC(3,1) consistency and ICC(2,1) absolute agreement for k = 2 reps."""
    X = np.c_[x1, x2]
    X = X[np.isfinite(X).all(1)]
    n, k = X.shape
    if n < 10:
        return np.nan, np.nan
    gm = X.mean()
    ssr = k * ((X.mean(1) - gm) ** 2).sum()
    ssc = n * ((X.mean(0) - gm) ** 2).sum()
    sse = ((X - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse), (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def year_center(feat: pd.DataFrame, cols) -> pd.DataFrame:
    """Subtract the drill x draft-year mean: clip trimming differs by year (e.g. the 2024 DB 45-degree
    clips are half as long), which otherwise inflates between-player variance and ICCs."""
    f = feat.copy()
    cols = [c for c in cols if c in f]
    f[cols] = f[cols] - f.groupby(["drillCanon", "draftYear"])[cols].transform("mean")
    return f


def rep_pairs(feat: pd.DataFrame, drill: str, flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """First two reps (by attempt number) of a drill per player, wide (suffix _1/_2)."""
    g = feat[feat["drillCanon"] == drill]
    if flags is not None:
        g = g[~g["eventId"].isin(flags.loc[flags["anyFlag"], "eventId"])]
    g = g.sort_values(["nflId", "attempt"])
    g = g.assign(rep=g.groupby("nflId").cumcount() + 1)
    g = g[g["rep"] <= 2]
    w = g.pivot(index="nflId", columns="rep")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.dropna(subset=["eventId_2"]) if "eventId_2" in w else w.iloc[:0]


def reliability(feat: pd.DataFrame, drill: str, features, flags=None, center: bool = False) -> pd.DataFrame:
    if center:
        feat = year_center(feat, features)
    w = rep_pairs(feat, drill, flags)
    rows = []
    for f in features:
        if f"{f}_1" not in w or f"{f}_2" not in w:
            continue
        a, b = w[f"{f}_1"].astype(float), w[f"{f}_2"].astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        i3, i2 = icc(a[ok], b[ok])
        sem = np.sqrt(np.var(np.r_[a[ok], b[ok]]) * (1 - i3)) if np.isfinite(i3) else np.nan
        rows.append({"drillCanon": drill, "feature": f, "nPairs": int(ok.sum()), "r": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                     "ICC31": i3, "ICC21": i2, "meanDiff": float((b[ok] - a[ok]).mean()),
                     "mean": float(np.r_[a[ok], b[ok]].mean()), "SEM": sem})
    return pd.DataFrame(rows)


SPEED_FEATS = ["peakSpeed", "tTo3", "tTo5", "peakAccel", "peakDecel", "duration"]
CUT_FEATS = ["cut_entrySpeed", "cut_minSpeed", "cut_speedRetained", "cut_decelTime", "cut_decelDist",
             "cut_peakDecel", "cut_meanDecel", "cut_peakReaccel", "cut_tGain2", "cut_codTime", "cut_exit05Speed"]
REV_FEATS = ["nRev", "revRate", "rev_latPeak", "rev_brakeTime", "rev_overshoot", "rev_deadTime", "rev_reaccelTime",
             "rev_peakLatDecel"]
FORTY_FEATS = ["tClip", "t10", "t10_zero", "t40", "split2_10", "split10_20", "split20_40", "fly10_40",
               "v5", "v10", "v10rel", "d90"]
HB_FEATS = ["hbStart", "hbTurnTime", "hbPeakYaw", "hbEntrySpeed", "hbMinSpeed", "hbSpeedRetained", "hbPeakReaccel",
            "hbSpeed1s"]


def split_half(feat: pd.DataFrame, drill: str, cols=REV_COLS, flags=None) -> pd.DataFrame:
    """Within-attempt reliability for single-rep lateral drills: median over odd vs even reversals,
    correlated across players, Spearman-Brown corrected to the full attempt. ``*_tempo`` partials
    out the attempt's reversal rate (the leader's tempo, shared by both halves) and draft year.
    ``sessionAutocorr`` = correlation between consecutive players in event order (shared leader /
    session conditions)."""
    g = feat[feat["drillCanon"] == drill]
    if flags is not None:
        g = g[~g["eventId"].isin(flags.loc[flags["anyFlag"], "eventId"])]
    g = g.sort_values(["draftYear", "eventId"]).groupby("nflId", sort=False).first()
    rows = []
    for c in cols:
        o, e = g.get(f"rev_{c}_odd"), g.get(f"rev_{c}_even")
        if o is None:
            continue
        ok = o.notna() & e.notna() & g["revRate"].notna()
        if ok.sum() < 15:
            continue
        r = np.corrcoef(o[ok], e[ok])[0, 1]
        X = np.c_[np.ones(ok.sum()), g.loc[ok, "revRate"], g.loc[ok, "draftYear"] == 2024, g.loc[ok, "draftYear"] == 2025]
        res = [v[ok] - X @ np.linalg.lstsq(X, v[ok], rcond=None)[0] for v in (o, e)]
        rt = np.corrcoef(*res)[0, 1]
        full = g[f"rev_{c}"].dropna().values
        rows.append({"drillCanon": drill, "feature": f"rev_{c}", "n": int(ok.sum()), "rHalf": r, "spearmanBrown": 2 * r / (1 + r),
                     "rHalf_tempo": rt, "spearmanBrown_tempo": 2 * rt / (1 + rt),
                     "rWithTempo": np.corrcoef(g.loc[ok, f"rev_{c}"], g.loc[ok, "revRate"])[0, 1],
                     "sessionAutocorr": np.corrcoef(full[:-1], full[1:])[0, 1]})
    return pd.DataFrame(rows)


def cross_drill(feat: pd.DataFrame, drills, features, flags=None, min_n: int = 25) -> pd.DataFrame:
    """Parallel-forms reliability for single-rep drills: player-level correlation of the same
    feature across different drills (e.g. the main break of curl vs comeback vs out). Reports the
    median pairwise r and the standardized alpha of the k-drill composite."""
    g = feat[feat["drillCanon"].isin(drills)]
    if flags is not None:
        g = g[~g["eventId"].isin(flags.loc[flags["anyFlag"], "eventId"])]
    rows = []
    for f in features:
        w = g.pivot_table(index="nflId", columns="drillCanon", values=f, aggfunc="mean")
        w = w.loc[:, w.notna().sum() >= min_n]
        if w.shape[1] < 2:
            continue
        cm = w.corr(min_periods=min_n).values
        iu = np.triu_indices_from(cm, 1)
        rr = cm[iu]
        rr = rr[np.isfinite(rr)]
        if not len(rr):
            continue
        k, rbar = w.shape[1], float(np.mean(rr))
        rows.append({"feature": f, "kDrills": k, "nPlayersMed": int(np.median(w.notna().sum())), "rMedian": float(np.median(rr)),
                     "rMin": float(rr.min()), "rMax": float(rr.max()), "alphaStd": k * rbar / (1 + (k - 1) * rbar)})
    return pd.DataFrame(rows)


def year_consistency(feat: pd.DataFrame) -> pd.DataFrame:
    """Per drill, median frames / path length / end distance by year; flag a year that differs >20%."""
    f = feat.assign(endDist=np.hypot(feat["endX"], feat["endY"]))
    t = f.groupby(["drillCanon", "draftYear"])[["frames", "pathLen", "endDist", "peakSpeed"]].median().unstack()
    out = []
    for d, row in t.iterrows():
        for c in ("frames", "pathLen", "peakSpeed"):
            v = row[c].dropna()
            if len(v) >= 2 and (v.max() / v.min() > 1.2):
                out.append({"drillCanon": d, "metric": c, **{str(y): round(x, 1) for y, x in v.items()}})
    return pd.DataFrame(out)


def consistency_check(feat: pd.DataFrame, drills, feature="duration") -> dict:
    """Is rep-to-rep variability a player trait? Correlate a player's |rep1-rep2| (as % of drill
    SD) in one half of his drills with the other half (split by drill, Spearman-Brown corrected)."""
    rows = []
    for d in drills:
        w = rep_pairs(feat, d)
        if f"{feature}_1" not in w or len(w) < 20:
            continue
        diff = (w[f"{feature}_1"] - w[f"{feature}_2"]).abs() / np.nanstd(np.r_[w[f"{feature}_1"], w[f"{feature}_2"]])
        rows.append(pd.DataFrame({"nflId": w.index, "drill": d, "absDiff": diff.values}))
    if not rows:
        return {}
    long = pd.concat(rows)
    long["rank"] = long.groupby("drill")["absDiff"].rank(pct=True)
    wide = long.pivot(index="nflId", columns="drill", values="rank")
    cols = list(wide.columns)
    rs = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.permutation(cols)
        h1, h2 = wide[p[: len(p) // 2]].mean(axis=1), wide[p[len(p) // 2:]].mean(axis=1)
        ok = wide[p[: len(p) // 2]].notna().sum(axis=1).ge(1) & wide[p[len(p) // 2:]].notna().sum(axis=1).ge(1)
        rs.append(np.corrcoef(h1[ok], h2[ok])[0, 1])
    r = float(np.nanmean(rs))
    return {"feature": feature, "nDrills": len(cols), "nPlayers": int(wide.notna().sum(axis=1).ge(2).sum()),
            "splitHalfR": r, "spearmanBrown": 2 * r / (1 + r)}


# ----------------------------------------------------------------------------- stopwatch validity / redundancy
def stopwatch_validity(feat: pd.DataFrame, res: pl.DataFrame, flags=None) -> pd.DataFrame:
    """Best (min) tracked time per player vs the official time: r, bias (tracking - official), SD of
    the difference. Agility clips with too few turns (incomplete) are dropped."""
    r = res.select("nflId", "group", "forty", "ten_yd_split", "three_cone", "short_shuttle").to_pandas()
    r["fly_official"] = r["forty"] - r["ten_yd_split"]
    rows = []
    specs = [("FORTY_YARD_DASH", "forty", ["tClip", "t40", "t40_onset", "t40_zero", "t2", "split2_10"]),
             ("FORTY_YARD_DASH", "forty", ["t40_minus_t2"]),
             ("FORTY_YARD_DASH", "ten_yd_split", ["t10", "t10_onset", "t10_zero", "split2_10"]),
             ("FORTY_YARD_DASH", "fly_official", ["fly10_40"]),
             ("THREE_CONE_DRILL", "three_cone", ["agTime_first", "agTime_onset", "agTime_zero", "agTime_anchor"]),
             ("SHORT_SHUTTLE", "short_shuttle", ["agTime_first", "agTime_onset", "agTime_zero", "agTime_anchor",
                                                 "agAfterFirstTurn"])]
    feat = feat.assign(t40_minus_t2=feat.get("t40") - feat.get("t2"))
    for drill, off, cands in specs:
        g = feat[feat["drillCanon"] == drill]
        if drill == "SHORT_SHUTTLE":
            g = g[g["agTurns"] >= 2]
        if drill == "THREE_CONE_DRILL":
            g = g[g["agTurns"] >= 3]
        for c in cands:
            best = g.groupby("nflId")[c].min().rename("trk").reset_index().merge(r, on="nflId").dropna(subset=["trk", off])
            if len(best) < 10:
                continue
            e = best["trk"] - best[off]
            # robust: drop |e - median| > 1 s (wrong clip) and report how many
            keep = (e - e.median()).abs() <= 1.0
            bk, ek = best[keep], e[keep]
            b, _ = np.polyfit(bk[off], bk["trk"], 1)
            rows.append({"official": off, "tracking": c, "n": len(best), "nDropped": int((~keep).sum()),
                         "r": np.corrcoef(bk["trk"], bk[off])[0, 1], "rAll": np.corrcoef(best["trk"], best[off])[0, 1],
                         "bias": ek.mean(), "sdDiff": ek.std(), "slope": b,
                         "within0.05": float(((ek - ek.mean()).abs() <= .05).mean())})
            for grp in GROUPS:
                m = bk["group"] == grp
                if m.sum() >= 15:
                    rows[-1][f"r_{grp}"] = np.corrcoef(bk.loc[m, "trk"], bk.loc[m, off])[0, 1]
    return pd.DataFrame(rows)


def player_level(feat: pd.DataFrame, drill: str, features, flags=None) -> pd.DataFrame:
    g = feat[feat["drillCanon"] == drill]
    if flags is not None:
        g = g[~g["eventId"].isin(flags.loc[flags["anyFlag"], "eventId"])]
    return g.groupby("nflId")[list(features)].mean()


def redundancy(feat, res, specs, flags=None) -> pd.DataFrame:
    """Correlation of player-mean tracking features with stopwatch / NGS numbers within group,
    plus R^2 of the feature on forty + weight (how much a stopwatch already explains)."""
    r = res.select("nflId", "group", "forty", "ten_yd_split", "three_cone", "short_shuttle",
                   "ngs_athleticism_score", "combine_weight").to_pandas().set_index("nflId")
    rows = []
    for grp, drill, feats in specs:
        pl_ = player_level(feat, drill, feats, flags).join(r, how="inner")
        pl_ = pl_[pl_["group"] == grp]
        for f in feats:
            row = {"group": grp, "drillCanon": drill, "feature": f, "n": int(pl_[f].notna().sum())}
            for o in ("forty", "ten_yd_split", "three_cone", "short_shuttle", "ngs_athleticism_score", "combine_weight"):
                ok = pl_[[f, o]].dropna()
                row[f"r_{o}"] = np.corrcoef(ok[f], ok[o])[0, 1] if len(ok) >= 15 else np.nan
                row[f"n_{o}"] = len(ok)
            ok = pl_[[f, "forty", "combine_weight"]].dropna()
            if len(ok) >= 20:
                X = np.c_[np.ones(len(ok)), ok["forty"], ok["combine_weight"]]
                beta, *_ = np.linalg.lstsq(X, ok[f], rcond=None)
                row["R2_forty_wt"] = 1 - np.var(ok[f] - X @ beta) / np.var(ok[f])
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
    return plt


def fig_paths(df: pl.DataFrame, drills, path, n=30, seed=0):
    plt = _plt()
    cols = 4
    rows = (len(drills) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.2))
    rng = np.random.default_rng(seed)
    for ax, d in zip(np.ravel(axs), drills):
        sub = df.filter(pl.col("drillCanon") == d)
        evs = sub["eventId"].unique().to_list()
        for e in rng.choice(evs, min(n, len(evs)), replace=False):
            tr = sub.filter(pl.col("eventId") == e)
            ax.plot(tr["xd"], tr["yd"], lw=.5, alpha=.8)
        ax.set_title(d[:34])
        ax.set_aspect("equal", "datalim")
        ax.plot(0, 0, "k.", ms=3)
    for ax in np.ravel(axs)[len(drills):]:
        ax.axis("off")
    fig.suptitle("Drill paths in normalize_drill frame (xd, yd), 30 random attempts each", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fig_dir_trim(dn: pd.DataFrame, tr: pd.DataFrame, path):
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))
    x = np.arange(len(dn))
    axs[0].bar(x - .2, dn["errMedian"], .4, label="median")
    axs[0].bar(x + .2, dn["errP90"], .4, label="p90")
    axs[0].set_xticks(x, dn["bin"].astype(str), rotation=45)
    axs[0].set_yscale("log")
    axs[0].set_title("|dir - position heading| (deg) by s (yd/s)")
    axs[0].legend()
    axs[1].hist(tr["s0"], bins=40, color="C0")
    axs[1].axvline(ONSET, color="k", lw=.8)
    axs[1].set_title(f"speed at first frame ({tr['startMoving'].mean():.0%} > {ONSET})")
    axs[2].hist(tr["s1rel"], bins=40, color="C1")
    axs[2].set_title(f"last-frame speed / peak ({tr['endFast'].mean():.0%} > 0.5)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fig_stopwatch(feat, res, path, flags=None):
    plt = _plt()
    r = res.select("nflId", "group", "forty", "ten_yd_split", "three_cone", "short_shuttle").to_pandas()
    specs = [("FORTY_YARD_DASH", "t40", "forty"), ("FORTY_YARD_DASH", "t40_minus_t2", "forty"),
             ("FORTY_YARD_DASH", "t10", "ten_yd_split"), ("FORTY_YARD_DASH", "split2_10", "ten_yd_split"),
             ("THREE_CONE_DRILL", "agTime_onset", "three_cone"), ("SHORT_SHUTTLE", "agAfterFirstTurn", "short_shuttle")]
    feat = feat.assign(t40_minus_t2=feat["t40"] - feat["t2"])
    fig, axs = plt.subplots(1, 6, figsize=(13, 2.5))
    for ax, (d, c, o) in zip(axs, specs):
        g = feat[feat["drillCanon"] == d]
        if "agTurns" in g and d != "FORTY_YARD_DASH":
            g = g[g["agTurns"] >= (2 if d == "SHORT_SHUTTLE" else 3)]
        m = g.groupby("nflId")[c].min().reset_index().merge(r, on="nflId").dropna(subset=[c, o])
        for i, grp in enumerate(GROUPS):
            s = m[m["group"] == grp]
            ax.scatter(s[o], s[c], s=4, label=grp, color=f"C{i}")
        e = m[c] - m[o]
        k = (e - e.median()).abs() <= 1.0
        x = np.linspace(m[o].min(), m[o].max(), 2)
        ax.plot(x, x + e[k].mean(), "k--", lw=.6)
        ax.set_xlabel(f"official {o}")
        ax.set_ylabel(f"tracking {c}")
        ax.set_title(f"r={np.corrcoef(m.loc[k, c], m.loc[k, o])[0, 1]:.2f} (n={k.sum()}, {(~k).sum()} wrong clips)\n"
                     f"bias {e[k].mean():+.2f} s (dashed)")
    axs[0].legend(markerscale=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fig_reliability(rel: pd.DataFrame, path):
    plt = _plt()
    r = rel.sort_values("ICC31_yc")
    fig, ax = plt.subplots(figsize=(5.5, max(3, .14 * len(r))))
    lab = (r["drillCanon"].str.slice(0, 22) + " | " + r["feature"] + " (n=" + r["nPairs"].astype(str) + ")").values
    v = r["ICC31_yc"].values
    ax.barh(np.arange(len(r)), v, color=np.where(v >= .7, "C2", np.where(v >= .5, "C1", "C3")))
    ax.plot(r["ICC31"], np.arange(len(r)), "k|", ms=5, label="raw (not year-centred)")
    ax.legend(fontsize=5, loc="lower right")
    ax.set_yticks(np.arange(len(r)), lab, fontsize=5)
    for v in (.5, .7):
        ax.axvline(v, color="k", lw=.5, ls="--")
    ax.set_xlabel("test-retest ICC(3,1), first two reps, year-centred")
    ax.set_xlim(-0.3, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_lateral(df: pl.DataFrame, path, seed=3):
    """Lateral velocity of three OL mirror attempts with the detected shuffle-phase end."""
    plt = _plt()
    sub = df.filter(pl.col("drillCanon") == "PASS_PRO_MIRROR_DRILL")
    evs = np.random.default_rng(seed).choice(sub["eventId"].unique().to_list(), 3, replace=False)
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.3), sharey=True)
    for ax, e in zip(axs, evs):
        g = sub.filter(pl.col("eventId") == e).to_pandas()
        k = kinematics(g, ("xd", "yd"))
        endi = mirror_phase_end(k["speed"])
        P = k["pos"][:endi] - k["pos"][:endi].mean(0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        lat = (k["pos"] - k["pos"][:endi].mean(0)) @ vt[0]
        v = savgol_filter(lat, 7, 2, deriv=1, delta=DT)
        ax.plot(k["t"], v, lw=.9, label="lateral v")
        ax.plot(k["t"], k["speed"], lw=.7, color="grey", label="speed")
        ax.axvline(k["t"][min(endi, len(k["t"]) - 1)], color="r", lw=.8, ls="--", label="sprint-out")
        ax.axhline(0, color="k", lw=.4)
        ax.set_xlabel("s")
    axs[0].set_ylabel("yd/s")
    axs[0].legend(fontsize=5)
    fig.suptitle("PASS_PRO_MIRROR_DRILL: lateral velocity (principal axis of shuffle phase)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def md(df: pd.DataFrame, floatfmt=".2f") -> str:
    """Markdown table without the tabulate dependency."""
    def f(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute attempt features")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = load_results()
    df = load_frames()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    # 1. coverage
    cov = coverage_table(df, res)
    cov.to_csv(OUT / "coverage.csv", index=False)
    print("## coverage\n", md(cov.drop(columns=["nYears"])))

    # 2. kinematic QA
    sc, sc_d = speed_check(df)
    print("## speed check", {k: round(v, 4) for k, v in sc.items()}, "\n", sc_d.head(5).to_string())
    dn = dir_noise(df)
    print("## dir noise\n", md(dn))
    tr = trim_stats(df)
    ts = trim_summary(tr)
    ts.to_csv(OUT / "trim.csv")
    print("## trim\n", ts.to_string())
    print("overall", tr[["startMoving", "endFast", "peakAtEnd", "padded"]].mean().round(3).to_dict(),
          tr.groupby("draftYear")[["startMoving", "padded"]].mean().round(3).to_dict())
    fig_dir_trim(dn, tr, FIG / "dq_dir_noise_trim.png")

    fpath = OUT / "attempt_features.parquet"
    if args.refresh or not fpath.exists():
        attempt_features(df).to_parquet(fpath)
    feat = pd.read_parquet(fpath).merge(res.select("nflId", "group").to_pandas(), on="nflId", how="left")
    print("## break detection rate (lib detect_cuts vs detect_breaks)\n",
          feat.groupby("drillCanon")[["nCutsLib", "nCuts"]].agg(lambda v: (v > 0).mean()).round(2).to_string())

    geo = geometry(feat)
    geo.to_csv(OUT / "geometry.csv", index=False)
    print("## geometry\n", geo.to_string())
    yc = year_consistency(feat)
    yc.to_csv(OUT / "year_consistency.csv", index=False)
    print("## year consistency\n", yc.to_string())
    fl = flag_outliers(feat)
    fl[["eventId", "nflId", "drillCanon", "group", "duration", "pathLen", "peakSpeed", "z_duration", "z_pathLen",
        "z_peakSpeed", "truncated", "slowRep", "longRep", "anyFlag"]].to_parquet(OUT / "outlier_flags.parquet")
    print("## outliers", fl[["truncated", "slowRep", "longRep", "anyFlag"]].sum().to_dict())
    print(fl.groupby("drillCanon")[["truncated", "slowRep", "longRep", "anyFlag"]].sum().query("anyFlag>0").to_string())
    fig_paths(df, ["FORTY_YARD_DASH", "SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT",
                   "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "W_DRILL",
                   "PASS_PRO_MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
                   "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "PASS_RUSH_DRILL"], FIG / "dq_paths.png")
    fig_lateral(df, FIG / "dq_mirror_lateral.png")

    # 3. stopwatch validity
    sv = stopwatch_validity(feat, res, fl)
    sv.to_csv(OUT / "stopwatch_validity.csv", index=False)
    print("## stopwatch\n", sv.round(3).to_string())
    fig_stopwatch(feat, res, FIG / "dq_stopwatch.png", fl)

    # 4. reliability (test-retest where 2 reps exist)
    rel_specs = {d: SPEED_FEATS + CUT_FEATS for d in sum(IDEA_DRILLS.values(), [])}
    rel_specs["FORTY_YARD_DASH"] = ["peakSpeed", "peakAccel", "tTo5"] + FORTY_FEATS
    for d in LATERAL_DRILLS:
        rel_specs[d] = SPEED_FEATS + REV_FEATS
    rel_specs["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"] = SPEED_FEATS + CUT_FEATS + HB_FEATS
    rel_specs["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL"] = SPEED_FEATS + CUT_FEATS + HB_FEATS
    for d in ("GAUNTLET_DRILL", "LINE_DRILL", "PASS_RUSH_DROP"):
        rel_specs[d] = SPEED_FEATS + CUT_FEATS
    for d in ("THREE_CONE_DRILL", "SHORT_SHUTTLE"):
        rel_specs[d] = ["agTime_anchor", "peakSpeed", "peakAccel", "peakDecel", "cut_peakDecel", "cut_codTime"]
    for d, (tg, _) in HEADING_BREAKS.items():
        rel_specs.setdefault(d, SPEED_FEATS + CUT_FEATS)
        rel_specs[d] = list(dict.fromkeys(rel_specs[d] + HB_FEATS))
    rel = pd.concat([reliability(feat, d, f, fl) for d, f in rel_specs.items()], ignore_index=True)
    relc = pd.concat([reliability(feat, d, f, fl, center=True) for d, f in rel_specs.items()], ignore_index=True)
    rel = rel.merge(relc[["drillCanon", "feature", "ICC31"]].rename(columns={"ICC31": "ICC31_yc"}),
                    on=["drillCanon", "feature"], how="left")
    rel.to_csv(OUT / "reliability.csv", index=False)
    print("## reliability\n", rel[rel["nPairs"] >= 30].round(2).to_string())
    fig_reliability(rel[(rel["nPairs"] >= 30) & rel["feature"].isin(
        ["peakSpeed", "tTo5", "peakAccel", "peakDecel", "cut_entrySpeed", "cut_minSpeed", "cut_speedRetained",
         "cut_decelTime", "cut_peakDecel", "cut_peakReaccel", "cut_codTime", "split2_10", "fly10_40", "t40", "t10",
         "v10rel", "hbTurnTime", "hbPeakYaw", "hbEntrySpeed", "duration"])], FIG / "dq_reliability.png")

    # single-rep drills: within-attempt split-half (lateral) and cross-drill parallel forms
    rev_cols = [f"rev_{c}_{h}" for c in REV_COLS for h in ("odd", "even")]
    sh = pd.concat([split_half(year_center(feat, rev_cols), d, flags=fl) for d in LATERAL_DRILLS], ignore_index=True)
    print("lateral features by year\n", feat[feat.drillCanon.isin(LATERAL_DRILLS)].groupby(["drillCanon", "draftYear"])[
        ["frames", "nRev", "revRate", "rev_latPeak", "rev_overshoot", "rev_deadTime", "rev_brakeTime"]].median().round(2).to_string())
    sh.to_csv(OUT / "split_half_reversals.csv", index=False)
    print("## split-half reversals\n", sh.round(2).to_string())
    xd_specs = {
        "WR route breaks": (IDEA_DRILLS["Brake Check (WR)"] + ["SLOT_SAIL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT"],
                            ["peakSpeed", "peakAccel"] + CUT_FEATS),
        "DB breaks": (["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "W_DRILL", "BOX_DRILL", "TERYL_AUSTIN_DRILL_1",
                       "TERYL_AUSTIN_DRILL_2", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"], ["peakSpeed", "peakAccel"] + CUT_FEATS),
        "OL lateral": (["PASS_PRO_MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"], REV_FEATS + ["peakSpeed", "peakAccel"]),
        "DL burst": (["PASS_RUSH_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_AND_CLUB_DRILL",
                      "FOUR_BAG_AGILITY_DRILL", "BODY_CONTROL_DRILL", "RUN_THE_HOOP_DRILL"], ["peakSpeed", "peakAccel", "tTo3", "tTo5"]),
    }
    xd_specs["WR route breaks"][1].extend(HB_FEATS)
    xd = pd.concat([cross_drill(year_center(feat, f), d, f, fl).assign(set=k) for k, (d, f) in xd_specs.items()],
                   ignore_index=True)
    xd.to_csv(OUT / "cross_drill.csv", index=False)
    print("## cross-drill\n", xd.round(2).to_string())
    # all drills, a few generic features
    allrel = pd.concat([reliability(feat, d, ["peakSpeed", "peakAccel", "duration", "cut_codTime", "cut_peakDecel"], fl)
                        for d in feat["drillCanon"].unique()], ignore_index=True)
    allrel.to_csv(OUT / "reliability_all.csv", index=False)
    print(allrel[allrel.nPairs >= 30].pivot(index="drillCanon", columns="feature", values="ICC31").round(2).to_string())

    cons = [consistency_check(feat, [d for d in feat["drillCanon"].unique()], f)
            for f in ("duration", "peakSpeed", "peakAccel", "cut_codTime")]
    cons = pd.DataFrame([c for c in cons if c])
    cons.to_csv(OUT / "consistency.csv", index=False)
    print("## consistency trait\n", cons.round(3).to_string())

    # 5. redundancy
    forty_r = ["peakSpeed", "tClip", "split2_10", "fly10_40", "v10rel", "d90", "peakAccel"]
    red_specs = [(g, "FORTY_YARD_DASH", forty_r) for g in GROUPS] + [
        ("WR", "CURL_ROUTE_RIGHT", ["cut_entrySpeed", "cut_minSpeed", "cut_peakDecel", "cut_codTime"]),
        ("WR", "COMEBACK_ROUTE_RIGHT", ["cut_entrySpeed", "cut_minSpeed", "cut_peakDecel", "cut_codTime"]),
        ("WR", "SPEED_OUT_ROUTE_LEFT", ["hbEntrySpeed", "hbMinSpeed", "hbTurnTime", "hbPeakReaccel"]),
        ("WR", "SLANT_ROUTE_LEFT", ["hbEntrySpeed", "hbMinSpeed", "hbTurnTime", "hbPeakReaccel"]),
        ("WR", "DAGGER_ROUTE_LEFT", ["hbEntrySpeed", "hbMinSpeed", "hbTurnTime", "cut_peakDecel"]),
        ("WR", "GAUNTLET_DRILL", ["peakSpeed", "duration"]),
        ("DB", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", ["hbEntrySpeed", "hbTurnTime", "hbPeakReaccel", "cut_codTime"]),
        ("DB", "W_DRILL", ["duration", "cut_codTime", "cut_peakDecel"]),
        ("DB", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", ["peakSpeed", "duration", "hbTurnTime", "hbPeakYaw", "cut_codTime"]),
        ("OL", "PASS_PRO_MIRROR_DRILL", ["rev_latPeak", "rev_brakeTime", "rev_overshoot", "rev_deadTime", "rev_peakLatDecel"]),
        ("OL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", ["rev_latPeak", "rev_brakeTime", "rev_peakLatDecel", "peakSpeed"]),
        ("DL", "PASS_RUSH_DRILL", ["peakSpeed", "tTo3", "tTo5", "peakAccel", "duration"]),
        ("DL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", ["rev_latPeak", "rev_brakeTime", "rev_peakLatDecel", "peakSpeed"]),
    ]
    red = redundancy(feat, res, red_specs, fl)
    relmap = rel.set_index(["drillCanon", "feature"])["ICC31_yc"].to_dict()
    relmap.update(sh.set_index(["drillCanon", "feature"])["spearmanBrown_tempo"].to_dict())
    red["reliability"] = [relmap.get((d, f), np.nan) for d, f in zip(red["drillCanon"], red["feature"])]
    red["uniqueReliable"] = red["reliability"] - red["R2_forty_wt"]
    red.to_csv(OUT / "redundancy.csv", index=False)
    cols = ["group", "drillCanon", "feature", "n", "r_forty", "r_ten_yd_split", "r_three_cone", "n_three_cone",
            "r_short_shuttle", "n_short_shuttle", "r_ngs_athleticism_score", "r_combine_weight", "R2_forty_wt",
            "reliability", "uniqueReliable"]
    print("## redundancy\n", red[cols].round(2).to_string())


if __name__ == "__main__":
    main()
