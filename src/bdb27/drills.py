"""Single-entity movement features for Combine drills (and cohort game reps).

A *trajectory* is a pandas DataFrame for one entity in one attempt/play, with
``frameId`` (10 Hz) and position columns (default ``xd, yd`` from
:func:`bdb.tracking.normalize_drill`, else ``x, y``); ``s``/``dir`` are used when
present. Velocities and speed derivatives are Savitzky-Golay smoothed.

* :func:`speed_profile` - peak speed, time to top speed / thresholds, peak accel/decel.
* :func:`detect_cuts` - change-of-direction events: turn angle, entry/min/exit speed,
  deceleration rate/time/distance into the break, re-acceleration and COD time.
* :func:`drill_summary` - both of the above for every player-attempt.
* :func:`reaction_lag`, :func:`flip_latencies` - leader/follower lag (mirror drills).
* :func:`response_after_event` - latency to accelerate after an event (e.g. pass_forward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from .tracking import DT

MOTION_SPEED = 0.5  # yd/s; movement onset threshold


def _pd(df) -> pd.DataFrame:
    return df.to_pandas() if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame) else df


def _xy_cols(traj: pd.DataFrame, xy=None) -> tuple[str, str]:
    if xy:
        return xy
    return ("xd", "yd") if {"xd", "yd"} <= set(traj.columns) else ("x", "y")


def _sg(v: np.ndarray, window: int, deriv: int = 0) -> np.ndarray:
    if len(v) < window or window < 3:
        return np.gradient(v, DT) if deriv else v
    return savgol_filter(v, window, 2, deriv=deriv, delta=DT)


def kinematics(traj, xy=None, window: int = 5) -> dict[str, np.ndarray]:
    """Arrays for one trajectory: t, pos (n,2), vel (n,2), speed, dsdt, heading (deg, math, unwrapped).

    Velocity is the smoothed derivative of position (frame-consistent after any
    drill rotation); speed uses ``s`` when present.
    """
    traj = _pd(traj).sort_values("frameId")
    cx, cy = _xy_cols(traj, xy)
    t = (traj["frameId"].to_numpy(float) - traj["frameId"].iloc[0]) * DT
    pos = traj[[cx, cy]].to_numpy(float)
    vel = np.c_[_sg(pos[:, 0], window, 1), _sg(pos[:, 1], window, 1)]
    speed = traj["s"].to_numpy(float) if "s" in traj and traj["s"].notna().all() else np.linalg.norm(vel, axis=1)
    speed = _sg(speed, window)
    head = np.degrees(np.unwrap(np.arctan2(vel[:, 1], vel[:, 0])))
    return {"t": t, "pos": pos, "vel": vel, "speed": speed, "dsdt": _sg(speed, window, 1), "heading": head}


def speed_profile(traj, xy=None, thresholds=(3.0, 5.0, 7.0), window: int = 5) -> dict:
    """Peak speed, time to top speed (95% of peak) and thresholds from motion onset,
    peak acceleration/deceleration (yd/s^2, decel positive), distance and duration."""
    k = kinematics(traj, xy, window)
    t, s, dsdt = k["t"], k["speed"], k["dsdt"]
    moving = np.flatnonzero(s > MOTION_SPEED)
    t0 = t[moving[0]] if len(moving) else np.nan
    i_pk = int(np.argmax(s))

    def t_reach(v):
        hit = np.flatnonzero(s >= v)
        return t[hit[0]] - t0 if len(hit) else np.nan

    out = {"peakSpeed": s[i_pk], "tPeakSpeed": t[i_pk] - t0, "tTopSpeed95": t_reach(0.95 * s[i_pk]),
           "peakAccel": np.nanmax(dsdt), "peakDecel": max(0.0, -np.nanmin(dsdt)),
           "distance": float(np.linalg.norm(np.diff(k["pos"], axis=0), axis=1).sum()),
           "duration": t[-1] - t0 if len(moving) else 0.0, "tMotionStart": t0}
    out.update({f"tTo{v:g}": t_reach(v) for v in thresholds})
    return out


def detect_cuts(traj, xy=None, min_turn: float = 30.0, min_drop: float = 0.75, window: int = 5,
                regain: float = 0.9) -> pd.DataFrame:
    """Change-of-direction events, found as speed minima with prominence >= ``min_drop``
    yd/s and a heading change >= ``min_turn`` deg between entry and exit speed peaks.

    Per cut: ``tCut`` (time of min speed), ``turnAngle`` (deg, + = left/CCW in the
    trajectory's frame), ``entrySpeed``/``minSpeed``/``exitSpeed``, ``speedRetained``,
    ``decelTime``/``decelDist``/``meanDecel``/``peakDecel`` (entry peak -> min),
    ``peakReaccel`` and ``timeToRegain`` (min -> ``regain`` x entry speed), and
    ``codTime`` (entry peak -> regain).
    """
    k = kinematics(traj, xy, window)
    t, s, dsdt, vel, pos = k["t"], k["speed"], k["dsdt"], k["vel"], k["pos"]
    mins, _ = find_peaks(-s, prominence=min_drop)
    bounds = np.r_[0, mins, len(s) - 1]
    rows = []
    for j, m in enumerate(mins):
        seg = s[bounds[j]:m + 1]  # decel start = last frame near the pre-cut peak speed
        i0 = bounds[j] + int(np.flatnonzero(seg >= 0.98 * seg.max())[-1])
        i1 = m + int(np.argmax(s[m:bounds[j + 2] + 1]))
        if i0 == m or i1 == m:
            continue
        turn = np.degrees(np.arctan2(vel[i0, 0] * vel[i1, 1] - vel[i0, 1] * vel[i1, 0], np.dot(vel[i0], vel[i1])))
        if abs(turn) < min_turn:
            continue
        after = np.flatnonzero(s[m:] >= regain * s[i0])
        t_regain = t[m + after[0]] - t[m] if len(after) else np.nan
        dec_t = t[m] - t[i0]
        rows.append({"tCut": t[m], "turnAngle": turn, "entrySpeed": s[i0], "minSpeed": s[m], "exitSpeed": s[i1],
                     "speedRetained": s[m] / s[i0], "decelTime": dec_t,
                     "decelDist": float(np.linalg.norm(np.diff(pos[i0:m + 1], axis=0), axis=1).sum()),
                     "meanDecel": (s[i0] - s[m]) / dec_t, "peakDecel": -np.min(dsdt[i0:m + 1]),
                     "peakReaccel": np.max(dsdt[m:i1 + 1]), "timeToRegain": t_regain, "codTime": dec_t + t_regain})
    return pd.DataFrame(rows, columns=["tCut", "turnAngle", "entrySpeed", "minSpeed", "exitSpeed", "speedRetained",
                                       "decelTime", "decelDist", "meanDecel", "peakDecel", "peakReaccel",
                                       "timeToRegain", "codTime"])


def drill_summary(df, key=None, xy=None, min_frames: int = 10, **cut_kw) -> pd.DataFrame:
    """Speed profile + cut aggregates for every player trajectory (ball rows skipped).

    ``key`` defaults to the attempt key (``draftYear, eventId`` / ``eventId``) or
    ``gameId, playId``; ``nflId`` is always added. Cut columns are prefixed
    ``cut1_`` (first cut) plus ``nCuts``, ``meanSpeedRetained``, ``maxPeakDecel``.
    """
    df = _pd(df)
    if "isBall" in df.columns:
        df = df[~df["isBall"].astype(bool)]
    if key is None:
        key = (["gameId", "playId"] if {"gameId", "playId"} <= set(df.columns)
               else [c for c in ("draftYear", "eventId") if c in df.columns])
    meta = [c for c in ("drillType", "drillName", "attempt") if c in df.columns and c not in key]
    rows = []
    for k, g in df.groupby(key + ["nflId"], sort=False, observed=True):
        if len(g) < min_frames:
            continue
        row = dict(zip(key + ["nflId"], k if isinstance(k, tuple) else (k,)))
        row.update({c: g[c].iloc[0] for c in meta})
        row.update(speed_profile(g, xy))
        cuts = detect_cuts(g, xy, **cut_kw)
        row.update(nCuts=len(cuts), meanSpeedRetained=cuts["speedRetained"].mean() if len(cuts) else np.nan,
                   maxPeakDecel=cuts["peakDecel"].max() if len(cuts) else np.nan)
        if len(cuts):
            row.update({f"cut1_{c}": v for c, v in cuts.iloc[0].items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ two entities
def _aligned_vel(leader, follower, xy, window):
    a, b = _pd(leader).sort_values("frameId"), _pd(follower).sort_values("frameId")
    common = np.intersect1d(a["frameId"], b["frameId"])
    a, b = a[a["frameId"].isin(common)], b[b["frameId"].isin(common)]
    return common, kinematics(a, xy, window)["vel"], kinematics(b, xy, window)["vel"]


def reaction_lag(leader, follower, axis: str = "y", xy=None, max_lag: float = 1.0, window: int = 5) -> dict:
    """Lag (s) maximizing the cross-correlation of the two entities' velocity along
    ``axis`` ('x', 'y' or 'both'). Positive = follower trails leader."""
    _, va, vb = _aligned_vel(leader, follower, xy, window)
    cols = {"x": [0], "y": [1], "both": [0, 1]}[axis]
    va, vb = va[:, cols] - va[:, cols].mean(0), vb[:, cols] - vb[:, cols].mean(0)
    best = (np.nan, -np.inf)
    for L in range(-int(max_lag / DT), int(max_lag / DT) + 1):
        x, y = (va[: len(va) - L], vb[L:]) if L >= 0 else (va[-L:], vb[: len(vb) + L])
        den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        if len(x) > 5 and den > 0 and (r := (x * y).sum() / den) > best[1]:
            best = (L * DT, r)
    return {"lag": best[0], "corr": best[1]}


def flip_latencies(leader, follower, axis: str = "y", xy=None, min_speed: float = 0.5,
                   max_lag: float = 1.5, window: int = 5) -> pd.DataFrame:
    """For each leader reversal along ``axis`` (velocity sign change, |v| > ``min_speed``
    on both sides), the time until the follower reverses the same way."""
    frames, va, vb = _aligned_vel(leader, follower, xy, window)
    c = {"x": 0, "y": 1}[axis]
    a, b = va[:, c], vb[:, c]

    def flips(v):
        sgn = np.where(v > min_speed, 1, np.where(v < -min_speed, -1, 0))
        idx = np.flatnonzero(sgn)
        return [(idx[i + 1], sgn[idx[i + 1]]) for i in range(len(idx) - 1) if sgn[idx[i]] != sgn[idx[i + 1]]]

    fb = flips(b)
    rows = []
    for i, sg in flips(a):
        nxt = [j for j, sb in fb if sb == sg and 0 <= j - i <= max_lag / DT]
        rows.append({"frameId": frames[i], "newSign": sg, "latency": (nxt[0] - i) * DT if nxt else np.nan})
    return pd.DataFrame(rows, columns=["frameId", "newSign", "latency"])


def response_after_event(traj, event_frame: int, xy=None, window_s: float = 1.5, speed_gain: float = 1.0,
                         window: int = 5) -> dict:
    """Reaction to an event at ``event_frame`` (e.g. pass_forward in game tracking):
    time to peak acceleration, time to gain ``speed_gain`` yd/s, and heading change."""
    k = kinematics(traj, xy, window)
    frames = _pd(traj).sort_values("frameId")["frameId"].to_numpy()
    sel = np.flatnonzero((frames >= event_frame) & (frames <= event_frame + window_s / DT))
    if not len(sel):
        return {"tPeakAccel": np.nan, "peakAccel": np.nan, "tSpeedGain": np.nan, "headingChange": np.nan}
    i0, s, a = sel[0], k["speed"], k["dsdt"]
    j = sel[int(np.argmax(a[sel]))]
    gain = sel[s[sel] >= s[i0] + speed_gain]
    return {"tPeakAccel": (frames[j] - event_frame) * DT, "peakAccel": a[j],
            "tSpeedGain": (frames[gain[0]] - event_frame) * DT if len(gain) else np.nan,
            "headingChange": k["heading"][sel[-1]] - k["heading"][i0]}
