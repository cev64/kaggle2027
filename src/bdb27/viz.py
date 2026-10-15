"""Field plotting and play animation (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from .tracking import FIELD_LENGTH, FIELD_WIDTH

HASH_Y = (70.75 / 3, FIELD_WIDTH - 70.75 / 3)  # NFL hashes: 70'9" from each sideline
SIDE_COLORS = {"offense": "#d62728", "defense": "#1f77b4", "ball": "#8c564b"}


def draw_field(ax=None, figsize=(12, 5.33), los_x: float | None = None, first_down_x: float | None = None,
               color: str = "#2e7d32"):
    """Draw a to-scale 120 x 53.3 yd field (end zones, 5-yd lines, numbers, hash marks)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.add_patch(plt.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, fc=color, ec="white", lw=2, zorder=0))
    for x0 in (0, 110):
        ax.add_patch(plt.Rectangle((x0, 0), 10, FIELD_WIDTH, fc="#1b5e20", ec="white", lw=1.5, zorder=0))
    for x in range(15, 110, 5):
        ax.axvline(x, color="white", lw=1.2 if x % 10 == 0 else 0.6, zorder=1)
    for x in range(11, 110):  # 1-yd ticks at sidelines and hashes
        for y in (0.4, HASH_Y[0], HASH_Y[1], FIELD_WIDTH - 0.4):
            ax.plot([x, x], [y - 0.35, y + 0.35], color="white", lw=0.6, zorder=1)
    for x in range(20, 110, 10):
        num = x - 10 if x <= 60 else 110 - x
        for y, rot in ((12, 0), (FIELD_WIDTH - 12, 180)):
            ax.text(x, y, str(num), color="white", fontsize=12, ha="center", va="center", rotation=rot, zorder=1)
    if los_x is not None:
        ax.axvline(los_x, color="#1565c0", lw=2, zorder=2)
    if first_down_x is not None:
        ax.axvline(first_down_x, color="#fdd835", lw=2, zorder=2)
    ax.set_xlim(-2, FIELD_LENGTH + 2)
    ax.set_ylim(-2, FIELD_WIDTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _colors(frame: pd.DataFrame, palette: dict | None):
    col = next((c for c in ("side", "club") if c in frame.columns), None)
    if col is None:
        return ["white"] * len(frame)
    keys = frame[col].astype(str)
    if palette is None:
        palette = dict(SIDE_COLORS) if col == "side" else {}
        for i, k in enumerate(k for k in keys.unique() if k not in palette and k != "football"):
            palette[k] = plt.cm.tab10(i)
        palette.setdefault("football", SIDE_COLORS["ball"])
    return [palette.get(k, "white") for k in keys]


def plot_frame(frame, ax=None, arrows: bool = True, labels: str | None = "jerseyNumber", palette=None,
               arrow_scale: float = 0.5, **field_kw):
    """Plot one frame: players colored by side/club, ball in brown, velocity arrows (vx, vy)."""
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") and not isinstance(frame, pd.DataFrame) else frame
    if ax is None or field_kw:
        ax = draw_field(ax, **field_kw)
    ball = frame["isBall"].astype(bool) if "isBall" in frame.columns else pd.Series(False, index=frame.index)
    pl_ = frame[~ball]
    artists = [ax.scatter(pl_["x"], pl_["y"], c=_colors(pl_, palette), s=120, ec="black", lw=0.8, zorder=4)]
    if ball.any():
        b = frame[ball]
        artists.append(ax.scatter(b["x"], b["y"], c=SIDE_COLORS["ball"], s=50, marker="o", ec="white", zorder=5))
    if arrows and {"vx", "vy"} <= set(pl_.columns):
        artists.append(ax.quiver(pl_["x"], pl_["y"], pl_["vx"] * arrow_scale, pl_["vy"] * arrow_scale,
                                 angles="xy", scale_units="xy", scale=1, width=0.002, color="black", zorder=3))
    if labels and labels in pl_.columns:
        for x, y, t in zip(pl_["x"], pl_["y"], pl_[labels]):
            if pd.notna(t):
                artists.append(ax.text(x, y, str(int(t)) if isinstance(t, float) else str(t),
                                       fontsize=6, ha="center", va="center", color="white", zorder=6))
    return ax, artists


def animate_play(play, out_path: str | Path, fps: int = 10, figsize=(12, 5.33), **plot_kw) -> Path:
    """Animate one play to GIF (Pillow) or MP4 (ffmpeg) depending on the file suffix."""
    play = play.to_pandas() if hasattr(play, "to_pandas") and not isinstance(play, pd.DataFrame) else play
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.sort(play["frameId"].unique())
    fig, ax = plt.subplots(figsize=figsize)
    los = play["losX"].dropna().iloc[0] if "losX" in play.columns and play["losX"].notna().any() else None
    draw_field(ax, los_x=los)
    title = ax.set_title("")
    state: list = []
    by_frame = dict(tuple(play.groupby("frameId")))

    def update(fid):
        for a in state:
            a.remove()
        state.clear()
        state.extend(plot_frame(by_frame[fid], ax=ax, **plot_kw)[1])
        ev = by_frame[fid]["event"].dropna() if "event" in play.columns else []
        title.set_text(f"game {play['gameId'].iloc[0]} play {play['playId'].iloc[0]} frame {fid}"
                       + (f" - {ev.iloc[0]}" if len(ev) else ""))
        return state

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    writer = animation.PillowWriter(fps=fps) if out_path.suffix.lower() == ".gif" else animation.FFMpegWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def plot_trace(traj, x: str = "xd", y: str = "yd", color: str = "s", ax=None, cmap: str = "viridis",
               label: str | None = None, colorbar: bool = True):
    """Plot one drill/route trace (e.g. drill-relative ``xd, yd``) colored by ``color`` (speed)."""
    traj = traj.to_pandas() if hasattr(traj, "to_pandas") and not isinstance(traj, pd.DataFrame) else traj
    traj = traj.sort_values("frameId")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj[x], traj[y], color="0.8", lw=1, zorder=1)
    sc = ax.scatter(traj[x], traj[y], c=traj[color] if color in traj else None, cmap=cmap, s=14, zorder=2,
                    label=label)
    if colorbar and color in traj:
        plt.colorbar(sc, ax=ax, label=color)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{x} (yd)")
    ax.set_ylabel(f"{y} (yd)")
    return ax
