"""Writeup visuals: hero animation (docs/fig/hero.gif, hero_static.png) and pipeline schematic (docs/fig/pipeline.png).

Hero: Xavier Legette vs Keon Coleman (2024 class, picks 32 and 33). Left, their Combine 40 speed-distance traces
build up; right, each one's fastest regular-season vertical-route rep (snap to throw) as speed vs time.
Pairs were selected by scanning same-class WR pairs for a typical shrinkage ratio (see README note in the printout
and the docstring of `choose_pair` below).

Run:  python scripts/figures_hero.py [hero|pipeline|all]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "fig"

# ---------------------------------------------------------------- style
BG = "#0E1520"
PANEL = "#141E2C"
INK = "#F2F4F7"
MUTED = "#8C98AA"
GRID = "#243246"
# Okabe-Ito (colorblind-safe) accents
C_A = "#E69F00"   # orange: first player
C_B = "#56B4E9"   # sky blue: second player
C_HI = "#F0E442"  # yellow: gap / emphasis
C_GREEN = "#009E73"

_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Inter" if "Inter" in _have else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})

# Players. Chosen with `choose_pair` logic: same draft class, both 1st-round WRs picked back to back, Combine top-speed gap
# of ~10% showing up as ~4-5% in games (ratio 0.44 in logs; the WR cross-section slope is 0.42).
PLAYERS = [
    dict(nflId=57153, name="Xavier Legette", short="Legette", color=C_A, pick="No. 32"),
    dict(nflId=57154, name="Keon Coleman", short="Coleman", color=C_B, pick="No. 33"),
]
VERTICAL = ["GO", "POST", "CORNER"]


# ---------------------------------------------------------------- data
def load_hero_data():
    cap = (pl.read_parquet(PROC / "capacity_players.parquet")
           .filter(pl.col("nflId").is_in([p["nflId"] for p in PLAYERS]))
           .select("nflId", "vmax", "forty", "draftYear", "draftOverallPick"))
    capd = {r["nflId"]: r for r in cap.to_dicts()}
    out = []
    for p in PLAYERS:
        nid = p["nflId"]
        # Combine: the attempt with the highest observed speed
        t = (pl.scan_parquet(PROC / "combine_tracking_clean.parquet")
             .filter((pl.col("drillCanon") == "FORTY_YARD_DASH") & ~pl.col("isBall") & (pl.col("nflId") == nid))
             .select("eventId", "frameId", "tSec", "xd", "s").collect())
        best = t.group_by("eventId").agg(m=pl.col("s").max()).sort("m", descending=True)["eventId"][0]
        t = t.filter(pl.col("eventId") == best).sort("frameId")
        # Game: fastest vertical route with a real window (>= 3 s), snap to throw
        w = (pl.scan_parquet(PROC / "windows_WR.parquet")
             .filter((pl.col("nflId") == nid) & pl.col("routeRan").is_in(VERTICAL) & (pl.col("tWindow") >= 3.0))
             .sort("peakSpeed", descending=True).head(1).collect().row(0, named=True))
        g = (pl.scan_parquet(PROC / f"game_tracking_{w['season']}_clean.parquet")
             .filter((pl.col("gameId") == w["gameId"]) & (pl.col("playId") == w["playId"]) & (pl.col("nflId") == nid)
                     & (pl.col("framesFromSnap") >= 0))
             .select("framesFromSnap", "s", "event").sort("framesFromSnap").collect())
        n_end = int(round(w["tWindow"] * 10))
        g = g.filter(pl.col("framesFromSnap") <= n_end)
        gt = g["framesFromSnap"].to_numpy() / 10.0
        gs = g["s"].to_numpy()
        ct = t["tSec"].to_numpy()
        cx = np.clip(t["xd"].to_numpy(), 0, 40)
        cs = t["s"].to_numpy()
        out.append(dict(**p, vmax=capd[nid]["vmax"], forty=capd[nid]["forty"], pickN=capd[nid]["draftOverallPick"],
                        year=capd[nid]["draftYear"], ct=ct, cx=cx, cs=cs, obs_peak=float(cs.max()),
                        gt=gt, gs=gs, gpeak=float(gs.max()), gpeak_t=float(gt[gs.argmax()]),
                        route=w["routeRan"], season=w["season"], gameId=w["gameId"], playId=w["playId"],
                        window=w["tWindow"]))
    return out


# ---------------------------------------------------------------- hero
def interp_to(t_now, tt, vv):
    """values up to t_now, with an interpolated head point"""
    m = tt <= t_now
    if m.sum() == 0:
        return tt[:0], vv[:0]
    k = m.sum()
    xs, ys = list(tt[:k]), list(vv[:k])
    if k < len(tt):
        f = (t_now - tt[k - 1]) / (tt[k] - tt[k - 1])
        xs.append(t_now)
        ys.append(vv[k - 1] + f * (vv[k] - vv[k - 1]))
    return np.array(xs), np.array(ys)


class Hero:
    W, H = 12.0, 6.75  # inches; 100 dpi -> 1200x675

    def __init__(self, D):
        self.D = D
        a, b = D
        self.fig = plt.figure(figsize=(self.W, self.H), dpi=100)
        f = self.fig
        f.text(0.045, 0.94, "Forty cents on the dollar", fontsize=27, fontweight="bold", va="center")
        f.text(0.045, 0.875, f"Two 2024 first-round receivers, picks {a['pickN']:.0f} and {b['pickN']:.0f}. "
               "A 10.9% Combine speed edge becomes 4.6% on Sundays.",
               fontsize=12.5, color=MUTED, va="center")

        self.axL = f.add_axes([0.075, 0.30, 0.395, 0.45])
        self.axR = f.add_axes([0.575, 0.30, 0.395, 0.45], sharey=self.axL)
        for ax in (self.axL, self.axR):
            ax.set_ylim(0, 13.4)
            ax.set_yticks([0, 4, 8, 12])
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(length=0, labelsize=12, pad=6)
        self.axR.tick_params(labelleft=False)
        self.axL.set_xlim(0, 40)
        self.axL.set_xticks([0, 10, 20, 30, 40])
        self.axL.set_xlabel("distance (yd)", fontsize=12.5, labelpad=6)
        self.axR.set_xlim(0, 6.4)
        self.axR.set_xticks([0, 2, 4, 6])
        self.axR.set_xlabel("seconds after the snap", fontsize=12.5, labelpad=6)
        self.axL.set_ylabel("speed (yd/s)", fontsize=12.5, labelpad=4)

        f.text(0.075, 0.825, "NFL Combine 40-yard dash", fontsize=15, fontweight="bold", va="center")
        f.text(0.575, 0.825, "Fastest regular-season vertical-route rep", fontsize=15, fontweight="bold", va="center")
        f.text(0.075, 0.788, "top speed, from the fitted sprint curve", fontsize=11.5, color=MUTED, va="center")
        f.text(0.575, 0.788, "snap to throw, 2024 and 2025 regular seasons", fontsize=11.5, color=MUTED, va="center")

        # Combine vmax reference lines (both panels: left is the Combine, right shows the same level faded)
        self.lines = {}
        for p in D:
            c = p["color"]
            self.axL.axhline(p["vmax"], color=c, lw=1.4, ls=(0, (4, 3)), alpha=0.9)
            self.axR.axhline(p["vmax"], color=c, lw=1.1, ls=(0, (4, 3)), alpha=0.35)
            l1, = self.axL.plot([], [], color=c, lw=3.4, solid_capstyle="round")
            h1, = self.axL.plot([], [], "o", color=c, ms=9, mec=BG, mew=1.6, zorder=5)
            l2, = self.axR.plot([], [], color=c, lw=3.4, solid_capstyle="round")
            h2, = self.axR.plot([], [], "o", color=c, ms=9, mec=BG, mew=1.6, zorder=5)
            self.lines[p["short"]] = (l1, h1, l2, h2)
        # direct labels (always on)
        a, b = D
        self.axL.text(39.5, a["vmax"] + 0.28, f"{a['short']}  {a['vmax']:.1f}", color=a["color"], fontsize=14.5,
                      fontweight="bold", va="bottom", ha="right")
        self.axL.text(39.5, b["vmax"] - 1.5, f"{b['short']}  {b['vmax']:.1f}", color=b["color"], fontsize=14.5,
                      fontweight="bold", va="top", ha="right")
        self.axL.text(39.5, 0.5, "dashes: fitted Combine top speed", color=MUTED, fontsize=10.5, ha="right",
                      va="bottom")
        # animated, revealed later
        self.rev = []
        self.clock = f.text(0.97, 0.935, "", fontsize=13, color=MUTED, ha="right", va="center")
        self.panel_notes()

    def panel_notes(self):
        a, b = self.D
        # Game peak labels + shrinkage brackets (right panel) -- revealed at end
        self.rev_artists = []
        ax = self.axR
        for p in self.D:
            c = p["color"]
            x = p["gpeak_t"]
            ax.plot([x, x], [p["gpeak"], p["vmax"]], color=c, lw=2.2, alpha=0.0, zorder=3)
        # bottom strip: the punchline
        f = self.fig
        self.strip = []
        ce, ge = a["vmax"] - b["vmax"], a["gpeak"] - b["gpeak"]
        self.edge = (ce, ge)
        y0 = 0.115
        t1 = f.text(0.075, y0 + 0.03, f"{ce:.2f}", fontsize=34, fontweight="bold", color=C_HI, va="center")
        t1b = f.text(0.075, y0 - 0.045, "yd/s Combine edge,\nLegette over Coleman", fontsize=12, color=MUTED, va="center", linespacing=1.25)
        t2 = f.text(0.40, y0 + 0.03, f"{ge:.2f}", fontsize=34, fontweight="bold", color=C_HI, va="center")
        t2b = f.text(0.40, y0 - 0.045, "yd/s game edge\n(best vertical rep each)", fontsize=12, color=MUTED, va="center", linespacing=1.25)
        t3 = f.text(0.72, y0 + 0.03, f"{ge / ce * 100:.0f}¢", fontsize=34, fontweight="bold", color=C_HI, va="center")
        t3b = f.text(0.72, y0 - 0.045, "on the dollar here. WR estimates\nrange 27–38¢ (n = 65)", fontsize=12, color=MUTED, va="center", linespacing=1.25)
        arr1 = f.text(0.315, y0 + 0.03, "→", fontsize=30, color=MUTED, va="center", ha="center")
        arr2 = f.text(0.64, y0 + 0.03, "=", fontsize=30, color=MUTED, va="center", ha="center")
        self.strip = [t1, t1b, t2, t2b, t3, t3b, arr1, arr2]
        for t in self.strip:
            t.set_alpha(0)
        # Game-peak value labels and the drop annotation
        self.peak_txt = []
        for k, p in enumerate(self.D):
            c = p["color"]
            drop = (1 - p["gpeak"] / p["vmax"]) * 100
            mid = 0.5 * (p["gpeak"] + p["vmax"])
            side = -1 if k == 0 else 1
            td = ax.text(p["gpeak_t"] + side * 0.12, mid, f"{p['gpeak']:.1f}  ·  −{drop:.0f}%", color=c, fontsize=14,
                         fontweight="bold", va="center", ha="right" if side < 0 else "left", alpha=0)
            self.peak_txt.append((td, td))
        self.drop_lines = [ax.plot([p["gpeak_t"]] * 2, [p["gpeak"], p["vmax"]], color=p["color"], lw=2.2, alpha=0)[0]
                           for p in self.D]
        self.peak_dots = [ax.plot([p["gpeak_t"]], [p["gpeak"]], "o", color=p["color"], ms=11, mec=INK, mew=1.8, alpha=0, zorder=6)[0]
                          for p in self.D]
        self.thr = []
        for p in self.D:
            x = p["gt"][-1]
            self.thr.append(ax.annotate("throw", xy=(x, p["gs"][-1]), xytext=(0, -16), textcoords="offset points",
                                        color=p["color"], fontsize=10.5, ha="center", va="top", alpha=0))

    def set_state(self, t_anim, reveal, speed=1.2):
        td = t_anim * speed
        for p in self.D:
            l1, h1, l2, h2 = self.lines[p["short"]]
            # Combine: speed vs distance, time-driven
            tx, ty = interp_to(td, p["ct"], p["cs"])
            _, xs = interp_to(td, p["ct"], p["cx"])
            l1.set_data(xs, ty)
            h1.set_data(xs[-1:], ty[-1:])
            gx, gy = interp_to(td, p["gt"], p["gs"])
            l2.set_data(gx, gy)
            h2.set_data(gx[-1:], gy[-1:])
        self.clock.set_text(f"t = {min(td, 6.0):.1f} s")
        for (tx, td_), dl, dot, p in zip(self.peak_txt, self.drop_lines, self.peak_dots, self.D):
            for art in (tx, td_, dl, dot):
                art.set_alpha(reveal)
        for t in self.thr:
            t.set_alpha(min(1.0, reveal))
        for t in self.strip:
            t.set_alpha(reveal)

    def set_final(self):
        self.clock.set_text("")
        self.set_state(6.0 / 1.2, 1.0)
        self.clock.set_text("")

    def frame(self, t_anim, reveal):
        self.set_state(t_anim, reveal)
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba()).copy()).convert("RGB")


def make_hero():
    FIG.mkdir(parents=True, exist_ok=True)
    D = load_hero_data()
    for p in D:
        print(f"{p['name']}: Combine vmax {p['vmax']:.2f} (observed {p['obs_peak']:.2f}, 40 = {p['forty']}), "
              f"game best vertical {p['gpeak']:.2f} on a {p['route']} in {p['season']} "
              f"(game {p['gameId']}, play {p['playId']}, window {p['window']:.1f}s), game/Combine {p['gpeak'] / p['vmax']:.3f}")
    h = Hero(D)
    fps = 20
    build = 5.0
    frames, durs = [], []
    for i in range(int(build * fps) + 1):
        frames.append(h.frame(i / fps, 0.0))
        durs.append(1000 // fps)
    n_fade = 6
    for k in range(1, n_fade + 1):
        frames.append(h.frame(build, k / n_fade))
        durs.append(1000 // fps)
    durs[-1] = 1800  # hold on the final frame
    total = sum(durs) / 1000
    print(f"animation length {total:.2f}s, {len(frames)} frames")
    pal = frames[-1].quantize(colors=40, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in frames]
    q[0].save(FIG / "hero.gif", save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True, disposal=1)
    # static: full-resolution final frame
    h.fig.set_dpi(200)
    h.set_final()
    h.fig.savefig(FIG / "hero_static.png", dpi=200)
    plt.close(h.fig)
    print("hero.gif", (FIG / "hero.gif").stat().st_size / 1e6, "MB;",
          "hero_static.png", (FIG / "hero_static.png").stat().st_size / 1e6, "MB")


# ---------------------------------------------------------------- pipeline
def spark_axes(fig, cx, cy, w, h):
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def make_pipeline():
    FIG.mkdir(parents=True, exist_ok=True)
    W, H = 16.0, 5.4
    fig = plt.figure(figsize=(W, H), dpi=100)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    bg.text(0.55, 4.85, "From a 40-yard dash to a Sunday forecast", fontsize=26, fontweight="bold", va="center")
    bg.text(0.55, 4.35, "Five steps, every one computed from player tracking data", fontsize=14, color=MUTED, va="center")

    steps = [
        ("1", "Tracked Combine 40", "speed sampled 10x a second\nfor every attempt", C_B),
        ("2", "Capacity curve", "mono-exponential fit gives\ntop speed and burst", C_B),
        ("3", "Game windows", "regular-season reps from the\nsnap to the throw", C_A),
        ("4", "Elasticity by position", "how much of a Combine edge\nsurvives into games", C_A),
        ("5", "Sunday Speed", "projected game top speed,\nwith its uncertainty", C_HI),
    ]
    n = len(steps)
    bw, gap = 2.78, 0.43
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    yb, hb = 0.55, 3.35
    centers = []
    for i, (num, title, sub, col) in enumerate(steps):
        x = x0 + i * (bw + gap)
        cx = x + bw / 2
        centers.append(cx)
        bg.add_patch(FancyBboxPatch((x, yb), bw, hb, boxstyle="round,pad=0,rounding_size=0.18", fc=PANEL, ec=GRID, lw=1.4))
        bg.text(x + 0.22, yb + hb - 0.34, num, fontsize=15, fontweight="bold", color=BG, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.32", fc=col, ec="none"))
        bg.text(x + 0.55, yb + hb - 0.34, title, fontsize=14.5, fontweight="bold", va="center")
        bg.text(x + 0.22, yb + 0.58, sub, fontsize=11.8, color=MUTED, va="center", linespacing=1.35)
        if i < n - 1:
            bg.add_patch(FancyArrowPatch((x + bw + 0.05, yb + hb / 2), (x + bw + gap - 0.05, yb + hb / 2),
                                         arrowstyle="-|>", mutation_scale=22, color=MUTED, lw=2))

    sw, sh = bw - 0.5, 1.5
    sy = yb + 1.72
    px = lambda i: centers[i]

    # 1 tracked 40: noisy sampled speed trace with dots
    ax = spark_axes(fig, px(0) / W, 0, 0, 0)
    ax.remove()
    def mk(i):
        a = spark_axes(fig, px(i) / W, sy / H, sw / W, sh / H)
        return a
    a1 = mk(0)
    t = np.linspace(0, 4.8, 49)
    v = 11.4 * (1 - np.exp(-(t - 0.07) / 1.35)).clip(0)
    rng = np.random.default_rng(3)
    a1.plot(t, v + rng.normal(0, 0.12, t.size), "o", color=C_B, ms=3.8, alpha=0.95)
    a1.plot(t, v, color=C_B, lw=1.2, alpha=0.35)
    a1.set_xlim(-0.1, 5.0)
    a1.set_ylim(0, 13.5)
    a1.axhline(0, color=GRID, lw=1.2)
    a1.text(4.9, 1.0, "time", color=MUTED, fontsize=10.5, ha="right")
    a1.text(0.1, 12.3, "speed", color=MUTED, fontsize=10.5, ha="left")

    # 2 capacity curve: fit + asymptote + burst band
    a2 = mk(1)
    t2 = np.linspace(0, 6, 120)
    v2 = 12.0 * (1 - np.exp(-(t2 - 0.07) / 1.4)).clip(0)
    a2.axhline(12.0, color=C_HI, lw=1.4, ls=(0, (4, 3)))
    a2.plot(t2, v2, color=C_B, lw=3.2)
    a2.axvspan(1.0, 2.6, color=C_B, alpha=0.18, lw=0)
    a2.text(0.1, 12.3, "top speed", color=C_HI, fontsize=11.5, fontweight="bold", va="bottom")
    a2.text(1.8, 1.0, "burst", color=C_B, fontsize=11.5, fontweight="bold", ha="center", va="bottom")
    a2.set_xlim(0, 6)
    a2.set_ylim(0, 15)
    a2.axhline(0, color=GRID, lw=1.2)

    # 3 game window: speed from snap to throw
    a3 = mk(2)
    t3 = np.linspace(0, 3.6, 80)
    v3 = 9.4 * (1 - np.exp(-t3 / 1.25)) + 0.15 * np.sin(t3 * 3)
    a3.plot(t3, v3, color=C_A, lw=3.2)
    a3.axvline(0, color=MUTED, lw=1.2)
    a3.axvline(3.6, color=MUTED, lw=1.2)
    a3.text(0.1, 12.2, "snap", color=MUTED, fontsize=11.5, ha="left", va="bottom")
    a3.text(3.5, 12.2, "throw", color=MUTED, fontsize=11.5, ha="right", va="bottom")
    a3.plot([t3[-1]], [v3[-1]], "o", color=C_A, ms=8, mec=BG, mew=1.5)
    a3.set_xlim(-0.3, 3.9)
    a3.set_ylim(0, 15)
    a3.axhline(0, color=GRID, lw=1.2)

    # 4 elasticity bars by position (summary values from docs/translation.md, with body weight held fixed)
    a4 = mk(3)
    pos = ["WR", "DB", "DL"]
    el = [0.27, 0.38, 0.42]
    cols = [C_A, C_A, C_A]
    xs = np.arange(3)
    a4.bar(xs, el, width=0.56, color=cols, alpha=0.95)
    a4.axhline(1.0, color=C_HI, lw=1.4, ls=(0, (4, 3)))
    a4.text(2.45, 1.03, "1.0 = full pass-through", color=C_HI, fontsize=10.5, ha="right", va="bottom")
    for x, e, p in zip(xs, el, pos):
        a4.text(x, e + 0.04, f"{e:.2f}", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=INK)
        a4.text(x, 0.03, p, ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=BG)
    a4.set_ylim(0, 1.25)
    a4.set_xlim(-0.6, 2.6)
    a4.axhline(0, color=GRID, lw=1.2)

    # 5 projection: line with interval band and a highlighted player
    a5 = mk(4)
    xx = np.linspace(-1, 1, 50)
    yy = 0.4 * xx
    a5.fill_between(xx, yy - 0.28, yy + 0.28, color=C_HI, alpha=0.16, lw=0)
    a5.plot(xx, yy, color=C_HI, lw=3.2)
    a5.plot([0.55], [0.22], "o", color=C_HI, ms=10, mec=BG, mew=1.6)
    a5.plot([0.55, 0.55], [0.22 - 0.28, 0.22 + 0.28], color=C_HI, lw=1.6, alpha=0.9)
    a5.axhline(-0.82, color=GRID, lw=1.2)
    a5.set_xlim(-1.1, 1.15)
    a5.set_ylim(-0.82, 0.8)
    a5.text(-1.0, 0.72, "game", color=MUTED, fontsize=10.5, va="top")
    a5.text(1.1, -0.74, "Combine", color=MUTED, fontsize=10.5, ha="right", va="bottom")

    out = FIG / "pipeline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("pipeline.png", out.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("hero", "all"):
        make_hero()
    if what in ("pipeline", "all"):
        make_pipeline()
