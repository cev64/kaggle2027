import matplotlib

matplotlib.use("Agg")
import polars as pl  # noqa: E402

from bdb27 import tracking as T  # noqa: E402
from bdb27 import viz  # noqa: E402


def test_plot_and_animate(tmp_path, clean, plays):
    df = T.prepare_tracking(clean, plays).filter((pl.col("playId") == 1) & (pl.col("frameId") <= 3))
    ax, artists = viz.plot_frame(df.filter(pl.col("frameId") == 1))
    assert len(artists) >= 3
    out = viz.animate_play(df, tmp_path / "play.gif", fps=5)
    assert out.exists() and out.stat().st_size > 0
    viz.plot_trace(df.filter(pl.col("nflId") == 1105), x="x", y="y")
