#!/usr/bin/env python
"""Raw tracking -> QA'd, cleaned, standardized parquet + docs/data_qa.md.

Handles every tracking kind found under data/raw (BDB 2027 combine_tracking and
game_tracking_<season>, or legacy tracking_week_N / 2026 input files), one file
at a time to bound memory:

    python scripts/build_processed.py [--raw data/raw] [--out data/processed] [--kinds ...]

Outputs: data/processed/<file>_clean.parquet, combine_drill_summary.parquet (Combine),
and the markdown QA report (default docs/data_qa.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from bdb27 import clean as C  # noqa: E402
from bdb27 import tracking as T  # noqa: E402


def _context(raw: Path) -> pl.DataFrame | None:
    for name in ("plays", "player_play", "supplementary"):
        try:
            return T.load_table(name, raw)
        except FileNotFoundError:
            continue
    return None


def process_file(path: Path, kind: str, raw: Path, out: Path, context, smooth: bool, drill_summary: bool):
    """Clean + prepare one tracking file; returns (section name, QA report, stats)."""
    t0 = time.time()
    df = T.cached_frame(path, out / "cache").collect()
    cleaned, report = C.clean_tracking(df)
    prepared = T.prepare_tracking(cleaned, None if kind == "combine_tracking" else context)
    if smooth:
        cols = ("xd", "yd") if "xd" in prepared.columns else ("x", "y")
        prepared = C.smooth_kinematics(prepared, cols=cols)
    dest = out / f"{path.stem}_clean.parquet"
    prepared.write_parquet(dest, compression="zstd")
    if kind == "combine_tracking" and drill_summary:
        from bdb27.drills import drill_summary as summarize
        summarize(prepared).to_parquet(out / "combine_drill_summary.parquet")
    key = T.play_key(df)
    stats = {"rows": df.height, "plays/attempts": df.select(key).n_unique(), "rows after cleaning": prepared.height,
             "output": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
             "seconds": round(time.time() - t0, 1)}
    print(f"  {path.name}: {stats}")
    return path.name, report, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--report", type=Path, default=ROOT / "docs/data_qa.md")
    ap.add_argument("--kinds", nargs="*", default=None, help=f"subset of {T.TRACKING_KINDS}")
    ap.add_argument("--limit", type=int, default=None, help="process at most N files per kind (smoke runs)")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-drill-summary", action="store_true")
    args = ap.parse_args(argv)

    found = T.discover_files(args.raw) if args.raw.exists() else {"all": []}
    kinds = [k for k in (args.kinds or T.TRACKING_KINDS) if found.get(k)]
    if not kinds:
        print(f"No tracking files found under {args.raw}.\n"
              "Download the competition data first: set KAGGLE_API_TOKEN (or ~/.kaggle/access_token), accept the\n"
              "rules on kaggle.com, then run scripts/download_kaggle.sh. Expected e.g. combine_tracking.csv,\n"
              "game_tracking_2023.csv (BDB 2027) or tracking_week_1.csv (BDB 2025).", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    context = _context(args.raw)
    reports, stats = {}, {}
    for kind in kinds:
        print(f"[{kind}]")
        for path in found[kind][: args.limit]:
            name, rep, st = process_file(path, kind, args.raw, args.out, context,
                                         not args.no_smooth, not args.no_drill_summary)
            reports[name] = rep
            stats.update({f"{name} {k}": v for k, v in st.items() if k != "seconds"})
    C.write_qa_markdown(reports, args.report, stats=stats)
    print(f"QA report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
