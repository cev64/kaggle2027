"""Download nflverse external data into data/external and print row counts.

Usage::

    python scripts/fetch_external.py                     # 2022-2025, all datasets
    python scripts/fetch_external.py --seasons 2024 2025 --datasets pbp participation
    python scripts/fetch_external.py --refresh           # ignore the cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdb import external


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(external.DEFAULT_SEASONS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(external.DATASETS), default=None)
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    results = external.fetch_all(args.seasons, args.datasets, args.refresh)
    failures = 0
    print(f"{'dataset':<16}{'season':>8}{'rows':>12}")
    for (name, season), rows in results.items():
        label = "all" if season is None else str(season)
        if isinstance(rows, str):
            failures += 1
            print(f"{name:<16}{label:>8}  {rows}")
        else:
            print(f"{name:<16}{label:>8}{rows:>12,}")
    size_mb = sum(p.stat().st_size for p in external.EXTERNAL_DIR.rglob("*.parquet")) / 1e6
    print(f"\ncache: {external.EXTERNAL_DIR} ({size_mb:,.1f} MB), failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
