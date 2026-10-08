"""
WealthOS — Unified Runner
==========================
Runs the production deployers in sequence, prints a consolidated summary,
then regenerates dashboard.html.

Weekly workflow (Friday 3:15pm):
    python run_all.py

Dry-run (print plan, don't execute):
    python run_all.py --dry-run

Momentum deployers are parked in momentum/ and are NOT run here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent

STEPS = [
    ("1/2  NIFTYSHOP  (weekly, Nifty 50)",   "NiftyShop",  "niftyshop_deploy.py"),
    ("2/2  MIDCAPSHOP  (weekly, Midcap 50)", "MidcapShop", "midcap_niftyshop_deploy.py"),
]


def _section(title: str) -> None:
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}\n")


def _run(script: str, extra_args: list[str] | None = None) -> bool:
    """Run a deployer script in a subprocess. Returns True on success."""
    cmd = [sys.executable, str(BASE / script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(BASE))
    return result.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="WealthOS — run production deployers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would run, don't execute")
    a = p.parse_args()

    start = datetime.now()
    print(f"\nWealthOS Unified Runner  —  {start.strftime('%A, %d %B %Y  %H:%M')}")
    print("Running: NiftyShop + MidcapShop (weekly)")

    if a.dry_run:
        print("\n  DRY RUN — would execute:")
        for _, _, script in STEPS:
            print(f"    python {script}")
        print("    python dashboard.py")
        return

    results: dict[str, bool] = {}
    for title, name, script in STEPS:
        _section(title)
        results[name] = _run(script)

    elapsed = (datetime.now() - start).seconds
    _section("CONSOLIDATED SUMMARY")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:<25}  {status}")
    print(f"\n  Completed in {elapsed}s  —  {datetime.now().strftime('%H:%M:%S')}")

    # Auto-regenerate dashboard
    _section("DASHBOARD")
    dash_ok = _run("dashboard.py", ["--open"])
    print(f"  Dashboard: {'dashboard.html written' if dash_ok else 'FAILED'}")

    if any(not v for v in results.values()) or not dash_ok:
        print("\n  One or more steps failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
