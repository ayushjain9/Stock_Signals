"""
WealthOS — Unified Runner
==========================
Runs production deployers in sequence and prints a consolidated summary.

Weekly workflow (Friday 3:15pm):
    python run_all.py --reversion

Monthly workflow (rebalance Sunday):
    python run_all.py --momentum
    python run_all.py --momentum --save   # also update holdings files

Full review (all four):
    python run_all.py

Dry-run (print plan, don't execute):
    python run_all.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent


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
    p.add_argument("--momentum",  action="store_true", help="Run momentum deployers only (monthly)")
    p.add_argument("--reversion", action="store_true", help="Run NiftyShop deployers only (weekly)")
    p.add_argument("--save",      action="store_true", help="Pass --save to momentum deployers (updates holdings files)")
    p.add_argument("--dry-run",   action="store_true", help="Print what would run, don't execute")
    a = p.parse_args()

    run_mom = not a.reversion   # default: run everything
    run_rev = not a.momentum
    if a.momentum:
        run_mom, run_rev = True, False
    if a.reversion:
        run_mom, run_rev = False, True

    start = datetime.now()
    print(f"\nWealthOS Unified Runner  —  {start.strftime('%A, %d %B %Y  %H:%M')}")
    what = []
    if run_mom: what.append("Momentum (monthly)")
    if run_rev: what.append("NiftyShop (weekly)")
    print(f"Running: {' + '.join(what)}")

    if a.dry_run:
        print("\n  DRY RUN — would execute:")
        if run_mom:
            save_flag = " --save" if a.save else ""
            print(f"    python nifty50_Momentum_deploy.py{save_flag}")
            print(f"    python midcap_momentum_deploy.py{save_flag}")
        if run_rev:
            print(f"    python niftyshop_deploy.py")
            print(f"    python midcap_niftyshop_deploy.py")
        return

    results: dict[str, bool] = {}
    momentum_args = ["--save"] if a.save else []

    if run_mom:
        _section("1/2  NIFTY 50 MOMENTUM  (monthly rebalance)")
        results["Nifty50 Momentum"] = _run("nifty50_Momentum_deploy.py", momentum_args)

        _section("2/2  MIDCAP 50 MOMENTUM  (monthly rebalance)")
        results["Midcap Momentum"]  = _run("midcap_momentum_deploy.py", momentum_args)

    if run_rev:
        label_n = "1/2" if not run_mom else "1/2"
        _section(f"{label_n}  NIFTYSHOP  (weekly, Nifty 50)")
        results["NiftyShop"]  = _run("niftyshop_deploy.py")

        _section(f"2/2  MIDCAPSHOP  (weekly, Midcap 50)")
        results["MidcapShop"] = _run("midcap_niftyshop_deploy.py")

    elapsed = (datetime.now() - start).seconds
    _section("CONSOLIDATED SUMMARY")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:<25}  {status}")
    print(f"\n  Completed in {elapsed}s  —  {datetime.now().strftime('%H:%M:%S')}")

    if any(not v for v in results.values()):
        print("\n  One or more deployers failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
