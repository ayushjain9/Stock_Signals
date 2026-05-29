"""
WealthOS — Market Regime Detector
===================================

Detects the current market regime from Nifty 50 price vs DMA50/DMA200,
and outputs dynamic capital allocation guidance between Momentum and
Mean-Reversion strategies.

VALUE SLEEVE NOTE:
  value_screener.py operates on a 3-5 year horizon and is deliberately
  excluded from this dynamic allocation. Fund it separately; never
  rebalance it based on a short-term DMA signal.

REGIME CLASSIFICATION:
  BULL      price > DMA200  AND  DMA50 > DMA200   (trend up, structure intact)
  BEAR      price < DMA200  AND  DMA50 < DMA200   (trend down, structure broken)
  RECOVERY  price > DMA50   AND  price < DMA200   (bouncing, not yet confirmed)
  SIDEWAYS  anything else   (oscillating around moving averages)

DYNAMIC ALLOCATION (Momentum vs Mean-Reversion only):
  Regime      Momentum    Mean-Reversion
  ─────────────────────────────────────
  BULL           75%           25%
  RECOVERY       50%           50%
  SIDEWAYS       35%           65%
  BEAR           20%           80%

STABILITY RULE:
  A regime is only CONFIRMED after 5+ consecutive trading days.
  Do not rebalance on a TRANSITIONING regime — wait for confirmation.
  This prevents whipsaw on brief DMA crossovers.

USAGE:
  python regime_detector.py                      # default ₹15.5L liquid sleeve
  python regime_detector.py --capital 1550000    # custom capital
  python regime_detector.py --breadth            # add Nifty50 breadth check (~15s extra)
  python regime_detector.py --history 20         # show last 20 days of regime history
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

# ── Universe (matches existing deploy scripts exactly) ────────────────────────

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "INFOSYS", "SBIN",
    "HINDUNILVR", "ITC", "LT", "KOTAKBANK", "BAJFINANCE", "HCLTECH", "AXISBANK",
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "NTPC", "POWERGRID",
    "NESTLEIND", "WIPRO", "ONGC", "JSWSTEEL", "TATAMOTORS", "ADANIENT", "COALINDIA",
    "INDUSINDBK", "BAJAJFINSV", "TATASTEEL", "TECHM", "HDFCLIFE", "DIVISLAB",
    "DRREDDY", "CIPLA", "GRASIM", "APOLLOHOSP", "ADANIPORTS", "TRENT", "BEL",
    "SHRIRAMFIN", "BAJAJ-AUTO", "EICHERMOT", "M&M", "BRITANNIA", "BPCL",
    "HEROMOTOCO", "HINDALCO", "SBILIFE",
]

# ── Regime → allocation ───────────────────────────────────────────────────────

ALLOCATION: dict[str, tuple[float, float]] = {
    "BULL":      (0.75, 0.25),
    "RECOVERY":  (0.50, 0.50),
    "SIDEWAYS":  (0.35, 0.65),
    "BEAR":      (0.20, 0.80),
}

# Days a regime must hold before it's considered confirmed
STABILITY_DAYS = 5

# Default sleeves from CLAUDE.md — used to split allocation proportionally
DEFAULT_SLEEVES = {
    "Nifty50 Momentum":  900_000,
    "Midcap Momentum":   200_000,
    "Nifty50 NiftyShop": 250_000,
    "Midcap NiftyShop":  200_000,
}
DEFAULT_CAPITAL = sum(DEFAULT_SLEEVES.values())  # ₹15.5L


# ── Regime logic ──────────────────────────────────────────────────────────────

def classify(price: float, dma50: float, dma200: float) -> str:
    if price > dma200 and dma50 > dma200:
        return "BULL"
    if price < dma200 and dma50 < dma200:
        return "BEAR"
    if price > dma50 and price < dma200:
        return "RECOVERY"
    return "SIDEWAYS"


def count_consecutive(series: pd.Series, current: str) -> int:
    """How many trailing days match the current regime."""
    count = 0
    for r in reversed(series.tolist()):
        if r == current:
            count += 1
        else:
            break
    return count


# ── Display helpers ───────────────────────────────────────────────────────────

REGIME_COLOR = {"BULL": "green", "BEAR": "red", "RECOVERY": "yellow", "SIDEWAYS": "cyan"}
REGIME_EMOJI = {"BULL": "🟢", "BEAR": "🔴", "RECOVERY": "🟡", "SIDEWAYS": "🔵"}


def rc(r: str) -> str:
    return REGIME_COLOR.get(r, "white")


def re_(r: str) -> str:
    return REGIME_EMOJI.get(r, "⚪")


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_nifty() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download("^NSEI", period="18mo", progress=False, auto_adjust=True)
    close = df["Close"].dropna()
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    result = close.to_frame(name="close")
    result["dma50"]  = result["close"].rolling(50).mean()
    result["dma200"] = result["close"].rolling(200).mean()
    result = result.dropna(subset=["dma200"])
    result["regime"] = result.apply(
        lambda r: classify(r["close"], r["dma50"], r["dma200"]), axis=1
    )
    return result


def fetch_breadth() -> tuple[int, int, float]:
    """Returns (above_200dma, total_valid, pct_above)."""
    import yfinance as yf
    tickers = [s + ".NS" for s in NIFTY50]
    raw = yf.download(tickers, period="14mo", progress=False, auto_adjust=True, threads=True)
    closes = raw["Close"]
    above, total = 0, 0
    for sym in NIFTY50:
        col = sym + ".NS"
        if col not in closes.columns:
            continue
        c = closes[col].dropna()
        if len(c) < 200:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-200:].mean()):
            above += 1
    pct = (above / total * 100) if total > 0 else 0.0
    return above, total, pct


# ── Main run ──────────────────────────────────────────────────────────────────

def run(capital: int, show_breadth: bool, history_days: int) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        f"[bold]WealthOS — Market Regime Detector[/bold]\n"
        f"[dim]{datetime.now().strftime('%A, %d %B %Y  %H:%M')}[/dim]",
        border_style="blue"
    ))
    console.print()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    console.print("  [dim]Fetching Nifty 50 data...[/dim]")
    df = fetch_nifty()

    last   = df.iloc[-1]
    price  = float(last["close"])
    dma50  = float(last["dma50"])
    dma200 = float(last["dma200"])
    regime = str(last["regime"])

    consecutive  = count_consecutive(df["regime"], regime)
    is_confirmed = consecutive >= STABILITY_DAYS

    mom_pct, mr_pct = ALLOCATION[regime]

    # ── Nifty snapshot ────────────────────────────────────────────────────────
    console.print("[bold underline]NIFTY 50 SNAPSHOT[/bold underline]")
    snap = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    snap.add_column("Label", style="dim", min_width=20)
    snap.add_column("Value", justify="right")
    snap.add_column("Context", style="dim")

    dist50  = (price / dma50  - 1) * 100
    dist200 = (price / dma200 - 1) * 100
    gap     = (dma50  / dma200 - 1) * 100

    snap.add_row("Current Price", f"[bold]₹{price:,.0f}[/bold]", "")
    snap.add_row(
        "DMA 50", f"₹{dma50:,.0f}",
        f"price {'[green]above' if dist50 > 0 else '[red]below'}[/] by {abs(dist50):.1f}%"
    )
    snap.add_row(
        "DMA 200", f"₹{dma200:,.0f}",
        f"price {'[green]above' if dist200 > 0 else '[red]below'}[/] by {abs(dist200):.1f}%"
    )
    snap.add_row(
        "DMA50 vs DMA200", f"{gap:+.1f}%",
        "[green]golden structure[/green]" if gap > 0 else "[red]death cross structure[/red]"
    )
    console.print(snap)

    # ── Regime + stability ────────────────────────────────────────────────────
    color = rc(regime)
    emoji = re_(regime)
    if is_confirmed:
        stability_text = f"[green]CONFIRMED ({consecutive} consecutive days)[/green]"
    else:
        stability_text = (
            f"[yellow]TRANSITIONING — {consecutive} day(s) only. "
            f"Need {STABILITY_DAYS} before rebalancing.[/yellow]"
        )

    console.print(f"\n[bold]REGIME:[/bold]  [{color}]{emoji}  {regime}[/{color}]   {stability_text}\n")

    checks = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    checks.add_column("Condition", style="dim", min_width=22)
    checks.add_column("Status")
    checks.add_row("price > DMA200", "[green]✓  YES[/green]" if price > dma200 else "[red]✗  NO[/red]")
    checks.add_row("DMA50 > DMA200", "[green]✓  YES[/green]" if dma50 > dma200  else "[red]✗  NO[/red]")
    checks.add_row("price > DMA50",  "[green]✓  YES[/green]" if price > dma50   else "[red]✗  NO[/red]")
    console.print(checks)

    # ── Breadth (optional) ────────────────────────────────────────────────────
    if show_breadth:
        console.print("  [dim]Fetching breadth data for all 50 stocks...[/dim]")
        above, total, pct = fetch_breadth()
        b_color = "green" if pct > 60 else ("red" if pct < 30 else "yellow")
        b_confirm = (
            "→ [green]confirms BULL[/green]" if pct > 60 else
            "→ [red]confirms BEAR[/red]"     if pct < 30 else
            "→ [yellow]mixed — no strong breadth signal[/yellow]"
        )
        console.print(
            f"\n[bold]BREADTH:[/bold]  [{b_color}]{above}/{total} stocks above DMA200 "
            f"({pct:.0f}%)[/{b_color}]  {b_confirm}"
        )

    # ── Allocation table ──────────────────────────────────────────────────────
    mom_total = capital * mom_pct
    mr_total  = capital * mr_pct

    # Split proportionally within each category using default sleeve ratios
    default_mom = DEFAULT_SLEEVES["Nifty50 Momentum"] + DEFAULT_SLEEVES["Midcap Momentum"]
    default_mr  = DEFAULT_SLEEVES["Nifty50 NiftyShop"] + DEFAULT_SLEEVES["Midcap NiftyShop"]

    n50_mom_w = DEFAULT_SLEEVES["Nifty50 Momentum"]  / default_mom
    mid_mom_w = DEFAULT_SLEEVES["Midcap Momentum"]   / default_mom
    n50_mr_w  = DEFAULT_SLEEVES["Nifty50 NiftyShop"] / default_mr
    mid_mr_w  = DEFAULT_SLEEVES["Midcap NiftyShop"]  / default_mr

    console.print(
        f"\n[bold underline]ALLOCATION GUIDANCE[/bold underline]"
        f"  [dim](Total liquid sleeve: ₹{capital/1e5:.1f}L)[/dim]\n"
    )

    alloc = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 2))
    alloc.add_column("Strategy",         style="bold", min_width=22)
    alloc.add_column("Category",         style="dim")
    alloc.add_column("Portfolio %",      justify="right")
    alloc.add_column("Target ₹",         justify="right")
    alloc.add_column("Default ₹",        justify="right", style="dim")

    def lakh(n: float) -> str:
        return f"₹{n/1e5:.2f}L"

    strategies = [
        ("Nifty50 Momentum",  "Momentum",       mom_pct * n50_mom_w, mom_total * n50_mom_w, DEFAULT_SLEEVES["Nifty50 Momentum"]),
        ("Midcap Momentum",   "Momentum",        mom_pct * mid_mom_w, mom_total * mid_mom_w, DEFAULT_SLEEVES["Midcap Momentum"]),
        ("Nifty50 NiftyShop", "Mean-Reversion",  mr_pct  * n50_mr_w,  mr_total  * n50_mr_w,  DEFAULT_SLEEVES["Nifty50 NiftyShop"]),
        ("Midcap NiftyShop",  "Mean-Reversion",  mr_pct  * mid_mr_w,  mr_total  * mid_mr_w,  DEFAULT_SLEEVES["Midcap NiftyShop"]),
    ]

    for name, cat, pct_of_total, target, default in strategies:
        alloc.add_row(name, cat, f"{pct_of_total*100:.0f}%", lakh(target), lakh(default))

    console.print(alloc)

    # ── Value sleeve note ─────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]VALUE SLEEVE — Excluded from dynamic allocation[/bold]\n\n"
        "[dim]value_screener.py scans 400+ mid/small caps for quality compounders.\n"
        "Holding horizon: 3-5 years. Review quarterly, exit when thesis breaks.\n"
        "Never exit a value position because of a DMA regime signal.[/dim]",
        border_style="dim",
        title="[dim]value_screener.py[/dim]",
    ))

    # ── Regime history ────────────────────────────────────────────────────────
    n = min(history_days, len(df))
    console.print(f"\n[bold underline]REGIME HISTORY — last {n} trading days[/bold underline]")

    hist = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    hist.add_column("Date",   min_width=12)
    hist.add_column("Price",  justify="right")
    hist.add_column("DMA50",  justify="right")
    hist.add_column("DMA200", justify="right")
    hist.add_column("Regime")

    for date, row in df.tail(n).iloc[::-1].iterrows():
        r = str(row["regime"])
        c = rc(r)
        hist.add_row(
            str(date.date()),
            f"₹{row['close']:,.0f}",
            f"₹{row['dma50']:,.0f}",
            f"₹{row['dma200']:,.0f}",
            f"[{c}]{re_(r)} {r}[/{c}]",
        )
    console.print(hist)

    # ── Action guidance ───────────────────────────────────────────────────────
    console.print()

    if not is_confirmed:
        console.print(Panel(
            f"[yellow bold]⚠  WAIT — Regime is TRANSITIONING[/yellow bold]\n\n"
            f"Current: {emoji} {regime}  ({consecutive} day(s))\n"
            f"Minimum {STABILITY_DAYS} consecutive days required before rebalancing.\n\n"
            f"Run this script again in a few days. Do not adjust allocations yet.",
            border_style="yellow",
            title="ACTION",
        ))
        console.print()
        return

    action_text = {
        "BULL": (
            "Momentum strategies are in their optimal environment.\n\n"
            "• Run nifty50_Momentum_deploy.py and midcap_momentum_deploy.py this Sunday.\n"
            "• NiftyShop is a support sleeve — maintain open positions, don't add fresh capital.\n"
            "• The 200DMA quality gate in NiftyShop already self-adjusts in bull markets."
        ),
        "BEAR": (
            "Trend-following is a losing game right now.\n\n"
            "• Do NOT force-exit momentum positions — let the exit buffer and hard stop work.\n"
            "• Redirect any fresh capital toward NiftyShop mean-reversion.\n"
            "• NiftyShop entries below 200DMA are blocked by the quality gate — that's correct.\n"
            "• This is the hardest regime to act in. Do less, not more."
        ),
        "RECOVERY": (
            "Market is bouncing but the bull structure isn't confirmed yet.\n\n"
            "• Run a balanced 50/50 split between Momentum and Mean-Reversion.\n"
            "• Don't chase momentum — wait for BULL confirmation (DMA50 crossing DMA200).\n"
            "• NiftyShop's 200DMA quality gate will naturally filter weak setups.\n"
            "• This regime can flip quickly — check again in 5-7 days."
        ),
        "SIDEWAYS": (
            "Range-bound markets are NiftyShop's home ground.\n\n"
            "• Mean-reversion is your primary alpha source right now.\n"
            "• The Momentum exit buffer protects you from constant churn — let it hold.\n"
            "• Run niftyshop_deploy.py every Friday. Avoid forcing new momentum entries.\n"
            "• Midcap NiftyShop: reminder that MDD is -11.93% — size accordingly."
        ),
    }.get(regime, "")

    console.print(Panel(
        f"[bold]{emoji}  {regime} REGIME — CONFIRMED ({consecutive} days)[/bold]\n\n"
        f"Allocation:  [bold]Momentum {mom_pct*100:.0f}%[/bold]  |  "
        f"[bold]Mean-Reversion {mr_pct*100:.0f}%[/bold]\n\n"
        f"{action_text}",
        border_style=rc(regime),
        title="ACTION GUIDANCE",
    ))
    console.print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="WealthOS Market Regime Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--capital", type=int, default=DEFAULT_CAPITAL,
        help=f"Total liquid sleeve capital in ₹ (default: {DEFAULT_CAPITAL:,} = ₹15.5L)"
    )
    p.add_argument(
        "--breadth", action="store_true",
        help="Fetch all 50 stocks and compute breadth (% above DMA200). Adds ~15s."
    )
    p.add_argument(
        "--history", type=int, default=10,
        help="Number of trading days to show in regime history (default: 10)"
    )
    args = p.parse_args()
    run(args.capital, args.breadth, args.history)


if __name__ == "__main__":
    main()
