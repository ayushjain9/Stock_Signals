"""
WealthOS — Market Regime Detector
===================================

Detects the current market regime from Nifty 50 price vs DMA50/DMA200,
and outputs dynamic capital allocation guidance across all three strategies:
Momentum, Mean-Reversion, and Value.

REGIME CLASSIFICATION:
  BULL      price > DMA200  AND  DMA50 > DMA200   (trend up, structure intact)
  BEAR      price < DMA200  AND  DMA50 < DMA200   (trend down, structure broken)
  RECOVERY  price > DMA50   AND  price < DMA200   (bouncing, not yet confirmed)
  SIDEWAYS  anything else   (oscillating around moving averages)

DYNAMIC ALLOCATION (all three categories):
  Regime      Momentum    Mean-Reversion    Value
  ────────────────────────────────────────────────
  BULL           65%            25%           10%
  RECOVERY       40%            40%           20%
  SIDEWAYS       25%            55%           20%
  BEAR           10%            20%           70%

CRITICAL RULE — VALUE ALLOCATION:
  The Value % above means: of any FRESH capital you deploy, this much goes to
  value_screener.py picks. It does NOT mean exit existing value positions.
  Value holdings exit ONLY when the fundamental thesis breaks:
    - ROE drops below 12% for 2 consecutive years
    - Debt/Equity crosses 1.0
    - PE re-rates to sector average (full value achieved)
    - Promoter pledge crosses 20%
  Never exit a value position because of a DMA regime signal.

STABILITY RULE:
  A regime is only CONFIRMED after 5+ consecutive trading days.
  Do not rebalance on a TRANSITIONING regime — wait for confirmation.
  This prevents whipsaw on brief DMA crossovers.

USAGE:
  python regime_detector.py                                   # default sleeves
  python regime_detector.py --capital 1550000                 # custom momentum+MR capital
  python regime_detector.py --value-capital 500000            # set value sleeve size
  python regime_detector.py --breadth                         # Nifty50 breadth check (~15s)
  python regime_detector.py --history 20                      # show last 20 days history
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
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

# ── Regime → allocation (Momentum %, Mean-Reversion %, Value %) ──────────────

ALLOCATION: dict[str, tuple[float, float, float]] = {
    "BULL":      (0.65, 0.25, 0.10),
    "RECOVERY":  (0.40, 0.40, 0.20),
    "SIDEWAYS":  (0.25, 0.55, 0.20),
    "BEAR":      (0.10, 0.20, 0.70),
}

# Days a regime must hold before it's considered confirmed
STABILITY_DAYS = 5

# Default sleeves from CLAUDE.md — used for intra-category proportional split
DEFAULT_SLEEVES = {
    "Nifty50 Momentum":  900_000,
    "Midcap Momentum":   200_000,
    "Nifty50 NiftyShop": 400_000,
    "Midcap NiftyShop":  200_000,
}
DEFAULT_CAPITAL       = sum(DEFAULT_SLEEVES.values())   # ₹15.5L (momentum + MR)
DEFAULT_VALUE_CAPITAL = 300_000                         # ₹3L default value sleeve


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


@dataclass
class RegimeSnapshot:
    label:            str    # BULL / RECOVERY / SIDEWAYS / BEAR
    price:            float
    dma50:            float
    dma200:           float
    consecutive_days: int    # how many days current regime has held
    is_confirmed:     bool   # True if >= STABILITY_DAYS
    price_vs_200_pct: float  # (price/dma200 - 1) * 100
    dma50_vs_200_pct: float  # (dma50/dma200 - 1) * 100
    as_of:            str    # date string


def get_regime() -> RegimeSnapshot:
    """
    Fetch NIFTY data and return current regime as a RegimeSnapshot.
    Importable by all deployers for the header banner.
    """
    df    = fetch_nifty()
    last  = df.iloc[-1]
    price = float(last["close"])
    dma50 = float(last["dma50"])
    dma200= float(last["dma200"])
    label = str(last["regime"])
    consec= count_consecutive(df["regime"], label)
    return RegimeSnapshot(
        label            = label,
        price            = price,
        dma50            = dma50,
        dma200           = dma200,
        consecutive_days = consec,
        is_confirmed     = consec >= STABILITY_DAYS,
        price_vs_200_pct = round((price / dma200 - 1) * 100, 1),
        dma50_vs_200_pct = round((dma50  / dma200 - 1) * 100, 1),
        as_of            = str(df.index[-1].date()),
    )


def print_regime_header() -> RegimeSnapshot:
    """
    Print a compact regime banner and return the snapshot.
    Call once near the top of each deployer's main().
    Trades proceed regardless — this is informational only.
    """
    r = get_regime()
    stability = (
        f"CONFIRMED ({r.consecutive_days} days)"
        if r.is_confirmed
        else f"TRANSITIONING ({r.consecutive_days} days — wait for {STABILITY_DAYS} before rebalancing)"
    )
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  MARKET REGIME: {r.label}  |  {stability}")
    print(f"  NIFTY ₹{r.price:,.0f}  |  vs 200DMA: {r.price_vs_200_pct:+.1f}%"
          f"  |  50-200DMA gap: {r.dma50_vs_200_pct:+.1f}%")
    print(f"  [Informational — trades proceed as backtested. Run regime_detector.py for full guidance]")
    print(f"{sep}\n")
    return r


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

def run(capital: int, value_capital: int, show_breadth: bool, history_days: int) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    total_capital = capital + value_capital

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

    mom_pct, mr_pct, val_pct = ALLOCATION[regime]

    # ── Nifty snapshot ────────────────────────────────────────────────────────
    console.print("[bold underline]NIFTY 50 SNAPSHOT[/bold underline]")
    snap = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    snap.add_column("Label",   style="dim", min_width=20)
    snap.add_column("Value",   justify="right")
    snap.add_column("Context", style="dim")

    dist50  = (price / dma50  - 1) * 100
    dist200 = (price / dma200 - 1) * 100
    gap     = (dma50  / dma200 - 1) * 100

    snap.add_row("Current Price",    f"[bold]₹{price:,.0f}[/bold]", "")
    snap.add_row(
        "DMA 50",  f"₹{dma50:,.0f}",
        f"price {'[green]above' if dist50  > 0 else '[red]below'}[/] by {abs(dist50):.1f}%"
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
    mom_total = capital       * mom_pct
    mr_total  = capital       * mr_pct
    val_total = value_capital * val_pct    # value % applied to its own sleeve

    # Intra-category splits using default sleeve proportions
    default_mom = DEFAULT_SLEEVES["Nifty50 Momentum"] + DEFAULT_SLEEVES["Midcap Momentum"]
    default_mr  = DEFAULT_SLEEVES["Nifty50 NiftyShop"] + DEFAULT_SLEEVES["Midcap NiftyShop"]

    n50_mom_w = DEFAULT_SLEEVES["Nifty50 Momentum"]  / default_mom
    mid_mom_w = DEFAULT_SLEEVES["Midcap Momentum"]   / default_mom
    n50_mr_w  = DEFAULT_SLEEVES["Nifty50 NiftyShop"] / default_mr
    mid_mr_w  = DEFAULT_SLEEVES["Midcap NiftyShop"]  / default_mr

    console.print(
        f"\n[bold underline]ALLOCATION GUIDANCE[/bold underline]"
        f"  [dim](Momentum+MR sleeve: ₹{capital/1e5:.1f}L  |  "
        f"Value sleeve: ₹{value_capital/1e5:.1f}L  |  "
        f"Total: ₹{total_capital/1e5:.1f}L)[/dim]\n"
    )

    def lakh(n: float) -> str:
        return f"₹{n/1e5:.2f}L"

    # ── Category summary ──────────────────────────────────────────────────────
    cat_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 2),
                    title="[dim]Category Summary[/dim]")
    cat_tbl.add_column("Category",        style="bold", min_width=18)
    cat_tbl.add_column("Allocation %",    justify="right")
    cat_tbl.add_column("Target ₹",        justify="right")
    cat_tbl.add_column("Applied to",      style="dim")

    cat_tbl.add_row(
        "Momentum",       f"{mom_pct*100:.0f}%",
        lakh(mom_total),  "Momentum sleeve"
    )
    cat_tbl.add_row(
        "Mean-Reversion", f"{mr_pct*100:.0f}%",
        lakh(mr_total),   "Momentum sleeve"
    )
    cat_tbl.add_row(
        "[yellow]Value[/yellow]",
        f"[yellow]{val_pct*100:.0f}%[/yellow]",
        f"[yellow]{lakh(val_total)}[/yellow]",
        "[yellow]Value sleeve (fresh capital only)[/yellow]"
    )
    console.print(cat_tbl)

    # ── Strategy detail ───────────────────────────────────────────────────────
    strat_tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 2),
                      title="[dim]Strategy Detail[/dim]")
    strat_tbl.add_column("Script",           style="bold",  min_width=28)
    strat_tbl.add_column("Category",         style="dim")
    strat_tbl.add_column("Portfolio %",      justify="right")
    strat_tbl.add_column("Target ₹",         justify="right")
    strat_tbl.add_column("Current Default",  justify="right", style="dim")

    strategies = [
        ("nifty50_Momentum_deploy.py",  "Momentum",       mom_pct * n50_mom_w, mom_total * n50_mom_w, DEFAULT_SLEEVES["Nifty50 Momentum"]),
        ("midcap_momentum_deploy.py",   "Momentum",        mom_pct * mid_mom_w, mom_total * mid_mom_w, DEFAULT_SLEEVES["Midcap Momentum"]),
        ("niftyshop_deploy.py",         "Mean-Reversion",  mr_pct  * n50_mr_w,  mr_total  * n50_mr_w,  DEFAULT_SLEEVES["Nifty50 NiftyShop"]),
        ("midcap_niftyshop_deploy.py",  "Mean-Reversion",  mr_pct  * mid_mr_w,  mr_total  * mid_mr_w,  DEFAULT_SLEEVES["Midcap NiftyShop"]),
    ]

    for name, cat, pct_of_total, target, default in strategies:
        strat_tbl.add_row(name, cat, f"{pct_of_total*100:.0f}%", lakh(target), lakh(default))

    # Value row — single script
    strat_tbl.add_row(
        "[yellow]value_screener.py[/yellow]",
        "[yellow]Value[/yellow]",
        f"[yellow]{val_pct*100:.0f}%[/yellow]",
        f"[yellow]{lakh(val_total)}[/yellow]",
        f"[dim]{lakh(value_capital)} (sleeve)[/dim]"
    )
    console.print(strat_tbl)

    # ── Value allocation rule (always visible) ────────────────────────────────
    console.print(
        "\n[dim]⚠  Value %  =  fresh capital deployment guide only.[/dim]\n"
        "[dim]   Exit value positions on thesis break (ROE, D/E, PE re-rate) — NEVER on a regime signal.[/dim]\n"
    )

    # ── Regime history ────────────────────────────────────────────────────────
    n = min(history_days, len(df))
    console.print(f"[bold underline]REGIME HISTORY — last {n} trading days[/bold underline]")

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
            "[bold]Momentum[/bold] — optimal environment. Run both deploy scripts this Sunday.\n"
            "[bold]Mean-Reversion[/bold] — support sleeve only. Maintain open positions; don't add fresh capital.\n"
            "[bold]Value[/bold] — minimal new buying (10%). Quality stocks are fairly priced in bull markets;\n"
            "  the screener will naturally surface fewer undervalued picks. Hold what you have."
        ),
        "BEAR": (
            "[bold]Momentum[/bold] — do NOT force-exit. Let the exit buffer and hard stop work.\n"
            "  Redirecting fresh momentum capital elsewhere is correct; panic-selling is not.\n"
            "[bold]Mean-Reversion[/bold] — NiftyShop quality gate (200DMA) blocks entries automatically.\n"
            "  Don't override it. Mean-reversion needs a floor to work from.\n"
            "[bold]Value[/bold] — this is your primary deployment target (70% of fresh capital).\n"
            "  Run value_screener.py now. Quality mid/small caps at distressed prices is\n"
            "  exactly what value investing is designed for. Look for HIGH CONVICTION picks.\n"
            "  Bear markets are when long-term wealth is built — but only in quality names."
        ),
        "RECOVERY": (
            "[bold]Momentum[/bold] — balanced deployment (40%). Don't chase; wait for BULL confirmation\n"
            "  (DMA50 crossing above DMA200) before increasing exposure.\n"
            "[bold]Mean-Reversion[/bold] — balanced (40%). NiftyShop's quality gate naturally filters weak setups.\n"
            "[bold]Value[/bold] — 20% fresh capital. Some quality names still trade at bear-phase discounts.\n"
            "  Run value_screener.py and look for stocks that held up fundamentally\n"
            "  during the downturn — those are the survivors worth accumulating."
        ),
        "SIDEWAYS": (
            "[bold]Momentum[/bold] — reduce new deployment (25%). Exit buffer protects existing positions\n"
            "  from constant churn — let it hold; don't force new entries.\n"
            "[bold]Mean-Reversion[/bold] — primary alpha source now (55%). NiftyShop thrives in range-bound\n"
            "  markets. Run niftyshop_deploy.py every Friday.\n"
            "  Midcap NiftyShop: MDD is -11.93% — size accordingly.\n"
            "[bold]Value[/bold] — 20% fresh capital. Sideways markets leave laggard sectors undervalued.\n"
            "  Screener can surface sector-specific opportunities even without a broad bear."
        ),
    }.get(regime, "")

    console.print(Panel(
        f"[bold]{emoji}  {regime} REGIME — CONFIRMED ({consecutive} days)[/bold]\n\n"
        f"[bold]Momentum {mom_pct*100:.0f}%[/bold]  |  "
        f"[bold]Mean-Reversion {mr_pct*100:.0f}%[/bold]  |  "
        f"[bold yellow]Value {val_pct*100:.0f}%[/bold yellow]\n\n"
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
        help=f"Momentum + MR sleeve capital in ₹ (default: {DEFAULT_CAPITAL:,} = ₹15.5L)"
    )
    p.add_argument(
        "--value-capital", type=int, default=DEFAULT_VALUE_CAPITAL,
        help=f"Value sleeve capital in ₹ (default: {DEFAULT_VALUE_CAPITAL:,} = ₹3L)"
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
    run(args.capital, args.value_capital, args.breadth, args.history)


if __name__ == "__main__":
    main()
