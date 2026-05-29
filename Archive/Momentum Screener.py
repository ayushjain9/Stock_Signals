"""
WealthOS — Momentum Signal Screener (Full Stack)
=================================================
Scores every stock 0-100 across four signal tiers:

  TREND (40 pts)        DMA alignment — direction
  STRENGTH (48 pts)     ADX + RSI + MACD — how strong is the move
  MOMENTUM (20 pts)     ROC 1M / 3M / 6M — raw price momentum
  CONFIRMATION (17 pts) 52W high, volume, BB squeeze

SIGNALS EXPLAINED
-----------------
  RSI (14)      Relative Strength Index. Measures speed of price changes.
                > 55 = bullish momentum. > 70 = overbought (momentum risk).
                < 45 = bearish momentum. < 30 = oversold.

  MACD (12,26,9) Moving Avg Convergence Divergence. MACD line > signal line
                = bullish crossover. Histogram growing = accelerating momentum.

  ADX (14)      Average Directional Index. Measures STRENGTH of trend, not
                direction. ADX > 25 = trending. > 40 = very strong trend.
                +DI > -DI = bullish trend. -DI > +DI = bearish trend.

  ROC           Rate of Change. Pure price momentum. 1M ROC > 5% means the
                stock gained 5%+ in the last month.

  52W High      Stocks within 5% of their 52-week high exhibit momentum
                continuation (empirically confirmed in Indian markets).

  Volume Surge  Volume > 1.5x the 20-day average on an up day = institutional
                accumulation. Smart money is entering.

  BB Squeeze    Bollinger Band width < 4% means volatility has compressed.
                This historically precedes explosive moves. Long the direction
                of the breakout.

USAGE
-----
  pip install yfinance pandas rich numpy
  python momentum_screener.py                      # Nifty 50
  python momentum_screener.py --symbols HDFCBANK ICICIBANK SBIN
  python momentum_screener.py --min-score 60       # only strong momentum
  python momentum_screener.py --output csv
"""
from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TATAMOTORS","ADANIENT","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","DIVISLAB","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "BRITANNIA","BPCL","HEROMOTOCO","HINDALCO","SBILIFE",
]


# ─────────────────────────────────────────────────────────────────────────────
# Indicator maths (pure numpy, no TA-lib dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed EMA."""
    k = 2 / (period + 1)
    out = np.zeros_like(arr, dtype=float)
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    delta = np.diff(closes)
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    avg_up = up[:period].mean()
    avg_dn = dn[:period].mean()
    for i in range(period, len(up)):
        avg_up = (avg_up * (period - 1) + up[i]) / period
        avg_dn = (avg_dn * (period - 1) + dn[i]) / period
    rs = avg_up / avg_dn if avg_dn > 1e-10 else 100.0
    return round(100 - 100 / (1 + rs), 2)


def calc_macd(closes: np.ndarray, fast=12, slow=26, signal=9) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = fast_ema - slow_ema
    sig_line  = _ema(macd_line[slow - 1:], signal)
    hist      = macd_line[-1] - sig_line[-1]
    return round(float(macd_line[-1]), 4), round(float(sig_line[-1]), 4), round(float(hist), 4)


def calc_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> tuple[float, float, float]:
    """Returns (ADX, +DI, -DI)."""
    n = len(closes)
    tr = np.maximum.reduce([highs[1:] - lows[1:],
                             np.abs(highs[1:] - closes[:-1]),
                             np.abs(lows[1:]  - closes[:-1])])
    pdm = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                   np.maximum(highs[1:] - highs[:-1], 0), 0.0)
    ndm = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                   np.maximum(lows[:-1] - lows[1:], 0), 0.0)

    def rma(arr):
        out = np.zeros(len(arr))
        out[period - 1] = arr[:period].sum()
        for i in range(period, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / period + arr[i]
        return out

    atr_r  = rma(tr)
    pdm_r  = rma(pdm)
    ndm_r  = rma(ndm)

    # Use last valid window
    pdi = 100 * pdm_r[-1] / atr_r[-1] if atr_r[-1] > 0 else 0
    ndi = 100 * ndm_r[-1] / atr_r[-1] if atr_r[-1] > 0 else 0

    dx_arr = 100 * np.abs(pdm_r - ndm_r) / (pdm_r + ndm_r + 1e-10)
    adx_r  = rma(dx_arr)
    return round(float(adx_r[-1]), 2), round(float(pdi), 2), round(float(ndi), 2)


def calc_bb_width(closes: np.ndarray, period: int = 20) -> float:
    p  = closes[-period:]
    mid = p.mean()
    std = p.std(ddof=1)
    return round(4 * std / mid * 100, 2)  # (upper - lower) / mid × 100


def calc_roc(closes: np.ndarray, period: int) -> float:
    if len(closes) <= period:
        return 0.0
    return round((closes[-1] / closes[-period] - 1) * 100, 2)


def calc_volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
    avg = volumes[-period - 1 : -1].mean()
    return round(float(volumes[-1] / avg), 2) if avg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

MAX_SCORE = 112  # sum of all possible points


@dataclass
class SignalSet:
    # DMA
    brutal_strength:  bool = False
    brutal_weakness:  bool = False
    golden_cross:     bool = False
    death_cross:      bool = False
    above_200:        bool = False
    above_50:         bool = False
    # Oscillators
    rsi:              float = 0.0
    macd_line:        float = 0.0
    macd_signal:      float = 0.0
    macd_hist:        float = 0.0
    adx:              float = 0.0
    pdi:              float = 0.0
    ndi:              float = 0.0
    # Momentum
    roc_1m:           float = 0.0
    roc_3m:           float = 0.0
    roc_6m:           float = 0.0
    # Confirmation
    pct_from_52h:     float = 0.0
    vol_ratio:        float = 0.0
    bb_width:         float = 0.0


@dataclass
class MomentumResult:
    symbol:       str
    price:        float
    dma25:        float
    dma50:        float
    dma200:       float
    signals:      SignalSet = field(default_factory=SignalSet)
    score:        int       = 0
    score_pct:    float     = 0.0
    fired:        list[str] = field(default_factory=list)
    tier_scores:  dict      = field(default_factory=dict)
    rating:       str       = "NO DATA"
    error:        str | None = None

    @property
    def rating_color(self) -> str:
        if self.score_pct >= 80: return "bold green"
        if self.score_pct >= 60: return "green"
        if self.score_pct >= 40: return "yellow"
        if self.score_pct >= 20: return "red"
        return "bold red"


def compute_score(s: SignalSet) -> tuple[int, float, list[str], dict]:
    fired = []
    tier = {"Trend": 0, "Strength": 0, "Momentum": 0, "Confirmation": 0}

    # ── TREND (40 pts) ──
    if s.brutal_strength:
        fired.append("Brutal Strength (+20)"); tier["Trend"] += 20
    if s.golden_cross and not s.brutal_strength:
        fired.append("Golden Cross (+10)"); tier["Trend"] += 10
    if s.above_200 and not s.brutal_strength:
        fired.append("Above 200DMA (+5)"); tier["Trend"] += 5
    if s.above_50 and not s.brutal_strength:
        fired.append("Above 50DMA (+5)"); tier["Trend"] += 5

    # ── STRENGTH (48 pts) ──
    if s.adx >= 40:
        fired.append(f"ADX {s.adx:.0f} very strong (+15)"); tier["Strength"] += 15
    elif s.adx >= 25:
        fired.append(f"ADX {s.adx:.0f} trending (+8)"); tier["Strength"] += 8

    if 55 <= s.rsi < 70:
        fired.append(f"RSI {s.rsi:.1f} bullish zone (+10)"); tier["Strength"] += 10
    elif s.rsi >= 70:
        fired.append(f"RSI {s.rsi:.1f} overbought momentum (+5)"); tier["Strength"] += 5

    if s.macd_line > s.macd_signal:
        fired.append("MACD bullish crossover (+10)"); tier["Strength"] += 10
    if s.macd_hist > 0 and s.macd_hist > abs(s.macd_signal) * 0.05:
        fired.append("MACD histogram growing (+5)"); tier["Strength"] += 5

    # ── MOMENTUM (20 pts) ──
    if s.roc_1m > 5:
        fired.append(f"ROC 1M +{s.roc_1m:.1f}% (+5)"); tier["Momentum"] += 5
    if s.roc_3m > 15:
        fired.append(f"ROC 3M +{s.roc_3m:.1f}% (+8)"); tier["Momentum"] += 8
    if s.roc_6m > 25:
        fired.append(f"ROC 6M +{s.roc_6m:.1f}% (+7)"); tier["Momentum"] += 7

    # ── CONFIRMATION (17 pts) ──
    if -5 <= s.pct_from_52h <= 0:
        fired.append(f"Near 52W high ({s.pct_from_52h:.1f}%) (+8)"); tier["Confirmation"] += 8
    if s.vol_ratio >= 1.5:
        fired.append(f"Volume surge {s.vol_ratio:.1f}x (+5)"); tier["Confirmation"] += 5
    if s.bb_width < 4:
        fired.append(f"BB Squeeze {s.bb_width:.1f}% (+4)"); tier["Confirmation"] += 4

    total = sum(tier.values())
    pct   = round(total / MAX_SCORE * 100, 1)
    return total, pct, fired, tier


def score_to_rating(pct: float) -> str:
    if pct >= 80: return "STRONG MOMENTUM"
    if pct >= 60: return "EMERGING MOMENTUM"
    if pct >= 40: return "NEUTRAL"
    if pct >= 20: return "WEAK"
    return "NO MOMENTUM"


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_score(symbols: list[str]) -> list[MomentumResult]:
    import yfinance as yf
    import pandas as pd

    tickers = [s + ".NS" for s in symbols]
    print(f"Fetching {len(tickers)} tickers (1y daily OHLCV)...")
    data = yf.download(
        tickers, period="1y", progress=False,
        auto_adjust=True, threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close  = data["Close"]
        high   = data["High"]
        low    = data["Low"]
        volume = data["Volume"]
    else:
        # single ticker
        close  = data[["Close"]];  close.columns  = [tickers[0]]
        high   = data[["High"]];   high.columns   = [tickers[0]]
        low    = data[["Low"]];    low.columns    = [tickers[0]]
        volume = data[["Volume"]]; volume.columns = [tickers[0]]

    results = []
    for sym, ticker in zip(symbols, tickers, strict=False):
        col = ticker if ticker in close.columns else sym
        if col not in close.columns or close[col].dropna().shape[0] < 200:
            results.append(MomentumResult(sym, 0, 0, 0, 0, error="Insufficient data"))
            continue

        c = close[col].dropna().values.astype(float)
        h = high[col].reindex(close[col].dropna().index).ffill().values.astype(float)
        lo = low[col].reindex(close[col].dropna().index).ffill().values.astype(float)
        v = volume[col].reindex(close[col].dropna().index).fillna(0).values.astype(float)

        price  = c[-1]
        dma25  = c[-25:].mean()
        dma50  = c[-50:].mean()
        dma200 = c[-200:].mean()

        rsi_val             = calc_rsi(c)
        macd_l, macd_s, hist = calc_macd(c)
        adx_val, pdi, ndi   = calc_adx(h, lo, c)
        bbw                 = calc_bb_width(c)
        roc1                = calc_roc(c, 22)
        roc3                = calc_roc(c, 66)
        roc6                = calc_roc(c, 132)
        vr                  = calc_volume_ratio(v)
        hi52                = c[-252:].max()
        pct52               = (price / hi52 - 1) * 100

        sig = SignalSet(
            brutal_strength = price > dma25 > dma50 > dma200,
            brutal_weakness = price < dma25 < dma50 < dma200,
            golden_cross    = dma50 > dma200,
            death_cross     = dma50 < dma200,
            above_200       = price > dma200,
            above_50        = price > dma50,
            rsi=rsi_val, macd_line=macd_l, macd_signal=macd_s, macd_hist=hist,
            adx=adx_val, pdi=pdi, ndi=ndi,
            roc_1m=roc1, roc_3m=roc3, roc_6m=roc6,
            pct_from_52h=pct52, vol_ratio=vr, bb_width=bbw,
        )

        score, pct, fired, tier_s = compute_score(sig)
        results.append(MomentumResult(
            symbol=sym, price=round(price, 2),
            dma25=round(dma25, 2), dma50=round(dma50, 2), dma200=round(dma200, 2),
            signals=sig, score=score, score_pct=pct,
            fired=fired, tier_scores=tier_s, rating=score_to_rating(pct),
        ))

    return sorted(results, key=lambda r: r.score_pct, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_rich_report(results: list[MomentumResult]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.panel import Panel
    except ImportError:
        print_plain(results); return

    console = Console()
    console.print(f"\n[bold]WealthOS — Momentum Screener[/bold]  "
                  f"[dim]{datetime.now().strftime('%d %b %Y %H:%M')}[/dim]\n")

    ratings = [
        ("STRONG MOMENTUM",    "bold green",  [r for r in results if r.rating == "STRONG MOMENTUM"]),
        ("EMERGING MOMENTUM",  "green",       [r for r in results if r.rating == "EMERGING MOMENTUM"]),
        ("NEUTRAL",            "yellow",      [r for r in results if r.rating == "NEUTRAL"]),
        ("WEAK",               "red",         [r for r in results if r.rating == "WEAK"]),
        ("NO MOMENTUM",        "bold red",    [r for r in results if r.rating == "NO MOMENTUM"]),
    ]

    for label, color, stocks in ratings:
        if not stocks:
            continue
        t = Table(title=f"[{color}]{label}[/{color}]  ({len(stocks)} stocks)",
                  box=box.SIMPLE_HEAVY, header_style="dim")
        t.add_column("Symbol",    width=13)
        t.add_column("Score",     justify="center", width=9)
        t.add_column("Price",     justify="right",  width=10)
        t.add_column("RSI",       justify="right",  width=7)
        t.add_column("ADX",       justify="right",  width=7)
        t.add_column("MACD",      justify="center", width=8)
        t.add_column("ROC 1M",    justify="right",  width=8)
        t.add_column("ROC 3M",    justify="right",  width=8)
        t.add_column("52W High",  justify="right",  width=9)
        t.add_column("Vol",       justify="right",  width=7)
        t.add_column("DMA Signal",               width=18)

        for r in stocks:
            s = r.signals
            dma_sig = ("✦ BS" if s.brutal_strength else
                       "✦ BW" if s.brutal_weakness else
                       "↑ GC" if s.golden_cross    else "↓ DC")
            macd_sym = "▲" if s.macd_line > s.macd_signal else "▼"
            vol_str  = f"{s.vol_ratio:.1f}x" if s.vol_ratio >= 1.5 else f"[dim]{s.vol_ratio:.1f}x[/dim]"
            t.add_row(
                r.symbol,
                f"[{color}]{r.score_pct:.0f}%[/{color}]",
                f"₹{r.price:.1f}",
                f"[green]{s.rsi:.0f}[/green]" if s.rsi > 55 else f"[red]{s.rsi:.0f}[/red]" if s.rsi < 45 else f"{s.rsi:.0f}",
                f"[green]{s.adx:.0f}[/green]" if s.adx > 25 else f"[dim]{s.adx:.0f}[/dim]",
                f"[green]{macd_sym}[/green]" if s.macd_line > s.macd_signal else f"[red]{macd_sym}[/red]",
                f"[green]+{s.roc_1m:.1f}%[/green]" if s.roc_1m > 0 else f"[red]{s.roc_1m:.1f}%[/red]",
                f"[green]+{s.roc_3m:.1f}%[/green]" if s.roc_3m > 0 else f"[red]{s.roc_3m:.1f}%[/red]",
                f"[green]{s.pct_from_52h:.1f}%[/green]" if s.pct_from_52h > -5 else f"[dim]{s.pct_from_52h:.1f}%[/dim]",
                vol_str,
                dma_sig,
            )
        console.print(t)

        # Print fired signals for top 3 in each section
        if label == "STRONG MOMENTUM":
            for r in stocks[:5]:
                console.print(f"  [bold]{r.symbol}[/bold]: " + " · ".join(r.fired))
            console.print()

    err = [r for r in results if r.error]
    if err:
        console.print(f"[dim]Errors: {', '.join(r.symbol for r in err)}[/dim]\n")


def print_plain(results: list[MomentumResult]) -> None:
    print(f"\nMomentum Screener  {datetime.now().strftime('%d %b %Y %H:%M')}\n")
    print(f"{'Symbol':12} {'Score':>7} {'RSI':>6} {'ADX':>6} {'MACD':>6} {'ROC1M':>7} {'ROC3M':>7} {'52W%':>7}  Rating")
    print("─" * 90)
    for r in results:
        s = r.signals
        macd = "▲" if s.macd_line > s.macd_signal else "▼"
        print(f"{r.symbol:12} {r.score_pct:>6.0f}% {s.rsi:>6.1f} {s.adx:>6.1f} {macd:>6} "
              f"{s.roc_1m:>+7.1f}% {s.roc_3m:>+7.1f}% {s.pct_from_52h:>+7.1f}%  {r.rating}")


def save_csv(results: list[MomentumResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","score_pct","score_raw","rating","price","dma25","dma50","dma200",
                    "rsi","macd_line","macd_signal","macd_hist","adx","pdi","ndi",
                    "roc_1m","roc_3m","roc_6m","pct_from_52w_high","vol_ratio","bb_width",
                    "brutal_strength","brutal_weakness","golden_cross","death_cross","signals_fired"])
        for r in results:
            s = r.signals
            w.writerow([r.symbol, r.score_pct, r.score, r.rating, r.price,
                        r.dma25, r.dma50, r.dma200,
                        s.rsi, s.macd_line, s.macd_signal, s.macd_hist,
                        s.adx, s.pdi, s.ndi,
                        s.roc_1m, s.roc_3m, s.roc_6m, s.pct_from_52h, s.vol_ratio, s.bb_width,
                        s.brutal_strength, s.brutal_weakness, s.golden_cross, s.death_cross,
                        " | ".join(r.fired)])
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WealthOS momentum screener")
    parser.add_argument("--symbols",   nargs="+")
    parser.add_argument("--watchlist", type=Path)
    parser.add_argument("--min-score", type=float, default=0, help="Only show stocks above this score %")
    parser.add_argument("--output",    choices=["table","csv","both"], default="table")
    parser.add_argument("--csv-file",  type=Path, default=Path("momentum_signals.csv"))
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.watchlist and args.watchlist.exists():
        symbols = [l.strip().upper() for l in args.watchlist.read_text().splitlines() if l.strip()]
    else:
        symbols = NIFTY50
        print(f"Running full Nifty 50 ({len(NIFTY50)} stocks).")

    results = fetch_and_score(symbols)

    if args.min_score > 0:
        results = [r for r in results if r.score_pct >= args.min_score]
        print(f"Filtered to {len(results)} stocks with score ≥ {args.min_score}%")

    if args.output in ("table", "both"):
        print_rich_report(results)
    if args.output in ("csv", "both"):
        save_csv(results, args.csv_file)


if __name__ == "__main__":
    main()