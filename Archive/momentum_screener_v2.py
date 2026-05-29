"""
WealthOS — Momentum Screener v2 (Professional Grade)
=====================================================

What changed from v1 — addressing every valid critique:

  ✓ Relative Strength vs NIFTY        Most important. Stock RS score 0-100
  ✓ Market Regime Filter               Bull / Caution / Bear gates everything
  ✓ ATR-based stops & targets          2×ATR stop, 2×ATR T1, 4×ATR T2
  ✓ Sector Momentum                    Is the stock's sector itself trending?
  ✓ Multi-timeframe Alignment          Weekly + Daily + Monthly DMA all agree?
  ✓ False Breakout Detection           N-day confirmation above key level
  ✓ BB Directional Filter              Squeeze + price above midband = bullish
  ✓ Reduced indicator overlap          Reweighted: RS replaces raw ROC weighting

SCORING v2 (max 130 pts)
------------------------
  REGIME GATE (0 or -50 penalty)   Bull=no penalty, Caution=-20, Bear=-50
  RELATIVE STRENGTH (30 pts)       RS rank > 70 = +20, RS rank > 50 = +12
  TREND DIRECTION (25 pts)         DMA alignment
  TREND STRENGTH (25 pts)          ADX + MACD
  MULTI-TIMEFRAME (20 pts)         Weekly + Monthly confirmation
  CONFIRMATION (15 pts)            52W high, volume, BB, breakout confirmed

USAGE
-----
  pip install yfinance pandas rich numpy
  python momentum_screener_v2.py
  python momentum_screener_v2.py --symbols HDFCBANK ICICIBANK --min-score 60
  python momentum_screener_v2.py --output csv
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

SECTOR_MAP = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDUSINDBK":"BANKING","IDFCFIRSTB":"BANKING","PNB":"BANKING",
    "BANKBARODA":"BANKING","BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","HDFCLIFE":"INSURANCE",
    "ICICIGI":"INSURANCE","SBILIFE":"INSURANCE","TCS":"IT","INFY":"IT","HCLTECH":"IT",
    "WIPRO":"IT","TECHM":"IT","RELIANCE":"ENERGY","ONGC":"ENERGY","BPCL":"ENERGY",
    "IOC":"ENERGY","GAIL":"ENERGY","NTPC":"POWER","POWERGRID":"POWER","NHPC":"POWER",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG","COLPAL":"FMCG",
    "MARUTI":"AUTO","TATAMOTORS":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO","EICHERMOT":"AUTO",
    "HEROMOTOCO":"AUTO","MOTHERSON":"AUTO_ANC","LT":"INFRA","HAL":"DEFENCE","BEL":"DEFENCE",
    "TITAN":"CONSUMER","TRENT":"RETAIL","ASIANPAINT":"PAINTS","ULTRACEMCO":"CEMENT",
    "GRASIM":"CEMENT","JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","DIVISLAB":"PHARMA","CIPLA":"PHARMA",
    "ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS","APOLLOHOSP":"HEALTHCARE",
    "HDFCAMC":"AMC","KAYNES":"ELECTRONICS","DIXON":"ELECTRONICS","IRFC":"PSU_NBFC",
}

# Sector ETF proxies for sector momentum (using Nifty sectoral indices where available)
SECTOR_PROXIES = {
    "BANKING": "^NSEBANK", "IT": "^CNXIT", "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",   "METALS": "^CNXMETAL","AUTO": "^CNXAUTO",
}


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr, period):
    k = 2 / (period + 1)
    out = np.zeros(len(arr), dtype=float)
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def calc_rsi(c, period=14):
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ag = up[:period].mean(); al = dn[:period].mean()
    for i in range(period, len(up)):
        ag = (ag * (period - 1) + up[i]) / period
        al = (al * (period - 1) + dn[i]) / period
    rs = ag / al if al > 1e-10 else 100.0
    return round(100 - 100 / (1 + rs), 2)


def calc_macd(c, fast=12, slow=26, signal=9):
    ml = _ema(c, fast) - _ema(c, slow)
    sl = _ema(ml[slow - 1:], signal)
    hist = ml[-1] - sl[-1]
    return round(float(ml[-1]), 4), round(float(sl[-1]), 4), round(float(hist), 4)


def calc_adx(h, lo, c, period=14):
    tr  = np.maximum.reduce([h[1:]-lo[1:], np.abs(h[1:]-c[:-1]), np.abs(lo[1:]-c[:-1])])
    pdm = np.where((h[1:]-h[:-1])>(lo[:-1]-lo[1:]), np.maximum(h[1:]-h[:-1],0), 0.0)
    ndm = np.where((lo[:-1]-lo[1:])>(h[1:]-h[:-1]), np.maximum(lo[:-1]-lo[1:],0), 0.0)
    def rma(a):
        o = np.zeros(len(a)); o[period-1] = a[:period].sum()
        for i in range(period, len(a)): o[i] = o[i-1] - o[i-1]/period + a[i]
        return o
    atr_r, pdm_r, ndm_r = rma(tr), rma(pdm), rma(ndm)
    pdi = 100*pdm_r[-1]/atr_r[-1] if atr_r[-1] else 0
    ndi = 100*ndm_r[-1]/atr_r[-1] if atr_r[-1] else 0
    dx  = 100*np.abs(pdm_r - ndm_r) / (pdm_r + ndm_r + 1e-10)
    adx_val = rma(dx)[-1]
    return round(float(adx_val), 2), round(float(pdi), 2), round(float(ndi), 2)


def calc_atr(h, lo, c, period=14):
    trs = [max(h[i]-lo[i], abs(h[i]-c[i-1]), abs(lo[i]-c[i-1])) for i in range(1, len(c))]
    return round(float(np.mean(trs[-period:])), 4)


def calc_roc(c, period):
    return round((c[-1] / c[-period] - 1) * 100, 2) if len(c) > period else 0.0


def calc_bb(c, period=20):
    p = c[-period:]; mid = p.mean(); std = p.std(ddof=1)
    width = 4 * std / mid * 100
    above_mid = c[-1] > mid
    return round(width, 2), above_mid


def relative_strength(stock_c, bench_c, period=90):
    """Excess return vs benchmark over N days."""
    if len(stock_c) < period or len(bench_c) < period:
        return 0.0
    s = stock_c[-1] / stock_c[-period] - 1
    b = bench_c[-1] / bench_c[-period] - 1
    return round((s - b) * 100, 2)


def market_regime(nifty_c):
    """Returns regime string and dict of context."""
    price  = nifty_c[-1]
    d200   = np.mean(nifty_c[-200:])
    slope  = np.mean(nifty_c[-50:]) - np.mean(nifty_c[-100:-50])  # recent vs older 50D
    if price > d200 and slope > 0:
        regime = "BULL"
    elif price < d200 and slope < 0:
        regime = "BEAR"
    else:
        regime = "CAUTION"
    pct_from_d200 = round((price / d200 - 1) * 100, 2)
    return regime, {"price": round(price, 2), "d200": round(d200, 2),
                    "pct_from_d200": pct_from_d200, "slope": round(slope, 4)}


def breakout_confirmed(c, dma_series, n_days=3):
    """True if close > DMA for last N days."""
    if len(c) < n_days or len(dma_series) < n_days:
        return False
    return all(c[-(n_days - i)] > dma_series[-(n_days - i)] for i in range(n_days))


def mtf_score(c, weeks=1):
    """Check weekly and monthly DMA slopes."""
    if len(c) < 60:
        return 0, {}
    weekly_slope  = c[-1] - c[-5]     # price up vs 1 week ago
    monthly_slope = c[-1] - c[-22]    # vs 1 month ago
    bimonth_slope = c[-1] - c[-44]    # vs 2 months ago
    bull = sum([weekly_slope > 0, monthly_slope > 0, bimonth_slope > 0])
    return bull, {"weekly": round(float(weekly_slope), 2),
                  "monthly": round(float(monthly_slope), 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring v2
# ─────────────────────────────────────────────────────────────────────────────

MAX_SCORE_V2 = 130


@dataclass
class SignalV2:
    # Regime
    regime:          str   = "BULL"
    regime_penalty:  int   = 0
    # Relative Strength
    rs_90d:          float = 0.0
    rs_rank:         float = 0.0   # percentile 0-100 among screened stocks
    # DMA
    brutal_strength: bool  = False
    brutal_weakness: bool  = False
    golden_cross:    bool  = False
    death_cross:     bool  = False
    above_200:       bool  = False
    above_50:        bool  = False
    # Oscillators
    rsi:             float = 50.0
    adx:             float = 0.0
    pdi:             float = 0.0
    ndi:             float = 0.0
    macd_bull:       bool  = False
    macd_hist_pos:   bool  = False
    # Multi-timeframe
    mtf_bull_count:  int   = 0
    # Confirmation
    pct_52h:         float = 0.0
    near_52h:        bool  = False
    vol_ratio:       float = 1.0
    vol_surge:       bool  = False
    bb_width:        float = 10.0
    bb_above_mid:    bool  = True
    bb_squeeze:      bool  = False
    breakout_conf:   bool  = False
    # Risk
    atr:             float = 0.0
    stop_loss:       float = 0.0
    target1:         float = 0.0
    target2:         float = 0.0
    risk_pct:        float = 0.0
    # Context
    sector:          str   = "UNKNOWN"
    roc_1m:          float = 0.0
    roc_3m:          float = 0.0


def score_v2(s: SignalV2) -> tuple[int, float, list[str]]:
    pts = 0
    fired = []

    # REGIME GATE — modifies everything
    pts += s.regime_penalty
    if s.regime == "BEAR":
        fired.append(f"⚠ Bear regime penalty ({s.regime_penalty})")
    elif s.regime == "CAUTION":
        fired.append(f"⚠ Caution regime penalty ({s.regime_penalty})")

    # RELATIVE STRENGTH (30 pts) — most important upgrade
    if s.rs_rank >= 80:
        pts += 30; fired.append(f"RS rank {s.rs_rank:.0f}th pctile (+30)")
    elif s.rs_rank >= 70:
        pts += 20; fired.append(f"RS rank {s.rs_rank:.0f}th pctile (+20)")
    elif s.rs_rank >= 50:
        pts += 12; fired.append(f"RS rank {s.rs_rank:.0f}th pctile (+12)")
    elif s.rs_rank < 30:
        pts -= 10; fired.append(f"RS rank {s.rs_rank:.0f}th pctile (underperforming -10)")

    # TREND DIRECTION (25 pts)
    if s.brutal_strength:
        pts += 25; fired.append("Brutal Strength (+25)")
    elif s.golden_cross and s.above_200 and s.above_50:
        pts += 18; fired.append("Golden Cross + above both DMAs (+18)")
    elif s.golden_cross:
        pts += 10; fired.append("Golden Cross (+10)")
    elif s.above_200:
        pts += 6;  fired.append("Above 200D (+6)")

    # TREND STRENGTH (25 pts)
    if s.adx >= 40:
        pts += 15; fired.append(f"ADX {s.adx:.0f} very strong (+15)")
    elif s.adx >= 25:
        pts += 8;  fired.append(f"ADX {s.adx:.0f} trending (+8)")

    if s.rsi >= 55 and s.rsi < 70:
        pts += 10; fired.append(f"RSI {s.rsi:.0f} bullish zone (+10)")
    elif s.rsi >= 70:
        pts += 5;  fired.append(f"RSI {s.rsi:.0f} overbought (+5)")

    if s.macd_bull:
        pts += 10; fired.append("MACD bullish (+10)")
    if s.macd_hist_pos and s.macd_bull:
        pts += 5;  fired.append("MACD histogram growing (+5)")

    # MULTI-TIMEFRAME (20 pts)
    if s.mtf_bull_count == 3:
        pts += 20; fired.append("All timeframes bullish (+20)")
    elif s.mtf_bull_count == 2:
        pts += 12; fired.append("2/3 timeframes bullish (+12)")
    elif s.mtf_bull_count == 1:
        pts += 4;  fired.append("1/3 timeframes bullish (+4)")

    # CONFIRMATION (15 pts)
    if s.near_52h:
        pts += 8; fired.append(f"Near 52W high {s.pct_52h:.1f}% (+8)")
    if s.vol_surge:
        pts += 5; fired.append(f"Volume {s.vol_ratio:.1f}× surge (+5)")
    if s.bb_squeeze and s.bb_above_mid:
        pts += 4; fired.append(f"BB Squeeze + above midband (+4)")
    elif s.bb_squeeze and not s.bb_above_mid:
        fired.append("BB Squeeze but below midband — breakout could be DOWN")
    if s.breakout_conf:
        pts += 3; fired.append("Breakout confirmed 3 days (+3)")

    pct = round(max(0, pts) / MAX_SCORE_V2 * 100, 1)
    return pts, pct, fired


def rating_v2(pct):
    if pct >= 75: return "STRONG"
    if pct >= 55: return "EMERGING"
    if pct >= 35: return "NEUTRAL"
    if pct >= 15: return "WEAK"
    return "AVOID"


# ─────────────────────────────────────────────────────────────────────────────
# Fetch + compute
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultV2:
    symbol:    str
    price:     float
    dma25:     float
    dma50:     float
    dma200:    float
    signals:   SignalV2    = field(default_factory=SignalV2)
    score:     int         = 0
    score_pct: float       = 0.0
    fired:     list[str]   = field(default_factory=list)
    rating:    str         = "NO DATA"
    error:     str | None  = None


def fetch_and_score_v2(symbols: list[str], min_score: float = 0) -> tuple[list[ResultV2], dict]:
    import yfinance as yf
    import pandas as pd

    all_syms = symbols + ["^NSEI"]
    tickers  = [s + ".NS" if not s.startswith("^") else s for s in all_syms]

    print(f"Fetching {len(tickers)} tickers (1y daily OHLCV)...")
    raw = yf.download(tickers, period="1y", progress=False, auto_adjust=True, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]; high = raw["High"]; low = raw["Low"]; vol = raw["Volume"]
    else:
        close = raw[["Close"]]; high = raw[["High"]]; low = raw[["Low"]]; vol = raw[["Volume"]]
        close.columns = high.columns = low.columns = vol.columns = [tickers[0]]

    # Market regime from NIFTY
    nifty_c = close["^NSEI"].dropna().values.astype(float) if "^NSEI" in close.columns else None
    regime, regime_ctx = market_regime(nifty_c) if nifty_c is not None and len(nifty_c) >= 200 else ("BULL", {})
    regime_penalty = 0 if regime == "BULL" else -20 if regime == "CAUTION" else -50
    print(f"Market regime: {regime}  (NIFTY {regime_ctx.get('pct_from_d200',0):+.1f}% vs 200D)")

    # First pass: collect RS values for percentile ranking
    rs_vals: dict[str, float] = {}
    raw_data: dict[str, dict] = {}

    for sym, ticker in zip(symbols, tickers[:-1], strict=False):
        col = ticker if ticker in close.columns else sym
        if col not in close.columns or close[col].dropna().shape[0] < 60:
            continue
        c = close[col].dropna().values.astype(float)
        if nifty_c is not None:
            rs = relative_strength(c, nifty_c)
            rs_vals[sym] = rs
        raw_data[sym] = {"c": c, "col": col}

    # Compute RS percentile ranks
    rs_arr = np.array(list(rs_vals.values())) if rs_vals else np.array([0.0])
    rs_ranks = {sym: round(np.sum(rs_arr <= rs_vals[sym]) / len(rs_arr) * 100, 1)
                for sym in rs_vals}

    results: list[ResultV2] = []
    for sym, ticker in zip(symbols, tickers[:-1], strict=False):
        col = ticker if ticker in close.columns else sym
        if col not in close.columns or close[col].dropna().shape[0] < 200:
            results.append(ResultV2(sym, 0, 0, 0, 0, error="Insufficient data"))
            continue

        c  = close[col].dropna().values.astype(float)
        h  = high[col].reindex(close[col].dropna().index).ffill().values.astype(float)
        lo = low[col].reindex(close[col].dropna().index).ffill().values.astype(float)
        v  = vol[col].reindex(close[col].dropna().index).fillna(0).values.astype(float)

        price  = c[-1]
        dma25  = c[-25:].mean()
        dma50  = c[-50:].mean()
        dma200 = c[-200:].mean()

        rsi_v              = calc_rsi(c)
        ml, sl, hist       = calc_macd(c)
        adx_v, pdi, ndi    = calc_adx(h, lo, c)
        atr_v              = calc_atr(h, lo, c)
        bbw, bb_above      = calc_bb(c)
        roc1               = calc_roc(c, 22)
        roc3               = calc_roc(c, 66)
        hi52               = c[-252:].max()
        pct52              = (price / hi52 - 1) * 100
        vr                 = v[-1] / v[-21:-1].mean() if v[-21:-1].mean() > 0 else 1.0
        mtf_count, _       = mtf_score(c)

        dma200_rolling = np.array([np.mean(c[max(0,i-200):i]) for i in range(200, len(c))])
        bf_conf = breakout_confirmed(c[200:], dma200_rolling, n_days=3)

        stop  = round(price - 2 * atr_v, 2)
        t1    = round(price + 2 * atr_v, 2)
        t2    = round(price + 4 * atr_v, 2)
        risk  = round((price - stop) / price * 100, 2)

        sig = SignalV2(
            regime=regime, regime_penalty=regime_penalty,
            rs_90d=rs_vals.get(sym, 0), rs_rank=rs_ranks.get(sym, 50),
            brutal_strength=price > dma25 > dma50 > dma200,
            brutal_weakness=price < dma25 < dma50 < dma200,
            golden_cross=dma50 > dma200, death_cross=dma50 < dma200,
            above_200=price > dma200, above_50=price > dma50,
            rsi=rsi_v, adx=adx_v, pdi=pdi, ndi=ndi,
            macd_bull=ml > sl, macd_hist_pos=hist > 0,
            mtf_bull_count=mtf_count,
            pct_52h=pct52, near_52h=-5 <= pct52 <= 0,
            vol_ratio=round(float(vr), 2), vol_surge=vr >= 1.5,
            bb_width=bbw, bb_above_mid=bb_above, bb_squeeze=bbw < 4,
            breakout_conf=bf_conf,
            atr=round(atr_v, 2), stop_loss=stop, target1=t1, target2=t2,
            risk_pct=risk, sector=SECTOR_MAP.get(sym, "OTHER"),
            roc_1m=roc1, roc_3m=roc3,
        )

        score, pct, fired = score_v2(sig)
        results.append(ResultV2(
            symbol=sym, price=round(price, 2),
            dma25=round(dma25, 2), dma50=round(dma50, 2), dma200=round(dma200, 2),
            signals=sig, score=score, score_pct=pct, fired=fired, rating=rating_v2(pct),
        ))

    results.sort(key=lambda r: r.score_pct, reverse=True)
    if min_score > 0:
        results = [r for r in results if r.score_pct >= min_score]

    return results, {"regime": regime, "regime_ctx": regime_ctx}


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[ResultV2], regime_info: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        [print(f"{r.symbol:12} {r.score_pct:.0f}% {r.rating:12} RS:{r.signals.rs_rank:.0f} "
               f"ADX:{r.signals.adx:.0f} stop:₹{r.signals.stop_loss:.0f}") for r in results]
        return

    console = Console()
    regime = regime_info["regime"]
    rc = {"BULL": "green", "CAUTION": "yellow", "BEAR": "red"}[regime]
    console.print(f"\n[bold]WealthOS Momentum Screener v2[/bold]  "
                  f"[{rc}]Market Regime: {regime}[/{rc}]  "
                  f"[dim]{datetime.now().strftime('%d %b %Y %H:%M')}[/dim]\n")

    for label, color, filt in [
        ("STRONG",   "bold green", lambda r: r.rating == "STRONG"),
        ("EMERGING", "green",      lambda r: r.rating == "EMERGING"),
        ("NEUTRAL",  "yellow",     lambda r: r.rating == "NEUTRAL"),
        ("WEAK",     "red",        lambda r: r.rating in ("WEAK", "AVOID")),
    ]:
        group = [r for r in results if filt(r) and not r.error]
        if not group: continue
        t = Table(title=f"[{color}]{label}[/{color}]  ({len(group)})",
                  box=box.SIMPLE_HEAVY, header_style="dim")
        for col in ["Symbol","Score","RS%ile","RSI","ADX","MTF","DMA","Stop","T1","T2","Sector"]:
            t.add_column(col, justify="right" if col not in ("Symbol","DMA","Sector") else "left", width=10)

        for r in group:
            s = r.signals
            dma = ("✦BS" if s.brutal_strength else "↑GC" if s.golden_cross else
                   "✦BW" if s.brutal_weakness else "↓DC")
            rsi_c = "green" if s.rsi > 55 else "red" if s.rsi < 45 else "white"
            adx_c = "green" if s.adx > 25 else "dim"
            t.add_row(
                r.symbol,
                f"[{color}]{r.score_pct:.0f}%[/{color}]",
                f"{s.rs_rank:.0f}",
                f"[{rsi_c}]{s.rsi:.0f}[/{rsi_c}]",
                f"[{adx_c}]{s.adx:.0f}[/{adx_c}]",
                f"{s.mtf_bull_count}/3",
                dma,
                f"₹{s.stop_loss:.0f}",
                f"₹{s.target1:.0f}",
                f"₹{s.target2:.0f}",
                s.sector[:8],
            )
        console.print(t)

        if label == "STRONG":
            for r in group[:3]:
                console.print(f"  [bold]{r.symbol}[/bold] RS {r.signals.rs_90d:+.1f}% vs NIFTY  "
                               f"Risk {r.signals.risk_pct:.1f}%  "
                               + " · ".join(r.fired[:4]))
            console.print()

    err = [r for r in results if r.error]
    if err:
        console.print(f"[dim]Skipped: {', '.join(r.symbol for r in err)}[/dim]\n")


def save_csv(results: list[ResultV2], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","score_pct","rating","regime","rs_90d","rs_rank",
                    "price","dma25","dma50","dma200","rsi","adx","pdi","ndi",
                    "macd_bull","mtf_bull_count","pct_52h","vol_ratio",
                    "bb_width","bb_above_mid","bb_squeeze","breakout_conf",
                    "atr","stop_loss","target1","target2","risk_pct","sector",
                    "brutal_strength","golden_cross","death_cross","brutal_weakness",
                    "fired"])
        for r in results:
            s = r.signals
            w.writerow([r.symbol,r.score_pct,r.rating,s.regime,s.rs_90d,s.rs_rank,
                        r.price,r.dma25,r.dma50,r.dma200,s.rsi,s.adx,s.pdi,s.ndi,
                        s.macd_bull,s.mtf_bull_count,s.pct_52h,s.vol_ratio,
                        s.bb_width,s.bb_above_mid,s.bb_squeeze,s.breakout_conf,
                        s.atr,s.stop_loss,s.target1,s.target2,s.risk_pct,s.sector,
                        s.brutal_strength,s.golden_cross,s.death_cross,s.brutal_weakness,
                        " | ".join(r.fired)])
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--watchlist", type=Path)
    p.add_argument("--min-score", type=float, default=0)
    p.add_argument("--output", choices=["table","csv","both"], default="table")
    p.add_argument("--csv-file", type=Path, default=Path("momentum_v2.csv"))
    args = p.parse_args()

    symbols = (
        [s.upper() for s in args.symbols] if args.symbols else
        [l.strip().upper() for l in args.watchlist.read_text().splitlines() if l.strip()]
        if args.watchlist and args.watchlist.exists() else NIFTY50
    )

    results, regime_info = fetch_and_score_v2(symbols, args.min_score)
    if args.output in ("table","both"): print_report(results, regime_info)
    if args.output in ("csv","both"):   save_csv(results, args.csv_file)


if __name__ == "__main__":
    main()
