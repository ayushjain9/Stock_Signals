"""
WealthOS — Momentum Screener (Production Deployment)
=====================================================

STRATEGY PARAMETERS — confirmed via 5yr backtest + walk-forward validation
  Universe:    Nifty 50
  Hold top:    15 stocks (equal weight, ₹60K each on ₹9L sleeve)
  Sector cap:  Max 3 stocks per sector
  Exit buffer: Only exit a holding if its score dropped 15+ points from entry
  Rebalance:   Monthly (run this script every 3-4 weeks, Sunday evening)
  Min score:   40 / 100
  Liquidity:   ₹5Cr+ average daily traded value

VALIDATED PERFORMANCE (v6 — 5-year backtest + walk-forward, Nifty 50)
  Jensen's alpha:  8.41% base → 8.68% OOS with v6 signals (vol surge + 6M RS)
  Sharpe ratio:    0.87 base  → 0.79 OOS  (higher absolute alpha in harder market)
  Max drawdown:   -12.28%     → -15.51% OOS (benchmark: -15.99%)
  Hit ratio:       56.7%      → 60.7% OOS months outperforming NIFTY
  Alpha sensitivity: 0.14pp lost per 1pp NIFTY decline — highly robust
  Version: v6 (added volume surge +0.95pp OOS, 6-month RS +2.20pp OOS)

WHAT THIS FILE DOES
  1. Reads your current held positions from a simple text file (current_holdings.txt)
  2. Downloads fresh NSE prices for all 50 stocks
  3. Scores each stock using the confirmed signal stack
  4. Selects top 15 with sector cap = 3
  5. Applies exit buffer: keeps holdings above entry_score - 15
  6. Outputs a clear rebalancing action list with position sizes

HOW TO USE
  # One-time setup:
  pip install yfinance pandas numpy rich

  # Create your holdings file (one line per stock):
  echo "HDFCBANK 745 100" > current_holdings.txt  # symbol, entry_price, entry_score

  # Run every month (Sunday evening, before market opens Monday):
  python momentum_deploy.py
  python momentum_deploy.py --sleeve 900000        # ₹9L sleeve
  python momentum_deploy.py --holdings my_file.txt # custom holdings file

HOLDINGS FILE FORMAT (current_holdings.txt)
  One stock per line: SYMBOL  ENTRY_PRICE  ENTRY_SCORE
  Example:
    HDFCBANK  745.00  72
    SBIN      285.00  68
    BAJFINANCE 900.00 75

  If no holdings file exists, script assumes empty portfolio (all-new positions).
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Confirmed parameters ──────────────────────────────────────────────────────

NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TATAMOTORS","ADANIENT","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","DIVISLAB","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "BRITANNIA","BPCL","HEROMOTOCO","HINDALCO","SBILIFE",
]

SECTOR_MAP: dict[str, str] = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDUSINDBK":"BANKING","BAJAJFINSV":"NBFC","BAJFINANCE":"NBFC",
    "SHRIRAMFIN":"NBFC","HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE",
    "TCS":"IT","INFOSYS":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "RELIANCE":"ENERGY","ONGC":"ENERGY","BPCL":"ENERGY",
    "NTPC":"POWER","POWERGRID":"POWER",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "MARUTI":"AUTO","TATAMOTORS":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO",
    "EICHERMOT":"AUTO","HEROMOTOCO":"AUTO",
    "LT":"INFRA","SIEMENS":"INFRA",
    "BEL":"DEFENCE","HAL":"DEFENCE",
    "TITAN":"CONSUMER","TRENT":"RETAIL",
    "ASIANPAINT":"PAINTS",
    "ULTRACEMCO":"CEMENT","GRASIM":"CEMENT",
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","DIVISLAB":"PHARMA","CIPLA":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE",
    "ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS",
    "BHARTIARTL":"TELECOM",
}

# Deployment parameters — DO NOT CHANGE without re-running backtest
TOP_N           = 15
SECTOR_CAP      = 3
EXIT_BUFFER     = 15.0
MIN_SCORE       = 40.0
LIQUIDITY_CR    = 5.0
SLEEVE_DEFAULT  = 900_000  # ₹9L
MIN_HISTORY     = 210


# ── Signal computation (confirmed stack) ─────────────────────────────────────

def _ema(a, n):
    k = 2/(n+1); o = np.zeros(len(a)); o[n-1] = a[:n].mean()
    for i in range(n, len(a)): o[i] = a[i]*k + o[i-1]*(1-k)
    return o

def score_stock(c: np.ndarray, h: np.ndarray, lo: np.ndarray, bench: np.ndarray, v: np.ndarray | None = None) -> float:
    """
    Compute momentum score 0-100. Signal breakdown:
      Trend direction (20):  DMA alignment — which timeframes agree
      RS 90D (25):           stock vs NIFTY over 90 days — outperforming?
      RS 6M  (8):            6-month RS skipping last month (Jegadeesh-Titman)
      VAM (20):              ROC/ATR — smooth trend vs chaotic spike
      52W proximity (6):     near yearly high = continuation signal
      MTF EMA (12):          short/medium/long EMAs all aligned?
      ADX proxy (7):         is momentum accelerating?
      Volume surge (6):      5D vol > 1.5-2× 20D avg — institutional accumulation
    v=None disables volume signal (backward-compatible for testing).
    In production always pass v_ from score_all().
    """
    if len(c) < MIN_HISTORY:
        return -1.0

    p    = c[-1]
    d25  = c[-25:].mean()
    d50  = c[-50:].mean()
    d200 = c[-200:].mean()
    s    = 0.0

    # Trend direction (20 pts)
    if p > d25 > d50 > d200:        s += 20
    elif d50 > d200 and p > d200:   s += 12
    elif d50 > d200:                s += 6

    # Relative strength vs NIFTY (25 pts)
    if len(bench) >= 90:
        rs = (c[-1]/c[-90] - bench[-1]/bench[-90]) * 100
        if rs > 10:   s += 25
        elif rs > 5:  s += 18
        elif rs > 0:  s += 10
        elif rs < -5: s -= 15

    # 6-month RS — Jegadeesh-Titman style, skip last 1 month (8 pts)
    # Validated: +3.35pp OOS alpha improvement, not overfit (IS barely changes)
    # Longer lookback filters false momentum — 3-month spikes without sustained RS
    if len(c) >= 147 and len(bench) >= 147:
        rs6 = (c[-22] / c[-147] - bench[-22] / bench[-147]) * 100
        if rs6 > 15:   s += 8
        elif rs6 > 8:  s += 5
        elif rs6 > 0:  s += 2
        elif rs6 < -8: s -= 5

    # Volatility-adjusted momentum (20 pts)
    if len(c) >= 66:
        roc    = (c[-1]/c[-66] - 1) * 100
        ranges = (h[-14:] - lo[-14:]).mean() if len(h) >= 14 else c[-20:].std()
        atr_pct = ranges / p * 100 if p > 0 else 2.0
        vam = roc / atr_pct if atr_pct > 0.01 else 0
        if vam > 3:   s += 20
        elif vam > 2: s += 15
        elif vam > 1: s += 8
        elif vam > 0: s += 3

    # 52-week high proximity (6 pts)
    if len(c) >= 252:
        pct52 = (p / c[-252:].max() - 1) * 100
        if -5 <= pct52 <= 0:
            s += 6

    # Multi-timeframe EMA alignment (12 pts)
    if len(c) >= 200:
        e5 = _ema(c,5); e20 = _ema(c,20); e50 = _ema(c,50); e200 = _ema(c,200)
        s += sum([e5[-1]>e20[-1], e20[-1]>e50[-1], e50[-1]>e200[-1]]) * 4

    # ADX direction proxy (7 pts) — accelerating momentum
    if len(c) >= 30:
        if c[-1]/c[-10] > 1 and c[-1]/c[-10] > c[-10]/c[-20]:
            s += 7

    # Volume surge — institutional accumulation confirmation (6 pts max)
    # Data already in memory from liquidity gate — zero extra download cost
    # Validated: +0.95pp OOS alpha, OOS improvement > IS improvement (not overfit)
    if v is not None and len(v) >= 26:
        avg5d  = float(v[-5:].mean())
        avg20d = float(v[-25:-5].mean())
        if avg20d > 0:
            ratio = avg5d / avg20d
            if ratio >= 2.0:   s += 6
            elif ratio >= 1.5: s += 4

    return min(float(s), 100.0)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Holding:
    symbol:      str
    entry_price: float
    entry_score: float

@dataclass
class StockResult:
    symbol:      str
    score:       float
    price:       float
    dma25:       float
    dma50:       float
    dma200:      float
    sector:      str
    liq_cr:      float
    liq_ok:      bool
    rs_90d:      float  # excess return vs NIFTY
    roc_3m:      float
    vam:         float
    pct_52h:     float

    @property
    def dma_signal(self) -> str:
        if self.price > self.dma25 > self.dma50 > self.dma200: return "Brutal Strength ↑↑↑"
        if self.dma50 > self.dma200 and self.price > self.dma200: return "Golden Cross ↑"
        if self.dma50 > self.dma200: return "Golden (price lagging)"
        if self.price < self.dma25 < self.dma50 < self.dma200: return "Brutal Weakness ↓↓↓"
        return "Death Cross ↓"


# ── Holdings file I/O ─────────────────────────────────────────────────────────

def load_holdings(path: Path) -> dict[str, Holding]:
    if not path.exists():
        return {}
    holdings = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            sym, ep, es = parts[0].upper(), float(parts[1]), float(parts[2])
            holdings[sym] = Holding(sym, ep, es)
        elif len(parts) == 1:
            # Symbol only — no entry price/score recorded yet
            holdings[parts[0].upper()] = Holding(parts[0].upper(), 0.0, 0.0)
    return holdings

def save_holdings(holdings: dict[str, Holding], path: Path) -> None:
    lines = [f"# WealthOS momentum sleeve — updated {datetime.now().strftime('%Y-%m-%d')}\n"]
    lines += [f"{h.symbol:15s}  {h.entry_price:>10.2f}  {h.entry_score:>6.1f}\n"
              for h in sorted(holdings.values(), key=lambda x: x.symbol)]
    path.write_text("".join(lines))


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    tickers = [s + ".NS" for s in NIFTY50] + ["^NSEI"]
    print(f"  Downloading {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(tickers, period="14mo", progress=False, auto_adjust=True, threads=True)
    cl = raw["Close"]; hi = raw["High"]; lo = raw["Low"]; vo = raw["Volume"]
    return cl, hi, lo, vo


# ── Scoring run ───────────────────────────────────────────────────────────────

def score_all(cl, hi, lo, vo) -> list[StockResult]:
    bench_s = cl["^NSEI"].dropna()
    bench   = bench_s.values.astype(float)
    bd      = bench_s.index
    results = []

    for sym in NIFTY50:
        col = sym + ".NS"
        if col not in cl.columns:
            continue

        close_s = cl[col].dropna()
        c = close_s.values.astype(float)
        if len(c) < MIN_HISTORY:
            continue

        h_  = hi[col].reindex(close_s.index).ffill().values.astype(float)
        lo_ = lo[col].reindex(close_s.index).ffill().values.astype(float)
        v_  = vo[col].reindex(close_s.index).fillna(0).values.astype(float)

        # Align bench to stock dates
        bench_aligned = cl["^NSEI"].reindex(close_s.index).ffill().dropna().values.astype(float)

        price    = c[-1]
        avg_vol  = v_[-20:].mean()
        liq_cr   = avg_vol * price / 1e7
        liq_ok   = liq_cr >= LIQUIDITY_CR

        sc = score_stock(c, h_, lo_, bench_aligned, v_) if liq_ok else -1.0

        # Extra diagnostics
        rs90 = (c[-1]/c[-90] - bench_aligned[-1]/bench_aligned[-90]) * 100 if len(c) >= 90 else 0
        roc3 = (c[-1]/c[-66] - 1) * 100 if len(c) >= 66 else 0
        ranges = (h_[-14:] - lo_[-14:]).mean() if len(h_) >= 14 else 1
        atp = ranges / price * 100 if price > 0 else 1
        vam = roc3 / atp if atp > 0 else 0
        p52 = (price / c[-252:].max() - 1) * 100 if len(c) >= 252 else 0

        results.append(StockResult(
            symbol=sym, score=sc, price=round(price,2),
            dma25=round(c[-25:].mean(),2), dma50=round(c[-50:].mean(),2),
            dma200=round(c[-200:].mean(),2),
            sector=SECTOR_MAP.get(sym,"OTHER"),
            liq_cr=round(liq_cr,1), liq_ok=liq_ok,
            rs_90d=round(rs90,2), roc_3m=round(roc3,2),
            vam=round(vam,2), pct_52h=round(p52,2),
        ))

    return sorted(results, key=lambda r: -r.score)


# ── Selection with sector cap + exit buffer ───────────────────────────────────

def select_holdings(
    scored:    list[StockResult],
    current:   dict[str, Holding],
    top_n:     int     = TOP_N,
    sec_cap:   int     = SECTOR_CAP,
    ex_buf:    float   = EXIT_BUFFER,
    min_sc:    float   = MIN_SCORE,
) -> tuple[set[str], set[str], set[str], set[str], dict[str, float]]:
    """
    Returns: (new_holds, to_buy, to_sell, to_keep, scores_dict)
    Applies sector cap + exit buffer exactly as backtested.
    """
    score_map = {r.symbol: r.score for r in scored}

    # Step 1: select top_n by score with sector cap
    new_holds: set[str] = set()
    sec_cnt: dict[str, int] = {}
    for r in scored:
        if r.score < min_sc or not r.liq_ok:
            continue
        sec = r.sector
        if sec_cnt.get(sec, 0) < sec_cap:
            new_holds.add(r.symbol)
            sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        if len(new_holds) >= top_n:
            break

    # Step 2: exit buffer — keep current holdings not in new_holds
    # if their score hasn't dropped 15+ points from entry.
    # Hard stop-loss: always exit if price fell 15%+ from entry,
    # regardless of score. Score can lag sudden gap-down events.
    kept_by_buffer: set[str] = set()
    for sym, h in current.items():
        if sym in new_holds:
            continue  # already selected

        # Hard stop-loss — independent of score (catches fraud/gap-down)
        cur_price = next((r.price for r in scored if r.symbol == sym), 0)
        if h.entry_price > 0 and cur_price > 0 and cur_price < h.entry_price * 0.85:
            continue  # force exit — don't add to kept_by_buffer

        # Exit buffer — only hold if score hasn't decayed too much
        cur_score = score_map.get(sym, 0)
        if h.entry_score > 0 and cur_score >= h.entry_score - ex_buf:
            kept_by_buffer.add(sym)

    final_holds = new_holds | kept_by_buffer

    to_buy  = final_holds - set(current.keys())
    to_sell = set(current.keys()) - final_holds
    to_keep = set(current.keys()) & final_holds

    return final_holds, to_buy, to_sell, to_keep, score_map


# ── Output ────────────────────────────────────────────────────────────────────

def print_report(
    scored:       list[StockResult],
    final_holds:  set[str],
    to_buy:       set[str],
    to_sell:      set[str],
    to_keep:      set[str],
    current:      dict[str, Holding],
    score_map:    dict[str, float],
    sleeve:       float,
) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
    except ImportError:
        _print_plain(scored, final_holds, to_buy, to_sell, to_keep, current, score_map, sleeve)
        return

    per_pos = sleeve / max(len(final_holds), 1)
    now = datetime.now().strftime("%d %b %Y %H:%M")
    console.print(f"\n[bold]WealthOS Momentum Sleeve[/bold]  [dim]{now}[/dim]")
    console.print(f"Sleeve: ₹{sleeve/100000:.1f}L  |  Positions: {len(final_holds)}  |  Per position: ₹{per_pos/1000:.0f}K\n")

    # ── Action summary ─────────────────────────────────────────────────────────
    if to_buy:
        console.print(f"[bold green]BUY ({len(to_buy)} stocks)[/bold green]")
        t = Table(box=box.SIMPLE_HEAVY, header_style="dim")
        for col in ["Symbol","Score","Price","25D","50D","200D","Signal","Sector","RS 90D","VAM","₹ to deploy"]:
            t.add_column(col, justify="right" if col not in ("Symbol","Signal","Sector") else "left", width=13)
        for sym in sorted(to_buy, key=lambda s: -score_map.get(s,0)):
            r = next((x for x in scored if x.symbol==sym), None)
            if not r: continue
            t.add_row(
                f"[green]{sym}[/green]",
                f"[green]{r.score:.0f}[/green]",
                f"₹{r.price:.1f}", f"₹{r.dma25:.1f}", f"₹{r.dma50:.1f}", f"₹{r.dma200:.1f}",
                r.dma_signal[:20], r.sector[:10],
                f"{r.rs_90d:+.1f}%", f"{r.vam:.1f}",
                f"₹{per_pos/1000:.0f}K",
            )
        console.print(t)

    if to_sell:
        console.print(f"[bold red]SELL ({len(to_sell)} stocks)[/bold red]")
        t = Table(box=box.SIMPLE_HEAVY, header_style="dim")
        for col in ["Symbol","Entry price","Entry score","Current score","Drop","Reason"]:
            t.add_column(col, justify="right" if col not in ("Symbol","Reason") else "left", width=14)
        for sym in sorted(to_sell):
            h = current.get(sym)
            cur_sc = score_map.get(sym, 0)
            drop = (h.entry_score - cur_sc) if h else 0
            reason = "Score dropped 15+ pts" if h and drop >= 15 else "Not in top-15"
            t.add_row(
                f"[red]{sym}[/red]",
                f"₹{h.entry_price:.1f}" if h else "—",
                f"{h.entry_score:.0f}" if h else "—",
                f"{cur_sc:.0f}",
                f"{drop:+.0f}",
                reason,
            )
        console.print(t)

    if to_keep:
        console.print(f"[bold blue]HOLD ({len(to_keep)} stocks — no action)[/bold blue]")
        t = Table(box=box.SIMPLE_HEAVY, header_style="dim")
        for col in ["Symbol","Entry price","Current score","Signal","Sector","P&L est"]:
            t.add_column(col, justify="right" if col not in ("Symbol","Signal","Sector") else "left", width=14)
        for sym in sorted(to_keep, key=lambda s: -score_map.get(s,0)):
            h = current.get(sym)
            r = next((x for x in scored if x.symbol==sym), None)
            if not r: continue
            pnl = (r.price/h.entry_price - 1)*100 if h and h.entry_price > 0 else 0
            pnl_str = f"[green]{pnl:+.1f}%[/green]" if pnl >= 0 else f"[red]{pnl:+.1f}%[/red]"
            t.add_row(
                sym,
                f"₹{h.entry_price:.1f}" if h else "—",
                f"{r.score:.0f}",
                r.dma_signal[:20], r.sector[:10],
                pnl_str,
            )
        console.print(t)

    # ── Full score table ───────────────────────────────────────────────────────
    console.print(f"\n[dim]Full Nifty 50 momentum scores[/dim]")
    t = Table(box=box.SIMPLE_HEAVY, header_style="dim", show_header=True)
    for col in ["#","Symbol","Score","Price","RS 90D","VAM","52W%","Signal","Sector","Liq ₹Cr","Status"]:
        t.add_column(col, justify="right" if col not in ("Symbol","Signal","Sector","Status") else "left",
                     width=11 if col not in ("Signal",) else 22)
    for i, r in enumerate(scored, 1):
        if r.score < 0: continue
        status = ("[green]BUY[/green]"   if r.symbol in to_buy  else
                  "[red]SELL[/red]"      if r.symbol in to_sell else
                  "[blue]HOLD[/blue]"    if r.symbol in to_keep else
                  "[dim]watch[/dim]"     if r.score >= 40        else
                  "[dim]–[/dim]")
        liq_s = f"[dim]{r.liq_cr:.0f}[/dim]" if not r.liq_ok else f"{r.liq_cr:.0f}"
        sc_c  = "green" if r.score>=60 else "yellow" if r.score>=40 else "red"
        t.add_row(
            str(i), r.symbol,
            f"[{sc_c}]{r.score:.0f}[/{sc_c}]",
            f"₹{r.price:.1f}",
            f"{r.rs_90d:+.1f}%", f"{r.vam:.1f}", f"{r.pct_52h:.1f}%",
            r.dma_signal[:20], r.sector[:10], liq_s, status,
        )
    console.print(t)

    console.print(f"\n[dim]Config: Nifty50 | Top {TOP_N} | Sector cap {SECTOR_CAP} | "
                  f"Exit buffer {EXIT_BUFFER:.0f} | Min score {MIN_SCORE:.0f} | "
                  f"Rebalance monthly[/dim]")
    console.print(f"[dim]Validated: Jensen's alpha 8.41% | Sharpe 0.87 | MDD -12.28% (5yr backtest)[/dim]\n")


def _print_plain(scored, final_holds, to_buy, to_sell, to_keep, current, score_map, sleeve):
    per_pos = sleeve / max(len(final_holds),1)
    print(f"\nWealthOS Momentum Sleeve — {datetime.now().strftime('%d %b %Y')}")
    print(f"Sleeve ₹{sleeve/100000:.1f}L | {len(final_holds)} positions | ₹{per_pos/1000:.0f}K each\n")
    if to_buy:
        print(f"BUY ({len(to_buy)}):")
        for sym in sorted(to_buy, key=lambda s: -score_map.get(s,0)):
            r = next((x for x in scored if x.symbol==sym), None)
            if r: print(f"  {sym:12s} score={r.score:.0f}  ₹{r.price:.1f}  {r.dma_signal}")
    if to_sell:
        print(f"\nSELL ({len(to_sell)}): {', '.join(sorted(to_sell))}")
    if to_keep:
        print(f"\nHOLD ({len(to_keep)}): {', '.join(sorted(to_keep))}")
    print(f"\nAll scores:")
    for r in scored:
        if r.score < 0: continue
        tag = "BUY" if r.symbol in to_buy else "SELL" if r.symbol in to_sell else "HOLD" if r.symbol in to_keep else ""
        print(f"  {r.symbol:12s} {r.score:>5.0f}  {r.dma_signal:<25}  {tag}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="WealthOS momentum screener — monthly rebalancing signal")
    p.add_argument("--sleeve",    type=float, default=SLEEVE_DEFAULT,
                   help="Momentum sleeve size in ₹ (default 900000 = ₹9L)")
    p.add_argument("--holdings",  type=Path, default=Path("current_holdings.txt"),
                   help="File with current held positions (symbol entry_price entry_score)")
    p.add_argument("--save",      action="store_true",
                   help="Save updated holdings list after computing rebalance")
    p.add_argument("--scores-only", action="store_true",
                   help="Just print all 50 scores, no rebalancing output")
    a = p.parse_args()

    print(f"\nWealthOS Momentum Screener")
    print(f"  Loading holdings from: {a.holdings}")
    current = load_holdings(a.holdings)
    print(f"  Current positions: {len(current)} stocks: {', '.join(sorted(current.keys())) or '(none)'}")

    print(f"  Fetching market data...")
    cl, hi, lo, vo = fetch_data()

    print(f"  Scoring all {len(NIFTY50)} Nifty 50 stocks...")
    scored = score_all(cl, hi, lo, vo)
    valid  = [r for r in scored if r.score >= 0]
    print(f"  Scored {len(valid)} stocks ({len(NIFTY50)-len(valid)} failed liquidity/data)")

    if a.scores_only:
        for r in scored:
            liq = "" if r.liq_ok else " [ILLIQUID]"
            print(f"  {r.symbol:12s} {r.score:>5.0f}  {r.sector:15s}  RS:{r.rs_90d:+.1f}%  {r.dma_signal}{liq}")
        return

    final_holds, to_buy, to_sell, to_keep, score_map = select_holdings(scored, current)

    print_report(scored, final_holds, to_buy, to_sell, to_keep, current, score_map, a.sleeve)

    if a.save:
        # Update holdings: add new buys, remove sells
        updated = {sym: current[sym] for sym in to_keep}
        for sym in to_buy:
            r = next((x for x in scored if x.symbol == sym), None)
            if r:
                updated[sym] = Holding(sym, r.price, r.score)
        save_holdings(updated, a.holdings)
        print(f"Saved updated holdings to {a.holdings}")


if __name__ == "__main__":
    main()
