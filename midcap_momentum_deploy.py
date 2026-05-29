"""
WealthOS — Midcap Momentum Screener (Production Deployment)
============================================================

SAME signal stack as momentum_deploy.py (v6 — validated).
DIFFERENT universe: Nifty Midcap 50 (F&O traded, liquid mid-caps).
DIFFERENT benchmark: Nifty Midcap 50 index (^NSMIDCP).

STRATEGY PARAMETERS
  Universe:    Nifty Midcap 50 (F&O traded stocks)
  Hold top:    10 stocks (equal weight — smaller universe than Nifty 50)
  Sector cap:  3 stocks per sector max
  Exit buffer: 15 pts — same as Nifty 50 version
  Rebalance:   Monthly (Sunday evening before Monday open)
  Min score:   40 / 100
  Liquidity:   ₹2Cr+ average daily value (midcaps less liquid)
  Benchmark:   Nifty Midcap 50 index (^NSMIDCP)

NOTE ON PERFORMANCE
  Midcap momentum is academically stronger than large-cap momentum
  (less efficient pricing → more persistent factor) but:
  - Higher volatility → larger drawdowns
  - Less institutional support → slower recovery on losers
  - Run as a SEPARATE sleeve from Nifty 50 momentum
  - Not backtested with walk-forward validation yet
  - Paper trade for 2 months before deploying real capital

SIGNAL STACK (identical to Nifty 50 version — v6 validated)
  Trend direction (20):  DMA alignment
  RS 90D vs midcap (25): stock vs ^NSMIDCP over 90 days
  RS 6M (8):             Jegadeesh-Titman 6-month RS
  VAM (20):              volatility-adjusted momentum
  52W proximity (6):     near yearly high
  MTF EMA (12):          multi-timeframe alignment
  ADX proxy (7):         momentum acceleration
  Volume surge (6):      institutional accumulation

USAGE
-----
  python midcap_momentum_deploy.py
  python midcap_momentum_deploy.py --sleeve 200000    # ₹2L sleeve
  python midcap_momentum_deploy.py --holdings my.txt
  python midcap_momentum_deploy.py --save             # save new holdings
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from regime_detector import print_regime_header
warnings.filterwarnings("ignore")

# ── Universe and sector map — imported from config.py (single source of truth) ─
from config import (
    MIDCAP50_LIST      as MIDCAP50,
    MIDCAP50_SECTOR_MAP as SECTOR_MAP,
)

# ── Parameters ────────────────────────────────────────────────────────────────
TOP_N        = 10      # top 10 from 50 stocks (same ratio as 15 from Nifty 50)
SECTOR_CAP   = 3
EXIT_BUFFER  = 15.0
MIN_SCORE    = 40.0
LIQUIDITY_CR = 2.0     # ₹2Cr/day — midcaps less liquid than large caps
SLEEVE_DEF   = 200_000 # ₹2L default — smaller sleeve for midcap
MIN_HISTORY  = 210
BENCHMARK    = "^NSMIDCP"  # Nifty Midcap 50 index


# ── Signal computation (v6 — identical to Nifty 50 version) ──────────────────
def _ema(a, n):
    k = 2/(n+1); o = np.zeros(len(a)); o[n-1] = a[:n].mean()
    for i in range(n, len(a)): o[i] = a[i]*k + o[i-1]*(1-k)
    return o


def score_stock(c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                bench: np.ndarray, v: np.ndarray | None = None) -> float:
    """
    Momentum score 0-100 — identical signal stack to Nifty 50 version.
    Bench here is the Midcap 50 index, not Nifty 50.
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

    # RS 90D vs Midcap 50 benchmark (25 pts)
    if len(bench) >= 90:
        rs = (c[-1]/c[-90] - bench[-1]/bench[-90]) * 100
        if rs > 10:   s += 25
        elif rs > 5:  s += 18
        elif rs > 0:  s += 10
        elif rs < -5: s -= 15

    # 6-month RS — Jegadeesh-Titman, skip last 1 month (8 pts)
    if len(c) >= 147 and len(bench) >= 147:
        rs6 = (c[-22] / c[-147] - bench[-22] / bench[-147]) * 100
        if rs6 > 15:   s += 8
        elif rs6 > 8:  s += 5
        elif rs6 > 0:  s += 2
        elif rs6 < -8: s -= 5

    # Volatility-adjusted momentum (20 pts)
    if len(c) >= 66:
        roc     = (c[-1]/c[-66] - 1) * 100
        ranges  = (h[-14:] - lo[-14:]).mean() if len(h) >= 14 else c[-20:].std()
        atr_pct = ranges / p * 100 if p > 0 else 2.0
        vam     = roc / atr_pct if atr_pct > 0.01 else 0
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

    # ADX direction proxy (7 pts)
    if len(c) >= 30:
        if c[-1]/c[-10] > 1 and c[-1]/c[-10] > c[-10]/c[-20]:
            s += 7

    # Volume surge (6 pts)
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
    entry_date:  str = ""   # YYYY-MM-DD; empty for legacy holdings without a date

@dataclass
class StockResult:
    symbol:   str
    score:    float
    price:    float
    dma25:    float
    dma50:    float
    dma200:   float
    sector:   str
    liq_cr:   float
    liq_ok:   bool
    rs_90d:   float
    roc_3m:   float
    vam:      float
    pct_52h:  float

    @property
    def dma_signal(self) -> str:
        if self.price > self.dma25 > self.dma50 > self.dma200: return "Brutal Strength ↑↑↑"
        if self.dma50 > self.dma200 and self.price > self.dma200: return "Golden Cross ↑"
        if self.dma50 > self.dma200: return "Golden (price lagging)"
        return "Death Cross / Weak ↓"


# ── Holdings I/O ──────────────────────────────────────────────────────────────
def load_holdings(path: Path) -> dict[str, Holding]:
    if not path.exists():
        return {}
    holdings = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            sym, ep, es, ed = parts[0].upper(), float(parts[1]), float(parts[2]), parts[3]
            holdings[sym] = Holding(sym, ep, es, ed)
        elif len(parts) >= 3:
            sym, ep, es = parts[0].upper(), float(parts[1]), float(parts[2])
            holdings[sym] = Holding(sym, ep, es)
        elif len(parts) == 1:
            holdings[parts[0].upper()] = Holding(parts[0].upper(), 0.0, 0.0)
    return holdings

def save_holdings(holdings: dict[str, Holding], path: Path) -> None:
    lines = [f"# WealthOS Midcap Momentum — updated {datetime.now().strftime('%Y-%m-%d')}\n",
             f"# symbol           entry_price  score  entry_date\n"]
    lines += [f"{h.symbol:15s}  {h.entry_price:>10.2f}  {h.entry_score:>6.1f}  {h.entry_date or datetime.now().strftime('%Y-%m-%d')}\n"
              for h in sorted(holdings.values(), key=lambda x: x.symbol)]
    path.write_text("".join(lines))


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_data():
    import yfinance as yf
    tickers = [s + ".NS" for s in MIDCAP50] + [BENCHMARK]
    print(f"  Downloading {len(tickers)} tickers (Nifty Midcap 50 + benchmark)...")
    raw = yf.download(tickers, period="18mo", progress=False,
                      auto_adjust=True, threads=True)
    return raw["Close"], raw["High"], raw["Low"], raw["Volume"]


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_all(cl, hi, lo, vo) -> list[StockResult]:
    bench_s = cl[BENCHMARK].dropna()
    bench   = bench_s.values.astype(float)
    results = []

    for sym in MIDCAP50:
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
        bench_aligned = cl[BENCHMARK].reindex(close_s.index).ffill().dropna().values.astype(float)

        price   = c[-1]
        avg_vol = v_[-20:].mean()
        liq_cr  = avg_vol * price / 1e7
        liq_ok  = liq_cr >= LIQUIDITY_CR

        sc = score_stock(c, h_, lo_, bench_aligned, v_) if liq_ok else -1.0

        rs90 = (c[-1]/c[-90] - bench_aligned[-1]/bench_aligned[-90]) * 100 if len(c) >= 90 else 0
        roc3 = (c[-1]/c[-66] - 1) * 100 if len(c) >= 66 else 0
        rngs = (h_[-14:] - lo_[-14:]).mean() if len(h_) >= 14 else 1
        atp  = rngs / price * 100 if price > 0 else 1
        vam  = roc3 / atp if atp > 0 else 0
        p52  = (price / c[-252:].max() - 1) * 100 if len(c) >= 252 else 0

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


# ── Selection ─────────────────────────────────────────────────────────────────
def select_holdings(scored, current, top_n=TOP_N, sec_cap=SECTOR_CAP,
                    ex_buf=EXIT_BUFFER, min_sc=MIN_SCORE):
    score_map = {r.symbol: r.score for r in scored}
    price_map = {r.symbol: r.price for r in scored}

    # Step 0: hard stop-loss gate — applies to ALL current holdings unconditionally.
    # Must run before top-N selection so a high-scoring but crashed stock
    # cannot be "saved" by remaining in the top-N list.
    hard_stopped: set[str] = set()
    for sym, h in current.items():
        cur_price = price_map.get(sym, 0)
        if h.entry_price > 0 and cur_price > 0 and cur_price < h.entry_price * 0.85:
            hard_stopped.add(sym)

    # Hard-stopped stocks are excluded from top-N even if their score is still high.
    new_holds: set[str] = set()
    sec_cnt: dict[str, int] = {}
    for r in scored:
        if r.symbol in hard_stopped:
            continue  # hard stop is unconditional — never re-select
        if r.score < min_sc or not r.liq_ok:
            continue
        sec = r.sector
        if sec_cnt.get(sec, 0) < sec_cap:
            new_holds.add(r.symbol)
            sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        if len(new_holds) >= top_n:
            break

    kept_by_buffer: set[str] = set()
    for sym, h in current.items():
        if sym in new_holds:
            continue
        if sym in hard_stopped:
            continue  # force exit — hard stop supersedes buffer
        cur_score = score_map.get(sym, 0)
        if h.entry_score > 0 and cur_score >= h.entry_score - ex_buf:
            kept_by_buffer.add(sym)

    final_holds = new_holds | kept_by_buffer
    to_buy  = final_holds - set(current.keys())
    to_sell = set(current.keys()) - final_holds
    to_keep = set(current.keys()) & final_holds
    return final_holds, to_buy, to_sell, to_keep, score_map


# ── Output ────────────────────────────────────────────────────────────────────
def print_report(scored, final_holds, to_buy, to_sell, to_keep,
                 current, score_map, sleeve):
    per_pos = sleeve / max(len(final_holds), 1)
    now = datetime.now().strftime("%d %b %Y %H:%M")

    print()
    print("=" * 72)
    print(f"  WealthOS — Midcap Momentum Sleeve    {now}")
    print(f"  Sleeve: ₹{sleeve/100000:.1f}L  |  "
          f"Positions: {len(final_holds)}/{TOP_N}  |  "
          f"Per position: ₹{per_pos/1000:.0f}K")
    print(f"  Universe: Nifty Midcap 50  |  Benchmark: {BENCHMARK}")
    print("=" * 72)

    # BUY
    if to_buy:
        print(f"\n  ✅ BUY ({len(to_buy)} stocks)")
        print(f"  {'Symbol':<12} {'Score':>6} {'Price':>9} {'25D':>9} {'200D':>9}"
              f" {'Signal':<22} {'Sector':<12} {'RS90D':>7} {'₹K':>5}")
        print(f"  {'─'*11} {'─'*6} {'─'*9} {'─'*9} {'─'*9}"
              f" {'─'*21} {'─'*11} {'─'*7} {'─'*5}")
        for sym in sorted(to_buy, key=lambda s: -score_map.get(s, 0)):
            r = next((x for x in scored if x.symbol == sym), None)
            if not r: continue
            print(f"  {sym:<12} {r.score:>6.0f} ₹{r.price:>8.1f} ₹{r.dma25:>8.1f}"
                  f" ₹{r.dma200:>8.1f} {r.dma_signal[:21]:<22}"
                  f" {r.sector[:11]:<12} {r.rs_90d:>+6.1f}%"
                  f" ₹{per_pos/1000:.0f}K")

    # SELL
    if to_sell:
        print(f"\n  🔴 SELL ({len(to_sell)} stocks)")
        print(f"  {'Symbol':<12} {'Entry ₹':>9} {'EntSc':>6} {'CurSc':>6}"
              f" {'Drop':>6} {'Reason'}")
        print(f"  {'─'*11} {'─'*9} {'─'*6} {'─'*6} {'─'*6} {'─'*25}")
        for sym in sorted(to_sell):
            h   = current.get(sym)
            cur = score_map.get(sym, 0)
            drop = (h.entry_score - cur) if h else 0
            cur_px = next((r.price for r in scored if r.symbol == sym), 0)
            reason = "Hard stop-loss (-15%)" if h and h.entry_price > 0 \
                     and cur_px > 0 and cur_px < h.entry_price * 0.85 \
                     else f"Score dropped {drop:.0f} pts" if drop >= EXIT_BUFFER \
                     else "Outside top-10"
            print(f"  {sym:<12} ₹{h.entry_price:>8.1f} {h.entry_score:>6.0f}"
                  f" {cur:>6.0f} {drop:>+5.0f}  {reason}" if h else
                  f"  {sym:<12} {'—':>9} {'—':>6} {cur:>6.0f} {'—':>6}  {reason}")

    # HOLD
    if to_keep:
        print(f"\n  🔵 HOLD ({len(to_keep)} stocks — no action)")
        print(f"  {'Symbol':<12} {'Score':>6} {'Price':>9} {'EntSc':>6} {'Chg':>6}")
        print(f"  {'─'*11} {'─'*6} {'─'*9} {'─'*6} {'─'*6}")
        for sym in sorted(to_keep):
            r = next((x for x in scored if x.symbol == sym), None)
            h = current.get(sym)
            if not r: continue
            esc = h.entry_score if h else 0
            print(f"  {sym:<12} {r.score:>6.0f} ₹{r.price:>8.1f}"
                  f" {esc:>6.0f} {r.score-esc:>+5.0f}")

    # Full ranking
    print(f"\n  FULL MIDCAP 50 RANKING (top 20)")
    print(f"  {'Rank':<5} {'Symbol':<12} {'Score':>6} {'Price':>9}"
          f" {'Signal':<22} {'Sector':<12} {'Liq₹Cr':>7} {'RS90D':>7}")
    print(f"  {'─'*4} {'─'*11} {'─'*6} {'─'*9}"
          f" {'─'*21} {'─'*11} {'─'*7} {'─'*7}")
    rank = 0
    for r in scored[:25]:
        if not r.liq_ok:
            continue
        rank += 1
        if rank > 20:
            break
        tag = " ← HOLD" if r.symbol in to_keep else \
              " ← BUY"  if r.symbol in to_buy  else \
              " ← BUF"  if r.symbol in (final_holds - to_buy - to_keep) else ""
        print(f"  {rank:<5} {r.symbol:<12} {r.score:>6.0f} ₹{r.price:>8.1f}"
              f" {r.dma_signal[:21]:<22} {r.sector[:11]:<12}"
              f" {r.liq_cr:>7.1f} {r.rs_90d:>+6.1f}%{tag}")

    print(f"\n  Sleeve ₹{sleeve/100000:.1f}L  |  "
          f"₹{per_pos/1000:.0f}K per position  |  "
          f"Benchmark: {BENCHMARK}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Midcap Momentum — monthly rebalancing signal (Nifty Midcap 50)")
    p.add_argument("--sleeve",   type=float, default=SLEEVE_DEF)
    p.add_argument("--holdings", type=str,   default="midcap_holdings.txt")
    p.add_argument("--save",     action="store_true",
                   help="Save new holdings to file after running")
    a = p.parse_args()

    print_regime_header()
    holdings_path = Path(a.holdings)

    print(f"\n  WealthOS — Midcap Momentum Screener")
    print(f"  Sleeve: ₹{a.sleeve/100000:.1f}L  |  "
          f"Top {TOP_N} stocks  |  Sector cap {SECTOR_CAP}  |  "
          f"Exit buffer {EXIT_BUFFER:.0f}pts")
    print(f"  Universe: Nifty Midcap 50  |  Liquidity gate: ₹{LIQUIDITY_CR}Cr+\n")

    current = load_holdings(holdings_path)
    if current:
        print(f"  Current holdings ({len(current)}): "
              f"{', '.join(sorted(current.keys()))}\n")
    else:
        print("  No current holdings — fresh portfolio run.\n")

    cl, hi, lo, vo = fetch_data()
    scored = score_all(cl, hi, lo, vo)

    final_holds, to_buy, to_sell, to_keep, score_map = select_holdings(
        scored, current)

    print_report(scored, final_holds, to_buy, to_sell, to_keep,
                 current, score_map, a.sleeve)

    if a.save:
        new_holdings = {}
        for sym in final_holds:
            r  = next((x for x in scored if x.symbol == sym), None)
            ep = current[sym].entry_price if sym in current and current[sym].entry_price > 0 \
                 else (r.price if r else 0)
            es = current[sym].entry_score if sym in current and sym not in to_buy \
                 else score_map.get(sym, 0)
            existing_date = current[sym].entry_date if sym in current and sym not in to_buy else ""
            new_holdings[sym] = Holding(sym, ep, es, existing_date or datetime.now().strftime("%Y-%m-%d"))
        save_holdings(new_holdings, holdings_path)
        print(f"  Saved {len(new_holdings)} holdings → {holdings_path}\n")


if __name__ == "__main__":
    main()
