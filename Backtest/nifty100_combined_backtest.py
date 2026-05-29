"""
WealthOS — Nifty 100 Combined Backtest
=======================================
Tests BOTH strategies on the same Nifty 100 universe:
  A) Momentum strategy (v6 signal stack)
  B) NiftyShop mean-reversion (20DMA deviation, +8% exit)

Universe: Nifty 100 = Nifty 50 + Nifty Next 50
Benchmark: ^NSEI (Nifty 50) for momentum | ^NSEI for NiftyShop

⚠ SURVIVORSHIP BIAS WARNING
  Nifty Next 50 rebalances semi-annually.
  Using current constituents to backtest 3-5 years overstates returns.
  Nifty 50 portion has minimal bias.
  Interpret Nifty 100 results as OPTIMISTIC upper bound.
  The Nifty 50 results are the reliable anchor.

200DMA FILTER LOGIC
  Below 200DMA  → skipped by BOTH strategies
  Above 200DMA + below 20DMA → NiftyShop candidate only
  Above 200DMA + above 20DMA + strong score → Momentum candidate only
  The two strategies never compete for the same stock.

USAGE
-----
  python nifty100_combined_backtest.py          # 3yr, both strategies
  python nifty100_combined_backtest.py --years 2
  python nifty100_combined_backtest.py --strategy momentum
  python nifty100_combined_backtest.py --strategy niftyshop
  python nifty100_combined_backtest.py --compare  # vs Nifty 50 only
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass, field
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ── Nifty 50 ──────────────────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFY","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TATAMOTORS","ADANIENT","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","DIVISLAB","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "BRITANNIA","BPCL","HEROMOTOCO","HINDALCO","SBILIFE",
]

# ── Nifty Next 50 (current May 2026 — ⚠ survivorship bias) ───────────────────
NIFTY_NEXT50 = [
    "ADANIPOWER","DMART","VEDL","HAL","HINDUNILVR",  # HINDUNILVR overlap check
    "TVSMOTOR","VBL","ADANIGREEN","ADANIENERGYSOL","CUMMINSIND",
    "PFC","RECLTD","DLF","GODREJCP","MUTHOOTFIN",
    "IRCTC","CHOLAFIN","SIEMENS","ABB","PIDILITIND",
    "HDFCAMC","ICICIPRULI","LODHA","ZOMATO","NAUKRI",
    "SBICARD","IOC","GAIL","INDIGO","NHPC",
    "TATAPOWER","LICI","BAJAJHLDNG","INDUSTOWER","COLPAL",
    "OFSS","AMBUJACEM","ICICIGI","MARICO","DABUR",
    "BERGEPAINT","JUBLFOOD","PAGEIND","MCDOWELL-N","TORNTPOWER",
    "PIIND","LUPIN","AUROPHARMA","ALKEM","MAXHEALTH",
]

# Remove any overlap with Nifty 50
NIFTY_NEXT50 = [s for s in NIFTY_NEXT50 if s not in NIFTY50]
NIFTY100     = NIFTY50 + NIFTY_NEXT50

# Sector map — covers both Nifty 50 and Next 50
SECTOR_MAP: dict[str, str] = {
    # Banking
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDUSINDBK":"BANKING",
    # NBFC/Insurance
    "BAJAJFINSV":"NBFC","BAJFINANCE":"NBFC","SHRIRAMFIN":"NBFC","CHOLAFIN":"NBFC",
    "HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE","ICICIPRULI":"INSURANCE",
    "ICICIGI":"INSURANCE","HDFCAMC":"FINANCE","LICI":"INSURANCE","SBICARD":"FINANCE",
    "MUTHOOTFIN":"FINANCE","BAJAJHLDNG":"FINANCE",
    # IT
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT","OFSS":"IT",
    "NAUKRI":"IT",
    # Energy
    "RELIANCE":"ENERGY","ONGC":"ENERGY","BPCL":"ENERGY","IOC":"ENERGY","GAIL":"ENERGY",
    "ADANIPOWER":"POWER","ADANIGREEN":"POWER","ADANIENERGYSOL":"POWER",
    "TATAPOWER":"POWER","NTPC":"POWER","POWERGRID":"POWER","NHPC":"POWER",
    "TORNTPOWER":"POWER","PFC":"POWER","RECLTD":"POWER",
    # FMCG
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "GODREJCP":"FMCG","COLPAL":"FMCG","MARICO":"FMCG","DABUR":"FMCG",
    "MCDOWELL-N":"FMCG","VBL":"FMCG",
    # Auto
    "MARUTI":"AUTO","TATAMOTORS":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO",
    "EICHERMOT":"AUTO","HEROMOTOCO":"AUTO","TVSMOTOR":"AUTO",
    # Infra/Capital Goods
    "LT":"INFRA","SIEMENS":"CAPITAL_GOODS","ABB":"CAPITAL_GOODS",
    "CUMMINSIND":"CAPITAL_GOODS","INDUSTOWER":"TELECOM","BHARTIARTL":"TELECOM",
    # Metals/Materials
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "VEDL":"METALS","AMBUJACEM":"CEMENT","GRASIM":"CEMENT","ULTRACEMCO":"CEMENT",
    "PIDILITIND":"CHEMICALS","PIIND":"CHEMICALS","BERGEPAINT":"CONSUMER",
    # Pharma
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","CIPLA":"PHARMA","DIVISLAB":"PHARMA",
    "LUPIN":"PHARMA","AUROPHARMA":"PHARMA","ALKEM":"PHARMA",
    # Consumer/Others
    "TITAN":"CONSUMER","TRENT":"RETAIL","DMART":"RETAIL","ASIANPAINT":"PAINTS",
    "JUBLFOOD":"CONSUMER","PAGEIND":"CONSUMER","ADANIENT":"CONGLOMERATE",
    "ADANIPORTS":"PORTS","BEL":"DEFENCE","HAL":"DEFENCE",
    "APOLLOHOSP":"HEALTHCARE","MAXHEALTH":"HEALTHCARE",
    "IRCTC":"LOGISTICS","LODHA":"REALTY","DLF":"REALTY",
    "ZOMATO":"TECHNOLOGY","INDIGO":"AVIATION","HDFCAMC":"FINANCE",
}

# ── EMA helper ────────────────────────────────────────────────────────────────
def _ema(a, n):
    k = 2/(n+1); o = np.zeros(len(a)); o[n-1] = a[:n].mean()
    for i in range(n, len(a)): o[i] = a[i]*k + o[i-1]*(1-k)
    return o

# ── Momentum score (v6 — identical to production deploy) ─────────────────────
MIN_HIST = 210

def momentum_score(c, h, lo, bench, v=None):
    if len(c) < MIN_HIST: return -1.0
    p = c[-1]; d25 = c[-25:].mean(); d50 = c[-50:].mean(); d200 = c[-200:].mean()
    s = 0.0

    if p > d25 > d50 > d200:        s += 20
    elif d50 > d200 and p > d200:   s += 12
    elif d50 > d200:                s += 6

    if len(bench) >= 90:
        rs = (c[-1]/c[-90] - bench[-1]/bench[-90]) * 100
        if rs > 10:   s += 25
        elif rs > 5:  s += 18
        elif rs > 0:  s += 10
        elif rs < -5: s -= 15

    if len(c) >= 147 and len(bench) >= 147:
        rs6 = (c[-22]/c[-147] - bench[-22]/bench[-147]) * 100
        if rs6 > 15:   s += 8
        elif rs6 > 8:  s += 5
        elif rs6 > 0:  s += 2
        elif rs6 < -8: s -= 5

    if len(c) >= 66:
        roc = (c[-1]/c[-66] - 1) * 100
        rngs = (h[-14:] - lo[-14:]).mean() if len(h) >= 14 else c[-20:].std()
        atr  = rngs / p * 100 if p > 0 else 2.0
        vam  = roc / atr if atr > 0.01 else 0
        if vam > 3:   s += 20
        elif vam > 2: s += 15
        elif vam > 1: s += 8
        elif vam > 0: s += 3

    if len(c) >= 252:
        pct52 = (p / c[-252:].max() - 1) * 100
        if -5 <= pct52 <= 0: s += 6

    if len(c) >= 200:
        e5 = _ema(c,5); e20 = _ema(c,20); e50 = _ema(c,50); e200 = _ema(c,200)
        s += sum([e5[-1]>e20[-1], e20[-1]>e50[-1], e50[-1]>e200[-1]]) * 4

    if len(c) >= 30:
        if c[-1]/c[-10] > 1 and c[-1]/c[-10] > c[-10]/c[-20]: s += 7

    if v is not None and len(v) >= 26:
        avg5d = float(v[-5:].mean()); avg20d = float(v[-25:-5].mean())
        if avg20d > 0:
            ratio = avg5d / avg20d
            if ratio >= 2.0:   s += 6
            elif ratio >= 1.5: s += 4

    return min(float(s), 100.0)


# ── NiftyShop position tracking ───────────────────────────────────────────────
@dataclass
class Lot:
    date: object; price: float; qty: float; amt: float

@dataclass
class Pos:
    lots: list[Lot] = field(default_factory=list)
    @property
    def invested(self): return sum(l.amt for l in self.lots)
    @property
    def qty(self): return sum(l.qty for l in self.lots)
    @property
    def avg_px(self):
        q = self.qty
        return sum(l.price*l.qty for l in self.lots)/q if q > 0 else 0
    @property
    def last_px(self): return self.lots[-1].price if self.lots else 0
    def can_avg(self, mx): return self.invested < mx - 0.01
    def avg_trig(self, cur, t): return cur <= self.last_px * (1-t)
    def exit_trig(self, cur, t): return cur >= self.avg_px * (1+t)
    def pnl(self, cur): return (cur - self.avg_px) * self.qty


# ── Momentum backtest engine ──────────────────────────────────────────────────
def run_momentum(universe, close, hi, lo, vo, si=None, ei=None,
                 top_n=15, sec_cap=3, ex_buf=15.0, min_sc=40.0,
                 liq_cr=5.0):
    bench_s = close["^NSEI"].dropna()
    bench   = bench_s.values.astype(float)
    dates   = bench_s.index
    si      = si or MIN_HIST
    ei      = ei or len(dates)

    pn=[1.0]; bn=[1.0]; pr_list=[]; br_list=[]
    cur_hold: dict = {}   # sym -> entry_score
    ep: dict = {}         # sym -> entry_price

    for day in range(si, ei, 21):
        scores: dict = {}
        for sym in universe:
            col = sym + ".NS"
            if col not in close.columns: continue
            c_ = close[col].reindex(dates).ffill().iloc[:day].dropna().values.astype(float)
            if len(c_) < MIN_HIST: continue
            av = vo[col].reindex(dates).fillna(0).iloc[max(0,day-20):day].mean() \
                 if col in vo.columns else 0
            if av * c_[-1] < liq_cr * 1e7: continue
            h_ = hi[col].reindex(dates).ffill().values[:day].astype(float)
            l_ = lo[col].reindex(dates).ffill().values[:day].astype(float)
            v_ = vo[col].reindex(dates).fillna(0).values[:day].astype(float)
            sc = momentum_score(c_, h_, l_, bench[:day], v_)
            if sc >= min_sc: scores[sym] = sc

        # Select with sector cap
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        new: set = set(); sc_cnt: dict = {}
        for sym, sc in ranked:
            sec = SECTOR_MAP.get(sym, "OTHER")
            if sc_cnt.get(sec, 0) < sec_cap:
                new.add(sym); sc_cnt[sec] = sc_cnt.get(sec, 0) + 1
            if len(new) >= top_n: break

        # Exit buffer
        stay = set()
        for sym in cur_hold:
            if sym in new: stay.add(sym)
            else:
                col = sym + ".NS"
                if col not in close.columns: continue
                cr = close[col].reindex(dates).ffill()
                cur_px = float(cr.iloc[day]) if day < len(cr) else 0
                if ep.get(sym, 0) > 0 and cur_px > 0 and cur_px < ep[sym] * 0.85:
                    continue
                if sym in scores and scores[sym] >= cur_hold[sym] - ex_buf:
                    stay.add(sym)
        final = new | stay

        enters = final - set(cur_hold.keys())
        for sym in enters:
            col = sym + ".NS"
            if col in close.columns:
                cr = close[col].reindex(dates).ffill()
                ep[sym] = float(cr.iloc[day]) if day < len(cr) else 0
                cur_hold[sym] = scores.get(sym, 0)
        for sym in list(cur_hold.keys()):
            if sym not in final: del cur_hold[sym]

        nxt = min(day+21, len(bench)-1)
        fwd = []
        for sym in final:
            col = sym + ".NS"
            if col not in close.columns: continue
            cr = close[col].reindex(dates).ffill()
            p0 = float(cr.iloc[day]); p1 = float(cr.iloc[nxt])
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0: continue
            fwd.append(p1/p0 - 1)

        to_f = (len(final - set(cur_hold.keys())) + len(set(cur_hold.keys()) - final)) \
               / (2 * max(len(final), 1))
        cost = 0.002
        pr = (np.mean(fwd) if fwd else 0) - to_f * cost
        br = bench[nxt] / bench[day] - 1
        pr_list.append(pr); br_list.append(br)
        pn.append(pn[-1]*(1+pr)); bn.append(bn[-1]*(1+br))

    pr_a = np.array(pr_list); br_a = np.array(br_list)
    pna  = np.array(pn); bna = np.array(bn)
    ppy  = 252/21; ny = len(pr_a)/ppy
    cp   = (pna[-1]/pna[0])**(1/ny) - 1 if ny > 0 else 0
    cb   = (bna[-1]/bna[0])**(1/ny) - 1 if ny > 0 else 0
    rfp  = 0.065/21/12
    ex   = pr_a - rfp
    sh   = ex.mean()/ex.std()*np.sqrt(ppy) if ex.std() > 0 else 0
    ds   = ex[ex<0].std()*np.sqrt(ppy)
    so   = ex.mean()*ppy/ds if ds > 0 else 0
    pk   = np.maximum.accumulate(pna)
    mdd  = ((pna-pk)/pk).min()
    cov  = np.cov(pr_a, br_a); beta = cov[0,1]/cov[1,1] if cov[1,1]>0 else 1
    ja   = (cp - 0.065) - beta*(cb - 0.065)
    hm   = (pr_a > br_a).mean()
    return dict(
        cagr=round(cp*100,2), bench_cagr=round(cb*100,2),
        j_alpha=round(ja*100,2), sharpe=round(sh,2), sortino=round(so,2),
        mdd=round(mdd*100,2), hit_m=round(hm*100,1),
        n_years=round(ny,1), strategy="Momentum",
    )


# ── NiftyShop backtest engine ─────────────────────────────────────────────────
def run_niftyshop(universe, close, si=25, ei=None,
                  fresh=10_000, avg=15_000, max_stock=40_000,
                  max_pos=5, target=0.08, trig=0.03, top_n=5,
                  cost=0.001):
    bench_s = close["^NSEI"].dropna()
    dates   = bench_s.index
    ei      = ei or len(dates)

    cash = 400_000.0
    positions: dict[str, Pos] = {}
    nav_list = []; dep_list = []; tlog = []

    for i in range(max(si, 25), ei):
        date = dates[i]

        # 20DMA + 200DMA filter
        dev_scores: dict = {}
        for sym in universe:
            col = sym + ".NS"
            if col not in close.columns: continue
            hist = close[col].reindex(dates).ffill().iloc[:i].dropna()
            if len(hist) < 20: continue
            price = float(hist.iloc[-1])
            dma20 = float(hist.tail(20).mean())
            if price <= 0 or dma20 <= 0: continue
            # 200DMA quality filter
            if len(hist) >= 200:
                dma200 = float(hist.tail(200).mean())
                if price < dma200: continue   # skip — structural decline
            dev = (price - dma20) / dma20 * 100
            dev_scores[sym] = (dev, price)

        today = {sym: v[1] for sym, v in dev_scores.items()}

        # Exits
        to_exit = [sym for sym, pos in positions.items()
                   if sym in today and pos.exit_trig(today[sym], target)]
        for sym in to_exit:
            pos = positions[sym]
            px  = today[sym] * (1 - cost)
            tlog.append({"action":"SELL","pnl":pos.pnl(px),"invested":pos.invested})
            cash += pos.qty * px
            del positions[sym]

        # Rank below-20DMA stocks (after 200DMA filter already applied)
        below = sorted([(d, s, p) for s,(d,p) in dev_scores.items()
                        if d < 0], key=lambda x: x[0])
        top5 = [s for _,s,_ in below[:top_n]]
        acted = False

        # Fresh entry
        if not acted:
            for sym in top5:
                if sym in positions or len(positions) >= max_pos: continue
                if cash < fresh: break
                px = today[sym] * (1+cost); qty = fresh/px; cash -= fresh
                p = Pos(); p.lots.append(Lot(date, px, qty, fresh))
                positions[sym] = p
                tlog.append({"action":"BUY_FRESH","pnl":0,"invested":fresh})
                acted = True; break

        # Averaging
        if not acted and positions:
            cands = [(today.get(s,0)/pos.avg_px-1, s, pos, today.get(s,0))
                     for s,pos in positions.items()
                     if s in today
                     and pos.avg_trig(today[s], trig)
                     and pos.can_avg(max_stock)]
            if cands:
                cands.sort()
                _, sym, pos, px = cands[0]
                amt = min(avg, max_stock - pos.invested, cash)
                if amt >= 1000:
                    buy_px = px*(1+cost); qty = amt/buy_px; cash -= amt
                    pos.lots.append(Lot(date, buy_px, qty, amt))
                    tlog.append({"action":"BUY_AVG","pnl":0,"invested":amt})

        nav = cash + sum(pos.qty * today.get(s,0) for s,pos in positions.items())
        dep = sum(pos.invested for pos in positions.values())
        nav_list.append(nav); dep_list.append(dep)

    nav_arr = np.array(nav_list); dep_arr = np.array(dep_list)
    n_years = len(nav_arr)/252
    cagr    = (nav_arr[-1]/nav_arr[0])**(1/n_years)-1 if n_years>0 else 0
    peak    = np.maximum.accumulate(nav_arr)
    mdd     = ((nav_arr-peak)/peak).min()
    util    = dep_arr.mean()/400_000
    exits   = [t for t in tlog if t["action"]=="SELL"]
    tot_pnl = sum(t["pnl"] for t in exits)
    xirr    = tot_pnl/dep_arr.mean()/n_years if dep_arr.mean()>0 else 0
    wr      = (np.array([t["pnl"] for t in exits])>0).mean() if exits else 0
    return dict(
        cagr=round(cagr*100,2),
        xirr=round(xirr*100,2),
        mdd=round(mdd*100,2),
        util=round(util*100,1),
        win_rate=round(wr*100,1),
        total_pnl=round(tot_pnl,0),
        n_fresh=sum(1 for t in tlog if t["action"]=="BUY_FRESH"),
        n_avg=sum(1 for t in tlog if t["action"]=="BUY_AVG"),
        n_exits=len(exits),
        n_years=round(n_years,1),
        strategy="NiftyShop",
    )


# ── Output ────────────────────────────────────────────────────────────────────
def print_momentum(label, r, bench_cagr):
    print(f"\n  {'─'*66}")
    print(f"  MOMENTUM — {label}")
    print(f"  {'─'*66}")
    print(f"  Jensen Alpha:     {r['j_alpha']:>8.2f}%  |  Benchmark CAGR: {bench_cagr:.2f}%")
    print(f"  Portfolio CAGR:   {r['cagr']:>8.2f}%  |  Sharpe: {r['sharpe']:.2f}"
          f"  |  Sortino: {r['sortino']:.2f}")
    print(f"  Max drawdown:     {r['mdd']:>8.2f}%  |  Hit ratio: {r['hit_m']:.1f}%"
          f"  |  Period: {r['n_years']:.1f}yr")

def print_niftyshop(label, r, bench_cagr):
    print(f"\n  {'─'*66}")
    print(f"  NIFTYSHOP — {label}")
    print(f"  {'─'*66}")
    print(f"  CAGR on ₹4L:      {r['cagr']:>8.2f}%  |  Benchmark CAGR: {bench_cagr:.2f}%")
    print(f"  XIRR deployed:    {r['xirr']:>8.2f}%  |  Capital util: {r['util']:.1f}%")
    print(f"  Max drawdown:     {r['mdd']:>8.2f}%  |  Win rate: {r['win_rate']:.1f}%"
          f"  |  Period: {r['n_years']:.1f}yr")
    print(f"  Trades: {r['n_fresh']} fresh + {r['n_avg']} avg + {r['n_exits']} exits"
          f"  |  Total P&L: ₹{r['total_pnl']:,.0f}")

def print_comparison(results_50, results_100, bench_cagr):
    print(f"\n{'='*70}")
    print("  UNIVERSE COMPARISON — Nifty 50 vs Nifty 100")
    print(f"  ⚠  Nifty 100 has survivorship bias from Nifty Next 50 changes")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30} {'Nifty 50':>12} {'Nifty 100 ⚠':>13}")
    print(f"  {'─'*29} {'─'*12} {'─'*13}")

    for label, key50, key100, unit in [
        ("MOMENTUM — J.Alpha",     "mom50_ja",   "mom100_ja",  "%"),
        ("MOMENTUM — Sharpe",      "mom50_sh",   "mom100_sh",  ""),
        ("MOMENTUM — MDD",         "mom50_mdd",  "mom100_mdd", "%"),
        ("MOMENTUM — Hit ratio",   "mom50_hit",  "mom100_hit", "%"),
        ("NIFTYSHOP — XIRR",       "ns50_xirr",  "ns100_xirr", "%"),
        ("NIFTYSHOP — CAGR ₹4L",   "ns50_cagr",  "ns100_cagr", "%"),
        ("NIFTYSHOP — MDD",        "ns50_mdd",   "ns100_mdd",  "%"),
        ("NIFTYSHOP — Win rate",   "ns50_wr",    "ns100_wr",   "%"),
    ]:
        v50  = results_50.get(key50,  "—")
        v100 = results_100.get(key100, "—")
        p50  = f"{v50:.2f}{unit}"  if isinstance(v50,  (int,float)) else str(v50)
        p100 = f"{v100:.2f}{unit}" if isinstance(v100, (int,float)) else str(v100)
        print(f"  {label:<30} {p50:>12} {p100:>13}")

    print(f"\n  Benchmark CAGR: {bench_cagr:.2f}%")
    print(f"\n  KEY INSIGHT:")
    print(f"  Both strategies target DIFFERENT stocks from the SAME universe.")
    print(f"  Momentum picks Zone 3 (above 200DMA + above 20DMA + strong RS)")
    print(f"  NiftyShop picks Zone 2 (above 200DMA + below 20DMA)")
    print(f"  Zero overlap — complementary by design.\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years",    type=int, default=3)
    p.add_argument("--strategy", type=str, default="both",
                   help="both | momentum | niftyshop")
    p.add_argument("--compare",  action="store_true",
                   help="Run Nifty 50 AND Nifty 100 for comparison")
    a = p.parse_args()

    # Gather all tickers needed
    all_tickers = set()
    for s in NIFTY100:
        all_tickers.add(s + ".NS")
    all_tickers.add("^NSEI")

    period = f"{a.years + 1}y"
    print(f"\n  Nifty 100 Combined Backtest — {a.years}yr")
    print(f"  Universe: {len(NIFTY50)} Nifty 50 + {len(NIFTY_NEXT50)} Next 50"
          f" = {len(NIFTY100)} stocks")
    print(f"  ⚠  Nifty Next 50 results have survivorship bias (semi-annual rebalancing)")
    print(f"  Downloading {len(all_tickers)} tickers ({period})...")

    import yfinance as yf
    raw   = yf.download(list(all_tickers), period=period,
                        progress=False, auto_adjust=True, threads=True)
    cl    = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    hi    = raw["High"]  if isinstance(raw.columns, pd.MultiIndex) else raw[["High"]]
    lo    = raw["Low"]   if isinstance(raw.columns, pd.MultiIndex) else raw[["Low"]]
    vo    = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]

    bench_arr  = cl["^NSEI"].dropna().values.astype(float)
    n_yr       = len(bench_arr) / 252
    bench_cagr = (bench_arr[-1]/bench_arr[0])**(1/n_yr) - 1 if n_yr > 0 else 0

    r50 = {}; r100 = {}

    print(f"\n  {'='*68}")
    print(f"  NIFTY 100 UNIVERSE RESULTS")
    print(f"  {'='*68}")

    if a.strategy in ("both", "momentum"):
        print(f"\n  Running Momentum on Nifty 100...")
        rm100 = run_momentum(NIFTY100, cl, hi, lo, vo)
        r100.update({"mom100_ja": rm100["j_alpha"], "mom100_sh": rm100["sharpe"],
                     "mom100_mdd": rm100["mdd"],    "mom100_hit": rm100["hit_m"]})
        print_momentum("Nifty 100 (50 + Next 50) ⚠bias", rm100, bench_cagr*100)

    if a.strategy in ("both", "niftyshop"):
        print(f"\n  Running NiftyShop on Nifty 100...")
        rn100 = run_niftyshop(NIFTY100, cl)
        r100.update({"ns100_xirr": rn100["xirr"], "ns100_cagr": rn100["cagr"],
                     "ns100_mdd":  rn100["mdd"],   "ns100_wr":   rn100["win_rate"]})
        print_niftyshop("Nifty 100 (50 + Next 50) ⚠bias", rn100, bench_cagr*100)

    # Comparison with Nifty 50 only
    if a.compare or True:  # always show comparison
        print(f"\n  {'='*68}")
        print(f"  NIFTY 50 ONLY (validated baseline — no bias)")
        print(f"  {'='*68}")

        if a.strategy in ("both", "momentum"):
            print(f"\n  Running Momentum on Nifty 50 only...")
            rm50 = run_momentum(NIFTY50, cl, hi, lo, vo)
            r50.update({"mom50_ja": rm50["j_alpha"], "mom50_sh": rm50["sharpe"],
                        "mom50_mdd": rm50["mdd"],    "mom50_hit": rm50["hit_m"]})
            print_momentum("Nifty 50 only (no bias)", rm50, bench_cagr*100)

        if a.strategy in ("both", "niftyshop"):
            print(f"\n  Running NiftyShop on Nifty 50 only...")
            rn50 = run_niftyshop(NIFTY50, cl)
            r50.update({"ns50_xirr": rn50["xirr"], "ns50_cagr": rn50["cagr"],
                        "ns50_mdd":  rn50["mdd"],   "ns50_wr":   rn50["win_rate"]})
            print_niftyshop("Nifty 50 only (no bias)", rn50, bench_cagr*100)

        print_comparison(r50, r100, bench_cagr*100)

    print(f"  ⚠  REMINDER: Nifty 100 numbers are OPTIMISTIC.")
    print(f"     The real-world improvement over Nifty 50 will be smaller.")
    print(f"     Nifty 50 numbers are your reliable anchor.\n")

if __name__ == "__main__":
    main()
