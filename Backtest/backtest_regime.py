"""
WealthOS — Regime Filter Backtest
===================================

Tests the POSITION COUNT version of the market regime filter.
This is the version ChatGPT recommended and we never tested.

WHAT THIS FILE TESTS
--------------------
When the market is in a bear/caution regime, reduce from 15 positions
to fewer (5, 10, or 0), keeping unused capital in cash at 6.5% annual.

This is DIFFERENT from the score penalty version (v2) which failed.
That reduced individual scores. This reduces POSITION COUNT.

REGIME DEFINITIONS TESTED
--------------------------
  1.  No filter (base, 15 stocks always)
  2.  NIFTY drawdown > 5%  → hold 10 stocks
  3.  NIFTY drawdown > 5%  → hold 5 stocks
  4.  NIFTY drawdown > 8%  → hold 10 stocks
  5.  NIFTY drawdown > 8%  → hold 5 stocks
  6.  NIFTY < 200DMA       → hold 5 (immediate trigger)
  7.  NIFTY < 200DMA       → hold 0 (full cash)
  8.  Graded: Bull=15, Caution=10, Bear=5
  9.  Persistent: NIFTY < 200DMA for 2+ periods → hold 5
  10. Persistent: NIFTY < 200DMA for 3+ periods → hold 5

KEY INSIGHT BEING TESTED
-------------------------
  We know base strategy OOS MDD = -16.34% (worse than benchmark -13.70%)
  We want to reduce OOS MDD to < -13.70% (under benchmark level)
  WITHOUT losing more than 1.5pp of Jensen's alpha

USAGE
-----
  python backtest_regime.py               # full sweep + walk-forward
  python backtest_regime.py --quick       # top 5 configs only
  python backtest_regime.py --plot        # show equity curves (requires matplotlib)
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass, field
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ── Universe (same as deploy.py) ──────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TATAMOTORS","ADANIENT","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","DIVISLAB","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "BRITANNIA","BPCL","HEROMOTOCO","HINDALCO","SBILIFE",
]
SECTOR_MAP: dict[str,str] = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDUSINDBK":"BANKING","BAJAJFINSV":"NBFC","BAJFINANCE":"NBFC",
    "SHRIRAMFIN":"NBFC","HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE",
    "TCS":"IT","INFOSYS":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "RELIANCE":"ENERGY","ONGC":"ENERGY","BPCL":"ENERGY",
    "NTPC":"POWER","POWERGRID":"POWER",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "MARUTI":"AUTO","TATAMOTORS":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO",
    "EICHERMOT":"AUTO","HEROMOTOCO":"AUTO",
    "LT":"INFRA","BEL":"DEFENCE","TITAN":"CONSUMER","TRENT":"RETAIL",
    "ASIANPAINT":"PAINTS","ULTRACEMCO":"CEMENT","GRASIM":"CEMENT",
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","DIVISLAB":"PHARMA","CIPLA":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE","ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS",
    "BHARTIARTL":"TELECOM","BAJAJFINSV":"NBFC","BAJFINANCE":"NBFC",
}

RISK_FREE_MONTHLY = 0.065 / 12   # 6.5% annual — cash earns this in bear regime
TOP_N_BASE        = 15
SECTOR_CAP        = 3
EXIT_BUFFER       = 15.0
MIN_SCORE         = 40.0
MIN_HISTORY       = 210
COST_PCT          = 0.002         # 0.2% round trip

# ── Signal (identical to deploy.py) ──────────────────────────────────────────
def _ema(a, n):
    k=2/(n+1); o=np.zeros(len(a)); o[n-1]=a[:n].mean()
    for i in range(n, len(a)): o[i]=a[i]*k+o[i-1]*(1-k)
    return o

def score_stock(c, h, lo, bench):
    if len(c) < MIN_HISTORY: return -1.0
    p=c[-1]; d25=c[-25:].mean(); d50=c[-50:].mean(); d200=c[-200:].mean(); s=0.0
    if p>d25>d50>d200:         s+=20
    elif d50>d200 and p>d200:  s+=12
    elif d50>d200:             s+=6
    if len(bench)>=90:
        rs=(c[-1]/c[-90] - bench[-1]/bench[-90])*100
        if rs>10:   s+=25
        elif rs>5:  s+=18
        elif rs>0:  s+=10
        elif rs<-5: s-=15
    if len(c)>=66:
        roc=(c[-1]/c[-66]-1)*100
        rng=(h[-14:]-lo[-14:]).mean() if len(h)>=14 else c[-20:].std()
        ap=rng/p*100 if p>0 else 2.0
        vam=roc/ap if ap>0.01 else 0
        if vam>3:   s+=20
        elif vam>2: s+=15
        elif vam>1: s+=8
        elif vam>0: s+=3
    if len(c)>=252:
        p52=(p/c[-252:].max()-1)*100
        if -5<=p52<=0: s+=6
    if len(c)>=200:
        e5=_ema(c,5); e20=_ema(c,20); e50=_ema(c,50); e200=_ema(c,200)
        s+=sum([e5[-1]>e20[-1], e20[-1]>e50[-1], e50[-1]>e200[-1]])*4
    if len(c)>=30:
        if c[-1]/c[-10]>1 and c[-1]/c[-10]>c[-10]/c[-20]: s+=7
    return min(float(s), 100.0)

# ── Regime detection ──────────────────────────────────────────────────────────
@dataclass
class RegimeCfg:
    name:             str
    description:      str
    # Drawdown-based trigger
    dd_threshold:     float = 0.0    # 0 = disabled. e.g. 0.05 = 5% below peak
    dd_n_stocks:      int   = 15     # stocks to hold when drawdown triggered
    # 200DMA-based trigger
    dma_trigger:      bool  = False  # True = NIFTY < 200DMA triggers
    dma_n_stocks:     int   = 5      # stocks in bear regime
    dma_persistent:   int   = 1      # must be below 200DMA for N periods
    # Graded (3-tier)
    graded:           bool  = False  # True = use 3-tier graded system
    # For graded: Bull=15, Caution (NIFTY 0-5% below 200DMA)=10, Bear (>5% below)=5

def detect_regime_n_stocks(nifty_prices, nifty_200dma, period_idx, cfg, below_200_history):
    """
    Determine how many stocks to hold this period.
    All detection uses PREVIOUS period's data — no look-ahead.
    Returns: (n_stocks_to_hold, regime_label)
    """
    if period_idx < 1:
        return TOP_N_BASE, "BULL"

    p  = nifty_prices[period_idx - 1]   # yesterday's price (no look-ahead)
    d200 = nifty_200dma[period_idx - 1]

    # --- GRADED regime ---
    if cfg.graded:
        pct_vs_200 = (p / d200 - 1) * 100 if d200 > 0 else 0
        if pct_vs_200 >= -3:         return 15, "BULL"
        elif pct_vs_200 >= -8:       return 10, "CAUTION"
        else:                        return 5,  "BEAR"

    # --- 200DMA trigger ---
    if cfg.dma_trigger:
        below = p < d200
        if cfg.dma_persistent > 1:
            # Must be below 200DMA for N consecutive periods
            below = sum(below_200_history[-cfg.dma_persistent:]) >= cfg.dma_persistent
        if below:
            return cfg.dma_n_stocks, "BEAR"
        return TOP_N_BASE, "BULL"

    # --- Drawdown trigger ---
    if cfg.dd_threshold > 0:
        # Compute current drawdown from recent peak
        recent = nifty_prices[max(0, period_idx-252):period_idx]
        if len(recent) > 0:
            peak = recent.max()
            dd = (p / peak - 1)
            if dd < -cfg.dd_threshold:
                return cfg.dd_n_stocks, f"DRAWDOWN({dd*100:.1f}%)"

    return TOP_N_BASE, "BULL"


# ── Core backtest ──────────────────────────────────────────────────────────────
def run(regime_cfg: RegimeCfg, cl, hi, lo, vo, si=None, ei=None) -> dict:
    bench_s = cl["^NSEI"].dropna()
    ba = bench_s.values.astype(float)
    bd = bench_s.index
    si = si or MIN_HISTORY
    ei = ei or len(ba) - 21

    # Pre-compute NIFTY 200DMA
    nifty_200dma = np.array([
        ba[max(0,i-200):i].mean() if i >= 200 else ba[:i].mean() if i > 0 else ba[0]
        for i in range(len(ba))
    ])

    pn=[1.0]; bn=[1.0]; pd_=[]
    cur:set=set(); entry_p:dict={}; entry_s:dict={}
    to_list=[]; tlog=[]; ppr=[]; pbr=[]
    regime_history=[]
    below_200_hist=[]  # for persistent regime detection

    for day in range(si, ei, 21):
        # Detect regime (uses previous period data only)
        below_200_now = ba[day-1] < nifty_200dma[day-1] if day >= 1 else False
        below_200_hist.append(below_200_now)
        n_to_hold, regime_label = detect_regime_n_stocks(
            ba, nifty_200dma, day, regime_cfg, below_200_hist
        )
        regime_history.append(regime_label)

        # Score stocks
        scores: dict[str,float] = {}
        for sym in NIFTY50:
            col = sym+".NS"
            if col not in cl.columns: continue
            c_=cl[col].reindex(bd).ffill().iloc[:day].dropna().values.astype(float)
            if len(c_)<MIN_HISTORY: continue
            av=vo[col].reindex(bd).fillna(0).iloc[max(0,day-20):day].mean() if col in vo.columns else 0
            if av*c_[-1]<5e7: continue
            h_=hi[col].reindex(bd).ffill().values[:day].astype(float)
            l_=lo[col].reindex(bd).ffill().values[:day].astype(float)
            sc=score_stock(c_,h_,l_,ba[:day])
            if sc>=MIN_SCORE: scores[sym]=sc

        # Select top n_to_hold with sector cap
        ranked=sorted(scores.items(),key=lambda x:-x[1])
        new:set=set(); sc_cnt:dict={}
        for sym,sc in ranked:
            sec=SECTOR_MAP.get(sym,"OTHER")
            if sc_cnt.get(sec,0)<SECTOR_CAP:
                new.add(sym); sc_cnt[sec]=sc_cnt.get(sec,0)+1
            if len(new)>=n_to_hold: break

        # Exit buffer
        stay=set()
        for sym in cur:
            if sym in new: stay.add(sym)
            elif sym in scores and scores[sym]>=entry_s.get(sym,0)-EXIT_BUFFER:
                # BUT: if regime reduced n_to_hold, don't keep more than n_to_hold
                if len(new|stay) < int(n_to_hold * 1.2):
                    stay.add(sym)
        final = new | stay

        exits=cur-final; enters=final-cur
        nxt=min(day+21, len(ba)-1)

        if cur: to_list.append((len(exits)+len(enters))/(2*max(len(final),1)))

        # Log exits
        for sym in exits:
            col=sym+".NS"
            if col not in cl.columns: continue
            cr=cl[col].reindex(bd).ffill()
            ep=float(cr.iloc[day]) if day<len(cr) else np.nan
            if np.isnan(ep): continue
            enp=entry_p.get(sym,ep)
            tlog.append({"symbol":sym,"return_pct":(ep/enp-1)*100 if enp>0 else 0})

        for sym in enters:
            col=sym+".NS"
            if col in cl.columns:
                cr=cl[col].reindex(bd).ffill()
                entry_p[sym]=float(cr.iloc[day]) if day<len(cr) else 0
                entry_s[sym]=scores.get(sym,0)

        # Forward returns
        fwd=[]; cash_weight=max(0, 1 - len(final)/TOP_N_BASE)
        for sym in final:
            col=sym+".NS"
            if col not in cl.columns: continue
            cr=cl[col].reindex(bd).ffill()
            p0=float(cr.iloc[day]); p1=float(cr.iloc[nxt])
            if np.isnan(p0) or np.isnan(p1) or p0<=0: continue
            fwd.append(p1/p0-1)

        to_frac=(len(exits)+len(enters))/(2*max(len(final),1)) if final else 0
        equity_ret = np.mean(fwd) if fwd else 0
        # Cash component earns risk-free
        stock_weight = 1 - cash_weight
        pr = stock_weight*equity_ret + cash_weight*RISK_FREE_MONTHLY - to_frac*COST_PCT
        br = ba[nxt]/ba[day]-1

        ppr.append(pr); pbr.append(br)
        pn.append(pn[-1]*(1+pr)); bn.append(bn[-1]*(1+br))
        pd_.append(bd[nxt]); cur=final

    pr=np.array(ppr); br_=np.array(pbr)
    pna=np.array(pn);  bna=np.array(bn)
    ppy=252/21; ny=len(pr)/ppy if ppy else 1

    cp=(pna[-1]/pna[0])**(1/ny)-1 if ny>0 else 0
    cb=(bna[-1]/bna[0])**(1/ny)-1 if ny>0 else 0
    rf=0.065/12; ex=pr-rf
    sh=(ex.mean()/ex.std()*np.sqrt(ppy)) if ex.std()>0 else 0
    ds=ex[ex<0].std()*np.sqrt(ppy)
    so=(ex.mean()*ppy/ds) if ds>0 else 0
    pk=np.maximum.accumulate(pna); mdd=((pna-pk)/pk).min()
    bpk=np.maximum.accumulate(bna); bmdd=((bna-bpk)/bna).min()
    cal=cp/abs(mdd) if mdd<0 else 0
    cov=np.cov(pr,br_); beta=cov[0,1]/cov[1,1] if cov[1,1]>0 else 1
    ja=(cp-0.065)-beta*(cb-0.065)
    hm=(pr>br_).mean()
    tr=[t["return_pct"] for t in tlog]
    ht=(np.array(tr)>0).mean() if tr else 0
    ato=np.mean(to_list)*100 if to_list else 0

    bull_pct  = sum(1 for r in regime_history if r=="BULL")/len(regime_history)*100 if regime_history else 100
    bear_pct  = 100 - bull_pct

    return dict(
        name=regime_cfg.name, description=regime_cfg.description,
        port_nav=pna, bench_nav=bna, dates=pd_,
        cagr=round(cp*100,2), bench_cagr=round(cb*100,2),
        alpha=round((cp-cb)*100,2), j_alpha=round(ja*100,2),
        beta=round(float(beta),3), sharpe=round(sh,2), sortino=round(so,2),
        mdd=round(mdd*100,2), bmdd=round(bmdd*100,2), calmar=round(cal,2),
        hit_m=round(hm*100,1), hit_t=round(ht*100,1),
        ann_to=round(ato*ppy,0), n_periods=len(pr), n_years=round(ny,1),
        bull_pct=round(bull_pct,1), bear_pct=round(bear_pct,1),
    )

# ── All regime configurations ─────────────────────────────────────────────────
ALL_REGIMES = [
    RegimeCfg("base",           "No filter — 15 stocks always"),
    RegimeCfg("dd5_10",         "Drawdown >5% → 10 stocks",  dd_threshold=0.05, dd_n_stocks=10),
    RegimeCfg("dd5_5",          "Drawdown >5% → 5 stocks",   dd_threshold=0.05, dd_n_stocks=5),
    RegimeCfg("dd8_10",         "Drawdown >8% → 10 stocks",  dd_threshold=0.08, dd_n_stocks=10),
    RegimeCfg("dd8_5",          "Drawdown >8% → 5 stocks",   dd_threshold=0.08, dd_n_stocks=5),
    RegimeCfg("dd12_5",         "Drawdown >12% → 5 stocks",  dd_threshold=0.12, dd_n_stocks=5),
    RegimeCfg("200dma_5",       "NIFTY <200DMA → 5 stocks",  dma_trigger=True, dma_n_stocks=5),
    RegimeCfg("200dma_0",       "NIFTY <200DMA → CASH",      dma_trigger=True, dma_n_stocks=0),
    RegimeCfg("200dma_p2_5",    "NIFTY <200DMA 2 periods → 5", dma_trigger=True, dma_n_stocks=5, dma_persistent=2),
    RegimeCfg("200dma_p3_5",    "NIFTY <200DMA 3 periods → 5", dma_trigger=True, dma_n_stocks=5, dma_persistent=3),
    RegimeCfg("graded",         "Bull=15, Caution=10, Bear=5", graded=True),
]

QUICK_REGIMES = [
    ALL_REGIMES[0],   # base
    ALL_REGIMES[4],   # dd8_5
    ALL_REGIMES[6],   # 200dma_5
    ALL_REGIMES[9],   # persistent_3
    ALL_REGIMES[10],  # graded
]

# ── Data load ─────────────────────────────────────────────────────────────────
def load_data():
    import yfinance as yf
    tickers=[s+".NS" if not s.startswith("^") else s for s in NIFTY50+["^NSEI"]]
    print(f"  Downloading {len(tickers)} tickers (6y history)...")
    raw=yf.download(tickers,period="6y",progress=False,auto_adjust=True,threads=True)
    if isinstance(raw.columns,pd.MultiIndex):
        cl=raw["Close"]; hi=raw["High"]; lo=raw["Low"]; vo=raw["Volume"]
    else:
        cl=raw[["Close"]]; hi=raw[["High"]]; lo=raw[["Low"]]; vo=raw[["Volume"]]
        cl.columns=hi.columns=lo.columns=vo.columns=[tickers[0]]
    return cl,hi,lo,vo

# ── Print results ─────────────────────────────────────────────────────────────
def print_sweep(results: list[dict]):
    print(f"\n{'='*100}")
    print("REGIME FILTER SWEEP — full comparison")
    print(f"{'='*100}")
    print(f"\n{'Config':<22} {'JAlpha':>7} {'Sharpe':>7} {'Sortino':>8} {'MDD%':>7} {'Calmar':>7} {'HitM%':>6} {'Bear%':>7} {'Turn%':>7}")
    print("─"*90)

    base = next(r for r in results if r['name']=='base')
    for r in results:
        is_base = r['name']=='base'
        mdd_better = r['mdd'] > base['mdd']  # less negative = better
        alpha_ok   = r['j_alpha'] >= base['j_alpha'] - 1.5
        winner     = " ★" if mdd_better and alpha_ok and not is_base else ""
        marker     = " (base)" if is_base else winner
        print(f"  {r['name']:<20} {r['j_alpha']:>7.2f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['mdd']:>7.2f} {r['calmar']:>7.2f} {r['hit_m']:>6.1f} "
              f"{r['bear_pct']:>7.1f} {r['ann_to']:>7.0f}{marker}")

    print(f"\n  ★ = better MDD than base AND alpha drop < 1.5pp")
    print(f"\nBase MDD: {base['mdd']:.2f}%  |  Base J.Alpha: {base['j_alpha']:.2f}%")

def print_walkforward(results_is: list[dict], results_oos: list[dict]):
    base_is  = next(r for r in results_is  if r['name']=='base')
    base_oos = next(r for r in results_oos if r['name']=='base')

    print(f"\n{'='*100}")
    print("WALK-FORWARD VALIDATION — in-sample vs out-of-sample")
    print(f"{'='*100}")
    print(f"\n  In-sample NIFTY:      {base_is['bench_cagr']:.2f}% CAGR")
    print(f"  Out-of-sample NIFTY:  {base_oos['bench_cagr']:.2f}% CAGR\n")

    print(f"{'Config':<22} {'IS Alpha':>9} {'OOS Alpha':>10} {'IS MDD':>8} {'OOS MDD':>9} {'BenchOOS MDD':>13} {'Verdict'}")
    print("─"*95)

    bench_oos_mdd = base_oos['bmdd']
    for r_is, r_oos in zip(results_is, results_oos):
        mdd_under_bench = r_oos['mdd'] > bench_oos_mdd  # portfolio MDD less severe than bench
        alpha_ok = r_oos['j_alpha'] >= base_oos['j_alpha'] - 1.5
        is_base = r_is['name']=='base'
        if is_base:
            verdict = "(base)"
        elif mdd_under_bench and alpha_ok:
            verdict = "★ MDD < benchmark, alpha OK"
        elif mdd_under_bench:
            verdict = "~ MDD < benchmark, alpha cost"
        elif alpha_ok:
            verdict = "~ Good alpha, MDD not fixed"
        else:
            verdict = "✗ Both worse"
        print(f"  {r_is['name']:<20} {r_is['j_alpha']:>9.2f} {r_oos['j_alpha']:>10.2f} "
              f"{r_is['mdd']:>8.2f} {r_oos['mdd']:>9.2f} {bench_oos_mdd:>13.2f}  {verdict}")

    print(f"\n  ★ = OOS MDD less severe than NIFTY AND alpha drop < 1.5pp (the goal)")
    print(f"  Benchmark OOS MDD: {bench_oos_mdd:.2f}%  |  Base OOS MDD: {base_oos['mdd']:.2f}%")

    # Best config analysis
    winners = [(r_is,r_oos) for r_is,r_oos in zip(results_is,results_oos)
               if r_oos['mdd'] > bench_oos_mdd
               and r_oos['j_alpha'] >= base_oos['j_alpha'] - 1.5
               and r_is['name'] != 'base']
    if winners:
        best_is, best_oos = max(winners, key=lambda x: x[1]['j_alpha'])
        print(f"\n{'='*70}")
        print(f"BEST CONFIG: {best_is['name']} — {best_is['description']}")
        print(f"{'='*70}")
        print(f"  Full-period:     J.Alpha {best_is['j_alpha']:.2f}%  Sharpe {best_is['sharpe']:.2f}  MDD {best_is['mdd']:.2f}%")
        print(f"  In-sample:       J.Alpha {best_is['j_alpha']:.2f}%  Sharpe {best_is['sharpe']:.2f}  MDD {best_is['mdd']:.2f}%")
        print(f"  Out-of-sample:   J.Alpha {best_oos['j_alpha']:.2f}%  Sharpe {best_oos['sharpe']:.2f}  MDD {best_oos['mdd']:.2f}%")
        print(f"  Bear regime:     {best_oos['bear_pct']:.1f}% of periods in reduced mode")
        alpha_cost = base_oos['j_alpha'] - best_oos['j_alpha']
        mdd_improvement = base_oos['mdd'] - best_oos['mdd']
        print(f"  Alpha cost:      -{alpha_cost:.2f}pp  (from {base_oos['j_alpha']:.2f}% to {best_oos['j_alpha']:.2f}%)")
        print(f"  MDD improvement: {mdd_improvement:.2f}pp  (from {base_oos['mdd']:.2f}% to {best_oos['mdd']:.2f}%)")
        print(f"  On ₹9L sleeve:   Saves ₹{abs(mdd_improvement)/100*900000:,.0f} in worst drawdown")
        print(f"                   Costs ₹{alpha_cost/100*900000:,.0f} per year in alpha")
    else:
        print(f"\n  No config met both criteria (MDD < benchmark AND alpha within 1.5pp).")
        print(f"  Best trade-off shown in table above.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",  action="store_true", help="Run only top 5 configs")
    p.add_argument("--plot",   action="store_true", help="Plot equity curves")
    a = p.parse_args()

    regimes_to_run = QUICK_REGIMES if a.quick else ALL_REGIMES
    print(f"\nRegime filter backtest — {len(regimes_to_run)} configs")
    print(f"Loading data...")
    cl, hi, lo, vo = load_data()

    bench_s = cl["^NSEI"].dropna()
    n = len(bench_s)
    split = int(n * 0.6)

    print(f"\nRunning full-period sweep ({len(regimes_to_run)} configs)...")
    results_full = []
    for rc in regimes_to_run:
        r = run(rc, cl, hi, lo, vo)
        results_full.append(r)
        print(f"  {rc.name:<20} J.Alpha {r['j_alpha']:>6.2f}%  Sharpe {r['sharpe']:.2f}  MDD {r['mdd']:.2f}%  Bear {r['bear_pct']:.0f}%")

    print_sweep(results_full)

    print(f"\nRunning walk-forward ({len(regimes_to_run)} configs × 2 periods)...")
    results_is  = [run(rc, cl, hi, lo, vo, si=MIN_HISTORY, ei=split)  for rc in regimes_to_run]
    results_oos = [run(rc, cl, hi, lo, vo, si=split, ei=n-21)         for rc in regimes_to_run]
    print_walkforward(results_is, results_oos)

    if a.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            colors = plt.cm.tab10(np.linspace(0, 1, len(results_full)))
            for r, col in zip(results_full, colors):
                axes[0].plot(r['port_nav'], label=r['name'], color=col, linewidth=1.2)
            axes[0].set_title("Full period — equity curves"); axes[0].legend(fontsize=7)
            for r, col in zip(results_oos, colors):
                axes[1].plot(r['port_nav'], label=r['name'], color=col, linewidth=1.2)
            axes[1].set_title("Out-of-sample — equity curves"); axes[1].legend(fontsize=7)
            plt.tight_layout(); plt.savefig("regime_equity_curves.png", dpi=150)
            print("\nSaved: regime_equity_curves.png")
        except Exception as e:
            print(f"Plot failed: {e}")

if __name__ == "__main__":
    main()
