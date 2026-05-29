"""
WealthOS — Backtest v6 (Signal Improvement Test)
=================================================

PURPOSE
-------
Tests whether adding real ADX and volume surge to the confirmed
base strategy (sc3, buf15, Nifty 50) improves performance.

BASE STRATEGY (confirmed optimal, backtest_v3_final.py):
  Universe:    Nifty 50
  Top N:       15
  Sector cap:  3
  Exit buffer: 15 pts
  Signals:     DMA + RS(90D) + VAM + 52W + MTF + ADX_proxy
  Result:      Jensen's alpha 8.41%, Sharpe 0.87

WHAT WE'RE TESTING (4 variations + base = 5 configs):
  v6a:  Base (no changes) — must reproduce ~8.41% alpha as sanity check
  v6b:  + Real ADX (replaces proxy)
  v6c:  + Volume surge (new signal, 6 pts max)
  v6d:  + Graduated 52W high (cliff-edge fix)
  v6e:  + 6-month RS alongside 90D RS
  v6f:  All four changes combined

SUCCESS CRITERIA
----------------
  An addition is WORTH KEEPING if:
    Full-period J.Alpha improves vs base
    OOS J.Alpha improves or holds within 0.5pp of base
    Sharpe improves or holds
    Hit ratio doesn't drop
  
  Rejected if any of the above worsens meaningfully.
  Rule: don't change what's working unless data clearly supports it.

USAGE
-----
  python backtest_v6_signals.py           # full sweep + walk-forward
  python backtest_v6_signals.py --quick   # skip walk-forward
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass
import numpy as np, pandas as pd
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
    "BHARTIARTL":"TELECOM",
}

# ── Signal flags ────────────────────────────────────────────────────────────
@dataclass
class SignalCfg:
    name:            str
    use_real_adx:    bool  = False   # True = real ADX; False = proxy
    use_vol_surge:   bool  = False   # True = score volume surge
    use_grad_52w:    bool  = False   # True = graduated 52W; False = cliff
    use_6m_rs:       bool  = False   # True = add 6M RS signal alongside 90D

# ── Indicator library ────────────────────────────────────────────────────────
def _ema(a, n):
    k=2/(n+1); o=np.zeros(len(a)); o[n-1]=a[:n].mean()
    for i in range(n,len(a)): o[i]=a[i]*k+o[i-1]*(1-k)
    return o

def calc_adx(h, lo, c, period=14):
    """Real ADX with +DI/-DI direction. Returns (adx, pdi, ndi)."""
    if len(c) < period*3: return 0.0, 0.0, 0.0
    tr  = np.maximum.reduce([h[1:]-lo[1:], np.abs(h[1:]-c[:-1]), np.abs(lo[1:]-c[:-1])])
    pdm = np.where((h[1:]-h[:-1])>(lo[:-1]-lo[1:]), np.maximum(h[1:]-h[:-1],0), 0.0)
    ndm = np.where((lo[:-1]-lo[1:])>(h[1:]-h[:-1]), np.maximum(lo[:-1]-lo[1:],0), 0.0)
    def rma(a):
        o=np.zeros(len(a)); o[period-1]=a[:period].sum()
        for i in range(period,len(a)): o[i]=o[i-1]-o[i-1]/period+a[i]
        return o
    atr_r,pdm_r,ndm_r = rma(tr),rma(pdm),rma(ndm)
    with np.errstate(invalid='ignore', divide='ignore'):
        pdi = np.where(atr_r>0, 100*pdm_r/atr_r, 0.0)
        ndi = np.where(atr_r>0, 100*ndm_r/atr_r, 0.0)
        denom = pdi+ndi
        dx  = np.where(denom>0, 100*np.abs(pdi-ndi)/denom, 0.0)
    adx_arr = rma(np.nan_to_num(dx[period:]))
    return round(float(adx_arr[-1]),1), round(float(pdi[-1]),1), round(float(ndi[-1]),1)

def vol_surge_pts(v):
    """Score volume surge: 5D avg vs 20D avg."""
    if len(v) < 26: return 0
    avg5  = v[-5:].mean()
    avg20 = v[-25:-5].mean()
    if avg20 <= 0: return 0
    ratio = avg5 / avg20
    if ratio >= 2.0: return 6
    if ratio >= 1.5: return 4
    return 0

def pct52_grad(pct):
    """Graduated 52W high scoring — fixes the cliff edge."""
    if -2 <= pct <= 2:    return 9   # at or just above high = breakout
    if -5 <= pct < -2:   return 6   # near high (same as original)
    if -15 <= pct < -5:  return 3   # within 15% — was 0 before
    return 0

def pct52_cliff(pct):
    """Original cliff-edge scoring."""
    return 6 if -5 <= pct <= 0 else 0

def rs_90d(c, bench):
    if len(c)<90 or len(bench)<90: return 0.0
    return (c[-1]/c[-90] - bench[-1]/bench[-90])*100

def rs_6m(c, bench, skip=21):
    """6-month RS skipping last 1 month (Jegadeesh-Titman style)."""
    if len(c)<126+skip or len(bench)<126+skip: return 0.0
    return (c[-(skip+1)]/c[-(126+skip)] - bench[-(skip+1)]/bench[-(126+skip)])*100


# ── Scoring engine ───────────────────────────────────────────────────────────
def score_stock(c, h, lo, v, bench, cfg: SignalCfg) -> float:
    MIN_HISTORY = 210
    if len(c) < MIN_HISTORY: return -1.0
    p=c[-1]; d25=c[-25:].mean(); d50=c[-50:].mean(); d200=c[-200:].mean(); s=0.0

    # Signal 1: DMA alignment (20 pts) — unchanged
    if p>d25>d50>d200:        s+=20
    elif d50>d200 and p>d200: s+=12
    elif d50>d200:            s+=6

    # Signal 2: RS vs NIFTY (25 pts) — 90D always scored
    rs90 = rs_90d(c, bench)
    if rs90>10:   s+=25
    elif rs90>5:  s+=18
    elif rs90>0:  s+=10
    elif rs90<-5: s-=15

    # Signal 2b: 6-month RS — optional addition (8 pts)
    if cfg.use_6m_rs:
        rs6 = rs_6m(c, bench)
        if rs6>15:   s+=8
        elif rs6>8:  s+=5
        elif rs6>0:  s+=2
        elif rs6<-8: s-=5

    # Signal 3: VAM (20 pts) — unchanged
    if len(c)>=66:
        roc=(c[-1]/c[-66]-1)*100
        rng=(h[-14:]-lo[-14:]).mean() if len(h)>=14 else c[-20:].std()
        ap=rng/p*100 if p>0 else 2.0
        vam=roc/ap if ap>0.01 else 0
        if vam>3:   s+=20
        elif vam>2: s+=15
        elif vam>1: s+=8
        elif vam>0: s+=3

    # Signal 4: 52W high proximity (6-9 pts)
    if len(c)>=252:
        pct52=(p/c[-252:].max()-1)*100
        s += pct52_grad(pct52) if cfg.use_grad_52w else pct52_cliff(pct52)

    # Signal 5: MTF EMA alignment (12 pts) — unchanged
    if len(c)>=200:
        e5=_ema(c,5); e20=_ema(c,20); e50=_ema(c,50); e200=_ema(c,200)
        s+=sum([e5[-1]>e20[-1], e20[-1]>e50[-1], e50[-1]>e200[-1]])*4

    # Signal 6: ADX / momentum direction (7-10 pts)
    if cfg.use_real_adx and len(h)>=42 and len(lo)>=42:
        adx_val, pdi, ndi = calc_adx(h, lo, c)
        if adx_val>=40 and pdi>ndi:  s+=10
        elif adx_val>=25 and pdi>ndi: s+=7
        # If adx high but ndi>pdi (bearish direction): 0 pts, no penalty
    else:
        # Original proxy
        if len(c)>=30:
            if c[-1]/c[-10]>1 and c[-1]/c[-10]>c[-10]/c[-20]: s+=7

    # Signal 7: Volume surge (0-6 pts) — new, optional
    if cfg.use_vol_surge and len(v)>=26:
        s += vol_surge_pts(v)

    return min(float(s), 100.0)


# ── Core backtest (identical to v3_final structure) ─────────────────────────
def run(sig_cfg: SignalCfg, cl, hi, lo, vo, si=None, ei=None) -> dict:
    TOP_N=15; SECTOR_CAP=3; EXIT_BUFFER=15.0; MIN_SCORE=40.0
    MIN_HIST=210; COST_PCT=0.002

    bench_s=cl["^NSEI"].dropna(); ba=bench_s.values.astype(float); bd=bench_s.index
    si=si or MIN_HIST; ei=ei or len(ba)-21

    pn=[1.0]; bn=[1.0]; pd_=[bd[si]]
    cur:set=set(); ep:dict={}; es:dict={}
    to_list=[]; tlog=[]; ppr=[]; pbr=[]

    for day in range(si, ei, 21):
        scores:dict={}
        for sym in NIFTY50:
            col=sym+".NS"
            if col not in cl.columns: continue
            c_=cl[col].reindex(bd).ffill().iloc[:day].dropna().values.astype(float)
            if len(c_)<MIN_HIST: continue
            av=vo[col].reindex(bd).fillna(0).iloc[max(0,day-20):day].mean() if col in vo.columns else 0
            if av*c_[-1]<5e7: continue
            h_=hi[col].reindex(bd).ffill().values[:day].astype(float)
            l_=lo[col].reindex(bd).ffill().values[:day].astype(float)
            v_=vo[col].reindex(bd).fillna(0).values[:day].astype(float)
            sc=score_stock(c_,h_,l_,v_,ba[:day],sig_cfg)
            if sc>=MIN_SCORE: scores[sym]=sc

        # Select with sector cap
        ranked=sorted(scores.items(),key=lambda x:-x[1])
        new:set=set(); sc_cnt:dict={}
        for sym,sc in ranked:
            sec=SECTOR_MAP.get(sym,"OTHER")
            if sc_cnt.get(sec,0)<SECTOR_CAP:
                new.add(sym); sc_cnt[sec]=sc_cnt.get(sec,0)+1
            if len(new)>=TOP_N: break

        # Exit buffer
        stay=set()
        for sym in cur:
            if sym in new: stay.add(sym)
            elif sym in scores and scores[sym]>=es.get(sym,0)-EXIT_BUFFER:
                stay.add(sym)
        final=new|stay
        if len(final)>int(TOP_N*1.2):
            stays=sorted([(s,scores.get(s,0)) for s in stay-new],key=lambda x:-x[1])
            keep=set(s for s,_ in stays[:max(0,int(TOP_N*1.2)-len(new))])
            final=new|keep

        exits=cur-final; enters=final-cur
        nxt=min(day+21,len(ba)-1)
        if cur: to_list.append((len(exits)+len(enters))/(2*max(len(final),1)))

        for sym in exits:
            col=sym+".NS"
            if col not in cl.columns: continue
            cr=cl[col].reindex(bd).ffill()
            ep_=float(cr.iloc[day]) if day<len(cr) else np.nan
            if np.isnan(ep_): continue
            enp=ep.get(sym,ep_)
            tlog.append({"symbol":sym,"return_pct":(ep_/enp-1)*100 if enp>0 else 0})

        for sym in enters:
            col=sym+".NS"
            if col in cl.columns:
                cr=cl[col].reindex(bd).ffill()
                ep[sym]=float(cr.iloc[day]) if day<len(cr) else 0
                es[sym]=scores.get(sym,0)

        fwd=[]
        for sym in final:
            col=sym+".NS"
            if col not in cl.columns: continue
            cr=cl[col].reindex(bd).ffill()
            p0=float(cr.iloc[day]); p1=float(cr.iloc[nxt])
            if np.isnan(p0) or np.isnan(p1) or p0<=0: continue
            fwd.append(p1/p0-1)

        to_f=(len(exits)+len(enters))/(2*max(len(final),1)) if final else 0
        pr=(np.mean(fwd) if fwd else 0)-to_f*COST_PCT
        br=ba[nxt]/ba[day]-1
        ppr.append(pr); pbr.append(br)
        pn.append(pn[-1]*(1+pr)); bn.append(bn[-1]*(1+br))
        pd_.append(bd[nxt]); cur=final

    pr=np.array(ppr); br_=np.array(pbr)
    pna=np.array(pn);  bna=np.array(bn)
    ppy=252/21; ny=len(pr)/ppy if ppy else 1
    cp=(pna[-1]/pna[0])**(1/ny)-1 if ny>0 else 0
    cb=(bna[-1]/bna[0])**(1/ny)-1 if ny>0 else 0
    rfp=0.065/21/12; ex=pr-rfp
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

    return dict(
        name=sig_cfg.name,
        cagr=round(cp*100,2), bench_cagr=round(cb*100,2),
        alpha=round((cp-cb)*100,2), j_alpha=round(ja*100,2),
        beta=round(float(beta),3), sharpe=round(sh,2), sortino=round(so,2),
        mdd=round(mdd*100,2), bmdd=round(bmdd*100,2), calmar=round(cal,2),
        hit_m=round(hm*100,1), hit_t=round(ht*100,1),
        ann_to=round(ato*ppy,0), n_periods=len(pr), n_years=round(ny,1),
        port_nav=pna, bench_nav=bna,
    )


# ── Output ───────────────────────────────────────────────────────────────────
def print_sweep(results: list[dict], label="FULL PERIOD"):
    print(f"\n{'='*95}")
    print(f"SIGNAL IMPROVEMENT TEST — {label}")
    print(f"{'='*95}")
    print(f"\n{'Config':<38} {'JAlpha':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MDD%':>7} {'HitM%':>6} {'HitT%':>6} {'Turn%':>7}")
    print("─"*90)

    base = next((r for r in results if "base" in r['name'].lower()), results[0])

    for r in results:
        is_base = "base" in r['name'].lower()
        # Is it better? All four must hold or improve
        j_better  = r['j_alpha'] > base['j_alpha']
        sh_better = r['sharpe']  >= base['sharpe'] - 0.03   # allow tiny margin
        mdd_better= r['mdd']     >= base['mdd']    - 0.5    # allow tiny margin
        winner = " ★" if (j_better and sh_better and mdd_better and not is_base) else ""
        marker = " (base)" if is_base else winner

        j_diff = f"{r['j_alpha']-base['j_alpha']:+.2f}" if not is_base else ""
        print(f"  {r['name']:<36} {r['j_alpha']:>7.2f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['mdd']:>7.2f} {r['hit_m']:>6.1f} {r['hit_t']:>6.1f} {r['ann_to']:>7.0f}"
              f"  {j_diff}{marker}")

    print(f"\n  ★ = better J.Alpha + Sharpe held + MDD not worsened vs base")
    print(f"  Base J.Alpha: {base['j_alpha']:.2f}%  Sharpe: {base['sharpe']:.2f}  MDD: {base['mdd']:.2f}%")


def walkforward(configs: list[SignalCfg], cl, hi, lo, vo):
    bench=cl["^NSEI"].dropna(); n=len(bench); split=int(n*0.6)
    MIN_HIST=210
    print(f"\n{'='*95}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*95}")
    bench_is = cl["^NSEI"].dropna().iloc[MIN_HIST:split].values
    bench_oos = cl["^NSEI"].dropna().iloc[split:].values
    nifty_is_cagr  = (bench_is[-1]/bench_is[0])**(252/21/len(bench_is)*21)-1 if len(bench_is)>0 else 0
    nifty_oos_cagr = (bench_oos[-1]/bench_oos[0])**(252/21/len(bench_oos)*21)-1 if len(bench_oos)>0 else 0
    print(f"\n  In-sample NIFTY:      ~{nifty_is_cagr*100:.1f}% CAGR")
    print(f"  Out-of-sample NIFTY:  ~{nifty_oos_cagr*100:.1f}% CAGR")
    results_is  = [run(c,cl,hi,lo,vo,si=MIN_HIST,ei=split) for c in configs]
    results_oos = [run(c,cl,hi,lo,vo,si=split,ei=n-21)     for c in configs]
    print_sweep(results_is,  "IN-SAMPLE")
    print_sweep(results_oos, "OUT-OF-SAMPLE")

    base_is  = next(r for r in results_is  if "base" in r['name'].lower())
    base_oos = next(r for r in results_oos if "base" in r['name'].lower())

    print(f"\n{'Config':<38} {'IS Alpha':>9} {'OOS Alpha':>10} {'OOS Sharpe':>11} {'vs base OOS':>12}")
    print("─"*85)
    for r_is, r_oos in zip(results_is, results_oos):
        diff = r_oos['j_alpha'] - base_oos['j_alpha']
        print(f"  {r_is['name']:<36} {r_is['j_alpha']:>9.2f} {r_oos['j_alpha']:>10.2f} "
              f"{r_oos['sharpe']:>11.2f} {diff:>+12.2f}")
    print(f"\n  Positive 'vs base OOS' = signal adds value out-of-sample")
    print(f"  Negative = signal overfit to in-sample data")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Skip walk-forward")
    a=p.parse_args()

    configs = [
        SignalCfg("v6a_base",                                                     ),
        SignalCfg("v6b_real_adx",          use_real_adx=True                      ),
        SignalCfg("v6c_vol_surge",          use_vol_surge=True                     ),
        SignalCfg("v6d_grad_52w",           use_grad_52w=True                      ),
        SignalCfg("v6e_6m_rs",              use_6m_rs=True                         ),
        SignalCfg("v6f_adx+vol",            use_real_adx=True,  use_vol_surge=True ),
        SignalCfg("v6g_adx+vol+52w",        use_real_adx=True,  use_vol_surge=True,
                                            use_grad_52w=True                      ),
        SignalCfg("v6h_all",                use_real_adx=True,  use_vol_surge=True,
                                            use_grad_52w=True,  use_6m_rs=True     ),
        SignalCfg("v6i_vol+6mRS", use_vol_surge=True, use_6m_rs=True)
    ]

    import yfinance as yf
    tickers=[s+".NS" if not s.startswith("^") else s for s in NIFTY50+["^NSEI"]]
    print(f"\nLoading data ({len(tickers)} tickers, 6yr history)...")
    raw=yf.download(tickers,period="6y",progress=False,auto_adjust=True,threads=True)
    if isinstance(raw.columns,pd.MultiIndex):
        cl=raw["Close"]; hi=raw["High"]; lo=raw["Low"]; vo=raw["Volume"]
    else:
        cl=raw[["Close"]]; hi=raw[["High"]]; lo=raw[["Low"]]; vo=raw[["Volume"]]
        cl.columns=hi.columns=lo.columns=vo.columns=[tickers[0]]

    print(f"Running {len(configs)} signal configurations (full period)...")
    results=[]
    for cfg in configs:
        r=run(cfg,cl,hi,lo,vo)
        results.append(r)
        print(f"  {cfg.name:<35} J.Alpha {r['j_alpha']:>6.2f}%  "
              f"Sharpe {r['sharpe']:.2f}  MDD {r['mdd']:.2f}%")

    print_sweep(results)

    if not a.quick:
        print("\nRunning walk-forward validation...")
        walkforward(configs, cl, hi, lo, vo)

    # Final recommendation
    base = next(r for r in results if "base" in r['name'].lower())
    winners = [r for r in results if r['j_alpha']>base['j_alpha']
               and r['sharpe']>=base['sharpe']-0.03
               and r['mdd']>=base['mdd']-0.5
               and "base" not in r['name'].lower()]
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    if winners:
        best = max(winners, key=lambda r: r['j_alpha'])
        print(f"  Best config: {best['name']}")
        print(f"  J.Alpha improvement: {best['j_alpha']-base['j_alpha']:+.2f}pp")
        print(f"  Sharpe improvement:  {best['sharpe']-base['sharpe']:+.2f}")
        print(f"  MDD change:          {best['mdd']-base['mdd']:+.2f}pp")
        print(f"\n  IF walk-forward confirms this holds OOS → update deploy.py")
        print(f"  IF OOS alpha drops more than 1pp vs base OOS → reject")
    else:
        print("  No config improved on all three metrics vs base.")
        print("  Keep deploy.py unchanged. Base strategy is optimal.")

if __name__=="__main__":
    main()
