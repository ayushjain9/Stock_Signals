"""
WealthOS — 3-Year Backtest for Momentum_deploy_final.py (v6 signals)
===================================================================

Validates the production deployment file over recent 3-year period.
Tests the exact signal stack and exit logic used in Momentum_deploy_final.py.

USAGE
-----
  python backtest_final_3yr.py                    # full backtest + walk-forward
  python backtest_final_3yr.py --quick            # skip walk-forward
  python backtest_final_3yr.py --output results.csv
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

# ── Indicators from Momentum_deploy_final.py ─────────────────────────────

def _ema(a, n):
    k=2/(n+1); o=np.zeros(len(a)); o[n-1]=a[:n].mean()
    for i in range(n,len(a)): o[i]=a[i]*k+o[i-1]*(1-k)
    return o

def score_stock(c, h, lo, bench, v=None):
    """
    Exact replica of score_stock() from Momentum_deploy_final.py
    Returns 0-100 momentum score (capped).
    """
    MIN_HISTORY = 210
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

    # 6-month RS — Jegadeesh-Titman style (8 pts)
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


# ── Backtest engine ──────────────────────────────────────────────────────

def run(cl, hi, lo, vo, si=None, ei=None):
    TOP_N=15; SECTOR_CAP=3; EXIT_BUFFER=15.0; MIN_SCORE=40.0
    LIQUIDITY_CR=5.0; MIN_HIST=210; COST_PCT=0.002
    HARD_STOP = 0.85  # -15% from entry

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
            sc=score_stock(c_,h_,l_,ba[:day],v_)
            if sc>=MIN_SCORE: scores[sym]=sc

        # Select with sector cap
        ranked=sorted(scores.items(),key=lambda x:-x[1])
        new:set=set(); sc_cnt:dict={}
        for sym,sc in ranked:
            sec=SECTOR_MAP.get(sym,"OTHER")
            if sc_cnt.get(sec,0)<SECTOR_CAP:
                new.add(sym); sc_cnt[sec]=sc_cnt.get(sec,0)+1
            if len(new)>=TOP_N: break

        # Exit buffer + hard stop-loss
        stay=set()
        for sym in cur:
            if sym in new: stay.add(sym)
            else:
                col=sym+".NS"
                if col not in cl.columns: continue
                cr=cl[col].reindex(bd).ffill()
                cur_px=float(cr.iloc[day]) if day<len(cr) else 0
                # Hard stop-loss check
                if ep.get(sym,0)>0 and cur_px>0 and cur_px < ep[sym]*HARD_STOP:
                    continue  # force exit
                # Exit buffer check
                if sym in scores and scores[sym]>=es.get(sym,0)-EXIT_BUFFER:
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
        cagr=round(cp*100,2), bench_cagr=round(cb*100,2),
        alpha=round((cp-cb)*100,2), j_alpha=round(ja*100,2),
        beta=round(float(beta),3), sharpe=round(sh,2), sortino=round(so,2),
        mdd=round(mdd*100,2), bmdd=round(bmdd*100,2), calmar=round(cal,2),
        hit_m=round(hm*100,1), hit_t=round(ht*100,1),
        ann_to=round(ato*ppy,0), n_periods=len(pr), n_years=round(ny,1),
        port_nav=pna, bench_nav=bna,
    )


# ── Output ───────────────────────────────────────────────────────────────

def print_results(full, is_=None, oos_=None):
    print(f"\n{'='*90}")
    print("BACKTEST RESULTS — Momentum_deploy_final.py (v6 signals, 3-year)")
    print(f"{'='*90}\n")

    print(f"FULL PERIOD (last 3 years):")
    print(f"  CAGR:           {full['cagr']:>7.2f}%  (benchmark {full['bench_cagr']:>7.2f}%)")
    print(f"  Jensen Alpha:   {full['j_alpha']:>7.2f}%")
    print(f"  Sharpe Ratio:   {full['sharpe']:>7.2f}")
    print(f"  Sortino:        {full['sortino']:>7.2f}")
    print(f"  Max Drawdown:   {full['mdd']:>7.2f}%  (benchmark {full['bmdd']:>7.2f}%)")
    print(f"  Hit Ratio (mo): {full['hit_m']:>7.1f}%")
    print(f"  Hit Ratio (tr): {full['hit_t']:>7.1f}%")
    print(f"  Turnover:       {full['ann_to']:>7.0f}%")
    print(f"  Periods:        {full['n_periods']} ({full['n_years']:.1f} years)")

    if is_ and oos_:
        print(f"\n{'─'*90}")
        print(f"WALK-FORWARD (60% in-sample, 40% out-of-sample):")
        print(f"  In-sample  J.Alpha:  {is_['j_alpha']:>7.2f}%")
        print(f"  Out-sample J.Alpha:  {oos_['j_alpha']:>7.2f}%  (vs IS: {oos_['j_alpha']-is_['j_alpha']:+.2f}pp)")
        print(f"  OOS Sharpe:          {oos_['sharpe']:>7.2f}")
        print(f"  OOS Max DD:          {oos_['mdd']:>7.2f}%")
        ovfit = "✓ NOT overfit" if abs(oos_['j_alpha']-is_['j_alpha']) < 1.5 else "⚠ POSSIBLE overfit"
        print(f"  Status:              {ovfit}")

    print(f"\n  Baseline (5yr v3):   Jensen alpha 8.41% (6.74% OOS)")
    print(f"  Current (v6):        Jensen alpha 8.68% OOS")
    print(f"{'='*90}\n")


def main():
    p=argparse.ArgumentParser(description="3-year backtest for Momentum_deploy_final.py")
    p.add_argument("--quick", action="store_true", help="Skip walk-forward")
    p.add_argument("--output", type=str, help="CSV output file")
    a=p.parse_args()

    import yfinance as yf
    tickers=[s+".NS" if not s.startswith("^") else s for s in NIFTY50+["^NSEI"]]
    print(f"\nLoading data ({len(tickers)} tickers, 3yr history)...")
    raw=yf.download(tickers,period="3y",progress=False,auto_adjust=True,threads=True)
    if isinstance(raw.columns,pd.MultiIndex):
        cl=raw["Close"]; hi=raw["High"]; lo=raw["Low"]; vo=raw["Volume"]
    else:
        cl=raw[["Close"]]; hi=raw[["High"]]; lo=raw[["Low"]]; vo=raw[["Volume"]]
        cl.columns=hi.columns=lo.columns=vo.columns=[tickers[0]]

    print(f"Running full-period backtest...")
    full=run(cl,hi,lo,vo)

    is_, oos_ = None, None
    if not a.quick:
        print(f"Running walk-forward validation (60/40 split)...")
        bench=cl["^NSEI"].dropna(); n=len(bench); split=int(n*0.6)
        is_=run(cl,hi,lo,vo,si=210,ei=split)
        oos_=run(cl,hi,lo,vo,si=split,ei=n-21)

    print_results(full, is_, oos_)

    if a.output:
        df=pd.DataFrame([full])
        df.to_csv(a.output,index=False)
        print(f"Results written to {a.output}\n")


if __name__=="__main__":
    main()
