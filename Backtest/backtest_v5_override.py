"""
WealthOS — Momentum Backtester v5 (Stock-Level Score Override)
==============================================================

What changed from v4:
  NEW  score_override: high-scoring stocks bypass dynamic sector cap
  SET  threshold=0% as default (confirmed optimal from v4 sweep)

THE PROBLEM V4 HAD:
  Dynamic sector cap reduced NBFC when trailing return went negative.
  But BAJFINANCE was scoring 82/100 that same period.
  Result: missed BAJFINANCE's +11.6% avg return.

  Same for METALS (+9.6%), RETAIL (+103.9% OOS).
  These sectors dip briefly before exploding higher.
  The trailing return signal is correct about the sector being
  temporarily weak — but wrong to exclude a strongly-signalled stock.

THE OVERRIDE RULE:
  If a stock scores >= score_override (default 70) AND would be
  blocked by the dynamic cap AND adding it doesn't exceed base_cap:
  → include it anyway (max 1 extra slot per sector per period).

  Boundary: can never exceed base_cap regardless of override.
  So with base_cap=3, dynamic_cap=1: override allows up to 2 total,
  not 3 or more. Prevents runaway sector concentration.

WHY 70 AS DEFAULT:
  Score > 70 means: brutal strength OR golden cross + high RS + positive VAM.
  These stocks have multi-timeframe confirmation. They're not just
  one-period noise. 70/100 is the "convicted momentum" zone.

USAGE
-----
  python backtest_v5_override.py                              # best config
  python backtest_v5_override.py --score-override 65          # more permissive
  python backtest_v5_override.py --score-override 75          # more selective
  python backtest_v5_override.py --score-override 999         # disabled (= v4)
  python backtest_v5_override.py --sweep                      # full param sweep
  python backtest_v5_override.py --walkforward
  python backtest_v5_override.py --universe extended --walkforward
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ── Universe & sector map ────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TATAMOTORS","ADANIENT","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","DIVISLAB","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "BRITANNIA","BPCL","HEROMOTOCO","HINDALCO","SBILIFE",
]
EXTENDED = NIFTY50 + [
    "CANBK","UNIONBANK","IDFCFIRSTB","FEDERALBNK","PNB","BANKBARODA","RBLBANK",
    "MUTHOOTFIN","CHOLAFIN","RECLTD","PFC","IRFC",
    "ZOMATO","DMART","TATACONSUM","VARUNBEV","IRCTC",
    "APOLLOHOSP","MAXHEALTH","LALPATHLAB","FORTIS",
    "DIXON","KAYNES","HAVELLS","POLYCAB","CUMMINSIND",
    "TATAPOWER","ADANIGREEN","NHPC","SJVN","TORNTPOWER",
    "SIEMENS","ABB","BHEL","BEL","HAL","BEML",
    "AMBUJACEM","ACC","RAMCOCEM",
    "PERSISTENT","COFORGE","LTTS","MPHASIS","KPIT","OFSS",
    "SAIL","HINDZINC","NMDC","VEDL","NATIONALUM",
    "PAGEIND","RADICO","MCDOWELL-N","TATACHEM","PIIND","SRF",
]
SECTOR_MAP: dict[str,str] = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDUSINDBK":"BANKING","IDFCFIRSTB":"BANKING","PNB":"BANKING",
    "BANKBARODA":"BANKING","CANBK":"BANKING","UNIONBANK":"BANKING","FEDERALBNK":"BANKING",
    "BANDHANBNK":"BANKING","RBLBANK":"BANKING",
    "BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","MUTHOOTFIN":"NBFC","CHOLAFIN":"NBFC",
    "RECLTD":"NBFC","PFC":"NBFC","IRFC":"NBFC","SHRIRAMFIN":"NBFC",
    "HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE","ICICIGI":"INSURANCE",
    "TCS":"IT","INFOSYS":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "PERSISTENT":"IT","COFORGE":"IT","LTTS":"IT","MPHASIS":"IT","KPIT":"IT","OFSS":"IT",
    "RELIANCE":"ENERGY","ONGC":"ENERGY","BPCL":"ENERGY","IOC":"ENERGY","GAIL":"ENERGY",
    "NTPC":"POWER","POWERGRID":"POWER","NHPC":"POWER","TATAPOWER":"POWER",
    "ADANIGREEN":"POWER","SJVN":"POWER","TORNTPOWER":"POWER",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "TATACONSUM":"FMCG","VARUNBEV":"FMCG","RADICO":"FMCG","PAGEIND":"FMCG",
    "MCDOWELL-N":"FMCG",
    "MARUTI":"AUTO","TATAMOTORS":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO",
    "EICHERMOT":"AUTO","HEROMOTOCO":"AUTO",
    "LT":"INFRA","SIEMENS":"INFRA","ABB":"INFRA","BHEL":"INFRA","CUMMINSIND":"INFRA",
    "HAL":"DEFENCE","BEL":"DEFENCE","BEML":"DEFENCE",
    "TITAN":"CONSUMER","TRENT":"RETAIL","DMART":"RETAIL","IRCTC":"CONSUMER",
    "ZOMATO":"CONSUMER",
    "ASIANPAINT":"PAINTS",
    "ULTRACEMCO":"CEMENT","GRASIM":"CEMENT","AMBUJACEM":"CEMENT","ACC":"CEMENT",
    "RAMCOCEM":"CEMENT",
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "SAIL":"METALS","HINDZINC":"METALS","NMDC":"METALS","VEDL":"METALS",
    "NATIONALUM":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","DIVISLAB":"PHARMA","CIPLA":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE","MAXHEALTH":"HEALTHCARE","LALPATHLAB":"HEALTHCARE",
    "FORTIS":"HEALTHCARE",
    "ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS",
    "DIXON":"ELECTRONICS","KAYNES":"ELECTRONICS","HAVELLS":"ELECTRONICS","POLYCAB":"ELECTRONICS",
    "TATACHEM":"CHEMICALS","PIIND":"CHEMICALS","SRF":"CHEMICALS",
    "BHARTIARTL":"TELECOM",
}

def build_sec_uni(symbols):
    d = {}
    for s in symbols:
        sec = SECTOR_MAP.get(s, "OTHER")
        d.setdefault(sec, []).append(s)
    return d

# ── Indicators (unchanged from v4) ───────────────────────────────────────────
def _ema(a, n):
    k=2/(n+1); o=np.zeros(len(a)); o[n-1]=a[:n].mean()
    for i in range(n,len(a)): o[i]=a[i]*k+o[i-1]*(1-k)
    return o

def momentum_score(c, h, lo, bench, min_hist=210):
    if len(c)<min_hist: return -1.0
    p=c[-1]; d25=c[-25:].mean(); d50=c[-50:].mean(); d200=c[-200:].mean()
    s=0.0
    if p>d25>d50>d200:        s+=20
    elif d50>d200 and p>d200: s+=12
    elif d50>d200:            s+=6
    if len(bench)>=90:
        rs=(c[-1]/c[-90]-bench[-1]/bench[-90])*100
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

def sector_trailing_return(sector, sec_uni, cl, bd, day, rebal_days, lookback):
    stocks = sec_uni.get(sector, [])
    if not stocks: return 0.0
    stock_avgs = []
    for sym in stocks:
        col = sym+".NS"
        if col not in cl.columns: continue
        cr = cl[col].reindex(bd).ffill()
        rets = []
        for lb in range(lookback, 0, -1):
            end = day-(lb-1)*rebal_days; start = end-rebal_days
            if start<0 or end>=len(cr): continue
            p0=float(cr.iloc[start]); p1=float(cr.iloc[end])
            if not (np.isnan(p0) or np.isnan(p1) or p0<=0):
                rets.append(p1/p0-1)
        if rets: stock_avgs.append(np.mean(rets))
    return float(np.median(stock_avgs)) if stock_avgs else 0.0

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    universe:         list
    top_n:            int   = 15
    rebal_days:       int   = 21
    min_score:        float = 40.0
    sector_cap:       int   = 3
    exit_buffer:      float = 15.0
    weak_threshold:   float = 0.0     # v4 sweep winner: any negative → reduce
    weak_slots:       int   = 1
    lookback_periods: int   = 3
    score_override:   float = 70.0    # NEW: stock score >= this bypasses cap
    risk_free:        float = 0.065
    cost_pct:         float = 0.002
    min_history:      int   = 210
    label:            str   = ""

def _load(symbols, period="6y"):
    import yfinance as yf
    tickers=[s+".NS" if not s.startswith("^") else s for s in symbols+["^NSEI"]]
    raw=yf.download(tickers,period=period,progress=False,auto_adjust=True,threads=True)
    if isinstance(raw.columns,pd.MultiIndex):
        cl=raw["Close"]; hi=raw["High"]; lo=raw["Low"]; vo=raw["Volume"]
    else:
        cl=raw[["Close"]]; hi=raw[["High"]]; lo=raw[["Low"]]; vo=raw[["Volume"]]
        cl.columns=hi.columns=lo.columns=vo.columns=[tickers[0]]
    return cl,hi,lo,vo

def run(cfg:Cfg, cl, hi, lo, vo, si=None, ei=None) -> dict:
    bench_s=cl["^NSEI"].dropna(); ba=bench_s.values.astype(float); bd=bench_s.index
    si=si or cfg.min_history; ei=ei or len(ba)-cfg.rebal_days
    sec_uni=build_sec_uni(cfg.universe)

    pn=[1.0]; bn=[1.0]; pd_=[bd[si]]
    cur:set=set(); entry_p:dict={}; entry_s:dict={}
    to_list=[]; tlog=[]; ppr=[]; pbr=[]
    sec_ct:dict={}; sec_pnl:dict={}
    override_log:dict={}  # track override fires per sector

    for day in range(si, ei, cfg.rebal_days):
        # Step 1: Score all stocks
        scores:dict={}
        for sym in cfg.universe:
            col=sym+".NS"
            if col not in cl.columns: continue
            c_=cl[col].reindex(bd).ffill().iloc[:day].dropna().values.astype(float)
            if len(c_)<cfg.min_history: continue
            av=vo[col].reindex(bd).fillna(0).iloc[max(0,day-20):day].mean() if col in vo.columns else 0
            if av*c_[-1]<5e7: continue
            h_=hi[col].reindex(bd).ffill().values[:day].astype(float)
            l_=lo[col].reindex(bd).ffill().values[:day].astype(float)
            sc=momentum_score(c_,h_,l_,ba[:day],cfg.min_history)
            if sc>=cfg.min_score: scores[sym]=sc

        # Step 2: Dynamic sector caps
        eff_caps:dict={}
        warmup = day < cfg.min_history + cfg.lookback_periods*cfg.rebal_days
        for sec in sec_uni:
            if warmup:
                eff_caps[sec]=cfg.sector_cap; continue
            tr=sector_trailing_return(sec,sec_uni,cl,bd,day,cfg.rebal_days,cfg.lookback_periods)
            eff_caps[sec]=cfg.weak_slots if tr<cfg.weak_threshold else cfg.sector_cap

        # Step 3: Select top_n — with override logic
        ranked=sorted(scores.items(),key=lambda x:-x[1])
        new:set=set(); sc_cnt:dict={}

        for sym,sc in ranked:
            sec=SECTOR_MAP.get(sym,"OTHER")
            dyn_cap=eff_caps.get(sec,cfg.sector_cap)  # dynamic cap (possibly reduced)
            base_cap=cfg.sector_cap                     # base cap (always 3)
            current=sc_cnt.get(sec,0)

            # Normal path: within dynamic cap
            if current < dyn_cap:
                new.add(sym); sc_cnt[sec]=current+1

            # Override path: blocked by dynamic cap BUT score is strong
            # AND we haven't hit the base cap yet (prevents runaway)
            elif (sc >= cfg.score_override       # stock is strongly signalled
                  and current == dyn_cap         # blocked by DYNAMIC cap (not base)
                  and current < base_cap):       # base cap not yet hit
                new.add(sym); sc_cnt[sec]=current+1
                override_log[sec]=override_log.get(sec,0)+1  # log for reporting

            if len(new)>=cfg.top_n: break

        # Step 4: Exit buffer
        if cfg.exit_buffer>0:
            stay=set()
            for sym in cur:
                if sym in new: stay.add(sym)
                elif sym in scores and scores[sym]>=entry_s.get(sym,0)-cfg.exit_buffer:
                    stay.add(sym)
            final=new|stay
            if len(final)>int(cfg.top_n*1.2):
                stays=sorted([(s,scores.get(s,0)) for s in stay-new],key=lambda x:-x[1])
                keep=set(s for s,_ in stays[:max(0,int(cfg.top_n*1.2)-len(new))])
                final=new|keep
        else:
            final=new

        exits=cur-final; enters=final-cur
        nxt=min(day+cfg.rebal_days,len(ba)-1)

        if cur: to_list.append((len(exits)+len(enters))/(2*max(len(final),1)))

        # Log exits
        for sym in exits:
            col=sym+".NS"
            if col not in cl.columns: continue
            cr=cl[col].reindex(bd).ffill()
            ep=float(cr.iloc[day]) if day<len(cr) else np.nan
            if np.isnan(ep): continue
            enp=entry_p.get(sym,ep); sec=SECTOR_MAP.get(sym,"OTHER")
            ret=(ep/enp-1)*100 if enp>0 else 0
            tlog.append({"symbol":sym,"return_pct":round(ret,2),"sector":sec})
            sec_ct[sec]=sec_ct.get(sec,0)+1
            sec_pnl.setdefault(sec,[]).append(ret)

        for sym in enters:
            col=sym+".NS"
            if col in cl.columns:
                cr=cl[col].reindex(bd).ffill()
                entry_p[sym]=float(cr.iloc[day]) if day<len(cr) else 0
                entry_s[sym]=scores.get(sym,0)

        # Forward returns
        fwd=[]
        for sym in final:
            col=sym+".NS"
            if col not in cl.columns: continue
            cr=cl[col].reindex(bd).ffill()
            p0=float(cr.iloc[day]); p1=float(cr.iloc[nxt])
            if np.isnan(p0) or np.isnan(p1) or p0<=0: continue
            fwd.append(p1/p0-1)

        to_frac=(len(exits)+len(enters))/(2*max(len(final),1)) if final else 0
        pr=(np.mean(fwd) if fwd else 0)-to_frac*cfg.cost_pct
        br=ba[nxt]/ba[day]-1
        ppr.append(pr); pbr.append(br)
        pn.append(pn[-1]*(1+pr)); bn.append(bn[-1]*(1+br))
        pd_.append(bd[nxt]); cur=final

    # ── Metrics ───────────────────────────────────────────────────────────────
    pr=np.array(ppr); br_=np.array(pbr)
    pna=np.array(pn); bna=np.array(bn)
    ppy=252/cfg.rebal_days; ny=len(pr)/ppy if ppy else 1
    cp=(pna[-1]/pna[0])**(1/ny)-1 if ny>0 else 0
    cb=(bna[-1]/bna[0])**(1/ny)-1 if ny>0 else 0
    rfp=cfg.risk_free/ppy; ex=pr-rfp
    sh=(ex.mean()/ex.std()*np.sqrt(ppy)) if ex.std()>0 else 0
    ds=ex[ex<0].std()*np.sqrt(ppy)
    so=(ex.mean()*ppy/ds) if ds>0 else 0
    pk=np.maximum.accumulate(pna); mdd=((pna-pk)/pk).min()
    bpk=np.maximum.accumulate(bna); bmdd=((bna-bpk)/bna).min()
    cal=cp/abs(mdd) if mdd<0 else 0
    cov=np.cov(pr,br_); beta=cov[0,1]/cov[1,1] if cov[1,1]>0 else 1
    ja=(cp-cfg.risk_free)-beta*(cb-cfg.risk_free)
    hm=(pr>br_).mean()
    tr=[t["return_pct"] for t in tlog]
    ht=(np.array(tr)>0).mean() if tr else 0
    ato=np.mean(to_list)*100 if to_list else 0; anto=ato*ppy; cd=anto/100*cfg.cost_pct*100
    ts=sum(sec_ct.values())
    se={k:v/ts*100 for k,v in sec_ct.items()} if ts else {}
    sp={k:round(np.mean(v),2) for k,v in sec_pnl.items()}
    n_periods=len(pr)
    ol={k:round(v/n_periods*100,1) for k,v in override_log.items()} if override_log else {}

    return dict(
        label=cfg.label or f"v5_thr{cfg.weak_threshold*100:.0f}_ovr{cfg.score_override:.0f}_buf{cfg.exit_buffer:.0f}",
        port_nav=pna, bench_nav=bna, port_dates=pd_,
        trade_log=tlog, period_pr=pr, period_br=br_,
        cagr_p=round(cp*100,2), cagr_b=round(cb*100,2),
        alpha=round((cp-cb)*100,2), j_alpha=round(ja*100,2),
        beta=round(float(beta),3), sharpe=round(sh,2), sortino=round(so,2),
        mdd=round(mdd*100,2), bmdd=round(bmdd*100,2), calmar=round(cal,2),
        hit_m=round(hm*100,1), hit_t=round(ht*100,1),
        avg_to=round(ato,1), ann_to=round(anto,0), cost_drag=round(cd,2),
        n_trades=len(tlog), n_periods=n_periods, n_years=round(ny,1),
        sector_exp=se, sector_pnl=sp, override_fires=ol,
    )

def print_r(r, verbose=True):
    print(f"\n{'='*72}\nBACKTEST — {r['label']}\n{'='*72}")
    print(f"\nRETURN──────────────────────────────")
    print(f"  Portfolio CAGR:         {r['cagr_p']:>8.2f}%")
    print(f"  Benchmark CAGR (NIFTY): {r['cagr_b']:>8.2f}%")
    print(f"  Alpha (simple):         {r['alpha']:>8.2f}%")
    print(f"  Alpha (Jensen's):       {r['j_alpha']:>8.2f}%  beta={r['beta']:.3f}")
    print(f"\nRISK────────────────────────────────")
    print(f"  Sharpe:                 {r['sharpe']:>8.2f}")
    print(f"  Sortino:                {r['sortino']:>8.2f}")
    print(f"  Max drawdown (port):    {r['mdd']:>8.2f}%")
    print(f"  Max drawdown (bench):   {r['bmdd']:>8.2f}%")
    print(f"  Calmar ratio:           {r['calmar']:>8.2f}")
    print(f"\nTRADING─────────────────────────────")
    print(f"  Hit ratio (months):     {r['hit_m']:>8.1f}%")
    print(f"  Hit ratio (trades):     {r['hit_t']:>8.1f}%  [{r['n_trades']} trades]")
    print(f"  Annual turnover:        {r['ann_to']:>8.0f}%")
    print(f"  Annual cost drag:       {r['cost_drag']:>8.2f}%")

    if verbose and r['sector_exp']:
        print(f"\nSECTOR EXPOSURE + OVERRIDE ACTIVITY────")
        pnl=r.get('sector_pnl',{}); ol=r.get('override_fires',{})
        for sec,pct in sorted(r['sector_exp'].items(),key=lambda x:-x[1]):
            bar='█'*int(pct/3)
            avg=pnl.get(sec,0)
            ovr=ol.get(sec,0)
            ovr_str=f"  override fired {ovr:.0f}% of periods" if ovr>0 else ""
            flag=" ⚠" if pct>30 else ""
            print(f"  {sec:14s}  {pct:5.1f}%  {bar:<8}  trade:{avg:>+7.1f}%{ovr_str}{flag}")

    print(f"\nINTERPRETATION──────────────────────")
    if r['j_alpha']>3:   print(f"  ✓ Real alpha {r['j_alpha']:.1f}% (Jensen's, beta-adjusted)")
    elif r['j_alpha']>0: print(f"  ~ Small alpha {r['j_alpha']:.1f}% — borderline")
    else:                print(f"  ✗ Negative Jensen's alpha")
    if r['hit_m']<50:    print(f"  ℹ Positive skew: wins bigger than it loses (expected for momentum)")
    if r['mdd']<r['bmdd']:print(f"  ✓ Portfolio drawdown smaller than benchmark")
    elif r['mdd']>r['bmdd']:print(f"  ⚠ Portfolio MDD ({r['mdd']:.1f}%) worse than benchmark ({r['bmdd']:.1f}%)")
    total_overrides=sum(r.get('override_fires',{}).values())
    if total_overrides>0:
        print(f"  ℹ Override fired across sectors — check override_fires in sector table")

def sweep(cfg:Cfg, cl, hi, lo, vo):
    print("\n"+"="*90+"\nV5 OVERRIDE SWEEP\n"+"="*90)
    print(f"\n{'Config':<52} {'JAlpha':>7} {'Sharpe':>7} {'Sortino':>8} {'MDD%':>7} {'HitM%':>6}")
    print("─"*85)
    cases=[
        # Override threshold sensitivity (threshold=0% fixed as v4 sweep winner)
        ("override=60  (permissive)",       0.0, 60),
        ("override=65",                     0.0, 65),
        ("override=70  (base)",             0.0, 70),
        ("override=75",                     0.0, 75),
        ("override=80  (selective)",        0.0, 80),
        ("override=999 (disabled = v4)",    0.0, 999),
        # Compare with different thresholds + override=70
        ("thr=-1% + override=70",          -1.0, 70),
        ("thr=0%  + override=70 (base)",    0.0, 70),
        ("thr=0%  + no override (v4)",      0.0, 999),
        # v3 baseline for reference
        ("v3 baseline (no dynamic cap)",  999.0, 999),
    ]
    for label,thr,ovr in cases:
        c=Cfg(**{**cfg.__dict__,'weak_threshold':thr/100,'score_override':float(ovr),'label':label})
        r=run(c,cl,hi,lo,vo)
        print(f"  {label:<50} {r['j_alpha']:>7.2f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['mdd']:>7.2f} {r['hit_m']:>6.1f}")

def walkforward(cfg:Cfg, cl, hi, lo, vo):
    bench=cl["^NSEI"].dropna(); n=len(bench); split=int(n*0.6)
    ri=run(cfg,cl,hi,lo,vo,si=cfg.min_history,ei=split)
    ro=run(cfg,cl,hi,lo,vo,si=split,ei=n-cfg.rebal_days)
    ri['label']="IN-SAMPLE  (train)"; ro['label']="OUT-OF-SAMPLE (test)"
    print("\n"+"="*72+"\nWALK-FORWARD VALIDATION\n"+"="*72)
    print_r(ri,verbose=True); print_r(ro,verbose=True)
    print(f"\nDEGRADATION CHECK───────────────────")
    sd=ri['sharpe']-ro['sharpe']; ad=ri['alpha']-ro['alpha']
    nifty_drop=ri['cagr_b']-ro['cagr_b']
    alpha_drop=ri['j_alpha']-ro['j_alpha']
    print(f"  Sharpe  in→out:     {ri['sharpe']:.2f} → {ro['sharpe']:.2f}  (Δ{sd:+.2f})")
    print(f"  J.Alpha in→out:     {ri['j_alpha']:.2f}% → {ro['j_alpha']:.2f}%  (Δ{alpha_drop:+.2f}pp)")
    print(f"  Hit M   in→out:     {ri['hit_m']:.1f}% → {ro['hit_m']:.1f}%")
    print(f"  NIFTY   in→out:     {ri['cagr_b']:.2f}% → {ro['cagr_b']:.2f}% (market Δ{nifty_drop:+.2f}pp)")
    ratio = alpha_drop/nifty_drop if nifty_drop!=0 else 0
    print(f"\n  Per 1pp market deterioration, alpha dropped: {ratio:.2f}pp")
    if abs(ratio)<0.5:
        print(f"  ✓ Strategy is robust — alpha barely reacts to market changes")
    else:
        print(f"  ⚠ Strategy alpha is sensitive to market environment")
    print(f"\n  OVERRIDE ACTIVITY comparison:")
    for label_str,r_ in [("IN-SAMPLE",ri),("OUT-OF-SAMPLE",ro)]:
        ol=r_.get('override_fires',{})
        if ol:
            top=sorted(ol.items(),key=lambda x:-x[1])[:5]
            pnl=r_.get('sector_pnl',{})
            print(f"\n  {label_str}:")
            for sec,pct in top:
                avg=pnl.get(sec,0)
                verdict="✓ correctly fired" if avg>2 else ("✗ fired on winner" if avg>8 else "~neutral")
                print(f"    {sec:14s}: fired {pct:.0f}% of periods  avg trade: {avg:+.1f}%  {verdict}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--universe",       choices=["nifty50","extended"],default="nifty50")
    p.add_argument("--top-n",          type=int,   default=15)
    p.add_argument("--rebal-days",     type=int,   default=21)
    p.add_argument("--min-score",      type=float, default=40)
    p.add_argument("--sector-cap",     type=int,   default=3)
    p.add_argument("--exit-buffer",    type=float, default=15)
    p.add_argument("--weak-threshold", type=float, default=0.0,
                   help="Sector trailing return threshold (default 0.0 = any negative)")
    p.add_argument("--weak-slots",     type=int,   default=1)
    p.add_argument("--lookback",       type=int,   default=3)
    p.add_argument("--score-override", type=float, default=70,
                   help="Stock score >= this bypasses dynamic cap (default 70, 999=disabled)")
    p.add_argument("--sweep",          action="store_true")
    p.add_argument("--walkforward",    action="store_true")
    a=p.parse_args()

    syms=EXTENDED if a.universe=="extended" else NIFTY50
    cfg=Cfg(
        universe=syms,top_n=a.top_n,rebal_days=a.rebal_days,
        min_score=a.min_score,sector_cap=a.sector_cap,exit_buffer=a.exit_buffer,
        weak_threshold=a.weak_threshold/100,weak_slots=a.weak_slots,
        lookback_periods=a.lookback,score_override=a.score_override,
        label=f"{a.universe} top{a.top_n} r{a.rebal_days} "
              f"sc{a.sector_cap} thr{a.weak_threshold} ovr{a.score_override:.0f} buf{a.exit_buffer:.0f}",
    )
    print(f"Loading {len(syms)} symbols...")
    cl,hi,lo,vo=_load(syms)
    r=run(cfg,cl,hi,lo,vo)
    print_r(r)
    if a.sweep:        sweep(cfg,cl,hi,lo,vo)
    if a.walkforward:  walkforward(cfg,cl,hi,lo,vo)

if __name__=="__main__": main()
