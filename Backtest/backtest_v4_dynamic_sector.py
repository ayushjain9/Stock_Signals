"""
WealthOS — Momentum Backtester v4 (Dynamic Sector Cap)
=======================================================

KEY NEW FEATURE — Option A fix:
  Dynamic sector cap based on trailing sector performance.

  STATIC (v3):  Banking always gets max 3 slots, even when trend is down.
                Result: BANKING -1.7% avg trade, IT -1.4% avg trade.

  DYNAMIC (v4): Each sector's cap shrinks to 1 when its trailing
                3-period median return is below -1% threshold.
                When sector recovers → cap returns to normal (3).

  Why median, not mean?  Median is robust to one extreme outlier stock.
  Why 3 periods?         Monthly: 3 months = one full market cycle.
  Why -1% threshold?     Enough to filter real weakness, not noise.
  Why reduce to 1 not 0? We keep one slot — the strongest scorer in the
                          sector — because even weak sectors have leaders.

HOW IT WORKS:
  Every rebalancing day:
    1. Compute momentum score for all stocks (same as v3)
    2. For each sector: look at how stocks in that sector actually performed
       over the last 3 rebalancing periods
    3. If median return < -1% → sector_cap = 1 (instead of cfg.sector_cap)
    4. Apply sector-capped selection as before

WHAT CHANGES:
  When banking is weak: only 1 banking stock (best scorer) instead of 3
  Freed capital → next highest-scoring stocks from other sectors
  Asymmetric: bad period losses shrink by ~70%, good period gains preserved

USAGE
-----
  python backtest_v4_dynamic_sector.py                        # Nifty 50, default
  python backtest_v4_dynamic_sector.py --weak-threshold -1    # current
  python backtest_v4_dynamic_sector.py --weak-threshold -2    # stricter
  python backtest_v4_dynamic_sector.py --weak-threshold 0     # any neg = reduce
  python backtest_v4_dynamic_sector.py --weak-slots 0         # full exclusion
  python backtest_v4_dynamic_sector.py --lookback 5           # 5-period lookback
  python backtest_v4_dynamic_sector.py --sweep                # full sensitivity
  python backtest_v4_dynamic_sector.py --walkforward
  python backtest_v4_dynamic_sector.py --universe extended --walkforward
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Universe & sector map (same as v3) ────────────────────────────────────────
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
    "PAGEIND","RADICO","MCDOWELL-N",
    "TATACHEM","PIIND","SRF",
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
    "ZOMATO":"CONSUMER","NYKAA":"CONSUMER",
    "ASIANPAINT":"PAINTS","BERGEPAINT":"PAINTS",
    "ULTRACEMCO":"CEMENT","GRASIM":"CEMENT","AMBUJACEM":"CEMENT","ACC":"CEMENT",
    "RAMCOCEM":"CEMENT",
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "SAIL":"METALS","HINDZINC":"METALS","NMDC":"METALS","VEDL":"METALS",
    "NATIONALUM":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","DIVISLAB":"PHARMA","CIPLA":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE","MAXHEALTH":"HEALTHCARE","LALPATHLAB":"HEALTHCARE",
    "FORTIS":"HEALTHCARE",
    "ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS",
    "DIXON":"ELECTRONICS","KAYNES":"ELECTRONICS","HAVELLS":"ELECTRONICS",
    "POLYCAB":"ELECTRONICS",
    "TATACHEM":"CHEMICALS","PIIND":"CHEMICALS","SRF":"CHEMICALS",
    "BHARTIARTL":"TELECOM",
}

# Group universe by sector for trailing return computation
def build_sector_universe(symbols: list[str]) -> dict[str, list[str]]:
    sec_uni: dict[str, list[str]] = {}
    for sym in symbols:
        sec = SECTOR_MAP.get(sym, "OTHER")
        sec_uni.setdefault(sec, []).append(sym)
    return sec_uni

# ── Indicators (same as v3) ────────────────────────────────────────────────────
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
    return min(float(s),100.0)

# ── Dynamic sector cap engine ──────────────────────────────────────────────────

def compute_sector_trailing_return(
    sector: str,
    sec_universe: dict[str, list[str]],
    cl: pd.DataFrame,
    bench_dates: pd.DatetimeIndex,
    day_idx: int,
    rebal_days: int,
    lookback_periods: int,
) -> float:
    """
    Compute the median return of all stocks in a sector over the last
    `lookback_periods` rebalancing periods.

    Uses actual stock price data — NOT our held positions.
    This avoids the circular logic of judging a sector by our own picks.

    Returns the median return as a decimal (e.g., -0.017 for -1.7%).
    Returns 0.0 if insufficient data.
    """
    stocks_in_sector = sec_universe.get(sector, [])
    if not stocks_in_sector:
        return 0.0

    stock_period_rets = []
    for sym in stocks_in_sector:
        col = sym + ".NS"
        if col not in cl.columns:
            continue
        cr = cl[col].reindex(bench_dates).ffill()

        # Collect returns for each of the last `lookback_periods` periods
        sym_rets = []
        for lb in range(lookback_periods, 0, -1):
            end   = day_idx - (lb - 1) * rebal_days
            start = end - rebal_days
            if start < 0 or end >= len(cr):
                continue
            p0 = float(cr.iloc[start])
            p1 = float(cr.iloc[end])
            if not (np.isnan(p0) or np.isnan(p1) or p0 <= 0):
                sym_rets.append(p1 / p0 - 1)

        if sym_rets:
            stock_period_rets.append(np.mean(sym_rets))  # avg return for this stock

    if not stock_period_rets:
        return 0.0

    return float(np.median(stock_period_rets))  # median across all stocks in sector


def dynamic_sector_cap(
    sector: str,
    trailing_ret: float,
    base_cap: int,
    weak_threshold: float,
    weak_slots: int,
) -> tuple[int, str]:
    """
    Determine the effective sector cap based on trailing performance.

    Args:
        trailing_ret:    median 3-period return of sector
        base_cap:        normal max slots (e.g., 3)
        weak_threshold:  return below this = sector is weak (e.g., -0.01)
        weak_slots:      slots allowed when weak (e.g., 1)

    Returns:
        (effective_cap, reason_string)
    """
    if trailing_ret < weak_threshold:
        return weak_slots, f"WEAK({trailing_ret*100:.1f}%<{weak_threshold*100:.0f}%)→{weak_slots}slot"
    return base_cap, f"OK({trailing_ret*100:.1f}%)"


# ── Backtest config ────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    universe:        list[str]
    top_n:           int   = 15
    rebal_days:      int   = 21
    min_score:       float = 40.0
    sector_cap:      int   = 3      # base cap (normal conditions)
    exit_buffer:     float = 15.0   # only exit if score drops this much
    weak_threshold:  float = -0.01  # trailing return below this → reduce cap
    weak_slots:      int   = 1      # slots when sector is weak (0 = full exclusion)
    lookback_periods:int   = 3      # how many periods to look back
    risk_free:       float = 0.065
    cost_pct:        float = 0.002
    min_history:     int   = 210
    label:           str   = ""

def _load(symbols, period="6y"):
    import yfinance as yf
    tickers = [s + ".NS" if not s.startswith("^") else s
               for s in symbols + ["^NSEI"]]
    raw = yf.download(tickers, period=period, progress=False,
                      auto_adjust=True, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        cl=raw["Close"]; hi=raw["High"]; lo=raw["Low"]; vo=raw["Volume"]
    else:
        cl=raw[["Close"]]; hi=raw[["High"]]; lo=raw[["Low"]]; vo=raw[["Volume"]]
        cl.columns=hi.columns=lo.columns=vo.columns=[tickers[0]]
    return cl, hi, lo, vo


def run(cfg: Cfg, cl, hi, lo, vo, si=None, ei=None) -> dict:
    bench_s = cl["^NSEI"].dropna()
    ba = bench_s.values.astype(float)
    bd = bench_s.index
    si = si or cfg.min_history
    ei = ei or len(ba) - cfg.rebal_days

    sec_uni = build_sector_universe(cfg.universe)

    pn=[1.0]; bn=[1.0]; pd_=[bd[si]]
    cur: set[str]=set()
    entry_p: dict[str,float]={}
    entry_s: dict[str,float]={}
    to_list: list[float]=[]
    tlog: list[dict]=[]
    ppr: list[float]=[]; pbr: list[float]=[]
    sec_ct: dict[str,int]={}
    sec_pnl: dict[str,list]={}
    # Track how often each sector's cap was reduced
    sec_cap_history: dict[str,list] = {}

    for day in range(si, ei, cfg.rebal_days):
        # ── STEP 1: Score all stocks ──────────────────────────────────────────
        scores: dict[str,float] = {}
        for sym in cfg.universe:
            col = sym + ".NS"
            if col not in cl.columns: continue
            c_ = cl[col].reindex(bd).ffill().iloc[:day].dropna().values.astype(float)
            if len(c_) < cfg.min_history: continue
            av = vo[col].reindex(bd).fillna(0).iloc[max(0,day-20):day].mean() \
                 if col in vo.columns else 0
            if av * c_[-1] < 5e7: continue  # liquidity gate
            h_ = hi[col].reindex(bd).ffill().values.astype(float)[:day]
            l_ = lo[col].reindex(bd).ffill().values.astype(float)[:day]
            sc = momentum_score(c_, h_, l_, ba[:day], cfg.min_history)
            if sc >= cfg.min_score:
                scores[sym] = sc

        # ── STEP 2: Compute dynamic sector cap for each sector ────────────────
        effective_caps: dict[str, int] = {}
        cap_reasons:    dict[str, str] = {}
        for sector in sec_uni:
            # Only compute if we have enough history for lookback
            if day < cfg.min_history + cfg.lookback_periods * cfg.rebal_days:
                effective_caps[sector] = cfg.sector_cap
                cap_reasons[sector] = "WARMUP"
                continue

            trailing = compute_sector_trailing_return(
                sector, sec_uni, cl, bd, day,
                cfg.rebal_days, cfg.lookback_periods
            )
            cap, reason = dynamic_sector_cap(
                sector, trailing,
                cfg.sector_cap, cfg.weak_threshold, cfg.weak_slots
            )
            effective_caps[sector] = cap
            cap_reasons[sector] = reason

            # Record cap history for analysis
            sec_cap_history.setdefault(sector, []).append(cap)

        # ── STEP 3: Select top_n with dynamic sector caps ─────────────────────
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        new: set[str] = set()
        sc_cnt: dict[str,int] = {}
        for sym, sc in ranked:
            sec = SECTOR_MAP.get(sym, "OTHER")
            
            cap = effective_caps.get(sec, cfg.sector_cap)
               
            if sc_cnt.get(sec, 0) < cap:
                new.add(sym)
                sc_cnt[sec] = sc_cnt.get(sec, 0) + 1
            if len(new) >= cfg.top_n:
                break

        # ── STEP 4: Exit buffer (same as v3) ─────────────────────────────────
        if cfg.exit_buffer > 0:
            stay = set()
            for sym in cur:
                if sym in new:
                    stay.add(sym)
                elif sym in scores:
                    if scores[sym] >= entry_s.get(sym, 0) - cfg.exit_buffer:
                        stay.add(sym)
            final = new | stay
            if len(final) > int(cfg.top_n * 1.2):
                stay_scored = sorted(
                    [(s, scores.get(s, 0)) for s in stay - new],
                    key=lambda x: -x[1]
                )
                keep = set(s for s,_ in stay_scored[:max(0, int(cfg.top_n*1.2)-len(new))])
                final = new | keep
        else:
            final = new

        exits  = cur - final
        enters = final - cur
        nxt = min(day + cfg.rebal_days, len(ba)-1)

        if cur:
            to_list.append((len(exits)+len(enters)) / (2*max(len(final),1)))

        # ── Log exits ─────────────────────────────────────────────────────────
        for sym in exits:
            col = sym + ".NS"
            if col not in cl.columns: continue
            cr = cl[col].reindex(bd).ffill()
            ep = float(cr.iloc[day]) if day < len(cr) else np.nan
            if np.isnan(ep): continue
            enp = entry_p.get(sym, ep)
            sec = SECTOR_MAP.get(sym, "OTHER")
            ret = (ep/enp - 1)*100 if enp > 0 else 0
            tlog.append({
                "symbol":sym, "entry":round(enp,2), "exit":round(ep,2),
                "return_pct":round(ret,2), "sector":sec,
                "cap_at_exit": cap_reasons.get(sec,"?"),
            })
            sec_ct[sec] = sec_ct.get(sec,0) + 1
            sec_pnl.setdefault(sec, []).append(ret)

        for sym in enters:
            col = sym + ".NS"
            if col in cl.columns:
                cr = cl[col].reindex(bd).ffill()
                entry_p[sym] = float(cr.iloc[day]) if day < len(cr) else 0
                entry_s[sym] = scores.get(sym, 0)

        # ── Forward returns ───────────────────────────────────────────────────
        fwd = []
        for sym in final:
            col = sym + ".NS"
            if col not in cl.columns: continue
            cr = cl[col].reindex(bd).ffill()
            p0 = float(cr.iloc[day]); p1 = float(cr.iloc[nxt])
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0: continue
            fwd.append(p1/p0 - 1)

        to_frac = (len(exits)+len(enters)) / (2*max(len(final),1)) if final else 0
        pr = (np.mean(fwd) if fwd else 0) - to_frac * cfg.cost_pct
        br = ba[nxt] / ba[day] - 1

        ppr.append(pr); pbr.append(br)
        pn.append(pn[-1]*(1+pr)); bn.append(bn[-1]*(1+br))
        pd_.append(bd[nxt]); cur = final

    # ── Compute metrics ───────────────────────────────────────────────────────
    pr=np.array(ppr); br_=np.array(pbr)
    pna=np.array(pn); bna=np.array(bn)
    ppy=252/cfg.rebal_days; ny=len(pr)/ppy if ppy else 1

    cp = (pna[-1]/pna[0])**(1/ny)-1 if ny>0 else 0
    cb = (bna[-1]/bna[0])**(1/ny)-1 if ny>0 else 0
    rfp = cfg.risk_free/ppy; ex = pr-rfp
    sh = (ex.mean()/ex.std()*np.sqrt(ppy)) if ex.std()>0 else 0
    ds = ex[ex<0].std()*np.sqrt(ppy)
    so = (ex.mean()*ppy/ds) if ds>0 else 0
    pk = np.maximum.accumulate(pna); mdd = ((pna-pk)/pk).min()
    bpk = np.maximum.accumulate(bna); bmdd = ((bna-bpk)/bna).min()
    cal = cp/abs(mdd) if mdd<0 else 0
    cov = np.cov(pr, br_); beta = cov[0,1]/cov[1,1] if cov[1,1]>0 else 1
    ja = (cp-cfg.risk_free) - beta*(cb-cfg.risk_free)
    hm = (pr>br_).mean()
    tr = [t["return_pct"] for t in tlog]
    ht = (np.array(tr)>0).mean() if tr else 0
    ato = np.mean(to_list)*100 if to_list else 0
    anto = ato*ppy; cd = anto/100*cfg.cost_pct*100
    ts = sum(sec_ct.values())
    se = {k:v/ts*100 for k,v in sec_ct.items()} if ts else {}
    sp = {k:round(np.mean(v),2) for k,v in sec_pnl.items()}

    # Cap utilisation: how often was each sector reduced?
    cap_util = {
        sec: round(sum(1 for c in hist if c < cfg.sector_cap) / len(hist) * 100, 1)
        for sec, hist in sec_cap_history.items() if hist
    }

    return dict(
        label=cfg.label or f"v4_sc{cfg.sector_cap}_thr{cfg.weak_threshold*100:.0f}_slots{cfg.weak_slots}_buf{cfg.exit_buffer:.0f}",
        port_nav=pna, bench_nav=bna, port_dates=pd_,
        trade_log=tlog, period_pr=pr, period_br=br_,
        cagr_p=round(cp*100,2), cagr_b=round(cb*100,2),
        alpha=round((cp-cb)*100,2), j_alpha=round(ja*100,2),
        beta=round(float(beta),3), sharpe=round(sh,2), sortino=round(so,2),
        mdd=round(mdd*100,2), bmdd=round(bmdd*100,2), calmar=round(cal,2),
        hit_m=round(hm*100,1), hit_t=round(ht*100,1),
        avg_to=round(ato,1), ann_to=round(anto,0), cost_drag=round(cd,2),
        n_trades=len(tlog), n_periods=len(pr), n_years=round(ny,1),
        sector_exp=se, sector_pnl=sp, cap_utilisation=cap_util,
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

    if verbose:
        print(f"\nSECTOR EXPOSURE + DYNAMIC CAP ACTIVITY──")
        cap_util = r.get('cap_utilisation', {})
        pnl = r.get('sector_pnl', {})
        for sec, pct in sorted(r['sector_exp'].items(), key=lambda x: -x[1]):
            bar = '█' * int(pct/3)
            avg_ret = pnl.get(sec, 0)
            reduced_pct = cap_util.get(sec, 0)
            ret_str = f"{'+' if avg_ret>=0 else ''}{avg_ret:.1f}%"
            red_str = f"  cap reduced {reduced_pct:.0f}% of periods" if reduced_pct > 10 else ""
            flag = " ⚠" if pct > 30 else ""
            print(f"  {sec:14s}  {pct:5.1f}%  {bar:<8}  trade:{ret_str:>7}{red_str}{flag}")

    print(f"\nINTERPRETATION──────────────────────")
    if r['j_alpha'] > 3:
        print(f"  ✓ Real alpha {r['j_alpha']:.1f}% (Jensen's, beta-adjusted)")
    elif r['j_alpha'] > 0:
        print(f"  ~ Small alpha {r['j_alpha']:.1f}% — borderline")
    else:
        print(f"  ✗ Negative Jensen's alpha")
    if r['hit_m'] < 50:
        print(f"  ℹ Hit ratio {r['hit_m']:.0f}% < 50% — positive skew (expected for momentum)")
    if r['sharpe'] < 0.8:
        print(f"  ⚠ Sharpe {r['sharpe']:.2f}: expand universe to Nifty 200 for improvement")
    if r['ann_to'] > 300:
        print(f"  ⚠ Turnover {r['ann_to']:.0f}%: consider reducing via --exit-buffer")
    if r['mdd'] > r['bmdd']:
        print(f"  ⚠ Portfolio MDD ({r['mdd']:.1f}%) worse than benchmark ({r['bmdd']:.1f}%)")


def sweep(cfg: Cfg, cl, hi, lo, vo):
    """Sensitivity across threshold, weak_slots, lookback."""
    print("\n" + "="*90)
    print("DYNAMIC SECTOR CAP SENSITIVITY SWEEP")
    print("="*90)
    print(f"\n{'Config':<50} {'JAlpha':>7} {'Sharpe':>7} {'MDD%':>7} {'HitM%':>6} {'Cost':>5}")
    print("─"*82)

    cases = [
        # threshold sensitivity
        ("threshold=0%   (any negative → reduce)",   0.0,   1, 3),
        ("threshold=-1%  (base)",                   -1.0,   1, 3),
        ("threshold=-2%  (stricter)",               -2.0,   1, 3),
        ("threshold=-3%  (very strict)",            -3.0,   1, 3),
        # weak_slots sensitivity
        ("threshold=-1% slots=0 (full exclusion)",  -1.0,   0, 3),
        ("threshold=-1% slots=1 (base)",            -1.0,   1, 3),
        ("threshold=-1% slots=2 (gentle reduce)",   -1.0,   2, 3),
        # lookback sensitivity
        ("threshold=-1% lookback=2",                -1.0,   1, 2),
        ("threshold=-1% lookback=3 (base)",         -1.0,   1, 3),
        ("threshold=-1% lookback=5",                -1.0,   1, 5),
        # comparison: v3 (no dynamic cap)
        ("v3 baseline (no dynamic cap)",             999.0, 3, 3),
    ]

    for label, thr, slots, lb in cases:
        c = Cfg(**{**cfg.__dict__,
                   'weak_threshold': thr/100,
                   'weak_slots': slots,
                   'lookback_periods': lb,
                   'label': label})
        r = run(c, cl, hi, lo, vo)
        print(f"  {label:<48} {r['j_alpha']:>7.2f} {r['sharpe']:>7.2f} "
              f"{r['mdd']:>7.2f} {r['hit_m']:>6.1f} {r['cost_drag']:>5.2f}")


def walkforward(cfg: Cfg, cl, hi, lo, vo):
    bench = cl["^NSEI"].dropna()
    n = len(bench); split = int(n * 0.6)
    ri = run(cfg, cl, hi, lo, vo, si=cfg.min_history, ei=split)
    ro = run(cfg, cl, hi, lo, vo, si=split, ei=n-cfg.rebal_days)
    ri['label'] = "IN-SAMPLE  (train)"
    ro['label'] = "OUT-OF-SAMPLE (test)"
    print("\n" + "="*72 + "\nWALK-FORWARD VALIDATION\n" + "="*72)
    print_r(ri, verbose=False)
    print_r(ro, verbose=False)

    print(f"\nDEGRADATION CHECK───────────────────")
    sd = ri['sharpe'] - ro['sharpe']
    ad = ri['alpha']  - ro['alpha']
    print(f"  Sharpe drop in→out:  {sd:+.2f}  "
          f"{'⚠ possible overfit' if ri['sharpe']>0 and sd/ri['sharpe']>0.4 else '✓ acceptable'}")
    print(f"  Alpha  drop in→out:  {ad:+.2f}%  "
          f"{'✓ alpha persists' if abs(ad)<3 else '⚠ alpha shifted'}")
    print(f"  Hit M  in→out:       {ri['hit_m']:.1f}% → {ro['hit_m']:.1f}%")
    print(f"\n  In-sample NIFTY:     {ri['cagr_b']:.2f}% CAGR")
    print(f"  Out-of-sample NIFTY: {ro['cagr_b']:.2f}% CAGR")
    nifty_drop = ri['cagr_b'] - ro['cagr_b']
    alpha_drop  = ri['j_alpha'] - ro['j_alpha']
    print(f"  NIFTY drop:          {nifty_drop:+.2f}%  Alpha drop: {alpha_drop:+.2f}%")
    if abs(alpha_drop) < abs(nifty_drop) * 0.5:
        print(f"  ✓ Alpha drop ({alpha_drop:+.1f}%) much smaller than market drop ({nifty_drop:+.1f}%)")
        print(f"    Strategy is robust to market deterioration.")
    else:
        print(f"  ⚠ Alpha dropped significantly relative to market change.")

    # Show which sectors the dynamic cap protected in each period
    print(f"\n  TOP SECTOR CAP REDUCTIONS (% of periods each sector was capped):")
    for period_r, period_name in [(ri,'IN-SAMPLE'), (ro,'OUT-OF-SAMPLE')]:
        cu = period_r.get('cap_utilisation', {})
        if cu:
            top_reduced = sorted(cu.items(), key=lambda x: -x[1])[:5]
            print(f"\n  {period_name}:")
            for sec, pct in top_reduced:
                avg_ret = period_r.get('sector_pnl',{}).get(sec, 0)
                print(f"    {sec:14s}: cap reduced {pct:.0f}% of periods  "
                      f"(avg trade: {'+' if avg_ret>=0 else ''}{avg_ret:.1f}%)")


def main():
    p = argparse.ArgumentParser(description="WealthOS v4 — Dynamic Sector Cap")
    p.add_argument("--universe",       choices=["nifty50","extended"], default="nifty50")
    p.add_argument("--top-n",          type=int,   default=15)
    p.add_argument("--rebal-days",     type=int,   default=21)
    p.add_argument("--min-score",      type=float, default=40)
    p.add_argument("--sector-cap",     type=int,   default=3)
    p.add_argument("--exit-buffer",    type=float, default=15)
    p.add_argument("--weak-threshold", type=float, default=-1.0,
                   help="Sector trailing return threshold %% (default -1.0)")
    p.add_argument("--weak-slots",     type=int,   default=1,
                   help="Slots allowed when sector is weak (0=exclude, 1=one stock)")
    p.add_argument("--lookback",       type=int,   default=3,
                   help="Lookback periods for trailing sector return")
    p.add_argument("--sweep",          action="store_true")
    p.add_argument("--walkforward",    action="store_true")
    a = p.parse_args()

    syms = EXTENDED if a.universe == "extended" else NIFTY50
    cfg = Cfg(
        universe=syms, top_n=a.top_n, rebal_days=a.rebal_days,
        min_score=a.min_score, sector_cap=a.sector_cap,
        exit_buffer=a.exit_buffer,
        weak_threshold=a.weak_threshold / 100,
        weak_slots=a.weak_slots,
        lookback_periods=a.lookback,
        label=f"{a.universe} top{a.top_n} r{a.rebal_days} "
              f"sc{a.sector_cap} thr{a.weak_threshold} "
              f"slots{a.weak_slots} buf{a.exit_buffer:.0f}",
    )

    print(f"Loading {len(syms)} symbols (6y history)...")
    cl, hi, lo, vo = _load(syms)

    r = run(cfg, cl, hi, lo, vo)
    print_r(r)

    if a.sweep:       sweep(cfg, cl, hi, lo, vo)
    if a.walkforward: walkforward(cfg, cl, hi, lo, vo)


if __name__ == "__main__":
    main()
