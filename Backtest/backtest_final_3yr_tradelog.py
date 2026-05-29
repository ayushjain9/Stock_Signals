"""
WealthOS — 3-Year Backtest Trade Log Export
============================================

Generates detailed trade-by-trade log showing:
- Stock symbol
- Entry date, entry price, entry score
- Exit date, exit price, exit reason
- Return %
- Holding period

Exports to Excel for analysis.
"""
from __future__ import annotations
import argparse, warnings
from datetime import datetime
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

def _ema(a, n):
    k=2/(n+1); o=np.zeros(len(a)); o[n-1]=a[:n].mean()
    for i in range(n,len(a)): o[i]=a[i]*k+o[i-1]*(1-k)
    return o

def score_stock(c, h, lo, bench, v=None):
    MIN_HISTORY = 210
    if len(c) < MIN_HISTORY:
        return -1.0
    p=c[-1]; d25=c[-25:].mean(); d50=c[-50:].mean(); d200=c[-200:].mean(); s=0.0
    if p>d25>d50>d200: s+=20
    elif d50>d200 and p>d200: s+=12
    elif d50>d200: s+=6
    if len(bench)>=90:
        rs=(c[-1]/c[-90]-bench[-1]/bench[-90])*100
        if rs>10: s+=25
        elif rs>5: s+=18
        elif rs>0: s+=10
        elif rs<-5: s-=15
    if len(c)>=147 and len(bench)>=147:
        rs6=(c[-22]/c[-147]-bench[-22]/bench[-147])*100
        if rs6>15: s+=8
        elif rs6>8: s+=5
        elif rs6>0: s+=2
        elif rs6<-8: s-=5
    if len(c)>=66:
        roc=(c[-1]/c[-66]-1)*100
        ranges=(h[-14:]-lo[-14:]).mean() if len(h)>=14 else c[-20:].std()
        atr_pct=ranges/p*100 if p>0 else 2.0
        vam=roc/atr_pct if atr_pct>0.01 else 0
        if vam>3: s+=20
        elif vam>2: s+=15
        elif vam>1: s+=8
        elif vam>0: s+=3
    if len(c)>=252:
        pct52=(p/c[-252:].max()-1)*100
        if -5<=pct52<=0: s+=6
    if len(c)>=200:
        e5=_ema(c,5); e20=_ema(c,20); e50=_ema(c,50); e200=_ema(c,200)
        s+=sum([e5[-1]>e20[-1], e20[-1]>e50[-1], e50[-1]>e200[-1]])*4
    if len(c)>=30:
        if c[-1]/c[-10]>1 and c[-1]/c[-10]>c[-10]/c[-20]: s+=7
    if v is not None and len(v)>=26:
        avg5d=float(v[-5:].mean()); avg20d=float(v[-25:-5].mean())
        if avg20d>0:
            ratio=avg5d/avg20d
            if ratio>=2.0: s+=6
            elif ratio>=1.5: s+=4
    return min(float(s),100.0)

def run_with_trades(cl, hi, lo, vo):
    TOP_N=15; SECTOR_CAP=3; EXIT_BUFFER=15.0; MIN_SCORE=40.0
    LIQUIDITY_CR=5.0; MIN_HIST=210; COST_PCT=0.002
    HARD_STOP=0.85

    bench_s=cl["^NSEI"].dropna(); ba=bench_s.values.astype(float); bd=bench_s.index
    si=MIN_HIST; ei=len(ba)-21

    cur:set=set(); ep:dict={}; es:dict={}; entry_dates:dict={}
    trades=[]  # list of {symbol, entry_date, entry_price, entry_score, exit_date, exit_price, exit_reason, return_pct}

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
            if sym in new:
                stay.add(sym)
            else:
                col=sym+".NS"
                if col not in cl.columns: continue
                cr=cl[col].reindex(bd).ffill()
                cur_px=float(cr.iloc[day]) if day<len(cr) else 0
                # Hard stop-loss check
                if ep.get(sym,0)>0 and cur_px>0 and cur_px<ep[sym]*HARD_STOP:
                    # Force exit — hard stop
                    exit_date=bd[day].strftime('%Y-%m-%d')
                    exit_reason="Hard stop-loss (-15%)"
                    ret_pct=(cur_px/ep[sym]-1)*100 if ep[sym]>0 else 0
                    trades.append({
                        'symbol': sym,
                        'entry_date': entry_dates.get(sym, ''),
                        'entry_price': round(ep.get(sym,0),2),
                        'entry_score': round(es.get(sym,0),1),
                        'exit_date': exit_date,
                        'exit_price': round(cur_px,2),
                        'exit_reason': exit_reason,
                        'return_%': round(ret_pct,2),
                        'days_held': (datetime.strptime(exit_date,'%Y-%m-%d')-datetime.strptime(entry_dates.get(sym,'2000-01-01'),'%Y-%m-%d')).days,
                        'sector': SECTOR_MAP.get(sym,'OTHER'),
                    })
                    continue
                # Exit buffer check
                if sym in scores and scores[sym]>=es.get(sym,0)-EXIT_BUFFER:
                    stay.add(sym)
                else:
                    # Exit buffer triggered
                    exit_date=bd[day].strftime('%Y-%m-%d')
                    exit_reason=f"Score dropped {es.get(sym,0)-scores.get(sym,0):.0f} pts"
                    cur_px_close=float(cr.iloc[day]) if day<len(cr) else ep.get(sym,0)
                    ret_pct=(cur_px_close/ep.get(sym,1)-1)*100
                    trades.append({
                        'symbol': sym,
                        'entry_date': entry_dates.get(sym, ''),
                        'entry_price': round(ep.get(sym,0),2),
                        'entry_score': round(es.get(sym,0),1),
                        'exit_date': exit_date,
                        'exit_price': round(cur_px_close,2),
                        'exit_reason': exit_reason,
                        'return_%': round(ret_pct,2),
                        'days_held': (datetime.strptime(exit_date,'%Y-%m-%d')-datetime.strptime(entry_dates.get(sym,'2000-01-01'),'%Y-%m-%d')).days,
                        'sector': SECTOR_MAP.get(sym,'OTHER'),
                    })

        final=new|stay

        # Record entries
        for sym in (final-cur):
            col=sym+".NS"
            if col in cl.columns:
                cr=cl[col].reindex(bd).ffill()
                ep[sym]=float(cr.iloc[day]) if day<len(cr) else 0
                es[sym]=scores.get(sym,0)
                entry_dates[sym]=bd[day].strftime('%Y-%m-%d')

        cur=final

    return pd.DataFrame(trades)

def main():
    p=argparse.ArgumentParser(description="Export trade log from 3-year backtest")
    p.add_argument("--output", type=str, default="backtest_final_3yr_trades.xlsx", help="Output Excel file")
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

    print(f"Extracting trades...")
    trades_df=run_with_trades(cl,hi,lo,vo)

    # Summary stats
    if len(trades_df)>0:
        winning_trades=trades_df[trades_df['return_%']>0]
        losing_trades=trades_df[trades_df['return_%']<=0]
        print(f"\nTRADE SUMMARY:")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Winning: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
        print(f"  Losing: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
        print(f"  Avg return: {trades_df['return_%'].mean():.2f}%")
        print(f"  Avg days held: {trades_df['days_held'].mean():.0f}")
        print(f"  Best trade: {trades_df['return_%'].max():.2f}%")
        print(f"  Worst trade: {trades_df['return_%'].min():.2f}%")

    # Write to Excel with formatting
    print(f"\nWriting to {a.output}...")
    with pd.ExcelWriter(a.output, engine='openpyxl') as writer:
        trades_df.to_excel(writer, sheet_name='Trades', index=False)

        # Add summary sheet
        summary_df=pd.DataFrame({
            'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate %', 'Avg Return %', 'Avg Days Held', 'Best Trade %', 'Worst Trade %'],
            'Value': [
                len(trades_df),
                len(winning_trades) if len(trades_df)>0 else 0,
                len(losing_trades) if len(trades_df)>0 else 0,
                round(len(winning_trades)/len(trades_df)*100,1) if len(trades_df)>0 else 0,
                round(trades_df['return_%'].mean(),2) if len(trades_df)>0 else 0,
                round(trades_df['days_held'].mean(),0) if len(trades_df)>0 else 0,
                round(trades_df['return_%'].max(),2) if len(trades_df)>0 else 0,
                round(trades_df['return_%'].min(),2) if len(trades_df)>0 else 0,
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"✓ Trade log exported to {a.output}\n")

if __name__=="__main__":
    main()
