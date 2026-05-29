"""
WealthOS — Quality-Value Screener (Mid & Small Cap)
====================================================

WHAT THIS DOES
--------------
Scans 400+ NSE mid and small cap stocks weekly.
Scores each stock on 8 fundamental metrics (0-100 points).
Shortlists the top 15-20 quality businesses trading at reasonable prices.

THIS IS NOT A MOMENTUM SCREENER.
  Momentum: follow the price trend, rebalance monthly
  Value:     find quality businesses the market has underpriced, hold 3-5 years

THE HONEST LIMITATION
---------------------
This screener narrows 400 stocks to 15. It cannot tell you:
  - WHY the stock is cheap (temporary or permanent?)
  - Whether management is trustworthy
  - Whether the competitive moat is durable

WORKFLOW (takes 20 min/week)
------------------------------
  1. Run this script Sunday evening
  2. Look at top 10 stocks
  3. For top 3-5: spend 10 min each on screener.in — read last 2 years
     of key metrics and management commentary
  4. Build a concentrated 5-8 stock portfolio
  5. Review quarterly (not monthly — this is a 3-5 year strategy)

THE 8 METRICS AND WHY
----------------------
  ROE > 15% (5yr avg)      25pts  Real compounding ability — not a one-year blip
  Revenue growth 15%+ 3yr  20pts  Growing business — not cheap because shrinking
  Profit margin trend       15pts  Pricing power — margins expanding = moat
  Debt/Equity < 0.5         15pts  Safety — low debt survives bad years
  PE vs sector median       10pts  Cheap relative to peers — the actual value signal
  Promoter holding > 50%     8pts  Skin in the game — management aligned
  Operating cashflow +ve     5pts  Real profits — filters accounting tricks
  EPS growth consistent      2pts  No one-off years — sustainable earnings

EXIT LOGIC (very different from momentum)
------------------------------------------
  Hold as long as the thesis holds. The thesis for each stock is:
  "Quality business (high ROE, low debt, growing) trading below peers."
  Exit when:
  - ROE drops below 12% for 2 consecutive years
  - Debt/Equity crosses 1.0
  - PE re-rates to sector average (full value achieved)
  - Promoter pledge crosses 20%

USAGE
-----
  pip install yfinance pandas numpy rich requests
  python value_screener.py                          # full scan
  python value_screener.py --top 10                 # show top 10 only
  python value_screener.py --min-score 60           # high conviction only
  python value_screener.py --sector IT              # one sector only
  python value_screener.py --export value_picks.csv # save results
"""
from __future__ import annotations
import argparse, warnings, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Universe: Nifty Midcap 150 + Nifty Smallcap 250 ─────────────────────────
# Curated list — largest and most liquid mid/small caps on NSE

MIDCAP = [
    # FINANCIALS
    "MUTHOOTFIN","CHOLAFIN","MANAPPURAM","SUNDARMFIN","AAVAS","HOMEFIRST",
    "CREDITACC","UJJIVANSFB","EQUITASBNK","SURYODAY",
    # IT/TECH
    "PERSISTENT","COFORGE","LTTS","MPHASIS","KPIT","OFSS","TATAELXSI",
    "MASTEK","ZENSAR","CYIENT","BIRLASOFT","NIITTECH","HEXAWARE",
    # PHARMA / HEALTHCARE
    "AUROPHARMA","IPCALAB","NATCO","AJANTPHARMA","GLAND","LAURUSLABS",
    "GLENMARK","STAR","RAINBOW","KRSNAA","VIJAYABANK","METROPOLIS",
    # AUTO / ANC
    "BHARATFORG","MOTHERSON","BOSCHLTD","MINDAIND","SUNDRMFAST",
    "GABRIEL","WABCO","SUPRAJIT","ENDURANCE","CRAFTSMAN",
    # CONSUMER / RETAIL
    "DMART","TRENT","ABFRL","VMART","MANYAVAR","VBL","RADICO","MCDOWELL",
    "PAGEIND","WHIRLPOOL","BATAINDIA","RELAXO","VSTIND",
    # INDUSTRIALS / CAPITAL GOODS
    "CUMMINSIND","GRINDWELL","SCHAEFFLER","TIMKEN","SKFINDIA",
    "THERMAX","ELGIEQUIP","BHEL","BEML","RITES","IRCON",
    # CHEMICALS
    "PIIND","DEEPAKNITRITE","TATACHEM","VINATIORGA","AARTI","NOCIL",
    "GALAXYSURF","FINEORG","NAVINFLUOR","SRF","BALCHEMICALS",
    # CEMENT / INFRA
    "RAMCOCEM","JKCEMENT","HEIDELBERG","DBCORP","NUVOCO","ORIENTCEM",
    # METALS / MINING
    "NMDC","MOIL","HINDZINC","NALCO","WELCORP","RATNAMANI","APL",
    # REAL ESTATE
    "BRIGADE","PRESTIGE","SOBHA","MAHLIFE","GODREJPROP","OBEROIRLTY",
    # ENERGY
    "IREDA","SJVN","THERMAX","PFCLTD","RECLTD","IRFC",
    # ELECTRONICS / MANUFACTURING
    "DIXON","KAYNES","AMBER","HAVELLS","POLYCAB","KEI","RR",
    # MEDIA / TELECOM
    "ZEEL","SUNTV","NAZARA","ROUTE","TATACOMM",
    # MISCELLANEOUS
    "WHIRLPOOL","VSTIND","JYOTHY","MARICO","EMAMILTD","BAJAJCON",
    "ZYDUSLIFE","IOLCP","GMMPFAUDLR","NITINAUTO","CRAFTSMAN",
    # MORE MID-CAPS COMMONLY DISCUSSED
    "HAPPSTMNDS","LATENTVIEW","TANLA","INTELLECT","NUCLEUS",
    "CARTRADE","NYKAA","PAYTM","ZOMATO","POLICYBZR",
    "IRCTC","RAILVIKAS","RVNL","TEXRAIL","CESC",
    "APLAPOLLO","JINDALSAW","JSHL","NRBBEARING",
    "PGHH","GILLETTE","NESTLE","COLPAL","HINDPETRO",
]

# Remove duplicates
MIDCAP = list(dict.fromkeys(MIDCAP))

SECTOR_MAP: dict[str, str] = {
    "MUTHOOTFIN":"NBFC","CHOLAFIN":"NBFC","MANAPPURAM":"NBFC","SUNDARMFIN":"NBFC",
    "AAVAS":"NBFC","HOMEFIRST":"NBFC","CREDITACC":"NBFC","UJJIVANSFB":"BANKING",
    "EQUITASBNK":"BANKING","SURYODAY":"BANKING",
    "PERSISTENT":"IT","COFORGE":"IT","LTTS":"IT","MPHASIS":"IT","KPIT":"IT",
    "OFSS":"IT","TATAELXSI":"IT","MASTEK":"IT","ZENSAR":"IT","CYIENT":"IT",
    "BIRLASOFT":"IT","HEXAWARE":"IT","HAPPSTMNDS":"IT","LATENTVIEW":"IT",
    "TANLA":"IT","INTELLECT":"IT","NUCLEUS":"IT",
    "AUROPHARMA":"PHARMA","IPCALAB":"PHARMA","NATCO":"PHARMA","AJANTPHARMA":"PHARMA",
    "GLAND":"PHARMA","LAURUSLABS":"PHARMA","GLENMARK":"PHARMA","ZYDUSLIFE":"PHARMA",
    "STAR":"HEALTHCARE","RAINBOW":"HEALTHCARE","KRSNAA":"HEALTHCARE","METROPOLIS":"HEALTHCARE",
    "BHARATFORG":"AUTO_ANC","MOTHERSON":"AUTO_ANC","BOSCHLTD":"AUTO_ANC",
    "SUNDRMFAST":"AUTO_ANC","GABRIEL":"AUTO_ANC","ENDURANCE":"AUTO_ANC",
    "CRAFTSMAN":"AUTO_ANC","SUPRAJIT":"AUTO_ANC","NRBBEARING":"AUTO_ANC",
    "DMART":"RETAIL","TRENT":"RETAIL","VMART":"RETAIL","MANYAVAR":"RETAIL",
    "BATAINDIA":"CONSUMER","RELAXO":"CONSUMER","PAGEIND":"CONSUMER",
    "RADICO":"FMCG","MCDOWELL":"FMCG","VBL":"FMCG","MARICO":"FMCG",
    "EMAMILTD":"FMCG","BAJAJCON":"FMCG","JYOTHY":"FMCG","PGHH":"FMCG",
    "GILLETTE":"FMCG","COLPAL":"FMCG",
    "CUMMINSIND":"INDUSTRIALS","GRINDWELL":"INDUSTRIALS","SCHAEFFLER":"INDUSTRIALS",
    "TIMKEN":"INDUSTRIALS","SKFINDIA":"INDUSTRIALS","THERMAX":"INDUSTRIALS",
    "ELGIEQUIP":"INDUSTRIALS","BEML":"INDUSTRIALS",
    "PIIND":"CHEMICALS","DEEPAKNITRITE":"CHEMICALS","TATACHEM":"CHEMICALS",
    "VINATIORGA":"CHEMICALS","AARTI":"CHEMICALS","NOCIL":"CHEMICALS",
    "GALAXYSURF":"CHEMICALS","FINEORG":"CHEMICALS","NAVINFLUOR":"CHEMICALS",
    "SRF":"CHEMICALS","BALCHEMICALS":"CHEMICALS","IOLCP":"CHEMICALS",
    "RAMCOCEM":"CEMENT","JKCEMENT":"CEMENT","HEIDELBERG":"CEMENT",
    "NUVOCO":"CEMENT","ORIENTCEM":"CEMENT",
    "NMDC":"METALS","MOIL":"METALS","HINDZINC":"METALS","NALCO":"METALS",
    "RATNAMANI":"METALS","APLAPOLLO":"METALS","JINDALSAW":"METALS","JSHL":"METALS",
    "BRIGADE":"REALTY","PRESTIGE":"REALTY","SOBHA":"REALTY","MAHLIFE":"REALTY",
    "GODREJPROP":"REALTY","OBEROIRLTY":"REALTY",
    "IREDA":"POWER","SJVN":"POWER","RECLTD":"NBFC","IRFC":"NBFC","PFCLTD":"NBFC",
    "RITES":"INFRA","IRCON":"INFRA","RVNL":"INFRA","TEXRAIL":"INFRA","RAILVIKAS":"INFRA",
    "DIXON":"ELECTRONICS","KAYNES":"ELECTRONICS","AMBER":"ELECTRONICS",
    "HAVELLS":"ELECTRONICS","POLYCAB":"ELECTRONICS","KEI":"ELECTRONICS",
    "ZEEL":"MEDIA","SUNTV":"MEDIA","NAZARA":"CONSUMER_TECH",
    "ROUTE":"TELECOM","TATACOMM":"TELECOM",
    "NYKAA":"CONSUMER_TECH","PAYTM":"FINTECH","ZOMATO":"CONSUMER_TECH",
    "POLICYBZR":"FINTECH","IRCTC":"CONSUMER","CARTRADE":"CONSUMER_TECH",
    "BHEL":"INDUSTRIALS","CESC":"POWER","HINDPETRO":"ENERGY","WHIRLPOOL":"CONSUMER",
    "ABFRL":"RETAIL","VSTIND":"FMCG","NITINAUTO":"AUTO_ANC","GMMPFAUDLR":"INDUSTRIALS",
    "MINDAIND":"AUTO_ANC","WABCO":"AUTO_ANC","DBCORP":"MEDIA","WELCORP":"METALS",
    "APL":"METALS","NESTLE":"FMCG","JYOTHY":"FMCG","NIITTECH":"IT",
}


# ── Scoring engine ────────────────────────────────────────────────────────────

@dataclass
class ValueResult:
    symbol:        str
    name:          str          = ""
    sector:        str          = "OTHER"
    price:         float        = 0
    mkt_cap_cr:    float        = 0

    # Raw metrics
    pe:            float        = 0      # trailing PE
    pb:            float        = 0      # price to book
    roe:           float        = 0      # return on equity (TTM)
    roe_avg:       float        = 0      # avg ROE from annual data
    revenue_cagr:  float        = 0      # 3-year revenue CAGR
    margin_now:    float        = 0      # current net margin %
    margin_3y_ago: float        = 0      # margin 3 years ago
    debt_eq:       float        = 0      # debt/equity
    op_cashflow:   float        = 0      # operating cashflow
    eps_3y:        list         = field(default_factory=list)  # 3 years EPS
    promoter_pct:  float        = 0      # promoter holding %
    sector_median_pe: float     = 0      # sector median PE

    # Scores
    score:         int          = 0
    score_pct:     float        = 0.0
    fired:         list         = field(default_factory=list)
    rating:        str          = "UNSCORED"
    error:         str | None   = None


def score_stock(r: ValueResult, sector_medians: dict[str, float]) -> tuple[int, list[str]]:
    """
    Score 0-100 on 8 fundamental metrics.
    Each metric is either pass/fail or graded.
    """
    pts = 0
    fired = []

    # 1. ROE — 25 pts
    # Use avg ROE from annual data if available, else TTM
    roe = r.roe_avg if r.roe_avg > 0 else r.roe
    if roe >= 20:
        pts += 25; fired.append(f"ROE {roe:.1f}% excellent (+25)")
    elif roe >= 15:
        pts += 18; fired.append(f"ROE {roe:.1f}% good (+18)")
    elif roe >= 12:
        pts += 10; fired.append(f"ROE {roe:.1f}% acceptable (+10)")
    elif roe > 0:
        pts += 3;  fired.append(f"ROE {roe:.1f}% low (+3)")
    else:
        fired.append(f"ROE {roe:.1f}% negative (0)")

    # 2. Revenue growth — 20 pts
    rev = r.revenue_cagr
    if rev >= 20:
        pts += 20; fired.append(f"Revenue CAGR {rev:.1f}% strong (+20)")
    elif rev >= 15:
        pts += 15; fired.append(f"Revenue CAGR {rev:.1f}% good (+15)")
    elif rev >= 10:
        pts += 8;  fired.append(f"Revenue CAGR {rev:.1f}% moderate (+8)")
    elif rev >= 0:
        pts += 3;  fired.append(f"Revenue CAGR {rev:.1f}% slow (+3)")
    else:
        fired.append(f"Revenue CAGR {rev:.1f}% negative (0)")

    # 3. Margin trend — 15 pts
    if r.margin_now > 0 and r.margin_3y_ago > 0:
        margin_change = r.margin_now - r.margin_3y_ago
        if r.margin_now >= 15 and margin_change >= 0:
            pts += 15; fired.append(f"Margin {r.margin_now:.1f}% high & stable (+15)")
        elif r.margin_now >= 10 and margin_change >= 0:
            pts += 10; fired.append(f"Margin {r.margin_now:.1f}% good (+10)")
        elif margin_change >= 2:
            pts += 8;  fired.append(f"Margin expanding {margin_change:+.1f}pp (+8)")
        elif r.margin_now >= 8:
            pts += 5;  fired.append(f"Margin {r.margin_now:.1f}% adequate (+5)")
        else:
            fired.append(f"Margin {r.margin_now:.1f}% thin or compressing (0)")
    elif r.margin_now >= 10:
        pts += 8; fired.append(f"Margin {r.margin_now:.1f}% (+8, no 3yr comparison)")

    # 4. Debt/Equity — 15 pts
    d = r.debt_eq
    if d <= 0:
        pts += 15; fired.append("Debt-free or net cash (+15)")
    elif d <= 0.3:
        pts += 15; fired.append(f"D/E {d:.2f} very low (+15)")
    elif d <= 0.5:
        pts += 10; fired.append(f"D/E {d:.2f} comfortable (+10)")
    elif d <= 1.0:
        pts += 5;  fired.append(f"D/E {d:.2f} moderate (+5)")
    else:
        fired.append(f"D/E {d:.2f} high debt (0)")

    # 5. PE vs sector median — 10 pts
    sec_pe = sector_medians.get(r.sector, 0)
    if r.pe > 0 and sec_pe > 0:
        pe_discount = (sec_pe - r.pe) / sec_pe * 100
        if pe_discount >= 30:
            pts += 10; fired.append(f"PE {r.pe:.0f} vs sector {sec_pe:.0f} ({pe_discount:.0f}% discount +10)")
        elif pe_discount >= 15:
            pts += 7;  fired.append(f"PE {r.pe:.0f} vs sector {sec_pe:.0f} ({pe_discount:.0f}% discount +7)")
        elif pe_discount >= 5:
            pts += 4;  fired.append(f"PE {r.pe:.0f} vs sector {sec_pe:.0f} (slight discount +4)")
        elif pe_discount < -20:
            fired.append(f"PE {r.pe:.0f} expensive vs sector {sec_pe:.0f} (0)")
        else:
            pts += 2;  fired.append(f"PE {r.pe:.0f} near sector median (+2)")
    elif r.pe > 0 and r.pe < 20:
        pts += 6; fired.append(f"PE {r.pe:.0f} absolute low (+6)")

    # 6. Promoter holding — 8 pts
    ph = r.promoter_pct
    if ph >= 60:
        pts += 8; fired.append(f"Promoter {ph:.0f}% high conviction (+8)")
    elif ph >= 50:
        pts += 6; fired.append(f"Promoter {ph:.0f}% majority (+6)")
    elif ph >= 35:
        pts += 3; fired.append(f"Promoter {ph:.0f}% moderate (+3)")
    else:
        fired.append(f"Promoter {ph:.0f}% low (0)")

    # 7. Operating cashflow — 5 pts
    if r.op_cashflow > 0:
        pts += 5; fired.append("Operating cashflow positive (+5)")
    else:
        fired.append("Operating cashflow negative (0) — check carefully")

    # 8. EPS consistency — 2 pts
    if len(r.eps_3y) >= 3:
        growing = all(r.eps_3y[i] > r.eps_3y[i-1] for i in range(1, len(r.eps_3y)))
        if growing:
            pts += 2; fired.append("EPS growing consistently 3 years (+2)")
        else:
            fired.append("EPS inconsistent — one-off years possible (0)")

    return max(0, pts), fired


def rating_from_score(pct: float) -> str:
    if pct >= 80: return "HIGH CONVICTION"
    if pct >= 65: return "STRONG"
    if pct >= 50: return "MODERATE"
    if pct >= 35: return "WEAK"
    return "AVOID"


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_fundamentals(symbols: list[str]) -> list[ValueResult]:
    import yfinance as yf

    BATCH = 20
    results = []

    print(f"  Fetching fundamental data for {len(symbols)} stocks...")
    print(f"  (This may take 3-5 minutes — annual report data is large)\n")

    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i+BATCH]
        tickers_str = [s + ".NS" for s in batch]

        for sym, ticker_str in zip(batch, tickers_str):
            try:
                t = yf.Ticker(ticker_str)
                info = t.info or {}

                # Basic info
                price    = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
                mkt_cap  = float(info.get("marketCap") or 0) / 1e7  # convert to crores
                name     = info.get("longName") or info.get("shortName") or sym
                pe       = float(info.get("trailingPE") or 0)
                pb       = float(info.get("priceToBook") or 0)
                roe_ttm  = float(info.get("returnOnEquity") or 0) * 100  # convert to %
                margin   = float(info.get("profitMargins") or 0) * 100
                de_ratio = float(info.get("debtToEquity") or 0) / 100   # yfinance gives as %, convert
                op_cf    = float(info.get("operatingCashflow") or 0)
                promo    = float(info.get("heldPercentInsiders") or 0) * 100

                # Revenue CAGR from annual income statement
                try:
                    fin = t.financials  # annual, columns = most recent first
                    if fin is not None and not fin.empty and fin.shape[1] >= 3:
                        rev_row = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
                        if rev_row is not None:
                            rev_new = float(rev_row.iloc[0])
                            rev_old = float(rev_row.iloc[min(3, fin.shape[1]-1)])
                            n_years = min(3, fin.shape[1]-1)
                            rev_cagr = ((rev_new / rev_old) ** (1/n_years) - 1) * 100 if rev_old > 0 else 0
                        else:
                            rev_cagr = 0

                        # Margin 3 years ago
                        net_row = fin.loc["Net Income"] if "Net Income" in fin.index else None
                        if net_row is not None and rev_row is not None:
                            idx = min(3, fin.shape[1]-1)
                            rev_old_v = float(rev_row.iloc[idx]) if not pd.isna(rev_row.iloc[idx]) else 0
                            ni_old_v  = float(net_row.iloc[idx]) if not pd.isna(net_row.iloc[idx]) else 0
                            margin_old = (ni_old_v / rev_old_v * 100) if rev_old_v > 0 else 0
                        else:
                            margin_old = 0
                    else:
                        rev_cagr = 0; margin_old = 0
                except Exception:
                    rev_cagr = 0; margin_old = 0

                # ROE from balance sheet (avg over available years)
                try:
                    bs = t.balance_sheet
                    fin2 = t.financials
                    roe_list = []
                    if bs is not None and fin2 is not None and not bs.empty and not fin2.empty:
                        eq_row = None
                        for eq_label in ["Stockholders Equity", "Total Stockholder Equity",
                                         "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                            if eq_label in bs.index:
                                eq_row = bs.loc[eq_label]; break
                        ni_row = fin2.loc["Net Income"] if "Net Income" in fin2.index else None
                        if eq_row is not None and ni_row is not None:
                            for col_i in range(min(4, len(eq_row))):
                                try:
                                    eq = float(eq_row.iloc[col_i])
                                    ni = float(ni_row.iloc[col_i])
                                    if eq > 0: roe_list.append(ni/eq*100)
                                except Exception:
                                    pass
                    roe_avg = float(np.mean(roe_list)) if roe_list else roe_ttm
                except Exception:
                    roe_avg = roe_ttm

                # EPS consistency
                try:
                    fin3 = t.financials
                    eps_list = []
                    if fin3 is not None and "Diluted EPS" in fin3.index:
                        for v in fin3.loc["Diluted EPS"].values[:3]:
                            if not pd.isna(v): eps_list.append(float(v))
                    elif fin3 is not None and "Basic EPS" in fin3.index:
                        for v in fin3.loc["Basic EPS"].values[:3]:
                            if not pd.isna(v): eps_list.append(float(v))
                    eps_list.reverse()  # oldest first
                except Exception:
                    eps_list = []

                sector = SECTOR_MAP.get(sym, "OTHER")

                r = ValueResult(
                    symbol=sym, name=name[:30], sector=sector,
                    price=round(price, 2), mkt_cap_cr=round(mkt_cap, 0),
                    pe=round(pe, 1), pb=round(pb, 2),
                    roe=round(roe_ttm, 1), roe_avg=round(roe_avg, 1),
                    revenue_cagr=round(rev_cagr, 1),
                    margin_now=round(margin, 1), margin_3y_ago=round(margin_old, 1),
                    debt_eq=round(de_ratio, 2), op_cashflow=op_cf,
                    eps_3y=eps_list, promoter_pct=round(promo, 1),
                )
                results.append(r)
                time.sleep(0.15)  # be polite to Yahoo

            except Exception as e:
                results.append(ValueResult(sym, error=str(e)[:60]))

        done = min(i+BATCH, len(symbols))
        print(f"  Progress: {done}/{len(symbols)} stocks fetched...")

    return results


# ── Sector median PE computation ──────────────────────────────────────────────

def compute_sector_medians(results: list[ValueResult]) -> dict[str, float]:
    sector_pes: dict[str, list] = {}
    for r in results:
        if r.pe > 0 and r.pe < 200 and not r.error:
            sector_pes.setdefault(r.sector, []).append(r.pe)
    return {sec: round(float(np.median(vals)), 1)
            for sec, vals in sector_pes.items() if vals}


# ── Output ────────────────────────────────────────────────────────────────────

def print_report(results: list[ValueResult], top_n: int = 20) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
    except ImportError:
        _plain_print(results, top_n); return

    now = datetime.now().strftime("%d %b %Y %H:%M")
    console.print(f"\n[bold]WealthOS Quality-Value Screener[/bold]  [dim]{now}  Mid & Small Cap Universe[/dim]\n")

    # Summary
    scored = [r for r in results if not r.error and r.score > 0]
    high = sum(1 for r in scored if r.rating == "HIGH CONVICTION")
    strong = sum(1 for r in scored if r.rating == "STRONG")
    console.print(f"Scanned {len(results)} stocks  |  High conviction: {high}  |  Strong: {strong}  |  Showing top {top_n}\n")

    for rating, color in [
        ("HIGH CONVICTION", "bold green"),
        ("STRONG",          "green"),
        ("MODERATE",        "yellow"),
    ]:
        group = [r for r in results[:top_n] if r.rating == rating]
        if not group: continue

        t = Table(title=f"[{color}]{rating}[/{color}]  ({len(group)} stocks)",
                  box=box.SIMPLE_HEAVY, header_style="dim")
        for col, w in [("Symbol",10),("Score",8),("Price",10),("Mkt Cap",10),
                       ("ROE%",8),("RevCAGR",8),("Margin",8),
                       ("D/E",6),("PE",6),("SectorPE",9),("Promo%",8),("Sector",12)]:
            t.add_column(col, justify="right" if col not in ("Symbol","Sector") else "left", width=w)

        for r in group:
            sc_pe = r.sector_median_pe
            pe_str = f"[green]{r.pe:.0f}[/green]" if r.pe > 0 and sc_pe > 0 and r.pe < sc_pe else f"{r.pe:.0f}"
            roe_str = f"[green]{r.roe_avg:.0f}%[/green]" if r.roe_avg >= 15 else f"{r.roe_avg:.0f}%"
            de_str = f"[green]{r.debt_eq:.2f}[/green]" if r.debt_eq <= 0.5 else f"[red]{r.debt_eq:.2f}[/red]"
            cagr_str = f"[green]{r.revenue_cagr:.0f}%[/green]" if r.revenue_cagr >= 15 else f"{r.revenue_cagr:.0f}%"
            mktcap_str = f"₹{r.mkt_cap_cr:,.0f}Cr"
            t.add_row(
                r.symbol, f"[{color}]{r.score}[/{color}]",
                f"₹{r.price:.1f}", mktcap_str,
                roe_str, cagr_str, f"{r.margin_now:.1f}%",
                de_str, pe_str, f"{sc_pe:.0f}" if sc_pe else "—",
                f"{r.promoter_pct:.0f}%", r.sector[:10],
            )
        console.print(t)

        if rating == "HIGH CONVICTION":
            for r in group[:3]:
                console.print(f"  [bold]{r.symbol}[/bold]: " + " · ".join(r.fired[:4]))
            console.print()

    # Warning box
    console.print("[dim]⚠  This screener identifies QUANTITATIVELY strong stocks. Always verify:[/dim]")
    console.print("[dim]   1. Read last 2yr annual report key highlights on screener.in[/dim]")
    console.print("[dim]   2. Check promoter pledge % (this screener uses insider% as proxy)[/dim]")
    console.print("[dim]   3. Verify the business hasn't changed fundamentally since last report[/dim]\n")


def _plain_print(results, top_n):
    print(f"\nQuality-Value Screener  {datetime.now().strftime('%d %b %Y')}")
    print(f"{'Symbol':12} {'Score':>6} {'ROE':>7} {'RevCGR':>8} {'D/E':>6} {'PE':>6} {'Sector'}")
    print("─"*70)
    for r in results[:top_n]:
        print(f"{r.symbol:12} {r.score:>6} {r.roe_avg:>7.1f}% {r.revenue_cagr:>7.1f}% "
              f"{r.debt_eq:>6.2f} {r.pe:>6.1f} {r.sector}")


def save_csv(results: list[ValueResult], path: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "symbol": r.symbol, "name": r.name, "sector": r.sector,
            "score": r.score, "rating": r.rating, "price": r.price,
            "mkt_cap_cr": r.mkt_cap_cr, "pe": r.pe, "pb": r.pb,
            "roe_avg": r.roe_avg, "revenue_cagr": r.revenue_cagr,
            "margin_now": r.margin_now, "margin_3y_ago": r.margin_3y_ago,
            "debt_eq": r.debt_eq, "promoter_pct": r.promoter_pct,
            "sector_median_pe": r.sector_median_pe,
            "signals_fired": " | ".join(r.fired),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Quality-Value screener for mid/small cap NSE stocks")
    p.add_argument("--top",       type=int,   default=20, help="Show top N results")
    p.add_argument("--min-score", type=int,   default=0,  help="Minimum score to show")
    p.add_argument("--sector",    type=str,   default="", help="Filter by sector")
    p.add_argument("--export",    type=Path,  default=None)
    p.add_argument("--symbols",   nargs="+",  help="Custom symbol list")
    a = p.parse_args()

    symbols = [s.upper() for s in a.symbols] if a.symbols else MIDCAP
    if a.sector:
        symbols = [s for s in symbols if SECTOR_MAP.get(s,"").upper() == a.sector.upper()]
        print(f"Filtered to {len(symbols)} stocks in sector {a.sector.upper()}")

    print(f"\nQuality-Value Screener — {len(symbols)} mid/small cap stocks")
    print(f"Universe: Nifty Midcap 150 + Nifty Smallcap 250 equivalent")
    print(f"Metrics: ROE (5yr) · Revenue growth · Margins · Debt · PE vs sector · Promoter\n")

    # Fetch
    results = fetch_fundamentals(symbols)

    # Compute sector medians from scanned data
    sector_medians = compute_sector_medians(results)

    # Score all
    for r in results:
        if r.error or r.price == 0:
            r.rating = "NO DATA"; continue
        r.sector_median_pe = sector_medians.get(r.sector, 0)
        r.score, r.fired = score_stock(r, sector_medians)
        r.score_pct = round(r.score / 100, 3)
        r.rating = rating_from_score(r.score)

    # Sort and filter
    results.sort(key=lambda r: (-r.score, r.symbol))
    if a.min_score:
        results = [r for r in results if r.score >= a.min_score]

    print_report(results, top_n=a.top)

    if a.export:
        save_csv(results, a.export)


if __name__ == "__main__":
    main()
