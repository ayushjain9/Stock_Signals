"""
WealthOS — NiftyShop Multi-Universe Backtest
============================================
Runs the NiftyShop mean-reversion strategy on three universes:
  A) Nifty 50        (baseline — already validated)
  B) Nifty200 Momentum 30  (high-momentum large-midcaps)
  C) Nifty Midcap 50       (midcap stocks with F&O)

⚠ SURVIVORSHIP BIAS WARNING
  Nifty200 Momentum 30 rebalances semi-annually (up to 20 changes/cycle).
  Nifty Midcap 50 rebalances quarterly.
  This backtest uses CURRENT constituents as a fixed universe.
  Results will be OPTIMISTIC vs real live performance.
  The comparison between universes is still valid on a relative basis.

STRATEGY RULES (identical for all three universes)
  Fresh buy:   ₹10,000 per stock
  Avg buy:     ₹15,000 (triggered at -3% below last buy)
  Max/stock:   ₹40,000
  Max stocks:  5 simultaneously
  Exit:        +8% above average buy price
  Priority:    SELL → AVERAGE → FRESH BUY

USAGE
-----
  python niftyshop_universe_backtest.py          # all 3 universes, 3yr
  python niftyshop_universe_backtest.py --years 2
  python niftyshop_universe_backtest.py --universe nifty50
  python niftyshop_universe_backtest.py --universe momentum30
  python niftyshop_universe_backtest.py --universe midcap50
"""
from __future__ import annotations
import argparse, warnings
from dataclasses import dataclass, field
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ── Universe definitions ──────────────────────────────────────────────────────

NIFTY50 = {
    "RELIANCE":"Reliance","HDFCBANK":"HDFC Bank","BHARTIARTL":"Airtel",
    "TCS":"TCS","ICICIBANK":"ICICI Bank","SBIN":"SBI","HINDUNILVR":"HUL",
    "INFY":"Infosys","BAJFINANCE":"Bajaj Finance","ITC":"ITC","LT":"L&T",
    "MARUTI":"Maruti","M&M":"M&M","HCLTECH":"HCL Tech","KOTAKBANK":"Kotak",
    "SUNPHARMA":"Sun Pharma","ULTRACEMCO":"Ultratech","TITAN":"Titan",
    "AXISBANK":"Axis Bank","NTPC":"NTPC","BAJAJFINSV":"Bajaj Finserv",
    "ONGC":"ONGC","ADANIPORTS":"Adani Ports","BEL":"BEL",
    "POWERGRID":"Power Grid","COALINDIA":"Coal India","NESTLEIND":"Nestle",
    "APOLLOHOSP":"Apollo Hosp","DIVISLAB":"Divi's","CIPLA":"Cipla",
    "TECHM":"Tech M","TATACONSUM":"Tata Consumer","JSWSTEEL":"JSW Steel",
    "INDUSINDBK":"IndusInd","ASIANPAINT":"Asian Paints",
    "HINDALCO":"Hindalco","EICHERMOT":"Eicher","HEROMOTOCO":"Hero Moto",
    "BRITANNIA":"Britannia","DRREDDY":"Dr Reddy","BAJAJ-AUTO":"Bajaj Auto",
    "GRASIM":"Grasim","TATASTEEL":"Tata Steel","WIPRO":"Wipro",
    "TRENT":"Trent","ADANIENT":"Adani Ent","SHRIRAMFIN":"Shriram Fin",
    "UPL":"UPL","TATAMOTORS":"Tata Motors","SBICARD":"SBI Cards",
}

# Current Nifty200 Momentum 30 constituents (May 2026)
# ⚠ Survivorship bias: index changed ~20 stocks in June 2025 rebalancing
MOMENTUM30 = {
    "BHARTIARTL":"Bharti Airtel","SBIN":"State Bank of India",
    "BAJFINANCE":"Bajaj Finance","MARUTI":"Maruti Suzuki",
    "SHRIRAMFIN":"Shriram Finance","ASIANPAINT":"Asian Paints",
    "HINDALCO":"Hindalco Industries","EICHERMOT":"Eicher Motors",
    "SBILIFE":"SBI Life Insurance","TVSMOTOR":"TVS Motor",
    "INDIGO":"InterGlobe Aviation","MUTHOOTFIN":"Muthoot Finance",
    "HDFCBANK":"HDFC Bank","KOTAKBANK":"Kotak Bank",
    "M&M":"Mahindra & Mahindra","ADANIPORTS":"Adani Ports",
    "TATAPOWER":"Tata Power","HEROMOTOCO":"Hero MotoCorp",
    "OFSS":"Oracle Financial Services","ICICIBANK":"ICICI Bank",
    "AXISBANK":"Axis Bank","BAJAJFINSV":"Bajaj Finserv",
    "HCLTECH":"HCL Technologies","NTPC":"NTPC",
    "POWERGRID":"Power Grid","COALINDIA":"Coal India",
    "APOLLOHOSP":"Apollo Hospitals","TITAN":"Titan",
    "BEL":"BEL","ONGC":"ONGC",
}

# Current Nifty Midcap 50 constituents (May 2026)
# ⚠ Survivorship bias: rebalances quarterly
MIDCAP50 = {
    "BSE":"BSE Ltd","BHEL":"BHEL","POLYCAB":"Polycab India",
    "LUPIN":"Lupin","INDUSTOWER":"Indus Towers","MARICO":"Marico",
    "GMRAIRPORTS":"GMR Airports","HINDPETRO":"HPCL","NHPC":"NHPC",
    "DABUR":"Dabur India","SRF":"SRF Ltd","PERSISTENT":"Persistent Systems",
    "HAVELLS":"Havells India","POLICYBZR":"PB Fintech","NMDC":"NMDC",
    "MCX":"MCX","FEDERALBNK":"Federal Bank","SUZLON":"Suzlon Energy",
    "AUBANK":"AU Small Finance Bank","IDFCFIRSTB":"IDFC First Bank",
    "OBEROIRLTY":"Oberoi Realty","MFSL":"Max Financial Services",
    "ABCAPITAL":"Aditya Birla Capital","BALKRISIND":"Balkrishna Ind",
    "TATACHEM":"Tata Chemicals","CESC":"CESC","PIIND":"PI Industries",
    "ZYDUSLIFE":"Zydus Lifesciences","KAJARIACER":"Kajaria Ceramics",
    "VOLTAS":"Voltas","SUNDARMFIN":"Sundaram Finance",
    "ASTRAL":"Astral Ltd","CONCOR":"Container Corp",
    "GLENMARK":"Glenmark Pharma","GODREJPROP":"Godrej Properties",
    "INDHOTEL":"Indian Hotels","JKCEMENT":"JK Cement",
    "LTTS":"L&T Tech Services","MRF":"MRF",
    "PAGEIND":"Page Industries","PHOENIXLTD":"Phoenix Mills",
    "PRESTIGE":"Prestige Estates","RAMCOCEM":"Ramco Cements",
    "SYNGENE":"Syngene International","TATACOMM":"Tata Comms",
    "TORNTPHARM":"Torrent Pharma","ABFRL":"Aditya Birla Fashion",
    "CHAMBLFERT":"Chambal Fertilisers","CRISIL":"CRISIL",
    "METROPOLIS":"Metropolis Healthcare",
}

UNIVERSES = {
    "nifty50":     ("Nifty 50 (50 large caps)",             NIFTY50,     "^NSEI"),
    "momentum30":  ("Nifty200 Momentum 30 ⚠bias",           MOMENTUM30,  "^NSEI"),
    "midcap50":    ("Nifty Midcap 50 ⚠bias",               MIDCAP50,    "^NSMIDCP"),
}

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    fresh:     float = 10_000
    avg:       float = 15_000
    max_stock: float = 40_000
    max_pos:   int   = 5
    target:    float = 0.08
    trig:      float = 0.03
    top_n:     int   = 5
    cost:      float = 0.001

# ── Portfolio types ───────────────────────────────────────────────────────────
@dataclass
class Lot:
    date:  object   # pd.Timestamp of the buy
    price: float
    qty:   float
    amt:   float

@dataclass
class Pos:
    lots: list[Lot] = field(default_factory=list)

    @property
    def invested(self): return sum(l.amt for l in self.lots)
    @property
    def qty(self):      return sum(l.qty for l in self.lots)
    @property
    def avg_px(self):
        q = self.qty
        return sum(l.price * l.qty for l in self.lots) / q if q > 0 else 0
    @property
    def last_px(self):  return self.lots[-1].price if self.lots else 0
    def can_avg(self, mx): return self.invested < mx - 0.01
    def avg_trig(self, cur, t): return cur <= self.last_px * (1 - t)
    def exit_trig(self, cur, t): return cur >= self.avg_px * (1 + t)
    def pnl(self, cur): return (cur - self.avg_px) * self.qty

# ── Engine ────────────────────────────────────────────────────────────────────
def run(universe: dict, close: pd.DataFrame, bench_col: str,
        cfg: Cfg, si: int = 25, ei: int | None = None) -> dict:

    tickers = {sym: sym + ".NS" for sym in universe}
    bench_s = close[bench_col].dropna() if bench_col in close.columns else None
    if bench_s is None:
        print(f"  Warning: benchmark {bench_col} not found, using ^NSEI")
        bench_s = close.get("^NSEI", close.iloc[:, 0]).dropna()

    dates = bench_s.index
    ei    = ei or len(dates)

    cash = 400_000.0
    positions: dict[str, Pos] = {}
    nav_list, deployed_list, tlog = [], [], []

    for i in range(max(si, 25), ei):
        date = dates[i]

        # 20DMA deviation for all stocks
        scores: dict[str, tuple[float, float]] = {}
        for sym, col in tickers.items():
            if col not in close.columns: continue
            hist = close[col].reindex(dates).ffill().iloc[:i].dropna()
            if len(hist) < 20: continue
            price = float(hist.iloc[-1])
            dma20 = float(hist.tail(20).mean())
            if price <= 0 or dma20 <= 0: continue
            scores[sym] = ((price - dma20) / dma20 * 100, price)

        today = {sym: v[1] for sym, v in scores.items()}

        # STEP 1: exits
        to_exit = [sym for sym, pos in positions.items()
                   if sym in today and pos.exit_trig(today[sym], cfg.target)]
        for sym in to_exit:
            pos      = positions[sym]
            px       = today[sym] * (1 - cfg.cost)
            pnl      = pos.pnl(px)
            entry_dt = pos.lots[0].date
            hold_days = (date - entry_dt).days if hasattr(date - entry_dt, "days") else 0
            cash    += pos.qty * px
            tlog.append({"action": "SELL", "pnl": pnl,
                          "hold": hold_days, "invested": pos.invested})
            del positions[sym]

        # STEP 2: fresh or average
        below = sorted([(d, s, p) for s, (d, p) in scores.items() if d < 0],
                       key=lambda x: x[0])
        top5  = [s for _, s, _ in below[:cfg.top_n]]
        acted = False

        # fresh entry
        if not acted:
            for sym in top5:
                if sym in positions or len(positions) >= cfg.max_pos: continue
                if cash < cfg.fresh: break
                px     = today[sym] * (1 + cfg.cost)
                qty    = cfg.fresh / px
                cash  -= cfg.fresh
                p       = Pos(); p.lots.append(Lot(date, px, qty, cfg.fresh))
                positions[sym] = p
                tlog.append({"action": "BUY_FRESH", "pnl": 0,
                              "hold": 0, "invested": cfg.fresh})
                acted = True; break

        # averaging
        if not acted and positions:
            cands = [(today.get(s, 0) / pos.avg_px - 1,  s, pos, today.get(s, 0))
                     for s, pos in positions.items()
                     if s in today
                     and pos.avg_trig(today[s], cfg.trig)
                     and pos.can_avg(cfg.max_stock)]
            if cands:
                cands.sort()
                _, sym, pos, px = cands[0]
                amt  = min(cfg.avg, cfg.max_stock - pos.invested, cash)
                if amt >= 1000:
                    buy_px = px * (1 + cfg.cost)
                    qty    = amt / buy_px
                    cash  -= amt
                    pos.lots.append(Lot(date, buy_px, qty, amt))
                    tlog.append({"action": "BUY_AVG", "pnl": 0,
                                  "hold": 0, "invested": amt})

        # NAV
        deployed = 0.0
        nav      = cash
        for sym, pos in positions.items():
            if sym in today:
                nav      += pos.qty * today[sym]
                deployed += pos.invested
        nav_list.append(nav)
        deployed_list.append(deployed)

    # metrics
    nav_arr  = np.array(nav_list)
    dep_arr  = np.array(deployed_list)
    n_years  = len(nav_arr) / 252

    cagr     = (nav_arr[-1] / nav_arr[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak     = np.maximum.accumulate(nav_arr)
    mdd      = ((nav_arr - peak) / peak).min()
    util     = dep_arr.mean() / 400_000

    exits    = [t for t in tlog if t["action"] == "SELL"]
    tot_pnl  = sum(t["pnl"] for t in exits)
    win_rate = (np.array([t["pnl"] for t in exits]) > 0).mean() if exits else 0
    avg_dep  = dep_arr.mean()
    xirr     = tot_pnl / avg_dep / n_years if avg_dep > 0 else 0

    fresh    = [t for t in tlog if t["action"] == "BUY_FRESH"]
    avgs     = [t for t in tlog if t["action"] == "BUY_AVG"]

    return dict(
        cagr         = round(cagr * 100, 2),
        xirr         = round(xirr * 100, 2),
        mdd          = round(mdd * 100, 2),
        util         = round(util * 100, 1),
        win_rate     = round(win_rate * 100, 1),
        total_pnl    = round(tot_pnl, 0),
        n_fresh      = len(fresh),
        n_avg        = len(avgs),
        n_exits      = len(exits),
        n_years      = round(n_years, 1),
        final_nav    = round(float(nav_arr[-1]), 0),
        nav_arr      = nav_arr,
    )

# ── Output ────────────────────────────────────────────────────────────────────
def print_result(label: str, r: dict, bench_cagr: float) -> None:
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    print(f"  CAGR on ₹4L total:        {r['cagr']:>8.2f}%  (benchmark {bench_cagr:.2f}%)")
    print(f"  XIRR on deployed capital: {r['xirr']:>8.2f}%")
    print(f"  Capital utilisation:      {r['util']:>8.1f}%")
    print(f"  Max drawdown:             {r['mdd']:>8.2f}%")
    print(f"  Win rate:                 {r['win_rate']:>8.1f}%")
    print(f"  Total P&L (closed):      ₹{r['total_pnl']:>8,.0f}")
    print(f"  Final portfolio value:   ₹{r['final_nav']:>8,.0f}")
    print(f"  Fresh buys:               {r['n_fresh']:>8d}")
    print(f"  Averaging buys:           {r['n_avg']:>8d}")
    print(f"  Exits:                    {r['n_exits']:>8d}")
    print(f"  Period:                   {r['n_years']:>8.1f} years")

def print_comparison(results: dict, bench_cagr: float) -> None:
    print(f"\n{'='*78}")
    print("  UNIVERSE COMPARISON — NiftyShop (₹10K fresh, ₹15K avg, +8% exit)")
    print(f"  ⚠  Momentum30 and Midcap50 results have survivorship bias — will be optimistic")
    print(f"{'='*78}")

    headers = ["Metric", "Nifty 50", "Momentum 30 ⚠", "Midcap 50 ⚠"]
    print(f"\n  {headers[0]:<30} {headers[1]:>14} {headers[2]:>15} {headers[3]:>13}")
    print(f"  {'─'*29} {'─'*14} {'─'*15} {'─'*13}")

    rows = [
        ("CAGR on ₹4L total",       "cagr",      "%"),
        ("vs benchmark",             "__bench",   "pp"),
        ("XIRR deployed capital",    "xirr",      "%"),
        ("Capital utilisation",      "util",      "%"),
        ("Max drawdown",             "mdd",       "%"),
        ("Win rate",                 "win_rate",  "%"),
        ("Total closed P&L",         "total_pnl", "₹"),
        ("Fresh buys",               "n_fresh",   ""),
        ("Averaging buys",           "n_avg",     ""),
        ("Exits",                    "n_exits",   ""),
    ]

    keys = ["nifty50", "momentum30", "midcap50"]
    for label, key, unit in rows:
        vals = []
        for k in keys:
            if k not in results:
                vals.append("N/A")
                continue
            if key == "__bench":
                v = results[k]["cagr"] - bench_cagr
                vals.append(f"{v:+.2f}pp")
            elif key == "total_pnl":
                vals.append(f"₹{results[k][key]:,.0f}")
            elif unit in ("%", "pp"):
                vals.append(f"{results[k][key]:.2f}{unit}")
            else:
                vals.append(str(results[k][key]))

        v0, v1, v2 = vals
        print(f"  {label:<30} {v0:>14} {v1:>15} {v2:>13}")

    print(f"\n  Benchmark CAGR (Nifty 50): {bench_cagr:.2f}%")
    print()
    print("  KEY INTERPRETATION:")
    print("  • Higher XIRR = mean-reversion edge is stronger in that universe")
    print("  • Higher utilisation = more trading opportunities")
    print("  • Lower MDD = safer for capital preservation")
    print("  • More fresh buys = more stocks fell below 20DMA (choppy universe)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years",     type=int, default=3)
    p.add_argument("--universe",  type=str, default="all",
                   help="all | nifty50 | momentum30 | midcap50")
    a = p.parse_args()

    cfg = Cfg()
    to_run = list(UNIVERSES.keys()) if a.universe == "all" \
             else [a.universe] if a.universe in UNIVERSES \
             else list(UNIVERSES.keys())

    # Gather all needed tickers
    all_tickers = set()
    for key in to_run:
        _, univ, bench = UNIVERSES[key]
        all_tickers.update(sym + ".NS" for sym in univ)
        all_tickers.add(bench)
    all_tickers.add("^NSEI")
    all_tickers.add("^NSMIDCP")

    period = f"{a.years + 1}y"
    print(f"\n  NiftyShop Universe Backtest — {a.years}yr")
    print(f"  Strategy: ₹10K fresh | ₹15K avg | +8% exit | 5 stocks max")
    print(f"  ⚠  Momentum30 + Midcap50 use current constituents → survivorship bias")
    print(f"\n  Downloading {len(all_tickers)} tickers ({period})...")

    import yfinance as yf
    raw   = yf.download(list(all_tickers), period=period,
                        progress=False, auto_adjust=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    # Benchmark for comparison (Nifty 50)
    bench_s    = close["^NSEI"].dropna()
    bench_arr  = bench_s.values.astype(float)
    n_yr       = len(bench_arr) / 252
    bench_cagr = (bench_arr[-1] / bench_arr[0]) ** (1/n_yr) - 1 if n_yr > 0 else 0

    results = {}
    for key in to_run:
        label, univ, bench_col = UNIVERSES[key]
        print(f"\n  Running {label}...")
        r = run(univ, close, bench_col, cfg)
        results[key] = r
        print_result(label, r, bench_cagr * 100)

    if len(results) > 1:
        print_comparison(results, bench_cagr * 100)

    print(f"\n  ⚠  REMINDER: Momentum30 and Midcap50 results are OPTIMISTIC due to")
    print(f"     survivorship bias. Real live performance will be lower.")
    print(f"     Use Nifty 50 results as your anchor — no survivorship bias there.\n")

if __name__ == "__main__":
    main()
