"""
WealthOS — Point-in-Time Backtest Harness
=========================================

WHY THIS FILE EXISTS
--------------------
Every backtest in Backtest/ pulls `period="6y"` of *today's* index constituents.
That is survivorship bias: stocks that were dropped from the index (usually after
falling) are excluded, and stocks that were promoted into it (after rising) are
included for years before they actually qualified. Your own research already says
so — backtest_momentum.py prints a SURVIVORSHIP BIAS WARNING and nifty100_combined
concludes "survivorship bias was the entire story." This harness fixes it.

WHAT "POINT-IN-TIME" MEANS HERE
-------------------------------
At every rebalance date the investable universe is reconstructed as it (approximately)
was *on that date*, not as it is today:

  • Nifty 50  → ranked top 50 by trailing traded value from the Nifty 100 superset
                (your Nifty 50 + Nifty Next 50). A name only enters the universe once
                it is actually large/liquid enough; a name that shrinks drops out and
                its subsequent losses are NOT magically removed from history.
  • Midcap 50 → see HONEST LIMITATIONS below. Without a midcap-150 superset this
                degrades to snapshot; supply --midcap-superset or a membership CSV.

This validates the DEPLOYED code: it imports score_stock() and select_holdings()
straight from nifty50_Momentum_deploy.py. The momentum numbers it prints are the
numbers your live screener would have produced, on the universe that actually existed.

HONEST LIMITATIONS (read these before quoting any number)
---------------------------------------------------------
  1. The Nifty Next 50 superset (config + this file) is *current*. Names that fell
     out of the Nifty 100 entirely (rare for recent Nifty-50 members) are still
     missing → a small residual survivorship bias remains. This is a big reduction,
     not a perfect elimination. The gold standard is a real NSE constituents file;
     pass it with --membership-n50 <csv> (schema: symbol,start,end).
  2. Truly delisted/merged names with no Yahoo history (e.g. HDFC Ltd post-merger)
     cannot be priced and are skipped. Their absence still understates downside.
  3. yfinance is the only source. Bad splits / missing days flow straight through.
     This harness reports how many symbols failed to load — it does NOT hide them.
  4. Midcap momentum is NOT walk-forward validated (your own deployer says so). Treat
     midcap PIT numbers as directional until you supply a proper midcap superset.

USAGE
-----
  python pit_backtest.py                                  # all strategies, 1/2/3y, mode=both
  python pit_backtest.py --strategy momentum-n50          # one strategy
  python pit_backtest.py --years 3                        # single horizon
  python pit_backtest.py --mode pit                       # PIT only (the trustworthy number)
  python pit_backtest.py --mode snapshot                  # reproduce the biased number
  python pit_backtest.py --cost 0.0025 --rf 0.065         # tune costs / risk-free
  python pit_backtest.py --membership-n50 nifty50_members.csv   # gold-standard PIT
  python pit_backtest.py --midcap-superset midcap150.txt       # better midcap PIT

OPTIONAL real-membership CSV schema (one row per (symbol, spell in index)):
  symbol,start,end
  HDFCBANK,,                 # in for the whole window
  JIOFIN,2023-09-01,         # entered 2023-09-01, still in
  HDFC,,2023-07-13           # left on merger date
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import the ACTUAL deployed logic (single source of truth) ─────────────────
# score_stock and select_holdings are byte-identical between the Nifty 50 and
# Midcap deployers (the midcap docstring states "identical signal stack"), so we
# reuse the Nifty 50 versions for both universes. StockResult / Holding are the
# dataclasses select_holdings() consumes.
from nifty50_Momentum_deploy import (  # noqa: E402
    score_stock,
    select_holdings,
    StockResult,
    Holding,
)
from config import (  # noqa: E402
    NIFTY50_LIST,
    NIFTY50_SECTOR_MAP,
    MIDCAP50_LIST,
    MIDCAP50_SECTOR_MAP,
    NIFTY50_BENCHMARK,
    MIDCAP50_BENCHMARK,
)

MIN_HISTORY = 210
TRADING_DAYS = 252
REBAL_STEP = 21      # momentum: monthly
WEEK_STEP = 5        # niftyshop: weekly


# ── Nifty Next 50 superset (for PIT reconstruction of the Nifty 50) ───────────
# Harvested from Backtest/nifty100_combined_backtest.py. Current (May 2026) — see
# HONEST LIMITATIONS #1. Overlaps with Nifty 50 are removed at build time.
NIFTY_NEXT50 = [
    "ADANIPOWER", "DMART", "VEDL", "HAL", "TVSMOTOR", "VBL", "ADANIGREEN",
    "ADANIENERGYSOL", "CUMMINSIND", "PFC", "RECLTD", "DLF", "GODREJCP", "MUTHOOTFIN",
    "IRCTC", "CHOLAFIN", "SIEMENS", "ABB", "PIDILITIND", "HDFCAMC", "ICICIPRULI",
    "LODHA", "ZOMATO", "NAUKRI", "SBICARD", "IOC", "GAIL", "INDIGO", "NHPC",
    "TATAPOWER", "LICI", "BAJAJHLDNG", "INDUSTOWER", "COLPAL", "OFSS", "AMBUJACEM",
    "ICICIGI", "MARICO", "DABUR", "BERGEPAINT", "JUBLFOOD", "PAGEIND", "MCDOWELL-N",
    "TORNTPOWER", "PIIND", "LUPIN", "AUROPHARMA", "ALKEM", "MAXHEALTH",
]

# Sector map for Next-50 names not already in NIFTY50_SECTOR_MAP (for sector cap).
NEXT50_SECTOR_MAP = {
    "ADANIPOWER": "POWER", "DMART": "RETAIL", "VEDL": "METALS", "HAL": "DEFENCE",
    "TVSMOTOR": "AUTO", "VBL": "FMCG", "ADANIGREEN": "POWER", "ADANIENERGYSOL": "POWER",
    "CUMMINSIND": "CAPITAL_GOODS", "PFC": "POWER", "RECLTD": "POWER", "DLF": "REALTY",
    "GODREJCP": "FMCG", "MUTHOOTFIN": "FINANCE", "IRCTC": "LOGISTICS", "CHOLAFIN": "NBFC",
    "SIEMENS": "CAPITAL_GOODS", "ABB": "CAPITAL_GOODS", "PIDILITIND": "CHEMICALS",
    "HDFCAMC": "FINANCE", "ICICIPRULI": "INSURANCE", "LODHA": "REALTY",
    "ZOMATO": "E-commerce", "NAUKRI": "IT", "SBICARD": "FINANCE", "IOC": "ENERGY",
    "GAIL": "ENERGY", "INDIGO": "Aviation", "NHPC": "POWER", "TATAPOWER": "POWER",
    "LICI": "INSURANCE", "BAJAJHLDNG": "FINANCE", "INDUSTOWER": "TELECOM",
    "COLPAL": "FMCG", "OFSS": "IT", "AMBUJACEM": "CEMENT", "ICICIGI": "INSURANCE",
    "MARICO": "FMCG", "DABUR": "FMCG", "BERGEPAINT": "PAINTS", "JUBLFOOD": "CONSUMER",
    "PAGEIND": "CONSUMER", "MCDOWELL-N": "FMCG", "TORNTPOWER": "POWER",
    "PIIND": "CHEMICALS", "LUPIN": "PHARMA", "AUROPHARMA": "PHARMA", "ALKEM": "PHARMA",
    "MAXHEALTH": "HEALTHCARE",
}


# ── Strategy specs ────────────────────────────────────────────────────────────
@dataclass
class MomentumSpec:
    key:          str
    label:        str
    snapshot:     list[str]            # today's index constituents
    superset:     list[str]            # PIT reconstruction pool (may == snapshot)
    sector_map:   dict[str, str]
    benchmark:    str
    top_n:        int
    sector_cap:   int
    liquidity_cr: float
    index_size:   int                  # how many names the index holds
    pit_capable:  bool                 # False → pit degrades to snapshot (warn)


# ── Data download (loud about failures, cached within a run) ──────────────────
_DL_CACHE: dict[tuple, tuple] = {}


def download(symbols: list[str], benchmark: str, years: int) -> tuple[pd.DataFrame, ...]:
    import yfinance as yf
    key = (tuple(sorted(symbols)), benchmark, years)
    if key in _DL_CACHE:
        print(f"  (reusing cached download of {len(symbols)+1} tickers)")
        return _DL_CACHE[key]
    period = f"{years}y"
    tickers = [s + ".NS" for s in symbols] + [benchmark]
    print(f"  Downloading {len(tickers)} tickers, period={period} ...")
    raw = yf.download(tickers, period=period, progress=False,
                      auto_adjust=True, threads=True)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Unexpected single-ticker frame — need >1 ticker.")
    cl, hi, lo, vo = raw["Close"], raw["High"], raw["Low"], raw["Volume"]

    # Report what actually loaded — never silently drop.
    wanted = set(s + ".NS" for s in symbols)
    got = set(c for c in cl.columns if cl[c].notna().sum() >= MIN_HISTORY)
    missing = sorted(s for s in wanted if s not in got)
    if missing:
        print(f"  ⚠  {len(missing)}/{len(wanted)} symbols unusable "
              f"(<{MIN_HISTORY} bars or not on Yahoo): {', '.join(m[:-3] for m in missing)}")
    if benchmark not in cl.columns or cl[benchmark].notna().sum() < MIN_HISTORY:
        raise RuntimeError(f"Benchmark {benchmark} failed to load — cannot proceed.")
    _DL_CACHE[key] = (cl, hi, lo, vo)
    return cl, hi, lo, vo


# ── Aligned arrays ────────────────────────────────────────────────────────────
def align(series_frame: pd.DataFrame, col: str, index: pd.DatetimeIndex) -> np.ndarray:
    if col not in series_frame.columns:
        return np.full(len(index), np.nan)
    return series_frame[col].reindex(index).values.astype(float)


# ── Point-in-time membership ──────────────────────────────────────────────────
class Membership:
    """Returns the investable set as of any rebalance day index."""

    def __init__(self, mode: str, spec: MomentumSpec, bd: pd.DatetimeIndex,
                 closes: dict, vols: dict, real_csv: Path | None,
                 reconstitute_days: int = 126):
        self.mode = mode
        self.spec = spec
        self.bd = bd
        self.closes = closes
        self.vols = vols
        self.recon = reconstitute_days
        self._cache: dict[int, set[str]] = {}
        self.real = self._load_csv(real_csv) if real_csv else None

    def _load_csv(self, path: Path) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        df = pd.read_csv(path)
        rows = []
        for _, r in df.iterrows():
            s = str(r["symbol"]).strip().upper()
            start = pd.to_datetime(r["start"]) if str(r.get("start", "")).strip() else pd.Timestamp.min
            end = pd.to_datetime(r["end"]) if str(r.get("end", "")).strip() else pd.Timestamp.max
            rows.append((s, start, end))
        print(f"  Using real membership file: {len(rows)} spells for {len({r[0] for r in rows})} symbols")
        return rows

    def asof(self, day: int) -> set[str]:
        if self.real is not None:
            d = self.bd[day]
            return {s for (s, a, b) in self.real if a <= d <= b}
        if self.mode == "snapshot" or not self.spec.pit_capable:
            return set(self.spec.snapshot)
        # PIT: reconstitute on a fixed cadence, hold membership between.
        anchor = day - (day % self.recon)
        if anchor not in self._cache:
            self._cache[anchor] = self._rank_top(anchor)
        return self._cache[anchor]

    def _rank_top(self, day: int) -> set[str]:
        """Top index_size names by trailing 126-day average traded value."""
        lo = max(0, day - 126)
        tv = {}
        for sym in self.spec.superset:
            c = self.closes.get(sym)
            v = self.vols.get(sym)
            if c is None or v is None:
                continue
            cc, vv = c[lo:day + 1], v[lo:day + 1]
            m = ~np.isnan(cc) & ~np.isnan(vv)
            # need enough history to even be scoreable later
            if np.count_nonzero(~np.isnan(c[:day + 1])) < MIN_HISTORY or m.sum() < 60:
                continue
            tv[sym] = float(np.nanmean(cc[m] * vv[m]))
        ranked = sorted(tv, key=lambda s: -tv[s])
        return set(ranked[: self.spec.index_size])


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(port: pd.Series, bench: pd.Series, trade_rets: list[float],
            turnovers: list[float], rf: float) -> dict:
    port = port.dropna()
    bench = bench.reindex(port.index).ffill()
    rp = port.pct_change().dropna()
    rb = bench.pct_change().reindex(rp.index).fillna(0)
    n_years = (port.index[-1] - port.index[0]).days / 365.25
    n_years = max(n_years, 1e-9)

    cagr_p = port.iloc[-1] ** (1 / n_years) - 1      # port starts at 1.0
    cagr_b = (bench.iloc[-1] / bench.iloc[0]) ** (1 / n_years) - 1

    daily_rf = rf / TRADING_DAYS
    ex = rp - daily_rf
    sharpe = ex.mean() / rp.std() * np.sqrt(TRADING_DAYS) if rp.std() > 0 else 0.0
    downside = rp[rp < 0].std()
    sortino = ex.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else 0.0

    if rb.var() > 0:
        beta = np.cov(rp, rb)[0, 1] / rb.var()
    else:
        beta = 1.0
    jensen = (cagr_p - rf) - beta * (cagr_b - rf)

    peak = port.cummax()
    mdd = ((port - peak) / peak).min()
    bpeak = bench.cummax()
    bmdd = ((bench - bpeak) / bpeak).min()
    calmar = cagr_p / abs(mdd) if mdd < 0 else 0.0

    # monthly hit ratio
    pm = port.resample("ME").last().pct_change().dropna()
    bm = bench.resample("ME").last().pct_change().reindex(pm.index).fillna(0)
    hit_m = (pm > bm).mean() * 100 if len(pm) else 0.0
    hit_t = (np.array(trade_rets) > 0).mean() * 100 if trade_rets else 0.0
    ann_to = (np.mean(turnovers) * (TRADING_DAYS / REBAL_STEP) * 100) if turnovers else 0.0

    return dict(
        cagr=cagr_p * 100, bench_cagr=cagr_b * 100, alpha=(cagr_p - cagr_b) * 100,
        jensen=jensen * 100, beta=beta, sharpe=sharpe, sortino=sortino,
        mdd=mdd * 100, bmdd=bmdd * 100, calmar=calmar,
        hit_m=hit_m, hit_t=hit_t, ann_to=ann_to,
        n_trades=len(trade_rets), n_years=n_years,
    )


# ── Momentum engine ───────────────────────────────────────────────────────────
def run_momentum(spec: MomentumSpec, mode: str, years: int, cost: float,
                 rf: float, real_csv: Path | None,
                 data_years: int) -> dict | None:
    pool = sorted(set(spec.superset) | set(spec.snapshot))
    cl, hi, lo, vo = download(pool, spec.benchmark, data_years)

    bench_s = cl[spec.benchmark].dropna()
    bd = bench_s.index
    bench_arr = bench_s.values.astype(float)

    closes = {s: align(cl, s + ".NS", bd) for s in pool}
    highs = {s: align(hi, s + ".NS", bd) for s in pool}
    lows = {s: align(lo, s + ".NS", bd) for s in pool}
    vols = {s: align(vo, s + ".NS", bd) for s in pool}

    member = Membership(mode, spec, bd, closes, vols, real_csv)

    # window start: `years` back from the last bar, but never before we have MIN_HISTORY
    last = bd[-1]
    win_start = last - pd.DateOffset(years=years)
    si = int(np.searchsorted(bd.values, np.datetime64(win_start)))
    si = max(si, MIN_HISTORY)
    ei = len(bd) - 1
    if ei - si < REBAL_STEP * 3:
        print(f"  ✗ Not enough data for {years}y window (have {ei - si} bars after lookback).")
        return None

    nav = pd.Series(np.nan, index=bd[si:ei + 1], dtype=float)
    nav.iloc[0] = 1.0
    positions: dict[str, Holding] = {}
    trade_rets: list[float] = []
    turnovers: list[float] = []

    rebal_days = list(range(si, ei, REBAL_STEP))
    for t in rebal_days:
        members = member.asof(t)
        scored: list[StockResult] = []
        for sym in members:
            a = closes[sym][: t + 1]
            idx = np.where(~np.isnan(a))[0]
            if len(idx) < MIN_HISTORY:
                continue
            c = a[idx]
            h = highs[sym][: t + 1][idx]
            l = lows[sym][: t + 1][idx]
            v = vols[sym][: t + 1][idx]
            b = bench_arr[: t + 1]
            b = b[~np.isnan(b)]
            b = b[-len(c):] if len(b) >= len(c) else b
            price = c[-1]
            avg_vol = np.nanmean(v[-20:]) if len(v) >= 20 else np.nanmean(v)
            liq_cr = avg_vol * price / 1e7
            liq_ok = liq_cr >= spec.liquidity_cr
            sc = score_stock(c, h, l, b, v) if liq_ok else -1.0
            scored.append(StockResult(
                symbol=sym, score=sc, price=round(float(price), 2),
                dma25=float(c[-25:].mean()), dma50=float(c[-50:].mean()),
                dma200=float(c[-200:].mean()),
                sector=spec.sector_map.get(sym, "OTHER"),
                liq_cr=round(liq_cr, 1), liq_ok=liq_ok,
                rs_90d=0.0, roc_3m=0.0, vam=0.0, pct_52h=0.0,
            ))
        scored.sort(key=lambda r: -r.score)

        final, to_buy, to_sell, to_keep, score_map = select_holdings(
            scored, positions, top_n=spec.top_n, sec_cap=spec.sector_cap,
        )
        price_now = {r.symbol: r.price for r in scored}

        # realise returns on exits
        for sym in to_sell:
            h = positions.get(sym)
            px = price_now.get(sym, 0.0)
            if h and h.entry_price > 0 and px > 0:
                trade_rets.append(px / h.entry_price - 1)

        # update holdings (mirror deployer --save behaviour)
        new_positions = {s: positions[s] for s in to_keep}
        for sym in to_buy:
            new_positions[sym] = Holding(sym, price_now.get(sym, 0.0),
                                         score_map.get(sym, 0.0),
                                         str(bd[t].date()))
        positions = new_positions

        # turnover + cost
        n = max(len(final), 1)
        one_way = (len(to_buy) + len(to_sell)) / (2 * n)
        turnovers.append(one_way)
        cost_frac = (len(to_buy) + len(to_sell)) / n * cost

        # daily NAV over [t, nt], equal weight, buy-and-hold between rebalances
        nt = min(t + REBAL_STEP, ei)
        held = [s for s in final if not np.isnan(closes[s][t]) and closes[s][t] > 0]
        base = nav.loc[bd[t]] * (1 - cost_frac)
        w = 1.0 / len(held) if held else 0.0
        p0 = {s: closes[s][t] for s in held}
        for d in range(t + 1, nt + 1):
            gross = 0.0
            for s in held:
                pd_ = closes[s][d]
                if np.isnan(pd_):
                    pd_ = closes[s][d - 1] if d - 1 >= 0 and not np.isnan(closes[s][d - 1]) else p0[s]
                gross += w * (pd_ / p0[s])
            nav.loc[bd[d]] = base * (gross if held else 1.0)

    bench_nav = pd.Series(bench_arr[si:ei + 1] / bench_arr[si], index=bd[si:ei + 1])
    m = metrics(nav, bench_nav, trade_rets, turnovers, rf)
    m["strategy"] = spec.label
    m["mode"] = mode if spec.pit_capable or mode == "snapshot" else "snapshot*"
    m["years"] = years
    return m


# ── NiftyShop (mean-reversion) engine ─────────────────────────────────────────
@dataclass
class _Pos:
    qty: float = 0.0
    invested: float = 0.0
    last_buy: float = 0.0
    cost_basis: float = 0.0  # for avg price = cost_basis / qty

    @property
    def avg(self) -> float:
        return self.cost_basis / self.qty if self.qty > 0 else 0.0


def run_niftyshop(spec: MomentumSpec, mode: str, years: int, cost: float, rf: float,
                  real_csv: Path | None, data_years: int,
                  capital: float, fresh: float, avg_amt: float,
                  max_stock: float, profit: float = 0.08, avg_trig: float = 0.03,
                  max_pos: int = 5, idle_yield: float = 0.065) -> dict | None:
    pool = sorted(set(spec.superset) | set(spec.snapshot))
    cl, hi, lo, vo = download(pool, spec.benchmark, data_years)
    bench_s = cl[spec.benchmark].dropna()
    bd = bench_s.index
    bench_arr = bench_s.values.astype(float)
    closes = {s: align(cl, s + ".NS", bd) for s in pool}
    vols = {s: align(vo, s + ".NS", bd) for s in pool}
    member = Membership(mode, spec, bd, closes, vols, real_csv)

    last = bd[-1]
    si = int(np.searchsorted(bd.values, np.datetime64(last - pd.DateOffset(years=years))))
    si = max(si, MIN_HISTORY)
    ei = len(bd) - 1
    if ei - si < WEEK_STEP * 4:
        print(f"  ✗ Not enough data for {years}y window.")
        return None

    cash = capital
    book: dict[str, _Pos] = {}
    trade_rets: list[float] = []
    nav = pd.Series(np.nan, index=bd[si:ei + 1], dtype=float)

    def price_at(sym, d):
        p = closes[sym][d]
        return p if not np.isnan(p) else np.nan

    week_days = list(range(si, ei, WEEK_STEP))
    for wi, t in enumerate(week_days):
        members = member.asof(t)

        # candidate stats as of t
        cand = {}
        for sym in members:
            a = closes[sym][: t + 1]
            idx = np.where(~np.isnan(a))[0]
            if len(idx) < 200:
                continue
            c = a[idx]
            price = c[-1]
            dma20 = c[-20:].mean()
            dma200 = c[-200:].mean()
            if dma20 <= 0 or price <= 0 or price < dma200:   # 200DMA quality gate
                continue
            cand[sym] = {"price": price, "dev": (price - dma20) / dma20 * 100}

        # ---- priority: SELL → AVERAGE → FRESH (one decision per week) ----
        # 1. exits (all triggered)
        sold = False
        for sym in list(book.keys()):
            p = price_at(sym, t)
            if np.isnan(p):
                continue
            pos = book[sym]
            if pos.qty > 0 and p >= pos.avg * (1 + profit):
                proceeds = pos.qty * p * (1 - cost)
                cash += proceeds
                trade_rets.append((p / pos.avg) - 1)
                del book[sym]
                sold = True

        if not sold:
            # 2. averaging — worst held first
            did_avg = False
            held_sorted = sorted(
                book.items(),
                key=lambda kv: (price_at(kv[0], t) / kv[1].last_buy - 1)
                if kv[1].last_buy > 0 and not np.isnan(price_at(kv[0], t)) else 0.0,
            )
            for sym, pos in held_sorted:
                p = price_at(sym, t)
                if np.isnan(p):
                    continue
                if p <= pos.last_buy * (1 - avg_trig) and pos.invested + avg_amt <= max_stock \
                        and cash >= avg_amt * (1 + cost):
                    q = avg_amt / p
                    cash -= avg_amt * (1 + cost)
                    pos.qty += q
                    pos.invested += avg_amt
                    pos.cost_basis += q * p
                    pos.last_buy = p
                    did_avg = True
                    break

            if not did_avg and len(book) < max_pos:
                # 3. fresh — top-5 most-fallen not held
                fresh_list = sorted(
                    [(d["dev"], s, d) for s, d in cand.items()
                     if d["dev"] < 0 and s not in book],
                    key=lambda x: x[0],
                )[:5]
                if fresh_list and cash >= fresh * (1 + cost):
                    _, sym, d = fresh_list[0]
                    p = d["price"]
                    q = fresh / p
                    cash -= fresh * (1 + cost)
                    book[sym] = _Pos(qty=q, invested=fresh, last_buy=p, cost_basis=q * p)

        # daily NAV from t to next week step; idle cash earns the liquid-fund yield
        # (the deployer explicitly recommends parking idle ₹ at ~6.5% p.a.)
        nt = min(t + WEEK_STEP, ei)
        drf = (1 + idle_yield) ** (1 / TRADING_DAYS) - 1
        for dday in range(t, nt + 1):
            mkt = 0.0
            for sym, pos in book.items():
                p = price_at(sym, dday)
                if np.isnan(p):
                    p = pos.avg
                mkt += pos.qty * p
            idle = cash * (1 + drf) ** (dday - t)
            nav.loc[bd[dday]] = (idle + mkt) / capital
        cash *= (1 + drf) ** (nt - t)   # realise interest into cash for next block

    nav.iloc[0] = 1.0 if np.isnan(nav.iloc[0]) else nav.iloc[0]
    nav = nav.ffill()
    bench_nav = pd.Series(bench_arr[si:ei + 1] / bench_arr[si], index=bd[si:ei + 1])
    m = metrics(nav, bench_nav, trade_rets, [], rf)
    m["strategy"] = spec.label
    m["mode"] = mode if spec.pit_capable or mode == "snapshot" else "snapshot*"
    m["years"] = years
    m["ann_to"] = float("nan")  # turnover not tracked the same way for MR
    return m


# ── Specs ─────────────────────────────────────────────────────────────────────
def build_specs() -> dict[str, MomentumSpec]:
    n50_super = sorted(set(NIFTY50_LIST) | (set(NIFTY_NEXT50) - set(NIFTY50_LIST)))
    n50_sector = {**NEXT50_SECTOR_MAP, **NIFTY50_SECTOR_MAP}
    return {
        "momentum-n50": MomentumSpec(
            key="momentum-n50", label="Nifty 50 Momentum",
            snapshot=NIFTY50_LIST, superset=n50_super, sector_map=n50_sector,
            benchmark=NIFTY50_BENCHMARK, top_n=15, sector_cap=3,
            liquidity_cr=5.0, index_size=50, pit_capable=True,
        ),
        "momentum-midcap": MomentumSpec(
            key="momentum-midcap", label="Midcap 50 Momentum",
            snapshot=MIDCAP50_LIST, superset=MIDCAP50_LIST,
            sector_map=MIDCAP50_SECTOR_MAP, benchmark=MIDCAP50_BENCHMARK,
            top_n=10, sector_cap=3, liquidity_cr=2.0, index_size=50,
            pit_capable=False,   # no midcap-150 superset → snapshot (warn)
        ),
        "niftyshop-n50": MomentumSpec(
            key="niftyshop-n50", label="NiftyShop (Nifty 50)",
            snapshot=NIFTY50_LIST, superset=sorted(set(NIFTY50_LIST) | set(NIFTY_NEXT50)),
            sector_map=NIFTY50_SECTOR_MAP, benchmark=NIFTY50_BENCHMARK,
            top_n=5, sector_cap=99, liquidity_cr=0.0, index_size=50, pit_capable=True,
        ),
        "niftyshop-midcap": MomentumSpec(
            key="niftyshop-midcap", label="MidcapShop (Midcap 50)",
            snapshot=MIDCAP50_LIST, superset=MIDCAP50_LIST,
            sector_map=MIDCAP50_SECTOR_MAP, benchmark=MIDCAP50_BENCHMARK,
            top_n=5, sector_cap=99, liquidity_cr=0.0, index_size=50, pit_capable=False,
        ),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    hdr = (f"\n  {'Strategy':<22} {'Mode':<9} {'Yr':>2} {'CAGR':>7} {'Bench':>7} "
           f"{'Alpha':>7} {'JAlpha':>7} {'Beta':>5} {'Shrp':>5} {'Sort':>5} "
           f"{'MDD':>7} {'Calmar':>6} {'HitM':>5} {'HitT':>5} {'Turn':>6} {'Trd':>4}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 3))
    for r in rows:
        to = "  n/a" if (isinstance(r["ann_to"], float) and np.isnan(r["ann_to"])) else f"{r['ann_to']:>5.0f}%"
        print(f"  {r['strategy']:<22} {r['mode']:<9} {r['years']:>2} "
              f"{r['cagr']:>6.1f}% {r['bench_cagr']:>6.1f}% {r['alpha']:>+6.1f}% "
              f"{r['jensen']:>+6.1f}% {r['beta']:>5.2f} {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
              f"{r['mdd']:>6.1f}% {r['calmar']:>6.2f} {r['hit_m']:>4.0f}% {r['hit_t']:>4.0f}% "
              f"{to} {r['n_trades']:>4}")


def print_bias(rows: list[dict]) -> None:
    """Snapshot vs PIT delta — the survivorship-bias measurement."""
    by = {}
    for r in rows:
        by.setdefault((r["strategy"], r["years"]), {})[r["mode"]] = r
    deltas = [(k, v) for k, v in by.items() if "snapshot" in v and "pit" in v]
    if not deltas:
        return
    print("\n  SURVIVORSHIP-BIAS DELTA  (snapshot minus point-in-time)")
    print(f"  {'Strategy':<22} {'Yr':>2} {'ΔCAGR':>8} {'ΔJAlpha':>9} {'ΔSharpe':>8} {'ΔMDD':>8}")
    print("  " + "─" * 64)
    for (strat, yr), v in sorted(deltas):
        s, p = v["snapshot"], v["pit"]
        print(f"  {strat:<22} {yr:>2} "
              f"{s['cagr']-p['cagr']:>+7.1f}% {s['jensen']-p['jensen']:>+8.1f}% "
              f"{s['sharpe']-p['sharpe']:>+8.2f} "
              f"{s['mdd']-p['mdd']:>+7.1f}%")
    print("\n  Positive ΔCAGR / ΔJAlpha = the original (snapshot) backtest was inflated by")
    print("  survivorship. The point-in-time column is the number to trust.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="WealthOS point-in-time backtest")
    p.add_argument("--strategy", default="all",
                   choices=["all", "momentum-n50", "momentum-midcap",
                            "niftyshop-n50", "niftyshop-midcap"])
    p.add_argument("--years", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--mode", default="both", choices=["both", "pit", "snapshot"])
    p.add_argument("--cost", type=float, default=0.002,
                   help="per-side cost fraction (STT+exch+slippage), default 0.20%%")
    p.add_argument("--rf", type=float, default=0.065, help="risk-free rate")
    p.add_argument("--membership-n50", type=Path, default=None,
                   help="real NSE constituents CSV (symbol,start,end) — gold standard")
    args = p.parse_args()

    specs = build_specs()
    keys = list(specs) if args.strategy == "all" else [args.strategy]
    modes = ["snapshot", "pit"] if args.mode == "both" else [args.mode]
    data_years = max(args.years) + 2   # +2y lookback for 200DMA/52W indicators

    print("=" * 78)
    print("  WealthOS — POINT-IN-TIME BACKTEST")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}  |  horizons={args.years}  modes={modes}")
    print(f"  cost/side={args.cost*100:.2f}%  rf={args.rf*100:.1f}%")
    print("=" * 78)

    rows: list[dict] = []
    for key in keys:
        spec = specs[key]
        if not spec.pit_capable and "pit" in modes:
            print(f"\n  ⚠  {spec.label}: no point-in-time superset available → "
                  f"'pit' degrades to snapshot (shown as 'snapshot*'). "
                  f"Supply a real membership file for a true PIT run.")
        for yr in args.years:
            for mode in modes:
                # skip redundant pit run when it would just equal snapshot
                if mode == "pit" and not spec.pit_capable and "snapshot" in modes:
                    continue
                print(f"\n── {spec.label} | {yr}y | {mode} "
                      f"{'─'*max(0, 40-len(spec.label))}")
                try:
                    runner = run_niftyshop if key.startswith("niftyshop") else run_momentum
                    if key.startswith("niftyshop"):
                        cap = 400_000 if "n50" in key else 200_000
                        fr = 10_000 if "n50" in key else 5_000
                        av = 15_000 if "n50" in key else 7_500
                        mx = 40_000 if "n50" in key else 20_000
                        r = run_niftyshop(spec, mode, yr, args.cost, args.rf,
                                          args.membership_n50, data_years,
                                          cap, fr, av, mx)
                    else:
                        r = run_momentum(spec, mode, yr, args.cost, args.rf,
                                         args.membership_n50 if key == "momentum-n50" else None,
                                         data_years)
                    if r:
                        rows.append(r)
                except Exception as e:  # noqa: BLE001
                    print(f"  ✗ run failed: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print_table(rows)
    print_bias(rows)

    print("\n  CAVEATS (do not omit when quoting these):")
    print("   • PIT removes most, not all, survivorship bias (Next-50 superset is current).")
    print("   • 'snapshot*' rows are NOT point-in-time — midcap lacks a 150-name superset.")
    print("   • Truly delisted/merged names with no Yahoo history are absent → downside understated.")
    print("   • Single data source (yfinance); cost model is an estimate, not your actual fills.")
    print()


if __name__ == "__main__":
    main()
