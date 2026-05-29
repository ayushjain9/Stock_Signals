"""
WealthOS — Momentum Strategy Backtester
========================================

Backtests the momentum screener v3 signal on historical NSE data.

WHAT THIS VALIDATES
-------------------
  CAGR                  Compound annual growth rate vs NIFTY benchmark
  Sharpe ratio          Risk-adjusted return (monthly, annualized, rf=6.5%)
  Sortino ratio         Downside-only risk-adjusted return
  Max drawdown          Worst peak-to-trough loss (portfolio + benchmark)
  Calmar ratio          CAGR / |max drawdown|
  Hit ratio (months)    % of rebalancing periods portfolio > benchmark
  Hit ratio (trades)    % of individual position exits that were profitable
  Turnover              Avg % of portfolio replaced per period (cost driver)
  Annual cost drag      Realistic Indian equity transaction cost impact
  Sector exposure       Avg allocation per sector across all periods
  Alpha (Jensen's)      Portfolio return minus beta-adjusted benchmark return

BIAS PREVENTION — critical
---------------------------
  Look-ahead bias    Signal computed on close[t-1], return measured from close[t]
                     to close[t+rebal_days]. NEVER uses current period data.
  Survivorship bias  Pass your own universe. If you use today's Nifty 500,
                     you have survivorship bias. Accept it and note it.
  Point-in-time      Sector map is static (limitation). For 5yr backtests
                     this introduces mild error on ~10-15% of stocks.

TRANSACTION COST MODEL
-----------------------
  Per trade: brokerage (₹20 flat or %) + STT (0.1% sell) + exchange (0.018% each way)
  Realistic round-trip for ₹1L position: ~0.14-0.21%
  This is applied on portfolio fraction turned over at each rebalancing.

USAGE
-----
  pip install yfinance pandas numpy scipy rich
  python backtest_momentum.py                          # Nifty 50, 5yr, top 15
  python backtest_momentum.py --years 3 --top-n 20
  python backtest_momentum.py --universe nifty200      # requires nifty200.txt
  python backtest_momentum.py --rebal-days 5           # weekly (high turnover)
  python backtest_momentum.py --output results.csv

INTERPRETING RESULTS
---------------------
  Sharpe > 1.0    Good. > 1.5 is excellent for a pure momentum strategy.
  Max DD < 30%    Acceptable. > 40% tests whether you'd actually hold.
  Hit ratio 55%+  Strategy wins more months than it loses vs benchmark.
  Turnover < 200% Annual turnover under 200% keeps cost drag < 0.5%/yr.
  Alpha > 3%      Meaningful outperformance after costs. < 2% may be noise.
  Sector < 30%    No sector dominates more than 30% (concentration risk).
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Universe definitions
# ─────────────────────────────────────────────────────────────────────────────

NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","BAJFINANCE","HCLTECH","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","NESTLEIND","WIPRO","ONGC",
    "JSWSTEEL","TATAMOTORS","ADANIENT","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "TATASTEEL","TECHM","HDFCLIFE","DIVISLAB","DRREDDY","CIPLA","GRASIM","APOLLOHOSP",
    "ADANIPORTS","TRENT","BEL","SHRIRAMFIN","BAJAJ-AUTO","EICHERMOT","M&M",
    "BRITANNIA","BPCL","HEROMOTOCO","HINDALCO","SBILIFE",
]

# Add extra liquid mid-caps to the universe for a richer test
EXTENDED = NIFTY50 + [
    "CANBK","UNIONBANK","IDFCFIRSTB","FEDERALBNK","BANDHANBNK","RBLBANK",
    "PNB","BANKBARODA","MUTHOOTFIN","CHOLAFIN","RECLTD","PFC","IRFC",
    "ZOMATO","DMART","TATACONSUM","VARUNBEV","NYKAA","PAYTM","POLICYBZR",
    "APOLLOHOSP","MAXHEALTH","FORTIS","LALPATHLAB",
    "DIXON","KAYNES","AMBER","HAVELLS","POLYCAB",
    "TATAPOWER","TORNTPOWER","ADANIGREEN","SJVN","NHPC",
    "SIEMENS","ABB","BHEL","BEL","HAL","BEML",
    "AMBUJACEM","ACC","RAMCOCEM",
    "PERSISTENT","COFORGE","LTTS","KPIT","MPHASIS","OFSS",
    "SAIL","HINDZINC","MOIL","NMDC","VEDL",
    "PAGEIND","MCDOWELL-N","RADICO","UNITDSPR",
    "IRFC","NATIONALUM","RATNAMANI","CUMMINSIND",
]

SECTOR_MAP: dict[str, str] = {
    "HDFCBANK":"BANKING","ICICIBANK":"BANKING","SBIN":"BANKING","AXISBANK":"BANKING",
    "KOTAKBANK":"BANKING","INDUSINDBK":"BANKING","IDFCFIRSTB":"BANKING","PNB":"BANKING",
    "BANKBARODA":"BANKING","CANBK":"BANKING","UNIONBANK":"BANKING","FEDERALBNK":"BANKING",
    "BANDHANBNK":"BANKING","RBLBANK":"BANKING",
    "BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","MUTHOOTFIN":"NBFC","CHOLAFIN":"NBFC",
    "RECLTD":"NBFC","PFC":"NBFC","IRFC":"NBFC",
    "HDFCLIFE":"INSURANCE","ICICIGI":"INSURANCE","SBILIFE":"INSURANCE",
    "TCS":"IT","INFOSYS":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "PERSISTENT":"IT","COFORGE":"IT","LTTS":"IT","KPIT":"IT","MPHASIS":"IT","OFSS":"IT",
    "RELIANCE":"ENERGY","ONGC":"ENERGY","BPCL":"ENERGY","IOC":"ENERGY","GAIL":"ENERGY",
    "NTPC":"POWER","POWERGRID":"POWER","NHPC":"POWER","TATAPOWER":"POWER",
    "TORNTPOWER":"POWER","ADANIGREEN":"POWER","SJVN":"POWER",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "TATACONSUM":"FMCG","VARUNBEV":"FMCG","MCDOWELL-N":"FMCG",
    "MARUTI":"AUTO","TATAMOTORS":"AUTO","M&M":"AUTO","BAJAJ-AUTO":"AUTO",
    "EICHERMOT":"AUTO","HEROMOTOCO":"AUTO","MOTHERSON":"AUTO",
    "LT":"INFRA","SIEMENS":"INFRA","ABB":"INFRA","CUMMINSIND":"INFRA","BHEL":"INFRA",
    "HAL":"DEFENCE","BEL":"DEFENCE","BEML":"DEFENCE",
    "TITAN":"CONSUMER","TRENT":"RETAIL","DMART":"RETAIL","PAGEIND":"CONSUMER",
    "ASIANPAINT":"PAINTS","BERGEPAINT":"PAINTS",
    "ULTRACEMCO":"CEMENT","GRASIM":"CEMENT","AMBUJACEM":"CEMENT","ACC":"CEMENT",
    "JSWSTEEL":"METALS","TATASTEEL":"METALS","HINDALCO":"METALS","COALINDIA":"METALS",
    "SAIL":"METALS","HINDZINC":"METALS","MOIL":"METALS","NMDC":"METALS","VEDL":"METALS",
    "SUNPHARMA":"PHARMA","DRREDDY":"PHARMA","DIVISLAB":"PHARMA","CIPLA":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE","MAXHEALTH":"HEALTHCARE","FORTIS":"HEALTHCARE",
    "LALPATHLAB":"HEALTHCARE",
    "ADANIENT":"CONGLOMERATE","ADANIPORTS":"PORTS","ADANIGREEN":"POWER",
    "DIXON":"ELECTRONICS","KAYNES":"ELECTRONICS","AMBER":"ELECTRONICS",
    "HAVELLS":"ELECTRONICS","POLYCAB":"ELECTRONICS",
    "NATIONALUM":"METALS","RATNAMANI":"METALS",
    "ZOMATO":"CONSUMER","NYKAA":"CONSUMER","PAYTM":"FINTECH","POLICYBZR":"FINTECH",
}


# ─────────────────────────────────────────────────────────────────────────────
# Signal engine (simplified for speed — core momentum signals only)
# Uses strict look-ahead prevention: only data up to index i-1 is used
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    k = 2 / (n + 1)
    out = np.zeros(len(arr), dtype=float)
    out[n - 1] = arr[:n].mean()
    for i in range(n, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def momentum_score(
    price_hist: np.ndarray,
    high_hist: np.ndarray,
    low_hist: np.ndarray,
    bench_hist: np.ndarray,
    min_history: int = 210,
) -> float:
    """
    Compute momentum score on data available up to current period (no look-ahead).
    Mirrors the v3 screener signals.
    Returns 0-100, or -1 if insufficient data.
    """
    if len(price_hist) < min_history:
        return -1.0

    p     = price_hist[-1]
    d25   = price_hist[-25:].mean()
    d50   = price_hist[-50:].mean()
    d200  = price_hist[-200:].mean()

    score = 0.0

    # ── Tier 3: DMA alignment + ADX direction (20 pts) ──
    if p > d25 > d50 > d200:
        score += 20
    elif d50 > d200 and p > d200:
        score += 12
    elif d50 > d200:
        score += 6

    # ── Tier 1: RS vs benchmark (25 pts) ──
    if len(bench_hist) >= 90:
        stock_ret_90 = price_hist[-1] / price_hist[-90] - 1
        bench_ret_90 = bench_hist[-1] / bench_hist[-90] - 1
        rs = (stock_ret_90 - bench_ret_90) * 100
        if rs > 10:   score += 25
        elif rs > 5:  score += 18
        elif rs > 0:  score += 10
        elif rs < -5: score -= 15

    # ── Tier 4: Volatility-adjusted momentum (20 pts) ──
    if len(price_hist) >= 66:
        roc = (price_hist[-1] / price_hist[-66] - 1) * 100
        # ATR proxy using daily range
        if len(high_hist) >= 14 and len(low_hist) >= 14:
            ranges = high_hist[-14:] - low_hist[-14:]
            atr = ranges.mean()
            atr_pct = atr / p * 100 if p > 0 else 2.0
        else:
            atr_pct = price_hist[-20:].std() / price_hist[-20:].mean() * 100
        vam = roc / atr_pct if atr_pct > 0.01 else 0
        if vam > 3.0:   score += 20
        elif vam > 2.0: score += 15
        elif vam > 1.0: score += 8
        elif vam > 0:   score += 3

    # ── Tier 5: Confirmation — 52W high proximity (6 pts) ──
    if len(price_hist) >= 252:
        hi52 = price_hist[-252:].max()
        pct52 = (p / hi52 - 1) * 100
        if -5 <= pct52 <= 0:
            score += 6

    # ── Tier 5: MTF EMA alignment (from Tier 2 in backtest weight context) ──
    if len(price_hist) >= 200:
        e5  = _ema(price_hist, 5)
        e20 = _ema(price_hist, 20)
        e50 = _ema(price_hist, 50)
        e200 = _ema(price_hist, 200)
        mtf = sum([e5[-1] > e20[-1], e20[-1] > e50[-1], e50[-1] > e200[-1]])
        score += mtf * 4  # 0-12 bonus (max 12 when all 3 aligned)

    return min(float(score), 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Transaction cost model
# ─────────────────────────────────────────────────────────────────────────────

def round_trip_cost_pct(position_value: float) -> float:
    """Realistic Indian equity round-trip cost as fraction of trade value."""
    brokerage_each = min(20, position_value * 0.0003)
    stt = position_value * 0.001        # sell side only
    exchange = position_value * 0.00018 * 2
    gst = brokerage_each * 0.18 * 2
    total = brokerage_each * 2 + stt + exchange + gst
    return total / position_value


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioState:
    holdings: dict[str, float] = field(default_factory=dict)  # symbol → shares
    cash: float = 0.0
    nav: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    universe:       list[str]
    years:          int     = 5
    top_n:          int     = 15
    rebal_days:     int     = 21       # 21 = monthly, 5 = weekly
    initial_capital: float  = 1_000_000
    min_score:      float   = 40.0     # don't buy low-conviction stocks
    risk_free:      float   = 0.065    # India 10yr G-Sec
    benchmark:      str     = "^NSEI"
    cost_pct:       float   = 0.002    # 0.2% round-trip (conservative estimate)
    min_history:    int     = 210      # days needed before first signal


def run_backtest(cfg: BacktestConfig) -> dict:
    """Run the full backtest. Returns a dict of all metrics and timeseries."""
    import yfinance as yf

    period = f"{cfg.years + 1}y"  # extra year for warmup
    tickers = [s + ".NS" for s in cfg.universe] + [cfg.benchmark]

    print(f"Downloading {len(tickers)} tickers ({period})...")
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=True, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"];  high = raw["High"];  low = raw["Low"];  vol = raw["Volume"]
    else:
        close = raw[["Close"]]; high = raw[["High"]]; low = raw[["Low"]]; vol = raw[["Volume"]]
        close.columns = high.columns = low.columns = vol.columns = [tickers[0]]

    # Benchmark series
    bench_col = cfg.benchmark
    bench = close[bench_col].dropna() if bench_col in close.columns else None
    if bench is None or len(bench) < cfg.min_history:
        raise ValueError(f"Benchmark {cfg.benchmark} has insufficient data.")

    # Trim to requested years (keep warmup for signals)
    bench_arr = bench.values.astype(float)
    bench_dates = bench.index

    # Portfolio tracking
    port_nav   = [cfg.initial_capital]
    bench_nav  = [cfg.initial_capital]
    port_dates = [bench_dates[cfg.min_history]]

    current_holdings: set[str] = set()
    entry_prices: dict[str, float] = {}
    all_turnover: list[float] = []
    trade_log: list[dict] = []     # (symbol, entry, exit, return, hold_days, sector)
    period_returns_port: list[float] = []
    period_returns_bench: list[float] = []
    sector_counts: dict[str, int] = {}

    # Rebalancing dates (every rebal_days trading days, starting after warmup)
    start_idx = cfg.min_history
    rebal_indices = list(range(start_idx, len(bench_arr) - cfg.rebal_days, cfg.rebal_days))

    for r_idx, day_idx in enumerate(rebal_indices):
        # Compute signals for each stock using data up to day_idx-1 (strict)
        scores: dict[str, float] = {}
        for sym in cfg.universe:
            col = sym + ".NS"
            if col not in close.columns:
                continue
            close_s = close[col].dropna()
            if len(close_s) <= day_idx or close_s.iloc[day_idx] != close_s.iloc[day_idx]:
                continue  # NaN check
            # LOOK-AHEAD PREVENTION: use data up to day_idx-1, not day_idx
            stock_close_hist = close[col].reindex(bench_dates[:day_idx]).ffill().dropna().values.astype(float)
            if len(stock_close_hist) < cfg.min_history:
                continue
            high_hist = high[col].reindex(bench_dates[:day_idx]).ffill().dropna().values.astype(float)
            low_hist  = low[col].reindex(bench_dates[:day_idx]).ffill().dropna().values.astype(float)

            # Liquidity gate: avg daily value > ₹5Cr
            vol_s = vol[col].reindex(bench_dates[:day_idx]).fillna(0).values
            avg_vol = vol_s[-20:].mean()
            price_now = stock_close_hist[-1]
            if avg_vol * price_now < 5e7:  # 5Cr = 5e7 rupees
                continue

            sc = momentum_score(stock_close_hist, high_hist, low_hist, bench_arr[:day_idx], cfg.min_history)
            if sc >= cfg.min_score:
                scores[sym] = sc

        # Select top N by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        new_holdings = set(s for s, _ in ranked[:cfg.top_n])

        # Turnover
        exits  = current_holdings - new_holdings
        enters = new_holdings - current_holdings
        if current_holdings:
            to = (len(exits) + len(enters)) / 2 / max(len(new_holdings), 1)
            all_turnover.append(to)

        # Forward return period: day_idx to day_idx + rebal_days
        next_idx = day_idx + cfg.rebal_days
        if next_idx >= len(bench_arr):
            break

        # Portfolio return = equal-weight average of held stocks minus cost drag
        port_fwd_rets = []
        for sym in new_holdings:
            col = sym + ".NS"
            if col not in close.columns:
                continue
            close_reindexed = close[col].reindex(bench_dates).ffill()
            p_now  = float(close_reindexed.iloc[day_idx])
            p_next = float(close_reindexed.iloc[next_idx])
            if np.isnan(p_now) or np.isnan(p_next) or p_now <= 0:
                continue
            fwd_ret = p_next / p_now - 1
            port_fwd_rets.append(fwd_ret)

            # Log exits for hit ratio
            if sym in exits:
                entry_p = entry_prices.get(sym, p_now)
                hold_days_cnt = cfg.rebal_days * (1 + sum(1 for s, _ in all_turnover if sym in current_holdings))
                sector = SECTOR_MAP.get(sym, "OTHER")
                trade_log.append({
                    "symbol": sym, "entry_price": entry_p, "exit_price": p_now,
                    "return_pct": (p_now / entry_p - 1) * 100 if entry_p > 0 else 0,
                    "sector": sector,
                })
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Track entry prices
        for sym in enters:
            col = sym + ".NS"
            if col in close.columns:
                cr = close[col].reindex(bench_dates).ffill()
                entry_prices[sym] = float(cr.iloc[day_idx]) if not np.isnan(cr.iloc[day_idx]) else 0

        # Cost drag: proportion turned over × round-trip cost
        turnover_frac = (len(exits) + len(enters)) / (2 * max(len(new_holdings), 1))
        cost_drag = turnover_frac * cfg.cost_pct

        port_period_ret = (np.mean(port_fwd_rets) if port_fwd_rets else 0) - cost_drag
        bench_period_ret = bench_arr[next_idx] / bench_arr[day_idx] - 1

        period_returns_port.append(port_period_ret)
        period_returns_bench.append(bench_period_ret)
        port_nav.append(port_nav[-1] * (1 + port_period_ret))
        bench_nav.append(bench_nav[-1] * (1 + bench_period_ret))
        port_dates.append(bench_dates[next_idx])
        current_holdings = new_holdings

    port_nav  = np.array(port_nav)
    bench_nav = np.array(bench_nav)
    pr = np.array(period_returns_port)
    br = np.array(period_returns_bench)
    periods_per_year = 252 / cfg.rebal_days
    n_years = len(pr) / periods_per_year

    # ── Compute all metrics ──
    port_cagr  = (port_nav[-1] / port_nav[0]) ** (1 / n_years) - 1
    bench_cagr = (bench_nav[-1] / bench_nav[0]) ** (1 / n_years) - 1
    alpha      = port_cagr - bench_cagr

    rf_period = cfg.risk_free / periods_per_year
    excess_pr = pr - rf_period
    sharpe    = (excess_pr.mean() / excess_pr.std()) * np.sqrt(periods_per_year) if excess_pr.std() > 0 else 0
    downside_std = excess_pr[excess_pr < 0].std() * np.sqrt(periods_per_year)
    sortino   = (excess_pr.mean() * periods_per_year) / downside_std if downside_std > 0 else 0

    peak_p = np.maximum.accumulate(port_nav)
    mdd_p  = ((port_nav - peak_p) / peak_p).min()
    peak_b = np.maximum.accumulate(bench_nav)
    mdd_b  = ((bench_nav - peak_b) / bench_nav).min()

    calmar = port_cagr / abs(mdd_p) if mdd_p < 0 else 0
    hit_months = (pr > br).mean()
    hit_trades = np.mean([t["return_pct"] > 0 for t in trade_log]) if trade_log else 0
    avg_turnover_pct = np.mean(all_turnover) * 100 if all_turnover else 0
    annual_turnover  = avg_turnover_pct * periods_per_year
    annual_cost_drag = annual_turnover / 100 * cfg.cost_pct * 100

    # Beta and Jensen's alpha
    if len(br) > 5:
        cov_matrix = np.cov(pr, br)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0
        jensens_alpha = (port_cagr - cfg.risk_free) - beta * (bench_cagr - cfg.risk_free)
    else:
        beta = 1.0; jensens_alpha = alpha

    # Sector exposure
    total_sector = sum(sector_counts.values())
    sector_exposure = {k: v / total_sector * 100 for k, v in sector_counts.items()} if total_sector else {}

    return {
        "config":           cfg,
        "port_nav":         port_nav,
        "bench_nav":        bench_nav,
        "port_dates":       port_dates,
        "trade_log":        trade_log,
        "metrics": {
            "port_cagr":        round(port_cagr * 100, 2),
            "bench_cagr":       round(bench_cagr * 100, 2),
            "alpha":            round(alpha * 100, 2),
            "jensens_alpha":    round(jensens_alpha * 100, 2),
            "beta":             round(beta, 3),
            "sharpe":           round(sharpe, 2),
            "sortino":          round(sortino, 2),
            "mdd_portfolio":    round(mdd_p * 100, 2),
            "mdd_benchmark":    round(mdd_b * 100, 2),
            "calmar":           round(calmar, 2),
            "hit_ratio_months": round(hit_months * 100, 1),
            "hit_ratio_trades": round(hit_trades * 100, 1),
            "avg_turnover_pct": round(avg_turnover_pct, 1),
            "annual_turnover":  round(annual_turnover, 0),
            "annual_cost_drag": round(annual_cost_drag, 2),
            "n_trades":         len(trade_log),
            "n_periods":        len(pr),
            "years_backtested": round(n_years, 1),
            "sector_exposure":  sector_exposure,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    m = results["metrics"]
    cfg = results["config"]
    print(f"\n{'='*72}")
    print(f"MOMENTUM STRATEGY BACKTEST — v3 Screener")
    print(f"Universe: {len(cfg.universe)} stocks | Top {cfg.top_n} held | "
          f"Rebalance every {cfg.rebal_days}d | {m['years_backtested']:.1f} years")
    print(f"{'='*72}")

    print(f"\n{'RETURN METRICS':─<38}")
    print(f"  Portfolio CAGR:           {m['port_cagr']:>8.2f}%")
    print(f"  Benchmark CAGR (NIFTY):   {m['bench_cagr']:>8.2f}%")
    print(f"  Alpha (simple):           {m['alpha']:>8.2f}%")
    print(f"  Alpha (Jensen's, beta-adj):{m['jensens_alpha']:>7.2f}%")
    print(f"  Beta vs NIFTY:            {m['beta']:>8.3f}")

    print(f"\n{'RISK METRICS':─<38}")
    print(f"  Sharpe ratio (annualized):{m['sharpe']:>8.2f}  (>1.0 = good, >1.5 = excellent)")
    print(f"  Sortino ratio:            {m['sortino']:>8.2f}  (>1.5 = good)")
    print(f"  Max drawdown (portfolio): {m['mdd_portfolio']:>8.2f}%")
    print(f"  Max drawdown (benchmark): {m['mdd_benchmark']:>8.2f}%")
    print(f"  Calmar ratio:             {m['calmar']:>8.2f}  (>0.5 = acceptable)")

    print(f"\n{'TRADING METRICS':─<38}")
    print(f"  Hit ratio (months):       {m['hit_ratio_months']:>8.1f}%  (>55% = strategy wins more)")
    print(f"  Hit ratio (individual):   {m['hit_ratio_trades']:>8.1f}%  (>55% = most positions profitable)")
    print(f"  Avg period turnover:      {m['avg_turnover_pct']:>8.1f}%  (% portfolio replaced per rebal)")
    print(f"  Annual turnover:          {m['annual_turnover']:>8.0f}%  (<200% = cost-efficient)")
    print(f"  Annual cost drag:         {m['annual_cost_drag']:>8.2f}%  (lost to transaction costs)")
    print(f"  Total closed trades:      {m['n_trades']:>8d}")

    sec = m["sector_exposure"]
    if sec:
        print(f"\n{'SECTOR EXPOSURE (avg across periods)':─<38}")
        for sector, pct in sorted(sec.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 2.5)
            flag = " ⚠ concentrated" if pct > 30 else ""
            print(f"  {sector:14s}  {pct:5.1f}%  {bar}{flag}")

    # Interpretation
    print(f"\n{'INTERPRETATION':─<38}")
    if m["jensens_alpha"] > 3:
        print("  ✓ Meaningful alpha after adjusting for benchmark beta.")
    elif m["jensens_alpha"] > 0:
        print("  ~ Small positive alpha. May not be statistically significant.")
    else:
        print("  ✗ Negative Jensen's alpha — strategy doesn't compensate for risk taken.")

    if m["sharpe"] >= 1.5:
        print("  ✓ Excellent risk-adjusted return (Sharpe ≥ 1.5).")
    elif m["sharpe"] >= 1.0:
        print("  ✓ Good risk-adjusted return (Sharpe ≥ 1.0).")
    else:
        print("  ~ Below-average risk-adjusted return. Review signal quality.")

    if m["annual_turnover"] > 300:
        print(f"  ⚠ High turnover ({m['annual_turnover']:.0f}%). Cost drag is "
              f"₹{m['annual_cost_drag']:.2f}%/yr — consider less frequent rebalancing.")

    max_conc = max(sec.values()) if sec else 0
    max_conc_sec = max(sec, key=sec.get) if sec else ""
    if max_conc > 30:
        print(f"  ⚠ {max_conc_sec} at {max_conc:.1f}% — sector concentration risk.")

    print(f"\n{'SURVIVORSHIP BIAS WARNING':─<38}")
    print("  This backtest uses current-universe stocks only. Stocks that")
    print("  delisted or crashed are absent. True historical alpha is likely")
    print("  1-3% lower than reported here.")


def save_csv(results: dict, path: Path) -> None:
    m = results["metrics"]
    summary = pd.DataFrame([{k: v for k, v in m.items() if k != "sector_exposure"}])
    summary.to_csv(path.with_suffix("") / "_summary.csv", index=False)

    trade_df = pd.DataFrame(results["trade_log"])
    if not trade_df.empty:
        trade_df.to_csv(path, index=False)
        print(f"Trade log: {path}  |  Summary: {path.with_suffix('')}_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Momentum v3 backtester")
    p.add_argument("--universe",    choices=["nifty50", "extended"], default="nifty50")
    p.add_argument("--universe-file", type=Path, help="Custom symbol list, one per line")
    p.add_argument("--years",       type=int,   default=5)
    p.add_argument("--top-n",       type=int,   default=15)
    p.add_argument("--rebal-days",  type=int,   default=21, help="21=monthly, 5=weekly")
    p.add_argument("--min-score",   type=float, default=40)
    p.add_argument("--capital",     type=float, default=1_000_000)
    p.add_argument("--output",      type=Path,  default=None)
    args = p.parse_args()

    if args.universe_file and args.universe_file.exists():
        symbols = [l.strip().upper() for l in args.universe_file.read_text().splitlines() if l.strip()]
    elif args.universe == "extended":
        symbols = EXTENDED
    else:
        symbols = NIFTY50

    cfg = BacktestConfig(
        universe=symbols,
        years=args.years,
        top_n=args.top_n,
        rebal_days=args.rebal_days,
        initial_capital=args.capital,
        min_score=args.min_score,
    )

    print(f"Universe: {len(symbols)} stocks | Top {cfg.top_n} | "
          f"Rebalance {cfg.rebal_days}d | {cfg.years}yr backtest")

    results = run_backtest(cfg)
    print_results(results)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
