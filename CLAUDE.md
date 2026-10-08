# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Activate the local venv (already created)
.venv\Scripts\activate          # Windows PowerShell
source .venv/bin/activate       # bash/WSL

# Dependencies (already installed in .venv)
pip install yfinance pandas numpy rich requests beautifulsoup4 scipy openpyxl
```

## Running the Scripts

All scripts are standalone — run directly with Python. Data is fetched live from Yahoo Finance (NSE symbols via `.NS` suffix, e.g. `HDFCBANK.NS`).

---

### NiftyShop Deploy — Nifty 50 (weekly mean-reversion)

```bash
python niftyshop_deploy.py                               # weekly signal (run every Friday 3:15pm)
python niftyshop_deploy.py --capital 400000              # ₹4L sleeve
python niftyshop_deploy.py --fresh 6250 --avg 9375       # custom position sizes
python niftyshop_deploy.py --status                      # portfolio snapshot only
python niftyshop_deploy.py --history                     # closed P&L
```

### Weekly run (both Shop deployers + dashboard)

```bash
python run_all.py              # NiftyShop + MidcapShop, then dashboard.py
python run_all.py --dry-run
```

### Parked: momentum (not deployed, not tested)

`momentum/nifty50_Momentum_deploy.py` and `momentum/midcap_momentum_deploy.py` are parked. They are NOT run by
`run_all.py` and NOT shown on the dashboard. They still run standalone
(`python momentum/nifty50_Momentum_deploy.py`) and read their paper holdings from `momentum/`.
5y point-in-time test (2026-10-08): v6 momentum 6.3% CAGR vs equal-weight Nifty 50 10.5% — it does not earn its complexity.

### MidcapShop Deploy — Nifty Midcap 50 (weekly mean-reversion)

```bash
python midcap_niftyshop_deploy.py                        # weekly signal (every Friday 3:15pm)
python midcap_niftyshop_deploy.py --capital 200000
python midcap_niftyshop_deploy.py --fresh 5000 --avg 7500
python midcap_niftyshop_deploy.py --status
python midcap_niftyshop_deploy.py --history
```

### Value Screener (mid & small cap fundamentals)

```bash
python value_screener.py                                 # full scan, 400+ stocks
python value_screener.py --top 10
python value_screener.py --min-score 60
python value_screener.py --sector IT
python value_screener.py --export value_picks.csv
```

---

## Architecture

### Folder structure

```
Stock_Signals/
├── niftyshop_deploy.py             ← Nifty 50 mean-reversion, weekly (PRODUCTION)
├── midcap_niftyshop_deploy.py      ← Midcap 50 mean-reversion, weekly (PRODUCTION)
├── config.py                       ← Universes, sector maps, benchmarks (single source of truth)
├── regime_detector.py              ← Market regime banner (imported by deployers)
├── run_all.py                      ← Runs both Shop deployers, then dashboard.py
├── dashboard.py                    ← Writes dashboard.html from the Shop trade logs
├── value_screener.py               ← Quality-value screener (~148 tickers, not 400+)
└── momentum/                       ← PARKED momentum deployers + paper holdings (not deployed)
```

The `Backtest/` (R&D chain v1→v6, NiftyShop/Nifty 100 universe tests) and `research/`
(momentum_screener_v3.py) folders were deleted on 2026-10-08; recover them from git
history if needed (`git log --all -- Backtest/`).

---

### Production strategies

| File | Strategy | Universe | Frequency | Sleeve |
|------|----------|----------|-----------|--------|
| `niftyshop_deploy.py` | Mean-reversion | Nifty 50 | Weekly (Friday 3:15pm) | ₹4L |
| `midcap_niftyshop_deploy.py` | Mean-reversion | Nifty Midcap 50 | Weekly (Friday 3:15pm) | ₹2L |

---

### Momentum signal stack (v6 — used in both momentum deployers)

Eight signals, score capped at 100 pts:

| Signal | Max pts | Notes |
|--------|---------|-------|
| Trend direction (DMA alignment) | 20 | price > 25D > 50D > 200D = "Brutal Strength" |
| RS vs benchmark 90D | 25 | 90-day excess return vs index |
| RS vs benchmark 6M | 8 | Jegadeesh-Titman: 6M lookback, skip last 1M; validated +2.20pp OOS |
| Volatility-adjusted momentum (VAM) | 20 | ROC(66d) / ATR% — filters chaotic spikes |
| 52W high proximity | 6 | within -5% of yearly high |
| Multi-timeframe EMA alignment | 12 | E5 > E20 > E50 > E200 |
| ADX direction proxy | 7 | price acceleration over 10-day windows |
| Volume surge | 6 | 5D avg > 1.5–2× 20D avg; validated +0.95pp OOS |
| Gate | BLOCKED | liquidity below threshold |

Nifty 50 benchmark: `^NSEI`. Midcap 50 benchmark: `^NSEMDCP50` (was wrongly `^NSMIDCP` = Nifty Next 50 until 2026-10-08). Liquidity gate: ₹5Cr (Nifty 50), ₹2Cr (Midcap 50).

**Signals tested and rejected in v6:** real ADX (underperformed every config), graduated 52W high (+0.21pp OOS — noise level).

> ⚠ The per-signal "+X.XXpp OOS" figures above come from the **survivorship-biased**
> snapshot backtests in `Backtest/`. They are signal-selection evidence, not net
> strategy performance. For net, point-in-time numbers see "Validated performance" below.

---

### Momentum strategy parameters

```
Nifty 50 version:
  Hold top:    15 stocks (equal weight, ₹60K each on ₹9L sleeve)
  Sector cap:  Max 3 stocks per sector
  Exit buffer: Only exit if score dropped 15+ pts from entry
  Hard stop:   Force exit if price down 15%+ from entry
  Rebalance:   Monthly (~21 trading days)
  Min score:   40 / 100
  Min history: 210 trading days

Midcap 50 version:
  Hold top:    10 stocks (smaller universe)
  Sector cap:  Max 3 stocks per sector
  Same exit buffer, hard stop, min score, min history
  Not walk-forward validated AND not point-in-time validated — all midcap numbers are survivorship-biased. Paper trade
  2 months first.
```

> NOTE — "hard stop" is checked only on the monthly rebalance, not intraday. A holding
> can fall well below −15% mid-month before the next run catches it.

### NiftyShop strategy parameters (mean-reversion)

```
Fresh entry:   Buy 1 stock/week from top 5 most-fallen below 20DMA
               Only if NOT already in portfolio AND slots available
Quality gate:  Stock must be ABOVE 200DMA (long-term uptrend intact)
Avg entry:     If top 5 all held → average worst-held stock
               Trigger: current price < last_buy_price × 0.97 (-3%)
               Cap: total invested per stock ≤ max_per_stock
               NEVER average a stock below its 200DMA (or with unknown 200DMA) — added
               2026-10-08; the 200DMA gate now protects averaging, not just fresh entry.
               Applied in both deployers and dashboard.py. Not backtested.
Exit:          current price ≥ avg_buy_price × 1.08 (+8%)
Priority:      SELL → AVERAGE → FRESH BUY
Max positions: 5 stocks simultaneously

Nifty 50:      ₹10K fresh, ₹15K avg, ₹40K max/stock
Midcap 50:     ₹5K–6.25K fresh, ₹7.5K–9.375K avg (smaller sizes, higher risk)
```

### NiftyShop performance

**Point-in-time validated (idle cash @6.5%, as of 2026-05-29; pit_backtest.py has since been removed):**

```
Nifty 50, 3y PIT:   CAGR  8.6%  |  Jensen alpha +1.6%  |  MDD -4.7%  |  beta 0.26
Nifty 50, 2y PIT:   CAGR  7.0%  |  Jensen alpha +1.7%  |  MDD -4.8%
Nifty 50, 1y PIT:   CAGR  6.1%  |  Jensen alpha +3.3%  |  MDD -4.2%
Midcap 50:          survivorship-biased (snapshot only) — 3y lagged bench (~11% vs 19%)
```

It is a LOW-RISK / LOW-RETURN sleeve: ~6-9% CAGR with small alpha but a genuine
drawdown advantage (-4.7% vs index -20%). The "100% hit ratio" is the mean-reversion
illusion — winners sell at +8%, losers are never sold (no stop), so realised trades
always look green while capital sits idle.

**Old claim (Backtest/niftyshop_universe_backtest.py — SURVIVORSHIP-BIASED, NOT validated):**

```
Without 200DMA filter:  XIRR ~13-21%   ← NOT reproduced point-in-time; real ≈ 6-9% CAGR
With 200DMA filter:     XIRR ~16-18%   ← was "estimated", never measured
200DMA filter is ACTIVE on both deployers
```

### Holdings file format (for the parked momentum deployers)

```
# One stock per line: SYMBOL  ENTRY_PRICE  ENTRY_SCORE
HDFCBANK    745.00    72
SBIN        285.00    68
BAJFINANCE  900.00    75
```

Files: `momentum/current_holdings.txt` (Nifty 50), `momentum/midcap_holdings.txt` (Midcap 50). Paper only — no real momentum positions.
Comments (`#`) and blank lines are ignored. Symbol-only lines are treated as zero-basis entries.

### Trade log format (for NiftyShop deployers)

```
date,symbol,action,price,quantity,amount
2026-05-28,ONGC,BUY_FRESH,274.05,36,10000
2026-06-06,ONGC,BUY_AVG,261.00,57,15000
2026-08-15,ONGC,SELL,319.00,93,29667
```

Files: `trade_log.csv` (Nifty 50), `midcap_trade_log.csv` (Midcap 50).
Actions: `BUY_FRESH | BUY_AVG | SELL`.

---

### Data layer

All files call `yfinance.download()` directly — no database, no caching. Tickers use the `.NS` suffix for NSE-listed stocks. Benchmarks: `^NSEI` (Nifty 50), `^NSEMDCP50` (Nifty Midcap 50). Do NOT use `^NSMIDCP` — on Yahoo that ticker is the Nifty **Next** 50. An `18mo` look-back is fetched for the momentum deployers.

### Shared constants

Universes, sector maps and benchmarks live in `config.py` and the four deployers import them.
Exceptions that still embed their own (stale) lists: `regime_detector.py` (NIFTY50 for breadth), `value_screener.py`.

---

### Validated performance (Nifty 50 momentum)

**Point-in-time validated (daily NAV, survivorship-corrected, as of 2026-05-29; pit_backtest.py has since been removed):**

| Horizon | CAGR (PIT) | Bench | Jensen α (PIT) | Sharpe | Max DD |
|---------|-----------:|------:|---------------:|-------:|-------:|
| 3y      | 15.6%      | 8.2%  | **+7.2%**      | 0.56   | **−20.5%** |
| 2y      | 0.4%       | 1.8%  | **−0.6%**      | −0.21  | −18.2% |
| 1y      | −1.8%      | −5.2% | +4.4%          | −0.42  | −14.3% |

Three things the old numbers got wrong:
1. **Drawdown is −20.5%, not −15.51%.** The old figure came from monthly-step NAV that
   hid intra-month drops; the PIT test used daily NAV.
2. **The alpha is real but NOT stationary.** The 3y alpha (+7.2%) rides the oldest third
   of the window; the trailing **2 years delivered ~0% alpha and negative Sharpe.**
3. Survivorship bias inflated the old alpha by only ~1-1.7pp — so the edge is mostly
   genuine, just smaller and lumpier than 8.68% implied.

**Old claims (survivorship-biased snapshot backtests — kept for reference only):**

- *v3 baseline:* Jensen's alpha 8.41% (6.74% OOS) · Sharpe 0.87 · MDD −12.28% · Hit 56.7% · Turnover ~330%
- *v6:* Jensen's alpha 8.68% OOS · Sharpe 0.79 OOS · MDD −15.51% OOS · Hit 60.7% OOS
  — all from `Backtest/` runs on **today's constituents over multiple years** (survivorship bias).
