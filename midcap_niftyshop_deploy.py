"""
WealthOS — MidcapShop Production Deploy
========================================
Mean-reversion strategy on Nifty Midcap 50. Run every Friday at 3:20pm.
Signal only — you execute manually in Zerodha.

⚠ RISK WARNING (backed by our own backtest)
  Midcap 50 NiftyShop showed MDD -11.93% vs Nifty 50's -4.70%.
  Mid-caps fall harder and recover slower than large-caps.
  Averaging into a mid-cap in structural decline is more dangerous.
  200DMA filter is ACTIVE to mitigate this — but does not eliminate it.
  Start with smaller position sizes (₹5K fresh, ₹7.5K avg) until validated.

STRATEGY RULES (identical to niftyshop_deploy.py)
  Universe:       Nifty Midcap 50
  Signal:         Weekly 20DMA deviation scan
  Quality filter: Only stocks ABOVE 200DMA (critical for midcaps)
  Fresh entry:    Buy 1 stock/week from top 5 most-fallen below 20DMA
  Avg entry:      If top 5 all held → average worst held stock
                  Trigger: -3% below last buy price
                  Cap: total invested per stock ≤ max_per_stock
                  Never average a stock that is below its 200DMA
                  (or whose 200DMA can't be determined)
  Exit:           current price ≥ avg_buy_price × 1.08 (+8%)
  Priority:       SELL → AVERAGE → FRESH BUY
  Max positions:  5 stocks

TRACKING FILES (same directory as this script)
  midcap_trade_log.csv — separate from niftyshop trade log

TRADE LOG FORMAT:
  date,symbol,action,price,quantity,amount
  2026-05-28,HAVELLS,BUY_FRESH,1450.00,6,8700
  2026-06-06,HAVELLS,BUY_AVG,1389.00,10,13890
  2026-08-15,HAVELLS,SELL,1672.00,16,26752

  Actions: BUY_FRESH | BUY_AVG | SELL

USAGE
-----
  python midcap_niftyshop_deploy.py                    # weekly signal
  python midcap_niftyshop_deploy.py --capital 200000   # ₹2L sleeve
  python midcap_niftyshop_deploy.py --fresh 5000 --avg 7500
  python midcap_niftyshop_deploy.py --status           # portfolio only
  python midcap_niftyshop_deploy.py --history          # closed P&L

WEEKLY WORKFLOW (5 min every Friday)
-------------------------------------
  3:15pm → python midcap_niftyshop_deploy.py
  3:20pm → read the signal
  3:25pm → execute 1 trade in Zerodha
  3:30pm → add 1 row to midcap_trade_log.csv
"""
from __future__ import annotations
import argparse, csv, warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from regime_detector import print_regime_header
warnings.filterwarnings("ignore")

# ── Universe — imported from config.py (single source of truth) ───────────────
from config import MIDCAP50_NAMES as MIDCAP50

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    total_capital: float = 200_000   # ₹2L default — smaller for midcap
    fresh_amount:  float = 5_000     # ₹5K — smaller than Nifty 50 version
    avg_amount:    float = 7_500     # ₹7.5K
    max_per_stock: float = 20_000    # ₹20K max per midcap position
    max_positions: int   = 5
    profit_target: float = 0.08      # +8% above avg buy price
    avg_trigger:   float = 0.03      # -3% below last buy price
    top_n:         int   = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "midcap_trade_log.csv"   # separate from Nifty 50 log
LOG_COLS = ["date", "symbol", "action", "price", "quantity", "amount"]

# ── Trade log ─────────────────────────────────────────────────────────────────
def ensure_log() -> None:
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLS)
        print(f"  ✅ Created midcap_trade_log.csv at {LOG_FILE}")

def load_log() -> pd.DataFrame:
    ensure_log()
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    if df.empty:
        return df
    for col in ["price", "quantity", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["action"] = df["action"].str.strip().str.upper()
    return df.dropna(subset=["price", "quantity", "amount"])

# ── Portfolio model ───────────────────────────────────────────────────────────
@dataclass
class Lot:
    date:     object
    action:   str
    price:    float
    quantity: float
    amount:   float

@dataclass
class Position:
    symbol: str
    lots:   list[Lot] = field(default_factory=list)

    @property
    def total_invested(self) -> float:
        return sum(l.amount for l in self.lots)

    @property
    def total_qty(self) -> float:
        return sum(l.quantity for l in self.lots)

    @property
    def avg_price(self) -> float:
        tq = self.total_qty
        return sum(l.price * l.quantity for l in self.lots) / tq if tq > 0 else 0

    @property
    def last_buy_price(self) -> float:
        return self.lots[-1].price if self.lots else 0

    @property
    def n_lots(self) -> int:
        return len(self.lots)

    def can_average(self, avg_amount: float, max_per_stock: float) -> bool:
        return self.total_invested + avg_amount <= max_per_stock

    def apply_sell(self, sold_qty: float) -> bool:
        """FIFO lot reduction. Returns True if position is fully closed."""
        remaining = sold_qty
        new_lots = []
        for lot in self.lots:
            if remaining <= 0:
                new_lots.append(lot)
            elif lot.quantity <= remaining:
                remaining -= lot.quantity
            else:
                kept_frac = (lot.quantity - remaining) / lot.quantity
                new_lots.append(Lot(
                    date=lot.date, action=lot.action, price=lot.price,
                    quantity=lot.quantity - remaining,
                    amount=lot.amount * kept_frac,
                ))
                remaining = 0
        self.lots = new_lots
        return self.total_qty <= 0.5

    def avg_triggered(self, cur: float, trig: float) -> bool:
        return cur <= self.last_buy_price * (1 - trig)

    def exit_triggered(self, cur: float, target: float) -> bool:
        return cur >= self.avg_price * (1 + target)

    def unrealised_pnl(self, cur: float) -> float:
        return (cur - self.avg_price) * self.total_qty

    def pnl_pct(self, cur: float) -> float:
        return (cur / self.avg_price - 1) * 100 if self.avg_price > 0 else 0


def build_portfolio(log: pd.DataFrame) -> dict[str, Position]:
    """Reconstruct open positions from full trade log. Supports partial exits (FIFO)."""
    positions: dict[str, Position] = {}
    for _, row in log.iterrows():
        sym, action = row["symbol"], row["action"]
        if action in ("BUY_FRESH", "BUY_AVG"):
            if sym not in positions:
                positions[sym] = Position(sym)
            positions[sym].lots.append(Lot(
                date=row["date"], action=action,
                price=float(row["price"]),
                quantity=float(row["quantity"]),
                amount=float(row["amount"]),
            ))
        elif action == "SELL" and sym in positions:
            if positions[sym].apply_sell(float(row["quantity"])):
                del positions[sym]
    return positions


def get_closed_trades(log: pd.DataFrame) -> list[dict]:
    """Replay log to extract closed positions with P&L. Supports partial exits (FIFO)."""
    positions: dict[str, Position] = {}
    closed = []
    for _, row in log.iterrows():
        sym, action = row["symbol"], row["action"]
        if action in ("BUY_FRESH", "BUY_AVG"):
            if sym not in positions:
                positions[sym] = Position(sym)
            positions[sym].lots.append(Lot(
                date=row["date"], action=action,
                price=float(row["price"]),
                quantity=float(row["quantity"]),
                amount=float(row["amount"]),
            ))
        elif action == "SELL" and sym in positions:
            pos       = positions[sym]
            sold_qty  = float(row["quantity"])
            proceeds  = float(row["amount"])
            sold_cost = (sold_qty / pos.total_qty) * pos.total_invested if pos.total_qty > 0 else pos.total_invested
            pnl       = proceeds - sold_cost
            entry_dt  = pos.lots[0].date
            exit_dt   = row["date"]
            hold      = (exit_dt - entry_dt).days if hasattr(exit_dt - entry_dt, "days") else 0
            closed.append({
                "symbol":   sym,
                "entry":    str(entry_dt)[:10],
                "exit":     str(exit_dt)[:10],
                "hold_days":hold,
                "n_lots":   pos.n_lots,
                "invested": round(sold_cost, 2),
                "proceeds": round(proceeds, 2),
                "pnl":      round(pnl, 2),
                "pnl_pct":  round(pnl / sold_cost * 100, 2) if sold_cost > 0 else 0.0,
            })
            if pos.apply_sell(sold_qty):
                del positions[sym]
    return closed


# ── Market data ───────────────────────────────────────────────────────────────
def fetch_prices() -> dict[str, dict]:
    """
    Download 12 months of Midcap 50 prices.
    200DMA quality filter is CRITICAL for midcaps — stocks below 200DMA
    can stay depressed for 1-2 years. Only buy quality dips.
    """
    import yfinance as yf
    tickers = [f"{s}.NS" for s in MIDCAP50]
    raw = yf.download(tickers, period="12mo", progress=False,
                      auto_adjust=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    result   = {}
    filtered = {}

    for sym in MIDCAP50:
        col = sym + ".NS"
        if col not in close.columns:
            continue
        s = close[col].dropna()
        if len(s) < 20:
            continue
        price = float(s.iloc[-1])
        dma20 = float(s.tail(20).mean())
        if dma20 <= 0 or price <= 0:
            continue

        # 200DMA quality filter — non-negotiable for midcaps
        # Require 200 bars minimum — stocks with less history (recent IPOs)
        # cannot be evaluated on long-term trend and are excluded.
        if len(s) < 200:
            continue   # insufficient history for 200DMA quality check
        dma200 = float(s.tail(200).mean())
        if price < dma200:
            filtered[sym] = {
                "name":      MIDCAP50[sym],
                "price":     round(price, 2),
                "dma200":    round(dma200, 2),
                "deviation": round((price - dma20) / dma20 * 100, 2),
            }
            continue   # skip — structural decline, not a dip

        result[sym] = {
            "name":      MIDCAP50[sym],
            "price":     round(price, 2),
            "dma20":     round(dma20, 2),
            "deviation": round((price - dma20) / dma20 * 100, 2),
            "above200":  True,
        }

    result["__filtered__"] = filtered   # type: ignore[assignment]
    return result


# ── Signal ────────────────────────────────────────────────────────────────────
def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch latest close for any symbols — no DMA filter. Used for portfolio holdings."""
    import yfinance as yf
    if not symbols:
        return {}
    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(tickers, period="5d", progress=False,
                      auto_adjust=True, threads=True)
    if raw.empty:
        return {}
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    result: dict[str, float] = {}
    for sym in symbols:
        col = sym + ".NS"
        if col in close.columns:
            s = close[col].dropna()
            if len(s) > 0:
                result[sym] = round(float(s.iloc[-1]), 2)
    return result


def _inject_portfolio_prices(portfolio: dict, mkt: dict, filtered: dict) -> None:
    """
    Add current prices for holdings not captured by the DMA filter — without this,
    a held stock below its 200DMA would vanish (no exit check, no P&L).
    above200: True = above 200DMA (already in mkt), False = below (from the
    200DMA filter), None = unknown (left the universe / <200 bars).
    Averaging is only allowed when above200 is True.
    """
    for sym in portfolio:
        if sym in mkt or sym not in filtered:
            continue
        d = filtered[sym]
        mkt[sym] = {
            "name":      d["name"],
            "price":     d["price"],
            "dma20":     0.0,
            "deviation": d["deviation"],
            "dma200":    d["dma200"],
            "above200":  False,
        }
    missing = [s for s in portfolio if s not in mkt]
    if not missing:
        return
    live = fetch_current_prices(missing)
    for sym, price in live.items():
        mkt[sym] = {
            "name":      MIDCAP50.get(sym, sym),
            "price":     price,
            "dma20":     0.0,
            "deviation": 0.0,
            "above200":  None,   # unknown — not fetched
        }


def generate_signal(cfg: Config) -> None:
    today_str = datetime.now().strftime("%Y-%m-%d")
    dow       = datetime.now().strftime("%A")

    print()
    print("═" * 68)
    print(f"  MIDCAPSHOP SIGNAL — {dow}, {today_str}")
    print(f"  ₹{cfg.total_capital:,.0f} capital  │  "
          f"Fresh ₹{cfg.fresh_amount:,.0f}  │  "
          f"Avg ₹{cfg.avg_amount:,.0f}  │  "
          f"Max/stock ₹{cfg.max_per_stock:,.0f}")
    print(f"  Universe: Nifty Midcap 50  │  200DMA filter: ACTIVE")
    print("═" * 68)

    log       = load_log()
    portfolio = build_portfolio(log)

    print("\n  Fetching Midcap 50 prices (12mo, 200DMA filter active)...")
    mkt_raw  = fetch_prices()
    filtered = mkt_raw.pop("__filtered__", {})
    mkt      = {k: v for k, v in mkt_raw.items() if not k.startswith("__")}

    # Ensure portfolio holdings always have a live price (they may be below 200DMA)
    _inject_portfolio_prices(portfolio, mkt, filtered)

    if not mkt:
        print("  ❌ Could not fetch prices. Check internet.")
        return

    n_filtered = len(filtered)
    if n_filtered:
        print(f"  ℹ  {n_filtered} stock(s) blocked by 200DMA filter:")
        for sym, d in sorted(filtered.items(), key=lambda x: x[1]["deviation"]):
            print(f"     {sym:<13} price ₹{d['price']:.0f}  "
                  f"below 200DMA ₹{d['dma200']:.0f}  "
                  f"20DMA dev {d['deviation']:+.1f}%")

    # STEP 1: SELL
    _header("STEP 1 — EXIT CHECK")
    exit_signals: list[tuple] = []
    for sym, pos in portfolio.items():
        if sym not in mkt: continue
        cur = mkt[sym]["price"]
        if pos.exit_triggered(cur, cfg.profit_target):
            exit_signals.append((sym, pos, cur))

    if exit_signals:
        for sym, pos, cur in exit_signals:
            pnl_pct = pos.pnl_pct(cur)
            pnl_rs  = pos.unrealised_pnl(cur)
            print(f"\n  ✅ SELL {sym} — {MIDCAP50.get(sym, sym)}")
            print(f"     Avg buy price : ₹{pos.avg_price:.2f}")
            print(f"     Current price : ₹{cur:.2f}  ({pnl_pct:+.1f}%)")
            print(f"     Target price  : ₹{pos.avg_price*(1+cfg.profit_target):.2f}")
            print(f"     Expected P&L  : ₹{pnl_rs:,.0f}")
            print(f"     Qty to sell   : {pos.total_qty:.0f} shares")
            print(f"     Lots held     : {pos.n_lots}  │  First buy: {pos.lots[0].date}")
    else:
        print("  No exits triggered.")

    # STEP 2: AVERAGING
    _header("STEP 2 — AVERAGING CHECK  (if no exit today)")
    avg_candidates: list[tuple] = []
    for sym, pos in portfolio.items():
        if sym not in mkt: continue
        cur       = mkt[sym]["price"]
        gap_pct   = (cur / pos.last_buy_price - 1) * 100
        triggered = pos.avg_triggered(cur, cfg.avg_trigger)
        can_avg   = pos.can_average(cfg.avg_amount, cfg.max_per_stock)
        above200  = mkt[sym].get("above200")
        avg_candidates.append((gap_pct, sym, pos, cur, triggered, can_avg, above200))

    avg_candidates.sort(key=lambda x: x[0])   # worst first
    avg_action: tuple | None = None

    for gap_pct, sym, pos, cur, triggered, can_avg, above200 in avg_candidates:
        new_total = pos.total_invested + cfg.avg_amount
        if triggered and above200 is not True:
            why   = "below 200DMA" if above200 is False else "200DMA unknown"
            label = f"🚫 NO AVERAGING — {why}"
            note  = "  → rule: never average a stock below its 200DMA"
        elif triggered and can_avg:
            label = "✅ AVERAGE"
            note  = (f"  → spend ₹{cfg.avg_amount:,.0f}, "
                     f"total ₹{new_total:,.0f} / ₹{cfg.max_per_stock:,.0f}")
        elif triggered and not can_avg:
            label = "⛔ AT CAP"
            note  = f"  → at ₹{pos.total_invested:,.0f} max"
        else:
            label = f"⬜ watching  (needs -{cfg.avg_trigger*100:.0f}%, at {gap_pct:.1f}%)"
            note  = ""

        print(f"\n  {label}  {sym} — {MIDCAP50.get(sym, sym)}")
        print(f"     Last buy  : ₹{pos.last_buy_price:.2f}")
        print(f"     Current   : ₹{cur:.2f}  ({gap_pct:+.1f}% vs last buy)")
        print(f"     Avg price : ₹{pos.avg_price:.2f}  │  "
              f"Invested: ₹{pos.total_invested:,.0f}  ({pos.n_lots} lot{'s' if pos.n_lots>1 else ''})")
        if note:
            print(f"    {note}")

        if triggered and can_avg and above200 is True and not exit_signals and avg_action is None:
            avg_action = (sym, cur)

    if not avg_candidates:
        print("  No open positions to check.")
    if avg_action and not exit_signals:
        sym, cur = avg_action
        print(f"\n  ⚡ ACTION: AVERAGE {sym} @ ~₹{cur:.2f}"
              f"  → ₹{cfg.avg_amount:,.0f} (~{cfg.avg_amount/cur:.0f} shares)")
    elif not exit_signals and not avg_action:
        print("\n  No averaging today.")

    # STEP 3: NEW ENTRY
    _header("STEP 3 — NEW ENTRY SCAN  (if no exit and no avg today)")
    slots = cfg.max_positions - len(portfolio)
    below = sorted(
        [(d["deviation"], sym, d)
         for sym, d in mkt.items()
         if d["deviation"] < 0 and sym not in portfolio],
        key=lambda x: x[0]
    )

    print(f"\n  Slots free: {slots} / {cfg.max_positions}")
    print(f"\n  {'Rank':<5} {'Symbol':<13} {'Company':<30}"
          f" {'Price':>9} {'20DMA':>9} {'Dev%':>7}  Signal")
    print(f"  {'─'*4} {'─'*12} {'─'*29} {'─'*9} {'─'*9} {'─'*7}  {'─'*10}")

    entry_action: tuple | None = None
    for i, (dev, sym, d) in enumerate(below[:10], 1):
        in_top5   = i <= cfg.top_n
        can_enter = (in_top5 and slots > 0
                     and not exit_signals and not avg_action)
        marker = "→ BUY" if (can_enter and entry_action is None) else \
                 "top5"  if in_top5 else ""
        print(f"  {i:<5} {sym:<13} {d['name'][:29]:<30}"
              f" ₹{d['price']:>8.2f} ₹{d['dma20']:>8.2f}"
              f" {dev:>6.1f}%  {marker}")
        if can_enter and entry_action is None:
            entry_action = (sym, d["price"])

    if not below:
        print("  No Midcap 50 stocks below 20DMA (and above 200DMA) today.")
    elif slots == 0:
        print(f"\n  All {cfg.max_positions} slots occupied.")
    elif exit_signals or avg_action:
        print("\n  (Sell/average takes priority today.)")
    elif entry_action:
        sym, price = entry_action
        print(f"\n  ⚡ ACTION: BUY {sym} ({MIDCAP50.get(sym, sym)})"
              f" @ ~₹{price:.2f}"
              f" → ₹{cfg.fresh_amount:,.0f} (~{cfg.fresh_amount/price:.0f} shares)")

    # TODAY'S ACTION
    print()
    print("═" * 68)
    if exit_signals:
        action = "SELL " + ", ".join(s for s,_,_ in exit_signals)
    elif avg_action:
        sym, cur = avg_action
        action = f"AVERAGE {sym} @ ~₹{cur:.2f}  (₹{cfg.avg_amount:,.0f})"
    elif entry_action:
        sym, price = entry_action
        action = f"BUY {sym} @ ~₹{price:.2f}  (₹{cfg.fresh_amount:,.0f})"
    else:
        action = "NOTHING — hold all positions, check again next Friday"

    print(f"  TODAY'S ACTION: {action}")
    print("═" * 68)

    if "NOTHING" not in action:
        print()
        print("  AFTER EXECUTING → add to midcap_trade_log.csv:")
        if exit_signals:
            for sym, pos, cur in exit_signals:
                print(f"  {today_str},{sym},SELL,<price>,{pos.total_qty:.0f},<amount>")
        elif avg_action:
            sym, cur = avg_action
            print(f"  {today_str},{sym},BUY_AVG,<price>,{int(cfg.avg_amount/cur)},<amount>")
        elif entry_action:
            sym, price = entry_action
            print(f"  {today_str},{sym},BUY_FRESH,<price>,{int(cfg.fresh_amount/price)},<amount>")
        print("  Replace <price> and <amount> with Zerodha confirmation values.")

    _print_portfolio(portfolio, mkt, cfg)


def _print_portfolio(portfolio, mkt, cfg):
    print()
    _header("CURRENT MIDCAP PORTFOLIO")
    if not portfolio:
        print("  No open positions.\n")
        return

    print(f"\n  {'Symbol':<13} {'Invested':>10} {'Lots':>5} {'Avg ₹':>9}"
          f" {'Cur ₹':>9} {'P&L%':>7} {'to exit':>9} {'P&L ₹':>9}")
    print(f"  {'─'*12} {'─'*10} {'─'*5} {'─'*9} {'─'*9} {'─'*7} {'─'*9} {'─'*9}")

    tot_inv = 0; tot_pnl = 0
    for sym, pos in sorted(portfolio.items()):
        cur    = mkt.get(sym, {}).get("price", 0)
        pnl    = pos.unrealised_pnl(cur) if cur else 0
        pp     = pos.pnl_pct(cur) if cur else 0
        target = pos.avg_price * (1 + cfg.profit_target)
        to_tgt = (cur / target - 1) * 100 if cur and target else 0
        icon   = "▲" if pnl >= 0 else "▼"
        avg_hit = bool(cur and pos.avg_triggered(cur, cfg.avg_trigger))
        ok200   = mkt.get(sym, {}).get("above200") is True
        alert  = " ← EXIT?" if (cur and pos.exit_triggered(cur, cfg.profit_target)) else \
                 " ← AVG?" if (avg_hit and ok200
                                and pos.can_average(cfg.avg_amount, cfg.max_per_stock)) else \
                 " ⊘ <200DMA, no avg" if (avg_hit and not ok200) else ""
        print(f"  {sym:<13} ₹{pos.total_invested:>8,.0f} {pos.n_lots:>5}"
              f" ₹{pos.avg_price:>8.2f} ₹{cur:>8.2f}"
              f" {pp:>+6.1f}% {to_tgt:>+8.1f}%"
              f" {icon}₹{abs(pnl):>7,.0f}{alert}")
        tot_inv += pos.total_invested
        tot_pnl += pnl

    idle = cfg.total_capital - tot_inv
    icon = "▲" if tot_pnl >= 0 else "▼"
    print(f"  {'─'*12} {'─'*10}")
    print(f"  {'TOTAL':<13} ₹{tot_inv:>8,.0f}{'':>35} "
          f"{icon}₹{abs(tot_pnl):>7,.0f}")
    print(f"\n  Deployed: ₹{tot_inv:,.0f}  │  "
          f"Idle: ₹{idle:,.0f}  │  "
          f"Slots: {cfg.max_positions - len(portfolio)} free")
    print(f"  ⚠  Idle ₹{idle:,.0f} → park in liquid fund (~6.5% p.a.)\n")

    print("  ZERODHA ALERTS (7.5% above avg buy price):")
    for sym, pos in sorted(portfolio.items()):
        print(f"    {sym:<13} alert @ ₹{pos.avg_price*1.075:.2f}"
              f"  (avg ₹{pos.avg_price:.2f})")
    print()


def print_history() -> None:
    log    = load_log()
    closed = get_closed_trades(log)
    if not closed:
        print("\n  No closed trades yet.\n")
        return

    print()
    print("═" * 72)
    print("  MIDCAPSHOP — CLOSED TRADE HISTORY")
    print("═" * 72)
    print(f"\n  {'Symbol':<13} {'Entry':>10} {'Exit':>10} {'Days':>5}"
          f" {'Lots':>4} {'Invested':>10} {'P&L ₹':>9} {'P&L%':>7}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*5} {'─'*4} {'─'*10} {'─'*9} {'─'*7}")

    tot_pnl = 0; wins = 0
    for t in closed:
        icon = "✅" if t["pnl"] >= 0 else "❌"
        print(f"  {icon} {t['symbol']:<11} {t['entry']:>10} {t['exit']:>10}"
              f" {t['hold_days']:>5} {t['n_lots']:>4}"
              f" ₹{t['invested']:>8,.0f} ₹{t['pnl']:>7,.0f} {t['pnl_pct']:>+6.1f}%")
        tot_pnl += t["pnl"]
        wins += 1 if t["pnl"] >= 0 else 0

    n = len(closed)
    print(f"\n  Closed: {n}  │  Win rate: {wins/n*100:.1f}%  │  "
          f"Total P&L: ₹{tot_pnl:,.0f}  │  "
          f"Avg hold: {sum(t['hold_days'] for t in closed)/n:.0f} days\n")


def _header(title: str) -> None:
    print()
    print(f"  {'─'*66}")
    print(f"  {title}")
    print(f"  {'─'*66}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="MidcapShop — weekly mean-reversion")
    p.add_argument("--capital", type=float, default=200_000)
    p.add_argument("--fresh",   type=float, default=5_000)
    p.add_argument("--avg",     type=float, default=7_500)
    p.add_argument("--max",     type=float, default=20_000)
    p.add_argument("--status",  action="store_true")
    p.add_argument("--history", action="store_true")
    a = p.parse_args()

    cfg = Config(total_capital=a.capital, fresh_amount=a.fresh,
                 avg_amount=a.avg, max_per_stock=a.max)

    print_regime_header()
    print(f"\n  WealthOS MidcapShop — Mean Reversion on Nifty Midcap 50")
    ensure_log()

    if a.history:
        print_history(); return

    if a.status:
        log       = load_log()
        portfolio = build_portfolio(log)
        mkt_raw   = fetch_prices()
        filtered  = mkt_raw.pop("__filtered__", {})
        mkt       = {k: v for k, v in mkt_raw.items() if not k.startswith("__")}
        _inject_portfolio_prices(portfolio, mkt, filtered)
        _print_portfolio(portfolio, mkt, cfg); return

    generate_signal(cfg)


if __name__ == "__main__":
    main()
