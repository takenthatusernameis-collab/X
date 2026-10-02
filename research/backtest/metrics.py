"""Risk and performance metrics computed from an equity curve.

All metrics are deterministic functions of the equity curve and
trades. No external state is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np


@dataclass(frozen=True)
class TradeSummary:
    n_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float


@dataclass
class Metrics:
    """Performance metrics for one backtest run."""
    n_periods: int
    total_return: float
    annualized_return: float
    vol_annual: float
    sharpe: float
    max_drawdown: float
    calmar: float
    n_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    turnover_ratio: float
    final_cash: float
    final_position_shares: float


def trade_summary_from_fills(fills, fill_prices: np.ndarray) -> TradeSummary:
    """Round-trip PnL summary from fills.

    A round trip is completed when cumulative signed shares return to
    zero or change sign. PnL = shares * (exit - entry) - commission.
    """
    fills = list(fills)
    if not fills:
        return TradeSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    cum = 0.0
    entry_price = 0.0
    entry_shares_abs = 0.0
    entry_commission = 0.0
    profits = []
    losses = []
    entry_shares = 0.0

    for f in fills:
        entry_shares += f.shares
        if abs(entry_shares) < 1e-12:
            # round trip complete: shares went back to zero
            pnl = entry_shares_abs * (f.price - entry_price) - entry_commission - f.commission
        elif entry_shares * (entry_shares - f.shares) < 0 and abs(f.shares) >= abs(entry_shares):
            # direction change: exit old position, enter new
            pnl = entry_shares_abs * (f.price - entry_price) - entry_commission - f.commission
            entry_shares_abs = abs(f.shares)
            entry_price = f.price
            entry_commission = f.commission
            cum += pnl
            if pnl > 0:
                profits.append(pnl)
            else:
                losses.append(pnl)
            continue
        else:
            entry_shares_abs = abs(entry_shares)
            entry_price = f.price
            entry_commission = f.commission
            continue
        cum += pnl
        if pnl > 0:
            profits.append(pnl)
        else:
            losses.append(pnl)

    n = len(profits) + len(losses)
    winning = len(profits)
    losing = len(losses)
    gp = float(sum(profits))
    gl = abs(float(sum(losses)))
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0.0
    win_rate = winning / n if n else 0.0
    return TradeSummary(n, winning, losing, win_rate, gp, gl, pf)


def compute_metrics(
    equity_curve: np.ndarray,
    fills=None,
    fill_prices=None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> Metrics:
    """Compute performance and risk metrics from an equity curve.

    Args:
        equity_curve: equity after each bar (1-D numpy array).
        fills: fill records, used to compute round-trip PnL stats.
        fill_prices: matching prices array, used with fills.
        risk_free: annual risk-free rate.
        periods_per_year: bars per year for annualization.

    Returns:
        Metrics with total/annualized return, volatility, Sharpe,
        maximum drawdown, Calmar, trade stats and turnover.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    n = len(eq)
    if n < 2:
        raise ValueError("equity curve must have at least 2 points")

    starts, ends = eq[0], eq[-1]
    total_return = ends / starts - 1.0
    n_years = n / periods_per_year
    annualized_return = (1.0 + max(total_return, -1.0 + 1e-12)) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    returns = np.diff(eq) / eq[:-1]
    vol_annual = float(np.std(returns, ddof=1)) * np.sqrt(periods_per_year) if n > 1 else 0.0
    excess = annualized_return - risk_free
    sharpe = excess / vol_annual if vol_annual > 0 else 0.0

    running_max = np.maximum.accumulate(eq)
    drawdown = (running_max - eq) / running_max
    max_drawdown = float(np.max(drawdown))
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    trade = trade_summary_from_fills(fills, fill_prices) if (fills is not None and fill_prices is not None) else TradeSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    # turnover: sum of absolute shares traded * fill price / average equity
    turnover = 0.0
    if fills is not None and fill_prices is not None:
        fill_prices = np.asarray(fill_prices, dtype=np.float64)
        notional_turnover = np.sum(np.abs(np.asarray([f.shares for f in fills])) * fill_prices)
        avg_equity = np.mean(eq)
        turnover = notional_turnover / avg_equity if avg_equity > 0 else 0.0

    # final position state
    cum_shares = sum(f.shares for f in fills) if fills is not None else 0.0

    return Metrics(
        n_periods=n,
        total_return=total_return,
        annualized_return=annualized_return,
        vol_annual=vol_annual,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        calmar=calmar,
        n_trades=trade.n_trades,
        winning_trades=trade.winning_trades,
        losing_trades=trade.losing_trades,
        win_rate=trade.win_rate,
        gross_profit=trade.gross_profit,
        gross_loss=trade.gross_loss,
        profit_factor=trade.profit_factor,
        turnover_ratio=turnover,
        final_cash=ends - cum_shares * eq[-1],
        final_position_shares=cum_shares,
    )
