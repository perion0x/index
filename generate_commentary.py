"""Generate a Substack-ready commodity commentary from GSCI / BCOM data.

Run:
    python generate_commentary.py               # prints Markdown to stdout
    python generate_commentary.py --out post.md # writes to file

No Streamlit dependency — pure analytics + prose generation.
"""

import argparse
import os
import sys
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CONSTANTS  (mirror app.py)
# ============================================================
TRADING_DAYS = 252
ROLL_VOL_LONG = 252
ROLL_VOL_SHORT = 21
ROLL_CORR_LONG = 90
ROLL_CORR_SHORT = 30
ROLL_BETA_WINDOW = 252

GSCI_FILE = "SPGSCI_historical.csv"
BCOM_FILE = "BCOM_historical.csv"
DEFAULT_RF_ANNUAL = 0.045


# ============================================================
# DATA LAYER
# ============================================================
def _extract_close(df: pd.DataFrame, label: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df["Close"].iloc[:, 0]
        elif "Close" in df.columns.get_level_values(1):
            return df.xs("Close", axis=1, level=1).iloc[:, 0]
    else:
        if "Close" in df.columns:
            return df["Close"]
    raise ValueError(f"{label}: cannot locate 'Close' column")


def load_data() -> pd.DataFrame:
    for path in (GSCI_FILE, BCOM_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file: {path}")
    spgsci = pd.read_csv(GSCI_FILE, index_col=0, parse_dates=True, header=[0, 1])
    bcom = pd.read_csv(BCOM_FILE, index_col=0, parse_dates=True, header=[0, 1])
    sp_close = _extract_close(spgsci, GSCI_FILE)
    bcom_close = _extract_close(bcom, BCOM_FILE)
    df = pd.DataFrame({"SPGSCI": sp_close, "BCOM": bcom_close}).dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ============================================================
# ANALYTICS  (pure functions, no Streamlit)
# ============================================================
def _safe_div(num, den):
    if isinstance(den, pd.Series):
        return num / den.replace(0, np.nan)
    return np.nan if (den == 0 or np.isnan(den)) else num / den


def rf_to_daily_log(rf_annual: float) -> float:
    return float(np.log(1.0 + rf_annual) / TRADING_DAYS)


def cagr_from_log(log_returns, periods: int = TRADING_DAYS):
    return np.exp(log_returns.mean() * periods) - 1.0


def ann_vol_from_log(log_returns, periods: int = TRADING_DAYS):
    return log_returns.std() * np.sqrt(periods)


def sharpe_from_log(log_returns, rf_annual: float, periods: int = TRADING_DAYS):
    rf_d = rf_to_daily_log(rf_annual)
    std = log_returns.std()
    return _safe_div((log_returns.mean() - rf_d) * np.sqrt(periods), std)


def sortino_from_log(log_returns, rf_annual: float, periods: int = TRADING_DAYS):
    rf_d = rf_to_daily_log(rf_annual)
    shortfall = (log_returns - rf_d).clip(upper=0.0)
    dd_dev = np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods)
    excess = cagr_from_log(log_returns, periods) - rf_annual
    return _safe_div(excess, dd_dev)


def drawdown_from_log(log_returns):
    cum = np.exp(log_returns.cumsum())
    return (cum / cum.cummax() - 1.0) * 100.0


def calmar_from_log(log_returns, drawdown, periods: int = TRADING_DAYS):
    cagr = cagr_from_log(log_returns, periods)
    max_dd_abs = drawdown.min().abs() / 100.0
    return _safe_div(cagr, max_dd_abs)


def historical_var_cvar(log_returns, alpha: float):
    var = log_returns.quantile(alpha)
    cvar = pd.Series(
        {c: log_returns[c][log_returns[c] <= var[c]].mean() for c in log_returns.columns}
    )
    return var, cvar


def newey_west_sharpe_se(log_returns: pd.Series, lags: Optional[int] = None,
                         periods: int = TRADING_DAYS) -> float:
    r = log_returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    sr_ann = sharpe_from_log(r, 0.0, periods)
    if pd.isna(sr_ann) or np.isinf(sr_ann):
        return np.nan
    sr_daily = sr_ann / np.sqrt(periods)
    iid_var = (1.0 + 0.5 * sr_daily ** 2) / n
    centered = r - r.mean()
    var0 = (centered ** 2).mean()
    if var0 == 0:
        return np.nan
    adj = 1.0
    for k in range(1, lags + 1):
        rho_k = (centered.iloc[k:].values * centered.iloc[:-k].values).mean() / var0
        adj += 2.0 * (1.0 - k / (lags + 1.0)) * rho_k
    adj = max(adj, 1e-6)
    se_daily = np.sqrt(iid_var * adj)
    return float(se_daily * np.sqrt(periods))


def probabilistic_sharpe_ratio(log_returns: pd.Series, sr_benchmark: float = 0.0,
                                periods: int = TRADING_DAYS) -> float:
    r = log_returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan
    sr_hat_ann = sharpe_from_log(r, 0.0, periods)
    if pd.isna(sr_hat_ann) or np.isinf(sr_hat_ann):
        return np.nan
    sr_hat = sr_hat_ann / np.sqrt(periods)
    sr_star = sr_benchmark / np.sqrt(periods)
    g3 = float(r.skew())
    g4 = float(r.kurtosis()) + 3.0
    denom = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0:
        return np.nan
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def lower_tail_dependence(log_ret: pd.DataFrame, alpha: float = 0.05) -> float:
    if len(log_ret) < int(1 / alpha):
        return np.nan
    q_a = log_ret["SPGSCI"].quantile(alpha)
    q_b = log_ret["BCOM"].quantile(alpha)
    a_tail = log_ret["SPGSCI"] <= q_a
    b_tail = log_ret["BCOM"] <= q_b
    if a_tail.sum() == 0:
        return np.nan
    return float((a_tail & b_tail).sum() / a_tail.sum())


def max_drawdown_duration(drawdown: pd.DataFrame) -> pd.Series:
    out = {}
    for col in drawdown.columns:
        underwater = drawdown[col] < 0
        if not underwater.any():
            out[col] = 0
            continue
        runs = (underwater != underwater.shift()).cumsum()
        run_lengths = underwater.groupby(runs).sum()
        out[col] = int(run_lengths.max())
    return pd.Series(out)


def _metric_bundle(log_ret: pd.DataFrame, rf_annual: float) -> Dict:
    cum_wealth = np.exp(log_ret.cumsum())
    rebased_100 = cum_wealth / cum_wealth.iloc[0] * 100.0
    drawdown = drawdown_from_log(log_ret)
    cagr = cagr_from_log(log_ret) * 100.0
    ann_vol = ann_vol_from_log(log_ret) * 100.0
    sharpe = sharpe_from_log(log_ret, rf_annual)
    sortino = sortino_from_log(log_ret, rf_annual)
    calmar = calmar_from_log(log_ret, drawdown)
    var_95, cvar_95 = historical_var_cvar(log_ret, 0.05)
    var_99, cvar_99 = historical_var_cvar(log_ret, 0.01)
    correlation = log_ret["SPGSCI"].corr(log_ret["BCOM"])
    sharpe_se = pd.Series({c: newey_west_sharpe_se(log_ret[c]) for c in log_ret.columns})
    psr = pd.Series({c: probabilistic_sharpe_ratio(log_ret[c], 0.0) for c in log_ret.columns})
    dd_duration = max_drawdown_duration(drawdown)
    tail_dep_5 = lower_tail_dependence(log_ret, 0.05)
    return {
        "log_ret": log_ret, "cum_wealth": cum_wealth, "rebased_100": rebased_100,
        "drawdown": drawdown, "cagr": cagr, "ann_vol": ann_vol,
        "sharpe": sharpe, "sharpe_se": sharpe_se, "psr": psr,
        "sortino": sortino, "calmar": calmar,
        "var_95": var_95, "var_99": var_99, "cvar_95": cvar_95, "cvar_99": cvar_99,
        "correlation": correlation, "dd_duration": dd_duration, "tail_dep_5": tail_dep_5,
    }


def compute_metrics(df: pd.DataFrame, rf_annual: float) -> Dict:
    log_ret = np.log(df / df.shift(1)).dropna()
    bundle = _metric_bundle(log_ret, rf_annual)
    cov = log_ret.cov()
    beta_bcom_vs_gsci = cov.loc["BCOM", "SPGSCI"] / cov.loc["SPGSCI", "SPGSCI"]
    vol_252 = log_ret.rolling(ROLL_VOL_LONG).std() * np.sqrt(TRADING_DAYS)
    roll_sharpe = (
        (log_ret.rolling(ROLL_VOL_LONG).mean() - rf_to_daily_log(rf_annual))
        / log_ret.rolling(ROLL_VOL_LONG).std()
        * np.sqrt(TRADING_DAYS)
    )
    roll_corr_long = log_ret["SPGSCI"].rolling(ROLL_CORR_LONG).corr(log_ret["BCOM"])
    annual_log = log_ret.resample("YE").sum()
    annual_simple = (np.exp(annual_log) - 1.0) * 100.0
    monthly_log = log_ret.resample("ME").sum()
    monthly_simple = (np.exp(monthly_log) - 1.0) * 100.0
    bundle.update({
        "beta_bcom_vs_gsci": beta_bcom_vs_gsci,
        "vol_252": vol_252,
        "roll_sharpe": roll_sharpe,
        "roll_corr_long": roll_corr_long,
        "annual_simple": annual_simple,
        "monthly_simple": monthly_simple,
    })
    return bundle


def slice_period(df: pd.DataFrame, start: pd.Timestamp, rf_annual: float) -> Dict:
    sub = df[df.index >= pd.Timestamp(start)].copy()
    if len(sub) < 2:
        return {"empty": True, "df": sub}
    log_ret = np.log(sub / sub.shift(1)).dropna()
    bundle = _metric_bundle(log_ret, rf_annual)
    cum_pct = (sub / sub.iloc[0] - 1.0) * 100.0
    vol_roll_short = log_ret.rolling(ROLL_VOL_SHORT).std() * np.sqrt(TRADING_DAYS)
    roll_corr_short = log_ret["SPGSCI"].rolling(ROLL_CORR_SHORT).corr(log_ret["BCOM"])
    bundle.update({
        "empty": False, "df": sub, "cum_pct": cum_pct,
        "vol_roll_short": vol_roll_short, "roll_corr_short": roll_corr_short,
    })
    return bundle


# ============================================================
# NARRATIVE HELPERS
# ============================================================
def _sign(val: float, pos: str, neg: str, flat: str = "flat", threshold: float = 0.05) -> str:
    if val > threshold:
        return pos
    if val < -threshold:
        return neg
    return flat


def _vol_regime(recent_vol: float, long_vol: float) -> str:
    ratio = recent_vol / long_vol if long_vol else 1.0
    if ratio > 1.20:
        return "elevated relative to the trailing year"
    if ratio < 0.80:
        return "subdued relative to the trailing year"
    return "in line with the trailing year"


def _sharpe_tone(sr: float) -> str:
    if sr > 1.0:
        return "strong"
    if sr > 0.5:
        return "decent"
    if sr > 0.0:
        return "marginally positive"
    return "negative — investors were not compensated for the volatility taken"


def _drawdown_context(max_dd: float) -> str:
    if max_dd > -10:
        return "shallow"
    if max_dd > -20:
        return "moderate"
    if max_dd > -40:
        return "severe"
    return "catastrophic"


def _correlation_tone(corr: float, hist_corr: float) -> str:
    direction = "higher" if corr > hist_corr + 0.05 else (
        "lower" if corr < hist_corr - 0.05 else "broadly in line with"
    )
    if corr > 0.90:
        quality = "near-lockstep"
    elif corr > 0.75:
        quality = "high"
    elif corr > 0.50:
        quality = "moderate"
    else:
        quality = "low"
    return quality, direction


def _month_name(n: int) -> str:
    return ["January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"][n - 1]


def _ordinal(n: int) -> str:
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{suffixes.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th')}"


def _pct(v: float, decimals: int = 1) -> str:
    return f"{v:+.{decimals}f}%"


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


# ============================================================
# COMMENTARY BUILDER
# ============================================================
def generate_commentary(rf_annual: float = DEFAULT_RF_ANNUAL,
                        event_date: Optional[date] = date(2022, 2, 24)) -> str:
    # ── load & compute ──────────────────────────────────────
    df = load_data()
    r = compute_metrics(df, rf_annual)

    today = datetime.now()
    current_year = today.year
    current_month = today.month

    ytd_start = pd.Timestamp(date(current_year, 1, 1))
    ytd = slice_period(df, ytd_start, rf_annual)

    data_through = df.index.max().strftime("%-d %B %Y")
    data_from = df.index.min().strftime("%Y")

    # ── derived signals ─────────────────────────────────────
    # YTD totals
    ytd_gsci = ytd["cum_pct"]["SPGSCI"].iloc[-1] if not ytd["empty"] else None
    ytd_bcom = ytd["cum_pct"]["BCOM"].iloc[-1] if not ytd["empty"] else None

    # Leader / laggard
    if ytd_gsci is not None and ytd_bcom is not None:
        if abs(ytd_gsci - ytd_bcom) < 0.5:
            ytd_leader_sentence = (
                f"The two indices are virtually neck-and-neck year-to-date, "
                f"with GSCI at {_pct(ytd_gsci)} and BCOM at {_pct(ytd_bcom)}."
            )
        elif ytd_gsci > ytd_bcom:
            gap = ytd_gsci - ytd_bcom
            ytd_leader_sentence = (
                f"The S&P GSCI is outperforming BCOM by {_pct(gap, 1)} year-to-date "
                f"({_pct(ytd_gsci)} vs {_pct(ytd_bcom)}), reflecting its heavier tilt "
                f"toward energy."
            )
        else:
            gap = ytd_bcom - ytd_gsci
            ytd_leader_sentence = (
                f"Bloomberg BCOM is leading the S&P GSCI by {_pct(gap, 1)} year-to-date "
                f"({_pct(ytd_bcom)} vs {_pct(ytd_gsci)}), consistent with BCOM's broader "
                f"diversification dampening energy-driven swings."
            )
    else:
        ytd_leader_sentence = "Insufficient year-to-date data is available."

    # Volatility regime
    gsci_recent_vol = ytd["ann_vol"]["SPGSCI"] if not ytd["empty"] else None
    gsci_long_vol = r["ann_vol"]["SPGSCI"]
    bcom_recent_vol = ytd["ann_vol"]["BCOM"] if not ytd["empty"] else None
    bcom_long_vol = r["ann_vol"]["BCOM"]

    vol_comment = ""
    if gsci_recent_vol is not None:
        gsci_vol_desc = _vol_regime(gsci_recent_vol, gsci_long_vol)
        bcom_vol_desc = _vol_regime(bcom_recent_vol, bcom_long_vol)
        if gsci_vol_desc == bcom_vol_desc:
            vol_comment = (
                f"Short-term realised volatility is {gsci_vol_desc} for both indices. "
                f"GSCI is running at {_fmt(gsci_recent_vol)}% annualised versus its "
                f"long-run average of {_fmt(gsci_long_vol)}%; BCOM at "
                f"{_fmt(bcom_recent_vol)}% versus {_fmt(bcom_long_vol)}%."
            )
        else:
            vol_comment = (
                f"GSCI short-term vol is {gsci_vol_desc} ({_fmt(gsci_recent_vol)}% "
                f"vs {_fmt(gsci_long_vol)}% long-run), while BCOM vol is {bcom_vol_desc} "
                f"({_fmt(bcom_recent_vol)}% vs {_fmt(bcom_long_vol)}% long-run)."
            )

    # Rolling Sharpe trend (last reading vs 6 months ago)
    roll_sr = r["roll_sharpe"].dropna()
    gsci_sr_now = roll_sr["SPGSCI"].iloc[-1] if not roll_sr.empty else np.nan
    bcom_sr_now = roll_sr["BCOM"].iloc[-1] if not roll_sr.empty else np.nan
    lookback_6m = max(0, len(roll_sr) - 126)
    gsci_sr_6m = roll_sr["SPGSCI"].iloc[lookback_6m] if len(roll_sr) > 126 else np.nan
    bcom_sr_6m = roll_sr["BCOM"].iloc[lookback_6m] if len(roll_sr) > 126 else np.nan

    sharpe_comment = ""
    if not np.isnan(gsci_sr_now):
        gsci_dir = "improved" if gsci_sr_now > gsci_sr_6m else "deteriorated"
        bcom_dir = "improved" if bcom_sr_now > bcom_sr_6m else "deteriorated"
        sharpe_comment = (
            f"On a rolling one-year basis, the risk-adjusted picture has "
            f"{gsci_dir} for GSCI (Sharpe: {_fmt(gsci_sr_now, 2)}) and "
            f"{bcom_dir} for BCOM ({_fmt(bcom_sr_now, 2)}) over the past six months. "
            f"A Sharpe below zero means the index has not compensated holders for "
            f"the volatility absorbed."
        ) if gsci_dir != bcom_dir else (
            f"On a rolling one-year basis, risk-adjusted returns have "
            f"{gsci_dir} for both indices over the past six months "
            f"(GSCI: {_fmt(gsci_sr_now, 2)}, BCOM: {_fmt(bcom_sr_now, 2)})."
        )

    # Drawdown
    gsci_max_dd = r["drawdown"]["SPGSCI"].min()
    bcom_max_dd = r["drawdown"]["BCOM"].min()
    gsci_current_dd = r["drawdown"]["SPGSCI"].iloc[-1]
    bcom_current_dd = r["drawdown"]["BCOM"].iloc[-1]
    dd_context_gsci = _drawdown_context(gsci_max_dd)
    dd_context_bcom = _drawdown_context(bcom_max_dd)

    # Correlation
    ytd_corr = ytd["correlation"] if not ytd["empty"] else np.nan
    hist_corr = r["correlation"]
    if not np.isnan(ytd_corr):
        corr_quality, corr_vs_hist = _correlation_tone(ytd_corr, hist_corr)
        corr_comment = (
            f"The YTD return correlation between GSCI and BCOM stands at "
            f"{_fmt(ytd_corr, 3)} — {corr_quality}, and {corr_vs_hist} the "
            f"full-history figure of {_fmt(hist_corr, 3)}."
        )
    else:
        corr_comment = f"Full-history correlation between the two indices is {_fmt(hist_corr, 3)}."

    # Tail dependence
    tail = r["tail_dep_5"]
    if pd.notna(tail):
        tail_comment = (
            f"Lower-tail dependence — the probability that BCOM is in its worst "
            f"5% of days when GSCI is also in its worst 5% — sits at "
            f"{tail*100:.0f}% historically. "
            + (
                "This high co-crash rate means holding both indices together "
                "offers limited stress-period diversification."
                if tail > 0.60 else
                "This means the indices still tend to diverge during sell-offs, "
                "offering some diversification benefit across the two benchmarks."
                if tail < 0.40 else
                "The two indices move together in roughly half of stress episodes."
            )
        )
    else:
        tail_comment = ""

    # Seasonality for current month
    monthly_simple = r["monthly_simple"]
    month_gsci = monthly_simple["SPGSCI"][monthly_simple.index.month == current_month]
    month_bcom = monthly_simple["BCOM"][monthly_simple.index.month == current_month]
    avg_gsci = month_gsci.mean()
    avg_bcom = month_bcom.mean()
    win_gsci = (month_gsci > 0).mean() * 100
    win_bcom = (month_bcom > 0).mean() * 100
    n_years = len(month_gsci)
    month_name = _month_name(current_month)

    seasonal_gsci_tone = _sign(avg_gsci, "historically bullish", "historically bearish",
                               "historically mixed", threshold=0.3)
    seasonal_bcom_tone = _sign(avg_bcom, "historically bullish", "historically bearish",
                               "historically mixed", threshold=0.3)
    seasonal_comment = (
        f"Looking at the calendar, {month_name} has been {seasonal_gsci_tone} for GSCI "
        f"across {n_years} years of data: average return {_pct(avg_gsci, 2)}, "
        f"positive {win_gsci:.0f}% of the time. "
        f"BCOM's {month_name} record is {seasonal_bcom_tone}: "
        f"average {_pct(avg_bcom, 2)}, win rate {win_bcom:.0f}%."
    )

    # Best and worst calendar years for context
    ann = r["annual_simple"]
    gsci_best_yr = ann["SPGSCI"].idxmax().year
    gsci_worst_yr = ann["SPGSCI"].idxmin().year
    gsci_best_ret = ann["SPGSCI"].max()
    gsci_worst_ret = ann["SPGSCI"].min()

    # Full-history summary sentence
    full_cagr_gsci = r["cagr"]["SPGSCI"]
    full_cagr_bcom = r["cagr"]["BCOM"]
    full_vol_gsci = r["ann_vol"]["SPGSCI"]
    full_sharpe_gsci = r["sharpe"]["SPGSCI"]
    full_sharpe_bcom = r["sharpe"]["BCOM"]
    psr_gsci = r["psr"]["SPGSCI"]
    psr_bcom = r["psr"]["BCOM"]
    beta = r["beta_bcom_vs_gsci"]

    # Sharpe inference
    sr_gsci_se = r["sharpe_se"]["SPGSCI"]
    sr_bcom_se = r["sharpe_se"]["BCOM"]
    gsci_sr_lo = full_sharpe_gsci - 1.96 * sr_gsci_se if pd.notna(sr_gsci_se) else np.nan
    gsci_sr_hi = full_sharpe_gsci + 1.96 * sr_gsci_se if pd.notna(sr_gsci_se) else np.nan
    sharpe_inference = ""
    if not np.isnan(gsci_sr_lo):
        sharpe_inference = (
            f"Accounting for return autocorrelation via Newey-West standard errors, "
            f"the GSCI Sharpe 95% confidence interval is "
            f"[{_fmt(gsci_sr_lo, 3)}, {_fmt(gsci_sr_hi, 3)}]; "
            f"the Probabilistic Sharpe Ratio (Bailey & Lopez de Prado) gives "
            f"a {psr_gsci*100:.0f}% probability the true GSCI Sharpe is positive "
            f"and {psr_bcom*100:.0f}% for BCOM."
        )

    # ── assemble markdown ────────────────────────────────────
    lines = []

    # Header
    lines += [
        f"# Commodity Index Monitor — {today.strftime('%B %Y')}",
        f"",
        f"*S&P GSCI vs Bloomberg BCOM | Data through {data_through} | "
        f"Risk-free rate assumed: {rf_annual*100:.2f}%*",
        f"",
        "---",
        "",
    ]

    # Section 1: Scoreboard
    lines += [
        "## Year-to-Date Scoreboard",
        "",
        ytd_leader_sentence,
        "",
    ]

    if not ytd["empty"]:
        ytd_cagr_gsci = ytd["cagr"]["SPGSCI"]
        ytd_cagr_bcom = ytd["cagr"]["BCOM"]
        ytd_sharpe_gsci = ytd["sharpe"]["SPGSCI"]
        ytd_sharpe_bcom = ytd["sharpe"]["BCOM"]
        ytd_maxdd_gsci = ytd["drawdown"]["SPGSCI"].min()
        ytd_maxdd_bcom = ytd["drawdown"]["BCOM"].min()
        ytd_vol_gsci = ytd["ann_vol"]["SPGSCI"]
        ytd_vol_bcom = ytd["ann_vol"]["BCOM"]

        lines += [
            "| Metric | S&P GSCI | Bloomberg BCOM |",
            "|---|---|---|",
            f"| Total Return (YTD) | {_pct(ytd_gsci, 2)} | {_pct(ytd_bcom, 2)} |",
            f"| CAGR (ann., YTD) | {_pct(ytd_cagr_gsci, 2)} | {_pct(ytd_cagr_bcom, 2)} |",
            f"| Ann. Volatility | {_fmt(ytd_vol_gsci, 2)}% | {_fmt(ytd_vol_bcom, 2)}% |",
            f"| Sharpe Ratio | {_fmt(ytd_sharpe_gsci, 3)} | {_fmt(ytd_sharpe_bcom, 3)} |",
            f"| Max Drawdown | {_fmt(ytd_maxdd_gsci, 2)}% | {_fmt(ytd_maxdd_bcom, 2)}% |",
            "",
        ]

    # Section 2: Volatility & Risk Pulse
    lines += [
        "## Risk Pulse",
        "",
    ]
    if vol_comment:
        lines += [vol_comment, ""]
    if sharpe_comment:
        lines += [sharpe_comment, ""]

    lines += [
        f"Drawdown context: GSCI's all-time peak-to-trough drawdown is "
        f"{_fmt(gsci_max_dd, 1)}% ({dd_context_gsci}); BCOM's is {_fmt(bcom_max_dd, 1)}% "
        f"({dd_context_bcom}). Currently, GSCI sits {_fmt(gsci_current_dd, 1)}% and "
        f"BCOM {_fmt(bcom_current_dd, 1)}% below their respective all-time highs.",
        "",
    ]

    # Section 3: Correlation & Diversification
    lines += [
        "## Correlation & Diversification",
        "",
        corr_comment,
        "",
        f"The full-sample beta of BCOM regressed on GSCI is {_fmt(beta, 3)}, meaning "
        f"a {_pct(1.0, 0)} move in GSCI has historically corresponded to a "
        f"~{_pct(beta*100, 1)} move in BCOM.",
        "",
    ]
    if tail_comment:
        lines += [tail_comment, ""]

    # Section 4: Seasonality
    lines += [
        f"## Seasonality: {month_name}",
        "",
        seasonal_comment,
        "",
    ]

    # Section 5: Long-Run Context
    lines += [
        "## Long-Run Context",
        "",
        f"Across the full sample from {data_from}, GSCI has compounded at "
        f"{_fmt(full_cagr_gsci, 2)}% per year with {_fmt(full_vol_gsci, 2)}% annualised "
        f"volatility (Sharpe: {_fmt(full_sharpe_gsci, 3)}). BCOM's CAGR is "
        f"{_fmt(full_cagr_bcom, 2)}% (Sharpe: {_fmt(full_sharpe_bcom, 3)}).",
        "",
        f"GSCI's best calendar year was {gsci_best_yr} "
        f"({_pct(gsci_best_ret, 1)}); its worst was {gsci_worst_yr} "
        f"({_pct(gsci_worst_ret, 1)}).",
        "",
    ]

    # Section 6: Quant Corner
    lines += [
        "## Quant Corner",
        "",
        "Standard Sharpe ratios overstate statistical confidence when daily "
        "returns exhibit autocorrelation — a well-documented feature of commodity "
        "indices. The figures below use Newey-West HAC standard errors.",
        "",
        f"| | S&P GSCI | Bloomberg BCOM |",
        f"|---|---|---|",
        f"| Sharpe (ann.) | {_fmt(full_sharpe_gsci, 3)} | {_fmt(full_sharpe_bcom, 3)} |",
    ]
    if pd.notna(sr_gsci_se):
        bcom_sr_lo = full_sharpe_bcom - 1.96 * sr_bcom_se
        bcom_sr_hi = full_sharpe_bcom + 1.96 * sr_bcom_se
        lines += [
            f"| HAC SE | {_fmt(sr_gsci_se, 3)} | {_fmt(sr_bcom_se, 3)} |",
            f"| 95% CI | [{_fmt(gsci_sr_lo, 3)}, {_fmt(gsci_sr_hi, 3)}] | "
            f"[{_fmt(bcom_sr_lo, 3)}, {_fmt(bcom_sr_hi, 3)}] |",
            f"| P(true SR > 0) | {psr_gsci*100:.0f}% | {psr_bcom*100:.0f}% |",
        ]
    lines += ["", sharpe_inference, ""]

    # Tail risk
    var_gsci = r["var_95"]["SPGSCI"] * 100
    cvar_gsci = r["cvar_95"]["SPGSCI"] * 100
    var_bcom = r["var_95"]["BCOM"] * 100
    cvar_bcom = r["cvar_95"]["BCOM"] * 100
    lines += [
        f"**Historical tail risk (daily, full sample):** GSCI VaR 95%: "
        f"{_fmt(var_gsci, 2)}%, CVaR 95%: {_fmt(cvar_gsci, 2)}%. "
        f"BCOM VaR 95%: {_fmt(var_bcom, 2)}%, CVaR 95%: {_fmt(cvar_bcom, 2)}%.",
        "",
    ]

    # Footer
    lines += [
        "---",
        "",
        "*This commentary is generated programmatically from index price data. "
        "It is for informational purposes only and does not constitute investment advice. "
        "All returns are price-return only (no roll yield, no dividends). "
        "CAGR is computed geometrically from log returns.*",
        "",
    ]

    return "\n".join(lines)


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Substack commodity commentary")
    parser.add_argument("--out", metavar="FILE", default=None,
                        help="Write Markdown to FILE instead of stdout")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF_ANNUAL * 100,
                        help="Risk-free rate in %% (default: %(default)s)")
    parser.add_argument("--event-date", default="2022-02-24",
                        help="Event-impact start date YYYY-MM-DD (default: %(default)s)")
    args = parser.parse_args()

    event_date = datetime.strptime(args.event_date, "%Y-%m-%d").date()
    commentary = generate_commentary(rf_annual=args.rf / 100.0, event_date=event_date)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(commentary)
        print(f"Saved to {args.out}", file=sys.stderr)
    else:
        print(commentary)


if __name__ == "__main__":
    main()
