"""GSCI vs BCOM Commodity Index Analytics Dashboard.

Streamlit application benchmarking the S&P GSCI against the Bloomberg
Commodity Index (BCOM). Provides full-history, year-to-date, seasonality,
event-impact, and side-by-side comparison views.

Returns are computed in log space; performance metrics are reported as CAGR
(geometric annualised return) so that compounding is handled correctly.

Layers
------
- Data         :: load_data, validate inputs, yfinance fetch
- Analytics    :: _metric_bundle (shared core), helper risk-metric functions
- Statistical  :: Newey-West Sharpe SE, Probabilistic Sharpe Ratio,
                  rolling beta, within-year drawdown, DD duration,
                  lower-tail dependence
- Plotting     :: cached matplotlib figure builders (all charts unified)
- UI           :: Streamlit tabs / sidebar controls
"""

import os
from datetime import date, datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Patch
from scipy import stats


# ============================================================
# CONSTANTS
# ============================================================
TRADING_DAYS = 252
ROLL_VOL_LONG = 252
ROLL_VOL_SHORT = 21
ROLL_CORR_LONG = 90
ROLL_CORR_SHORT = 30
ROLL_BETA_WINDOW = 252

COLOR_GSCI = "#1f77b4"
COLOR_BCOM = "#ff7f0e"
COLOR_BG = "#fafbfc"
GRID_COLOR = "#e0e0e0"

GSCI_FILE = "SPGSCI_historical.csv"
BCOM_FILE = "BCOM_historical.csv"

DEFAULT_WAR_DATE = date(2022, 2, 24)
DEFAULT_RF_ANNUAL = 0.045

# Unified font stack used everywhere (UI + matplotlib).
FONT_STACK = ["Source Sans Pro", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


# ============================================================
# GLOBAL FONT / STYLE CONFIG (matplotlib)
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.color": GRID_COLOR,
    "axes.facecolor": COLOR_BG,
    "figure.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.facecolor": "white",
})


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GSCI vs BCOM Analytics", layout="wide")

st.markdown(
    f"""
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stTabs, .stButton, .stSelectbox, .stNumberInput, .stDateInput,
        .stCaption, .stHeading, p, span, div, label {{
            font-family: {", ".join(f"'{f}'" for f in FONT_STACK)} !important;
        }}
        .block-container {{padding-top: 1.5rem; padding-bottom: 1.5rem;}}
        h1 {{font-weight: 700; letter-spacing: -0.5px; font-size: 1.8rem;}}
        h2 {{font-weight: 600; font-size: 1.25rem; margin-top: 1.2rem;
             border-bottom: 1.5px solid #e0e0e0; padding-bottom: 0.3rem;}}
        h3 {{font-weight: 500; font-size: 1rem; color: #444;}}
        [data-testid="stMetricDelta"] {{font-size: 0.8rem;}}

        /* Hide Streamlit dataframe keyboard shortcut overlay */
        div[data-testid="stDataFrame"] div[class*="glideDataEditor"] > div:last-child,
        div[data-testid="stDataFrame"] [class*="keyhint"],
        div[data-testid="stDataFrame"] svg[title*="keyboard"],
        div[data-testid="stDataFrame"] svg[title*="arrow"] {{
            display: none !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Commodity Index Benchmarking")
st.caption(
    f"S&P GSCI vs Bloomberg Commodity Index (BCOM)  |  "
    f"Data through {datetime.now().strftime('%d %b %Y')}"
)


# ============================================================
# YAHOO FINANCE FETCH HELPERS
# ============================================================
def _normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Ensure yfinance output has MultiIndex columns (Price, Ticker)."""
    if df.empty:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df


def _extract_close(df: pd.DataFrame, label: str) -> pd.Series:
    """Robustly extract the Close price from flat or MultiIndex columns."""
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df["Close"].iloc[:, 0]
        elif "Close" in df.columns.get_level_values(1):
            return df.xs("Close", axis=1, level=1).iloc[:, 0]
    else:
        if "Close" in df.columns:
            return df["Close"]
    raise ValueError(f"{label}: cannot locate 'Close' column")


def fetch_latest_data() -> bool:
    """Pull latest EOD data from Yahoo Finance and overwrite CSVs."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    ok = True

    try:
        with st.spinner("Downloading S&P GSCI..."):
            sp = yf.download("^SPGSCI", start="1984-01-01", end=today_str, progress=False)
            if sp.empty:
                st.error("Yahoo Finance returned empty data for ^SPGSCI.")
                ok = False
            else:
                _normalize_yf(sp, "^SPGSCI").to_csv(GSCI_FILE)

        with st.spinner("Downloading BCOM..."):
            # NOTE: Replace "^BCOM" with your actual Yahoo Finance ticker if different
            bc = yf.download("^BCOM", start="1991-01-01", end=today_str, progress=False)
            if bc.empty:
                st.error("Yahoo Finance returned empty data for BCOM.")
                ok = False
            else:
                _normalize_yf(bc, "^BCOM").to_csv(BCOM_FILE)

        if ok:
            st.success(f"Data refreshed through {today_str}")
    except Exception as exc:
        st.error(f"Fetch failed: {exc}")
        ok = False

    return ok


def _data_age(path: str) -> str:
    """Human-readable age of a file."""
    if not os.path.exists(path):
        return "missing"
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    delta = datetime.now() - mtime
    if delta.days > 0:
        return f"{delta.days}d ago"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600}h ago"
    else:
        return f"{delta.seconds // 60}m ago"


# ============================================================
# DATA LAYER
# ============================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and align SPGSCI and BCOM close-price series."""
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
# ANALYTICS LAYER — helpers
# ============================================================
def _safe_div(num, den):
    """Divide returning NaN where denominator is zero or NaN."""
    if isinstance(den, pd.Series):
        return num / den.replace(0, np.nan)
    return np.nan if (den == 0 or np.isnan(den)) else num / den


def rf_to_daily_log(rf_annual: float) -> float:
    """Convert an annual simple risk-free rate to its daily log equivalent."""
    return float(np.log(1.0 + rf_annual) / TRADING_DAYS)


def cagr_from_log(log_returns, periods: int = TRADING_DAYS):
    """Compound annualised return (CAGR) from log returns."""
    return np.exp(log_returns.mean() * periods) - 1.0


def ann_vol_from_log(log_returns, periods: int = TRADING_DAYS):
    """Annualised volatility from log returns."""
    return log_returns.std() * np.sqrt(periods)


def sharpe_from_log(log_returns, rf_annual: float, periods: int = TRADING_DAYS):
    """Annualised Sharpe ratio using log returns."""
    rf_d = rf_to_daily_log(rf_annual)
    std = log_returns.std()
    return _safe_div((log_returns.mean() - rf_d) * np.sqrt(periods), std)


def sortino_from_log(log_returns, rf_annual: float, periods: int = TRADING_DAYS):
    """Sortino ratio with MAR = rf_annual."""
    rf_d = rf_to_daily_log(rf_annual)
    shortfall = (log_returns - rf_d).clip(upper=0.0)
    dd_dev = np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods)
    excess = cagr_from_log(log_returns, periods) - rf_annual
    return _safe_div(excess, dd_dev)


def drawdown_from_log(log_returns):
    """Percentage drawdown series (negative numbers, in %) from log returns."""
    cum = np.exp(log_returns.cumsum())
    return (cum / cum.cummax() - 1.0) * 100.0


def calmar_from_log(log_returns, drawdown, periods: int = TRADING_DAYS):
    """Calmar ratio = CAGR / |Max Drawdown|."""
    cagr = cagr_from_log(log_returns, periods)
    max_dd_abs = drawdown.min().abs() / 100.0
    return _safe_div(cagr, max_dd_abs)


def historical_var_cvar(log_returns, alpha: float):
    """Historical VaR and CVaR at confidence level (1 - alpha)."""
    var = log_returns.quantile(alpha)
    cvar = pd.Series(
        {c: log_returns[c][log_returns[c] <= var[c]].mean() for c in log_returns.columns}
    )
    return var, cvar


# ============================================================
# ANALYTICS LAYER — statistical inference
# ============================================================
def newey_west_sharpe_se(log_returns: pd.Series, lags: Optional[int] = None,
                        periods: int = TRADING_DAYS) -> float:
    """HAC (Newey-West) standard error of the annualised Sharpe ratio.

    Daily commodity returns exhibit serial correlation; the IID Sharpe SE
    of Lo (2002) understates uncertainty. The HAC-adjusted SE multiplies
    the IID SE by sqrt(1 + 2 * sum_{k=1..q} (1 - k/(q+1)) * rho_k), the
    Bartlett kernel adjustment, where rho_k is the lag-k autocorrelation
    of returns.
    """
    r = log_returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan
    if lags is None:
        # Newey & West (1994) automatic-bandwidth rule of thumb.
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))

    sr_ann = sharpe_from_log(r, 0.0, periods)
    if pd.isna(sr_ann) or np.isinf(sr_ann):
        return np.nan

    # Lo (2002) IID SE for the *daily* Sharpe; scaled to annual at the end.
    sr_daily = sr_ann / np.sqrt(periods)
    iid_var = (1.0 + 0.5 * sr_daily ** 2) / n

    # Bartlett kernel autocorrelation correction.
    centered = r - r.mean()
    var0 = (centered ** 2).mean()
    if var0 == 0:
        return np.nan
    adj = 1.0
    for k in range(1, lags + 1):
        rho_k = (centered.iloc[k:].values * centered.iloc[:-k].values).mean() / var0
        adj += 2.0 * (1.0 - k / (lags + 1.0)) * rho_k
    adj = max(adj, 1e-6)  # guard against negative inflation factors

    se_daily = np.sqrt(iid_var * adj)
    return float(se_daily * np.sqrt(periods))


def probabilistic_sharpe_ratio(log_returns: pd.Series, sr_benchmark: float = 0.0,
                               periods: int = TRADING_DAYS) -> float:
    """Probability that the true (annualised) Sharpe exceeds sr_benchmark.

    Bailey & Lopez de Prado (2012). Adjusts the normal-distribution Sharpe
    test for finite-sample skew and kurtosis of returns:

        PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1)
                       / sqrt(1 - g3 * SR_hat + (g4 - 1) / 4 * SR_hat^2) )

    where SR is non-annualised, g3 = skew, g4 = (non-excess) kurtosis.
    """
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
    g4 = float(r.kurtosis()) + 3.0  # pandas returns excess kurtosis
    denom = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0:
        return np.nan
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def rolling_beta(log_ret: pd.DataFrame, dep: str, indep: str,
                 window: int = ROLL_BETA_WINDOW) -> pd.Series:
    """Rolling OLS beta of `dep` returns on `indep` returns."""
    cov = log_ret[dep].rolling(window).cov(log_ret[indep])
    var = log_ret[indep].rolling(window).var()
    return cov / var.replace(0, np.nan)


def within_year_max_dd(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Max drawdown computed *within* each calendar year (peaks reset Jan 1).

    The original ``drawdown.resample('YE').min()`` reported the deepest
    all-time-peak drawdown reached during each year — which during a
    multi-year bear market is just the year-end depth, not the year's
    own peak-to-trough. This recomputes each year from a fresh peak.
    """
    rows = {}
    for year, group in log_ret.groupby(log_ret.index.year):
        if len(group) < 2:
            continue
        cum = np.exp(group.cumsum())
        dd = (cum / cum.cummax() - 1.0) * 100.0
        rows[year] = dd.min()
    return pd.DataFrame(rows).T.sort_index()


def max_drawdown_duration(drawdown: pd.DataFrame) -> pd.Series:
    """Longest consecutive run (in trading days) below the prior peak."""
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


def lower_tail_dependence(log_ret: pd.DataFrame, alpha: float = 0.05) -> float:
    """Empirical lower-tail dependence: P(B in bottom alpha | A in bottom alpha).

    Pearson correlation tells you about co-movement on average; tail
    dependence tells you whether they crash together. Far more relevant
    for stress-period diversification questions.
    """
    if len(log_ret) < int(1 / alpha):
        return np.nan
    q_a = log_ret["SPGSCI"].quantile(alpha)
    q_b = log_ret["BCOM"].quantile(alpha)
    a_tail = log_ret["SPGSCI"] <= q_a
    b_tail = log_ret["BCOM"] <= q_b
    if a_tail.sum() == 0:
        return np.nan
    return float((a_tail & b_tail).sum() / a_tail.sum())


# ============================================================
# ANALYTICS LAYER — bundle (shared core)
# ============================================================
def _metric_bundle(log_ret: pd.DataFrame, rf_annual: float) -> Dict[str, object]:
    """Core metric bundle shared by full-history and sliced-period flows.

    Reduces ~70% duplication between compute_metrics and slice_period.
    Returns only metrics that are well-defined for both contexts.
    """
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

    # Statistical inference on Sharpe.
    sharpe_se = pd.Series({
        c: newey_west_sharpe_se(log_ret[c]) for c in log_ret.columns
    })
    psr = pd.Series({
        c: probabilistic_sharpe_ratio(log_ret[c], 0.0) for c in log_ret.columns
    })

    # Drawdown duration (in trading days).
    dd_duration = max_drawdown_duration(drawdown)

    # Lower-tail dependence.
    tail_dep_5 = lower_tail_dependence(log_ret, 0.05)

    return {
        "log_ret": log_ret,
        "cum_wealth": cum_wealth,
        "rebased_100": rebased_100,
        "drawdown": drawdown,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se,
        "psr": psr,
        "sortino": sortino,
        "calmar": calmar,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "correlation": correlation,
        "dd_duration": dd_duration,
        "tail_dep_5": tail_dep_5,
    }


@st.cache_data
def compute_metrics(df: pd.DataFrame, rf_annual: float) -> Dict[str, object]:
    """Full-history metric bundle, plus rolling and beta series."""
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
    roll_beta_bcom = rolling_beta(log_ret, "BCOM", "SPGSCI", ROLL_BETA_WINDOW)

    annual_log = log_ret.resample("YE").sum()
    annual_simple = (np.exp(annual_log) - 1.0) * 100.0
    annual_vol_yr = log_ret.resample("YE").std() * np.sqrt(TRADING_DAYS) * 100.0
    annual_dd_within = within_year_max_dd(log_ret)  # corrected within-year DD

    monthly_log = log_ret.resample("ME").sum()
    monthly_simple = (np.exp(monthly_log) - 1.0) * 100.0

    bundle.update({
        "beta_bcom_vs_gsci": beta_bcom_vs_gsci,
        "vol_252": vol_252,
        "roll_sharpe": roll_sharpe,
        "roll_corr_long": roll_corr_long,
        "roll_beta_bcom": roll_beta_bcom,
        "annual_simple": annual_simple,
        "annual_vol_yr": annual_vol_yr,
        "annual_dd_within": annual_dd_within,
        "monthly_simple": monthly_simple,
    })
    return bundle


@st.cache_data
def slice_period(df: pd.DataFrame, start: pd.Timestamp, rf_annual: float) -> Dict:
    """Metric bundle for a price slice starting at ``start``."""
    sub = df[df.index >= pd.Timestamp(start)].copy()
    if len(sub) < 2:
        return {"empty": True, "df": sub}

    log_ret = np.log(sub / sub.shift(1)).dropna()
    bundle = _metric_bundle(log_ret, rf_annual)

    cum_pct = (sub / sub.iloc[0] - 1.0) * 100.0
    vol_roll_short = log_ret.rolling(ROLL_VOL_SHORT).std() * np.sqrt(TRADING_DAYS)
    roll_corr_short = log_ret["SPGSCI"].rolling(ROLL_CORR_SHORT).corr(log_ret["BCOM"])

    bundle.update({
        "empty": False,
        "df": sub,
        "cum_pct": cum_pct,
        "vol_roll_short": vol_roll_short,
        "roll_corr_short": roll_corr_short,
    })
    return bundle


def tail_risk_table(var_95, cvar_95, var_99, cvar_99) -> pd.DataFrame:
    """Tidy tail-risk table (formatted as percentages)."""
    out = pd.DataFrame(
        {
            "VaR 95% (daily)": var_95 * 100,
            "CVaR 95% (daily)": cvar_95 * 100,
            "VaR 99% (daily)": var_99 * 100,
            "CVaR 99% (daily)": cvar_99 * 100,
        }
    )
    return out.map(lambda v: f"{v:.2f}%")  # applymap is deprecated


def sharpe_inference_table(sharpe, sharpe_se, psr) -> pd.DataFrame:
    """Sharpe + Newey-West 95% CI + P(true SR > 0)."""
    rows = []
    for col in sharpe.index:
        sr = sharpe[col]
        se = sharpe_se[col]
        if pd.notna(se):
            lo, hi = sr - 1.96 * se, sr + 1.96 * se
            ci = f"[{lo:.3f}, {hi:.3f}]"
        else:
            ci = "n/a"
        rows.append({
            "Index": "S&P GSCI" if col == "SPGSCI" else "Bloomberg BCOM",
            "Sharpe (ann.)": f"{sr:.3f}",
            "HAC SE": f"{se:.3f}" if pd.notna(se) else "n/a",
            "95% CI (Newey-West)": ci,
            "PSR (P[SR>0])": f"{psr[col]*100:.1f}%" if pd.notna(psr[col]) else "n/a",
        })
    return pd.DataFrame(rows)


# ============================================================
# PLOTTING LAYER — base helpers
# ============================================================
def style_ax(ax, title: Optional[str] = None) -> None:
    """Light per-axis polish on top of the global rcParams."""
    if title:
        ax.set_title(title, pad=10)


def _line_panel(series_or_df, ylabel: str, title: str,
                figsize=(14, 4), color_map=None, hline=None):
    """Generic single-axis line chart used to replace st.line_chart calls."""
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(series_or_df, pd.Series):
        ax.plot(series_or_df.index, series_or_df.values,
                linewidth=1.2, color=COLOR_GSCI)
    else:
        for col in series_or_df.columns:
            color = (color_map or {}).get(col)
            label = "S&P GSCI" if col == "SPGSCI" else (
                "Bloomberg BCOM" if col == "BCOM" else col)
            ax.plot(series_or_df.index, series_or_df[col],
                    linewidth=1.2, label=label, color=color)
        ax.legend(loc="best", frameon=True)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    style_ax(ax, title)
    fig.tight_layout()
    return fig


# ============================================================
# PLOTTING LAYER — figures
# ============================================================
@st.cache_data
def fig_index_performance(rebased_100: pd.DataFrame):
    """Index Performance (Rebased to 100) — linear scale."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(rebased_100.index, 100, rebased_100["SPGSCI"],
                    alpha=0.12, color=COLOR_GSCI)
    ax.fill_between(rebased_100.index, 100, rebased_100["BCOM"],
                    alpha=0.12, color=COLOR_BCOM)
    ax.plot(rebased_100.index, rebased_100["SPGSCI"], color=COLOR_GSCI,
            linewidth=1.3, label="S&P GSCI")
    ax.plot(rebased_100.index, rebased_100["BCOM"], color=COLOR_BCOM,
            linewidth=1.3, label="Bloomberg BCOM")
    ax.axhline(100, color="black", linewidth=0.5)
    ax.set_ylabel("Index Level (rebased to 100)")
    style_ax(ax, "Index Performance (Rebased to 100)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    return fig


@st.cache_data
def fig_drawdown(drawdown: pd.DataFrame, title: str = "Drawdown from Peak (%)"):
    """Underwater (drawdown) chart."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.fill_between(drawdown.index, 0, drawdown["SPGSCI"], alpha=0.25,
                    color=COLOR_GSCI, label="S&P GSCI")
    ax.fill_between(drawdown.index, 0, drawdown["BCOM"], alpha=0.25,
                    color=COLOR_BCOM, label="Bloomberg BCOM")
    ax.plot(drawdown.index, drawdown["SPGSCI"], color=COLOR_GSCI, linewidth=0.7)
    ax.plot(drawdown.index, drawdown["BCOM"], color=COLOR_BCOM, linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    style_ax(ax, title)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


@st.cache_data
def fig_distribution(log_ret, var_95, cvar_95, col, color, title):
    """Histogram + KDE for a single index, with VaR/CVaR markers."""
    fig, ax = plt.subplots(figsize=(8, 4))
    series_pct = log_ret[col] * 100
    ax.hist(series_pct, bins=80, density=True, alpha=0.55, color=color,
            edgecolor="white", linewidth=0.3)
    kde_x = np.linspace(series_pct.min(), series_pct.max(), 200)
    kde = stats.gaussian_kde(series_pct)
    ax.plot(kde_x, kde(kde_x), color=color, linewidth=2)
    ax.axvline(var_95[col] * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"VaR 95%: {var_95[col]*100:.2f}%")
    ax.axvline(cvar_95[col] * 100, color="darkred", linestyle=":", linewidth=1.5,
               label=f"CVaR 95%: {cvar_95[col]*100:.2f}%")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    style_ax(ax, title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


@st.cache_data
def fig_heatmaps(monthly_simple: pd.DataFrame):
    """Year x Month heatmap of monthly returns for both indices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    months_short = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for idx, col in enumerate(["SPGSCI", "BCOM"]):
        pivot = (
            monthly_simple[col]
            .groupby([monthly_simple.index.year, monthly_simple.index.month])
            .first()
            .unstack()
        )
        im = axes[idx].imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                              vmin=-15, vmax=15)
        axes[idx].set_xticks(range(12))
        axes[idx].set_xticklabels(months_short)
        axes[idx].set_yticks(range(0, len(pivot.index), 2))
        axes[idx].set_yticklabels(pivot.index[::2])
        axes[idx].set_title(col, fontweight="bold", fontsize=12)
        axes[idx].set_facecolor("#f5f5f5")
        axes[idx].grid(False)
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


@st.cache_data
def fig_monthly_profile(monthly_simple: pd.DataFrame):
    """Average monthly return profile (with std-dev error bars)."""
    monthly_avg = monthly_simple.groupby(monthly_simple.index.month).mean()
    monthly_std = monthly_simple.groupby(monthly_simple.index.month).std()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(1, 13)
    width = 0.35
    ax.bar(x - width / 2, monthly_avg["SPGSCI"], width,
           yerr=monthly_std["SPGSCI"], label="S&P GSCI", capsize=3,
           color=COLOR_GSCI, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, monthly_avg["BCOM"], width,
           yerr=monthly_std["BCOM"], label="Bloomberg BCOM", capsize=3,
           color=COLOR_BCOM, alpha=0.85, edgecolor="white")
    ax.plot(x, monthly_avg["SPGSCI"], color=COLOR_GSCI, marker="o",
            markersize=4, linewidth=1)
    ax.plot(x, monthly_avg["BCOM"], color=COLOR_BCOM, marker="o",
            markersize=4, linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Average Monthly Return (%)")
    style_ax(ax)
    ax.legend()
    fig.tight_layout()
    return fig


@st.cache_data
def fig_win_rate(monthly_simple: pd.DataFrame):
    """Win rate by calendar month — replaces st.bar_chart for font consistency."""
    win_spgsci = (monthly_simple["SPGSCI"]
                  .groupby(monthly_simple.index.month)
                  .apply(lambda x: (x > 0).mean() * 100))
    win_bcom = (monthly_simple["BCOM"]
                .groupby(monthly_simple.index.month)
                .apply(lambda x: (x > 0).mean() * 100))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, 13)
    width = 0.4
    ax.bar(x - width / 2, win_spgsci, width, label="S&P GSCI",
           color=COLOR_GSCI, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, win_bcom, width, label="Bloomberg BCOM",
           color=COLOR_BCOM, alpha=0.85, edgecolor="white")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    style_ax(ax, "Monthly Win Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, win_spgsci, win_bcom


@st.cache_data
def fig_boxplots(monthly_simple: pd.DataFrame):
    """Side-by-side boxplots of monthly returns by calendar month."""
    fig, ax = plt.subplots(figsize=(14, 5))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    spgsci_by_month = [
        monthly_simple["SPGSCI"][monthly_simple.index.month == m].values
        for m in range(1, 13)
    ]
    bcom_by_month = [
        monthly_simple["BCOM"][monthly_simple.index.month == m].values
        for m in range(1, 13)
    ]
    ax.boxplot(spgsci_by_month, positions=np.arange(1, 13) - 0.22, widths=0.35,
               patch_artist=True,
               boxprops=dict(facecolor=COLOR_GSCI, alpha=0.7, linewidth=1.5),
               medianprops=dict(color="white", linewidth=2))
    ax.boxplot(bcom_by_month, positions=np.arange(1, 13) + 0.22, widths=0.35,
               patch_artist=True,
               boxprops=dict(facecolor=COLOR_BCOM, alpha=0.7, linewidth=1.5),
               medianprops=dict(color="white", linewidth=2))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Monthly Return (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
    ax.legend(handles=[
        Patch(facecolor=COLOR_GSCI, alpha=0.7, label="S&P GSCI"),
        Patch(facecolor=COLOR_BCOM, alpha=0.7, label="Bloomberg BCOM"),
    ], loc="upper right")
    fig.tight_layout()
    return fig


@st.cache_data
def fig_scatter(log_ret: pd.DataFrame, corr: float, title_suffix: str = ""):
    """Scatter of daily SPGSCI vs BCOM returns colored by time."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(log_ret["SPGSCI"] * 100, log_ret["BCOM"] * 100,
                    alpha=0.6, c=range(len(log_ret)), cmap="viridis",
                    edgecolors="black", linewidth=0.3, s=50)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.4, label="1:1 Line")
    ax.set_xlabel("S&P GSCI Daily Return (%)")
    ax.set_ylabel("Bloomberg BCOM Daily Return (%)")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time (early → late)", fontsize=8)
    style_ax(ax, f"Correlation = {corr:.3f}{title_suffix}")
    ax.legend()
    fig.tight_layout()
    return fig


@st.cache_data
def fig_rolling_beta(roll_beta: pd.Series, full_beta: float):
    """Rolling 1Y beta of BCOM on SPGSCI, with full-sample reference line."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(roll_beta.index, roll_beta.values, linewidth=1.2,
            color="#2ca02c", label="Rolling 1Y β")
    ax.axhline(full_beta, color="black", linestyle="--", linewidth=1,
               label=f"Full-sample β = {full_beta:.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("β (BCOM regressed on SPGSCI)")
    style_ax(ax, "Rolling 1-Year Beta — BCOM vs S&P GSCI")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@st.cache_data
def fig_calendar_returns(annual_simple: pd.DataFrame):
    """Calendar-year returns as grouped bars (replaces st.bar_chart)."""
    yrs = annual_simple.index.year
    fig, ax = plt.subplots(figsize=(14, 4.5))
    x = np.arange(len(yrs))
    width = 0.4
    ax.bar(x - width / 2, annual_simple["SPGSCI"], width, label="S&P GSCI",
           color=COLOR_GSCI, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, annual_simple["BCOM"], width, label="Bloomberg BCOM",
           color=COLOR_BCOM, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(yrs, rotation=45, ha="right")
    ax.set_ylabel("Annual Return (%)")
    style_ax(ax, "Calendar Year Returns")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


@st.cache_data
def fig_comparison_bars(full_m: Dict, ytd_m: Dict):
    """Per-index comparison: full-history vs YTD on key risk-return metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["CAGR (%)", "Ann Vol (%)", "Sharpe", "Sortino", "Max DD (%)"]
    y_pos = np.arange(len(metrics))

    for ax, col, color, name in zip(
        axes,
        ["SPGSCI", "BCOM"],
        [COLOR_GSCI, COLOR_BCOM],
        ["S&P GSCI", "Bloomberg BCOM"],
    ):
        full_vals = [full_m[m][col] for m in metrics]
        ytd_vals = [ytd_m[m][col] for m in metrics]
        width = 0.35
        ax.barh(y_pos - width / 2, full_vals, width, label="Full History",
                color=color, alpha=0.6, edgecolor="white")
        ax.barh(y_pos + width / 2, ytd_vals, width, label="YTD",
                color=color, alpha=0.9, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(name)
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Data Source")
c1, c2 = st.sidebar.columns(2)
c1.caption(f"SPGSCI: {_data_age(GSCI_FILE)}")
c2.caption(f"BCOM: {_data_age(BCOM_FILE)}")

if st.sidebar.button("🔄 Refresh from Yahoo Finance", use_container_width=True):
    if fetch_latest_data():
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Inputs")
rf_annual = st.sidebar.number_input(
    "Risk-free rate (annual, %)",
    min_value=0.0, max_value=20.0,
    value=DEFAULT_RF_ANNUAL * 100, step=0.25,
    help="Used in Sharpe and Sortino. Converted to a daily log equivalent.",
) / 100.0


# ============================================================
# LOAD + COMPUTE
# ============================================================
# Auto-fetch if CSVs are missing (first run / deployment)
if not all(os.path.exists(f) for f in (GSCI_FILE, BCOM_FILE)):
    st.warning("CSV files not found. Fetching from Yahoo Finance...")
    if not fetch_latest_data():
        st.error("Unable to fetch data. Please check tickers / internet connection.")
        st.stop()
    st.cache_data.clear()
    st.rerun()

try:
    df = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

r = compute_metrics(df, rf_annual)

current_year = datetime.now().year
ytd_start = pd.Timestamp(date(current_year, 1, 1))
ytd = slice_period(df, ytd_start, rf_annual)


# ============================================================
# WAR-IMPACT INPUT (sidebar)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Event Impact")
data_min = df.index.min().date()
data_max = df.index.max().date()
war_default = DEFAULT_WAR_DATE if data_min <= DEFAULT_WAR_DATE <= data_max else data_min
war_date = st.sidebar.date_input(
    "Event date",
    value=war_default,
    min_value=data_min,
    max_value=data_max,
    help="Used in the 'Event Impact' tab. Default: Russia-Ukraine war start (24 Feb 2022).",
)
war_period = slice_period(df, pd.Timestamp(war_date), rf_annual)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Full History",
    f"YTD {current_year}",
    "Seasonality",
    "Event Impact",
    "Side-by-Side Comparison",
])


# ============================================================
# TAB 1 :: FULL HISTORY
# ============================================================
with tab1:
    st.header("Performance Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P GSCI CAGR", f"{r['cagr']['SPGSCI']:.2f}%")
    c2.metric("Bloomberg BCOM CAGR", f"{r['cagr']['BCOM']:.2f}%")
    c3.metric("S&P GSCI Ann. Volatility", f"{r['ann_vol']['SPGSCI']:.2f}%")
    c4.metric("Bloomberg BCOM Ann. Volatility", f"{r['ann_vol']['BCOM']:.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P GSCI Sharpe", f"{r['sharpe']['SPGSCI']:.3f}")
    c2.metric("Bloomberg BCOM Sharpe", f"{r['sharpe']['BCOM']:.3f}")
    c3.metric("S&P GSCI Max Drawdown", f"{r['drawdown']['SPGSCI'].min():.2f}%")
    c4.metric("Bloomberg BCOM Max Drawdown", f"{r['drawdown']['BCOM'].min():.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P GSCI DD Duration", f"{r['dd_duration']['SPGSCI']} days",
              help="Longest consecutive trading-day run below the prior peak.")
    c2.metric("Bloomberg BCOM DD Duration", f"{r['dd_duration']['BCOM']} days")
    c3.metric("Lower-Tail Dependence (5%)",
              f"{r['tail_dep_5']*100:.1f}%" if pd.notna(r['tail_dep_5']) else "n/a",
              help="P(BCOM in worst 5% | SPGSCI in worst 5%). Tells you whether they crash together — Pearson correlation does not.")
    c4.metric("Beta (BCOM vs SPGSCI)", f"{r['beta_bcom_vs_gsci']:.3f}")

    st.divider()

    st.subheader("Sharpe Ratio with Statistical Inference")
    st.caption(
        "Newey-West HAC standard errors account for autocorrelation in daily "
        "returns — naive SE understates uncertainty. PSR = probability that "
        "the *true* Sharpe is positive, adjusted for skew and excess kurtosis "
        "(Bailey & Lopez de Prado, 2012)."
    )
    st.dataframe(
        sharpe_inference_table(r["sharpe"], r["sharpe_se"], r["psr"]),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Index Performance (Rebased to 100)")
    st.caption(
        "Both series rebased to 100 on the first available date. Linear y-axis "
        "makes absolute index-level comparison and percentage outperformance "
        "directly readable."
    )
    st.pyplot(fig_index_performance(r["rebased_100"]))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rolling 1-Year Volatility")
        st.pyplot(_line_panel(
            r["vol_252"] * 100, "Annualised Volatility (%)",
            "Rolling 1-Year Volatility",
            color_map={"SPGSCI": COLOR_GSCI, "BCOM": COLOR_BCOM},
        ))
    with c2:
        st.subheader("Rolling 1-Year Sharpe Ratio")
        st.pyplot(_line_panel(
            r["roll_sharpe"], "Sharpe Ratio",
            "Rolling 1-Year Sharpe", hline=0,
            color_map={"SPGSCI": COLOR_GSCI, "BCOM": COLOR_BCOM},
        ))

    st.subheader("Rolling 90-Day Return Correlation")
    st.pyplot(_line_panel(
        r["roll_corr_long"], "Correlation",
        "Rolling 90-Day Correlation (SPGSCI vs BCOM)",
    ))

    st.subheader("Rolling 1-Year Beta — BCOM regressed on S&P GSCI")
    st.caption(
        "Full-sample beta averages over very different regimes (2014–16 oil "
        "collapse, COVID, 2022 energy spike). Rolling beta surfaces those "
        "regime breaks rather than smoothing them away."
    )
    st.pyplot(fig_rolling_beta(r["roll_beta_bcom"], r["beta_bcom_vs_gsci"]))

    st.subheader("Drawdown from Peak (%)")
    st.pyplot(fig_drawdown(r["drawdown"]))

    st.subheader("Calendar Year Returns")
    st.pyplot(fig_calendar_returns(r["annual_simple"]))

    st.subheader("Daily Return Distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_distribution(
            r["log_ret"], r["var_95"], r["cvar_95"],
            "SPGSCI", COLOR_GSCI, "S&P GSCI",
        ))
    with c2:
        st.pyplot(fig_distribution(
            r["log_ret"], r["var_95"], r["cvar_95"],
            "BCOM", COLOR_BCOM, "Bloomberg BCOM",
        ))

    st.subheader("Tail Risk (Historical VaR / CVaR)")
    st.caption("Daily losses at the 5% and 1% tails. CVaR = expected loss "
               "conditional on breaching the VaR threshold.")
    st.dataframe(
        tail_risk_table(r["var_95"], r["cvar_95"], r["var_99"], r["cvar_99"]),
        use_container_width=True,
    )

    with st.expander("View Annual Statistics Table"):
        # Note: annual_dd_within is now a true within-year max DD (peak resets
        # each Jan 1), not the year-end depth of an ongoing multi-year drawdown.
        annual_table = pd.DataFrame({
            "Year": r["annual_simple"].index.year,
            "SPGSCI Return": r["annual_simple"]["SPGSCI"].round(2).values,
            "BCOM Return": r["annual_simple"]["BCOM"].round(2).values,
            "SPGSCI Vol": r["annual_vol_yr"]["SPGSCI"].round(2).values,
            "BCOM Vol": r["annual_vol_yr"]["BCOM"].round(2).values,
            "SPGSCI Within-Yr DD": (
                r["annual_dd_within"].reindex(r["annual_simple"].index.year)
                ["SPGSCI"].round(2).values
            ),
            "BCOM Within-Yr DD": (
                r["annual_dd_within"].reindex(r["annual_simple"].index.year)
                ["BCOM"].round(2).values
            ),
        })
        for col in annual_table.columns[1:]:
            annual_table[col] = annual_table[col].map(lambda v: f"{v:.2f}%")
        st.dataframe(annual_table, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 :: YTD
# ============================================================
with tab2:
    st.header(f"Year-to-Date {current_year} Analysis")
    if ytd["empty"]:
        st.info("Not enough data points in the current year for YTD analysis.")
    else:
        st.caption(
            f"Period: {ytd['df'].index.min().date()} to {ytd['df'].index.max().date()}"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P GSCI CAGR (YTD)", f"{ytd['cagr']['SPGSCI']:.2f}%",
                  f"{ytd['cum_pct']['SPGSCI'].iloc[-1]:.2f}% total")
        c2.metric("Bloomberg BCOM CAGR (YTD)", f"{ytd['cagr']['BCOM']:.2f}%",
                  f"{ytd['cum_pct']['BCOM'].iloc[-1]:.2f}% total")
        c3.metric("S&P GSCI YTD Vol", f"{ytd['ann_vol']['SPGSCI']:.2f}%")
        c4.metric("Bloomberg BCOM YTD Vol", f"{ytd['ann_vol']['BCOM']:.2f}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P GSCI YTD Sharpe", f"{ytd['sharpe']['SPGSCI']:.3f}")
        c2.metric("Bloomberg BCOM YTD Sharpe", f"{ytd['sharpe']['BCOM']:.3f}")
        c3.metric("Correlation", f"{ytd['correlation']:.3f}")
        c4.metric("Beta (BCOM vs SPGSCI, full history)",
                  f"{r['beta_bcom_vs_gsci']:.3f}")

        st.divider()

        st.subheader(f"Index Performance Since 1 Jan {current_year} (Rebased to 100)")
        st.pyplot(fig_index_performance(ytd["rebased_100"]))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("YTD Drawdown (%)")
            st.pyplot(fig_drawdown(ytd["drawdown"], "YTD Drawdown (%)"))
        with c2:
            st.subheader("21-Day Rolling Volatility")
            st.pyplot(_line_panel(
                ytd["vol_roll_short"] * 100, "Annualised Vol (%)",
                "21-Day Rolling Volatility",
                color_map={"SPGSCI": COLOR_GSCI, "BCOM": COLOR_BCOM},
            ))

        st.subheader(f"Rolling 30-Day Correlation ({current_year})")
        st.pyplot(_line_panel(
            ytd["roll_corr_short"], "Correlation",
            f"Rolling 30-Day Correlation ({current_year})",
        ))

        st.subheader("Daily Return Scatter")
        st.pyplot(fig_scatter(ytd["log_ret"], ytd["correlation"]))

        st.subheader("YTD Tail Risk")
        st.dataframe(
            tail_risk_table(ytd["var_95"], ytd["cvar_95"],
                            ytd["var_99"], ytd["cvar_99"]),
            use_container_width=True,
        )


# ============================================================
# TAB 3 :: SEASONALITY
# ============================================================
with tab3:
    st.header("Seasonality Analysis")

    st.subheader("Monthly Return Heatmaps")
    st.pyplot(fig_heatmaps(r["monthly_simple"]))

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Average Monthly Return Profile")
        st.pyplot(fig_monthly_profile(r["monthly_simple"]))
    with c2:
        st.subheader("Win Rate by Month")
        win_fig, win_spgsci, win_bcom = fig_win_rate(r["monthly_simple"])
        st.pyplot(win_fig)

    st.subheader("Monthly Return Distribution by Calendar Month")
    st.pyplot(fig_boxplots(r["monthly_simple"]))

    with st.expander("View Seasonality Statistics Table"):
        monthly_avg = r["monthly_simple"].groupby(r["monthly_simple"].index.month).mean()
        seasonality_table = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "SPGSCI Avg": monthly_avg["SPGSCI"].round(2).values,
            "SPGSCI Win Rate": win_spgsci.round(1).values,
            "BCOM Avg": monthly_avg["BCOM"].round(2).values,
            "BCOM Win Rate": win_bcom.round(1).values,
        })
        for col in ["SPGSCI Avg", "SPGSCI Win Rate", "BCOM Avg", "BCOM Win Rate"]:
            seasonality_table[col] = seasonality_table[col].map(lambda v: f"{v:.2f}%")
        st.dataframe(seasonality_table, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 :: EVENT IMPACT
# ============================================================
with tab4:
    st.header("Event Impact Analysis")
    st.caption(
        f"Performance from selected event date (default: 24 Feb 2022, "
        f"Russia-Ukraine war). Adjust the date in the sidebar."
    )

    if war_period["empty"]:
        st.info("Not enough data after the selected event date.")
    else:
        st.markdown(f"**Event date:** {war_date}  |  "
                    f"**Window:** {war_period['df'].index.min().date()} → "
                    f"{war_period['df'].index.max().date()}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P GSCI CAGR", f"{war_period['cagr']['SPGSCI']:.2f}%",
                  f"{war_period['cum_pct']['SPGSCI'].iloc[-1]:.2f}% total")
        c2.metric("Bloomberg BCOM CAGR", f"{war_period['cagr']['BCOM']:.2f}%",
                  f"{war_period['cum_pct']['BCOM'].iloc[-1]:.2f}% total")
        c3.metric("S&P GSCI Vol", f"{war_period['ann_vol']['SPGSCI']:.2f}%")
        c4.metric("Bloomberg BCOM Vol", f"{war_period['ann_vol']['BCOM']:.2f}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P GSCI Sharpe", f"{war_period['sharpe']['SPGSCI']:.3f}")
        c2.metric("Bloomberg BCOM Sharpe", f"{war_period['sharpe']['BCOM']:.3f}")
        c3.metric("S&P GSCI Max DD", f"{war_period['drawdown']['SPGSCI'].min():.2f}%")
        c4.metric("Bloomberg BCOM Max DD", f"{war_period['drawdown']['BCOM'].min():.2f}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P GSCI DD Duration", f"{war_period['dd_duration']['SPGSCI']} days")
        c2.metric("Bloomberg BCOM DD Duration", f"{war_period['dd_duration']['BCOM']} days")
        c3.metric(
            "Lower-Tail Dependence (5%)",
            f"{war_period['tail_dep_5']*100:.1f}%"
            if pd.notna(war_period["tail_dep_5"]) else "n/a",
        )
        c4.metric("Correlation", f"{war_period['correlation']:.3f}")

        st.divider()

        st.subheader("Sharpe with Statistical Inference (Event Window)")
        st.dataframe(
            sharpe_inference_table(war_period["sharpe"], war_period["sharpe_se"],
                                   war_period["psr"]),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Index Performance Since Event (Rebased to 100)")
        st.pyplot(fig_index_performance(war_period["rebased_100"]))

        st.subheader("Drawdown Since Event (%)")
        st.pyplot(fig_drawdown(war_period["drawdown"], "Drawdown Since Event (%)"))

        st.subheader("Tail Risk Since Event")
        st.dataframe(
            tail_risk_table(war_period["var_95"], war_period["cvar_95"],
                            war_period["var_99"], war_period["cvar_99"]),
            use_container_width=True,
        )


# ============================================================
# TAB 5 :: SIDE-BY-SIDE COMPARISON
# ============================================================
with tab5:
    st.header(f"Full History vs YTD {current_year} Comparison")

    if ytd["empty"]:
        st.info("YTD data unavailable for comparison.")
    else:
        full_m = {
            "CAGR (%)": r["cagr"],
            "Ann Vol (%)": r["ann_vol"],
            "Sharpe": r["sharpe"],
            "Sortino": r["sortino"],
            "Calmar": r["calmar"],
            "Max DD (%)": r["drawdown"].min(),
            "Skewness": r["log_ret"].skew(),
            "Kurtosis": r["log_ret"].kurtosis(),
            "VaR 95% (daily %)": r["var_95"] * 100,
            "CVaR 95% (daily %)": r["cvar_95"] * 100,
        }
        ytd_m = {
            "CAGR (%)": ytd["cagr"],
            "Ann Vol (%)": ytd["ann_vol"],
            "Sharpe": ytd["sharpe"],
            "Sortino": ytd["sortino"],
            "Calmar": ytd["calmar"],
            "Max DD (%)": ytd["drawdown"].min(),
            "Skewness": ytd["log_ret"].skew(),
            "Kurtosis": ytd["log_ret"].kurtosis(),
            "VaR 95% (daily %)": ytd["var_95"] * 100,
            "CVaR 95% (daily %)": ytd["cvar_95"] * 100,
        }

        st.subheader("Risk-Return Metrics Comparison")
        st.pyplot(fig_comparison_bars(full_m, ytd_m))

        st.divider()
        st.subheader("Compact Metric Tables")
        c1, c2 = st.columns(2)
        with c1:
            comp_spgsci = pd.DataFrame({
                "Metric": list(full_m.keys()) + ["Correlation"],
                "Full History": [f"{full_m[m]['SPGSCI']:.3f}" for m in full_m]
                                + [f"{r['correlation']:.3f}"],
                f"YTD {current_year}": [f"{ytd_m[m]['SPGSCI']:.3f}" for m in ytd_m]
                                       + [f"{ytd['correlation']:.3f}"],
            })
            st.markdown("**S&P GSCI**")
            st.dataframe(comp_spgsci, use_container_width=True, hide_index=True)
        with c2:
            comp_bcom = pd.DataFrame({
                "Metric": list(full_m.keys()) + ["Correlation"],
                "Full History": [f"{full_m[m]['BCOM']:.3f}" for m in full_m]
                                + [f"{r['correlation']:.3f}"],
                f"YTD {current_year}": [f"{ytd_m[m]['BCOM']:.3f}" for m in ytd_m]
                                       + [f"{ytd['correlation']:.3f}"],
            })
            st.markdown("**Bloomberg BCOM**")
            st.dataframe(comp_bcom, use_container_width=True, hide_index=True)


# ============================================================
# SIDEBAR :: DOWNLOADS
# ============================================================
@st.cache_data
def build_export_full(df, log_ret, cum_wealth, drawdown, vol_252,
                      roll_sharpe, roll_beta):
    """Bundle the full-history series into a single export-ready DataFrame."""
    return pd.DataFrame({
        "SPGSCI_Close": df["SPGSCI"],
        "BCOM_Close": df["BCOM"],
        "SPGSCI_Daily_Return": log_ret["SPGSCI"],
        "BCOM_Daily_Return": log_ret["BCOM"],
        "SPGSCI_Cum_Wealth": cum_wealth["SPGSCI"],
        "BCOM_Cum_Wealth": cum_wealth["BCOM"],
        "SPGSCI_Drawdown": drawdown["SPGSCI"],
        "BCOM_Drawdown": drawdown["BCOM"],
        "SPGSCI_Roll_Vol_1Y": vol_252["SPGSCI"],
        "BCOM_Roll_Vol_1Y": vol_252["BCOM"],
        "SPGSCI_Roll_Sharpe_1Y": roll_sharpe["SPGSCI"],
        "BCOM_Roll_Sharpe_1Y": roll_sharpe["BCOM"],
        "BCOM_Roll_Beta_1Y": roll_beta,
    })


@st.cache_data
def build_export_period(period: Dict, label: str) -> pd.DataFrame:
    """Bundle a sliced-period analysis into an export-ready DataFrame."""
    return pd.DataFrame({
        f"SPGSCI_Close_{label}": period["df"]["SPGSCI"],
        f"BCOM_Close_{label}": period["df"]["BCOM"],
        f"SPGSCI_Return_{label}": period["log_ret"]["SPGSCI"],
        f"BCOM_Return_{label}": period["log_ret"]["BCOM"],
        f"SPGSCI_Cum_{label}": period["cum_pct"]["SPGSCI"],
        f"BCOM_Cum_{label}": period["cum_pct"]["BCOM"],
        f"SPGSCI_DD_{label}": period["drawdown"]["SPGSCI"],
        f"BCOM_DD_{label}": period["drawdown"]["BCOM"],
    })


st.sidebar.markdown("---")
st.sidebar.subheader("Download Data")
export_full = build_export_full(
    df, r["log_ret"], r["cum_wealth"], r["drawdown"],
    r["vol_252"], r["roll_sharpe"], r["roll_beta_bcom"],
)
st.sidebar.download_button(
    "Download Full History CSV",
    export_full.to_csv().encode("utf-8"),
    "commodity_index_full.csv",
    "text/csv",
)
if not ytd["empty"]:
    export_ytd = build_export_period(ytd, f"YTD{current_year}")
    st.sidebar.download_button(
        f"Download YTD {current_year} CSV",
        export_ytd.to_csv().encode("utf-8"),
        f"commodity_index_ytd_{current_year}.csv",
        "text/csv",
    )
