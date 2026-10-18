"""Shared helpers: DuckDB connection, statistics, explainable binning and pitch-ready plotting."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import config as C  # noqa: E402

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ci360")


@contextmanager
def timer(msg: str):
    t0 = time.time()
    log.info("START  %s", msg)
    yield
    log.info("DONE   %s (%.1fs)", msg, time.time() - t0)


# --------------------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------------------
def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{C.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={C.DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path(os.environ.get("CI360_TMP_DIR", Path(tempfile.gettempdir()) / "ci360_duckdb"))   # spill-to-disk area
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con



def ensure_dirs(*dirs: Path):
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)



def save_json(obj, path: Path):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def auc(y, score) -> float:
    """ROC-AUC via the Mann-Whitney rank formulation (higher score => more likely y=1)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, score) -> float:
    return 2 * auc(y, score) - 1


def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def wilson_ci(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def pav(values, weights, increasing=True):
    """Weighted pool-adjacent-violators: monotone fit of a sequence (used to make bin bad-rates monotone)."""
    v = list(np.asarray(values, float) if increasing else -np.asarray(values, float))
    w = list(np.asarray(weights, float))
    blocks = [[v[i], w[i], 1] for i in range(len(v))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            tw = blocks[i][1] + blocks[i + 1][1]
            tv = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / tw
            blocks[i] = [tv, tw, blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for val, _, cnt in blocks:
        out += [val] * cnt
    out = np.array(out)
    return out if increasing else -out


def calibrate_bins(x: pd.Series, y: pd.Series, edges, direction: str, missing_label="Missing",
                   min_points=0.0):
    """
    Explainable evidence-based sub-score.
      1) bucket a feature using business-readable edges (+ an explicit Missing bucket),
      2) measure the observed out-of-time bad rate in each bucket,
      3) enforce monotonicity with PAV in the expected risk direction,
      4) linearly map bad rate -> points: lowest-risk bucket = 100, highest-risk bucket = 0.
    direction='increasing' means risk increases with the feature value.
    Returns a DataFrame scorecard (bucket, lo, hi, n, bad_rate, bad_rate_monotone, points).
    """
    edges = list(edges)
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    xc = x.clip(lower=edges[0], upper=np.nextafter(edges[-1], -np.inf))
    code = pd.cut(xc, bins=edges, right=False, include_lowest=True, labels=False)
    df = pd.DataFrame({"code": code.values, "y": y.values})
    df.loc[x.isna().values, "code"] = -1
    g = df.groupby("code").agg(n=("y", "size"), bad=("y", "sum")).reset_index()
    g["bucket"] = [missing_label if c == -1 else f"[{_fmt_edge(edges[int(c)])}, {_fmt_edge(edges[int(c) + 1])})"
                   for c in g["code"]]
    rng = g[g["code"] >= 0].sort_values("code").copy()
    rng["lo"] = [edges[int(o)] for o in rng["code"]]
    rng["hi"] = [edges[int(o) + 1] for o in rng["code"]]
    rng["bad_rate"] = rng["bad"] / rng["n"]
    rng["bad_rate_monotone"] = pav(rng["bad_rate"].values, rng["n"].values, increasing=(direction == "increasing"))
    miss = g[g["code"] == -1].copy()
    if len(miss):
        miss["lo"], miss["hi"] = np.nan, np.nan
        miss["bad_rate"] = miss["bad"] / miss["n"]
        miss["bad_rate_monotone"] = miss["bad_rate"]
    card = pd.concat([rng, miss], ignore_index=True)
    # Points are linear in log-odds (Weight-of-Evidence logic, the credit-scorecard standard) so that
    # "twice the odds of going bad" costs the same number of points whatever the factor.
    br = card["bad_rate_monotone"].clip(1e-4, 1 - 1e-4)
    card["log_odds"] = np.log(br / (1 - br))
    lmin, lmax = card["log_odds"].min(), card["log_odds"].max()
    span = max(lmax - lmin, 1e-9)
    card["points"] = (min_points + (100 - min_points) * (lmax - card["log_odds"]) / span).round(1)
    return card[["bucket", "lo", "hi", "n", "bad", "bad_rate", "bad_rate_monotone", "log_odds", "points"]]


def _fmt_edge(v):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    return f"{v:g}"


def apply_card(x: pd.Series, card: pd.DataFrame, missing_label="Missing") -> pd.Series:
    """Map raw feature values to scorecard points (buckets with no dev observations inherit nearest bucket)."""
    rng = card[card["bucket"] != missing_label].sort_values("lo")
    los = rng["lo"].to_numpy(float)
    pts_arr = rng["points"].to_numpy(float)
    xv = np.asarray(x, dtype=float)
    idx = np.clip(np.searchsorted(los, xv, side="right") - 1, 0, len(los) - 1)
    out = pd.Series(pts_arr[idx], index=x.index, dtype=float)
    miss_pts = card.loc[card["bucket"] == missing_label, "points"]
    out[np.isnan(xv)] = float(miss_pts.iloc[0]) if len(miss_pts) else float(np.median(pts_arr))
    return out


def assign_tier(score: pd.Series, tiers) -> pd.Series:
    out = pd.Series(index=score.index, dtype=object)
    for lo, hi, name in tiers:
        out[(score >= lo) & (score < hi)] = name
    return out


# --------------------------------------------------------------------------------------
# Plotting (pitch-deck style)
# --------------------------------------------------------------------------------------
# BI Season 10 Fintech theme - identical hex codes to the dashboard (Tools/pbi/dashboard.html)
PAL = {
    "navy": "#0F172A",      # primary brand: titles, primary series
    "orange": "#F97316",    # accent: key highlights, secondary series
    "amber": "#F59E0B",     # accent 2 / Medium tier
    "green": "#10B981",     # High tier / healthy / safe
    "red": "#EF4444",       # Low tier / financial stress / at-risk
    "grey": "#64748B",      # neutral / secondary data
    "bg": "#F8FAFC", "card": "#FFFFFF",
    "light": (100 / 255, 116 / 255, 139 / 255, 0.18),   # neutral grey at 18% (volume bars, tracks)
    "ink": "#0F172A",
}
TIER_COLORS = {"High": PAL["green"], "Medium": PAL["amber"], "Low": PAL["red"]}




def status_cmap():
    """0 -> #EF4444, 50 -> #F59E0B, 100 -> #10B981 (points / health colour scale)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bi10_status", [PAL["red"], PAL["amber"], PAL["green"]])


def navy_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bi10_navy", [PAL["bg"], PAL["navy"]])


def tint_rgba(hex_color, a):
    from matplotlib.colors import to_rgba
    return to_rgba(hex_color, a)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PAL["grey"],
        "axes.labelcolor": PAL["ink"], "xtick.color": PAL["grey"], "ytick.color": PAL["grey"],
        "axes.grid": True, "grid.color": tint_rgba(PAL["grey"], 0.18), "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": PAL["card"], "savefig.facecolor": PAL["card"], "axes.facecolor": PAL["card"],
        "text.color": PAL["ink"], "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 300,
    })


def headline(fig, title: str, subtitle: str = ""):
    fig.text(0.012, 0.975, title, fontsize=17, fontweight="bold", color=PAL["navy"], va="top", ha="left")
    if subtitle:
        fig.text(0.012, 0.915, subtitle, fontsize=11.5, color=PAL["grey"], va="top", ha="left")
    # BI10 accent rule under the headline
    fig.add_artist(plt.Line2D([0.012, 0.075], [0.993, 0.993], color=PAL["orange"], lw=4, transform=fig.transFigure))


def save_fig(fig, path: Path, source: str | None = None, top=0.94, bottom=0.1, left=0.07, right=0.98, wspace=None):
    if source:
        fig.text(0.012, 0.018, f"Source: {source}", fontsize=8.5, color=PAL["grey"], ha="left", va="bottom", style="italic")
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def vnd(x):
    x = float(x)
    if abs(x) >= 1e12:
        return f"{x / 1e12:.1f}T"
    if abs(x) >= 1e9:
        return f"{x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"
