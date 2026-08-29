#!/usr/bin/env python3
"""Figures for the irreversibility budget feasibility study.
Reads sim/results/results.json; writes one PDF per figure. Never hardcodes numbers."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "sim", "results", "results.json")

plt.rcParams.update({
    "figure.figsize": (3.35, 2.0),
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "font.family": "serif",
    "pdf.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"


def wilson_err(p_hat, n, z=1.96):
    """Wilson-interval half-widths (lo, hi) for a proportion — sane at 0/1."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError("p_hat must be in [0, 1]")
    den = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / den
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    return max(0.0, p_hat - lo), max(0.0, hi - p_hat)

with open(RES) as f:
    data = json.load(f)
R = data["params"]["tolerance"]
res = data["results"]


def fig1():
    """Realized aggregate exposure, local gates vs budget, +/- collusion."""
    rows = [
        ("Local gates", "e1_local", VERM),
        ("Budget ledger", "e1_budget", BLUE),
        ("Local + collusion", "e1_local_attack", VERM),
        ("Budget + collusion", "e1_budget_attack", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 1.7))
    ys = np.arange(len(rows))[::-1]
    for y, (label, key, color) in zip(ys, rows):
        m = res[key]
        mean, p95 = m["exposure"] / R, m["exposure_p95"] / R
        ax.barh(y, mean, height=0.62, color=color,
                hatch="//" if "attack" in key else None,
                edgecolor="white", linewidth=0.5)
        ax.plot([p95], [y], marker="|", color="black", markersize=9,
                markeredgewidth=1.1)
        text = f"$P$(overdraw) = {m['overdraw']:.2f}"
        if mean > 1.5:  # long bar: label inside, clear of the p95 whisker
            ax.annotate(text, xy=(mean, y), xytext=(-6, 0),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=7, color="white",
                        bbox=dict(facecolor=color, edgecolor="none", pad=1.2))
        else:           # short bar: label outside, past the whisker
            ax.annotate(text, xy=(max(mean, p95), y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.55, 3.95)
    ax.annotate("tolerance $R$", xy=(1.0, 3.55), fontsize=7,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Realized aggregate exposure (fraction of $R$)")
    ax.set_xlim(0, 5.2)
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(HERE, "e1_exposure.pdf"))


def fig2():
    """Safety-liveness tradeoff vs budget size B/R."""
    bfs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    keys = [f"e2_bf{bf}" for bf in bfs]
    over = [res[k]["overdraw"] for k in keys]
    n = data["runs"]
    yerr = np.array([wilson_err(p_, n) for p_ in over]).T
    pre = [res[k]["exec_pre"] for k in keys]
    burst = [res[k]["exec_burst"] for k in keys]
    fig, ax = plt.subplots()
    ax.errorbar(bfs, over, yerr=yerr, color=VERM, marker="o",
                markersize=3.5, linewidth=1.2, capsize=2,
                label="$P$(overdraw)")
    ax.plot(bfs, pre, color=BLUE, marker="s", markersize=3.5,
            linewidth=1.2, label="pre-burst routine executed")
    ax.plot(bfs, burst, color=GREEN, marker="^", markersize=3.5,
            linewidth=1.2, linestyle="--", label="correlated burst admitted")
    ax.set_xlabel("Budget size $B/R$")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center right", frameon=False)
    fig.savefig(os.path.join(HERE, "e2_tradeoff.pdf"))


def fig3():
    """Mispricing sensitivity: scale declared charges by epsilon."""
    epss = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    keys = [f"e3_eps{eps}" for eps in epss]
    over = [res[k]["overdraw"] for k in keys]
    n = data["runs"]
    yerr = np.array([wilson_err(p_, n) for p_ in over]).T
    util = [res[k]["utility_frac"] for k in keys]
    fig, ax = plt.subplots()
    ax.errorbar(epss, over, yerr=yerr, color=VERM, marker="o",
                markersize=3.5, linewidth=1.2, capsize=2,
                label="$P$(overdraw)")
    ax.plot(epss, util, color=BLUE, marker="s", markersize=3.5,
            linewidth=1.2, label="proposed value executed")
    ax.set_xscale("log")
    ax.set_xticks(epss)
    ax.set_xticklabels([f"{e:g}" for e in epss])
    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.7)
    ax.annotate("underpriced", xy=(0.5, 0.55), fontsize=7, ha="center",
                style="italic")
    ax.annotate("overpriced", xy=(2.0, 0.55), fontsize=7, ha="center",
                style="italic")
    ax.set_xlabel(r"Charge scaling factor $\varepsilon$")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(os.path.join(HERE, "e3_mispricing.pdf"))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote e1_exposure.pdf, e2_tradeoff.pdf, e3_mispricing.pdf")
