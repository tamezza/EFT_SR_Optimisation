#!/usr/bin/env python
"""
fit_mass_peaks.py — Standalone Gaussian Fitting for VBF Boosted HH → 4b
=========================================================================

Fits m_H1 and m_H2 distributions with an iterative Gaussian procedure:
  1. Broad initial fit in a given range
  2. Refined fits in μ ± N·σ windows (N = 1.0, 1.5, 2.0, 2.5, 3.0)
  3. Best fit selected by χ²/ndf closest to 1
  4. Resolution parameters derived: p = σ × c

Produces three sets of plots per Higgs candidate:
  (A) Raw distribution — no fit, just the histogram
  (B) Distribution + best fit — with pull panel underneath
  (C) All-fits overview — every Gaussian candidate overlaid, best one highlighted

Usage:
  python fit_mass_peaks.py --signal-file path/to/signal.root [options]

  # Multiple files (they get merged):
  python fit_mass_peaks.py --signal-file file1.root file2.root

  # With a parquet file instead of ROOT:
  python fit_mass_peaks.py --parquet-file path/to/signal.parquet
"""

import os
import argparse
import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


# =============================================================================
# ATLAS-style matplotlib configuration
# =============================================================================

ATLAS_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.linewidth": 1.5,
    "xtick.major.size": 8,
    "xtick.minor.size": 4,
    "xtick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "xtick.direction": "in",
    "xtick.top": True,
    "xtick.minor.visible": True,
    "ytick.major.size": 8,
    "ytick.minor.size": 4,
    "ytick.major.width": 1.2,
    "ytick.minor.width": 0.8,
    "ytick.direction": "in",
    "ytick.right": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
}
plt.rcParams.update(ATLAS_STYLE)


def draw_atlas_label(ax, x, y, text="Internal Simulation", extra_lines=None):
    """Draw ATLAS label in bold-italic + qualifier text."""
    ax.text(x, y, "ATLAS", fontsize=16, fontweight="bold", fontstyle="italic",
            transform=ax.transAxes, ha="left", va="top")
    ax.text(x + 0.17, y, text, fontsize=14, fontstyle="italic",
            transform=ax.transAxes, ha="left", va="top")
    if extra_lines:
        for i, line in enumerate(extra_lines):
            ax.text(x, y - 0.065 * (i + 1), line, fontsize=11,
                    transform=ax.transAxes, ha="left", va="top")


# =============================================================================
# Configuration defaults
# =============================================================================

DEFAULT_PARAMS = {
    "c1": 124.0, "c2": 117.0,
    "p1": 1500.0, "p2": 1900.0,
    "cut": 1.6,
}

SELECTION_CUT = "pass_boosted_vbf_sel == 1 && boosted_vbf_has_btag == 0 && boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2"
MASS_VAR_H1 = "boosted_m_h1"
MASS_VAR_H2 = "boosted_m_h2"
WEIGHT_VAR = "mc_sf"

MASS_RANGE_PLOT = (50, 300)
FIT_RANGE_H1 = (100, 150)
FIT_RANGE_H2 = (100, 150)
NBINS = 50


# =============================================================================
# Gaussian model and fitting helpers
# =============================================================================

def gauss(x, A, mu, sigma):
    """Simple Gaussian."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _compute_chi2_ndf(x, y, y_err, popt, nparams=3):
    """χ²/ndf for a binned fit. Bins with zero error are excluded."""
    y_pred = gauss(x, *popt)
    valid = y_err > 0
    n_valid = int(np.sum(valid))
    if n_valid <= nparams:
        return np.inf, 0
    residuals = (y[valid] - y_pred[valid]) / y_err[valid]
    chi2 = float(np.sum(residuals ** 2))
    ndf = n_valid - nparams
    return chi2, ndf


def _do_single_fit(centers, counts, sumw2, fit_lo, fit_hi, init_A, init_mu, init_sigma):
    """
    Perform a single Gaussian fit in [fit_lo, fit_hi].
    Returns a result dict or None if the fit fails.
    """
    mask = (centers >= fit_lo) & (centers <= fit_hi)
    x_fit = centers[mask]
    y_fit = counts[mask]
    w2_fit = sumw2[mask]

    y_err = np.sqrt(w2_fit)
    y_err_safe = np.where(y_err > 0, y_err, np.inf)

    if len(x_fit) < 5 or np.sum(y_fit) == 0:
        return None

    try:
        popt, pcov = curve_fit(
            gauss, x_fit, y_fit,
            p0=[init_A, init_mu, init_sigma],
            sigma=y_err_safe,
            absolute_sigma=True,
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        chi2, ndf = _compute_chi2_ndf(x_fit, y_fit, y_err, popt)
        chi2_ndf = chi2 / ndf if ndf > 0 else np.inf

        return {
            "A": popt[0], "mu": popt[1], "sigma": abs(popt[2]),
            "A_err": perr[0], "mu_err": perr[1], "sigma_err": perr[2],
            "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2_ndf,
            "fit_range": (fit_lo, fit_hi),
            "x_fit": x_fit, "y_fit": y_fit, "y_err": y_err,
        }
    except Exception:
        return None


# =============================================================================
# Iterative fitting
# =============================================================================

def fit_mass_peak(m_arr, w_arr, fit_range, init_mu, nbins=NBINS,
                  mass_range=MASS_RANGE_PLOT):
    """
    Iterative Gaussian fit:
      1. Broad initial fit in fit_range
      2. Refined fits in μ ± N·σ for N = 1.0, 1.5, 2.0, 2.5, 3.0
      3. If the seed shifts significantly, re-iterate once
      4. Select fit with χ²/ndf closest to 1

    Returns dict with best-fit result + all candidates, or None.
    """
    # ----- Build weighted histogram with Sumw2 -----
    bin_edges = np.linspace(mass_range[0], mass_range[1], nbins + 1)
    counts, edges = np.histogram(m_arr, bins=bin_edges, weights=w_arr)
    sumw2, _ = np.histogram(m_arr, bins=bin_edges, weights=w_arr ** 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = centers[1] - centers[0]

    # ----- Pass 0: broad initial fit -----
    mask0 = (centers >= fit_range[0]) & (centers <= fit_range[1])
    y0 = counts[mask0]
    if len(y0) < 5 or np.sum(y0) == 0:
        return None

    A0 = float(np.max(y0))
    sigma0 = 12.0

    initial_fit = _do_single_fit(centers, counts, sumw2,
                                 fit_range[0], fit_range[1],
                                 A0, init_mu, sigma0)
    if initial_fit is None:
        print(f"    Initial broad fit failed in [{fit_range[0]}, {fit_range[1]}]")
        return None

    initial_fit["label"] = f"Broad [{fit_range[0]:.0f}, {fit_range[1]:.0f}]"
    mu_seed = initial_fit["mu"]
    sigma_seed = initial_fit["sigma"]

    print(f"    Initial fit: μ = {mu_seed:.1f} GeV, σ = {sigma_seed:.1f} GeV, "
          f"χ²/ndf = {initial_fit['chi2_ndf']:.2f} "
          f"({initial_fit['chi2']:.1f}/{initial_fit['ndf']})")

    # ----- Pass 1: scan over refined windows -----
    sigma_windows = [1.0, 1.5, 2.0, 2.5, 3.0]
    candidates = [initial_fit]

    for nsig in sigma_windows:
        lo = max(mu_seed - nsig * sigma_seed, mass_range[0])
        hi = min(mu_seed + nsig * sigma_seed, mass_range[1])

        refit = _do_single_fit(centers, counts, sumw2, lo, hi,
                               initial_fit["A"], mu_seed, sigma_seed)
        if refit is not None:
            refit["label"] = f"±{nsig:.1f}σ [{lo:.0f}, {hi:.0f}]"
            candidates.append(refit)
            print(f"    Window ±{nsig:.1f}σ [{lo:.0f}, {hi:.0f}]: "
                  f"μ = {refit['mu']:.1f}, σ = {refit['sigma']:.1f}, "
                  f"χ²/ndf = {refit['chi2_ndf']:.2f} "
                  f"({refit['chi2']:.1f}/{refit['ndf']})")

    # ----- Pass 2 (optional): re-iterate if seed shifted -----
    best_pass1 = min(candidates, key=lambda r: abs(r["chi2_ndf"] - 1.0))
    mu_refined = best_pass1["mu"]
    sigma_refined = best_pass1["sigma"]

    if abs(mu_refined - mu_seed) > 1.0 or abs(sigma_refined - sigma_seed) > 1.0:
        print(f"    Re-iterating with refined seed: "
              f"μ = {mu_refined:.1f}, σ = {sigma_refined:.1f}")
        for nsig in sigma_windows:
            lo = max(mu_refined - nsig * sigma_refined, mass_range[0])
            hi = min(mu_refined + nsig * sigma_refined, mass_range[1])

            refit = _do_single_fit(centers, counts, sumw2, lo, hi,
                                   best_pass1["A"], mu_refined, sigma_refined)
            if refit is not None:
                refit["label"] = f"Pass2 ±{nsig:.1f}σ [{lo:.0f}, {hi:.0f}]"
                candidates.append(refit)

    # ----- Select best: χ²/ndf closest to 1, with sanity filters -----
    valid = [c for c in candidates
             if 3.0 < c["sigma"] < 40.0
             and 80.0 < c["mu"] < 160.0
             and c["ndf"] > 2]

    if not valid:
        print(f"    WARNING: No valid fits — falling back to initial fit")
        valid = [initial_fit]

    best = min(valid, key=lambda r: abs(r["chi2_ndf"] - 1.0))

    # Attach histogram data and all candidates for downstream plotting
    best["full_centers"] = centers
    best["full_counts"] = counts
    best["full_errors"] = np.sqrt(sumw2)
    best["bin_width"] = bin_width
    best["all_candidates"] = candidates

    print(f"    ★ Best fit: μ = {best['mu']:.1f} ± {best['mu_err']:.1f} GeV, "
          f"σ = {best['sigma']:.1f} ± {best['sigma_err']:.1f} GeV, "
          f"χ²/ndf = {best['chi2_ndf']:.2f} "
          f"({best['chi2']:.1f}/{best['ndf']}), "
          f"range [{best['fit_range'][0]:.0f}, {best['fit_range'][1]:.0f}]")

    return best


# =============================================================================
# Reading data from ROOT or parquet
# =============================================================================

def read_signal_from_root(file_paths, selection, mass_var_h1, mass_var_h2, weight_var):
    """Read m_H1, m_H2, weights from ROOT TTree(s)."""
    import ROOT as root
    root.gROOT.SetBatch(True)

    all_m1, all_m2, all_w = [], [], []

    for fpath in file_paths:
        tf = root.TFile.Open(fpath, "READ")
        if not tf or tf.IsZombie():
            print(f"  WARNING: Cannot open {fpath}")
            continue

        tree = tf.Get("ttree")
        if not tree:
            print(f"  WARNING: No 'ttree' in {fpath}")
            tf.Close()
            continue

        # Apply selection via entry list
        n_pass = tree.Draw(">>elist", selection, "entrylist")
        elist = root.gDirectory.Get("elist")
        if not elist or n_pass <= 0:
            print(f"  WARNING: No events pass selection in {fpath}")
            tf.Close()
            continue

        tree.SetEntryList(elist)

        arrays = {}
        for var in [mass_var_h1, mass_var_h2, weight_var]:
            n = tree.Draw(var, selection, "goff")
            if n <= 0:
                break
            tree.SetEstimate(n)
            n = tree.Draw(var, selection, "goff")
            buf = tree.GetV1()
            arrays[var] = np.array([buf[i] for i in range(n)])

        tree.SetEntryList(0)
        tf.Close()

        if len(arrays) == 3:
            all_m1.append(arrays[mass_var_h1])
            all_m2.append(arrays[mass_var_h2])
            all_w.append(arrays[weight_var])
            print(f"  Loaded {n_pass} events from {os.path.basename(fpath)}")

    if not all_m1:
        return None, None, None

    return np.concatenate(all_m1), np.concatenate(all_m2), np.concatenate(all_w)


def read_signal_from_parquet(file_path, mass_var_h1, mass_var_h2, weight_var):
    """Read m_H1, m_H2, weights from a parquet file."""
    import pandas as pd
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df)} events from {os.path.basename(file_path)}")

    for col in [mass_var_h1, mass_var_h2, weight_var]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in parquet. "
                           f"Available: {list(df.columns[:20])}...")

    return (df[mass_var_h1].values,
            df[mass_var_h2].values,
            df[weight_var].values)


# =============================================================================
# Step-histogram drawing helper (ATLAS "HIST" style)
# =============================================================================

def _draw_step_histogram(ax, centers, counts, bin_width, **kwargs):
    """
    Draw an unfilled step histogram (like ROOT "HIST" draw option).
    Constructs the stepped outline from bin edges.
    """
    n = len(centers)
    x_steps = np.zeros(2 * n + 2)
    y_steps = np.zeros(2 * n + 2)

    for i in range(n):
        lo_edge = centers[i] - bin_width / 2.0
        hi_edge = centers[i] + bin_width / 2.0
        x_steps[2 * i + 1] = lo_edge
        x_steps[2 * i + 2] = hi_edge
        y_steps[2 * i + 1] = counts[i]
        y_steps[2 * i + 2] = counts[i]

    # Close at the edges
    x_steps[0] = centers[0] - bin_width / 2.0
    x_steps[-1] = centers[-1] + bin_width / 2.0
    y_steps[0] = 0.0
    y_steps[-1] = 0.0

    ax.plot(x_steps, y_steps, **kwargs)


# =============================================================================
# Plot A: Raw distributions (no fit)
# =============================================================================

def plot_raw_distributions(fit1, fit2, config_label, output_dir):
    """Plot m_H1 and m_H2 distributions with no fit overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, fit, var_label, default_c in [
        (ax1, fit1, r"$m_{H_1}$", DEFAULT_PARAMS["c1"]),
        (ax2, fit2, r"$m_{H_2}$", DEFAULT_PARAMS["c2"]),
    ]:
        centers = fit["full_centers"]
        counts = fit["full_counts"]
        errors = fit["full_errors"]
        bw = fit["bin_width"]

        # Unfilled step histogram
        _draw_step_histogram(ax, centers, counts, bw,
                             color="black", linewidth=1.5, label="Signal MC")

        # Error bars at bin centres
        ax.errorbar(centers, counts, yerr=errors,
                    fmt="none", ecolor="black", elinewidth=1, capsize=2)

        # Default centre reference line
        ax.axvline(default_c, color="gray", ls=":", lw=1.5, alpha=0.6,
                   label=f"Default centre: {default_c:.0f} GeV")

        ax.set_xlabel(f"{var_label} [GeV]")
        ax.set_ylabel(f"Weighted events / {bw:.1f} GeV")
        ax.set_xlim(MASS_RANGE_PLOT)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")

    # ATLAS label on left panel
    draw_atlas_label(ax1, 0.05, 0.95, "Internal Simulation",
                     extra_lines=[
                         r"$\sqrt{s}$ = 13 TeV",
                         r"VBF HH $\rightarrow$ 4b Boosted",
                     ])

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir,
                    f"raw_distributions_{config_label}.{ext}"))
    plt.close()
    print(f"  Saved: raw_distributions_{config_label}.pdf/png")


# =============================================================================
# Plot B: Best fit with pull panel
# =============================================================================

def plot_best_fit(fit1, fit2, config_label, output_dir):
    """
    Plot m_H1 and m_H2 with the best Gaussian fit overlaid.
    Includes a pull sub-panel underneath each distribution.
    """
    fig = plt.figure(figsize=(16, 7.5))
    outer = gridspec.GridSpec(1, 2, wspace=0.30)

    for col, fit, var_label, default_c in [
        (0, fit1, r"$m_{H_1}$", DEFAULT_PARAMS["c1"]),
        (1, fit2, r"$m_{H_2}$", DEFAULT_PARAMS["c2"]),
    ]:
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3.5, 1], hspace=0.05,
        )
        ax_main = fig.add_subplot(inner[0])
        ax_pull = fig.add_subplot(inner[1], sharex=ax_main)

        centers = fit["full_centers"]
        counts = fit["full_counts"]
        errors = fit["full_errors"]
        bw = fit["bin_width"]

        # ---- Main panel ----
        _draw_step_histogram(ax_main, centers, counts, bw,
                             color="black", linewidth=1.5, label="Signal MC")
        ax_main.errorbar(centers, counts, yerr=errors,
                         fmt="none", ecolor="black", elinewidth=1, capsize=2)

        # Fit range shading
        fit_lo, fit_hi = fit["fit_range"]
        ax_main.axvspan(fit_lo, fit_hi, alpha=0.08, color="red",
                        label=f"Fit range [{fit_lo:.0f}, {fit_hi:.0f}] GeV")

        # Gaussian curve
        x_curve = np.linspace(fit_lo - 10, fit_hi + 10, 300)
        y_curve = gauss(x_curve, fit["A"], fit["mu"], fit["sigma"])

        chi2_ndf_str = f"{fit['chi2']:.1f}/{fit['ndf']} = {fit['chi2_ndf']:.2f}"
        ax_main.plot(x_curve, y_curve, color="#D32F2F", linewidth=2,
                     label=(f"Gaussian fit:\n"
                            f"  $\\mu$ = {fit['mu']:.1f} $\\pm$ {fit['mu_err']:.1f} GeV\n"
                            f"  $\\sigma$ = {fit['sigma']:.1f} $\\pm$ {fit['sigma_err']:.1f} GeV\n"
                            f"  $\\chi^2$/ndf = {chi2_ndf_str}"))

        # Reference lines
        ax_main.axvline(fit["mu"], color="#D32F2F", ls="--", alpha=0.4, lw=1)
        ax_main.axvline(default_c, color="gray", ls=":", alpha=0.5, lw=1.5,
                        label=f"Default centre: {default_c:.0f} GeV")

        ax_main.set_ylabel(f"Weighted events / {bw:.1f} GeV")
        ax_main.set_xlim(MASS_RANGE_PLOT)
        ax_main.set_ylim(bottom=0)
        ax_main.legend(fontsize=9, loc="upper right")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # ---- Pull panel ----
        fit_curve_at_bins = gauss(centers, fit["A"], fit["mu"], fit["sigma"])
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = np.where(errors > 0,
                            (counts - fit_curve_at_bins) / errors, 0.0)

        ax_pull.bar(centers, pull, width=bw, color="steelblue",
                    edgecolor="steelblue", alpha=0.7, linewidth=0.5)
        ax_pull.axhline(0, color="black", lw=0.8)
        ax_pull.axhline(+2, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax_pull.axhline(-2, color="gray", ls="--", lw=0.8, alpha=0.5)

        ax_pull.set_xlabel(f"{var_label} [GeV]")
        ax_pull.set_ylabel("Pull")
        ax_pull.set_ylim(-4, 4)
        ax_pull.yaxis.set_major_locator(MultipleLocator(2))
        ax_pull.yaxis.set_minor_locator(MultipleLocator(1))

    # ATLAS label on left main panel
    axes_list = fig.get_axes()
    draw_atlas_label(axes_list[0], 0.05, 0.92, "Internal Simulation",
                     extra_lines=[
                         r"$\sqrt{s}$ = 13 TeV",
                         r"VBF HH $\rightarrow$ 4b Boosted",
                     ])

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir,
                    f"best_fit_{config_label}.{ext}"))
    plt.close()
    print(f"  Saved: best_fit_{config_label}.pdf/png")


# =============================================================================
# Plot C: All fits overview
# =============================================================================

# Colour palette for candidate fits
_CANDIDATE_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#d62728",  # red
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
]


def plot_all_fits(fit1, fit2, config_label, output_dir):
    """
    Plot all Gaussian fit candidates on top of the data, highlighting the best.
    Each candidate is drawn in a different colour with its fit range and χ²/ndf
    shown in the legend.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, fit, var_label, default_c in [
        (ax1, fit1, r"$m_{H_1}$", DEFAULT_PARAMS["c1"]),
        (ax2, fit2, r"$m_{H_2}$", DEFAULT_PARAMS["c2"]),
    ]:
        centers = fit["full_centers"]
        counts = fit["full_counts"]
        errors = fit["full_errors"]
        bw = fit["bin_width"]
        candidates = fit["all_candidates"]

        # Data histogram
        _draw_step_histogram(ax, centers, counts, bw,
                             color="black", linewidth=1.5,
                             label="Signal MC", zorder=10)
        ax.errorbar(centers, counts, yerr=errors,
                    fmt="none", ecolor="black", elinewidth=1,
                    capsize=2, zorder=10)

        # Draw each candidate fit
        best_chi2_ndf = fit["chi2_ndf"]
        best_range = fit["fit_range"]

        for i, cand in enumerate(candidates):
            color = _CANDIDATE_COLORS[i % len(_CANDIDATE_COLORS)]
            clo, chi = cand["fit_range"]

            # Check if this is the best fit
            is_best = (abs(cand["chi2_ndf"] - best_chi2_ndf) < 1e-6
                       and abs(cand["fit_range"][0] - best_range[0]) < 0.1)

            x_c = np.linspace(clo - 5, chi + 5, 300)
            y_c = gauss(x_c, cand["A"], cand["mu"], cand["sigma"])

            lw = 2.5 if is_best else 1.2
            alpha = 1.0 if is_best else 0.6
            ls = "-" if is_best else "--"
            star = " ★" if is_best else ""

            label_str = cand.get("label", f"[{clo:.0f}, {chi:.0f}]")
            ax.plot(x_c, y_c, color=color, linewidth=lw, alpha=alpha, ls=ls,
                    label=(f"{label_str}{star}\n"
                           f"  μ={cand['mu']:.1f}, σ={cand['sigma']:.1f}, "
                           f"χ²/ndf={cand['chi2_ndf']:.2f}"))

            # Light vertical lines at fit range boundaries
            ax.axvline(clo, color=color, ls=":", lw=0.6, alpha=0.3)
            ax.axvline(chi, color=color, ls=":", lw=0.6, alpha=0.3)

        ax.set_xlabel(f"{var_label} [GeV]")
        ax.set_ylabel(f"Weighted events / {bw:.1f} GeV")
        ax.set_xlim(MASS_RANGE_PLOT)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper right", ncol=1)

    draw_atlas_label(ax1, 0.05, 0.95, "Internal Simulation",
                     extra_lines=[
                         r"$\sqrt{s}$ = 13 TeV",
                         r"VBF HH $\rightarrow$ 4b Boosted",
                         "All Gaussian fit candidates",
                     ])

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir,
                    f"all_fits_overview_{config_label}.{ext}"))
    plt.close()
    print(f"  Saved: all_fits_overview_{config_label}.pdf/png")


# =============================================================================
# Print summary table
# =============================================================================

def print_summary(fit1, fit2):
    """Print a summary table of fitted parameters and derived resolution."""
    c1 = fit1["mu"]
    c2 = fit2["mu"]
    sigma1 = fit1["sigma"]
    sigma2 = fit2["sigma"]
    p1 = sigma1 * c1
    p2 = sigma2 * c2

    print("\n" + "=" * 65)
    print("  Fit Results Summary")
    print("=" * 65)
    print(f"  {'':20s} {'m_H1':>20s} {'m_H2':>20s}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'Centre μ [GeV]':20s} "
          f"{c1:>9.2f} ± {fit1['mu_err']:.2f}    "
          f"{c2:>9.2f} ± {fit2['mu_err']:.2f}")
    print(f"  {'Width σ [GeV]':20s} "
          f"{sigma1:>9.2f} ± {fit1['sigma_err']:.2f}    "
          f"{sigma2:>9.2f} ± {fit2['sigma_err']:.2f}")
    print(f"  {'χ²/ndf':20s} "
          f"{fit1['chi2']:.1f}/{fit1['ndf']} = {fit1['chi2_ndf']:.2f}        "
          f"{fit2['chi2']:.1f}/{fit2['ndf']} = {fit2['chi2_ndf']:.2f}")
    print(f"  {'Fit range [GeV]':20s} "
          f"[{fit1['fit_range'][0]:.0f}, {fit1['fit_range'][1]:.0f}]"
          f"               [{fit2['fit_range'][0]:.0f}, {fit2['fit_range'][1]:.0f}]")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'p = σ × c [GeV²]':20s} "
          f"{p1:>20.0f} {p2:>20.0f}")
    print(f"  {'Default c [GeV]':20s} "
          f"{DEFAULT_PARAMS['c1']:>20.1f} {DEFAULT_PARAMS['c2']:>20.1f}")
    print(f"  {'Default p [GeV²]':20s} "
          f"{DEFAULT_PARAMS['p1']:>20.0f} {DEFAULT_PARAMS['p2']:>20.0f}")
    print(f"  {'Δc [GeV]':20s} "
          f"{c1 - DEFAULT_PARAMS['c1']:>+20.2f} "
          f"{c2 - DEFAULT_PARAMS['c2']:>+20.2f}")
    print("=" * 65)

    return {"c1": c1, "c2": c2, "sigma1": sigma1, "sigma2": sigma2,
            "p1": p1, "p2": p2}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit m_H1/m_H2 mass peaks for VBF Boosted HH → 4b SR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--signal-file", nargs="+",
                             help="Signal ROOT file(s) containing 'ttree'")
    input_group.add_argument("--parquet-file",
                             help="Signal parquet file")

    parser.add_argument("--output-dir", default="./fit_output",
                        help="Output directory for plots (default: ./fit_output)")
    parser.add_argument("--config-label", default="VBF_SM",
                        help="Label for output filenames (default: VBF_SM)")
    parser.add_argument("--selection", default=SELECTION_CUT,
                        help="ROOT TTree selection cut (ignored for parquet)")
    parser.add_argument("--mass-var-h1", default=MASS_VAR_H1,
                        help=f"m_H1 branch/column name (default: {MASS_VAR_H1})")
    parser.add_argument("--mass-var-h2", default=MASS_VAR_H2,
                        help=f"m_H2 branch/column name (default: {MASS_VAR_H2})")
    parser.add_argument("--weight-var", default=WEIGHT_VAR,
                        help=f"Weight branch/column name (default: {WEIGHT_VAR})")
    parser.add_argument("--nbins", type=int, default=NBINS,
                        help=f"Number of bins (default: {NBINS})")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  Mass Peak Fitting — VBF Boosted HH → 4b")
    print("=" * 65)

    # ----- Load data -----
    if args.signal_file:
        print(f"\n  Reading {len(args.signal_file)} ROOT file(s)...")
        print(f"  Selection: {args.selection}")
        m1, m2, w = read_signal_from_root(
            args.signal_file, args.selection,
            args.mass_var_h1, args.mass_var_h2, args.weight_var,
        )
    else:
        print(f"\n  Reading parquet file: {args.parquet_file}")
        m1, m2, w = read_signal_from_parquet(
            args.parquet_file,
            args.mass_var_h1, args.mass_var_h2, args.weight_var,
        )

    if m1 is None or len(m1) == 0:
        print("  ERROR: No events loaded. Exiting.")
        return

    print(f"  Total events: {len(m1)}")

    # ----- Data-driven seed: histogram peak -----
    def _find_peak(m_arr, w_arr, fit_range, nbins=60):
        edges = np.linspace(fit_range[0], fit_range[1], nbins + 1)
        counts, _ = np.histogram(m_arr, bins=edges, weights=w_arr)
        if np.sum(counts) == 0:
            return None
        centers = 0.5 * (edges[:-1] + edges[1:])
        return float(centers[np.argmax(counts)])

    seed_mu1 = _find_peak(m1, w, FIT_RANGE_H1) or 124.0
    seed_mu2 = _find_peak(m2, w, FIT_RANGE_H2) or 117.0
    print(f"  Seed μ from data: m_H1 = {seed_mu1:.1f} GeV, "
          f"m_H2 = {seed_mu2:.1f} GeV")

    # ----- Fit m_H1 -----
    print(f"\n  Fitting {args.mass_var_h1}...")
    fit1 = fit_mass_peak(m1, w, FIT_RANGE_H1, init_mu=seed_mu1, nbins=args.nbins)

    # ----- Fit m_H2 -----
    print(f"\n  Fitting {args.mass_var_h2}...")
    fit2 = fit_mass_peak(m2, w, FIT_RANGE_H2, init_mu=seed_mu2, nbins=args.nbins)

    if fit1 is None or fit2 is None:
        print("\n  ERROR: One or both fits failed. Cannot produce plots.")
        return

    # ----- Summary -----
    params = print_summary(fit1, fit2)

    # ----- Save parameters to JSON -----
    import json
    json_out = os.path.join(args.output_dir, f"fit_results_{args.config_label}.json")
    json_data = {
        "c1": params["c1"], "c2": params["c2"],
        "sigma1": params["sigma1"], "sigma2": params["sigma2"],
        "p1": params["p1"], "p2": params["p2"],
        "fit1": {
            "mu": fit1["mu"], "mu_err": fit1["mu_err"],
            "sigma": fit1["sigma"], "sigma_err": fit1["sigma_err"],
            "chi2": fit1["chi2"], "ndf": fit1["ndf"],
            "chi2_ndf": fit1["chi2_ndf"],
            "fit_range": list(fit1["fit_range"]),
            "n_candidates": len(fit1["all_candidates"]),
        },
        "fit2": {
            "mu": fit2["mu"], "mu_err": fit2["mu_err"],
            "sigma": fit2["sigma"], "sigma_err": fit2["sigma_err"],
            "chi2": fit2["chi2"], "ndf": fit2["ndf"],
            "chi2_ndf": fit2["chi2_ndf"],
            "fit_range": list(fit2["fit_range"]),
            "n_candidates": len(fit2["all_candidates"]),
        },
        "defaults": DEFAULT_PARAMS,
    }
    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n  Saved: {json_out}")

    # ----- Plots -----
    print(f"\n  Producing plots in {args.output_dir}/")
    label = args.config_label

    plot_raw_distributions(fit1, fit2, label, args.output_dir)
    plot_best_fit(fit1, fit2, label, args.output_dir)
    plot_all_fits(fit1, fit2, label, args.output_dir)

    print("\n  All done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
