#!/usr/bin/env python
"""
SR Optimization for VBF Boosted HH → 4b Analysis (v2)
=======================================================

Optimizes the signal region (SR) definition parameters:
  c1, c2       : mass-plane ellipse centres [GeV]
  p1, p2       : resolution parameters [GeV^2]
  cut_value    : SR boundary discriminant threshold

for each configuration (MC campaign × tagger × jet kinematics mode).

The procedure follows ATLAS HDBS-2022-02:
  1. Fit the 2D signal mass peak  → (c1, c2)
  2. Fit resolution from widths   → (p1, p2)
  3. Scan cut value to maximise S/√B
  4. (Optional) joint 5-parameter optimisation via Nelder-Mead

Weight handling
---------------
  mc_sf [fb] = generatorWeight × xsec × kfactor × BR × gen_filter_eff / Σw_init
  Full physics weight = mc_sf × luminosity_boosted [fb⁻¹] × pileupweight_boosted [1]

  By default only mc_sf is used (shape-only). This is sufficient for:
    - Mass peak fitting (μ, σ are shape parameters — luminosity only scales the
      amplitude, pileup weight is uncorrelated with Higgs mass)
    - SR cut optimisation (luminosity cancels in S/√B ratio)
    - Signal efficiency (ratio, so luminosity cancels)

  Use --full-weight to include luminosity × pileup weight. This is needed for:
    - Reporting absolute S/√B numbers at a given luminosity
    - Data/MC comparison or stacked histogram plots
    - Inputs to statistical fitting (HistFitter, pyhf)

  The optimal SR parameters (c1, c2, p1, p2, cut) are IDENTICAL regardless
  of whether --full-weight is used.  Only absolute S, B, S/√B values change.

Outputs per configuration:
  - Raw mass distributions (no fit)
  - Fitted 1D mass projections with pull panels
  - All Gaussian fit candidates overlay
  - S/√B vs cut-value scan plot
  - Signal efficiency vs cut-value
  - Optimised mass-plane contour with marginal projections
  - Summary comparison table across all configurations

Usage:
  python optimize_sr.py [--base-dir DIR] [--bkg-dir DIR] [--full-weight]
"""

import ROOT as root
import os
import sys
import glob
import json
import argparse
import numpy as np
from scipy.optimize import curve_fit, minimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# Disable ROOT graphics and stats
root.gROOT.SetBatch(True)
root.gStyle.SetOptStat(0)
root.gStyle.SetOptTitle(0)


# ============================================================
# ATLAS Style for matplotlib
# ============================================================

ATLAS_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.linewidth": 1.5,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 8,
    "xtick.minor.size": 4,
    "xtick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.major.size": 8,
    "ytick.minor.size": 4,
    "ytick.major.width": 1.2,
    "ytick.minor.width": 0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
}
plt.rcParams.update(ATLAS_STYLE)


def draw_atlas_label(ax, x=0.05, y=0.95, label="Internal Simulation",
                     energy_lumi=True, channel=True, fontsize=15,
                     extra_lines=None):
    """
    Draw ATLAS-style label on a matplotlib axis.
    """
    ax.text(x, y, "ATLAS", fontsize=fontsize, fontweight="bold",
            fontstyle="italic", transform=ax.transAxes, va="top")
    ax.text(x + 0.13, y, label, fontsize=fontsize - 1,
            transform=ax.transAxes, va="top")

    offset = 0.065
    if energy_lumi:
        ax.text(x, y - offset, r"$\sqrt{s}$ = 13 TeV",
                fontsize=fontsize - 3, transform=ax.transAxes, va="top")
        offset += 0.055
    if channel:
        ax.text(x, y - offset, r"VBF $HH \rightarrow 4b$ Boosted",
                fontsize=fontsize - 3, transform=ax.transAxes, va="top")
        offset += 0.055
    if extra_lines:
        for line in extra_lines:
            ax.text(x, y - offset, line,
                    fontsize=fontsize - 3, transform=ax.transAxes, va="top")
            offset += 0.055


# ============================================================
# Configuration
# ============================================================

# Default parameters (from HDBS-2022-02)
DEFAULT_PARAMS = {
    "c1": 124.0, "c2": 117.0,
    "p1": 1500.0, "p2": 1900.0,
    "cut": 1.6,
}

MC_CAMPAIGNS = ["MC20", "MC23", "Combined"]
TAGGERS = ["GN2X", "GN3PV01"]
JET_MODES = ["DEFAULT", "bjr_v00", "bjr_v01"]

SELECTION_CUT = "pass_boosted_vbf_sel == 1 && boosted_vbf_has_btag == 0 && boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2"

# Variables to read from the trees
MASS_VAR_H1 = "boosted_m_h1"
MASS_VAR_H2 = "boosted_m_h2"

# Weight branches
#   mc_sf [fb] = generatorWeight × xsec[fb] × kfactor × BR × gen_filter_eff / Σw_init
#   luminosity_boosted [fb⁻¹] — integrated luminosity for boosted trigger GRL
#   pileupweight_boosted [dimensionless] — per-event pileup reweighting factor
WEIGHT_VAR = "mc_sf"
LUMI_VAR = "luminosity_boosted"
PILEUP_VAR = "pileupweight_boosted"

# Scan range for cut value
CUT_SCAN_MIN = 0.5
CUT_SCAN_MAX = 4.0
CUT_SCAN_STEP = 0.05

# Mass range for fits and histograms
MASS_RANGE = (50, 300)
MASS_RANGE_PLOT = (50, 200)
FIT_RANGE_H1 = (90, 160)  # restrict Gaussian fit to peak region
FIT_RANGE_H2 = (100, 150)
NBINS = 50

# Whether to use full physics weight (mc_sf × lumi × pileup)
# Set via --full-weight CLI flag
USE_FULL_WEIGHT = False

# Whether to draw background density contours on mass plane plots
# Set via --show-bkg-contours CLI flag
SHOW_BKG_CONTOURS = False


# ============================================================
# Helper: read arrays from TTree
# ============================================================

def read_tree_arrays(tree, variables, selection="", weight_var="mc_sf",
                     full_weight=False, lumi_var=LUMI_VAR,
                     pileup_var=PILEUP_VAR):
    """
    Read arrays from a TTree using TTree::Draw.

    If full_weight is True, reads mc_sf, luminosity_boosted, and
    pileupweight_boosted, then returns weight = mc_sf × lumi × pileup.
    Otherwise, returns weight = mc_sf.

    Returns dict of numpy arrays: {varname: array, ..., weight_var: array}
    """
    n_entries = tree.GetEntries()
    if n_entries == 0:
        return None

    n_pass = tree.Draw(">>elist", selection, "entrylist")
    elist = root.gDirectory.Get("elist")

    if not elist or n_pass <= 0:
        return None

    tree.SetEntryList(elist)

    # Determine which branches to read for the weight
    if full_weight:
        weight_branches = [weight_var, lumi_var, pileup_var]
    else:
        weight_branches = [weight_var]

    # Read mass variables + weight branches
    all_vars = list(variables) + weight_branches
    # Remove duplicates while preserving order
    seen = set()
    unique_vars = []
    for v in all_vars:
        if v not in seen:
            unique_vars.append(v)
            seen.add(v)

    arrays = {}
    # ROOT's internal Draw buffer defaults to 1M entries.
    # For large trees (e.g. QCDttbar) this overflows → segfault on GetV1().
    # SetEstimate tells ROOT to allocate a buffer large enough for n_pass entries.
    tree.SetEstimate(n_pass)

    for var in unique_vars:
        n = tree.Draw(var, selection, "goff")
        if n <= 0:
            tree.SetEntryList(0)
            return None
        buf = tree.GetV1()
        arrays[var] = np.array([buf[i] for i in range(n)])

    tree.SetEntryList(0)

    # Compute combined weight if requested
    if full_weight:
        arrays[weight_var] = (arrays[weight_var]
                              * arrays[lumi_var]
                              * arrays[pileup_var])

    return arrays


# ============================================================
# Helper: find ROOT files
# ============================================================

def _find_sample_dir(base_dir, mc_campaign, sample, tagger, jet_mode):
    """
    Locate the sample directory, handling directory structure and naming quirks.

    Expected layout:
        base_dir/{mc_campaign}/{sample}_{tagger}_{jet_mode}/RootSamp/

    MC23 naming inconsistency: GN3PV01 folders may be named GN3XPV01.
    """
    # Primary name
    folder = f"{sample}_{tagger}_{jet_mode}"
    root_samp = os.path.join(base_dir, mc_campaign, folder, "RootSamp")
    if os.path.exists(root_samp):
        return root_samp

    # Handle GN3PV01 ↔ GN3XPV01 inconsistency
    alt_tagger = None
    if tagger == "GN3PV01":
        alt_tagger = "GN3XPV01"
    elif tagger == "GN3XPV01":
        alt_tagger = "GN3PV01"

    if alt_tagger is not None:
        folder_alt = f"{sample}_{alt_tagger}_{jet_mode}"
        root_samp_alt = os.path.join(base_dir, mc_campaign, folder_alt, "RootSamp")
        if os.path.exists(root_samp_alt):
            return root_samp_alt

    return None


def find_signal_file(base_dir, mc_campaign, tagger, jet_mode,
                     signal_label="Kappa2V", signal_coupling="l1cvv0cv1"):
    """
    Find the merged signal ROOT file for a given configuration.

    signal_label   : folder prefix (e.g. "Kappa2V", "SM")
    signal_coupling: coupling tag in the ROOT filename (e.g. "l1cvv0cv1")
                     The κ₂V=0 point is l1cvv0cv1; SM is l1cvv1cv1.
    """
    root_samp = _find_sample_dir(base_dir, mc_campaign, signal_label,
                                 tagger, jet_mode)
    if root_samp is None:
        return None

    # Try coupling-specific patterns first, then progressively broader
    for pattern in [
        os.path.join(root_samp, f"boosted_skim_VBFhh_{signal_coupling}_{mc_campaign.lower()}*__Nominal.root"),
        os.path.join(root_samp, f"boosted_skim_VBFhh_{signal_coupling}_mc2*__Nominal.root"),
        os.path.join(root_samp, f"boosted_skim_*{signal_coupling}*__Nominal.root"),
        os.path.join(root_samp, f"*{signal_coupling}*__Nominal.root"),
        os.path.join(root_samp, f"*{signal_coupling}*.root"),
        os.path.join(root_samp, "boosted_skim_VBFhh_*__Nominal.root"),
        os.path.join(root_samp, "*__Nominal.root"),
        os.path.join(root_samp, "*.root"),
    ]:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


def find_background_files(base_dir, mc_campaign, tagger, jet_mode,
                          bkg_processes=None):
    """
    Find background ROOT files for specific processes only.

    Layout:
      base_dir/{mc_campaign}/{PROCESS}_{TAGGER}_{MODE}/RootSamp/*.root

    Only folders whose prefix is in bkg_processes are included.
    Default bkg_processes: ["QCD", "ttbar"]
      - QCD folder contains: boosted_*_JZ_JZ*_*_fullsim_mc20__Nominal.root
      - ttbar folder contains: boosted_*_ttbar_*_fullsim_mc20__Nominal.root
    All files from both are merged into a single combined background.

    Handles GN3PV01 ↔ GN3XPV01 naming inconsistency.

    Returns list of (filepath, process_label).
    """
    if bkg_processes is None:
        bkg_processes = ["QCD", "ttbar"]

    mc_dir = os.path.join(base_dir, mc_campaign)
    if not os.path.isdir(mc_dir):
        return []

    bkg_files = []

    # Build list of tagger names to check (primary + alternate)
    tagger_names = [tagger]
    if tagger == "GN3PV01":
        tagger_names.append("GN3XPV01")
    elif tagger == "GN3XPV01":
        tagger_names.append("GN3PV01")

    # For each background process, try all tagger name variants
    for process in bkg_processes:
        for t in tagger_names:
            folder = f"{process}_{t}_{jet_mode}"
            root_samp = os.path.join(mc_dir, folder, "RootSamp")
            if not os.path.exists(root_samp):
                continue
            for f in sorted(glob.glob(os.path.join(root_samp, "*.root"))):
                bkg_files.append((f, process))

    return bkg_files


# ============================================================
# Step 1 & 2: Fit mass peaks and derive resolution parameters
# ============================================================

def gauss(x, A, mu, sigma):
    """Simple Gaussian for peak fitting."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _compute_chi2_ndf(x, y, y_err, popt, nparams=3):
    """
    Compute χ²/ndf for a weighted histogram fit.
    Bins with zero error are excluded (empty bins).
    """
    y_pred = gauss(x, *popt)
    valid = y_err > 0
    n_valid = int(np.sum(valid))
    if n_valid <= nparams:
        return np.inf, 0
    residuals = (y[valid] - y_pred[valid]) / y_err[valid]
    chi2 = float(np.sum(residuals ** 2))
    ndf = n_valid - nparams
    return chi2, ndf


def _do_single_fit(centers, counts, sumw2, fit_lo, fit_hi,
                   init_A, init_mu, init_sigma):
    """
    Perform a single Gaussian fit in [fit_lo, fit_hi].
    Uses proper Sumw2 errors as weights in the fit.
    Returns result dict or None.
    """
    mask = (centers >= fit_lo) & (centers <= fit_hi)
    x_fit = centers[mask]
    y_fit = counts[mask]
    w2_fit = sumw2[mask]

    # Bin errors: sqrt(sum of weights^2)
    y_err = np.sqrt(w2_fit)
    # Replace zero errors with inf to exclude from fit
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


def fit_mass_peak(m_arr, w_arr, fit_range, init_mu, nbins=NBINS,
                  mass_range=MASS_RANGE_PLOT):
    """
    Iterative Gaussian fit with χ²/ndf-based window selection.

    Procedure:
      1. Broad initial fit in the given fit_range
      2. Use fitted (μ, σ) to define refined windows: μ ± N·σ
         for N = 1.0, 1.5, 2.0, 2.5, 3.0
      3. If the best candidate's seed shifted by > 1 GeV, re-iterate
      4. Select the fit closest to χ²/ndf = 1

    Returns result dict with best fit + all candidates, or None.
    """
    # Build weighted histogram with Sumw2
    bin_edges = np.linspace(mass_range[0], mass_range[1], nbins + 1)
    counts, edges = np.histogram(m_arr, bins=bin_edges, weights=w_arr)
    sumw2, _ = np.histogram(m_arr, bins=bin_edges, weights=w_arr ** 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = centers[1] - centers[0]

    # --- Pass 0: broad initial fit ---
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

    # --- Pass 1: scan over refined windows ---
    sigma_windows = [1.0, 1.5, 2.0, 2.5, 3.0]
    candidates = [initial_fit]  # include the broad fit as a candidate

    for nsig in sigma_windows:
        lo = max(mu_seed - nsig * sigma_seed, mass_range[0])
        hi = min(mu_seed + nsig * sigma_seed, mass_range[1])

        refit = _do_single_fit(centers, counts, sumw2,
                               lo, hi,
                               initial_fit["A"], mu_seed, sigma_seed)
        if refit is not None:
            refit["label"] = f"±{nsig:.1f}σ [{lo:.0f}, {hi:.0f}]"
            candidates.append(refit)
            print(f"    Window ±{nsig:.1f}σ [{lo:.0f}, {hi:.0f}]: "
                  f"μ = {refit['mu']:.1f}, σ = {refit['sigma']:.1f}, "
                  f"χ²/ndf = {refit['chi2_ndf']:.2f} "
                  f"({refit['chi2']:.1f}/{refit['ndf']})")

    # --- Pass 2 (optional): re-iterate if seed shifted significantly ---
    best_pass1 = min(candidates, key=lambda r: abs(r["chi2_ndf"] - 1.0))
    mu_refined = best_pass1["mu"]
    sigma_refined = best_pass1["sigma"]

    if abs(mu_refined - mu_seed) > 1.0 or abs(sigma_refined - sigma_seed) > 1.0:
        print(f"    Re-iterating with refined seed: "
              f"μ = {mu_refined:.1f}, σ = {sigma_refined:.1f}")
        for nsig in sigma_windows:
            lo = max(mu_refined - nsig * sigma_refined, mass_range[0])
            hi = min(mu_refined + nsig * sigma_refined, mass_range[1])

            refit = _do_single_fit(centers, counts, sumw2,
                                   lo, hi,
                                   best_pass1["A"], mu_refined, sigma_refined)
            if refit is not None:
                refit["label"] = f"Pass2 ±{nsig:.1f}σ [{lo:.0f}, {hi:.0f}]"
                candidates.append(refit)

    # --- Select best fit: closest to χ²/ndf = 1, with sanity filters ---
    valid = [c for c in candidates
             if 3.0 < c["sigma"] < 40.0
             and 80.0 < c["mu"] < 160.0
             and c["ndf"] > 2]

    if not valid:
        print(f"    WARNING: No valid fits found, using initial fit")
        valid = [initial_fit]

    best = min(valid, key=lambda r: abs(r["chi2_ndf"] - 1.0))

    # Attach full histogram and all candidates for plotting
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


def _find_histogram_peak(m_arr, w_arr, fit_range, nbins=60):
    """
    Find the bin centre with the maximum weighted count inside fit_range.
    Returns the peak position [GeV], or None if the histogram is empty.
    """
    edges = np.linspace(fit_range[0], fit_range[1], nbins + 1)
    counts, _ = np.histogram(m_arr, bins=edges, weights=w_arr)
    if np.sum(counts) == 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[np.argmax(counts)])


def fit_signal_parameters(sig_m1, sig_m2, sig_w):
    """
    Fit both mass peaks and derive the SR parameters.
    Returns dict with c1, c2, sigma1, sigma2, p1, p2, fit results.

    The initial μ seed for each fit is determined from the data:
    the bin with the maximum weighted count in the fit range is used.
    Falls back to hardcoded values (124, 117 GeV) if the histogram is empty.
    """
    # Data-driven seed: find peak of weighted histogram
    seed_mu1 = _find_histogram_peak(sig_m1, sig_w, FIT_RANGE_H1)
    seed_mu2 = _find_histogram_peak(sig_m2, sig_w, FIT_RANGE_H2)

    # Fallback to physics expectations if histogram is empty
    if seed_mu1 is None:
        seed_mu1 = 124.0
    if seed_mu2 is None:
        seed_mu2 = 117.0

    print(f"  Seed μ from data: m_H1 = {seed_mu1:.1f} GeV, "
          f"m_H2 = {seed_mu2:.1f} GeV")

    print("  Fitting m_H1 peak...")
    fit1 = fit_mass_peak(sig_m1, sig_w, FIT_RANGE_H1, init_mu=seed_mu1)

    print("  Fitting m_H2 peak...")
    fit2 = fit_mass_peak(sig_m2, sig_w, FIT_RANGE_H2, init_mu=seed_mu2)

    if fit1 is None or fit2 is None:
        print("  ERROR: Fit failed, using default parameters")
        return {
            "c1": DEFAULT_PARAMS["c1"], "c2": DEFAULT_PARAMS["c2"],
            "sigma1": DEFAULT_PARAMS["p1"] / DEFAULT_PARAMS["c1"],
            "sigma2": DEFAULT_PARAMS["p2"] / DEFAULT_PARAMS["c2"],
            "p1": DEFAULT_PARAMS["p1"], "p2": DEFAULT_PARAMS["p2"],
            "fit1": None, "fit2": None, "status": "FAILED",
        }

    c1 = fit1["mu"]
    c2 = fit2["mu"]
    sigma1 = fit1["sigma"]
    sigma2 = fit2["sigma"]

    # Resolution parameters: sigma = p / c  =>  p = sigma * c
    p1 = sigma1 * c1
    p2 = sigma2 * c2

    print(f"  Fitted: c1 = {c1:.1f} ± {fit1['mu_err']:.1f} GeV, "
          f"σ1 = {sigma1:.1f} ± {fit1['sigma_err']:.1f} GeV, "
          f"χ²/ndf = {fit1.get('chi2_ndf', 0):.2f}")
    print(f"  Fitted: c2 = {c2:.1f} ± {fit2['mu_err']:.1f} GeV, "
          f"σ2 = {sigma2:.1f} ± {fit2['sigma_err']:.1f} GeV, "
          f"χ²/ndf = {fit2.get('chi2_ndf', 0):.2f}")
    print(f"  Derived: p1 = {p1:.0f} GeV², p2 = {p2:.0f} GeV²")

    return {
        "c1": c1, "c2": c2,
        "sigma1": sigma1, "sigma2": sigma2,
        "p1": p1, "p2": p2,
        "fit1": fit1, "fit2": fit2,
        "status": "OK",
    }


# ============================================================
# SR discriminant
# ============================================================

def sr_discriminant(m1, m2, c1, c2, p1, p2):
    """Eq.(1): sqrt( (m1*(m1-c1)/p1)^2 + (m2*(m2-c2)/p2)^2 )"""
    return np.sqrt(
        (m1 * (m1 - c1) / p1) ** 2 +
        (m2 * (m2 - c2) / p2) ** 2
    )


# ============================================================
# Step 3: Scan cut values to maximise S/√B
# ============================================================

def scan_cut_values(sig_m1, sig_m2, sig_w,
                    bkg_m1, bkg_m2, bkg_w,
                    c1, c2, p1, p2,
                    cut_min=CUT_SCAN_MIN, cut_max=CUT_SCAN_MAX,
                    cut_step=CUT_SCAN_STEP):
    """
    Scan SR cut values and compute figures of merit.
    Returns list of dicts with cut, S, B, S/√B, S/B, signal_eff.
    """
    sig_D = sr_discriminant(sig_m1, sig_m2, c1, c2, p1, p2)
    total_S = np.sum(sig_w)

    has_bkg = bkg_m1 is not None and len(bkg_m1) > 0
    if has_bkg:
        bkg_D = sr_discriminant(bkg_m1, bkg_m2, c1, c2, p1, p2)
    else:
        bkg_D = None

    cut_values = np.arange(cut_min, cut_max + cut_step / 2, cut_step)
    results = []

    for cut in cut_values:
        S = np.sum(sig_w[sig_D < cut])

        if has_bkg:
            B = np.sum(bkg_w[bkg_D < cut])
        else:
            B = 0

        sig_eff = S / total_S if total_S > 0 else 0

        if B > 0:
            SoverSqrtB = S / np.sqrt(B)
            SoverB = S / B
        else:
            SoverSqrtB = 0
            SoverB = 0

        results.append({
            "cut": float(cut),
            "S": float(S),
            "B": float(B),
            "S/sqrt(B)": float(SoverSqrtB),
            "S/B": float(SoverB),
            "sig_eff": float(sig_eff),
        })

    return results


def find_optimal_cut(scan_results, metric="S/sqrt(B)"):
    """Find the cut value that maximises the given metric."""
    if not scan_results:
        return DEFAULT_PARAMS["cut"]

    has_bkg = any(r["B"] > 0 for r in scan_results)

    if has_bkg:
        best = max(scan_results, key=lambda r: r[metric])
    else:
        # Without background, find cut giving ~76% signal efficiency
        target_eff = 0.76
        best = min(scan_results, key=lambda r: abs(r["sig_eff"] - target_eff))
        print(f"  INFO: No background provided. Targeting {target_eff*100:.0f}% "
              f"signal efficiency → cut = {best['cut']:.2f}")

    return best["cut"]


# ============================================================
# Step 4: Joint 5-parameter optimisation (optional)
# ============================================================

def joint_optimization(sig_m1, sig_m2, sig_w,
                       bkg_m1, bkg_m2, bkg_w,
                       initial_params):
    """
    Joint optimisation of (c1, c2, p1, p2, cut) via Nelder-Mead.
    Returns optimised dict {c1, c2, p1, p2, cut, S/sqrt(B)}.
    """
    if bkg_m1 is None or len(bkg_m1) == 0:
        print("  INFO: Skipping joint optimisation (no background)")
        return None

    def neg_sensitivity(params):
        c1, c2, p1, p2, cut = params

        if p1 < 100 or p2 < 100 or cut < 0.3 or cut > 5.0:
            return 0
        if c1 < 80 or c1 > 160 or c2 < 80 or c2 > 160:
            return 0

        sig_D = sr_discriminant(sig_m1, sig_m2, c1, c2, p1, p2)
        bkg_D = sr_discriminant(bkg_m1, bkg_m2, c1, c2, p1, p2)

        S = np.sum(sig_w[sig_D < cut])
        B = np.sum(bkg_w[bkg_D < cut])

        if B <= 0 or S <= 0:
            return 0

        return -S / np.sqrt(B)

    x0 = [initial_params["c1"], initial_params["c2"],
          initial_params["p1"], initial_params["p2"],
          initial_params["cut"]]

    print("  Running joint 5-parameter optimisation (Nelder-Mead)...")
    result = minimize(neg_sensitivity, x0=x0, method='Nelder-Mead',
                      options={'xatol': 0.05, 'fatol': 0.001, 'maxiter': 10000,
                               'adaptive': True})

    if result.success or result.fun < 0:
        c1_opt, c2_opt, p1_opt, p2_opt, cut_opt = result.x
        fom = -result.fun
        print(f"  Joint optimisation converged after {result.nit} iterations")
        print(f"  Optimal: c1={c1_opt:.1f}, c2={c2_opt:.1f}, "
              f"p1={p1_opt:.0f}, p2={p2_opt:.0f}, cut={cut_opt:.2f}")
        print(f"  S/√B = {fom:.3f}")
        return {
            "c1": c1_opt, "c2": c2_opt,
            "p1": p1_opt, "p2": p2_opt,
            "cut": cut_opt,
            "S/sqrt(B)": fom,
            "sigma1": p1_opt / c1_opt,
            "sigma2": p2_opt / c2_opt,
            "converged": True,
            "niter": result.nit,
        }
    else:
        print(f"  WARNING: Joint optimisation did not converge: {result.message}")
        return None


# ============================================================
# Plotting helpers
# ============================================================

def _draw_step_histogram(ax, centers, counts, bin_width,
                         orientation="vertical", **kwargs):
    """
    Draw an unfilled step histogram (ATLAS / ROOT "HIST" style).
    orientation: "vertical" (normal) or "horizontal" (for right marginal).
    """
    n = len(centers)
    half_bw = bin_width / 2.0

    step_x = np.zeros(2 * n + 2)
    step_y = np.zeros(2 * n + 2)

    step_x[0] = centers[0] - half_bw
    step_y[0] = 0.0

    for i in range(n):
        step_x[2 * i + 1] = centers[i] - half_bw
        step_y[2 * i + 1] = counts[i]
        step_x[2 * i + 2] = centers[i] + half_bw
        step_y[2 * i + 2] = counts[i]

    step_x[-1] = centers[-1] + half_bw
    step_y[-1] = 0.0

    if orientation == "vertical":
        ax.plot(step_x, step_y, **kwargs)
    else:
        ax.plot(step_y, step_x, **kwargs)


# Colour palette for all-fits overview
_CANDIDATE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#d62728",
    "#aec7e8", "#ffbb78",
]


# ============================================================
# Plot A: Raw distributions (no fit)
# ============================================================

def plot_raw_distributions(fit_params, config_label, output_dir):
    """Plot m_H1 and m_H2 distributions with no fit overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, fit, var_label, default_c in [
        (ax1, fit_params["fit1"], r"$m_{H_1}$", DEFAULT_PARAMS["c1"]),
        (ax2, fit_params["fit2"], r"$m_{H_2}$", DEFAULT_PARAMS["c2"]),
    ]:
        if fit is None:
            ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='red')
            continue

        centers = fit["full_centers"]
        counts = fit["full_counts"]
        errors = fit["full_errors"]
        bw = fit["bin_width"]

        _draw_step_histogram(ax, centers, counts, bw,
                             color="black", linewidth=1.5, label="Signal MC")
        ax.errorbar(centers, counts, yerr=errors,
                    fmt="none", ecolor="black", elinewidth=1, capsize=2)
        ax.axvline(default_c, color="gray", ls=":", lw=1.5, alpha=0.6,
                   label=f"Default centre: {default_c:.0f} GeV")

        ax.set_xlabel(f"{var_label} [GeV]")
        ax.set_ylabel(f"Weighted events / {bw:.1f} GeV")
        ax.set_xlim(MASS_RANGE_PLOT)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")

    draw_atlas_label(ax1, x=0.05, y=0.95, fontsize=13)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir,
                    f"raw_distributions_{config_label}.{ext}"),
                    bbox_inches='tight')
    plt.close()
    print(f"  Saved: raw_distributions_{config_label}")


# ============================================================
# Plot B: Best fit with pull panel
# ============================================================

def plot_mass_fits(fit_params, config_label, output_dir):
    """
    Plot 1D mass distributions as ATLAS-style unfilled step histograms
    with Gaussian fit overlaid, plus a pull panel underneath.
    """
    fig = plt.figure(figsize=(16, 7.5))
    outer = gridspec.GridSpec(1, 2, wspace=0.32)

    for col, (fit, var_label, default_c) in enumerate([
        (fit_params["fit1"], r"$m_{H_1}$", DEFAULT_PARAMS["c1"]),
        (fit_params["fit2"], r"$m_{H_2}$", DEFAULT_PARAMS["c2"]),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3.5, 1], hspace=0.05,
        )

        ax_main = fig.add_subplot(inner[0])
        ax_pull = fig.add_subplot(inner[1], sharex=ax_main)

        if fit is None:
            ax_main.text(0.5, 0.5, "Fit failed", transform=ax_main.transAxes,
                         ha='center', va='center', fontsize=16, color='red')
            continue

        # ---- Data: unfilled step histogram ----
        full_x = fit["full_centers"]
        full_y = fit["full_counts"]
        full_err = fit["full_errors"]
        bw = fit["bin_width"]

        _draw_step_histogram(ax_main, full_x, full_y, bw,
                             color='black', linewidth=1.5, label='Signal MC')
        ax_main.errorbar(full_x, full_y, yerr=full_err,
                         fmt='none', ecolor='black', elinewidth=1, capsize=2)

        # ---- Fit range shading ----
        fit_lo, fit_hi = fit["fit_range"]
        ax_main.axvspan(fit_lo, fit_hi, alpha=0.06, color='red', zorder=0,
                        label=f'Fit range [{fit_lo:.0f}, {fit_hi:.0f}] GeV')

        # ---- Gaussian fit curve ----
        chi2_ndf = fit.get("chi2_ndf", 0)
        chi2 = fit.get("chi2", 0)
        ndf = fit.get("ndf", 0)

        x_curve = np.linspace(fit_lo - 10, fit_hi + 10, 300)
        y_curve = gauss(x_curve, fit["A"], fit["mu"], fit["sigma"])

        ax_main.plot(x_curve, y_curve, color='#D32F2F', linewidth=2,
                     label=(f'Gaussian fit\n'
                            f'  $\\mu$ = {fit["mu"]:.1f} $\\pm$ '
                            f'{fit["mu_err"]:.1f} GeV\n'
                            f'  $\\sigma$ = {fit["sigma"]:.1f} $\\pm$ '
                            f'{fit["sigma_err"]:.1f} GeV\n'
                            f'  $\\chi^2$/ndf = {chi2:.1f}/{ndf}'
                            f' = {chi2_ndf:.2f}'))

        # ---- Reference lines ----
        ax_main.axvline(fit["mu"], color='#D32F2F', ls='--', alpha=0.35, lw=1)
        ax_main.axvline(default_c, color='gray', ls=':', alpha=0.6, lw=1.5,
                        label=f'Default $c$ = {default_c} GeV')

        # ---- Axis styling ----
        ax_main.set_ylabel(f'Weighted events / {bw:.1f} GeV', fontsize=14)
        ax_main.set_xlim(MASS_RANGE_PLOT)
        y_max = np.max(full_y) * 1.55
        ax_main.set_ylim(0, y_max)
        ax_main.legend(fontsize=9, loc='upper right', handlelength=1.8)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # ---- ATLAS label ----
        if col == 0:
            draw_atlas_label(ax_main, x=0.05, y=0.95, fontsize=13)

        # ---- Pull panel ----
        fit_curve_at_bins = gauss(full_x, fit["A"], fit["mu"], fit["sigma"])
        with np.errstate(divide='ignore', invalid='ignore'):
            pull = np.where(full_err > 0,
                            (full_y - fit_curve_at_bins) / full_err, 0.0)

        ax_pull.bar(full_x, pull, width=bw * 0.85, color='steelblue',
                    edgecolor='steelblue', alpha=0.7)
        ax_pull.axhline(0, color='black', linewidth=0.8)
        ax_pull.axhline(2, color='gray', ls='--', linewidth=0.6, alpha=0.5)
        ax_pull.axhline(-2, color='gray', ls='--', linewidth=0.6, alpha=0.5)
        ax_pull.set_xlabel(f'{var_label} [GeV]', fontsize=14)
        ax_pull.set_ylabel('Pull', fontsize=12)
        ax_pull.set_ylim(-4, 4)
        ax_pull.set_xlim(MASS_RANGE_PLOT)
        ax_pull.yaxis.set_major_locator(MultipleLocator(2))
        ax_pull.yaxis.set_minor_locator(MultipleLocator(1))

    fig.savefig(os.path.join(output_dir, f"mass_fits_{config_label}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"mass_fits_{config_label}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mass_fits_{config_label}")


# ============================================================
# Plot B2: Individual mass fit (standalone, one per Higgs)
# ============================================================

def plot_single_mass_fit(fit, var_label, var_tag, default_c,
                         fit_params_dict, config_label, output_dir):
    """
    Standalone ATLAS-style mass fit plot for a single Higgs candidate.
    Upper panel: data (step histogram) + Gaussian fit + fit range shading.
    Lower panel: pull = (data − fit) / σ.

    Parameters:
        fit           : dict from fit_mass_peak (best fit result)
        var_label     : LaTeX label, e.g. r"$m_{H_1}$"
        var_tag       : short tag for filename, e.g. "mH1"
        default_c     : default centre value (reference line)
        fit_params_dict: full fit_params dict (for sigma/p annotation)
        config_label  : e.g. "MC20_GN3PV01_DEFAULT"
        output_dir    : directory to save plots
    """
    if fit is None:
        return

    fig = plt.figure(figsize=(8, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.05)

    ax_main = fig.add_subplot(gs[0])
    ax_pull = fig.add_subplot(gs[1], sharex=ax_main)

    # ---- Data: unfilled step histogram ----
    full_x = fit["full_centers"]
    full_y = fit["full_counts"]
    full_err = fit["full_errors"]
    bw = fit["bin_width"]

    _draw_step_histogram(ax_main, full_x, full_y, bw,
                         color='black', linewidth=1.5, label='Signal MC')
    ax_main.errorbar(full_x, full_y, yerr=full_err,
                     fmt='none', ecolor='black', elinewidth=1, capsize=2)

    # ---- Fit range shading ----
    fit_lo, fit_hi = fit["fit_range"]
    ax_main.axvspan(fit_lo, fit_hi, alpha=0.06, color='red', zorder=0,
                    label=f'Fit range [{fit_lo:.0f}, {fit_hi:.0f}] GeV')

    # ---- Gaussian fit curve ----
    chi2_ndf = fit.get("chi2_ndf", 0)
    chi2 = fit.get("chi2", 0)
    ndf = fit.get("ndf", 0)

    x_curve = np.linspace(fit_lo - 10, fit_hi + 10, 300)
    y_curve = gauss(x_curve, fit["A"], fit["mu"], fit["sigma"])

    ax_main.plot(x_curve, y_curve, color='#D32F2F', linewidth=2,
                 label=(f'Gaussian fit\n'
                        f'  $\\mu$ = {fit["mu"]:.1f} $\\pm$ '
                        f'{fit["mu_err"]:.1f} GeV\n'
                        f'  $\\sigma$ = {fit["sigma"]:.1f} $\\pm$ '
                        f'{fit["sigma_err"]:.1f} GeV\n'
                        f'  $\\chi^2$/ndf = {chi2:.1f}/{ndf}'
                        f' = {chi2_ndf:.2f}'))

    # ---- Reference lines ----
    ax_main.axvline(fit["mu"], color='#D32F2F', ls='--', alpha=0.35, lw=1)
    ax_main.axvline(default_c, color='gray', ls=':', alpha=0.6, lw=1.5,
                    label=f'Default $c$ = {default_c} GeV')

    # ---- Derived parameter annotation ----
    p_val = fit["sigma"] * fit["mu"]
    ax_main.text(0.05, 0.55,
                 f'$p$ = $\\sigma \\times c$ = {p_val:.0f} GeV$^2$',
                 fontsize=10, color='#D32F2F', alpha=0.8,
                 transform=ax_main.transAxes, va='top')

    # ---- Axis styling ----
    ax_main.set_ylabel(f'Weighted events / {bw:.1f} GeV', fontsize=14)
    ax_main.set_xlim(MASS_RANGE_PLOT)
    y_max = np.max(full_y) * 1.55
    ax_main.set_ylim(0, y_max)
    ax_main.legend(fontsize=10, loc='upper right', handlelength=1.8)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- ATLAS label ----
    draw_atlas_label(ax_main, x=0.05, y=0.95, fontsize=14)
    ax_main.text(0.05, 0.68, config_label.replace("_", " / "),
                 fontsize=9, color='gray', transform=ax_main.transAxes)

    # ---- Pull panel ----
    fit_curve_at_bins = gauss(full_x, fit["A"], fit["mu"], fit["sigma"])
    with np.errstate(divide='ignore', invalid='ignore'):
        pull = np.where(full_err > 0,
                        (full_y - fit_curve_at_bins) / full_err, 0.0)

    ax_pull.bar(full_x, pull, width=bw * 0.85, color='steelblue',
                edgecolor='steelblue', alpha=0.7)
    ax_pull.axhline(0, color='black', linewidth=0.8)
    ax_pull.axhline(2, color='gray', ls='--', linewidth=0.6, alpha=0.5)
    ax_pull.axhline(-2, color='gray', ls='--', linewidth=0.6, alpha=0.5)
    ax_pull.set_xlabel(f'{var_label} [GeV]', fontsize=14)
    ax_pull.set_ylabel('Pull', fontsize=12)
    ax_pull.set_ylim(-4, 4)
    ax_pull.set_xlim(MASS_RANGE_PLOT)
    ax_pull.yaxis.set_major_locator(MultipleLocator(2))
    ax_pull.yaxis.set_minor_locator(MultipleLocator(1))

    fname = f"fit_{var_tag}_{config_label}"
    fig.savefig(os.path.join(output_dir, f"{fname}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{fname}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# Plot D2: Standalone S/√B cut scan
# ============================================================

def plot_cut_scan_standalone(scan_results, optimal_cut, config_label,
                             output_dir, fitted_params, has_bkg=True):
    """
    Standalone S/√B vs cut value plot with ATLAS styling.
    Includes fitted SR parameter annotation.
    """
    if not has_bkg:
        print("  Skipping standalone S/√B plot (no background)")
        return

    cuts = np.array([r["cut"] for r in scan_results])
    fom = np.array([r["S/sqrt(B)"] for r in scan_results])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Main curve
    ax.plot(cuts, fom, 'b-', linewidth=2, label=r'S / $\sqrt{B}$')

    # Optimal cut
    fom_at_opt = fom[np.argmin(np.abs(cuts - optimal_cut))]
    ax.axvline(optimal_cut, color='red', ls='--', linewidth=1.5,
               label=f'Optimal cut = {optimal_cut:.2f}'
                     f' (S/$\\sqrt{{B}}$ = {fom_at_opt:.3f})')

    # Default cut
    fom_at_def = fom[np.argmin(np.abs(cuts - DEFAULT_PARAMS["cut"]))]
    ax.axvline(DEFAULT_PARAMS["cut"], color='gray', ls=':', linewidth=1.5,
               label=f'Default cut = {DEFAULT_PARAMS["cut"]:.1f}'
                     f' (S/$\\sqrt{{B}}$ = {fom_at_def:.3f})')

    # Mark optimal point
    ax.plot(optimal_cut, fom_at_opt, 'ro', markersize=8, zorder=5)

    # Axis styling
    ax.set_xlabel('SR cut value $D_{\\mathrm{cut}}$', fontsize=14)
    ax.set_ylabel(r'S / $\sqrt{B}$', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(cuts[0], cuts[-1])
    ax.set_ylim(0, max(fom) * 1.25)

    # ATLAS label
    draw_atlas_label(ax, x=0.05, y=0.95, fontsize=14)

    # SR parameters box
    box_text = (f'$c_1$ = {fitted_params["c1"]:.1f} GeV\n'
                f'$c_2$ = {fitted_params["c2"]:.1f} GeV\n'
                f'$p_1$ = {fitted_params["p1"]:.0f} GeV$^2$\n'
                f'$p_2$ = {fitted_params["p2"]:.0f} GeV$^2$')
    ax.text(0.05, 0.72, box_text, fontsize=10, transform=ax.transAxes,
            va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                      alpha=0.3, edgecolor='gray'))

    # Weight mode annotation
    weight_note = ("full weight" if USE_FULL_WEIGHT
                   else "mc_sf only (optimal cut unchanged)")
    ax.text(0.95, 0.03, weight_note, fontsize=8, color='gray',
            transform=ax.transAxes, ha='right', va='bottom')

    # Config label
    ax.text(0.05, 0.55, config_label.replace("_", " / "),
            fontsize=9, color='gray', transform=ax.transAxes)

    fig.tight_layout()
    fname = f"cut_scan_SoverSqrtB_{config_label}"
    fig.savefig(os.path.join(output_dir, f"{fname}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{fname}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ============================================================
# Plot C: All fits overview
# ============================================================

def plot_all_fits_overview(fit_params, config_label, output_dir):
    """
    Plot all Gaussian fit candidates on top of the data, highlighting
    the best one.  Each candidate drawn in a different colour with its
    fit range and χ²/ndf shown in the legend.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, fit, var_label in [
        (ax1, fit_params["fit1"], r"$m_{H_1}$"),
        (ax2, fit_params["fit2"], r"$m_{H_2}$"),
    ]:
        if fit is None:
            ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='red')
            continue

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

            is_best = (abs(cand["chi2_ndf"] - best_chi2_ndf) < 1e-6
                       and abs(cand["fit_range"][0] - best_range[0]) < 0.1)

            x_c = np.linspace(clo - 5, chi + 5, 300)
            y_c = gauss(x_c, cand["A"], cand["mu"], cand["sigma"])

            lw = 2.5 if is_best else 1.2
            alpha = 1.0 if is_best else 0.6
            ls = "-" if is_best else "--"
            star = " [BEST]" if is_best else ""

            label_str = cand.get("label", f"[{clo:.0f}, {chi:.0f}]")
            ax.plot(x_c, y_c, color=color, linewidth=lw, alpha=alpha, ls=ls,
                    label=(f"{label_str}{star}\n"
                           f"  μ={cand['mu']:.1f}, σ={cand['sigma']:.1f}, "
                           f"χ²/ndf={cand['chi2_ndf']:.2f}"))

            ax.axvline(clo, color=color, ls=":", lw=0.6, alpha=0.3)
            ax.axvline(chi, color=color, ls=":", lw=0.6, alpha=0.3)

        ax.set_xlabel(f"{var_label} [GeV]")
        ax.set_ylabel(f"Weighted events / {bw:.1f} GeV")
        ax.set_xlim(MASS_RANGE_PLOT)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper right", ncol=1)

    draw_atlas_label(ax1, x=0.05, y=0.95, fontsize=13,
                     extra_lines=["All Gaussian fit candidates"])

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir,
                    f"all_fits_overview_{config_label}.{ext}"),
                    bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_fits_overview_{config_label}")


# ============================================================
# Save mass histograms to CSV
# ============================================================

def save_mass_histogram(fit_params, config_label, output_dir):
    """
    Save mass distribution histograms + best-fit Gaussian to CSV files.
    """
    for fit, tag in [(fit_params["fit1"], "mH1"), (fit_params["fit2"], "mH2")]:
        if fit is None:
            continue

        fname = os.path.join(output_dir, f"mass_histogram_{tag}_{config_label}.csv")

        full_x = fit["full_centers"]
        full_y = fit["full_counts"]
        full_err = fit["full_errors"]

        fit_curve = gauss(full_x, fit["A"], fit["mu"], fit["sigma"])

        with np.errstate(divide='ignore', invalid='ignore'):
            pull = np.where(full_err > 0, (full_y - fit_curve) / full_err, 0.0)

        fit_lo, fit_hi = fit.get("fit_range", (full_x[0], full_x[-1]))

        with open(fname, 'w') as f:
            f.write(f"# Mass histogram + Gaussian fit — {tag} — {config_label}\n")
            f.write(f"# mu = {fit['mu']:.4f} +/- {fit['mu_err']:.4f} GeV\n")
            f.write(f"# sigma = {fit['sigma']:.4f} +/- {fit['sigma_err']:.4f} GeV\n")
            f.write(f"# amplitude = {fit['A']:.6f} +/- {fit['A_err']:.6f}\n")
            f.write(f"# chi2 = {fit.get('chi2', 0):.4f}\n")
            f.write(f"# ndf = {fit.get('ndf', 0)}\n")
            f.write(f"# chi2_ndf = {fit.get('chi2_ndf', 0):.4f}\n")
            f.write(f"# fit_range = [{fit_lo:.1f}, {fit_hi:.1f}] GeV\n")
            f.write(f"# nbins = {len(full_x)}\n")
            f.write(f"# bin_width = {full_x[1] - full_x[0]:.2f} GeV\n")
            f.write(f"# weight_mode = {'full (mc_sf × lumi × pileup)' if USE_FULL_WEIGHT else 'mc_sf only'}\n")
            f.write(f"# n_candidates = {len(fit.get('all_candidates', []))}\n")
            f.write(f"#\n")
            f.write("bin_center,counts,error,fit_value,pull\n")
            for i in range(len(full_x)):
                f.write(f"{full_x[i]:.2f},{full_y[i]:.6f},"
                        f"{full_err[i]:.6f},{fit_curve[i]:.6f},{pull[i]:.4f}\n")

        print(f"  Saved histogram: {fname}")


# ============================================================
# Plot: Cut scan
# ============================================================

def plot_cut_scan(scan_results, optimal_cut, config_label, output_dir,
                  has_bkg=True):
    """Plot S/√B and signal efficiency vs cut value — ATLAS style."""
    cuts = [r["cut"] for r in scan_results]
    sig_eff = [r["sig_eff"] for r in scan_results]

    if has_bkg:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fom = [r["S/sqrt(B)"] for r in scan_results]
        sob = [r["S/B"] for r in scan_results]

        # S/√B
        axes[0].plot(cuts, fom, 'b-', linewidth=2)
        axes[0].axvline(optimal_cut, color='red', ls='--', linewidth=1.5,
                        label=f'Optimal: {optimal_cut:.2f}')
        axes[0].axvline(DEFAULT_PARAMS["cut"], color='gray', ls=':',
                        label=f'Default: {DEFAULT_PARAMS["cut"]}')
        axes[0].set_xlabel('SR cut value', fontsize=14)
        axes[0].set_ylabel(r'S / $\sqrt{B}$', fontsize=14)
        axes[0].legend(fontsize=11)

        weight_note = ("full weight" if USE_FULL_WEIGHT
                       else "mc_sf only (optimal cut unchanged)")
        axes[0].text(0.95, 0.05, weight_note, fontsize=8, color='gray',
                     transform=axes[0].transAxes, ha='right', va='bottom')

        # S/B
        axes[1].plot(cuts, sob, 'g-', linewidth=2)
        axes[1].axvline(optimal_cut, color='red', ls='--', linewidth=1.5)
        axes[1].axvline(DEFAULT_PARAMS["cut"], color='gray', ls=':')
        axes[1].set_xlabel('SR cut value', fontsize=14)
        axes[1].set_ylabel('S / B', fontsize=14)

        # Signal efficiency
        axes[2].plot(cuts, sig_eff, color='#D32F2F', linewidth=2)
        axes[2].axvline(optimal_cut, color='red', ls='--', linewidth=1.5)
        axes[2].axvline(DEFAULT_PARAMS["cut"], color='gray', ls=':')
        axes[2].set_xlabel('SR cut value', fontsize=14)
        axes[2].set_ylabel('Signal efficiency', fontsize=14)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        axes[0].plot(cuts, sig_eff, color='#D32F2F', linewidth=2)
        eff_at_opt = sig_eff[np.argmin(np.abs(np.array(cuts) - optimal_cut))]
        axes[0].axvline(optimal_cut, color='red', ls='--', linewidth=1.5,
                        label=f'Selected: {optimal_cut:.2f} '
                              f'(eff = {eff_at_opt:.1%})')
        axes[0].axvline(DEFAULT_PARAMS["cut"], color='gray', ls=':',
                        label=f'Default: {DEFAULT_PARAMS["cut"]}')
        axes[0].set_xlabel('SR cut value', fontsize=14)
        axes[0].set_ylabel('Signal efficiency', fontsize=14)
        axes[0].legend(fontsize=11)

        S_vals = [r["S"] for r in scan_results]
        axes[1].plot(cuts, S_vals, 'b-', linewidth=2)
        axes[1].axvline(optimal_cut, color='red', ls='--', linewidth=1.5)
        axes[1].axvline(DEFAULT_PARAMS["cut"], color='gray', ls=':')
        axes[1].set_xlabel('SR cut value', fontsize=14)
        axes[1].set_ylabel('Signal yield (sum of weights)', fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cut_scan_{config_label}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"cut_scan_{config_label}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Plot: Standalone mass plane (no marginals)
# ============================================================

def plot_massplane_standalone(sig_m1, sig_m2, sig_w,
                              bkg_m1, bkg_m2, bkg_w,
                              fitted_params, config_label, output_dir):
    """
    Clean standalone (m_H1, m_H2) mass plane.
    Signal as 2D colourmap, background as red contours,
    SR ellipses overlaid, no marginal projections.
    """
    has_bkg = bkg_m1 is not None and len(bkg_m1) > 0

    fig, ax = plt.subplots(figsize=(9, 8))

    plot_range = (50, 250)
    nbins_2d = 80

    # ---- 2D signal histogram ----
    h_sig, xedges, yedges = np.histogram2d(
        sig_m1, sig_m2, bins=nbins_2d,
        range=[plot_range, plot_range], weights=sig_w,
    )
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, h_sig.T, cmap='Blues', shading='auto')
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Signal events / bin', fontsize=12)

    # ---- 2D background contours (optional) ----
    if has_bkg and SHOW_BKG_CONTOURS:
        h_bkg, _, _ = np.histogram2d(
            bkg_m1, bkg_m2, bins=nbins_2d,
            range=[plot_range, plot_range], weights=bkg_w,
        )
        bkg_cx = 0.5 * (xedges[:-1] + xedges[1:])
        bkg_cy = 0.5 * (yedges[:-1] + yedges[1:])
        BX, BY = np.meshgrid(bkg_cx, bkg_cy)
        bkg_max = np.max(h_bkg)
        if bkg_max > 0:
            levels = [bkg_max * f for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
            ax.contour(BX, BY, h_bkg.T, levels=levels,
                       colors='#D32F2F', linewidths=0.8, alpha=0.5)

    # ---- SR contours ----
    m1_grid = np.linspace(plot_range[0] + 1, plot_range[1], 400)
    m2_grid = np.linspace(plot_range[0] + 1, plot_range[1], 400)
    M1, M2 = np.meshgrid(m1_grid, m2_grid)

    D_default = sr_discriminant(
        M1, M2,
        DEFAULT_PARAMS["c1"], DEFAULT_PARAMS["c2"],
        DEFAULT_PARAMS["p1"], DEFAULT_PARAMS["p2"])
    ax.contour(M1, M2, D_default,
               levels=[DEFAULT_PARAMS["cut"]],
               colors=['gray'], linewidths=[2], linestyles=['--'])

    D_fitted = sr_discriminant(
        M1, M2,
        fitted_params["c1"], fitted_params["c2"],
        fitted_params["p1"], fitted_params["p2"])
    ax.contour(M1, M2, D_fitted,
               levels=[fitted_params["cut"]],
               colors=['#D32F2F'], linewidths=[2.5], linestyles=['-'])

    # Centre markers
    ax.plot(DEFAULT_PARAMS["c1"], DEFAULT_PARAMS["c2"],
            '+', color='gray', ms=14, mew=2)
    ax.plot(fitted_params["c1"], fitted_params["c2"],
            '+', color='#D32F2F', ms=14, mew=2.5)

    # ---- Labels and legend ----
    ax.set_xlabel(r'$m_{H_1}$ [GeV]', fontsize=15)
    ax.set_ylabel(r'$m_{H_2}$ [GeV]', fontsize=15)
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)

    legend_handles = [
        Line2D([0], [0], color='gray', ls='--', lw=2,
               label=(f'Default SR: $c$=({DEFAULT_PARAMS["c1"]:.0f},'
                      f'{DEFAULT_PARAMS["c2"]:.0f}), '
                      f'cut={DEFAULT_PARAMS["cut"]}')),
        Line2D([0], [0], color='#D32F2F', ls='-', lw=2.5,
               label=(f'Fitted SR: $c$=({fitted_params["c1"]:.1f},'
                      f'{fitted_params["c2"]:.1f}), '
                      f'cut={fitted_params["cut"]:.2f}')),
    ]
    if has_bkg and SHOW_BKG_CONTOURS:
        legend_handles.append(
            Line2D([0], [0], color='#D32F2F', ls='-', lw=0.8, alpha=0.5,
                   label='Background contours'))
    ax.legend(handles=legend_handles, fontsize=9.5, loc='upper right')

    # ---- ATLAS label ----
    draw_atlas_label(ax, x=0.05, y=0.95, fontsize=14)
    ax.text(0.05, 0.88, config_label.replace("_", " / "),
            fontsize=10, color='gray', transform=ax.transAxes)

    # ---- SR parameter box ----
    box_text = (f'$c_1$ = {fitted_params["c1"]:.1f} GeV,  '
                f'$c_2$ = {fitted_params["c2"]:.1f} GeV\n'
                f'$p_1$ = {fitted_params["p1"]:.0f} GeV$^2$,  '
                f'$p_2$ = {fitted_params["p2"]:.0f} GeV$^2$\n'
                f'cut = {fitted_params["cut"]:.2f}')
    ax.text(0.05, 0.78, box_text, fontsize=9.5, transform=ax.transAxes,
            va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                      alpha=0.3, edgecolor='gray'))

    fig.tight_layout()
    fname = f"massplane_{config_label}"
    fig.savefig(os.path.join(output_dir, f"{fname}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{fname}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# Plot: Signal + Background mass plane with marginal histograms
# ============================================================

def plot_massplane_sig_bkg(sig_m1, sig_m2, sig_w,
                           bkg_m1, bkg_m2, bkg_w,
                           fitted_params, config_label, output_dir):
    """
    2D mass plane showing signal and background distributions,
    with 1D marginal histograms on top (m_H1) and right (m_H2).

    Signal: blue colourmap on 2D, blue step histograms on margins.
    Background: shown as contours on 2D, red step histograms on margins.
    SR contour (fitted) overlaid on the 2D plane.
    """
    has_bkg = bkg_m1 is not None and len(bkg_m1) > 0

    fig = plt.figure(figsize=(10, 9.5))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.02, hspace=0.02,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis('off')

    plot_range = (50, 250)
    nbins_2d = 80
    nbins_1d = 60

    # ---- 2D signal histogram ----
    h_sig, xedges, yedges = np.histogram2d(
        sig_m1, sig_m2, bins=nbins_2d,
        range=[plot_range, plot_range], weights=sig_w,
    )
    X, Y = np.meshgrid(xedges, yedges)
    im = ax_main.pcolormesh(X, Y, h_sig.T, cmap='Blues', shading='auto')

    cbar = fig.colorbar(im, ax=ax_corner, fraction=0.9,
                        orientation='vertical', pad=0.05)
    cbar.set_label('Signal events / bin', fontsize=11)

    # ---- 2D background contours (optional) ----
    if has_bkg and SHOW_BKG_CONTOURS:
        h_bkg, _, _ = np.histogram2d(
            bkg_m1, bkg_m2, bins=nbins_2d,
            range=[plot_range, plot_range], weights=bkg_w,
        )
        # Show background as density contours
        bkg_centers_x = 0.5 * (xedges[:-1] + xedges[1:])
        bkg_centers_y = 0.5 * (yedges[:-1] + yedges[1:])
        BX, BY = np.meshgrid(bkg_centers_x, bkg_centers_y)

        # Contour levels at 10%, 30%, 50%, 70%, 90% of max
        bkg_max = np.max(h_bkg)
        if bkg_max > 0:
            levels = [bkg_max * f for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
            ax_main.contour(BX, BY, h_bkg.T, levels=levels,
                            colors='#D32F2F', linewidths=0.8, alpha=0.5)

    # ---- SR contour (fitted) ----
    m1_grid = np.linspace(plot_range[0] + 1, plot_range[1], 400)
    m2_grid = np.linspace(plot_range[0] + 1, plot_range[1], 400)
    M1, M2 = np.meshgrid(m1_grid, m2_grid)

    D_default = sr_discriminant(
        M1, M2,
        DEFAULT_PARAMS["c1"], DEFAULT_PARAMS["c2"],
        DEFAULT_PARAMS["p1"], DEFAULT_PARAMS["p2"])
    ax_main.contour(M1, M2, D_default,
                    levels=[DEFAULT_PARAMS["cut"]],
                    colors=['gray'], linewidths=[1.8], linestyles=['--'])

    D_fitted = sr_discriminant(
        M1, M2,
        fitted_params["c1"], fitted_params["c2"],
        fitted_params["p1"], fitted_params["p2"])
    ax_main.contour(M1, M2, D_fitted,
                    levels=[fitted_params["cut"]],
                    colors=['#D32F2F'], linewidths=[2.5], linestyles=['-'])

    # Centre markers
    ax_main.plot(fitted_params["c1"], fitted_params["c2"],
                 '+', color='#D32F2F', ms=14, mew=2.5)
    ax_main.plot(DEFAULT_PARAMS["c1"], DEFAULT_PARAMS["c2"],
                 '+', color='gray', ms=12, mew=2)

    ax_main.set_xlabel(r'$m_{H_1}$ [GeV]', fontsize=14)
    ax_main.set_ylabel(r'$m_{H_2}$ [GeV]', fontsize=14)
    ax_main.set_xlim(plot_range)
    ax_main.set_ylim(plot_range)

    # Legend
    legend_handles = [
        Line2D([0], [0], color='gray', ls='--', lw=1.8,
               label=f'Default SR (cut={DEFAULT_PARAMS["cut"]})'),
        Line2D([0], [0], color='#D32F2F', ls='-', lw=2.5,
               label=(f'Fitted SR (cut={fitted_params["cut"]:.2f})')),
    ]
    if has_bkg and SHOW_BKG_CONTOURS:
        legend_handles.append(
            Line2D([0], [0], color='#D32F2F', ls='-', lw=0.8, alpha=0.5,
                   label='Background contours'))
    ax_main.legend(handles=legend_handles, fontsize=9, loc='upper right')

    # ---- Top marginal: m_H1 ----
    proj_edges = np.linspace(plot_range[0], plot_range[1], nbins_1d + 1)
    bw = proj_edges[1] - proj_edges[0]

    sig_counts_x, _ = np.histogram(sig_m1, bins=proj_edges, weights=sig_w)
    sig_sumw2_x, _ = np.histogram(sig_m1, bins=proj_edges, weights=sig_w**2)
    centers_x = 0.5 * (proj_edges[:-1] + proj_edges[1:])
    sig_err_x = np.sqrt(sig_sumw2_x)

    _draw_step_histogram(ax_top, centers_x, sig_counts_x, bw,
                         color='steelblue', linewidth=1.5, label='Signal')
    ax_top.errorbar(centers_x, sig_counts_x, yerr=sig_err_x,
                    fmt='none', ecolor='steelblue', elinewidth=0.8, capsize=1.5)

    if has_bkg:
        bkg_counts_x, _ = np.histogram(bkg_m1, bins=proj_edges, weights=bkg_w)
        bkg_sumw2_x, _ = np.histogram(bkg_m1, bins=proj_edges, weights=bkg_w**2)
        bkg_err_x = np.sqrt(bkg_sumw2_x)

        # Scale background for visual comparison
        sig_max = np.max(sig_counts_x) if np.max(sig_counts_x) > 0 else 1
        bkg_max_x = np.max(bkg_counts_x) if np.max(bkg_counts_x) > 0 else 1
        bkg_scale = sig_max / bkg_max_x * 0.8

        _draw_step_histogram(ax_top, centers_x, bkg_counts_x * bkg_scale, bw,
                             color='#D32F2F', linewidth=1.2, linestyle='--',
                             label=f'Background ($\\times${bkg_scale:.1e})')
        ax_top.legend(fontsize=8, loc='upper right')

    ax_top.axvline(fitted_params["c1"], color='#D32F2F', ls='--', lw=1, alpha=0.5)
    ax_top.axvline(DEFAULT_PARAMS["c1"], color='gray', ls=':', lw=1, alpha=0.5)
    ax_top.set_ylabel(f'Events / {bw:.1f} GeV', fontsize=11)
    ax_top.set_xlim(plot_range)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    draw_atlas_label(ax_top, x=0.03, y=0.90, fontsize=12)
    ax_top.text(0.03, 0.68, config_label.replace("_", " / "),
                fontsize=9, color='gray', transform=ax_top.transAxes)

    # ---- Right marginal: m_H2 (horizontal) ----
    sig_counts_y, _ = np.histogram(sig_m2, bins=proj_edges, weights=sig_w)
    sig_sumw2_y, _ = np.histogram(sig_m2, bins=proj_edges, weights=sig_w**2)
    centers_y = 0.5 * (proj_edges[:-1] + proj_edges[1:])
    sig_err_y = np.sqrt(sig_sumw2_y)

    _draw_step_histogram(ax_right, centers_y, sig_counts_y, bw,
                         orientation="horizontal",
                         color='steelblue', linewidth=1.5)
    ax_right.errorbar(sig_counts_y, centers_y, xerr=sig_err_y,
                      fmt='none', ecolor='steelblue', elinewidth=0.8, capsize=1.5)

    if has_bkg:
        bkg_counts_y, _ = np.histogram(bkg_m2, bins=proj_edges, weights=bkg_w)
        bkg_sumw2_y, _ = np.histogram(bkg_m2, bins=proj_edges, weights=bkg_w**2)
        bkg_err_y = np.sqrt(bkg_sumw2_y)

        sig_max_y = np.max(sig_counts_y) if np.max(sig_counts_y) > 0 else 1
        bkg_max_y = np.max(bkg_counts_y) if np.max(bkg_counts_y) > 0 else 1
        bkg_scale_y = sig_max_y / bkg_max_y * 0.8

        _draw_step_histogram(ax_right, centers_y, bkg_counts_y * bkg_scale_y, bw,
                             orientation="horizontal",
                             color='#D32F2F', linewidth=1.2, linestyle='--')

    ax_right.axhline(fitted_params["c2"], color='#D32F2F', ls='--', lw=1, alpha=0.5)
    ax_right.axhline(DEFAULT_PARAMS["c2"], color='gray', ls=':', lw=1, alpha=0.5)
    ax_right.set_xlabel(f'Events / {bw:.1f} GeV', fontsize=11)
    ax_right.set_ylim(plot_range)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fname = f"massplane_sig_bkg_{config_label}"
    fig.savefig(os.path.join(output_dir, f"{fname}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{fname}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# Plot: Optimised mass plane with marginal projections
# ============================================================

def plot_optimised_massplane(sig_m1, sig_m2, sig_w,
                             default_params, fitted_params, joint_params,
                             config_label, output_dir):
    """
    2D mass plane with marginal 1D projections on top and right.
    """
    fig = plt.figure(figsize=(10, 9.5))

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.02, hspace=0.02,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis('off')

    # ---- 2D histogram ----
    nbins_2d = 80
    h, xedges, yedges = np.histogram2d(
        sig_m1, sig_m2, bins=nbins_2d,
        range=[MASS_RANGE, MASS_RANGE], weights=sig_w,
    )
    X, Y = np.meshgrid(xedges, yedges)
    im = ax_main.pcolormesh(X, Y, h.T, cmap='Blues', shading='auto')

    cbar = fig.colorbar(im, ax=ax_corner, fraction=0.9,
                        orientation='vertical', pad=0.05)
    cbar.set_label('Events / bin', fontsize=11)

    # ---- Contour grid ----
    m1_grid = np.linspace(MASS_RANGE[0] + 1, MASS_RANGE[1], 400)
    m2_grid = np.linspace(MASS_RANGE[0] + 1, MASS_RANGE[1], 400)
    M1, M2 = np.meshgrid(m1_grid, m2_grid)

    # Default SR contour
    D_default = sr_discriminant(
        M1, M2,
        default_params["c1"], default_params["c2"],
        default_params["p1"], default_params["p2"])
    ax_main.contour(M1, M2, D_default,
                    levels=[default_params["cut"]],
                    colors=['gray'], linewidths=[2], linestyles=['--'])

    # Fitted SR contour
    D_fitted = sr_discriminant(
        M1, M2,
        fitted_params["c1"], fitted_params["c2"],
        fitted_params["p1"], fitted_params["p2"])
    ax_main.contour(M1, M2, D_fitted,
                    levels=[fitted_params["cut"]],
                    colors=['#D32F2F'], linewidths=[2.5], linestyles=['-'])

    # Joint-optimised SR contour
    if joint_params is not None:
        D_joint = sr_discriminant(
            M1, M2,
            joint_params["c1"], joint_params["c2"],
            joint_params["p1"], joint_params["p2"])
        ax_main.contour(M1, M2, D_joint,
                        levels=[joint_params["cut"]],
                        colors=['#2E7D32'], linewidths=[2], linestyles=['-.'])

    # Centre markers
    ax_main.plot(default_params["c1"], default_params["c2"],
                 '+', color='gray', ms=14, mew=2)
    ax_main.plot(fitted_params["c1"], fitted_params["c2"],
                 '+', color='#D32F2F', ms=14, mew=2)
    if joint_params is not None:
        ax_main.plot(joint_params["c1"], joint_params["c2"],
                     '+', color='#2E7D32', ms=14, mew=2)

    ax_main.set_xlabel(r'$m_{H_1}$ [GeV]', fontsize=14)
    ax_main.set_ylabel(r'$m_{H_2}$ [GeV]', fontsize=14)
    ax_main.set_xlim(MASS_RANGE)
    ax_main.set_ylim(MASS_RANGE)

    # Legend
    legend_handles = [
        Line2D([0], [0], color='gray', ls='--', lw=2,
               label=(f'Default: $c$=({default_params["c1"]:.0f},'
                      f'{default_params["c2"]:.0f}), '
                      f'cut={default_params["cut"]}')),
        Line2D([0], [0], color='#D32F2F', ls='-', lw=2.5,
               label=(f'Fitted: $c$=({fitted_params["c1"]:.1f},'
                      f'{fitted_params["c2"]:.1f}), '
                      f'cut={fitted_params["cut"]:.2f}')),
    ]
    if joint_params is not None:
        legend_handles.append(
            Line2D([0], [0], color='#2E7D32', ls='-.', lw=2,
                   label=(f'Joint opt: $c$=({joint_params["c1"]:.1f},'
                          f'{joint_params["c2"]:.1f}), '
                          f'cut={joint_params["cut"]:.2f}')))
    ax_main.legend(handles=legend_handles, fontsize=8.5, loc='upper right')

    # ---- Top marginal: m_H1 projection ----
    nbins_1d = 60
    proj_edges_x = np.linspace(MASS_RANGE[0], MASS_RANGE[1], nbins_1d + 1)
    proj_counts_x, _ = np.histogram(sig_m1, bins=proj_edges_x, weights=sig_w)
    proj_sumw2_x, _ = np.histogram(sig_m1, bins=proj_edges_x, weights=sig_w**2)
    proj_centers_x = 0.5 * (proj_edges_x[:-1] + proj_edges_x[1:])
    proj_err_x = np.sqrt(proj_sumw2_x)
    bw_x = proj_centers_x[1] - proj_centers_x[0]

    _draw_step_histogram(ax_top, proj_centers_x, proj_counts_x, bw_x,
                         color='black', linewidth=1.3)
    ax_top.errorbar(proj_centers_x, proj_counts_x, yerr=proj_err_x,
                    fmt='none', ecolor='black', elinewidth=0.8, capsize=1.5)

    ax_top.axvline(default_params["c1"], color='gray', ls=':', lw=1.2, alpha=0.6)
    ax_top.axvline(fitted_params["c1"], color='#D32F2F', ls='--', lw=1.2, alpha=0.6)
    if joint_params is not None:
        ax_top.axvline(joint_params["c1"], color='#2E7D32', ls='-.', lw=1.2, alpha=0.6)

    ax_top.set_ylabel('Events', fontsize=11)
    ax_top.set_xlim(MASS_RANGE)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    draw_atlas_label(ax_top, x=0.03, y=0.90, fontsize=12)

    # ---- Right marginal: m_H2 projection (horizontal step) ----
    proj_edges_y = np.linspace(MASS_RANGE[0], MASS_RANGE[1], nbins_1d + 1)
    proj_counts_y, _ = np.histogram(sig_m2, bins=proj_edges_y, weights=sig_w)
    proj_sumw2_y, _ = np.histogram(sig_m2, bins=proj_edges_y, weights=sig_w**2)
    proj_centers_y = 0.5 * (proj_edges_y[:-1] + proj_edges_y[1:])
    proj_err_y = np.sqrt(proj_sumw2_y)
    bw_y = proj_centers_y[1] - proj_centers_y[0]

    _draw_step_histogram(ax_right, proj_centers_y, proj_counts_y, bw_y,
                         orientation="horizontal",
                         color='black', linewidth=1.3)
    ax_right.errorbar(proj_counts_y, proj_centers_y, xerr=proj_err_y,
                      fmt='none', ecolor='black', elinewidth=0.8, capsize=1.5)

    ax_right.axhline(default_params["c2"], color='gray', ls=':', lw=1.2, alpha=0.6)
    ax_right.axhline(fitted_params["c2"], color='#D32F2F', ls='--', lw=1.2, alpha=0.6)
    if joint_params is not None:
        ax_right.axhline(joint_params["c2"], color='#2E7D32', ls='-.', lw=1.2, alpha=0.6)

    ax_right.set_xlabel('Events', fontsize=11)
    ax_right.set_ylim(MASS_RANGE)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.savefig(os.path.join(output_dir, f"massplane_optimised_{config_label}.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"massplane_optimised_{config_label}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: massplane_optimised_{config_label}")


# ============================================================
# Plot: Comparison summary
# ============================================================

def plot_comparison_summary(all_results, output_dir):
    """Bar charts of c1, c2, sigma1, sigma2, p1, p2, cut, signal eff."""
    valid = [r for r in all_results if r is not None and r["status"] == "OK"]
    if not valid:
        print("No valid results to compare")
        return

    labels = [r["config"] for r in valid]
    n = len(labels)
    x = np.arange(n)

    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)

    plot_specs = [
        (gs[0, 0], "c1", '$c_1$ [GeV]', 'Leading Higgs mass centre',
         'steelblue', DEFAULT_PARAMS["c1"]),
        (gs[0, 1], "c2", '$c_2$ [GeV]', 'Sub-leading Higgs mass centre',
         'steelblue', DEFAULT_PARAMS["c2"]),
        (gs[1, 0], "sigma1", '$\\sigma_1$ [GeV]', 'Leading Higgs mass resolution',
         'coral', DEFAULT_PARAMS["p1"] / DEFAULT_PARAMS["c1"]),
        (gs[1, 1], "sigma2", '$\\sigma_2$ [GeV]', 'Sub-leading Higgs mass resolution',
         'coral', DEFAULT_PARAMS["p2"] / DEFAULT_PARAMS["c2"]),
        (gs[2, 0], "p1", '$p_1$ [GeV²]', 'Resolution parameter p1',
         'goldenrod', DEFAULT_PARAMS["p1"]),
        (gs[2, 1], "p2", '$p_2$ [GeV²]', 'Resolution parameter p2',
         'goldenrod', DEFAULT_PARAMS["p2"]),
        (gs[3, 0], "cut", 'Optimal cut value', 'SR cut value',
         'mediumseagreen', DEFAULT_PARAMS["cut"]),
    ]

    for spec, key, ylabel, title, color, default_val in plot_specs:
        ax = fig.add_subplot(spec)
        vals = [r["fitted"][key] for r in valid]
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.axhline(default_val, color='red', ls='--',
                   label=f'Default: {default_val:.1f}')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Signal efficiency
    ax = fig.add_subplot(gs[3, 1])
    vals_opt = [r["sig_eff_at_optimal"] for r in valid]
    vals_def = [r.get("sig_eff_at_default", 0) for r in valid]
    w = 0.35
    ax.bar(x - w / 2, vals_def, w, color='gray', alpha=0.7, label='Default SR')
    ax.bar(x + w / 2, vals_opt, w, color='mediumseagreen', alpha=0.8,
           label='Optimised SR')
    ax.set_ylabel('Signal efficiency')
    ax.set_title('Signal efficiency in SR')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('SR Optimisation Comparison — All Configurations',
                 fontsize=15, y=1.01)
    fig.savefig(os.path.join(output_dir, "sr_optimisation_comparison.pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "sr_optimisation_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Save results
# ============================================================

def save_results_table(all_results, output_file):
    """Save a text comparison table of all optimisation results."""
    valid = [r for r in all_results if r is not None]

    with open(output_file, 'w') as f:
        f.write("=" * 160 + "\n")
        f.write("SR Optimisation Results — VBF Boosted HH → 4b\n")
        f.write(f"Weight mode: {'mc_sf × lumi × pileup' if USE_FULL_WEIGHT else 'mc_sf only (shape)'}\n")
        f.write("=" * 160 + "\n\n")

        f.write(f"Default parameters: c1={DEFAULT_PARAMS['c1']}, c2={DEFAULT_PARAMS['c2']}, "
                f"p1={DEFAULT_PARAMS['p1']}, p2={DEFAULT_PARAMS['p2']}, "
                f"cut={DEFAULT_PARAMS['cut']}\n\n")

        f.write(f"{'Config':<35} {'c1':>7} {'c2':>7} {'σ1':>7} {'σ2':>7} "
                f"{'p1':>8} {'p2':>8} {'cut':>7} {'sig_eff':>9} "
                f"{'Δc1':>7} {'Δc2':>7} {'Δcut':>7} {'status':>8}\n")
        f.write("-" * 160 + "\n")

        for r in valid:
            if r["status"] != "OK":
                f.write(f"  {r['config']:<33} {'—':>7} {'—':>7} {'—':>7} {'—':>7} "
                        f"{'—':>8} {'—':>8} {'—':>7} {'—':>9} "
                        f"{'—':>7} {'—':>7} {'—':>7} {r['status']:>8}\n")
                continue

            fp = r["fitted"]
            dc1 = fp["c1"] - DEFAULT_PARAMS["c1"]
            dc2 = fp["c2"] - DEFAULT_PARAMS["c2"]
            dcut = fp["cut"] - DEFAULT_PARAMS["cut"]

            f.write(f"  {r['config']:<33} {fp['c1']:>7.1f} {fp['c2']:>7.1f} "
                    f"{fp['sigma1']:>7.1f} {fp['sigma2']:>7.1f} "
                    f"{fp['p1']:>8.0f} {fp['p2']:>8.0f} "
                    f"{fp['cut']:>7.2f} {r['sig_eff_at_optimal']:>8.1%} "
                    f"{dc1:>+7.1f} {dc2:>+7.1f} {dcut:>+7.2f} {'OK':>8}\n")

        any_joint = any(r.get("joint") is not None for r in valid)
        if any_joint:
            f.write("\n\nJoint 5-parameter optimisation:\n")
            f.write("-" * 160 + "\n")
            f.write(f"{'Config':<35} {'c1':>7} {'c2':>7} {'σ1':>7} {'σ2':>7} "
                    f"{'p1':>8} {'p2':>8} {'cut':>7} {'S/√B':>9}\n")
            f.write("-" * 160 + "\n")

            for r in valid:
                jp = r.get("joint")
                if jp is None:
                    continue
                f.write(f"  {r['config']:<33} {jp['c1']:>7.1f} {jp['c2']:>7.1f} "
                        f"{jp['sigma1']:>7.1f} {jp['sigma2']:>7.1f} "
                        f"{jp['p1']:>8.0f} {jp['p2']:>8.0f} "
                        f"{jp['cut']:>7.2f} {jp['S/sqrt(B)']:>9.3f}\n")

        f.write("\n" + "=" * 160 + "\n")

    print(f"\nResults saved to: {output_file}")


def save_results_json(all_results, output_file):
    """Save all results as JSON for further processing."""
    serialisable = []
    for r in all_results:
        if r is None:
            continue
        entry = {
            "config": r["config"],
            "mc_campaign": r["mc_campaign"],
            "tagger": r["tagger"],
            "jet_mode": r["jet_mode"],
            "status": r["status"],
            "weight_mode": "full" if USE_FULL_WEIGHT else "mc_sf_only",
        }
        if r["status"] == "OK":
            entry["fitted"] = {
                "c1": r["fitted"]["c1"], "c2": r["fitted"]["c2"],
                "sigma1": r["fitted"]["sigma1"], "sigma2": r["fitted"]["sigma2"],
                "p1": r["fitted"]["p1"], "p2": r["fitted"]["p2"],
                "cut": r["fitted"]["cut"],
            }
            entry["sig_eff_at_optimal"] = r["sig_eff_at_optimal"]
            entry["sig_eff_at_default"] = r.get("sig_eff_at_default", None)
            if r.get("joint") is not None:
                entry["joint"] = {
                    "c1": r["joint"]["c1"], "c2": r["joint"]["c2"],
                    "p1": r["joint"]["p1"], "p2": r["joint"]["p2"],
                    "cut": r["joint"]["cut"],
                    "S/sqrt(B)": r["joint"]["S/sqrt(B)"],
                }
        serialisable.append(entry)

    with open(output_file, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"JSON results saved to: {output_file}")


# ============================================================
# Main optimisation for a single configuration
# ============================================================

def _read_signal_from_file(filepath):
    """Read signal arrays from a single ROOT file. Returns (m1, m2, w) or None."""
    tf = root.TFile.Open(filepath)
    if not tf or tf.IsZombie():
        print(f"  ERROR: Could not open {filepath}")
        return None

    tree = tf.Get("ttree")
    if not tree:
        print(f"  ERROR: No ttree found in {filepath}")
        tf.Close()
        return None

    print(f"  Entries: {tree.GetEntries()}")

    sig_data = read_tree_arrays(tree, [MASS_VAR_H1, MASS_VAR_H2],
                                SELECTION_CUT, WEIGHT_VAR,
                                full_weight=USE_FULL_WEIGHT)
    tf.Close()

    if sig_data is None:
        return None

    return (sig_data[MASS_VAR_H1], sig_data[MASS_VAR_H2], sig_data[WEIGHT_VAR])


# Campaigns to merge when running "Combined"
COMBINED_CAMPAIGNS = ["MC20", "MC23"]


def optimise_single_config(mc_campaign, tagger, jet_mode,
                           base_dir, bkg_dir=None,
                           output_dir=".", do_joint=False,
                           signal_label="Kappa2V",
                           signal_coupling="l1cvv0cv1",
                           bkg_processes=None):
    """
    Run the full SR optimisation for one (MC, tagger, jet_mode) configuration.
    """
    config_label = f"{mc_campaign}_{tagger}_{jet_mode}"
    print(f"\n{'=' * 70}")
    print(f"Optimising: {config_label} [{signal_label}]")
    print(f"Weight mode: {'full (mc_sf × lumi × pileup)' if USE_FULL_WEIGHT else 'mc_sf only (shape)'}")
    print(f"{'=' * 70}")

    result = {
        "config": config_label,
        "mc_campaign": mc_campaign,
        "tagger": tagger,
        "jet_mode": jet_mode,
        "status": "FAILED",
    }

    # --- Read signal ---
    if mc_campaign == "Combined":
        all_m1, all_m2, all_w = [], [], []
        for sub_mc in COMBINED_CAMPAIGNS:
            sig_file = find_signal_file(base_dir, sub_mc, tagger, jet_mode,
                                        signal_label=signal_label,
                                        signal_coupling=signal_coupling)
            if sig_file is None:
                print(f"  WARNING: No signal file for {sub_mc}_{tagger}_{jet_mode}")
                continue
            print(f"  Signal [{sub_mc}]: {sig_file}")
            arrays = _read_signal_from_file(sig_file)
            if arrays is not None:
                m1, m2, w = arrays
                all_m1.append(m1)
                all_m2.append(m2)
                all_w.append(w)
                print(f"    Events after selection: {len(m1)}")

        if not all_m1:
            print(f"  ERROR: No signal events from any sub-campaign")
            return result

        sig_m1 = np.concatenate(all_m1)
        sig_m2 = np.concatenate(all_m2)
        sig_w = np.concatenate(all_w)
        print(f"  Combined signal events: {len(sig_m1)}")

    else:
        sig_file = find_signal_file(base_dir, mc_campaign, tagger, jet_mode,
                                    signal_label=signal_label,
                                    signal_coupling=signal_coupling)
        if sig_file is None:
            print(f"  ERROR: No signal file found")
            return result
        print(f"  Signal: {sig_file}")

        arrays = _read_signal_from_file(sig_file)
        if arrays is None:
            print(f"  ERROR: No events pass selection")
            return result

        sig_m1, sig_m2, sig_w = arrays
        print(f"  Signal events after selection: {len(sig_m1)}")

    # --- Read background (if available) ---
    #   Background lives in the same directory tree as signal.
    #   Everything matching the tagger/mode that isn't the signal sample
    #   is treated as background.
    bkg_m1, bkg_m2, bkg_w = None, None, None

    bkg_campaigns = COMBINED_CAMPAIGNS if mc_campaign == "Combined" else [mc_campaign]
    all_bkg_files = []
    bkg_search_dir = bkg_dir if bkg_dir is not None else base_dir
    for sub_mc in bkg_campaigns:
        all_bkg_files.extend(
            find_background_files(bkg_search_dir, sub_mc, tagger, jet_mode,
                                  bkg_processes=bkg_processes)
        )

    if all_bkg_files:
        all_bkg_m1, all_bkg_m2, all_bkg_w = [], [], []
        for bf, proc in all_bkg_files:
            print(f"  Background [{proc}]: {bf}")
            tf_bkg = root.TFile.Open(bf)
            if not tf_bkg or tf_bkg.IsZombie():
                continue
            tree_bkg = tf_bkg.Get("ttree")
            if not tree_bkg:
                tf_bkg.Close()
                continue
            bkg_data = read_tree_arrays(tree_bkg, [MASS_VAR_H1, MASS_VAR_H2],
                                        SELECTION_CUT, WEIGHT_VAR,
                                        full_weight=USE_FULL_WEIGHT)
            tf_bkg.Close()
            if bkg_data is not None:
                all_bkg_m1.append(bkg_data[MASS_VAR_H1])
                all_bkg_m2.append(bkg_data[MASS_VAR_H2])
                all_bkg_w.append(bkg_data[WEIGHT_VAR])

        if all_bkg_m1:
            bkg_m1 = np.concatenate(all_bkg_m1)
            bkg_m2 = np.concatenate(all_bkg_m2)
            bkg_w = np.concatenate(all_bkg_w)
            print(f"  Total background events: {len(bkg_m1)}")
    else:
        print("  INFO: No background files found — will optimise using signal only")

    has_bkg = bkg_m1 is not None and len(bkg_m1) > 0

    # --- Step 1 & 2: Fit mass peaks ---
    fit_params = fit_signal_parameters(sig_m1, sig_m2, sig_w)

    # --- Step 3: Scan cut values ---
    print(f"\n  Scanning cut values [{CUT_SCAN_MIN}, {CUT_SCAN_MAX}] "
          f"with step {CUT_SCAN_STEP}...")
    scan_results = scan_cut_values(
        sig_m1, sig_m2, sig_w, bkg_m1, bkg_m2, bkg_w,
        fit_params["c1"], fit_params["c2"],
        fit_params["p1"], fit_params["p2"],
    )

    optimal_cut = find_optimal_cut(scan_results,
                                   metric="S/sqrt(B)" if has_bkg else "sig_eff")

    # Signal efficiency at default and optimal cut
    sig_D_default = sr_discriminant(sig_m1, sig_m2,
                                    DEFAULT_PARAMS["c1"], DEFAULT_PARAMS["c2"],
                                    DEFAULT_PARAMS["p1"], DEFAULT_PARAMS["p2"])
    sig_eff_default = np.sum(sig_w[sig_D_default < DEFAULT_PARAMS["cut"]]) / np.sum(sig_w)

    sig_D_fitted = sr_discriminant(sig_m1, sig_m2,
                                   fit_params["c1"], fit_params["c2"],
                                   fit_params["p1"], fit_params["p2"])
    sig_eff_optimal = np.sum(sig_w[sig_D_fitted < optimal_cut]) / np.sum(sig_w)

    print(f"\n  Optimal cut: {optimal_cut:.2f}")
    print(f"  Signal eff at default SR: {sig_eff_default:.1%}")
    print(f"  Signal eff at optimised SR: {sig_eff_optimal:.1%}")

    fitted = {
        "c1": fit_params["c1"], "c2": fit_params["c2"],
        "sigma1": fit_params["sigma1"], "sigma2": fit_params["sigma2"],
        "p1": fit_params["p1"], "p2": fit_params["p2"],
        "cut": optimal_cut,
    }

    # --- Step 4: Joint optimisation (optional) ---
    joint_result = None
    if do_joint and has_bkg:
        joint_result = joint_optimization(
            sig_m1, sig_m2, sig_w, bkg_m1, bkg_m2, bkg_w, fitted,
        )

    # --- Plots ---
    cfg_output = os.path.join(output_dir, f"SR_Optimisation_{config_label}")
    os.makedirs(cfg_output, exist_ok=True)

    if fit_params["fit1"] is not None:
        plot_raw_distributions(fit_params, config_label, cfg_output)
        plot_mass_fits(fit_params, config_label, cfg_output)
        plot_all_fits_overview(fit_params, config_label, cfg_output)
        save_mass_histogram(fit_params, config_label, cfg_output)

        # Standalone individual fit plots
        plot_single_mass_fit(fit_params["fit1"], r"$m_{H_1}$", "mH1",
                             DEFAULT_PARAMS["c1"], fit_params,
                             config_label, cfg_output)
        plot_single_mass_fit(fit_params["fit2"], r"$m_{H_2}$", "mH2",
                             DEFAULT_PARAMS["c2"], fit_params,
                             config_label, cfg_output)

    plot_cut_scan(scan_results, optimal_cut, config_label, cfg_output,
                  has_bkg=has_bkg)

    # Standalone S/√B cut scan
    plot_cut_scan_standalone(scan_results, optimal_cut, config_label,
                             cfg_output, fitted, has_bkg=has_bkg)

    # Signal + background mass plane with marginal histograms
    plot_massplane_sig_bkg(sig_m1, sig_m2, sig_w,
                           bkg_m1, bkg_m2, bkg_w,
                           fitted, config_label, cfg_output)

    # Standalone mass plane (no marginals)
    plot_massplane_standalone(sig_m1, sig_m2, sig_w,
                              bkg_m1, bkg_m2, bkg_w,
                              fitted, config_label, cfg_output)

    plot_optimised_massplane(sig_m1, sig_m2, sig_w,
                             DEFAULT_PARAMS, fitted, joint_result,
                             config_label, cfg_output)

    # --- Collect result ---
    result["status"] = fit_params["status"]
    result["fitted"] = fitted
    result["joint"] = joint_result
    result["sig_eff_at_optimal"] = sig_eff_optimal
    result["sig_eff_at_default"] = sig_eff_default
    result["scan_results"] = scan_results

    return result


# ============================================================
# Main
# ============================================================

def main():
    global SELECTION_CUT, WEIGHT_VAR, LUMI_VAR, PILEUP_VAR, USE_FULL_WEIGHT, SHOW_BKG_CONTOURS

    parser = argparse.ArgumentParser(
        description="SR Optimisation for VBF Boosted HH → 4b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weight modes:
  Default (--weight-var mc_sf):
    Uses mc_sf only [fb]. Sufficient for fitting (shape) and SR cut
    optimisation (luminosity cancels in S/√B). This is the recommended
    mode — optimal SR parameters are identical to full-weight mode.

  Full weight (--full-weight):
    Reads mc_sf × luminosity_boosted × pileupweight_boosted.
    Needed for absolute S/√B numbers at a given luminosity.
    Optimal SR parameters are IDENTICAL; only absolute yields change.
"""
    )
    parser.add_argument("--base-dir", default=".",
                        help="Base directory with signal sample folders")
    parser.add_argument("--bkg-dir", default=None,
                        help="Directory with background sample folders. "
                             "Defaults to --base-dir (same tree as signal). "
                             "Background = all folders matching tagger/mode "
                             "that are NOT the signal sample.")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for plots and tables")
    parser.add_argument("--mc-campaigns", nargs="+", default=MC_CAMPAIGNS,
                        help="MC campaigns to process")
    parser.add_argument("--taggers", nargs="+", default=TAGGERS,
                        help="Taggers to process")
    parser.add_argument("--jet-modes", nargs="+", default=JET_MODES,
                        help="Jet kinematics modes to process")
    parser.add_argument("--do-joint", action="store_true",
                        help="Run joint 5-parameter optimisation")
    parser.add_argument("--selection", default=SELECTION_CUT,
                        help="Selection cut string")
    parser.add_argument("--weight-var", default=WEIGHT_VAR,
                        help=f"Weight branch name (default: {WEIGHT_VAR})")
    parser.add_argument("--full-weight", action="store_true",
                        help="Use full physics weight: mc_sf × luminosity_boosted × "
                             "pileupweight_boosted. Default is mc_sf only (sufficient "
                             "for optimisation since luminosity cancels in S/√B).")
    parser.add_argument("--show-bkg-contours", action="store_true",
                        help="Draw background density contours (10%%-90%% of peak) "
                             "on mass plane plots. Off by default.")
    parser.add_argument("--signal-label", default="Kappa2V",
                        help="Signal sample folder prefix (default: Kappa2V = κ_2V=0, "
                             "l1cvv0cv1). Also accepts: SM, quad_M0_S1, etc.")
    parser.add_argument("--signal-coupling", default="l1cvv0cv1",
                        help="Coupling tag in the ROOT filename to select "
                             "(default: l1cvv0cv1 = κ_2V=0). "
                             "Use l1cvv1cv1 for SM.")
    parser.add_argument("--bkg-processes", nargs="+",
                        default=["QCD", "ttbar"],
                        help="Background process folder prefixes to include "
                             "(default: QCD ttbar). All files from matching "
                             "folders are merged into one background sample.")
    parser.add_argument("--lumi-var", default=LUMI_VAR,
                        help=f"Luminosity branch name (default: {LUMI_VAR})")
    parser.add_argument("--pileup-var", default=PILEUP_VAR,
                        help=f"Pileup weight branch name (default: {PILEUP_VAR})")
    args = parser.parse_args()

    # Update module-level variables
    SELECTION_CUT = args.selection
    WEIGHT_VAR = args.weight_var
    LUMI_VAR = args.lumi_var
    PILEUP_VAR = args.pileup_var
    USE_FULL_WEIGHT = args.full_weight
    SHOW_BKG_CONTOURS = args.show_bkg_contours

    print("=" * 70)
    print("SR Optimisation — VBF Boosted HH → 4b (v2)")
    print("=" * 70)
    print(f"Base dir:    {args.base_dir}")
    print(f"Bkg dir:     {args.bkg_dir if args.bkg_dir else '(same as base-dir)'}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Selection:   {SELECTION_CUT}")
    print(f"Weight:      {WEIGHT_VAR}" +
          (f" × {LUMI_VAR} × {PILEUP_VAR}" if USE_FULL_WEIGHT else " (shape only)"))
    print(f"Joint opt:   {args.do_joint}")
    print(f"Campaigns:   {args.mc_campaigns}")
    print(f"Taggers:     {args.taggers}")
    print(f"Jet modes:   {args.jet_modes}")
    print(f"Signal:      {args.signal_label} (coupling: {args.signal_coupling})")
    print(f"Background:  {args.bkg_processes}")

    if not USE_FULL_WEIGHT:
        print(f"\n  NOTE: Using mc_sf only. Optimal SR parameters are independent")
        print(f"        of luminosity. Use --full-weight for absolute S/√B values.")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for mc in args.mc_campaigns:
        for tagger_arg in args.taggers:
            for jet_mode in args.jet_modes:
                result = optimise_single_config(
                    mc, tagger_arg, jet_mode,
                    args.base_dir, args.bkg_dir,
                    args.output_dir, args.do_joint,
                    signal_label=args.signal_label,
                    signal_coupling=args.signal_coupling,
                    bkg_processes=args.bkg_processes,
                )
                all_results.append(result)

    # --- Summary outputs ---
    summary_dir = os.path.join(args.output_dir, "../Plots/SR_Optimisation_Summary")
    os.makedirs(summary_dir, exist_ok=True)

    save_results_table(all_results,
                       os.path.join(summary_dir, "sr_optimisation_results.txt"))
    save_results_json(all_results,
                      os.path.join(summary_dir, "sr_optimisation_results.json"))
    plot_comparison_summary(all_results, summary_dir)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("Summary of optimised parameters:")
    print("=" * 70)
    print(f"{'Config':<35} {'c1':>6} {'c2':>6} {'p1':>7} {'p2':>7} "
          f"{'cut':>6} {'eff':>7} {'Δc1':>5} {'Δc2':>5}")
    print("-" * 100)
    for r in all_results:
        if r is None or r["status"] != "OK":
            continue
        fp = r["fitted"]
        dc1 = fp["c1"] - DEFAULT_PARAMS["c1"]
        dc2 = fp["c2"] - DEFAULT_PARAMS["c2"]
        print(f"  {r['config']:<33} {fp['c1']:>6.1f} {fp['c2']:>6.1f} "
              f"{fp['p1']:>7.0f} {fp['p2']:>7.0f} "
              f"{fp['cut']:>6.2f} {r['sig_eff_at_optimal']:>6.1%} "
              f"{dc1:>+5.1f} {dc2:>+5.1f}")

    print("\n" + "=" * 70)
    print("All done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
