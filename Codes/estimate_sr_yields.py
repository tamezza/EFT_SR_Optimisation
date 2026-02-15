#!/usr/bin/env python3
"""
estimate_sr_yields.py — Expected event yields in the VBF HH→4b signal region

Computes the expected number of weighted events in the signal region for all
signal and background samples, using:

  weight = mc_sf × luminosity_boosted × pileupweight_boosted  →  [expected events]

The SR is defined by the elliptical discriminant:

  D = √[ (m₁(m₁ − c₁)/p₁)² + (m₂(m₂ − c₂)/p₂)² ]

Events with D < cut are inside the SR.

SR parameters can be supplied via:
  1. CLI flags (--c1, --c2, --p1, --p2, --cut)
  2. A JSON file from optimize_sr_v2.py (--sr-json)
  3. Defaults from HDBS-2022-02

Usage:
  python estimate_sr_yields.py \\
      --base-dir ../BaristaSAMPLES \\
      --mc-campaign MC20 \\
      --tagger GN3PV01 \\
      --jet-mode DEFAULT \\
      --output-dir ./yields

  # With optimised parameters from JSON:
  python estimate_sr_yields.py \\
      --base-dir ../BaristaSAMPLES \\
      --sr-json ./results/summary/sr_optimisation_results.json \\
      --sr-config MC20_GN3PV01_DEFAULT
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

# Try ROOT import
try:
    import ROOT as root
    root.gROOT.SetBatch(True)
    root.gErrorIgnoreLevel = root.kWarning
except ImportError:
    print("ERROR: PyROOT is required. Please set up ROOT first.")
    sys.exit(1)


# ============================================================
# Constants
# ============================================================

# Default SR parameters (HDBS-2022-02)
DEFAULT_SR = {
    "c1": 124.0, "c2": 117.0,
    "p1": 1500.0, "p2": 1900.0,
    "cut": 1.6,
}

SELECTION_CUT = "pass_boosted_vbf_sel == 1 && boosted_vbf_has_btag == 0 && boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2"

MASS_VAR_H1 = "boosted_m_h1"
MASS_VAR_H2 = "boosted_m_h2"

WEIGHT_VAR = "mc_sf"
LUMI_VAR = "luminosity_boosted"
PILEUP_VAR = "pileupweight_boosted"


# ============================================================
# SR discriminant
# ============================================================

def sr_discriminant(m1, m2, c1, c2, p1, p2):
    """D = √[ (m₁(m₁ − c₁)/p₁)² + (m₂(m₂ − c₂)/p₂)² ]"""
    return np.sqrt(
        (m1 * (m1 - c1) / p1) ** 2 +
        (m2 * (m2 - c2) / p2) ** 2
    )


# ============================================================
# File discovery
# ============================================================

def find_sample_dirs(base_dir, mc_campaign, tagger, jet_mode):
    """
    Find all sample directories matching (tagger, jet_mode) in
    base_dir/{mc_campaign}/.

    Returns list of (process_label, root_samp_path).
    Handles GN3PV01 ↔ GN3XPV01 naming inconsistency.
    """
    mc_dir = os.path.join(base_dir, mc_campaign)
    if not os.path.isdir(mc_dir):
        print(f"  WARNING: {mc_dir} does not exist")
        return []

    # Build suffixes to match
    tagger_names = [tagger]
    if tagger == "GN3PV01":
        tagger_names.append("GN3XPV01")
    elif tagger == "GN3XPV01":
        tagger_names.append("GN3PV01")

    suffixes = []
    for t in tagger_names:
        suffixes.append(f"_{t}_{jet_mode}")

    results = []
    for entry in sorted(os.listdir(mc_dir)):
        entry_path = os.path.join(mc_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Try each suffix
        process = None
        for suffix in suffixes:
            if entry.endswith(suffix):
                process = entry[:-len(suffix)]
                break

        if process is None:
            continue

        root_samp = os.path.join(entry_path, "RootSamp")
        if not os.path.exists(root_samp):
            continue

        root_files = sorted(glob.glob(os.path.join(root_samp, "*.root")))
        if root_files:
            results.append((process, root_files))

    return results


def filter_signal_files(root_files, signal_coupling):
    """
    Filter ROOT file list to keep only those matching the coupling tag.
    If no files match, return all files (fallback for background etc.).
    """
    matched = [f for f in root_files if signal_coupling in os.path.basename(f)]
    return matched if matched else root_files


# ============================================================
# Tree reading
# ============================================================

def read_sample(root_files, selection, signal_coupling=None):
    """
    Read m_H1, m_H2, and full physics weight from a list of ROOT files.

    Returns (m1, m2, weights) numpy arrays, or None if no events pass.
    Weight = mc_sf × luminosity_boosted × pileupweight_boosted [expected events].
    """
    all_m1, all_m2, all_w = [], [], []

    for fpath in root_files:
        # If signal_coupling is set, skip files that don't match
        if signal_coupling and signal_coupling not in os.path.basename(fpath):
            continue

        tf = root.TFile.Open(fpath)
        if not tf or tf.IsZombie():
            print(f"    WARNING: Cannot open {fpath}")
            continue

        tree = tf.Get("ttree")
        if not tree:
            tf.Close()
            continue

        n_entries = tree.GetEntries()
        if n_entries == 0:
            tf.Close()
            continue

        # Count events passing selection
        n_pass = tree.Draw(">>elist", selection, "entrylist")
        if n_pass <= 0:
            tf.Close()
            continue

        tree.SetEstimate(n_pass)

        # Read branches
        branches = [MASS_VAR_H1, MASS_VAR_H2, WEIGHT_VAR, LUMI_VAR, PILEUP_VAR]
        arrays = {}
        ok = True
        for var in branches:
            n = tree.Draw(var, selection, "goff")
            if n <= 0:
                ok = False
                break
            buf = tree.GetV1()
            arrays[var] = np.array([buf[i] for i in range(n)])

        tree.SetEntryList(0)
        tf.Close()

        if not ok:
            continue

        m1 = arrays[MASS_VAR_H1]
        m2 = arrays[MASS_VAR_H2]
        w = arrays[WEIGHT_VAR] * arrays[LUMI_VAR] * arrays[PILEUP_VAR]

        all_m1.append(m1)
        all_m2.append(m2)
        all_w.append(w)

    if not all_m1:
        return None

    return (np.concatenate(all_m1),
            np.concatenate(all_m2),
            np.concatenate(all_w))


# ============================================================
# Yield computation
# ============================================================

def compute_yields(m1, m2, weights, sr_params):
    """
    Compute expected yields before and after SR cut.

    Returns dict with:
      n_raw         : unweighted event count after selection
      n_total       : sum of weights after selection (before SR cut)
      n_sr          : sum of weights inside SR (D < cut)
      n_sr_err      : statistical uncertainty on n_sr (√Σw²)
      efficiency    : n_sr / n_total
    """
    D = sr_discriminant(m1, m2,
                        sr_params["c1"], sr_params["c2"],
                        sr_params["p1"], sr_params["p2"])

    mask_sr = D < sr_params["cut"]

    n_raw = len(m1)
    n_total = np.sum(weights)
    n_sr = np.sum(weights[mask_sr])
    n_sr_err = np.sqrt(np.sum(weights[mask_sr] ** 2))
    eff = n_sr / n_total if n_total != 0 else 0.0

    return {
        "n_raw": n_raw,
        "n_total": n_total,
        "n_sr": n_sr,
        "n_sr_err": n_sr_err,
        "efficiency": eff,
    }


# ============================================================
# Categorisation
# ============================================================

def categorise_process(process):
    """
    Assign a category to a process based on its folder prefix.

    Categories:
      Signal   : Kappa2V (κ₂V=0, the primary signal)
      VBF_SM   : SM VBF HH
      VBF_Quad : quad_M0_S1 (quadratic EFT terms)
      QCD      : QCD multijet (JZ slices)
      ttbar    : top-quark pair production
      Other    : anything else
    """
    p = process.lower()
    if p.startswith("kappa2v"):
        return "Signal"
    elif p == "sm":
        return "VBF_SM"
    elif p.startswith("quad"):
        return "VBF_Quad"
    elif p == "qcd":
        return "QCD"
    elif p == "ttbar":
        return "ttbar"
    else:
        return "Other"


# ============================================================
# Output formatting
# ============================================================

def print_yield_table(results, sr_params):
    """Print a formatted yield table."""
    # Header
    print("\n" + "=" * 100)
    print(f"  Expected event yields in the Signal Region")
    print(f"  SR parameters: c₁={sr_params['c1']:.1f}, c₂={sr_params['c2']:.1f}, "
          f"p₁={sr_params['p1']:.1f}, p₂={sr_params['p2']:.1f}, "
          f"cut={sr_params['cut']:.2f}")
    print(f"  Weight: mc_sf × luminosity_boosted × pileupweight_boosted")
    print("=" * 100)

    # Column headers
    header = (f"{'Category':<12} {'Process':<20} {'N_raw':>8} "
              f"{'N_total':>14} {'N_SR':>14} {'±stat':>12} {'Eff(%)':>8}")
    print(header)
    print("-" * 100)

    # Sort by category priority
    cat_order = {"Signal": 0, "VBF_SM": 1, "VBF_Quad": 2,
                 "QCD": 3, "ttbar": 4, "Other": 5}

    sorted_results = sorted(results,
                            key=lambda r: (cat_order.get(r["category"], 99),
                                           r["process"]))

    # Track totals
    total_signal = 0.0
    total_signal_err2 = 0.0
    total_bkg = 0.0
    total_bkg_err2 = 0.0

    prev_cat = None
    for r in sorted_results:
        # Separator between categories
        if r["category"] != prev_cat and prev_cat is not None:
            print("-" * 100)
        prev_cat = r["category"]

        line = (f"{r['category']:<12} {r['process']:<20} {r['n_raw']:>8d} "
                f"{r['n_total']:>14.4f} {r['n_sr']:>14.4f} "
                f"{'±':>2}{r['n_sr_err']:>10.4f} {r['efficiency']*100:>7.1f}%")
        print(line)

        # Accumulate totals
        if r["category"] in ("Signal",):
            total_signal += r["n_sr"]
            total_signal_err2 += r["n_sr_err"] ** 2
        elif r["category"] in ("QCD", "ttbar"):
            total_bkg += r["n_sr"]
            total_bkg_err2 += r["n_sr_err"] ** 2

    # Summary
    print("=" * 100)
    total_signal_err = np.sqrt(total_signal_err2)
    total_bkg_err = np.sqrt(total_bkg_err2)

    print(f"{'SIGNAL (S)':<33} {'':>8} {'':>14} "
          f"{total_signal:>14.4f} {'±':>2}{total_signal_err:>10.4f}")
    print(f"{'BACKGROUND (B)':<33} {'':>8} {'':>14} "
          f"{total_bkg:>14.4f} {'±':>2}{total_bkg_err:>10.4f}")

    if total_bkg > 0:
        s_over_sqrtb = total_signal / np.sqrt(total_bkg)
        s_over_b = total_signal / total_bkg
        print(f"\n  S/√B = {s_over_sqrtb:.4f}")
        print(f"  S/B  = {s_over_b:.6f}")
    elif total_signal > 0:
        print(f"\n  No background events in SR → S/√B undefined")

    print("=" * 100)


def save_yield_table(results, sr_params, output_dir, label):
    """Save yields to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, f"sr_yields_{label}.csv")
    with open(csv_path, "w") as f:
        f.write(f"# SR parameters: c1={sr_params['c1']}, c2={sr_params['c2']}, "
                f"p1={sr_params['p1']}, p2={sr_params['p2']}, cut={sr_params['cut']}\n")
        f.write(f"# Weight: mc_sf * luminosity_boosted * pileupweight_boosted\n")
        f.write("category,process,n_raw,n_total,n_sr,n_sr_err,efficiency\n")
        for r in results:
            f.write(f"{r['category']},{r['process']},{r['n_raw']},"
                    f"{r['n_total']:.6f},{r['n_sr']:.6f},"
                    f"{r['n_sr_err']:.6f},{r['efficiency']:.6f}\n")
    print(f"\nCSV saved: {csv_path}")

    # JSON
    json_path = os.path.join(output_dir, f"sr_yields_{label}.json")
    output = {
        "sr_parameters": sr_params,
        "selection": SELECTION_CUT,
        "weight": "mc_sf * luminosity_boosted * pileupweight_boosted",
        "samples": results,
    }
    # Add summary
    total_s = sum(r["n_sr"] for r in results if r["category"] == "Signal")
    total_b = sum(r["n_sr"] for r in results if r["category"] in ("QCD", "ttbar"))
    output["summary"] = {
        "total_signal": total_s,
        "total_background": total_b,
        "s_over_sqrt_b": total_s / np.sqrt(total_b) if total_b > 0 else None,
        "s_over_b": total_s / total_b if total_b > 0 else None,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved: {json_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate expected event yields in the VBF HH→4b SR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default SR (HDBS-2022-02):
  python estimate_sr_yields.py --base-dir ../BaristaSAMPLES --mc-campaign MC20 --tagger GN3PV01

  # Custom SR parameters:
  python estimate_sr_yields.py --base-dir ../BaristaSAMPLES --c1 124 --c2 117 --p1 1500 --p2 1900 --cut 1.4

  # From optimizer JSON:
  python estimate_sr_yields.py --base-dir ../BaristaSAMPLES --sr-json ./results/summary/sr_optimisation_results.json --sr-config MC20_GN3PV01_DEFAULT
        """,
    )

    # Directory options
    parser.add_argument("--base-dir", default="../BaristaSAMPLES",
                        help="Parent directory containing MC20/, MC23/")
    parser.add_argument("--mc-campaign", default="MC20",
                        help="MC campaign (default: MC20)")
    parser.add_argument("--tagger", default="GN3PV01",
                        help="Tagger (default: GN3PV01)")
    parser.add_argument("--jet-mode", default="DEFAULT",
                        help="Jet mode (default: DEFAULT)")
    parser.add_argument("--output-dir", default="./yields",
                        help="Output directory for CSV/JSON")

    # Sample selection
    parser.add_argument("--signal-label", default="Kappa2V",
                        help="Signal folder prefix (default: Kappa2V)")
    parser.add_argument("--signal-coupling", default="l1cvv0cv1",
                        help="Coupling tag in ROOT filename for signal "
                             "(default: l1cvv0cv1 = κ₂V=0)")
    parser.add_argument("--processes", nargs="+", default=None,
                        help="Only process these folder prefixes. "
                             "Default: all folders matching tagger/mode")
    parser.add_argument("--bkg-processes", nargs="+",
                        default=["QCD", "ttbar"],
                        help="Folder prefixes counted as background "
                             "(default: QCD ttbar)")

    # SR parameters (manual)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    parser.add_argument("--p1", type=float, default=None)
    parser.add_argument("--p2", type=float, default=None)
    parser.add_argument("--cut", type=float, default=None)

    # SR parameters (from optimizer JSON)
    parser.add_argument("--sr-json", default=None,
                        help="Path to sr_optimisation_results.json from optimizer")
    parser.add_argument("--sr-config", default=None,
                        help="Config key in JSON, e.g. MC20_GN3PV01_DEFAULT. "
                             "Uses 'fitted' parameters if available, else 'default'.")
    parser.add_argument("--sr-set", default="fitted",
                        choices=["default", "fitted", "joint"],
                        help="Which parameter set to use from JSON (default: fitted)")

    # Selection
    parser.add_argument("--selection", default=SELECTION_CUT,
                        help="TTree selection cut")

    args = parser.parse_args()

    # --- Resolve SR parameters ---
    sr_params = dict(DEFAULT_SR)  # start with defaults

    if args.sr_json:
        # Load from optimizer JSON
        with open(args.sr_json) as f:
            all_json = json.load(f)

        config_key = args.sr_config
        if config_key is None:
            config_key = f"{args.mc_campaign}_{args.tagger}_{args.jet_mode}"

        # Find matching config
        found = None
        for entry in all_json:
            if entry.get("config") == config_key:
                found = entry
                break

        if found is None:
            print(f"ERROR: Config '{config_key}' not found in {args.sr_json}")
            print(f"Available configs: {[e.get('config') for e in all_json]}")
            sys.exit(1)

        # Try requested parameter set, fall back gracefully
        param_key = f"{args.sr_set}_params"
        if param_key in found and found[param_key]:
            p = found[param_key]
            sr_params = {
                "c1": p["c1"], "c2": p["c2"],
                "p1": p["p1"], "p2": p["p2"],
                "cut": p.get("cut", p.get("best_cut", DEFAULT_SR["cut"])),
            }
            print(f"Loaded {args.sr_set} SR parameters from {args.sr_json}")
        else:
            print(f"WARNING: '{param_key}' not found in JSON, using defaults")

    # CLI overrides (take priority)
    if args.c1 is not None: sr_params["c1"] = args.c1
    if args.c2 is not None: sr_params["c2"] = args.c2
    if args.p1 is not None: sr_params["p1"] = args.p1
    if args.p2 is not None: sr_params["p2"] = args.p2
    if args.cut is not None: sr_params["cut"] = args.cut

    # --- Discover samples ---
    label = f"{args.mc_campaign}_{args.tagger}_{args.jet_mode}"

    print(f"\n{'=' * 70}")
    print(f"  SR Yield Estimation — VBF Boosted HH → 4b")
    print(f"{'=' * 70}")
    print(f"  Base dir:    {args.base_dir}")
    print(f"  Campaign:    {args.mc_campaign}")
    print(f"  Tagger:      {args.tagger}")
    print(f"  Jet mode:    {args.jet_mode}")
    print(f"  Selection:   {args.selection}")
    print(f"  Signal:      {args.signal_label} (coupling: {args.signal_coupling})")
    print(f"  Background:  {args.bkg_processes}")
    print(f"  SR params:   c₁={sr_params['c1']:.1f}, c₂={sr_params['c2']:.1f}, "
          f"p₁={sr_params['p1']:.1f}, p₂={sr_params['p2']:.1f}, "
          f"cut={sr_params['cut']:.2f}")
    print(f"  Weight:      mc_sf × luminosity_boosted × pileupweight_boosted")
    print(f"{'=' * 70}")

    sample_dirs = find_sample_dirs(args.base_dir, args.mc_campaign,
                                   args.tagger, args.jet_mode)

    if not sample_dirs:
        print("ERROR: No sample directories found")
        sys.exit(1)

    # Filter processes if requested
    if args.processes:
        sample_dirs = [(p, files) for p, files in sample_dirs
                       if p in args.processes]

    print(f"\n  Found {len(sample_dirs)} sample(s):")
    for proc, files in sample_dirs:
        print(f"    {proc}: {len(files)} file(s)")

    # --- Read samples and compute yields ---
    results = []

    for process, root_files in sample_dirs:
        category = categorise_process(process)

        # For signal, filter by coupling
        coupling_filter = None
        if category == "Signal":
            coupling_filter = args.signal_coupling

        print(f"\n  Processing: {process} [{category}] ...")

        data = read_sample(root_files, args.selection,
                           signal_coupling=coupling_filter)
        if data is None:
            print(f"    No events pass selection — skipping")
            continue

        m1, m2, w = data
        print(f"    Events after selection: {len(m1)}")

        yields = compute_yields(m1, m2, w, sr_params)
        yields["process"] = process
        yields["category"] = category
        results.append(yields)

        print(f"    N_total = {yields['n_total']:.4f},  "
              f"N_SR = {yields['n_sr']:.4f} ± {yields['n_sr_err']:.4f},  "
              f"eff = {yields['efficiency']*100:.1f}%")

    # --- Output ---
    if not results:
        print("\nERROR: No samples produced results")
        sys.exit(1)

    print_yield_table(results, sr_params)
    save_yield_table(results, sr_params, args.output_dir, label)


if __name__ == "__main__":
    main()
