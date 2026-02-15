#!/usr/bin/env python
"""
Plot 2D Mass Plane for all Jet Kinematics Modes and Taggers (SR only)

Loops over all configurations:
- MC campaigns: MC20, MC23
- Taggers: GN2X, GN3PV01
- Jet kinematics modes: DEFAULT, bjr_v00, bjr_v01

Draws only the SR contour (VBF formula < 1.6, red solid).
"""

import ROOT as root
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root.gStyle.SetOptStat(0)
root.gStyle.SetOptTitle(0)
root.gROOT.SetBatch(True)

BASE_DIR    = "../BaristaSAMPLES/"
OUTPUT_DIR  = "../Plots/SRonly/"
MC_CAMPAIGNS = ["MC20", "MC23"]
TAGGERS      = ["GN2X", "GN3PV01"]
JET_MODES    = ["DEFAULT", "bjr_v00", "bjr_v01"]
SIGNAL_TYPES = ["SM", "Kappa2V", "quad_M0_S1"]
SIGNAL_COUPLINGS = {
    "SM":        [None],
    "Kappa2V":   ["l1cvv0cv1", "l1cvv1cv1", "l1cvv2cv1"],
    "quad_M0_S1": ["M0_1", "S1_10"],
}
# ROOT TLatex display labels per coupling
SIGNAL_DISPLAY = {
    None:        "SM",
    "l1cvv0cv1": "#kappa_{2V} = 0",
    "l1cvv1cv1": "#kappa_{2V} = 1",
    "l1cvv2cv1": "#kappa_{2V} = 2",
    "M0_1":      "EFT M0_1 (Quad)",
    "S1_10":     "EFT S1_10 (Quad)",
}
SELECTION_CUT = "pass_boosted_vbf_sel == 1 && boosted_vbf_has_btag == 0 && boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2"

SR_CX = 124; SR_CY = 117
SR_P1 = 1500.0; SR_P2 = 1900.0; SR_CUT = 1.6
MASS_MIN = 50.0; MASS_MAX = 300.0; NBINS = 100


def ATLASLabel(x, y, text, color=1):
    l = root.TLatex(); l.SetNDC(); l.SetTextFont(72); l.SetTextColor(color); l.SetTextSize(0.04)
    delx = 0.115 * 550 * root.gPad.GetWh() / (472 * root.gPad.GetWw())
    l.DrawLatex(x, y, "ATLAS")
    if text:
        p = root.TLatex(); p.SetNDC(); p.SetTextSize(0.04); p.SetTextFont(42); p.SetTextColor(color)
        p.DrawLatex(x + delx, y, text)


def myText(x, y, text, color=1, size=0.025):
    l = root.TLatex(); l.SetTextSize(size); l.SetNDC(); l.SetTextColor(color); l.DrawLatex(x, y, text)


def get_folder_name(mc_campaign, signal_type, tagger, jet_mode):
    return os.path.join(mc_campaign, f"{signal_type}_{tagger}_{jet_mode}")


def find_root_files(folder_path, signal_type="SM", coupling=None):
    """
    Find ROOT files in folder_path/RootSamp.
    SM:        single merged file.
    Kappa2V:   all sub-campaign files for the given coupling.
    quad_M0_S1: all sub-campaign files for the given EFT operator.
    Always returns a list of file paths, or empty list if none found.
    """
    root_samp_path = os.path.join(folder_path, "RootSamp")
    if not os.path.exists(root_samp_path):
        print(f"  WARNING: RootSamp folder not found in {folder_path}")
        return []

    if signal_type == "Kappa2V" and coupling:
        pattern = os.path.join(root_samp_path, f"boosted_skim_VBFhh_{coupling}_*__Nominal.root")
        files = sorted(glob.glob(pattern))
        if files:
            return files
        print(f"  WARNING: No ROOT files for coupling {coupling} in {root_samp_path}")
        return []

    elif signal_type == "quad_M0_S1" and coupling:
        # e.g. boosted_skim_VBFhh_EFT_MGPy8EG_hhjj_vbf_M0_1_Quad_mc20__Nominal.root
        pattern = os.path.join(root_samp_path,
                               f"boosted_skim_VBFhh_EFT_MGPy8EG_hhjj_vbf_{coupling}_Quad_*__Nominal.root")
        files = sorted(glob.glob(pattern))
        if files:
            return files
        print(f"  WARNING: No ROOT files for EFT {coupling} in {root_samp_path}")
        return []

    else:
        # SM: single file patterns
        for pattern in [
            os.path.join(root_samp_path, "boosted_skim_VBFhh_*_SM_mc20__Nominal.root"),
            os.path.join(root_samp_path, "boosted_skim_VBFhh_*_SM_mc23__Nominal.root"),
            os.path.join(root_samp_path, "boosted_skim_VBFhh_*__Nominal.root"),
        ]:
            files = glob.glob(pattern)
            if files:
                return [files[0]]
        print(f"  WARNING: No ROOT files found in {root_samp_path}")
        return []


def vbf_sr_value(m1, m2, cx=SR_CX, cy=SR_CY, p1=SR_P1, p2=SR_P2):
    return np.sqrt((m1 * (m1 - cx) / p1) ** 2 + (m2 * (m2 - cy) / p2) ** 2)


def _make_sr_contour(n_grid=500):
    """Compute SR contour segments and label info using matplotlib, then close the figure."""
    m_arr = np.linspace(MASS_MIN + 1, MASS_MAX, n_grid)
    M1, M2 = np.meshgrid(m_arr, m_arr)
    Z = vbf_sr_value(np.ravel(M1), np.ravel(M2))

    fig, ax = plt.subplots()
    CS = ax.tricontour(np.ravel(M1), np.ravel(M2), Z, levels=[SR_CUT], colors=["red"])
    segments = []
    if hasattr(CS, 'allsegs') and len(CS.allsegs) > 0:
        for seg in CS.allsegs[0]:
            if len(seg) >= 2:
                segments.append(np.array(seg))
    plt.close(fig)
    return segments


def draw_sr_contour():
    """Draw SR contour (red solid) and label. Returns dict of ROOT objects to keep alive."""
    objects = {}
    segments = _make_sr_contour()

    graphs = []
    for seg in segments:
        if len(seg) < 2:
            continue
        gr = root.TGraph(len(seg))
        for i in range(len(seg)):
            gr.SetPoint(i, seg[i, 0], seg[i, 1])
        gr.SetLineColor(root.kRed + 1)
        gr.SetLineWidth(3)
        gr.SetLineStyle(1)
        gr.Draw("L SAME")
        graphs.append(gr)
    objects['sr_graphs'] = graphs

    label = root.TLatex(SR_CX - 9, SR_CY - 13, "SR")
    label.SetTextColor(root.kRed + 1)
    label.SetTextSize(0.035)
    label.SetTextFont(62)
    label.SetTextAlign(22)
    label.Draw()
    objects['label'] = label

    return objects


def calculate_sr_efficiency(hist_2d):
    """Calculate SR efficiency only."""
    total = hist_2d.Integral()
    sr_events = 0.0
    for i in range(1, hist_2d.GetNbinsX() + 1):
        for j in range(1, hist_2d.GetNbinsY() + 1):
            m1 = hist_2d.GetXaxis().GetBinCenter(i)
            m2 = hist_2d.GetYaxis().GetBinCenter(j)
            w = hist_2d.GetBinContent(i, j)
            if w == 0:
                continue
            if vbf_sr_value(m1, m2) < SR_CUT:
                sr_events += w
    sr_eff = sr_events / total if total > 0 else 0
    return {"total": total, "sr": sr_events, "sr_eff": sr_eff}


def _get_tree_name(root_file_path):
    f = root.TFile.Open(root_file_path)
    if not f or f.IsZombie():
        return None
    for name in ["ttree", "AnalysisMiniTree", "Nominal"]:
        if f.Get(name):
            f.Close(); return name
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, root.TTree):
            tname = key.GetName(); f.Close(); return tname
    f.Close()
    return None


def plot_mass_plane(root_file_paths, output_dir, config_label, mc_campaign, tagger, jet_mode,
                    signal_type="SM", coupling=None):
    if isinstance(root_file_paths, str):
        root_file_paths = [root_file_paths]

    tree_name = _get_tree_name(root_file_paths[0])
    if not tree_name:
        print(f"  ERROR: No TTree found in {root_file_paths[0]}"); return None

    chain = root.TChain(tree_name)
    for fp in root_file_paths:
        print(f"  Adding: {fp}")
        chain.Add(fp)
    print(f"  Chain has {chain.GetEntries()} entries (tree: {tree_name})")

    canvas = root.TCanvas(f"canvas_{config_label}", "Mass Plane", 900, 800)
    canvas.cd()
    canvas.SetLeftMargin(0.12); canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.12); canvas.SetTopMargin(0.08)

    hist_name = f"h2d_mass_plane_{config_label}"
    hist_2d = root.TH2D(hist_name, "", NBINS, MASS_MIN, MASS_MAX, NBINS, MASS_MIN, MASS_MAX)
    draw_cmd = f"boosted_m_h2:boosted_m_h1>>+{hist_name}"

    print(f"  Drawing 2D histogram...")
    entries = chain.Draw(draw_cmd, f"({SELECTION_CUT})*mc_sf", "COLZ")
    print(f"  Drew {entries} entries")

    hist_2d = root.gDirectory.Get(hist_name)
    if not hist_2d or hist_2d.GetEntries() == 0:
        print("  ERROR: No entries in histogram!"); return None

    hist_2d.SetTitle("")
    hist_2d.GetXaxis().SetTitle("m_{H_{1}} [GeV]")
    hist_2d.GetYaxis().SetTitle("m_{H_{2}} [GeV]")
    hist_2d.GetZaxis().SetTitle("a.u.")
    hist_2d.GetXaxis().SetTitleSize(0.045); hist_2d.GetYaxis().SetTitleSize(0.045)
    hist_2d.GetZaxis().SetTitleSize(0.045)
    hist_2d.GetXaxis().SetLabelSize(0.04); hist_2d.GetYaxis().SetLabelSize(0.04)
    hist_2d.GetZaxis().SetLabelSize(0.04)
    hist_2d.GetXaxis().SetTitleOffset(1.1); hist_2d.GetYaxis().SetTitleOffset(1.3)
    hist_2d.GetZaxis().SetTitleOffset(1.2)

    # Set 'Blues' palette to match optimize_sr.py
    from array import array as arr
    NRGBs = 5
    NCont = 255
    stops = arr('d', [0.00, 0.25, 0.50, 0.75, 1.00])
    red   = arr('d', [0.97, 0.75, 0.42, 0.19, 0.03])
    green = arr('d', [0.98, 0.83, 0.68, 0.44, 0.19])
    blue  = arr('d', [1.00, 0.93, 0.84, 0.69, 0.42])
    root.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
    root.gStyle.SetNumberContours(NCont)
    hist_2d.Draw("COLZ")
    region_objects = draw_sr_contour()
    eff = calculate_sr_efficiency(hist_2d)

    signal_label = SIGNAL_DISPLAY.get(coupling, signal_type)
    ATLASLabel(0.18, 0.87, "Working Progress")
    myText(0.15, 0.82, f"VBF HH #rightarrow 4b {signal_label} - {mc_campaign}", color=root.kBlack)
    myText(0.15, 0.77, f"Tagger: {tagger}, Mode: {jet_mode}", color=root.kBlack)
    myText(0.15, 0.72, f"SR efficiency: {100*eff['sr_eff']:.1f}%", color=root.kBlack)

    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, f"mass_plane_{config_label}")
    canvas.SaveAs(f"{output_base}.pdf")
    canvas.SaveAs(f"{output_base}.png")
    canvas.SaveAs(f"{output_base}.root")
    print(f"  Saved: {output_base}.pdf/.png/.root")
    print(f"  SR eff: {100*eff['sr_eff']:.1f}%")

    return {
        "config": config_label, "mc_campaign": mc_campaign, "tagger": tagger, "jet_mode": jet_mode,
        "entries": hist_2d.GetEntries(), "integral": hist_2d.Integral(),
        "mean_mH1": hist_2d.GetMean(1), "mean_mH2": hist_2d.GetMean(2),
        "rms_mH1": hist_2d.GetRMS(1), "rms_mH2": hist_2d.GetRMS(2),
        "sr_events": eff["sr"], "total_events": eff["total"], "sr_efficiency": eff["sr_eff"],
    }


def save_comparison_table(all_stats, output_file):
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("Mass Plane Comparison - SR Only\n")
        f.write("=" * 120 + "\n\n")
        f.write("Selection: " + SELECTION_CUT + "\n")
        f.write(f"SR: VBF formula (p1={SR_P1}, p2={SR_P2}, c=({SR_CX},{SR_CY})) < {SR_CUT}\n\n")
        f.write(f"{'Configuration':<40} {'Entries':>10} {'Mean mH1':>10} {'Mean mH2':>10} "
                f"{'RMS mH1':>10} {'RMS mH2':>10} {'SR Eff':>10}\n")
        f.write("-" * 120 + "\n")
        for mc in MC_CAMPAIGNS + ["Combined"]:
            f.write(f"\n{mc}:\n")
            for stat in all_stats:
                if stat and stat["mc_campaign"] == mc:
                    f.write(f"  {stat['tagger']}_{stat['jet_mode']:<30} "
                            f"{stat['entries']:>10.0f} {stat['mean_mH1']:>10.2f} {stat['mean_mH2']:>10.2f} "
                            f"{stat['rms_mH1']:>10.2f} {stat['rms_mH2']:>10.2f} "
                            f"{100*stat['sr_efficiency']:>9.1f}%\n")
        f.write("\n" + "=" * 120 + "\n")
    print(f"\nComparison table saved to: {output_file}")


def create_summary_plot(all_stats, output_dir):
    valid_stats = [s for s in all_stats if s is not None]
    if not valid_stats:
        print("No valid statistics to plot"); return

    canvas = root.TCanvas("canvas_summary", "SR Efficiency Comparison", 1400, 600)
    canvas.cd()
    canvas.SetLeftMargin(0.08); canvas.SetRightMargin(0.03)
    canvas.SetBottomMargin(0.22); canvas.SetTopMargin(0.08)

    n = len(valid_stats)
    h_sr = root.TH1D("h_sr_eff", "", n, 0, n)
    for i, stat in enumerate(valid_stats):
        label = f"{stat['mc_campaign']}_{stat['tagger']}_{stat['jet_mode']}"
        h_sr.GetXaxis().SetBinLabel(i + 1, label)
        h_sr.SetBinContent(i + 1, 100 * stat['sr_efficiency'])
    h_sr.SetFillColor(root.kRed - 7); h_sr.SetLineColor(root.kRed + 1)
    h_sr.GetYaxis().SetTitle("SR Efficiency [%]"); h_sr.GetYaxis().SetRangeUser(0, 100)
    h_sr.GetXaxis().SetLabelSize(0.03); h_sr.GetXaxis().LabelsOption("v")
    h_sr.Draw("BAR")

    ATLASLabel(0.12, 0.92, "Internal Simulation")
    myText(0.12, 0.87, "VBF HH #rightarrow 4b SM - SR Efficiency Comparison", color=root.kBlack)

    output_base = os.path.join(output_dir, "sr_efficiency_comparison")
    canvas.SaveAs(f"{output_base}.pdf"); canvas.SaveAs(f"{output_base}.png")
    print(f"Summary plot saved to: {output_base}.pdf/.png")


def main():
    print("=" * 70)
    print("Mass Plane Plotting - SR Only")
    print("=" * 70)
    print(f"SR: VBF formula (p1={SR_P1}, p2={SR_P2}, c=({SR_CX},{SR_CY})) < {SR_CUT}")

    if not os.path.exists(BASE_DIR):
        print(f"ERROR: Base directory not found: {BASE_DIR}"); return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_stats = []

    # ── Individual MC campaigns ──
    for mc_campaign in MC_CAMPAIGNS:
        for signal_type in SIGNAL_TYPES:
            couplings = SIGNAL_COUPLINGS[signal_type]
            for coupling in couplings:
                for tagger in TAGGERS:
                    for jet_mode in JET_MODES:
                        coup_str = f"_{coupling}" if coupling else ""
                        config_label = f"{mc_campaign}_{signal_type}{coup_str}_{tagger}_{jet_mode}"
                        folder_path = os.path.join(BASE_DIR, get_folder_name(mc_campaign, signal_type, tagger, jet_mode))
                        print(f"\n{'=' * 70}\nProcessing: {config_label}\nFolder: {folder_path}")
                        if not os.path.exists(folder_path):
                            print(f"  WARNING: Folder not found, skipping"); all_stats.append(None); continue
                        root_files = find_root_files(folder_path, signal_type, coupling)
                        if not root_files:
                            print(f"  WARNING: No ROOT file found, skipping"); all_stats.append(None); continue
                        output_dir = os.path.join(OUTPUT_DIR, f"H_MassPlane_Plots_{config_label}")
                        stats = plot_mass_plane(root_files, output_dir, config_label, mc_campaign, tagger, jet_mode,
                                                signal_type, coupling)
                        all_stats.append(stats)

    # ── Combined MC20+MC23 ──
    print(f"\n{'=' * 70}\nProcessing combined MC20+MC23...\n{'=' * 70}")
    for signal_type in SIGNAL_TYPES:
        couplings = SIGNAL_COUPLINGS[signal_type]
        for coupling in couplings:
            for tagger in TAGGERS:
                for jet_mode in JET_MODES:
                    coup_str = f"_{coupling}" if coupling else ""
                    config_label = f"Combined_{signal_type}{coup_str}_{tagger}_{jet_mode}"
                    print(f"\n{'=' * 70}\nProcessing: {config_label}")
                    combined_files = []
                    for mc in MC_CAMPAIGNS:
                        folder_path = os.path.join(BASE_DIR, get_folder_name(mc, signal_type, tagger, jet_mode))
                        if os.path.exists(folder_path):
                            rf = find_root_files(folder_path, signal_type, coupling)
                            combined_files.extend(rf)
                    if not combined_files:
                        print(f"  WARNING: No files found, skipping"); all_stats.append(None); continue
                    print(f"  Merging {len(combined_files)} files")
                    output_dir = os.path.join(OUTPUT_DIR, f"H_MassPlane_Plots_{config_label}")
                    stats = plot_mass_plane(combined_files, output_dir, config_label, "Combined", tagger, jet_mode,
                                            signal_type, coupling)
                    all_stats.append(stats)

    save_comparison_table(all_stats, os.path.join(OUTPUT_DIR, "H_mass_plane_comparison.txt"))
    summary_dir = os.path.join(OUTPUT_DIR, "H_Summary_Plots")
    os.makedirs(summary_dir, exist_ok=True)
    create_summary_plot(all_stats, summary_dir)
    print("\n" + "=" * 70 + "\nAll done!\n" + "=" * 70)


if __name__ == "__main__":
    main()
