#!/usr/bin/env python
"""
Plot 2D Mass Plane for all Jet Kinematics Modes and Taggers

This script loops over all configurations:
- MC campaigns: MC20, MC23
- Taggers: GN2X, GN3PV01
- Jet kinematics modes: DEFAULT, bjr_v00, bjr_v01

And creates a 2D mass plane plot for each configuration,
with SR, VR, and CR region overlays matching the bbbbarista
VBF quadrant-based definitions.

Contours drawn (matching HDBS-2022-02 Figure 2 style):
  SR:   VBF formula (Eq.1)  < 1.6   [red solid, full]
  VR:   CRVR formula < 170, anti-diagonal quadrants  [orange dashed]
  CR:   CRVR formula < 170, diagonal quadrants        [green dotted]
  Quadrant lines at (124, 117) connecting SR to outer boundary
"""

import ROOT as root
import os
import math
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root.gStyle.SetOptStat(0)
root.gStyle.SetOptTitle(0)
root.gROOT.SetBatch(True)

BASE_DIR = "."
MC_CAMPAIGNS = ["MC20", "MC23"]
TAGGERS      = ["GN2X", "GN3PV01"]
JET_MODES    = ["DEFAULT", "bjr_v00", "bjr_v01"]
SELECTION_CUT = "pass_boosted_vbf_sel == 1 && boosted_vbf_has_btag == 0 && boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2"

SR_CX = 124; SR_CY = 117
SR_P1 = 1500.0; SR_P2 = 1900.0; SR_CUT = 1.6
CRVR_P1 = 0.1; CRVR_P2 = 0.1
CRVR_CUT = 170.0    # single boundary for both VR and CR
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

def get_folder_name(mc_campaign, tagger, jet_mode):
    return f"SM_{mc_campaign}_{tagger}_{jet_mode}"

def find_merged_root_file(folder_path):
    root_samp_path = os.path.join(folder_path, "RootSamp")
    if not os.path.exists(root_samp_path):
        print(f"  WARNING: RootSamp folder not found in {folder_path}")
        return None
    for pattern in [
        os.path.join(root_samp_path, "boosted_skim_VBFhh_*_SM_mc20__Nominal.root"),
        os.path.join(root_samp_path, "boosted_skim_VBFhh_*_SM_mc23__Nominal.root"),
        os.path.join(root_samp_path, "boosted_skim_VBFhh_*__Nominal.root"),
    ]:
        files = glob.glob(pattern)
        if files:
            return files[0]
    print(f"  WARNING: No ROOT files found in {root_samp_path}")
    return None


def vbf_sr_value(m1, m2, cx=SR_CX, cy=SR_CY, p1=SR_P1, p2=SR_P2):
    return np.sqrt((m1 * (m1 - cx) / p1) ** 2 + (m2 * (m2 - cy) / p2) ** 2)

def crvr_value(m1, m2, cx=SR_CX, cy=SR_CY, p1=CRVR_P1, p2=CRVR_P2):
    return np.sqrt(((m1 - cx) / (p1 * np.log(m1))) ** 2 + ((m2 - cy) / (p2 * np.log(m2))) ** 2)


def _make_labeled_contour(func, level, label_text, m_range=(50, 300), n_grid=500,
                          manual_positions=None, gap_frac=0.12):
    m1_arr = np.linspace(m_range[0] + 1, m_range[1], n_grid)
    m2_arr = np.linspace(m_range[0] + 1, m_range[1], n_grid)
    M1, M2 = np.meshgrid(m1_arr, m2_arr)
    X, Y = np.ravel(M1), np.ravel(M2)
    Z = func(X, Y)
    fig, ax = plt.subplots()
    CS = ax.tricontour(X, Y, Z, levels=[level], colors=["red"])
    raw_segments = []
    if hasattr(CS, 'allsegs') and len(CS.allsegs) > 0:
        for seg in CS.allsegs[0]:
            if len(seg) >= 2:
                raw_segments.append(np.array(seg))
    fmt = {level: label_text}
    label_info = []
    try:
        if manual_positions is not None:
            clabels = ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=manual_positions)
        else:
            clabels = ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)
        for lbl in (clabels or []):
            x, y = lbl.get_position()
            angle = lbl.get_rotation()
            label_info.append((x, y, angle))
    except Exception:
        pass
    plt.close(fig)
    segments = _cut_gaps_in_segments(raw_segments, label_info, gap_frac)
    return segments, label_info


def _cut_gaps_in_segments(segments, label_info, gap_frac=0.12):
    if not label_info:
        return segments
    result = list(segments)
    for (lx, ly, _angle) in label_info:
        new_result = []
        for seg in result:
            if len(seg) < 2:
                new_result.append(seg); continue
            diffs = np.diff(seg, axis=0)
            ds = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            cum = np.concatenate([[0], np.cumsum(ds)])
            total_len = cum[-1]
            if total_len == 0:
                new_result.append(seg); continue
            dists = np.sqrt((seg[:, 0] - lx)**2 + (seg[:, 1] - ly)**2)
            closest_idx = np.argmin(dists)
            if dists[closest_idx] > 0.15 * total_len:
                new_result.append(seg); continue
            center_arc = cum[closest_idx]
            half_gap = gap_frac * total_len / 2.0
            arc_start = max(0, center_arc - half_gap)
            arc_end = min(total_len, center_arc + half_gap)
            def _interp_at(target_arc):
                idx = max(0, min(np.searchsorted(cum, target_arc) - 1, len(seg) - 2))
                t = (target_arc - cum[idx]) / ds[idx] if ds[idx] > 0 else 0
                return seg[idx] + t * diffs[idx]
            before = seg[cum < arc_start]
            if arc_start > 0:
                pt = _interp_at(arc_start)
                before = np.vstack([before, pt]) if len(before) > 0 else pt.reshape(1, 2)
            after = seg[cum > arc_end]
            if arc_end < total_len:
                pt = _interp_at(arc_end)
                after = np.vstack([pt, after]) if len(after) > 0 else pt.reshape(1, 2)
            if len(before) >= 2: new_result.append(before)
            if len(after) >= 2: new_result.append(after)
        result = new_result
    return result


def _clip_segments_to_quadrant(segments, cx, cy, quadrant="anti_diagonal"):
    clipped = []
    for seg in segments:
        if quadrant == "anti_diagonal":
            mask = ((seg[:, 0] > cx) & (seg[:, 1] < cy)) | ((seg[:, 0] < cx) & (seg[:, 1] > cy))
        else:
            mask = ((seg[:, 0] > cx) & (seg[:, 1] > cy)) | ((seg[:, 0] < cx) & (seg[:, 1] < cy))
        if not np.any(mask): continue
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if mask[0]: starts = np.concatenate([[0], starts])
        if mask[-1]: ends = np.concatenate([ends, [len(mask)]])
        for s, e in zip(starts, ends):
            chunk = seg[s:e]
            if len(chunk) >= 2: clipped.append(chunk)
    return clipped


def _filter_labels_by_quadrant(label_info, cx, cy, quadrant):
    filtered = []
    for (x, y, angle) in label_info:
        if quadrant == "anti_diagonal":
            if (x > cx and y < cy) or (x < cx and y > cy): filtered.append((x, y, angle))
        else:
            if (x > cx and y > cy) or (x < cx and y < cy): filtered.append((x, y, angle))
    return filtered


def _segments_to_tgraphs(segments):
    graphs = []
    for seg in segments:
        if len(seg) < 2: continue
        gr = root.TGraph(len(seg))
        for i in range(len(seg)): gr.SetPoint(i, seg[i, 0], seg[i, 1])
        graphs.append(gr)
    return graphs


def _draw_graphs(graphs, color, width, style):
    for gr in graphs:
        gr.SetLineColor(color); gr.SetLineWidth(width); gr.SetLineStyle(style)
        gr.Draw("L SAME")


def _make_labels(label_info, text, color, size=0.03):
    labels = []
    for (x, y, angle) in label_info:
        latex = root.TLatex(x, y, text)
        latex.SetTextColor(color); latex.SetTextSize(size); latex.SetTextFont(62)
        latex.SetTextAngle(angle); latex.SetTextAlign(22); latex.Draw()
        labels.append(latex)
    return labels


def _find_contour_crossings(func, level, cx, cy, m_range=(50, 300), n_grid=500):
    m_arr = np.linspace(m_range[0] + 1, m_range[1], n_grid)
    crossings = {}
    vals = func(cx, m_arr)
    idx = np.where(np.diff(np.sign(vals - level)))[0]
    crossings['vertical'] = [(cx, m_arr[i]) for i in idx]
    vals = func(m_arr, cy)
    idx = np.where(np.diff(np.sign(vals - level)))[0]
    crossings['horizontal'] = [(m_arr[i], cy) for i in idx]
    return crossings


def _draw_quadrant_lines(sr_crossings, crvr_crossings):
    """
    Draw quadrant divider lines between SR and CRVR outer boundary.
    Each divider is a boundary of BOTH a VR and a CR region,
    so each segment is drawn twice (overlaid):
      - orange dashed  (VR boundary)
      - green dotted   (CR boundary)
    The overlapping dash/dot patterns create an alternating effect.
    Returns TLine list.
    """
    lines = []

    sr_v = sr_crossings.get('vertical', [])
    cr_v = crvr_crossings.get('vertical', [])
    if sr_v and cr_v:
        sr_ys = sorted([p[1] for p in sr_v])
        cr_ys = sorted([p[1] for p in cr_v])
        if cr_ys and sr_ys:
            # Lower vertical (SR → bottom)
            if min(cr_ys) < min(sr_ys):
                l1 = root.TLine(SR_CX, min(cr_ys), SR_CX, min(sr_ys))
                l1.SetLineColor(root.kOrange + 1); l1.SetLineWidth(3); l1.SetLineStyle(2)
                lines.append(l1)
                l2 = root.TLine(SR_CX, min(cr_ys), SR_CX, min(sr_ys))
                l2.SetLineColor(root.kGreen + 3); l2.SetLineWidth(3); l2.SetLineStyle(3)
                lines.append(l2)
            # Upper vertical (SR → top)
            if max(sr_ys) < max(cr_ys):
                l1 = root.TLine(SR_CX, max(sr_ys), SR_CX, max(cr_ys))
                l1.SetLineColor(root.kOrange + 1); l1.SetLineWidth(3); l1.SetLineStyle(2)
                lines.append(l1)
                l2 = root.TLine(SR_CX, max(sr_ys), SR_CX, max(cr_ys))
                l2.SetLineColor(root.kGreen + 3); l2.SetLineWidth(3); l2.SetLineStyle(3)
                lines.append(l2)

    sr_h = sr_crossings.get('horizontal', [])
    cr_h = crvr_crossings.get('horizontal', [])
    if sr_h and cr_h:
        sr_xs = sorted([p[0] for p in sr_h])
        cr_xs = sorted([p[0] for p in cr_h])
        if cr_xs and sr_xs:
            # Left horizontal (SR → left)
            if min(cr_xs) < min(sr_xs):
                l1 = root.TLine(min(cr_xs), SR_CY, min(sr_xs), SR_CY)
                l1.SetLineColor(root.kOrange + 1); l1.SetLineWidth(3); l1.SetLineStyle(2)
                lines.append(l1)
                l2 = root.TLine(min(cr_xs), SR_CY, min(sr_xs), SR_CY)
                l2.SetLineColor(root.kGreen + 3); l2.SetLineWidth(3); l2.SetLineStyle(3)
                lines.append(l2)
            # Right horizontal (SR → right)
            if max(sr_xs) < max(cr_xs):
                l1 = root.TLine(max(sr_xs), SR_CY, max(cr_xs), SR_CY)
                l1.SetLineColor(root.kOrange + 1); l1.SetLineWidth(3); l1.SetLineStyle(2)
                lines.append(l1)
                l2 = root.TLine(max(sr_xs), SR_CY, max(cr_xs), SR_CY)
                l2.SetLineColor(root.kGreen + 3); l2.SetLineWidth(3); l2.SetLineStyle(3)
                lines.append(l2)

    return lines


def draw_region_contours():
    """
    Draw regions matching HDBS-2022-02 Figure 2 and analysis.py exactly:

      - Red solid:     SR contour (boosted_SR = 1.6, full)
      - Orange dashed: CRVR = 170 in anti-diagonal quadrants (= VR)
      - Green dotted:  CRVR = 170 in diagonal quadrants (= CR)
      - Quadrant divider lines between SR and CRVR outer boundary:
            upper vertical:    orange dashed (connects VR arcs)
            lower vertical:    green dotted  (connects CR arcs)
            left horizontal:   green dotted  (connects CR arcs)
            right horizontal:  orange dashed (connects VR arcs)

    analysis.py uses a SINGLE CRVR < 170 boundary for both VR and CR,
    distinguished only by which quadrant the event falls in.

    Returns dict of ROOT objects (must keep alive).
    """
    objects = {}

    # ---- SR contour (full, red solid) ----
    sr_segs, sr_label_info = _make_labeled_contour(vbf_sr_value, SR_CUT, "SR")
    sr_graphs = _segments_to_tgraphs(sr_segs)
    objects['sr_graphs'] = sr_graphs
    _draw_graphs(sr_graphs, root.kRed + 1, 3, 1)

    if not sr_label_info:
        sr_label_info = [(SR_CX - 9, SR_CY - 13, 0)]
    objects['labels'] = _make_labels(sr_label_info, "SR", root.kRed + 1, 0.035)

    """ 
    # ---- Single CRVR = 170 contour, split by quadrant ----
    crvr_segs, crvr_label_info = _make_labeled_contour(
        crvr_value, CRVR_CUT, "", gap_frac=0.0  # no inline label, we add our own
    )

    # VR: CRVR=170 in anti-diagonal quadrants → orange dashed
    vr_segs = _clip_segments_to_quadrant(crvr_segs, SR_CX, SR_CY, "anti_diagonal")
    vr_graphs = _segments_to_tgraphs(vr_segs)
    objects['vr_graphs'] = vr_graphs
    _draw_graphs(vr_graphs, root.kOrange + 1, 3, 2)

    # CR: CRVR=170 in diagonal quadrants → green dotted
    cr_segs = _clip_segments_to_quadrant(crvr_segs, SR_CX, SR_CY, "diagonal")
    cr_graphs = _segments_to_tgraphs(cr_segs)
    objects['cr_graphs'] = cr_graphs
    _draw_graphs(cr_graphs, root.kGreen + 3, 3, 3)

    # VR labels in anti-diagonal quadrants
    objects['labels'] += _make_labels(
        [(95, 155, 45), (175, 90, -45)], "VR", root.kOrange + 1, 0.03
    )
    # CR labels in diagonal quadrants
    objects['labels'] += _make_labels(
        [(175, 190, 45), (80, 70, 45)], "CR", root.kGreen + 3, 0.03
    )

    # ---- Quadrant divider lines between SR and CRVR outer boundary ----
    sr_crossings = _find_contour_crossings(vbf_sr_value, SR_CUT, SR_CX, SR_CY)
    crvr_crossings = _find_contour_crossings(crvr_value, CRVR_CUT, SR_CX, SR_CY)
    quad_lines = _draw_quadrant_lines(sr_crossings, crvr_crossings)
    objects['quad_lines'] = quad_lines
    for line in quad_lines:
        line.Draw()
    
    """

    return objects


def calculate_region_efficiencies(hist_2d):
    total_events = hist_2d.Integral()
    sr_events = vr_events = cr_events = 0.0
    for i in range(1, hist_2d.GetNbinsX() + 1):
        for j in range(1, hist_2d.GetNbinsY() + 1):
            m1 = hist_2d.GetXaxis().GetBinCenter(i)
            m2 = hist_2d.GetYaxis().GetBinCenter(j)
            w = hist_2d.GetBinContent(i, j)
            if w == 0: continue
            sr_val = vbf_sr_value(m1, m2)
            crvr_val = crvr_value(m1, m2)
            if sr_val < SR_CUT:
                sr_events += w
            elif crvr_val < CRVR_CUT:
                anti_diag = (m1 > SR_CX and m2 < SR_CY) or (m1 < SR_CX and m2 > SR_CY)
                diag = (m1 > SR_CX and m2 > SR_CY) or (m1 < SR_CX and m2 < SR_CY)
                if anti_diag: vr_events += w
                elif diag: cr_events += w
    eff = lambda n: n / total_events if total_events > 0 else 0
    return {"total": total_events, "sr": sr_events, "sr_eff": eff(sr_events),
            "vr": vr_events, "vr_eff": eff(vr_events), "cr": cr_events, "cr_eff": eff(cr_events)}


def _get_tree_name(root_file_path):
    """Determine the TTree name in a ROOT file."""
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


def plot_mass_plane(root_file_paths, output_dir, config_label, mc_campaign, tagger, jet_mode):
    """
    Create the 2D mass plane plot for a single configuration.
    root_file_paths: str or list of str (multiple files will be chained).
    """
    if isinstance(root_file_paths, str):
        root_file_paths = [root_file_paths]

    # Find tree name from first file
    tree_name = _get_tree_name(root_file_paths[0])
    if not tree_name:
        print(f"  ERROR: No TTree found in {root_file_paths[0]}"); return None

    # Build TChain
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
    weight_cmd = "mc_sf"

    print(f"  Drawing 2D histogram...")
    entries = chain.Draw(draw_cmd, f"({SELECTION_CUT})*{weight_cmd}", "COLZ")
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

    root.gStyle.SetPalette(root.kBird)
    hist_2d.Draw("COLZ")
    region_objects = draw_region_contours()
    eff = calculate_region_efficiencies(hist_2d)

    #ATLASLabel(0.18, 0.92, "Internal Simulation")
    #lumi_text = "#sqrt{s} = 13 TeV" if mc_campaign != "Combined" else "#sqrt{s} = 13 TeV, MC20+MC23"
    #myText(0.18, 0.87, lumi_text, color=root.kBlack)
    ATLASLabel(0.18, 0.87, "Working Progress")
    myText(0.15, 0.82, f"VBF HH #rightarrow 4b SM - {mc_campaign}", color=root.kBlack)
    myText(0.15, 0.77, f"Tagger: {tagger}, Mode: {jet_mode}", color=root.kBlack)
    myText(0.15, 0.72, f"SR: {100*eff['sr_eff']:.1f}%,  VR: {100*eff['vr_eff']:.1f}%,  CR: {100*eff['cr_eff']:.1f}%", color=root.kBlack)

    #legend = root.TLegend(0.52, 0.15, 0.84, 0.40)
    #legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(0.025)
    #if region_objects['sr_graphs']:
    #    legend.AddEntry(region_objects['sr_graphs'][0], f"SR (boosted_SR < {SR_CUT})", "l")
    #if region_objects['vr_graphs']:
    #    legend.AddEntry(region_objects['vr_graphs'][0], f"VR (CRVR < {CRVR_CUT:.0f}, anti-diag)", "l")
    #if region_objects['cr_graphs']:
    #    legend.AddEntry(region_objects['cr_graphs'][0], f"CR (CRVR < {CRVR_CUT:.0f}, diagonal)", "l")
    #legend.Draw()

    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, f"mass_plane_{config_label}")
    canvas.SaveAs(f"{output_base}.pdf"); canvas.SaveAs(f"{output_base}.png"); canvas.SaveAs(f"{output_base}.root")

    print(f"  Saved: {output_base}.pdf/.png/.root")
    print(f"  SR eff: {100*eff['sr_eff']:.1f}%,  VR eff: {100*eff['vr_eff']:.1f}%,  CR eff: {100*eff['cr_eff']:.1f}%")

    stats = {"config": config_label, "mc_campaign": mc_campaign, "tagger": tagger, "jet_mode": jet_mode,
             "entries": hist_2d.GetEntries(), "integral": hist_2d.Integral(),
             "mean_mH1": hist_2d.GetMean(1), "mean_mH2": hist_2d.GetMean(2),
             "rms_mH1": hist_2d.GetRMS(1), "rms_mH2": hist_2d.GetRMS(2),
             "sr_events": eff["sr"], "vr_events": eff["vr"], "cr_events": eff["cr"],
             "total_events": eff["total"],
             "sr_efficiency": eff["sr_eff"], "vr_efficiency": eff["vr_eff"], "cr_efficiency": eff["cr_eff"]}
    return stats


def save_comparison_table(all_stats, output_file):
    with open(output_file, 'w') as f:
        f.write("=" * 140 + "\n")
        f.write("Mass Plane Comparison - All Jet Kinematics Modes\n")
        f.write("=" * 140 + "\n\n")
        f.write("Selection: " + SELECTION_CUT + "\n")
        f.write(f"SR:  VBF formula (p1={SR_P1}, p2={SR_P2}, c=({SR_CX},{SR_CY})) < {SR_CUT}\n")
        f.write(f"VR:  CRVR formula (p={CRVR_P1}) < {CRVR_CUT}  [anti-diagonal quadrants]\n")
        f.write(f"CR:  CRVR formula (p={CRVR_P1}) < {CRVR_CUT}  [diagonal quadrants]\n\n")
        f.write(f"{'Configuration':<40} {'Entries':>10} {'Mean mH1':>10} {'Mean mH2':>10} "
                f"{'RMS mH1':>10} {'RMS mH2':>10} {'SR Eff':>10} {'VR Eff':>10} {'CR Eff':>10}\n")
        f.write("-" * 140 + "\n")
        all_mc = MC_CAMPAIGNS + ["Combined"]
        for mc in all_mc:
            f.write(f"\n{mc}:\n")
            for stat in all_stats:
                if stat and stat["mc_campaign"] == mc:
                    f.write(f"  {stat['tagger']}_{stat['jet_mode']:<30} "
                            f"{stat['entries']:>10.0f} {stat['mean_mH1']:>10.2f} {stat['mean_mH2']:>10.2f} "
                            f"{stat['rms_mH1']:>10.2f} {stat['rms_mH2']:>10.2f} "
                            f"{100*stat['sr_efficiency']:>9.1f}% {100*stat['vr_efficiency']:>9.1f}% {100*stat['cr_efficiency']:>9.1f}%\n")
        f.write("\n" + "=" * 140 + "\n\nComparison by Jet Kinematics Mode:\n" + "-" * 100 + "\n")
        for jet_mode in JET_MODES:
            f.write(f"\n{jet_mode}:\n")
            for stat in all_stats:
                if stat and stat["jet_mode"] == jet_mode:
                    delta_mH1 = stat['mean_mH1'] - SR_CX; delta_mH2 = stat['mean_mH2'] - SR_CY
                    f.write(f"  {stat['mc_campaign']}_{stat['tagger']}: "
                            f"mH1 = {stat['mean_mH1']:.2f} (d={delta_mH1:+.2f}), "
                            f"mH2 = {stat['mean_mH2']:.2f} (d={delta_mH2:+.2f}), "
                            f"SR = {100*stat['sr_efficiency']:.1f}%, "
                            f"VR = {100*stat['vr_efficiency']:.1f}%, "
                            f"CR = {100*stat['cr_efficiency']:.1f}%\n")
        f.write("\n" + "=" * 140 + "\nEnd of Comparison\n" + "=" * 140 + "\n")
    print(f"\nComparison table saved to: {output_file}")


def create_summary_plot(all_stats, output_dir):
    valid_stats = [s for s in all_stats if s is not None]
    if not valid_stats: print("No valid statistics to plot"); return
    canvas = root.TCanvas("canvas_summary", "Region Efficiency Comparison", 1400, 600)
    canvas.cd()
    canvas.SetLeftMargin(0.08); canvas.SetRightMargin(0.03)
    canvas.SetBottomMargin(0.22); canvas.SetTopMargin(0.08)
    n_configs = len(valid_stats)
    h_sr = root.TH1D("h_sr_eff", "", n_configs, 0, n_configs)
    h_vr = root.TH1D("h_vr_eff", "", n_configs, 0, n_configs)
    h_cr = root.TH1D("h_cr_eff", "", n_configs, 0, n_configs)
    for i, stat in enumerate(valid_stats):
        label = f"{stat['mc_campaign']}_{stat['tagger']}_{stat['jet_mode']}"
        h_sr.GetXaxis().SetBinLabel(i + 1, label)
        h_sr.SetBinContent(i + 1, 100 * stat['sr_efficiency'])
        h_vr.SetBinContent(i + 1, 100 * stat['vr_efficiency'])
        h_cr.SetBinContent(i + 1, 100 * stat['cr_efficiency'])
    h_sr.SetFillColor(root.kRed - 7); h_sr.SetLineColor(root.kRed + 1)
    h_sr.GetYaxis().SetTitle("Efficiency [%]"); h_sr.GetYaxis().SetRangeUser(0, 100)
    h_sr.GetXaxis().SetLabelSize(0.03); h_sr.GetXaxis().LabelsOption("v")
    h_vr.SetFillColor(root.kOrange - 3); h_vr.SetLineColor(root.kOrange + 2)
    h_cr.SetFillColor(root.kGreen - 6); h_cr.SetLineColor(root.kGreen + 2)
    hs = root.THStack("hs_eff", "")
    hs.Add(h_sr); hs.Add(h_vr); hs.Add(h_cr); hs.Draw("BAR")
    hs.GetYaxis().SetTitle("Efficiency [%]"); hs.GetYaxis().SetRangeUser(0, 100)
    hs.GetXaxis().SetLabelSize(0.03)
    for i in range(1, n_configs + 1): hs.GetXaxis().SetBinLabel(i, h_sr.GetXaxis().GetBinLabel(i))
    hs.GetXaxis().LabelsOption("v")
    leg = root.TLegend(0.70, 0.75, 0.95, 0.90)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    leg.AddEntry(h_sr, "SR", "f"); leg.AddEntry(h_vr, "VR", "f"); leg.AddEntry(h_cr, "CR", "f")
    leg.Draw()
    ATLASLabel(0.12, 0.92, "Internal Simulation")
    myText(0.12, 0.87, "VBF HH #rightarrow 4b SM - Region Efficiency Comparison", color=root.kBlack)
    output_base = os.path.join(output_dir, "region_efficiency_comparison")
    canvas.SaveAs(f"{output_base}.pdf"); canvas.SaveAs(f"{output_base}.png")
    print(f"Summary plot saved to: {output_base}.pdf/.png")


def main():
    print("=" * 70)
    print("Mass Plane Plotting - All Configurations")
    print("=" * 70)
    print(f"SR:  VBF formula (p1={SR_P1}, p2={SR_P2}, c=({SR_CX},{SR_CY})) < {SR_CUT}")
    print(f"VR:  CRVR formula (p={CRVR_P1}) < {CRVR_CUT} [anti-diagonal quadrants]")
    print(f"CR:  CRVR formula (p={CRVR_P1}) < {CRVR_CUT} [diagonal quadrants]")

    if not os.path.exists(BASE_DIR): print(f"ERROR: Base directory not found: {BASE_DIR}"); return
    all_stats = []

    # ── Individual MC campaigns ──
    for mc_campaign in MC_CAMPAIGNS:
        for tagger in TAGGERS:
            for jet_mode in JET_MODES:
                config_label = f"{mc_campaign}_{tagger}_{jet_mode}"
                folder_name = get_folder_name(mc_campaign, tagger, jet_mode)
                folder_path = os.path.join(BASE_DIR, folder_name)
                print(f"\n{'=' * 70}\nProcessing: {config_label}\nFolder: {folder_path}")
                if not os.path.exists(folder_path):
                    print(f"  WARNING: Folder not found, skipping"); all_stats.append(None); continue
                root_file = find_merged_root_file(folder_path)
                if not root_file:
                    print(f"  WARNING: No ROOT file found, skipping"); all_stats.append(None); continue
                output_dir = os.path.join(folder_path, f"H_MassPlane_Plots_{config_label}")
                stats = plot_mass_plane(root_file, output_dir, config_label, mc_campaign, tagger, jet_mode)
                all_stats.append(stats)

    # ── Combined MC20+MC23 ──
    print(f"\n{'=' * 70}")
    print("Now processing combined MC20+MC23 configurations...")
    print(f"{'=' * 70}")
    for tagger in TAGGERS:
        for jet_mode in JET_MODES:
            config_label = f"Combined_{tagger}_{jet_mode}"
            print(f"\n{'=' * 70}\nProcessing: {config_label}")

            # Collect ROOT files from both campaigns
            combined_files = []
            for mc in MC_CAMPAIGNS:
                folder_name = get_folder_name(mc, tagger, jet_mode)
                folder_path = os.path.join(BASE_DIR, folder_name)
                if not os.path.exists(folder_path):
                    print(f"  WARNING: {folder_path} not found"); continue
                rf = find_merged_root_file(folder_path)
                if rf:
                    combined_files.append(rf)

            if not combined_files:
                print(f"  WARNING: No files found for combined, skipping")
                all_stats.append(None)
                continue

            print(f"  Merging {len(combined_files)} files:")
            for cf in combined_files:
                print(f"    - {cf}")

            output_dir = os.path.join(BASE_DIR, f"H_MassPlane_Plots_{config_label}")
            stats = plot_mass_plane(
                combined_files, output_dir, config_label,
                "Combined", tagger, jet_mode
            )
            all_stats.append(stats)
    comparison_file = os.path.join(BASE_DIR, "H_mass_plane_comparison.txt")
    save_comparison_table(all_stats, comparison_file)
    summary_output_dir = os.path.join(BASE_DIR, "H_Summary_Plots")
    os.makedirs(summary_output_dir, exist_ok=True)
    create_summary_plot(all_stats, summary_output_dir)
    print("\n" + "=" * 70 + "\nAll done!\n" + "=" * 70)


if __name__ == "__main__":
    main()