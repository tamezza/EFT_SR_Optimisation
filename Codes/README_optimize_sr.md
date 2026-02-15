# Signal Region Optimisation for VBF Boosted HH → 4b

A complete framework for optimising the signal region definition in the ATLAS search for non-resonant Higgs boson pair production via vector boson fusion (VBF) in the fully hadronic boosted channel, following ATLAS HDBS-2022-02.

---

## Table of contents

1. [Physics motivation](#1-physics-motivation)
2. [The VBF HH → 4b channel](#2-the-vbf-hh--4b-channel)
3. [Event selection](#3-event-selection)
4. [Signal region definition](#4-signal-region-definition)
5. [Optimisation procedure](#5-optimisation-procedure)
6. [Monte Carlo weights](#6-monte-carlo-weights)
7. [Code overview](#7-code-overview)
8. [Output plots](#8-output-plots)
9. [Usage](#9-usage)
10. [Full CLI reference](#10-full-cli-reference)
11. [Companion scripts](#11-companion-scripts)
12. [Dependencies](#12-dependencies)

---

## 1. Physics motivation

### Higgs boson pair production

The discovery of the Higgs boson (h) at the LHC in 2012 opened the door to measuring its self-coupling, one of the last untested predictions of the Standard Model (SM). The trilinear self-coupling λ_HHH (often written κ_λ when normalised to the SM value) governs the shape of the Higgs potential and is directly accessible only through Higgs pair (HH) production.

At the LHC, HH production proceeds through two main mechanisms:

- **Gluon-gluon fusion (ggF):** Dominant production mode. A top-quark loop mediates gg → HH. Sensitive to the top Yukawa coupling (κ_t) and the trilinear self-coupling (κ_λ).

- **Vector boson fusion (VBF):** Two incoming quarks each radiate a W or Z boson, which then scatter to produce HH. The VBF process is uniquely sensitive to the quartic VVHH coupling (κ_2V), and also to the trilinear coupling (κ_λ) and the VVH coupling (κ_V). While the VBF cross-section is roughly 20× smaller than ggF in the SM, deviations in κ_2V can dramatically enhance it.

### Why κ_2V = 0?

In the SM, κ_2V = 1. If κ_2V deviated from 1, unitarity cancellations in longitudinal VV → HH scattering would be broken, causing the VBF HH cross-section to grow with energy. The point κ_2V = 0 is particularly interesting because:

- It maximises the VBF HH cross-section enhancement (the destructive interference between diagrams that keeps the SM cross-section small is removed).
- It serves as the primary benchmark for setting exclusion limits.
- In the coupling label convention, κ_2V = 0 corresponds to `l1cvv0cv1` (λ=1, c_VV=0, c_V=1).

The SM point is `l1cvv1cv1` (λ=1, c_VV=1, c_V=1).

### The EFT framework

The VBF HH process can be parameterised in an effective field theory (EFT) framework where the amplitude depends on three couplings: κ_λ, κ_V, and κ_2V. The differential cross-section is a polynomial in these couplings:

```
σ = A₁·κ_V⁴ + A₂·κ_2V² + A₃·κ_V²·κ_2V + A₄·κ_V²·κ_λ² + ...
```

Each term (quadratic, interference, SM) is generated as a separate Monte Carlo sample. The samples in the directory tree are labelled accordingly:

| Folder prefix | Physics content |
|---|---|
| `Kappa2V` | VBF HH signal at specific κ_2V values (l1cvv0cv1 = κ_2V=0, l1cvv1cv1 = SM) |
| `SM` | SM VBF HH (κ_λ=1, κ_V=1, κ_2V=1) |
| `quad_M0_S1` | Quadratic EFT terms (e.g. M₀ and S₁ operators) |
| `QCD` | QCD multijet background (JZ slices, weighted by cross-section) |
| `ttbar` | Top-quark pair production background |

---

## 2. The VBF HH → 4b channel

### Decay channel

Both Higgs bosons decay to b-quark pairs: HH → (bb)(bb) → 4b. This is the dominant decay mode (BR ≈ 33%) but suffers from enormous QCD multijet background.

### Boosted topology

At high transverse momentum (p_T ≳ 250 GeV), the two b-quarks from each H → bb decay are collimated into a single large-radius (large-R) jet. The analysis reconstructs two such "fat jets," each identified as a Higgs candidate using:

- **Large-R jet mass:** The invariant mass of the jet should be consistent with m_H ≈ 125 GeV.
- **B-tagging:** Sub-jets within the large-R jet are b-tagged using flavour-tagging algorithms (GN2X or GN3PV01 — graph neural networks trained on track and vertex information).

The two Higgs candidates are labelled H₁ (leading, highest p_T) and H₂ (sub-leading).

### VBF topology

VBF events have a distinctive signature: two forward jets (VBF jets) at high rapidity with a large rapidity gap between them:

- **m_VBFjj > 400 GeV:** The invariant mass of the VBF dijet system must be large, reflecting the high virtuality of the exchanged vector bosons.
- **Δη_VBFjj > 2:** The two VBF jets must be well-separated in pseudorapidity, characteristic of t-channel boson exchange.

---

## 3. Event selection

The selection applied before SR optimisation consists of two parts:

### Boosted VBF preselection

```
pass_boosted_vbf_sel == 1
```

This flag encapsulates the full preselection chain: trigger, large-R jet quality, kinematics, VBF jet identification, and overlap removal. It is computed upstream during ntuple production.

### B-tag veto

```
boosted_vbf_has_btag == 0
```

Events where the VBF jets are b-tagged are rejected. B-tagged VBF jets likely originate from gluon splitting (g → bb) rather than from light quarks in VBF, so vetoing them enriches the VBF topology.

### VBF topology cuts

```
boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2
```

These further enforce the VBF topology. The invariant mass and rapidity gap cuts suppress ggF contamination and QCD multijet background, both of which lack the characteristic forward-jet topology.

---

## 4. Signal region definition

### The mass plane

After selecting two Higgs candidates, the event is characterised by the pair (m_H1, m_H2). Signal events cluster around (125, 125) GeV, while backgrounds are broadly distributed.

### The elliptical discriminant

The signal region is defined by an elliptical cut on the mass plane using the discriminant:

```
D = √[ (m₁(m₁ − c₁) / p₁)² + (m₂(m₂ − c₂) / p₂)² ]
```

Events with **D < cut** are inside the signal region. This defines a closed contour (approximately elliptical near the peak) centred on (c₁, c₂) with widths controlled by (p₁, p₂).

### Physical meaning of the parameters

| Parameter | Formula | Meaning |
|---|---|---|
| c₁ | μ₁ from Gaussian fit | Peak position of m_H1 distribution [GeV] |
| c₂ | μ₂ from Gaussian fit | Peak position of m_H2 distribution [GeV] |
| σ₁ | σ from Gaussian fit | Width (resolution) of m_H1 [GeV] |
| σ₂ | σ from Gaussian fit | Width (resolution) of m_H2 [GeV] |
| p₁ | σ₁ × c₁ | Resolution parameter [GeV²] |
| p₂ | σ₂ × c₂ | Resolution parameter [GeV²] |
| cut | Maximises S/√B | Discriminant threshold |

Why p = σ × c instead of just σ? The factor of m₁(m₁ − c₁) in the discriminant means the variable being normalised is not simply (m₁ − c₁) but m₁(m₁ − c₁), which has units of GeV². Dividing by p = σ·c normalises this quantity to be dimensionless and O(1) near the peak, ensuring the discriminant treats both Higgs masses on an equal footing regardless of their absolute resolution.

### Default parameters (HDBS-2022-02)

| Parameter | Value |
|---|---|
| c₁ | 124 GeV |
| c₂ | 117 GeV |
| p₁ | 1500 GeV² |
| p₂ | 1900 GeV² |
| cut | 1.6 |

The sub-leading Higgs has a lower centre (117 vs 124 GeV) and worse resolution because it has lower p_T on average, leading to a broader, more asymmetric mass peak.

---

## 5. Optimisation procedure

### Step 1–2: Iterative Gaussian fitting

For each of m_H1 and m_H2, an iterative procedure determines the peak position (c) and width (σ):

**Initial seed (data-driven):**
Before fitting, the code histograms the weighted mass distribution inside the fit range and takes the bin with the maximum count as the initial μ seed. This means the starting point adapts automatically to the data — if a new tagger or jet calibration shifts the peak, the seed shifts with it. The fallback values (124 GeV for m_H1, 117 GeV for m_H2) are only used if the histogram is empty.

**Pass 1:**
1. Compute a weighted histogram in a broad seed range (e.g. [90, 160] GeV for m_H1).
2. Fit a Gaussian G(x) = A · exp(−(x−μ)²/2σ²) using scipy's curve_fit with Sumw2 errors.
3. Refit in progressively narrower windows: μ ± Nσ for N = 1.0, 1.5, 2.0, 2.5, 3.0.

**Pass 2 (if needed):**
If the best candidate's μ or σ shifted by more than 1 GeV from the initial seed, repeat the window scan using the updated values as the new seed. This handles cases where the initial range was not well-centred.

**Best fit selection:**
From all candidates (both passes), filter out unphysical results (σ outside [3, 40] GeV, μ outside [80, 160] GeV), then select the fit with χ²/ndf closest to 1.0.

The rationale for choosing χ²/ndf ≈ 1: too narrow a window gives few bins and an artificially good χ² but is not representative of the peak shape; too wide a window includes non-Gaussian tails and gives a poor χ². The window where the Gaussian approximation is best gives χ²/ndf closest to 1.

### Step 3: Cut value scan

Using the fitted (c₁, c₂, p₁, p₂), compute D for every signal and background event, then scan the cut threshold from 0.5 to 4.0 in steps of 0.05. At each cut value, compute:

| Metric | Formula | Meaning |
|---|---|---|
| S | Σ w_sig (D < cut) | Expected signal yield in SR |
| B | Σ w_bkg (D < cut) | Expected background yield in SR |
| S/√B | S / √B | Approximate discovery significance |
| S/B | S / B | Signal-to-background ratio |
| Signal efficiency | S / S_total | Fraction of signal events inside SR |

The optimal cut maximises S/√B. If no background is available, the cut targets 76% signal efficiency (a heuristic from HDBS-2022-02).

### Why S/√B?

For a counting experiment with S signal events and B background events, the statistical significance of observing the signal above the background-only hypothesis is approximately S/√B in the Gaussian limit (B >> 1). This is the leading-order approximation to the profile likelihood ratio. Maximising S/√B gives the best expected discovery sensitivity.

More precisely, the expected significance is √(2·((S+B)·ln(1+S/B) − S)), which reduces to S/√B when S << B.

### Step 4: Joint optimisation (optional)

Takes the Step 1–3 result as a starting point and simultaneously varies all five parameters (c₁, c₂, p₁, p₂, cut) using the Nelder-Mead simplex algorithm to maximise S/√B. This can find a slightly better solution because the parameters are correlated (e.g. shifting c₁ might allow a tighter cut), but in practice the improvement is usually small (a few percent).

Bounded region enforced: c₁, c₂ ∈ [80, 160] GeV, p₁, p₂ > 100 GeV², cut ∈ [0.3, 5.0].

---

## 6. Monte Carlo weights

### Event weight composition

Each MC event carries a weight that converts raw event counts into physical predictions:

```
w_event = mc_sf × luminosity_boosted × pileupweight_boosted
```

| Branch | Units | Meaning |
|---|---|---|
| `mc_sf` | fb | generatorWeight × σ_xsec × k-factor × BR × filter_eff / Σw_init |
| `luminosity_boosted` | fb⁻¹ | Integrated luminosity for the boosted trigger GRL |
| `pileupweight_boosted` | dimensionless | Per-event pileup reweighting factor |

The product mc_sf × luminosity gives expected events (fb × fb⁻¹ = dimensionless), and pileup weight corrects for the difference in pileup profile between MC and data.

### Why mc_sf alone is sufficient for SR optimisation

The figure of merit is S/√B, where S and B are sums of event weights:

```
S/√B = Σ w_sig / √(Σ w_bkg)
```

If full weight is used, every w becomes mc_sf × L × pw, where L is the luminosity (constant for all events in a campaign). Then:

```
S_full / √B_full = (L · Σ mc_sf_sig · pw) / √(L · Σ mc_sf_bkg · pw)
                  = √L · (Σ mc_sf_sig · pw) / √(Σ mc_sf_bkg · pw)
```

The √L is a global prefactor — it shifts the entire S/√B curve vertically but does not change which cut value is optimal. The pileup weight pw is a per-event correction that adjusts the pileup spectrum; it is uncorrelated with the Higgs mass, so it does not shift the Gaussian peak (μ, σ) and has negligible impact on the optimal SR boundary.

**Use full weight for:** absolute yield estimation, cutflow tables, data/MC comparisons, inputs to statistical fitting.

**Use mc_sf only for:** SR optimisation (this script), mass peak fitting, shape comparisons, relative efficiencies.

---

## 7. Code overview

### Scripts

| Script | Lines | Purpose |
|---|---|---|
| `optimize_sr_v2.py` | ~2230 | Main SR optimisation: fits mass peaks, scans cut, produces all plots |
| `estimate_sr_yields.py` | ~600 | Expected event yields in SR with full physics weight |
| `fit_mass_peaks.py` | ~780 | Standalone mass peak fitting (reference implementation) |
| `plot_mass_plane_SR_only.py` | — | SR contour for every signal coupling (ROOT-based) |
| `mass_plane_variants.py` | — | Full analysis geometry: SR + VR + CR + quadrant lines (ROOT-based) |
| `run_all_fits.sh` | — | Batch shell script for running fits across all configurations |
| `NormCross_EFT_VBF_Plot.py` | — | EFT normalised cross-section plotting (ROOT-based) |

### Version history

| Version | File | Key changes |
|---|---|---|
| v0 (original) | `optimize_sr.py` | Original SR optimiser. Basic Gaussian fit, cut scan, ROOT-style plots. |
| v1 (improved) | `optimize_sr_improved.py` | ATLAS-style matplotlib plots, pull panels, mass plane with SR contours, marginal projections. |
| v2 (current) | `optimize_sr_v2.py` | Aligned iterative fitting with `fit_mass_peaks.py` reference. Added: data-driven μ seed from histogram peak, configurable weight handling (`--full-weight`), GN3PV01↔GN3XPV01 naming, `--signal-coupling` for κ₂V=0, `--bkg-processes` filter, `--show-bkg-contours` flag, `SetEstimate()` segfault fix, three plot types per fit (raw, best+pull, all-fits), standalone per-Higgs fit plots, standalone S/√B cut scan, standalone mass plane, signal+background mass plane with marginal histograms. |

### Directory structure expected

```
BaristaSAMPLES/
├── MC20/
│   ├── Kappa2V_GN3PV01_DEFAULT/RootSamp/
│   │   ├── boosted_skim_VBFhh_l1cvv0cv1_mc20__Nominal.root  ← signal (κ₂V=0)
│   │   └── boosted_skim_VBFhh_l1cvv1cv1_mc20__Nominal.root  ← SM coupling
│   ├── QCD_GN3XPV01_DEFAULT/RootSamp/
│   │   └── boosted_*_JZ_JZ*_*_fullsim_mc20__Nominal.root     ← QCD multijet
│   ├── ttbar_GN3XPV01_DEFAULT/RootSamp/
│   │   └── boosted_*_ttbar_*_fullsim_mc20__Nominal.root       ← ttbar
│   ├── SM_GN3PV01_DEFAULT/RootSamp/
│   ├── quad_M0_S1_GN3PV01_DEFAULT/RootSamp/
│   └── ...
├── MC23/
│   └── (same structure, GN3PV01 folders may be named GN3XPV01)
```

**Naming quirks handled:**
- GN3PV01 ↔ GN3XPV01 tagger name inconsistency between MC20 and MC23
- Signal file selected by `--signal-coupling` (default `l1cvv0cv1`) to pick the right file when a folder contains multiple coupling points
- Background discovery by explicit `--bkg-processes` list (default: QCD, ttbar)

### Processing flow

```
┌──────────────────────────────────────────────┐
│ For each (MC campaign × tagger × jet mode):  │
│                                              │
│  1. Find signal ROOT file (Kappa2V folder,   │
│     l1cvv0cv1 coupling)                      │
│                                              │
│  2. Find background ROOT files (QCD + ttbar  │
│     folders, all files merged)               │
│                                              │
│  3. Read m_H1, m_H2, mc_sf from TTrees      │
│     with selection cuts applied              │
│                                              │
│  4. Fit Gaussians to m_H1, m_H2             │
│     → (c₁, σ₁), (c₂, σ₂)                   │
│     → p₁ = σ₁·c₁, p₂ = σ₂·c₂              │
│                                              │
│  5. Scan cut value to maximise S/√B          │
│     → optimal cut                            │
│                                              │
│  6. (Optional) joint 5-parameter optimise    │
│                                              │
│  7. Produce plots and save results           │
└──────────────────────────────────────────────┘
```

---

## 8. Output plots

For each configuration, the following plots are produced:

### Per-Higgs standalone fit

| File | Description |
|---|---|
| `fit_mH1_{label}.pdf` | m_H1 Gaussian fit with pull panel. Shows data as step histogram, fit curve, fit range shading, fitted μ and σ with uncertainties, χ²/ndf, derived p₁ = σ₁ × c₁ value, and default c₁ reference line. |
| `fit_mH2_{label}.pdf` | Same for m_H2. |

### Combined mass fits

| File | Description |
|---|---|
| `mass_fits_{label}.pdf` | Both m_H1 and m_H2 fits side-by-side on one figure. |
| `raw_distributions_{label}.pdf` | Raw histograms with no fit overlay. |
| `all_fits_overview_{label}.pdf` | Every Gaussian candidate overlaid in different colours, best one highlighted. Useful for seeing how the iterative window selection works. |

### Cut scan

| File | Description |
|---|---|
| `cut_scan_SoverSqrtB_{label}.pdf` | Standalone S/√B vs cut with optimal point marked, SR parameter box annotation, and S/√B values at both default and optimal cuts. |
| `cut_scan_{label}.pdf` | Three-panel plot: S/√B, S/B, and signal efficiency vs cut value. |

### Mass plane

**Background contours:** By default, mass plane plots show only the signal density and SR contours. Use `--show-bkg-contours` to overlay red isolines of the QCD+ttbar 2D density at 10%, 30%, 50%, 70%, and 90% of the background peak. The innermost contour (90%) shows where background is most concentrated; the outermost (10%) traces the broad tail.

| File | Description |
|---|---|
| `massplane_{label}.pdf` | **Standalone mass plane.** Clean 2D (m_H1, m_H2) plot with signal density (blue colourmap), background density contours (red), fitted and default SR contours, centre markers, and SR parameter annotation box. No marginal histograms. |
| `massplane_sig_bkg_{label}.pdf` | Signal + background mass plane with marginal 1D histograms on top (m_H1) and right (m_H2) showing both signal (blue) and background (red, scaled). SR contours overlaid. |
| `massplane_optimised_{label}.pdf` | Signal-only 2D mass plane with up to three SR contours overlaid: default (grey dashed), fitted (red solid), joint if available (green dash-dot). Marginal projections on top and right. |

### Data files

| File | Description |
|---|---|
| `mass_histogram_mH{1,2}_{label}.csv` | Bin centres, counts, errors, fit values, pull. Machine-readable. |
| `sr_optimisation_results.json` | All fitted and joint parameters, signal efficiencies, per-configuration. |

---

## 9. Usage

### Quick start: run everything

```bash
bash run_all_fits.sh
```

This runs both `optimize_sr_v2.py` (full SR optimisation with background) and `fit_mass_peaks.py` (standalone signal-only fits) across all MC campaigns × taggers × jet modes. Edit the configuration block at the top of the script to change:

```bash
SAMPLE="Kappa2V"              # Signal folder prefix (Kappa2V, SM, quad_M0_S1)
SIGNAL_COUPLING="l1cvv0cv1"   # κ₂V=0 (use l1cvv1cv1 for SM)
BKG_PROCESSES="QCD ttbar"     # Background processes
```

To run only one part:

```bash
bash run_all_fits.sh --optimize-only   # SR optimisation only
bash run_all_fits.sh --fits-only       # Standalone fits only
```

### Individual scripts

#### SR optimisation (single configuration)

```bash
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --output-dir ./results \
    --mc-campaigns MC20 \
    --taggers GN3PV01 \
    --jet-modes DEFAULT
```

#### SR optimisation (all configurations)

```bash
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --output-dir ./results
```

This loops over MC20 × MC23 × Combined, GN2X × GN3PV01, DEFAULT × bjr_v00 × bjr_v01 (18 configurations total) and produces a summary comparison at the end.

#### With joint optimisation

```bash
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --output-dir ./results \
    --do-joint
```

#### SM signal instead of κ₂V = 0

```bash
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --signal-label SM \
    --signal-coupling l1cvv1cv1 \
    --output-dir ./results
```

#### Full physics weight (for absolute S/√B)

```bash
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --output-dir ./results \
    --full-weight
```

Note: the optimal parameters will be identical to the default mode. Only the absolute S, B, S/√B numbers change.

#### With background contours on mass plane

```bash
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --output-dir ./results \
    --mc-campaigns MC20 \
    --taggers GN3PV01 \
    --jet-modes DEFAULT \
    --show-bkg-contours
```

#### Standalone mass peak fits

```bash
python fit_mass_peaks.py \
    --signal-file ../BaristaSAMPLES/MC20/Kappa2V_GN3PV01_DEFAULT/RootSamp/boosted_skim_VBFhh_l1cvv0cv1_mc20__Nominal.root \
    --config-label MC20_GN3PV01_DEFAULT \
    --output-dir ./fits/MC20_GN3PV01_DEFAULT
```

#### SR yield estimation

```bash
# With default SR parameters:
python estimate_sr_yields.py \
    --base-dir ../BaristaSAMPLES \
    --mc-campaign MC20 \
    --tagger GN3PV01

# With optimised cut:
python estimate_sr_yields.py \
    --base-dir ../BaristaSAMPLES \
    --cut 1.40

# From optimizer JSON:
python estimate_sr_yields.py \
    --base-dir ../BaristaSAMPLES \
    --sr-json ./results/summary/sr_optimisation_results.json \
    --sr-config MC20_GN3PV01_DEFAULT
```

#### Mass plane plots

```bash
# Full analysis geometry — SR + VR + CR + quadrant dividers:
python mass_plane_variants.py

# SR contour for every signal coupling (SM, κ₂V=0,1,2, EFT):
python plot_mass_plane_SR_only.py
```

Both are ROOT-based, self-contained, and loop over all MC × tagger × jet_mode automatically.

---

## 10. Full CLI reference

| Argument | Default | Description |
|---|---|---|
| `--base-dir` | `.` | Parent directory containing MC20/ and MC23/ |
| `--bkg-dir` | (same as base-dir) | Background search directory |
| `--output-dir` | `.` | Where to write plots and results |
| `--mc-campaigns` | MC20 MC23 Combined | MC campaigns to process |
| `--taggers` | GN2X GN3PV01 | Flavour taggers to loop over |
| `--jet-modes` | DEFAULT bjr_v00 bjr_v01 | Jet kinematics modes |
| `--signal-label` | Kappa2V | Signal sample folder prefix |
| `--signal-coupling` | l1cvv0cv1 | Coupling tag in ROOT filename |
| `--bkg-processes` | QCD ttbar | Background folder prefixes |
| `--do-joint` | off | Enable joint 5-parameter optimisation |
| `--full-weight` | off | Use mc_sf × luminosity × pileup |
| `--show-bkg-contours` | off | Draw background density contours (10%–90% of peak) on mass plane plots. Contours are isolines of the QCD+ttbar 2D density; innermost = 90% of peak, outermost = 10%. Useful for seeing where background concentrates relative to the SR ellipse. |
| `--selection` | pass_boosted_vbf_sel == 1 && ... | TTree selection cut |
| `--weight-var` | mc_sf | Weight branch name |
| `--lumi-var` | luminosity_boosted | Luminosity branch name |
| `--pileup-var` | pileupweight_boosted | Pileup weight branch name |

---

## 11. Companion scripts

### run_all_fits.sh

Master batch script that orchestrates both `optimize_sr_v2.py` and `fit_mass_peaks.py` across all configurations. Handles signal coupling filtering (`l1cvv0cv1` for κ₂V=0), GN3PV01/GN3XPV01 naming, combined MC20+MC23 mode, and prints a summary with success/skip counts.

```bash
bash run_all_fits.sh                    # Run everything
bash run_all_fits.sh --fits-only        # Only standalone fits
bash run_all_fits.sh --optimize-only    # Only SR optimisation
```

Edit the configuration block at the top to change signal sample, coupling, backgrounds, or campaigns.

### fit_mass_peaks.py

Standalone reference implementation of the iterative Gaussian fitting procedure. Produces three plots per sample: raw distribution, best fit with pull panel, and all-fits overview. Saves fit results to JSON.

Fixes applied: VBF topology cuts, `SetEstimate()` segfault fix, data-driven μ seed from histogram peak, aligned `FIT_RANGE_H2 = (100, 150)`.

```bash
python fit_mass_peaks.py \
    --signal-file ../BaristaSAMPLES/MC20/Kappa2V_GN3PV01_DEFAULT/RootSamp/boosted_skim_VBFhh_l1cvv0cv1_mc20__Nominal.root \
    --config-label MC20_GN3PV01_DEFAULT \
    --output-dir ./fits/MC20_GN3PV01_DEFAULT
```

### estimate_sr_yields.py

#### Purpose

While `optimize_sr_v2.py` answers **"What's the best SR shape?"** (using mc_sf only), `estimate_sr_yields.py` answers **"How many events do I actually expect in the SR?"** using the full physics weight.

#### What it does, step by step

**1. Auto-discover all samples** — scans `base_dir/{mc_campaign}/` and finds every folder matching the tagger and jet mode, then categorises them:

| Folder prefix | Category | Meaning |
|---|---|---|
| `Kappa2V` | Signal | VBF HH at κ₂V=0 (filtered by `--signal-coupling l1cvv0cv1`) |
| `SM` | VBF_SM | SM VBF HH (κ₂V=1) |
| `quad_M0_S1` | VBF_Quad | Quadratic EFT terms |
| `QCD` | QCD | QCD multijet (all JZ slices merged) |
| `ttbar` | ttbar | Top-quark pair production |

**2. Read events with full weight** — for each sample, reads m_H1, m_H2 and computes:

```
weight = mc_sf × luminosity_boosted × pileupweight_boosted
         [fb]     [fb⁻¹]              [dimensionless]
       = expected number of events
```

This is the key difference from the optimiser — here you get real event counts (e.g. "3.2 signal events", "1547 QCD events"), not arbitrary shape-only numbers.

**3. Apply SR cut** — for every event, compute D and count weighted events inside the SR:

```
N_total = Σ weights              (all events after preselection)
N_SR    = Σ weights  [D < cut]   (events inside the signal region)
±stat   = √(Σ w²)   [D < cut]   (statistical uncertainty)
Eff     = N_SR / N_total
```

**4. Output a yield table:**

```
====================================================================================================
  Expected event yields in the Signal Region
  SR parameters: c₁=124.0, c₂=117.0, p₁=1500.0, p₂=1900.0, cut=1.60
  Weight: mc_sf × luminosity_boosted × pileupweight_boosted
====================================================================================================
Category     Process                N_raw        N_total           N_SR        ±stat    Eff(%)
----------------------------------------------------------------------------------------------------
Signal       Kappa2V                 1842         3.2145         1.9834    ±    0.1523    61.7%
----------------------------------------------------------------------------------------------------
VBF_SM       SM                       423         0.0412         0.0254    ±    0.0034    61.6%
----------------------------------------------------------------------------------------------------
QCD          QCD                  1487231      1547.3920       312.5510    ±   18.2340    20.2%
----------------------------------------------------------------------------------------------------
ttbar        ttbar                  52340        84.2150        18.7630    ±    2.1450    22.3%
====================================================================================================
SIGNAL (S)                                                       1.9834    ±    0.1523
BACKGROUND (B)                                                 331.3140    ±   18.3596

  S/√B = 0.1089
  S/B  = 0.005986
====================================================================================================
```

(Numbers are illustrative, not from real data.)

**5. Save outputs** — CSV and JSON with the same data, including S, B, S/√B summary. The JSON is structured for downstream consumption by statistical fitting frameworks (HistFitter, pyhf).

#### Column definitions

| Column | Definition |
|---|---|
| N_raw | Unweighted event count after preselection (how many MC events passed) |
| N_total | Σ(mc_sf × lumi × pileup) after preselection (expected events before SR cut) |
| N_SR | Σ(mc_sf × lumi × pileup) where D < cut (expected events inside SR) |
| ±stat | √(Σw²) inside SR — statistical uncertainty from finite MC |
| Eff(%) | N_SR / N_total — fraction of preselected events that fall inside the SR |

#### Typical workflow

```bash
# Step 1: find optimal SR parameters
python optimize_sr_v2.py \
    --base-dir ../BaristaSAMPLES \
    --mc-campaigns MC20 --taggers GN3PV01 --jet-modes DEFAULT

# Step 2: see how many events you expect with those optimised parameters
python estimate_sr_yields.py \
    --base-dir ../BaristaSAMPLES --mc-campaign MC20 --tagger GN3PV01 \
    --sr-json ./results/summary/sr_optimisation_results.json \
    --sr-config MC20_GN3PV01_DEFAULT

# Compare with default SR (HDBS-2022-02):
python estimate_sr_yields.py \
    --base-dir ../BaristaSAMPLES --mc-campaign MC20 --tagger GN3PV01

# Or manually set the cut:
python estimate_sr_yields.py \
    --base-dir ../BaristaSAMPLES --mc-campaign MC20 --tagger GN3PV01 --cut 1.40
```

The yield numbers are what you would report in an analysis note or feed into a statistical fitting framework as expected event counts per process.

### Mass plane scripts — comparison

There are three separate mass plane tools, each serving a different purpose:

| Feature | `mass_plane_variants.py` | `plot_mass_plane_SR_only.py` | `optimize_sr_v2.py` mass planes |
|---|---|---|---|
| **Purpose** | Show full analysis region geometry | Show SR for every signal coupling | Compare fitted vs default SR |
| **Regions drawn** | SR + VR + CR + quadrant lines | SR only | SR only (default + fitted + joint) |
| **Signal samples** | One (typically SM) | All: SM, Kappa2V (κ₂V=0,1,2), quad (M0, S1) | One (Kappa2V κ₂V=0 default) |
| **Background** | No | No | Yes (optional, `--show-bkg-contours`) |
| **SR parameters** | Default only (HDBS-2022-02) | Default only | Default + fitted from Gaussian fits |
| **Plotting backend** | ROOT (TH2F, TCanvas) | ROOT (TH2F, TCanvas) | Matplotlib |
| **Loops over** | MC × tagger × jet_mode | MC × tagger × jet_mode × signal × coupling | MC × tagger × jet_mode |

### mass_plane_variants.py

Full analysis geometry: draws SR, VR, and CR region contours with quadrant divider lines matching the bbbbarista VBF quadrant-based region definitions (HDBS-2022-02 Figure 2 style). VR is the CRVR < 170 boundary in anti-diagonal quadrants, CR in diagonal quadrants. This is the plot you'd put in a note to explain the region layout.

```bash
python mass_plane_variants.py
```

Runs from the code directory. Loops over all MC × tagger × jet_mode configurations, saves ROOT plots into sample directories, plus a comparison summary.

### plot_mass_plane_SR_only.py

SR contour for every signal coupling point. Loops over SM, Kappa2V (κ₂V=0, 1, 2), and quad_M0_S1 (M0_1, S1_10) — one mass plane per coupling, so you can see how different BSM signals populate the mass plane relative to the SR.

```bash
python plot_mass_plane_SR_only.py
```

Saves to `../Plots/SRonly/`. Useful for comparing signal kinematics across coupling points.

### NormCross_EFT_VBF_Plot.py

ROOT-based script for plotting normalised differential cross-sections of all EFT terms (SM, quadratic, interference) overlaid with QCD background. Uses `TTree::Draw` with mc_sf weighting.

---

### Cross-script consistency

All scripts now share the same configuration:

| Setting | Value | Scripts |
|---|---|---|
| Selection cut | `pass_boosted_vbf_sel == 1 && boosted_vbf_has_btag == 0 && boosted_m_vbfjj > 400 && boosted_dEta_vbfjj > 2` | All 6 Python scripts |
| SetEstimate fix | `tree.SetEstimate(n_pass)` before `GetV1()` | optimize_sr_v2, fit_mass_peaks, estimate_sr_yields |
| FIT_RANGE_H2 | (100, 150) | optimize_sr_v2, fit_mass_peaks |
| Signal coupling | `l1cvv0cv1` (κ₂V=0) | optimize_sr_v2, estimate_sr_yields, run_all_fits.sh |
| Data-driven μ seed | Histogram peak in fit range | optimize_sr_v2, fit_mass_peaks |

---

## 12. Dependencies

- **Python** ≥ 3.8
- **ROOT** (PyROOT) — for reading TTrees
- **NumPy** — array operations
- **SciPy** — `curve_fit` (Gaussian fitting), `minimize` (Nelder-Mead)
- **Matplotlib** — all plotting (ATLAS-style configuration)
