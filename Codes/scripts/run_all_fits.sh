#!/bin/bash
# run_all_fits.sh
# ===============
# Master script to run SR optimisation and/or standalone mass peak fits
# across all MC campaigns, taggers, and jet modes.
#
# Run from the parent directory containing MC20/ and MC23/.
#
# Usage:
#   bash run_all_fits.sh                    # Run both optimize + fit
#   bash run_all_fits.sh --fits-only        # Only standalone fits
#   bash run_all_fits.sh --optimize-only    # Only SR optimisation
#
# Configuration:
#   SAMPLE           : Signal sample folder prefix (Kappa2V, SM, quad_M0_S1)
#   SIGNAL_COUPLING  : Coupling tag in ROOT filename (l1cvv0cv1 = kappa2V=0, l1cvv1cv1 = SM)
#   BKG_PROCESSES    : Background processes to merge (QCD ttbar)

# ============================================================
# Configuration — edit these
# ============================================================
BASE_DIR="../BaristaSAMPLES"
OUTPUT_DIR_OPT="../Plots/SR_Optimisation"
OUTPUT_DIR_FIT="../Plots/fit_results"

SAMPLE="Kappa2V"                  # Signal folder prefix
SIGNAL_COUPLING="l1cvv0cv1"       # kappa2V=0 (use l1cvv1cv1 for SM)
BKG_PROCESSES="QCD ttbar"         # Background processes to merge

MC_CAMPAIGNS="MC20 MC23"
TAGGERS="GN2X GN3PV01"
JET_MODES="DEFAULT bjr_v00 bjr_v01"

RUN_OPTIMIZE=true
RUN_FITS=true

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --fits-only)     RUN_OPTIMIZE=false ;;
        --optimize-only) RUN_FITS=false ;;
        --help)
            echo "Usage: bash run_all_fits.sh [--fits-only|--optimize-only]"
            exit 0 ;;
    esac
done

# ============================================================
# Helper: find the signal directory, handling GN3PV01/GN3XPV01
# ============================================================
find_signal_dir() {
    local mc_dir=$1
    local tagger=$2
    local mode=$3

    # Try exact name first
    local dir="${mc_dir}/${SAMPLE}_${tagger}_${mode}/RootSamp"
    if [ -d "$dir" ]; then
        echo "$dir"
        return
    fi

    # MC23 uses GN3XPV01 instead of GN3PV01
    if [ "$tagger" = "GN3PV01" ]; then
        dir="${mc_dir}/${SAMPLE}_GN3XPV01_${mode}/RootSamp"
        if [ -d "$dir" ]; then
            echo "$dir"
            return
        fi
    elif [ "$tagger" = "GN3XPV01" ]; then
        dir="${mc_dir}/${SAMPLE}_GN3PV01_${mode}/RootSamp"
        if [ -d "$dir" ]; then
            echo "$dir"
            return
        fi
    fi

    echo ""
}

echo "============================================================"
echo "  SR Optimisation & Fitting — VBF HH -> 4b"
echo "============================================================"
echo "  Signal:     ${SAMPLE} (coupling: ${SIGNAL_COUPLING})"
echo "  Background: ${BKG_PROCESSES}"
echo "  Base dir:   ${BASE_DIR}"
echo "  Campaigns:  ${MC_CAMPAIGNS}"
echo "  Taggers:    ${TAGGERS}"
echo "  Jet modes:  ${JET_MODES}"
echo "============================================================"
echo ""

# ============================================================
# 1. Full SR Optimisation (optimize_sr_v2.py)
# ============================================================
if $RUN_OPTIMIZE; then
    echo "====== Running SR Optimisation ======"
    echo ""

    python optimize_sr_v2.py \
        --base-dir "${BASE_DIR}" \
        --output-dir "${OUTPUT_DIR_OPT}" \
        --mc-campaigns ${MC_CAMPAIGNS} \
        --taggers ${TAGGERS} \
        --jet-modes ${JET_MODES} \
        --signal-label "${SAMPLE}" \
        --signal-coupling "${SIGNAL_COUPLING}" \
        --bkg-processes ${BKG_PROCESSES}

    echo ""
    echo "SR optimisation complete. Results in ${OUTPUT_DIR_OPT}/"
    echo ""
fi

# ============================================================
# 2. Standalone Mass Peak Fits (fit_mass_peaks.py)
# ============================================================
if $RUN_FITS; then
    echo "====== Running Standalone Mass Peak Fits ======"
    echo ""

    TOTAL=0
    SKIPPED=0
    SUCCESS=0

    # ----- Single campaigns: MC20 and MC23 -----
    for MC in ${MC_CAMPAIGNS}; do
        MC_DIR="${BASE_DIR}/${MC}"
        if [ ! -d "$MC_DIR" ]; then
            echo "SKIP: ${MC_DIR} not found"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        for TAGGER in ${TAGGERS}; do
            for MODE in ${JET_MODES}; do
                TOTAL=$((TOTAL + 1))
                LABEL="${MC}_${TAGGER}_${MODE}"
                SIGNAL_DIR=$(find_signal_dir "$MC_DIR" "$TAGGER" "$MODE")

                if [ -z "$SIGNAL_DIR" ] || [ ! -d "$SIGNAL_DIR" ]; then
                    echo "SKIP: No directory for $LABEL"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                # Filter by signal coupling
                FILES=$(ls ${SIGNAL_DIR}/*${SIGNAL_COUPLING}*Nominal.root 2>/dev/null)
                if [ -z "$FILES" ]; then
                    # Fallback: try all Nominal files
                    FILES=$(ls ${SIGNAL_DIR}/*Nominal.root 2>/dev/null)
                fi
                if [ -z "$FILES" ]; then
                    echo "SKIP: No Nominal.root in $SIGNAL_DIR"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                echo "=========================================="
                echo "Running: $LABEL"
                echo "  Dir: $SIGNAL_DIR"
                echo "=========================================="

                python fit_mass_peaks.py \
                    --signal-file ${FILES} \
                    --config-label "$LABEL" \
                    --output-dir "${OUTPUT_DIR_FIT}/${LABEL}"

                SUCCESS=$((SUCCESS + 1))
            done
        done
    done

    # ----- Combined: merge MC20 + MC23 -----
    for TAGGER in ${TAGGERS}; do
        for MODE in ${JET_MODES}; do
            TOTAL=$((TOTAL + 1))
            LABEL="Combined_${TAGGER}_${MODE}"

            FILES=""
            for MC in ${MC_CAMPAIGNS}; do
                DIR=$(find_signal_dir "${BASE_DIR}/${MC}" "$TAGGER" "$MODE")
                if [ -n "$DIR" ] && [ -d "$DIR" ]; then
                    # Filter by signal coupling
                    MC_FILES=$(ls ${DIR}/*${SIGNAL_COUPLING}*Nominal.root 2>/dev/null)
                    if [ -z "$MC_FILES" ]; then
                        MC_FILES=$(ls ${DIR}/*Nominal.root 2>/dev/null)
                    fi
                    FILES="$FILES $MC_FILES"
                fi
            done

            FILES=$(echo "$FILES" | xargs)  # trim whitespace
            if [ -z "$FILES" ]; then
                echo "SKIP: No files for $LABEL"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "=========================================="
            echo "Running: $LABEL (merged campaigns)"
            echo "=========================================="

            python fit_mass_peaks.py \
                --signal-file $FILES \
                --config-label "$LABEL" \
                --output-dir "${OUTPUT_DIR_FIT}/${LABEL}"

            SUCCESS=$((SUCCESS + 1))
        done
    done

    echo ""
    echo "============================================================"
    echo "  Standalone fits complete."
    echo "  Total: ${TOTAL}  |  Success: ${SUCCESS}  |  Skipped: ${SKIPPED}"
    echo "  Results in ${OUTPUT_DIR_FIT}/"
    echo "============================================================"
fi

echo ""
echo "All done!"
