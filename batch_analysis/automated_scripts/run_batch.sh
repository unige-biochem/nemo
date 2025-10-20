#!/usr/bin/env bash
# Exit immediately on setup errors, but allow jobs to fail
set -e

# ------------------------------
# User settings
# ------------------------------
PYTHON="/Users/andreadi/venv/bin/python"
SCRIPT="/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_quickvisual.py"
#SCRIPT="/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_link2dsegmentation.py"
#SCRIPT="/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_mesh.py"
ROOT_DIR="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER"
MAX_JOBS=12

# ------------------------------
# Find TIFF files
# ------------------------------
echo "--------------------------------------"
echo ">> Searching for TIFF files under: $ROOT_DIR"
echo "--------------------------------------"

mapfile -t TIFF_FILES < <(
  find "$ROOT_DIR" -type f \( -iname "*.tif" -o -iname "*.tiff" \) \
    ! -path "*/z_slice_segmentation/*" \
    ! -path "*/Extra/*" \
    | sort
)

NUM_FILES=${#TIFF_FILES[@]}
echo "Found $NUM_FILES TIFF files to process."
echo "Max parallel jobs: $MAX_JOBS"
echo "--------------------------------------"
echo "[!] Starting parallel processing..."
echo

# ------------------------------
# Define processing function
# ------------------------------
process_file() {
    local file="$1"
    echo ">> Processing $file ..."
    if ! "$PYTHON" "$SCRIPT" --img_path "$file"; then
        echo "[!] Failed on: $file" >&2
    fi
}

# Export for GNU Parallel
export -f process_file
export PYTHON
export SCRIPT

# ------------------------------
# Run in parallel
# ------------------------------
parallel --bar -j "$MAX_JOBS" process_file ::: "${TIFF_FILES[@]}"

echo
echo "--------------------------------------"
echo ">> All done (errors, if any, are printed above)."
echo "--------------------------------------"