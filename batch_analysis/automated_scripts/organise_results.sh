#!/usr/bin/env bash
set -e
set +H   # disable zsh history expansion for !

ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER_CLEAN"
BATCH_DIR="!batch-analysis_NEMO"

# ===== USER EDITABLE PART =====
SUBFOLDER="ALL_SAMPLING-MESHES_MAXPROJ"

FIG_NAMES=(
#    "sliced_raw.png"
#    "sliced_maxproj_raw.png"
#    "sliced_raw_sampling-mesh-raw.png"
    "sliced_raw_sampling-mesh_maxproj.png"
)
# ==============================

OUTDIR="$ROOT/$BATCH_DIR/$SUBFOLDER"
mkdir -p "$OUTDIR"

# build find expression
NAME_EXPR=()
for f in "${FIG_NAMES[@]}"; do
    NAME_EXPR+=( -name "$f" -o )
done
unset 'NAME_EXPR[-1]'

find "$ROOT" -type f \( "${NAME_EXPR[@]}" \) ! -path "*$BATCH_DIR*" \
-exec bash -c '
f="$1"
img=$(basename "$(dirname "$(dirname "$f")")")
size=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
time=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")
cp "$f" "'"$OUTDIR"'/${time}_${size}_${img}_$(basename "$f")"
' _ {} \;

#convert -delay 5 -loop 0 '/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER_CLEAN/!batch-analysis_NEMO/ALL_SAMPLING-MESHES_MAXPROJ'/*.png animation.gif