#!/usr/bin/env bash
set -e
set +H   # disable zsh history expansion for !

ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER_CLEAN"
OUT_ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/URI_25022026_PR_NEMO/EXP4_filter_membrane"

# ==============================
SUBFOLDER="ALL_CENTERLINES_FITTED"
FILE_NAMES=(
#"3d_midline_curve.csv"
"3d_midline_curve.png"
)


#SUBFOLDER="ALL_MORPHO_PROFILES"
#FILE_NAMES=(
#"rho-profile.png"
#"mesh_s-rho-phi.csv"
#)

#SUBFOLDER="ALL_CYLINDRICAL_PROJECTIONS"
#FILE_NAMES=(
#"cylindrical_projection.png"
#)

#SUBFOLDER="ALL_CYLINDRICAL_PROJECTIONS_CROPPED"
#FILE_NAMES=(
#"cylindrical_projection_cropped.png"
#)


#SUBFOLDER="ALL_SAMPLING-MESHES"
#"sliced_raw_sampling-mesh.png"
#"sliced_raw_sampling-mesh_maxproj.png"
#"sampling_mesh.ply"
# ==============================

OUTDIR="$OUT_ROOT/$SUBFOLDER"
mkdir -p "$OUTDIR"

# build find expression
NAME_EXPR=()
for f in "${FILE_NAMES[@]}"; do
    NAME_EXPR+=( -name "$f" -o )
done
unset 'NAME_EXPR[-1]'

find "$ROOT" -type f \( "${NAME_EXPR[@]}" \) \
-exec bash -c '
f="$1"
gastruloid=$(basename "$(dirname "$(dirname "$f")")")
size=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
time=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")
cp "$f" "'"$OUTDIR"'/${time}_${size}_${gastruloid}_$(basename "$f")"
' _ {} \;

#find "$ROOT" -type f \( "${NAME_EXPR[@]}" \) \
#-exec bash -c '
#f="$1"
#gastruloid=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
#size=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")
#time=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")")
#cp "$f" "'"$OUTDIR"'/${time}_${size}_${gastruloid}_$(basename "$f")"
#' _ {} \;

#convert -delay 5 -loop 0 '/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER_CLEAN/!batch-analysis_NEMO/ALL_SAMPLING-MESHES_MAXPROJ'/*.png animation.gif