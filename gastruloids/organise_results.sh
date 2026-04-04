#!/usr/bin/env bash
set -e
set +H

ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER_CLEAN"
OUT_ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/URI_25022026_PR_NEMO/EXP4_filter_membrane"

# ==============================
SUBFOLDER="collected_nemo_figures"
FILE_NAMES=(
"sliced_raw.pdf"
"sliced_maxproj_raw.pdf"
"sliced_raw_sampling-mesh.pdf"
"sliced_raw_sampling-mesh_maxproj.pdf"
"sampling_mesh.ply"
"sampling_mesh.ply_gauss_curv_r-50.0um.pdf"
"sampling_mesh.ply_mean_curv_r-50.0um.pdf"
"3d_midline_curve.pdf"
"rho-profile.pdf"
"cylindrical_projection.pdf"
"cylindrical_projection_cropped.pdf"
)
# ==============================

OUTDIR="$OUT_ROOT/$SUBFOLDER"
mkdir -p "$OUTDIR"

# Build find expression
NAME_EXPR=()
for f in "${FILE_NAMES[@]}"; do
    NAME_EXPR+=( -name "$f" -o )
done
unset 'NAME_EXPR[-1]'

# Export ROOT so the subshell can see it
export ROOT
export OUTDIR

find "$ROOT" -type f \( "${NAME_EXPR[@]}" \) \
-exec bash -c '
    file_path="$1"
    rel_path="${file_path#$ROOT/}"
    IFS="/" read -ra parts <<< "$rel_path"

    # Standard Metadata
    time_val="${parts[0]}"
    size_val="${parts[1]}"
    gastruloid_val="${parts[2]}"
    base_name=$(basename "$file_path")

    # Check if the file is inside a specific projection/layer subfolder
    # Based on ROOT/TIME/SIZE/GASTRULOID/results/LAYER_LABEL/filename
    # parts[0]=TIME, [1]=SIZE, [2]=GAST, [3]=results, [4]=LAYER_LABEL

    if [[ ${#parts[@]} -ge 5 && "${parts[3]}" == "results" ]]; then
        layer_val="${parts[4]}"
        new_name="${time_val}_${size_val}_${gastruloid_val}_${layer_val}_${base_name}"
    else
        # Fallback if it is a top-level gastruloid file (like a mesh or midline)
        new_name="${time_val}_${size_val}_${gastruloid_val}_${base_name}"
    fi

    cp "$file_path" "$OUTDIR/$new_name"
    echo "Copied: $new_name"
' _ {} \;