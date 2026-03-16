#!/usr/bin/env bash
set -e
set +H

ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER_CLEAN"
OUT_ROOT="/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/URI_25022026_PR_NEMO/EXP4_filter_membrane"

# ==============================
SUBFOLDER="all_nemo_figures"
FILE_NAMES=(
"3d_midline_curve.png"
"rho-profile.png"
"cylindrical_projection.png"
"cylindrical_projection_cropped.png"
"sliced_maxproj_raw.png"
"sliced_raw.png"
"sliced_raw_sampling-mesh.png"
"sliced_raw_sampling-mesh_maxproj.png"
"q-sphi-by-phi-profile-um_cropped.png"
"q-sphi-profile-um_cropped.png"
"S-profile-um_cropped.png"
"sampling_mesh.ply"
"distgraph_broad-scan.png"
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

    # 1. Get path relative to ROOT
    # This removes the ROOT prefix from the string
    rel_path="${file_path#$ROOT/}"

    # 2. Split the relative path into an array using "/" as delimiter
    IFS="/" read -ra parts <<< "$rel_path"

    # 3. Assign metadata based on top-level folder positions
    # Structure: ROOT/TIME/SIZE/GASTRULOID/...
    time_val="${parts[0]}"
    size_val="${parts[1]}"
    gastruloid_val="${parts[2]}"

    # 4. Get the filename
    base_name=$(basename "$file_path")

    # 5. Determine if we are in a layer subfolder to avoid name collisions
    # (Optional: Add the layer label to the filename if it exists)
    if [[ ${#parts[@]} -gt 4 ]]; then
        # If parts[3] is "results" and parts[5] is a layer, you might want it
        # but for now, we follow your requested pattern:
        new_name="${time_val}_${size_val}_${gastruloid_val}_${base_name}"
    else
        new_name="${time_val}_${size_val}_${gastruloid_val}_${base_name}"
    fi

    cp "$file_path" "$OUTDIR/$new_name"
    echo "Copied: $new_name"
' _ {} \;