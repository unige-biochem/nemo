#!/usr/bin/env bash
find '/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/' -type f \( -name "seg_raw_masks.mp4" -o -name "seg_raw_masks.gif" \) \
  -not -path "*/z_slice_segmentation/*" -not -path "*/!*" | while read -r file; do
    gas_dir=$(basename "$(dirname "$(dirname "$(dirname "$file")")")")   # e.g. Gas1
    size_dir=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$file")")")")")   # e.g. 200
    time_dir=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$file")")")")")")   # e.g. 72
    base=$(basename "$file")
    ext="${base##*.}"
    dest="$HOME/Downloads/${time_dir}h_${size_dir}_${gas_dir}_${base}"
    echo "→ Copying $file → $dest"
    cp "$file" "$dest"
done