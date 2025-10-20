#!/usr/bin/env bash

PYTHON="/Users/andreadi/venv/bin/python"
FILE="/Users/andreadi/Desktop/debug.tiff"

echo "--------------------------------------"
echo ">> Analysing TIFF file: $FILE"
echo "--------------------------------------"

#"$PYTHON" "/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_quickvisual.py" --img_path "$FILE"
#"$PYTHON" "/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_mesh.py" --img_path "$FILE" --overwrite --smooth --box_size 1

MIN_DIST=3.0
MAX_DIST=5.0
"$PYTHON" "/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"

LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
EXTR_MODE="radius" # "nearest"
EXTR_SIZE=30
"$PYTHON" "/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_tan-orient-extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"

AVG_MODE="radius" # "nearest"
AVG_SIZE=20
"$PYTHON" "/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts/run_tan-nem-order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"