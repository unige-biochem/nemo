#!/usr/bin/env bash

PYTHON="/Users/andreadi/venv/bin/python"
SCRIPTS_DIR="/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts"
MIN_DIST=-42.0
MAX_DIST=-41.0
LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
EXTR_MODE="radius" # "nearest"
EXTR_SIZE=30
AVG_MODE="radius" # "nearest"
AVG_SIZE=40

"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
"$PYTHON" "$SCRIPTS_DIR/run_tan-orient-extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
"$PYTHON" "$SCRIPTS_DIR/run_tan-nem-order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"