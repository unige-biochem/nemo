#!/usr/bin/env bash

PYTHON="/Users/andreadi/venv/bin/python"
SCRIPTS_DIR="/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Code/nemo/batch_analysis/automated_scripts"
#FILE="/Users/andreadi/Desktop/debug.tiff"

#FILE='/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/72/300/Gas1.tif'
#
#echo "--------------------------------------"
#echo ">> Analysing TIFF file: $FILE"
#echo "--------------------------------------"
#
##"$PYTHON" "$SCRIPTS_DIR/run_quickvisual.py" --img_path "$FILE"
##"$PYTHON" "$SCRIPTS_DIR/run_mesh.py" --img_path "$FILE" --overwrite --smooth --box_size 1
#
#MIN_DIST=-21.0
#MAX_DIST=-20.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"
#
#
#MIN_DIST=-36.0
#MAX_DIST=-35.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"
#
#
#
#
##=======================================================================================================================
#
#FILE='/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/96/300/Gas3.tif'
#
#echo "--------------------------------------"
#echo ">> Analysing TIFF file: $FILE"
#echo "--------------------------------------"
#
##"$PYTHON" "$SCRIPTS_DIR/run_quickvisual.py" --img_path "$FILE"
##"$PYTHON" "$SCRIPTS_DIR/run_mesh.py" --img_path "$FILE" --overwrite --smooth --box_size 1
#
#MIN_DIST=-21.0
#MAX_DIST=-20.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"
#
#
#MIN_DIST=-36.0
#MAX_DIST=-35.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"
#
#
#
##=======================================================================================================================
#
#FILE='/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/112/300/Gas2.tif'
#
#echo "--------------------------------------"
#echo ">> Analysing TIFF file: $FILE"
#echo "--------------------------------------"
#
##"$PYTHON" "$SCRIPTS_DIR/run_quickvisual.py" --img_path "$FILE"
##"$PYTHON" "$SCRIPTS_DIR/run_mesh.py" --img_path "$FILE" --overwrite --smooth --box_size 1
#
#MIN_DIST=-21.0
#MAX_DIST=-20.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"
#
#
#MIN_DIST=-36.0
#MAX_DIST=-35.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"
#
#
#DEST="/Users/andreadi/Desktop/test"
#mkdir -p "$DEST"
#
#cp -R "/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/72/300/Gas1" "$DEST/"
#cp -R "/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/96/300/Gas3" "$DEST/"
#cp -R "/Volumes/roux/AurelienRouxLab/Oriol/Experiments 2 photon with filter/Gastruloids size exp/EXP4_FILTER/112/300/Gas2" "$DEST/"


#=======================================================================================================================
FILE='/Users/andreadi/Physbio Dropbox/Konstantinos Andreadis/Academic/Data/1_gastruloid/!2025_corrected-membrane/104h_300_Gas1.tif'

echo "--------------------------------------"
echo ">> Analysing TIFF file: $FILE"
echo "--------------------------------------"

#"$PYTHON" "$SCRIPTS_DIR/run_quickvisual.py" --img_path "$FILE"
#"$PYTHON" "$SCRIPTS_DIR/run_mesh.py" --img_path "$FILE"

#MIN_DIST=-21.0
#MAX_DIST=-20.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"


#MIN_DIST=-36.0
#MAX_DIST=-35.0
#LAYER_LABEL="proj_${MIN_DIST}_to_${MAX_DIST}_um"
#EXTR_MODE="radius" # "nearest"
#EXTR_SIZE=30
#AVG_MODE="radius" # "nearest"
#AVG_SIZE=40
#
#"$PYTHON" "$SCRIPTS_DIR/run_projection.py" --img_path "$FILE" --dist_min "$MIN_DIST" --dist_max "$MAX_DIST"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_orient_extract.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --patch_mode "$EXTR_MODE" --patch_size "$EXTR_SIZE"
#"$PYTHON" "$SCRIPTS_DIR/run_tan_nem_order.py" --img_path "$FILE" --layer_label "$LAYER_LABEL" --avg_mode "$AVG_MODE" --avg_size "$AVG_SIZE"

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