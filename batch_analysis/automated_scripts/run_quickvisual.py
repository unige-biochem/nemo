"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Quickly visualise a .tiff file
Requires Python 3.9.6 and packages from requirements.txt.
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO scripts
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts import analysis, datahandler, visuals

# Import python essentials
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="", required=True, type=str, help="Path to the input TIFF image")
    args = parser.parse_args()
    return args


def main(img_path):
    print(f">> Attempting to quickly visualise image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None
    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    # ==== Choose Time Step and Channel ====
    t_select = 0
    c_select = 0
    # ==== [Optional] Reduce Resolution ====
    z_reduce_factor = 1  # 1 means no reduction
    xy_reduce_factor = 1  # 1 means no reduction
    # ==== [Optional] Normalise Intensities to [0, 1] ====
    normalise_intensities = False  # can be set to False
    # ==== [Optional] Overwrite Scaling with FIJI Values ====
    custom_scaling = None  # please use (z, y, x)
    # ==== [Optional] Overwrite unit with FIJI Values ====
    custom_unit = "um"  # as string
    # ==== Load Image ====
    img_load = analysis.load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select,
                                         reduce_xy=xy_reduce_factor,
                                         reduce_z=z_reduce_factor, norm_vals=normalise_intensities,
                                         custom_scaling=custom_scaling, custom_unit=custom_unit)
    if img_load is not None:
        img_raw, img_dim, img_scale, img_unit = img_load
        # ==== Create Folder Structure ====
        resdata_dir, resfig_dir = datahandler.create_resdirs(img_path)
        # ==== Plot Image Slices and Max Projections ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, max_proj=True, cmap='Greens', figsize=(20, 3),
                         savefig=os.path.join(resfig_dir, "sliced_maxproj_raw.png"), hidefig=True)
        z_i, y_i, x_i = int(img_dim[0] // 2), int(img_dim[1] // 2), int(img_dim[2] // 2)
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, x_i=x_i, y_i=y_i, z_i=z_i, figsize=(20, 3),
                         savefig=os.path.join(resfig_dir, "sliced_raw.png"), cmap="Greens_r", hidefig=True)
    else:
        return None

    try:
        sampl_mesh = datahandler.load_mesh(os.path.join(resdata_dir, "sampling_mesh.ply"), recalc_normals=True,
                                           clean=False)
    except:
        print(f"Could not find sampling mesh from {img_path}...")
        return None
    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[sampl_mesh], slice_depth=1,
                     savefig=os.path.join(resfig_dir, "sliced_raw_sampling-mesh.png"), hidefig=True)
    mesh_slice_max = np.array([img_scale[i] * img_dim[i] for i in range(len(img_scale))]).max() / 2
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[sampl_mesh],
                     slice_depth=mesh_slice_max, mesh_alpha=0.05,
                     savefig=os.path.join(resfig_dir, "sliced_raw_sampling-mesh_maxproj.png"), hidefig=True)
    return None


if __name__ == "__main__":
    print("======== START NEMO ========")
    args = parse_args()
    try:
        main(img_path=args.img_path)
    except:
        print(f"Failed to run batch analysis for {args.img_path}...")
    print("======== END NEMO ========")
