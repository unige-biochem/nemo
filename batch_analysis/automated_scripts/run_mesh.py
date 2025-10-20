"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Mesh a .tiff file
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
    parser.add_argument("--blur", default=5, type=float, help="Gaussian blur")
    parser.add_argument("--thresh", default=None, type=float, help="Threshold value")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, help="Smoothen mesh")
    parser.add_argument("--box_size", default=2, type=int, help="Size of the marching cubes box")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, help="Mesh again")
    args = parser.parse_args()
    return args


def main(img_path, img_blur_val, img_thresh_val, smooth, overwrite, box_size):
    print(f">> Attempting to mesh image {img_path}!")
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
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = datahandler.create_resdirs(img_path)

    if os.path.exists(os.path.join(resdata_dir, "sampling_mesh.ply")):
        print(f">> [!] Image already meshed!")
        if not overwrite:
            return None
    # ==== Blur Image ====
    sigma_ = analysis.rescale_val_xyz(val=img_blur_val, scale=img_scale)
    img_blur = analysis.gaussian_blur(img=img_raw, sigma=sigma_, renorm=False)

    # ==== Plot Image Slices ====
    visuals.plot_img(img=img_blur, scale=img_scale, unit=img_unit, cmap="inferno",
                     savefig=os.path.join(resfig_dir, "sliced_blur.png"), hidefig=True)
    # ==== Yen Threshold Image ====
    if img_thresh_val is None:
        # img_thresh_val = np.min([analysis.yen_thresh(img_blur[:, :, img_dim[2] // i]) for i in np.arange(2, 6)])
        img_thresh_val = np.min(
            [analysis.yen_thresh(img_blur[:, :, img_dim[2] // 2]),
             analysis.yen_thresh(img_blur[:, img_dim[1] // 2, :]),
             analysis.yen_thresh(img_blur[img_dim[0] // 2, :, :])])
        # img_thresh_val *= 2.0
        # img_thresh_val = np.min([analysis.yen_thresh(img_blur[:, :, int(img_dim[2] * i)]) for i in np.linspace(0.4, 0.6, 10)])

    # ==== Binarise Image using Threshold ====
    img_thresh = analysis.thresh_img(img_blur, img_thresh_val)

    # ==== Plot Image Slices ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, savefig=os.path.join(resfig_dir, "sliced_thresh.png"),
                     cmap="inferno", thresh_mask=img_thresh, hidefig=True)

    # ==== Segment Surface Mesh(es) ====
    full_mesh = analysis.marching_cubes(img=img_thresh, scale=img_scale, level=0.5, step_size=box_size)
    # ==== Save Mesh(es) ====
    datahandler.save_mesh(full_mesh, os.path.join(resdata_dir, "full_mesh.ply"))
    print(f"#FULL = {full_mesh.vertices.shape[0]}!")
    if smooth:
        # ==== Smooth Mesh(es) ====
        smooth_factor = 0.001
        smooth_iterations = 200
        full_mesh_smooth = analysis.taubin_smooth_mesh(mesh=full_mesh, n_iter=smooth_iterations,
                                                       pass_band=smooth_factor)
        # ==== Save Mesh(es) ====
        datahandler.save_mesh(full_mesh_smooth, os.path.join(resdata_dir, "full_mesh_smooth.ply"))
        # ==== Plot Image Slices with Mesh Overlay ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                         savefig=os.path.join(resfig_dir, "sliced_smooth_full-mesh.png"),
                         meshes=[full_mesh, full_mesh_smooth],
                         mesh_colors=["black", "purple"], hidefig=True)
    else:
        full_mesh_smooth = full_mesh.copy()
    full_mesh_smooth_subset_all = analysis.find_connected_meshes(mesh=full_mesh_smooth)
    sizes = [mesh.vertices.shape[0] for mesh in full_mesh_smooth_subset_all]
    full_mesh_smooth_subset = full_mesh_smooth_subset_all[np.argmax(sizes)]
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_smooth_full-mesh_sub.png"),
                     meshes=[full_mesh_smooth_subset],
                     mesh_colors=["purple"], hidefig=True)

    # ==== Select Sampling Mesh ====
    sampl_mesh = full_mesh_smooth_subset.copy()
    # ==== Save Sampling Mesh ====
    datahandler.save_mesh(sampl_mesh, os.path.join(resdata_dir, "sampling_mesh.ply"))
    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_raw_sampling-mesh.png"), meshes=[sampl_mesh],
                     hidefig=True)
    print(f"Number of sampling points: {len(sampl_mesh.vertices)} !")
    return None


if __name__ == "__main__":
    print("======== START NEMO ========")
    args = parse_args()
    main(img_path=args.img_path, img_blur_val=args.blur, img_thresh_val=args.thresh, smooth=args.smooth,
         overwrite=args.overwrite, box_size=args.box_size)
    print("======== END NEMO ========")
