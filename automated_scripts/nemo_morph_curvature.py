"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Mean and gaussian curvature
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_scaling,
    load_img_unit,
    curvature_by_srf_fit,
    interpolate_on_mesh
)
from module_scripts.datahandler import create_resdirs, load_mesh, save_array
from module_scripts.visuals import (
    plot_hist,
    plot_maxproj_pts,
    view_colored_mesh_multiple,
    color_scalar
)

# Import Python essentials
import numpy as np


def main(img_path, t_select, c_select, num_samples, radius, mesh_name, interp_k, show_figures=True, flip_normals=False,
         custom_basis=None, render=False, gauss_crop_range=None, mean_crop_range=None):
    print(f">> Attempting to perform cylindrical analysis of gastruloid {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    # ==== Load image ====
    img_scale = load_img_scaling(path=img_path)
    img_unit = load_img_unit(path=img_path)
    if img_scale is None:
        return None
    # ==== Create folder structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")
    mesh_path = os.path.join(resdata_dir, f"{mesh_name}.ply")
    if os.path.exists(mesh_path):
        mesh_curv = load_mesh(mesh_path)
        print(f"Number of sampling vertices: {len(mesh_curv.vertices)} !")
        if flip_normals:
            mesh_curv.invert()
    else:
        print(f"[!] No sampling mesh found for {img_path}!")
        return None

    # ==== Define crop range of Gauss & mean curvature ====
    gauss_exp = 1 / (np.ptp(mesh_curv.vertices, axis=0).mean() / 2) ** 2
    mean_exp = 1 / (np.ptp(mesh_curv.vertices, axis=0).mean() / 2)
    print(f"Gauss should be around {gauss_exp:.2e} (1/{img_unit}^2), "
          f"Mean should be around {mean_exp:.2e} (1/{img_unit})")

    curvature_patch_info = ["radius", radius]
    if curvature_patch_info[0] == "radius":
        patch_label = f"r-{curvature_patch_info[1]}{img_unit}"
    elif curvature_patch_info[0] == "nearest":
        patch_label = f"k-{curvature_patch_info[1]}"
    else:
        print(f"[!] Unknown patch type: {curvature_patch_info[0]}")
        return None

    # ==== [Optional] Exclude boundary vertices ====
    curvature_filter_boundary = False
    curvature_filter_boundary_factor = 0.0

    # ==== Calculate Gauss & mean curvature ====
    curvature_results = curvature_by_srf_fit(mesh_curv, num_sample=num_samples, patch_size=curvature_patch_info[1],
                                             patch_mode=curvature_patch_info[0],
                                             filter_boundary=curvature_filter_boundary, debug=True,
                                             gauss_crop_range=gauss_crop_range, mean_crop_range=mean_crop_range,
                                             boundary_excl_factor=curvature_filter_boundary_factor,
                                             custom_basis=custom_basis)

    c_gauss, c_mean, c_gauss_idxs, c_mean_idxs, c_tensors, c_tensors_mixed, tensor_idxs = curvature_results
    full_c_gauss = interpolate_on_mesh(mesh_curv, c_gauss_idxs, c_gauss, k=interp_k)
    full_c_mean = interpolate_on_mesh(mesh_curv, c_mean_idxs, c_mean, k=interp_k)

    # ==== Save Gauss & mean curvature ====
    save_array(full_c_gauss, f"{mesh_name}_gauss_curv_{patch_label}", header=f"gauss (1/{img_unit}^2)",
               folderpath=resdata_dir)
    save_array(full_c_mean, f"{mesh_name}_mean_curv_{patch_label}", header=f"mean (1/{img_unit})",
               folderpath=resdata_dir)
    curv_full_save_path = os.path.join(resdata_dir, f"{mesh_name}_full_curv_{patch_label}.npz")
    np.savez_compressed(curv_full_save_path, C_tensors=c_tensors,
                        C_tensors_mixed=c_tensors_mixed, tensor_idxs=tensor_idxs)
    print(f"Saved curvature results to {curv_full_save_path} !")

    # ==== Plot Gauss & mean curvature ====
    plot_hist(full_c_gauss, title=f"Gauss AVG = {full_c_gauss.mean():.2e} $(1/{img_unit}^2)$",
              savefig=os.path.join(resfig_dir, f"{mesh_name}_gauss_hist_{patch_label}.png"),
              hidefig=hidefig)
    plot_hist(full_c_mean, title=f"Mean AVG = {full_c_mean.mean():.2e} $(1/{img_unit})$",
              savefig=os.path.join(resfig_dir, f"{mesh_name}_mean_hist_{patch_label}.png"),
              hidefig=hidefig)
    plot_maxproj_pts(verts=mesh_curv.vertices, unit=img_unit, colors=full_c_gauss, cmap="coolwarm",
                     interp_grid_n=200, cmap_label=f"Gaussian Curvature $(1/{img_unit}^2)$",
                     savefig=os.path.join(resfig_dir, f"{mesh_name}_gauss_curv_{patch_label}.png"),
                     hidefig=hidefig)
    plot_maxproj_pts(verts=mesh_curv.vertices, unit=img_unit, colors=full_c_mean, cmap="Spectral",
                     interp_grid_n=200, cmap_label=f"Mean Curvature $(1/{img_unit})$",
                     savefig=os.path.join(resfig_dir, f"{mesh_name}_mean_curv_{patch_label}.png"),
                     hidefig=hidefig)

    save_array(np.column_stack(([num_samples], [radius], [interp_k])),
               name=f"{mesh_name}_curvature_parameters",
               header="numcalc,radius,interp_k", folderpath=resdata_dir)
    if render:
        # ==== 3D render result ====
        view_colored_mesh_multiple([mesh_curv, mesh_curv],
                                   [color_scalar(full_c_gauss, normalise=True, cmap="coolwarm"),
                                    color_scalar(full_c_mean, normalise=True, cmap="Spectral")],
                                   name_list=["Gauss", "Mean"])
    return mesh_curv, full_c_gauss, full_c_mean
