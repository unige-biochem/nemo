"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Inter-mesh thickness
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_scaling,
    load_img_unit,
    inter_dist_mesh,
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


def main(img_path, t_select, c_select, mesh_1_name, mesh_2_name, thickness_sampl_number, interp_k,
         thickness_crop_range=None, show_figures=True, render=False, plot_maxprojections=False, ):
    print(f">> Attempting to calculate the thickness between two meshes for {img_path}!")
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

    # ==== Select mesh(es) ====
    mesh_1 = load_mesh(os.path.join(resdata_dir, f"{mesh_1_name}.ply"))
    mesh_2 = load_mesh(os.path.join(resdata_dir, f"{mesh_2_name}.ply"))

    mesh_1_radii = np.linalg.norm(mesh_1.vertices - np.mean(mesh_1.vertices, axis=0), axis=-1)
    mesh_1_radii_avg = np.mean(mesh_1_radii)

    mesh_2_radii = np.linalg.norm(mesh_2.vertices - np.mean(mesh_2.vertices, axis=0), axis=-1)
    mesh_2_radii_avg = np.mean(mesh_2_radii)
    print(f"Approximate radius of inner mesh: {mesh_1_radii_avg} and of outer mesh: {mesh_2_radii_avg}")

    # ==== Calculate inter-mesh distance ====
    dist_vals, dist_idxs = inter_dist_mesh(mesh_1=mesh_1, mesh_2=mesh_2, num_sample=thickness_sampl_number,
                                           crop_range=thickness_crop_range, debug=True,
                                           allow_multiple_hits=False)
    full_dist_vals = interpolate_on_mesh(mesh_1, dist_idxs, dist_vals, k=interp_k)

    # ==== Save inter-mesh distance ====
    save_array(full_dist_vals, f"{mesh_1_name}_VS_{mesh_2_name}_thickness", header=f"dist ({img_unit})",
               folderpath=resdata_dir)

    # ==== Plot inter-mesh distance ====
    plot_hist(array=full_dist_vals, title=f"Thickness AVG = {full_dist_vals.mean():.2e} {img_unit}",
              xlim=thickness_crop_range,
              savefig=os.path.join(resfig_dir, f"{mesh_1_name}_VS_{mesh_2_name}_thickness_hist.png"),
              hidefig=hidefig)
    if plot_maxprojections:
        plot_maxproj_pts(verts=mesh_1.vertices, unit=img_unit, colors=full_dist_vals, cmap="coolwarm",
                         interp_grid_n=200, cmap_label=f"Thickness ({img_unit})",
                         savefig=os.path.join(resfig_dir, f"{mesh_1_name}_VS_{mesh_2_name}_thickness.png"),
                         hidefig=hidefig)
    save_array(np.array([[thickness_sampl_number, interp_k]]),
               name=f"{mesh_1_name}_VS_{mesh_2_name}_thickness_parameters",
               header="numcalc,interpnneigh", folderpath=resdata_dir)
    if render:
        # ==== 3D render result ====
        view_colored_mesh_multiple([mesh_1, mesh_2],
                                   [color_scalar(full_dist_vals, normalise=True, cmap="coolwarm"),
                                    "white"], mesh_blending_list=["opaque", "translucent"],
                                   mesh_opacity_list=[1.0, 0.3])
    return mesh_1, mesh_2, full_dist_vals
