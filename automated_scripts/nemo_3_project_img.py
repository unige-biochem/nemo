"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Project image
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import load_img_virtual, proj2mesh, scale_mesh, spherical_project
from module_scripts.datahandler import create_resdirs, save_array, save_mesh, load_mesh
from module_scripts.visuals import plot_img, plot_spherical_projection, plot_maxproj_pts, view_colored_mesh, \
    color_scalar

# Import python essentials
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(img_path, t_select, c_select, mesh_name, dist_min, dist_max, dist_num, proj_mode, flip_normals,
         show_figures=True, render=False, correct_offset_automatic=False, scale_down_mesh=True):
    print(f">> Attempting to project image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    # ==== Load Image ====
    img_load = load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load

    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    try:
        sampl_mesh = load_mesh(os.path.join(resdata_dir, f"{mesh_name}.ply"))
        print(f"Number of sampling vertices: {len(sampl_mesh.vertices)} !")
        if flip_normals:
            sampl_mesh.invert()
        if show_figures:
            plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[sampl_mesh],
                     slice_depth=1, show_mesh_normals=True)
    except:
        print(f">> Failed to load sampling mesh for {img_path}!")
        return None

    # ==== Projection Logic ====
    layer_label = f"{mesh_name}_proj_{dist_min}_to_{dist_max}_{img_unit}_{proj_mode}"
    dist_middle = dist_min + (dist_max - dist_min) / 2

    # ==== Specify Custom Minimum ====
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    if not os.path.exists(resdata_dir_layer):
        os.makedirs(resdata_dir_layer)
    if not os.path.exists(resfig_dir_layer):
        os.makedirs(resfig_dir_layer)

    # ==== Save Sampling Vertices and Normals ====
    save_array(sampl_mesh.vertices, "verts", header="x,y,z", folderpath=resdata_dir_layer)
    save_array(sampl_mesh.vertex_normals, "normals", header="nx,ny,nz", folderpath=resdata_dir_layer)

    # ==== Project onto Mesh ====
    dist_min_initscan = 0.0
    dist_max_initscan = 50.0
    dist_num_initscan = 50
    proj_initscan = proj2mesh(img=img_raw, mesh=sampl_mesh, min_dist_per_vert=None,
                              scale=img_scale, min_dist=dist_min_initscan, max_dist=dist_max_initscan,
                              num_dist=dist_num_initscan,
                              mode=proj_mode, show_proj=True,
                              savefig=os.path.join(resfig_dir_layer, "proj_kymo_initscan.pdf"),
                              unit=img_unit, normalise=False, hidefig=hidefig, return_full=True)
    distances_initscan, dist_points_initscan, radial_intensities_initscan = proj_initscan
    radial_intensities_initscan_meanprofile = np.nanmean(radial_intensities_initscan, axis=0)
    window_size = 5
    smoothed_profile = np.convolve(radial_intensities_initscan_meanprofile, np.ones(window_size) / window_size,
                                   mode='same')

    if correct_offset_automatic:

        # ==== Offset Projection Correction ====

        profile_ref = np.percentile(smoothed_profile, 95)
        if profile_ref > 0:
            norm_profile = smoothed_profile / profile_ref
        else:
            norm_profile = smoothed_profile
        entry_threshold = 0.7
        indices_above = np.argwhere(norm_profile > entry_threshold)
        if indices_above.size > 0:
            idx_first = indices_above[0][0]
            proj_offset = distances_initscan[idx_first]
        else:
            proj_offset = 0.0
    else:
        proj_offset = 0.0
    dist_min_corrected = dist_min + proj_offset
    dist_max_corrected = dist_max + proj_offset
    dist_middle_corrected = dist_middle + proj_offset
    save_array(np.array([proj_offset]), "offset", header="dist", folderpath=resdata_dir_layer)

    plt.figure()
    plt.plot(distances_initscan, smoothed_profile, "--")
    plt.plot(distances_initscan, radial_intensities_initscan_meanprofile)
    plt.axvline(x=proj_offset, color="red", label="OFFSET")
    plt.axvline(x=dist_middle_corrected, color="purple", label="PROJECTION")
    plt.savefig(os.path.join(resfig_dir_layer, "proj_kymo_initscan_smooth.pdf"), bbox_inches="tight", dpi=200)
    if show_figures:
        plt.show()
    else:
        plt.close()

    print(f">> Using projection OFFSET={proj_offset}...")

    proj_layer = proj2mesh(img=img_raw, mesh=sampl_mesh,
                           min_dist_per_vert=None,
                           scale=img_scale, min_dist=dist_min_corrected, max_dist=dist_max_corrected,
                           num_dist=dist_num, mode=proj_mode, show_proj=True,
                           savefig=os.path.join(resfig_dir_layer, f"proj_kymo.pdf"), unit=img_unit,
                           normalise=False, hidefig=hidefig)
    if scale_down_mesh:
        layer_mesh = scale_mesh(mesh=sampl_mesh, distance=dist_middle_corrected)
    else:
        layer_mesh = sampl_mesh.copy()
    save_mesh(layer_mesh, filepath=os.path.join(resdata_dir_layer, "layer_mesh.ply"))

    # ==== Save Projection ====
    save_array(proj_layer, "intensities", header="I", folderpath=resdata_dir_layer)

    # ==== Plot Projected Result ====
    mercator_x, mercator_y = spherical_project(pts=layer_mesh.vertices)
    plot_spherical_projection(phi=mercator_x, theta=mercator_y, intensities=proj_layer, cmap="inferno",
                              savefig=os.path.join(resfig_dir_layer, "mercator.pdf"), hidefig=hidefig)
    plot_maxproj_pts(verts=layer_mesh.vertices, colors=proj_layer, cmap="inferno", unit=img_unit,
                     savefig=os.path.join(resfig_dir_layer, "maxproj.pdf"), hidefig=hidefig)

    if render:
        view_colored_mesh(mesh=layer_mesh, mesh_blending="opaque",
                          vert_colors=color_scalar(scalar=proj_layer, normalise=True,
                                                   cmap="inferno"), img=img_raw, scale=img_scale,
                          mesh_opacity=1.0)
    save_array(np.column_stack(
        ([dist_min], [dist_max], [dist_num], [int(flip_normals)], [proj_offset],
         [int(scale_down_mesh)])),
        name="projection_parameters",
        header="dist_min,dist_max,dist_num,flip_normals,proj_offset,scale_down_mesh",
        folderpath=resdata_dir_layer)

    return layer_label, layer_mesh, proj_layer
