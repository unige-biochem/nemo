"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Project image
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import load_img_virtual, proj2mesh, scale_mesh, spherical_project
from module_scripts.datahandler import create_resdirs, save_array, save_mesh, load_mesh
from module_scripts.visuals import plot_img, plot_spherical_projection, plot_maxproj_pts, view_colored_mesh, \
    color_scalar

# Import Python essentials
import numpy as np
import matplotlib.pyplot as plt


def main(img_path, t_select, c_select, mesh_name, dist_min, dist_max, dist_num, proj_mode, flip_normals,
         correct_offset_automatic=False, scale_down_mesh=True, min_dist_per_vert=None,
         initial_broad_scan=False, show_figures=True, render=False, initial_broad_scan_min=0.0,
         initial_broad_scan_max=50.0, plot_half_projections=False, plot_sph_proj=True):
    print(f">> Attempting to project image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    # ==== Load image ====
    img_load = load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load

    # ==== Create folder structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    try:
        sampl_mesh = load_mesh(os.path.join(resdata_dir, f"{mesh_name}.ply"))
        print(f"Number of sampling vertices: {len(sampl_mesh.vertices)} !")
        if flip_normals:
            sampl_mesh.invert()
    except Exception as e:
        print(f">> Failed to load sampling mesh for {img_path} with error: {e}!")
        return None

    # ==== Projection logic ====
    layer_label = f"{mesh_name}_proj_{dist_min}_to_{dist_max}_{img_unit}_{proj_mode}"
    dist_middle = dist_min + (dist_max - dist_min) / 2

    # ==== Specify custom minimum ====
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    if not os.path.exists(resdata_dir_layer):
        os.makedirs(resdata_dir_layer)
    if not os.path.exists(resfig_dir_layer):
        os.makedirs(resfig_dir_layer)

    if min_dist_per_vert is not None:
        print(f"Minimal distance per vertex is not None!")
        save_array(array=min_dist_per_vert, header=f"dist({img_unit})",
                   folderpath=resdata_dir_layer,
                   name="minimal_distance_per_vertex")
    proj_offset = 0
    if initial_broad_scan:
        # ==== Project onto mesh ====
        dist_min_initscan = initial_broad_scan_min
        dist_max_initscan = initial_broad_scan_max
        dist_num_initscan = 50
        proj_initscan = proj2mesh(img=img_raw, mesh=sampl_mesh, min_dist_per_vert=min_dist_per_vert,
                                  scale=img_scale, min_dist=dist_min_initscan, max_dist=dist_max_initscan,
                                  num_dist=dist_num_initscan,
                                  mode=proj_mode, show_proj=True,
                                  savefig=os.path.join(resfig_dir_layer, "proj_kymo_initscan.png"),
                                  unit=img_unit, normalise=False, hidefig=hidefig, return_full=True)
        distances_initscan, dist_points_initscan, radial_intensities_initscan = proj_initscan
        radial_intensities_initscan_meanprofile = np.nanmean(radial_intensities_initscan, axis=0)
        window_size = min(5, len(radial_intensities_initscan_meanprofile))
        smoothed_profile = np.convolve(radial_intensities_initscan_meanprofile, np.ones(window_size) / window_size,
                                       mode='same')

        if correct_offset_automatic:
            # ==== Offset projection correction ====
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
        save_array(np.array([proj_offset]), "offset", header="dist", folderpath=resdata_dir_layer)

        plt.figure()
        plt.plot(distances_initscan, smoothed_profile, "--")
        plt.plot(distances_initscan, radial_intensities_initscan_meanprofile)
        plt.axvline(x=proj_offset, color="red", label="OFFSET")
        plt.axvline(x=dist_middle + proj_offset, color="purple", label="PROJECTION MIDDLE")
        plt.legend()
        plt.savefig(os.path.join(resfig_dir_layer, "proj_kymo_initscan_smooth.png"), bbox_inches="tight", dpi=300)
        if show_figures:
            plt.show()
        else:
            plt.close()

        print(f">> Using projection OFFSET={proj_offset}...")
    dist_min_corrected = dist_min + proj_offset
    dist_max_corrected = dist_max + proj_offset
    dist_middle_corrected = dist_middle + proj_offset

    proj_layer = proj2mesh(img=img_raw, mesh=sampl_mesh,
                           min_dist_per_vert=min_dist_per_vert,
                           scale=img_scale, min_dist=dist_min_corrected, max_dist=dist_max_corrected,
                           num_dist=dist_num, mode=proj_mode, show_proj=True,
                           savefig=os.path.join(resfig_dir_layer, f"proj_kymo.png"), unit=img_unit,
                           normalise=False, hidefig=hidefig)
    if scale_down_mesh:
        if min_dist_per_vert is not None:
            layer_mesh = scale_mesh(mesh=sampl_mesh, distance=min_dist_per_vert + dist_middle_corrected)
        else:
            layer_mesh = scale_mesh(mesh=sampl_mesh, distance=dist_middle_corrected)
    else:
        layer_mesh = sampl_mesh.copy()
    save_mesh(layer_mesh, filepath=os.path.join(resdata_dir_layer, "layer_mesh.ply"))

    plot_img(img=img_raw, scale=img_scale, unit=img_unit, cmap="Greys_r",
             savefig=os.path.join(resfig_dir_layer, f"projection-mesh_sliced.png"),
             meshes=[sampl_mesh, layer_mesh], show_mesh_normals=False,
             mesh_colors=["red", "blue"], hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, cmap="Greys_r",
             savefig=os.path.join(resfig_dir_layer, f"projection-mesh_sliced_normals.png"),
             meshes=[sampl_mesh, layer_mesh], show_mesh_normals=True,
             mesh_colors=["red", "blue"], hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, cmap="Greys",
             savefig=os.path.join(resfig_dir_layer, f"projection-mesh_maxproj.png"),
             meshes=[sampl_mesh, layer_mesh], max_proj=True,
             mesh_colors=["red", "blue"], hidefig=hidefig)

    # ==== Save projection ====
    save_array(proj_layer, "intensities", header="I", folderpath=resdata_dir_layer)

    try:
        if plot_sph_proj:
            # ==== Plot projected result ====
            sph_proj_phi, sph_proj_theta = spherical_project(pts=layer_mesh.vertices)
            plot_spherical_projection(phi=sph_proj_phi, theta=sph_proj_theta, intensities=proj_layer, cmap="inferno",
                                      savefig=os.path.join(resfig_dir_layer, "spherical_projection.png"),
                                      hidefig=hidefig)
        if plot_half_projections:
            plot_maxproj_pts(verts=layer_mesh.vertices, colors=proj_layer, cmap="inferno", unit=img_unit,
                             savefig=os.path.join(resfig_dir_layer, "maxproj.png"), hidefig=hidefig)
    except Exception as e:
        print(f"Encountered error during spherical projection: {e}")

    if render:
        view_colored_mesh(mesh=layer_mesh, mesh_blending="opaque",
                          vert_colors=color_scalar(scalar=proj_layer, normalise=True,
                                                   cmap="inferno"), img=img_raw, scale=img_scale,
                          mesh_opacity=1.0)
    save_array(np.array([[dist_min, dist_max, dist_num, int(flip_normals), proj_offset, int(scale_down_mesh)]]),
               name="projection_parameters",
               header="dist_min,dist_max,dist_num,flip_normals,proj_offset,scale_down_mesh",
               folderpath=resdata_dir_layer)

    return layer_label, layer_mesh, proj_layer
