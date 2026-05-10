"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Extract tangential nematic field
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from module_scripts.analysis import (
    load_img_unit,
    load_img_scaling,
    filter_normal_validity,
    coord_search_radius,
    coord_search_neighbours,
    tan_proj,
    tan_interp_batch,
    batch_2d_orientation,
    spherical_project,
    spherical_project_vectors
)
from module_scripts.datahandler import create_resdirs, load_array, load_mesh, save_array
from module_scripts.visuals import (
    plot_dir_field,
    plot_spherical_projection,
    view_colored_verts,
    view_colored_mesh_dir_field,
    color_scalar
)

# Import Python essentials
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(img_path, t_select, c_select, layer_label, patch_mode, patch_size, normal_validity_k,
         normal_validity_thresh, compute_num, grid_n_2dcurve_analysis, debug_2dcurve_analysis, show_figures=True,
         render=False):
    print(f">> Attempting to analysing projected layer {layer_label} image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures

    # ==== Load image ====
    img_unit = load_img_unit(path=img_path)
    img_scale = load_img_scaling(path=img_path)
    if img_scale is None:
        return None

    # ==== Create folder structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Load projected result ====
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    proj_layer = load_array("intensities", folderpath=resdata_dir_layer)
    proj_layer /= proj_layer.max()
    layer_mesh = load_mesh(os.path.join(resdata_dir_layer, "layer_mesh.ply"))

    # ==== Preselect vertices for 2D+ orientation analysis ====
    idxs_sel = np.arange(layer_mesh.vertices.shape[0])
    idxs_sel = filter_normal_validity(mesh=layer_mesh, idxs_sel=idxs_sel, k=normal_validity_k,
                                      threshold=normal_validity_thresh)

    # ==== Compute only points at an interval ====
    idxs_sel = np.random.choice(idxs_sel, size=int(compute_num))
    if len(idxs_sel) == 0:
        print("!! ERROR: No vertices were selected for analysis !!")
        return None
    else:
        print(f">> Selected initially {len(idxs_sel)} for analysis !")

    # ==== Tune plotting parameters ====
    if patch_mode == "radius":
        idxs_neigh = coord_search_radius(layer_mesh.vertices, custom_probes=layer_mesh.vertices[idxs_sel],
                                         r=patch_size)
        title_render = f"Extracted directors w.r.t {patch_size}{img_unit}"
        patch_label = f"r-{patch_size}{img_unit}"
    elif patch_mode == "nearest":
        idxs_neigh = coord_search_neighbours(layer_mesh.vertices, custom_probes=layer_mesh.vertices[idxs_sel],
                                             k=patch_size, n_process=8)
        title_render = f"Extracted directors w.r.t {patch_size - 1} neighbours"
        patch_label = f"k-{patch_size}"
    else:
        print(f"[!] Unknown patch type: {patch_mode}")
        return None

    # ==== Find nearest intensities ====
    proj_layer_neigh = [proj_layer[patch_idxs] for patch_idxs in idxs_neigh]

    # ==== Save vertices for 2D+ orientation analysis ====
    print(f"Num of directors to be calculated: {len(idxs_sel)} !")
    save_array(idxs_sel, "calcindeces", header="idx", folderpath=resdata_dir_layer)
    np.savez_compressed(os.path.join(resdata_dir_layer, "extraction_idxs_neigh.npz"),
                        idxs_neigh=np.asarray(idxs_neigh, dtype=object))
    # ==== Create the tangential bases ====
    neighbors_coords = [layer_mesh.vertices[patch] for patch in idxs_neigh]
    central_normals = layer_mesh.vertex_normals[idxs_sel]

    tan_cords, tan_x, tan_y = tan_proj(neighbors_coords, central_normals)

    # ==== Save the tangential bases ====
    save_array(tan_x, "tan_x", header="t1x,t1y,t1z", folderpath=resdata_dir_layer)
    save_array(tan_y, "tan_y", header="t2x,t2y,t2z", folderpath=resdata_dir_layer)

    # ==== Tune oocal orientation extraction accuracy ====
    box_size = grid_n_2dcurve_analysis // 3
    debug_vert_idx = None

    print(f">> Using local grid of NxN: {grid_n_2dcurve_analysis} and box size: {box_size}...")

    if debug_2dcurve_analysis:
        # ==== Pick single vertex for debug ====
        debug_vert_idx = np.random.choice(np.arange(idxs_sel.shape[0]))
        big_grid = tan_interp_batch(
            coords=[tan_cords[debug_vert_idx]],
            intensities=[proj_layer_neigh[debug_vert_idx]],
            grid_size=grid_n_2dcurve_analysis
        )[2][0]
        print(big_grid.shape)
    else:
        big_grid = np.vstack([grid.T for grid in tan_interp_batch(
            coords=tan_cords,
            intensities=proj_layer_neigh,
            grid_size=grid_n_2dcurve_analysis
        )[2]])

    # ==== Extract directors ====
    directors_2dcurved = batch_2d_orientation(
        big_grid=big_grid, box_size=box_size,
        vertices=layer_mesh.vertices[idxs_sel],
        tan_x=tan_x, tan_y=tan_y,
        debug=debug_2dcurve_analysis,
        debug_idx=debug_vert_idx,
        debug_line_length=1, unit=img_unit
    )

    if not debug_2dcurve_analysis:
        # ==== Save directors ====
        save_array(directors_2dcurved, "directors_2dcurved", header="x,y,z,vx,vy,vz",
                   folderpath=resdata_dir_layer)

        # ==== Plot directors ====
        plot_dir_field(directors=directors_2dcurved, title=title_render,
                       savefig=os.path.join(resfig_dir_layer, "directors_2dcurved.pdf"), veclength=10,
                       hidefig=hidefig)
    else:
        print(f"This was a debug run, not saving results...")
        return None

    sph_proj_phi, sph_proj_theta = spherical_project(pts=layer_mesh.vertices)
    vec_dir_phi, vec_dir_theta = spherical_project_vectors(directors_2dcurved[:, :3],
                                                           directors_2dcurved[:, 3:])

    plot_spherical_projection(phi=sph_proj_phi, theta=sph_proj_theta, intensities=proj_layer, cmap="Greys_r",
                              vec_pos_phi=sph_proj_phi[idxs_sel], vec_pos_theta=sph_proj_theta[idxs_sel],
                              vec_dir_phi=vec_dir_phi, vec_dir_theta=vec_dir_theta,
                              savefig=os.path.join(resfig_dir_layer, "spherical_projection_extracted-directors.pdf"),
                              hidefig=hidefig)
    if render:
        # ==== 3D render patch of 2D+ orientation analysis ====
        full_mesh_colors = np.array(["#FF0000" for _ in range(len(layer_mesh.vertices))])
        random_seed_idx = np.random.choice(range(len(idxs_neigh)))
        global_patch_idxs = idxs_neigh[random_seed_idx]
        full_mesh_colors[global_patch_idxs] = "#FFFF00"
        full_mesh_colors[idxs_sel[random_seed_idx]] = "#0000FF"
        view_colored_verts(
            verts=layer_mesh.vertices,
            colors=list(full_mesh_colors),
            use_orig_color=True
        )

        # ==== 3D render result of 2D+ orientation analysis ====
        view_colored_mesh_dir_field(mesh=layer_mesh, directors=directors_2dcurved,
                                    mesh_vert_colors=color_scalar(proj_layer, normalise=True,
                                                                  cmap="Greys_r"), vec_length=7,
                                    vec_edge_width=0.1)

    save_array(np.column_stack(
        ([patch_size], [normal_validity_k], [normal_validity_thresh], [compute_num], [grid_n_2dcurve_analysis])),
        name=f"{patch_label}_nematic-extraction_parameters",
        header=f"{patch_mode},normal_validity_k,normal_validity_thresh,compute_num,grid_n_2dcurve_analysis",
        folderpath=resdata_dir_layer)
    return layer_label, layer_mesh, proj_layer, idxs_neigh, idxs_sel, directors_2dcurved
