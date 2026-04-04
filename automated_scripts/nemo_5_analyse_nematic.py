"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Analyse tangential nematic order
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_scaling,
    coord_search_radius,
    coord_search_neighbours,
    avg_tan_nem_tens,
    spherical_project,
    spherical_project_vectors,
    load_img_unit
)
from module_scripts.datahandler import create_resdirs, load_array, load_mesh, save_array
from module_scripts.visuals import (
    plot_dir_field,
    plot_hist,
    plot_spherical_projection,
    view_colored_verts,
    view_colored_mesh_dir_field,
    color_scalar
)

# Import python essentials
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(img_path, t_select, c_select, layer_label, avg_mode, avg_size, show_figures=True, render=False):
    print(f">> Attempting to analysing the tangential field on projected layer {layer_label} image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures

    # ==== Load Image ====
    img_unit = load_img_unit(path=img_path)
    img_scale = load_img_scaling(path=img_path)
    if img_scale is None:
        return None
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Load Projected Result ====
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    proj_layer = load_array("intensities", folderpath=resdata_dir_layer)
    proj_layer /= proj_layer.max()
    layer_mesh = load_mesh(os.path.join(resdata_dir_layer, "layer_mesh.ply"))

    # ==== Load 2D+ Directors ====
    idxs_sel = load_array("calcindeces", folderpath=resdata_dir_layer).astype(int)
    tan_x = load_array("tan_x", folderpath=resdata_dir_layer)
    tan_y = load_array("tan_y", folderpath=resdata_dir_layer)
    directors_2dcurved = load_array("directors_2dcurved", folderpath=resdata_dir_layer)

    # ==== Tune Curved Nematic Analysis Number of Neighbours or Radius ====
    patch_type = avg_mode
    patch_size = avg_size

    # ==== Tune Plotting parameters ====
    vec_length = 10
    vec_edge_width = vec_length / 6
    plot2d_view = (20, 0)
    renderfigsize = (6, 5)
    histfigsize = (4, 3)
    veccoords = directors_2dcurved[:, :3]
    if patch_type == "radius":
        idxs_neigh = coord_search_radius(veccoords, r=patch_size)
        nematic_avg_label = f"r-{patch_size}{img_unit}"
        title_hist = f"Avg over {patch_size}{img_unit}: Order Scalar $S$"
        title_render = f"Avg over {patch_size}{img_unit}: Average Directors"
    elif patch_type == "nearest":
        idxs_neigh = coord_search_neighbours(veccoords, k=patch_size, n_process=8)
        nematic_avg_label = f"k-{patch_size}"
        title_hist = f"Avg over {patch_size - 1} neighbours: Order Scalar $S$"
        title_render = f"Avg over {patch_size - 1} neighbours: Average Directors"
    else:
        print(f"[!] Unknown patch type: {patch_type}")
        return None

    np.savez_compressed(os.path.join(resdata_dir_layer, f"{nematic_avg_label}_idxs_neigh.npz"),
             idxs_neigh=np.asarray(idxs_neigh, dtype=object))

    # ==== Calculate Curved Nematic Order ====
    s_2dcurv, n_avg_2dcurv = avg_tan_nem_tens(t1_cov=tan_x, t2_cov=tan_y, directors=directors_2dcurved,
                                              neigh_idxs=idxs_neigh)

    # ==== Save Curved Nematic Order ====
    save_array(s_2dcurv, name=f"S-order_2dcurved_{nematic_avg_label}", header="S",
               folderpath=resdata_dir_layer)
    save_array(np.column_stack((veccoords, n_avg_2dcurv)),
               name=f"directors-avg_2dcurved_{nematic_avg_label}",
               header="x,y,z,vx,vy,vz", folderpath=resdata_dir_layer)

    # ==== Plot Curved Nematic Order ====
    directors_2dcurved_avg = directors_2dcurved.copy()
    directors_2dcurved_avg[:, 3:] = n_avg_2dcurv

    savefig_render = os.path.join(resfig_dir_layer, f"field_intial-avg-nematic_{nematic_avg_label}.pdf")
    savefig_hist = os.path.join(resfig_dir_layer, f"hist_intial-order-s_{nematic_avg_label}.pdf")

    plot_dir_field(directors=directors_2dcurved_avg, veclength=vec_length, view_init=plot2d_view,
                   veccolor=s_2dcurv, cmap_label="order scalar $S$", title=title_render, manual_vminmax=[0, 1],
                   savefig=savefig_render, figsize=renderfigsize, show_axes=False, hidefig=hidefig)
    plot_hist(array=s_2dcurv, title=title_hist, savefig=savefig_hist,
              figsize=histfigsize, xlim=[0, 1], hidefig=hidefig)

    sph_proj_phi, sph_proj_theta = spherical_project(pts=layer_mesh.vertices)
    vec_dir_phi, vec_dir_theta = spherical_project_vectors(directors_2dcurved_avg[:, :3],
                                                           directors_2dcurved_avg[:, 3:])

    plot_spherical_projection(phi=sph_proj_phi, theta=sph_proj_theta, intensities=proj_layer, cmap="Greys_r",
                              vec_pos_phi=sph_proj_phi[idxs_sel], vec_pos_theta=sph_proj_theta[idxs_sel],
                              vec_dir_phi=vec_dir_phi, vec_dir_theta=vec_dir_theta, veccolor=s_2dcurv,
                              vec_manual_vminmax=[0, 1], vec_cmap_label="order scalar $S$",
                              savefig=os.path.join(resfig_dir_layer,
                                                   f"spherical_projection_field_avg-nematic_{nematic_avg_label}.pdf"),
                              hidefig=hidefig)

    if render:
        # ==== Visualise Averaging Patch ====
        patch_sel_idx = np.random.choice(range(len(idxs_neigh)))
        patch_color = np.array(["#FF0000" for _ in range(len(veccoords))])
        patch_color[idxs_neigh[patch_sel_idx]] = "#FFFF00"
        patch_color[idxs_neigh[patch_sel_idx][0]] = "#0000FF"
        view_colored_verts(verts=veccoords, colors=list(patch_color), use_orig_color=True)
        # ==== 3D Render Curved Nematic Order ====
        view_colored_mesh_dir_field(mesh=layer_mesh, directors=directors_2dcurved_avg,
                                    vec_colors=color_scalar(s_2dcurv, manual_vminmax=[0, 1]),
                                    mesh_vert_colors=color_scalar(proj_layer, normalise=True,
                                                                  cmap="Greys_r"),
                                    vec_length=vec_length, vec_edge_width=vec_edge_width)
    save_array(np.column_stack(([avg_size])),
               name=f"{nematic_avg_label}_nematic-analysis_parameters",
               header=f"{avg_mode}",
               folderpath=resdata_dir_layer)
    return layer_label, layer_mesh, proj_layer, directors_2dcurved, nematic_avg_label, directors_2dcurved_avg, s_2dcurv
