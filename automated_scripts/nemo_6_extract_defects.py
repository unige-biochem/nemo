"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Extract defects
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_unit,
    load_img_scaling,
    select_geodesic_defects,
    spherical_project,
    spherical_project_vectors,
)
from module_scripts.datahandler import create_resdirs, load_array, load_mesh, save_array
from module_scripts.visuals import (
    plot_dir_field,
    plot_spherical_projection,
    view_colored_mesh_dir_field,
    color_scalar
)

# Import python essentials
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(img_path, t_select, c_select, layer_label, nematic_avg_label, dist_cutoff_defect_localisation,
         max_candidates_defect_localisation,
         show_figures=True, render=False):
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
    directors_2dcurved_avg = load_array(name=f"directors-avg_2dcurved_{nematic_avg_label}",
                                        folderpath=resdata_dir_layer)
    s_2dcurv = load_array(name=f"S-order_2dcurved_{nematic_avg_label}",
                          folderpath=resdata_dir_layer)
    idxs_sel = load_array("calcindeces", folderpath=resdata_dir_layer).astype(int)

    # ==== Find Defects and Inter ====
    defect_idxs, rel_dists = select_geodesic_defects(s_2dcurv, layer_mesh, idxs_sel,
                                                     dist_cutoff=dist_cutoff_defect_localisation,
                                                     unit=img_unit,
                                                     max_candidates=max_candidates_defect_localisation)
    save_array(defect_idxs, name="defect-idxs", header="idx", folderpath=resdata_dir_layer)

    plot_dir_field(directors=directors_2dcurved_avg, veclength=1, veccolor="red",
                   marker=directors_2dcurved_avg[:, :3][defect_idxs],
                   pt_label="Defect Locations", vec_alpha=0.5, freq=5,
                   pt_color=color_scalar(np.linspace(0, 1, len(defect_idxs)), "Set1"),
                   pt_alpha=1.0, cmap_label="order parameter $S$", manual_vminmax=[0, 1],
                   savefig=os.path.join(resfig_dir_layer, f"defect-locations.pdf"), hidefig=hidefig)
    sph_proj_phi, sph_proj_theta = spherical_project(pts=layer_mesh.vertices)
    vec_dir_phi, vec_dir_theta = spherical_project_vectors(directors_2dcurved_avg[:, :3],
                                                           directors_2dcurved_avg[:, 3:])
    plot_spherical_projection(phi=sph_proj_phi, theta=sph_proj_theta, intensities=proj_layer, cmap="Greys_r",
                              vec_pos_phi=sph_proj_phi[idxs_sel], vec_pos_theta=sph_proj_theta[idxs_sel],
                              vec_dir_phi=vec_dir_phi, vec_dir_theta=vec_dir_theta, veccolor=s_2dcurv,
                              vec_manual_vminmax=[0, 1], vec_cmap_label="order scalar $S$",
                              savefig=os.path.join(resfig_dir_layer, f"defect-locations.pdf"), hidefig=hidefig,
                              marker_idxs=idxs_sel[defect_idxs],
                              marker_color=color_scalar(np.linspace(0, 1, len(defect_idxs)), "Set1"))
    defect_coords = layer_mesh.vertices[idxs_sel[defect_idxs]]
    save_array(defect_coords, name="defect_coords", header="x,y,z", folderpath=resdata_dir_layer)
    if render:
        # ==== 3D Render Results ====
        view_colored_mesh_dir_field(mesh=layer_mesh, directors=directors_2dcurved_avg, vec_edge_width=0.1,
                                    vec_colors=color_scalar(s_2dcurv, manual_vminmax=[0, 1]),
                                    marker_size=500,
                                    mesh_vert_colors=color_scalar(proj_layer, normalise=True,
                                                                  cmap="Greys"),
                                    markers=defect_coords,
                                    marker_colors=color_scalar(np.linspace(0, 1, len(defect_coords)),
                                                               "Set1"))
    save_array(np.column_stack(
        ([dist_cutoff_defect_localisation], [max_candidates_defect_localisation])),
        name=f"defect-extraction_parameters",
        header="dist_cutoff_defect_localisation,max_candidates_defect_localisation",
        folderpath=resdata_dir_layer)
    return layer_label, layer_mesh, proj_layer, directors_2dcurved_avg, s_2dcurv, defect_coords
