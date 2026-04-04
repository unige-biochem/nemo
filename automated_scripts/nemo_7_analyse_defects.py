"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Analyse defects
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_scaling,
    curvature_by_srf_fit,
    interpolate_on_mesh,
    coord_search_radius,
    coord_search_neighbours,
    tan_proj,
    curved_nem_charge,
    spherical_project,
    spherical_project_vectors,
    compute_defect_polarisations
)
from module_scripts.datahandler import create_resdirs, load_array, load_mesh, save_array
from module_scripts.visuals import (
    plot_dir_field,
    plot_hist,
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


def main(img_path, t_select, c_select, layer_label, nematic_avg_label, topcharge_mode, topcharge_size, topcurv_radius,
         topcurv_interpk, pol_mode, pol_size, show_figures=True, render=False):
    print(f">> Attempting to analysing the tangential field on projected layer {layer_label} image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures

    # ==== Load Image ====
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
    defect_idxs = load_array(name="defect-idxs", folderpath=resdata_dir_layer).astype(int)

    # ==== Calculate Gaussian Curvature for Topological Charge Analysis ====
    curv_charge_quick = curvature_by_srf_fit(mesh=layer_mesh, num_sample=3000, patch_size=topcurv_radius,
                                             patch_mode="radius", filter_boundary=False, debug=True,
                                             gauss_crop_range=None, boundary_excl_factor=0.0)
    # Results given as curv_charge_quick = C_gauss, C_mean, Gauss_idxs, mean_idxs, C_tensors, C_tensors_mixed, tensor_idxs
    gauss_curv_smooth = interpolate_on_mesh(mesh=layer_mesh, value_idxs=curv_charge_quick[2],
                                            values=curv_charge_quick[0], k=topcurv_interpk)

    # ==== Calculate Curved Topological Charge ====
    if topcharge_mode == "radius":
        charge_patch_idxs = coord_search_radius(layer_mesh.vertices,
                                                custom_probes=layer_mesh.vertices[idxs_sel[defect_idxs]],
                                                r=topcharge_size)
    elif topcharge_mode == "nearest":
        charge_patch_idxs = coord_search_neighbours(layer_mesh.vertices,
                                                    custom_probes=layer_mesh.vertices[idxs_sel[defect_idxs]],
                                                    k=topcharge_size)
    else:
        print(f"[!] Unknown patch type: {topcharge_mode}")
        return None

    defect_idxs_calc = [np.flatnonzero(idxs_sel == lst[0])[0]
                        for lst in charge_patch_idxs
                        if np.any(idxs_sel == lst[0])]

    _, tan_x_all, tan_y_all = tan_proj(layer_mesh.vertices[:, np.newaxis], layer_mesh.vertex_normals)
    m_charge, calc_charge_loop_idxs = curved_nem_charge(mesh=layer_mesh, directors=directors_2dcurved_avg,
                                                        calc_idxs=defect_idxs_calc,
                                                        director_indeces=idxs_sel,
                                                        tan_x=tan_x_all, tan_y=tan_y_all,
                                                        c_gauss=gauss_curv_smooth,
                                                        loop_angle_precision=1, patch_mode=topcharge_mode,
                                                        patch_size=topcharge_size, debug=True,
                                                        correct_orientation=True)
    print(f"Final number of defects: {len(m_charge)} !")
    plot_dir_field(directors=directors_2dcurved_avg, veclength=0.02, veccolor=s_2dcurv,
                   marker=np.vstack([layer_mesh.vertices[i] for i in calc_charge_loop_idxs]),
                   pt_label="Charge Calculation Line",
                   pt_alpha=1.0, cmap_label="order parameter $S$", manual_vminmax=[0, 1],
                   savefig=os.path.join(resfig_dir_layer, f"top-charge-loops.pdf"),
                   hidefig=hidefig)

    m_charge_extended = np.full(len(layer_mesh.vertices), 0.0)
    if topcharge_mode == "radius":
        charge_patch_idxs_calc = coord_search_radius(layer_mesh.vertices,
                                                     custom_probes=layer_mesh.vertices[
                                                         idxs_sel[defect_idxs_calc]],
                                                     r=topcharge_size)
    elif topcharge_mode == "nearest":
        charge_patch_idxs_calc = coord_search_neighbours(layer_mesh.vertices,
                                                         custom_probes=layer_mesh.vertices[
                                                             idxs_sel[defect_idxs_calc]],
                                                         k=topcharge_size)
    else:
        charge_patch_idxs_calc = None
        print(f"[!] Unknown patch type: {topcharge_mode}")
    if isinstance(charge_patch_idxs_calc, list):
        flat_idxs = np.concatenate([np.atleast_1d(np.array(x)) for x in charge_patch_idxs_calc if len(x) > 0])
    else:
        flat_idxs = np.ravel(charge_patch_idxs_calc)

    m_charge_extended[flat_idxs] = np.repeat(m_charge, [len(np.atleast_1d(x)) for x in charge_patch_idxs_calc])
    m_charge_extended = m_charge_extended.ravel()

    plot_dir_field(directors=directors_2dcurved_avg, veclength=10, veccolor=m_charge_extended[idxs_sel],
                   cmap="rainbow",
                   title=r"TOTAL CHARGE$\approx$" + f"{np.nansum(m_charge):.3}",
                   cmap_label="topological charge $m$",
                   manual_vminmax=[-1, 1], show_axes=False,
                   marker=layer_mesh.vertices[idxs_sel[defect_idxs_calc]],
                   savefig=os.path.join(resfig_dir_layer, f"top-charge-shaded.pdf"),
                   hidefig=hidefig)

    # ==== Save Curved Topological Charge ====
    save_array(m_charge, name=f"top-charge_2dcurved", header="m",
               folderpath=resdata_dir_layer)
    save_array(defect_idxs_calc, name=f"top-charge_2dcurved_idxs", header="idx",
               folderpath=resdata_dir_layer)

    # ==== Plot Curved Topological Charge ====
    plot_hist(m_charge, title=f"Sum(m)={np.nansum(m_charge)}",
              savefig=os.path.join(resfig_dir_layer, f"hist_top-charge.pdf"),
              hidefig=hidefig)

    sph_proj_phi, sph_proj_theta = spherical_project(pts=layer_mesh.vertices)
    vec_dir_phi, vec_dir_theta = spherical_project_vectors(directors_2dcurved_avg[:, :3],
                                                           directors_2dcurved_avg[:, 3:])
    plot_spherical_projection(
        phi=sph_proj_phi,
        theta=sph_proj_theta,
        intensities=proj_layer,
        vec_pos_phi=sph_proj_phi[idxs_sel],
        vec_pos_theta=sph_proj_theta[idxs_sel],
        vec_dir_phi=vec_dir_phi,
        vec_dir_theta=vec_dir_theta,
        vec_cmap_label="order scalar $S$",
        hexgridsize=150,
        scale_factor=10,
        cmap="Greys_r",
        veccolor=s_2dcurv,
        vec_manual_vminmax=[0, 1],
        arrow_alpha=1.0, figsize=(20, 13),
        marker_idxs=idxs_sel[defect_idxs_calc],
        marker_color=color_scalar(m_charge, manual_vminmax=[-1, 1], cmap="rainbow"),
        savefig=os.path.join(resfig_dir_layer, f"defect-charges.pdf"),
        hidefig=hidefig
    )

    pol_vecfield, pol_idxs = compute_defect_polarisations(
        mesh=layer_mesh,
        idxs_sel=idxs_sel,
        directors=directors_2dcurved_avg,
        vertex_normals=layer_mesh.vertex_normals.copy(),
        defect_idxs_calc=defect_idxs_calc,
        m_charge=np.round(m_charge, 2),
        patch_type=pol_mode,
        patch_size=pol_size, show_profile=True, hidefig=hidefig
    )
    charge_pol_linked_idxs = np.argsort(defect_idxs_calc)[
        np.searchsorted(defect_idxs_calc, pol_idxs, sorter=np.argsort(defect_idxs_calc))]

    save_array(pol_vecfield, name=f"def-pol_2dcurved", header="x,y,z,vx,vy,vz",
               folderpath=resdata_dir_layer)
    save_array(charge_pol_linked_idxs, name=f"def-pol_2dcurved_idxs", header="idx",
               folderpath=resdata_dir_layer)

    rotation_angles = [1, 1, 1]
    sph_proj_phi, sph_proj_theta = spherical_project(pts=layer_mesh.vertices, rotate=rotation_angles)
    vec_dir_phi, vec_dir_theta = spherical_project_vectors(directors_2dcurved_avg[:, :3],
                                                           directors_2dcurved_avg[:, 3:],
                                                           rotate=rotation_angles)
    pol_dir_phi, pol_dir_theta = spherical_project_vectors(pol_vecfield[:, :3], pol_vecfield[:, 3:],
                                                           rotate=rotation_angles)
    plot_spherical_projection(
        phi=sph_proj_phi,
        theta=sph_proj_theta,
        intensities=proj_layer,
        vec_pos_phi=sph_proj_phi[idxs_sel],
        vec_pos_theta=sph_proj_theta[idxs_sel],
        vec_dir_phi=vec_dir_phi,
        vec_dir_theta=vec_dir_theta,
        vec_cmap_label="order scalar $S$",
        hexgridsize=400,
        scale_factor=2,
        cmap="Greys_r", vec_width=0.0015,
        veccolor=s_2dcurv, vec_manual_vminmax=[0, 1],
        arrow_alpha=1.0, figsize=(22, 8),
        marker_idxs=idxs_sel[defect_idxs_calc],
        marker_color=color_scalar(m_charge, manual_vminmax=[-1, 1], cmap="rainbow"),
        marker_vec=(pol_dir_phi, pol_dir_theta, idxs_sel[pol_idxs]),
        marker_vec_scale=20, marker_vec_width=0.005, aspect="equal",
        marker_vec_color=color_scalar(m_charge[charge_pol_linked_idxs], manual_vminmax=[-1, 1],
                                      cmap="rainbow"),
        savefig=os.path.join(resfig_dir_layer, f"defect-polarisations.pdf"), hidefig=hidefig
    )
    if render:
        view_colored_mesh_dir_field(mesh=layer_mesh, directors=directors_2dcurved_avg, mesh_shading="flat",
                                    mesh_vert_colors=color_scalar(proj_layer, normalise=True,
                                                                  cmap="Greys_r"),
                                    vec_colors=color_scalar(s_2dcurv, manual_vminmax=[0, 1]),
                                    markers=layer_mesh.vertices[idxs_sel[defect_idxs_calc]], vec_edge_width=0.3,
                                    marker_colors=color_scalar(m_charge, manual_vminmax=[-1, 1],
                                                               cmap="rainbow"), marker_size=400,
                                    marker_vectors=pol_vecfield, marker_vectors_length=20, vec_length=10,
                                    marker_vector_width=3,
                                    marker_vectors_color=color_scalar(m_charge[charge_pol_linked_idxs],
                                                                      manual_vminmax=[-1, 1],
                                                                      cmap="rainbow"), )
    save_array(np.column_stack(
        ([topcharge_size], [topcurv_radius], [topcurv_interpk])),
        name=f"defect-analysis-charge_parameters",
        header=f"{topcharge_mode},topcurv_radius,topcurv_interpk",
        folderpath=resdata_dir_layer)

    save_array(np.column_stack(
        ([pol_size])),
        name=f"defect-analysis-polarisation_parameters",
        header=f"{pol_mode}",
        folderpath=resdata_dir_layer)
    return layer_label, layer_mesh, proj_layer, directors_2dcurved_avg, s_2dcurv, defect_idxs_calc, m_charge, charge_pol_linked_idxs, pol_vecfield
