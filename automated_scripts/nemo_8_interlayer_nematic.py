"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Analyse inter-layer nematic order
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_scaling,
    inter_layer_order,
    spherical_project,
    spherical_project_vectors
)
from module_scripts.datahandler import create_resdirs, load_array, load_mesh, save_array
from module_scripts.visuals import (
    plot_hist,
    plot_dir_field,
    plot_spherical_projection,
    view_colored_mesh_multiple,
    view_colored_mesh_dir_field,
    color_scalar
)
# Import Python essentials
import numpy as np


def main(img_path, t_select, c_select, layer_name_1, patch_label_1, layer_name_2, patch_label_2, show_figures=True,
         render=False):
    print(f">> Attempting to analyse the order between {layer_name_1} and {layer_name_2}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures

    # ==== Load image ====
    img_scale = load_img_scaling(path=img_path)
    if img_scale is None:
        return None
    # ==== Create folder structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    print(
        f"Choosing layer 1 with name {layer_name_1} and patch label {patch_label_1} | layer 2 with name {layer_name_2} and patch label {patch_label_2}")
    layer_mesh_1 = load_mesh(os.path.join(resdata_dir, layer_name_1, "layer_mesh.ply"))
    layer_mesh_2 = load_mesh(os.path.join(resdata_dir, layer_name_2, "layer_mesh.ply"))
    if (layer_mesh_1 is None) or (layer_mesh_2 is None):
        print(f"[!] Mesh {layer_mesh_1} and/or {layer_mesh_2} are not available!")
        return None

    inter_layer_s, field_1, field_2 = inter_layer_order(layer_name_1=layer_name_1,
                                                        patch_label_1=patch_label_1,
                                                        layer_name_2=layer_name_2,
                                                        patch_label_2=patch_label_2,
                                                        resdata_dir=resdata_dir,
                                                        director_name_prefix="directors-avg_2dcurved_")
    save_array(inter_layer_s,
               f"{layer_name_1}_{patch_label_1}_VS_{layer_name_2}_{patch_label_2}_interlayer_order",
               header="strength", folderpath=resdata_dir)

    plot_hist(inter_layer_s, title=r"Inter-Layer Nematic Order $S$", xlim=[0, 1],
              savefig=os.path.join(resfig_dir,
                                   f"{layer_name_1}_{patch_label_1}_VS_{layer_name_2}_{patch_label_2}_hist_inter_layer_s.png"),
              hidefig=hidefig)

    plot_dir_field(directors=field_1, veclength=10, view_init=(20, 0),
                   veccolor=inter_layer_s,
                   cmap_label=r"Inter-Layer Nematic Order $S$", show_axes=False, manual_vminmax=[0, 1],
                   cmap="Spectral",
                   savefig=os.path.join(resfig_dir,
                                        f"{layer_name_1}_{patch_label_1}_VS_{layer_name_2}_{patch_label_2}_nematic-field_inter_layer_s.png"),
                   hidefig=hidefig)
    plotted_vecfields = np.concatenate((field_2, field_1), axis=0)
    try:
        sph_proj_phi, sph_proj_theta = spherical_project(pts=plotted_vecfields[:, :3])
        vec_dir_phi, vec_dir_theta = spherical_project_vectors(plotted_vecfields[:, :3],
                                                               plotted_vecfields[:, 3:])
        plot_spherical_projection(phi=sph_proj_phi, theta=sph_proj_theta, intensities=np.ones_like(sph_proj_phi),
                                  cmap="Greys", vec_pos_phi=sph_proj_phi, vec_pos_theta=sph_proj_theta,
                                  vec_dir_phi=vec_dir_phi, vec_dir_theta=vec_dir_theta,
                                  veccolor=np.concatenate((np.ones(len(field_2)) * 0.5, inter_layer_s), axis=0),
                                  vec_manual_vminmax=[0, 1], vec_cmap="Spectral",
                                  vec_cmap_label=r"Inter-Layer Nematic Order $S$",
                                  savefig=os.path.join(resfig_dir,
                                                       f"{layer_name_1}_{patch_label_1}_VS_{layer_name_2}_{patch_label_2}_nematic-field_inter_layer_s.png"),
                                  hidefig=hidefig)
    except Exception as e:
        print(f"Encountered error during spherical projection: {e}")

    proj_layer_1 = load_array("intensities", folderpath=os.path.join(resdata_dir, layer_name_1))
    proj_layer_2 = load_array("intensities", folderpath=os.path.join(resdata_dir, layer_name_2))

    s_2dcurv_layer_1 = load_array(name=f"S-order_2dcurved_{patch_label_1}",
                                  folderpath=os.path.join(resdata_dir, layer_name_1))
    s_2dcurv_layer_2 = load_array(name=f"S-order_2dcurved_{patch_label_2}",
                                  folderpath=os.path.join(resdata_dir, layer_name_2))
    if render:
        import trimesh
        view_colored_mesh_multiple(mesh_list=[layer_mesh_1, layer_mesh_2],
                                   vert_colors_list=[
                                       color_scalar(proj_layer_1, normalise=True,
                                                    cmap="Greens_r"),
                                       color_scalar(proj_layer_2, normalise=True,
                                                    cmap="Blues_r")])
        mask = np.ones(plotted_vecfields.shape[0], dtype=bool)
        view_colored_mesh_dir_field(mesh=layer_mesh_1, directors=plotted_vecfields[mask],
                                    vec_colors=color_scalar(
                                        np.concatenate((np.ones(len(field_2)) * 0.5, inter_layer_s), axis=0)[
                                            mask],
                                        manual_vminmax=[0, 1], cmap="Spectral", ),
                                    mesh_vert_colors="black",
                                    vec_edge_width=0.5, vec_length=10)

        view_colored_mesh_dir_field(mesh=trimesh.util.concatenate(layer_mesh_1, layer_mesh_2),
                                    directors=plotted_vecfields,
                                    vec_colors=color_scalar(
                                        np.concatenate((s_2dcurv_layer_2, s_2dcurv_layer_1)),
                                        manual_vminmax=[0, 1],
                                        cmap="Spectral"),
                                    mesh_vert_colors=color_scalar(np.concatenate(
                                        (proj_layer_1, proj_layer_2)), normalise=True,
                                        cmap="Greys_r"),
                                    vec_edge_width=0.3, vec_length=20)
    return layer_mesh_1, layer_mesh_2, proj_layer_1, proj_layer_2, field_1, field_2, plotted_vecfields, s_2dcurv_layer_1, s_2dcurv_layer_2, inter_layer_s
