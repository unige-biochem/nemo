"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Mesh image
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts import analysis
from module_scripts.visuals import plot_img, view_mesh
from module_scripts.datahandler import create_resdirs, save_array, save_mesh

# Import python essentials
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(img_path, t_select, c_select, box_size, smooth_factor, smooth_iterations, show_figures=True,
         split_mode="inner/out", render=False):
    print(f">> Attempting to mesh image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Load Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    img_load = analysis.load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load

    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Load Preprocessed Image ====
    img_thresh_load = analysis.load_img_virtual(path=os.path.join(resfig_dir, "img_thresholded.tiff"), t_sel_idx=0,
                                                c_sel_idx=0)
    img_thresh_raw, img_thresh_dim, img_thresh_scale, img_thresh_unit = img_thresh_load

    # ==== Segment Surface Mesh(es) ====
    full_mesh = analysis.marching_cubes(img=img_thresh_raw, scale=img_thresh_scale, level=0.5, step_size=box_size)
    full_mesh_name = "full_mesh"

    # ==== Save Mesh(es) ====
    save_mesh(full_mesh, os.path.join(resdata_dir, f"{full_mesh_name}.ply"))

    # ==== Plot Image Slices with Mesh Overlay ====
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[full_mesh],
             show_mesh_normals=True, cmap="Greys_r",
             savefig=os.path.join(resfig_dir, f"sliced_raw_{full_mesh_name}.pdf"), hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[full_mesh], cmap="Greys",
             max_proj=True, savefig=os.path.join(resfig_dir,
                                                 f"sliced_raw_{full_mesh_name}_maxproj.pdf"),
             hidefig=hidefig)
    print(f"MESH {full_mesh_name}: # VERTICES = {len(full_mesh.vertices)} !")

    if split_mode == "top/bottom":
        # ==== Select TOP / BOTTOM Mesh ====
        mesh_sel_mask = np.einsum('ij,ij->i', np.array([[1, 0, 0] for _ in range(len(full_mesh.vertices))]),
                                  full_mesh.vertex_normals) < 0
    elif split_mode == "inner/out":
        # ==== Select INNER / OUTER Mesh ====
        mesh_sel_mask = np.einsum('ij,ij->i', full_mesh.vertices - np.mean(full_mesh.vertices, axis=0),
                                  full_mesh.vertex_normals) > 0
    else:
        print(f"[!] Unrecognised splitting mode {split_mode} !")
        return None

    # ==== Apply Sub-Mesh Selection ====
    inner_mesh = analysis.sel_submesh(mesh=full_mesh, mask=mesh_sel_mask)
    outer_mesh = analysis.sel_submesh(mesh=full_mesh, mask=~mesh_sel_mask)
    save_mesh(inner_mesh, os.path.join(resdata_dir, "inner_mesh.ply"))
    save_mesh(outer_mesh, os.path.join(resdata_dir, "outer_mesh.ply"))

    plot_img(img=img_raw, scale=img_scale, unit=img_unit, cmap="Greys_r",
             savefig=os.path.join(resfig_dir, f"sliced_raw_{full_mesh_name}-split.pdf"),
             meshes=[inner_mesh, outer_mesh], show_mesh_normals=True,
             mesh_colors=["red", "blue"], hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, cmap="Greys",
             savefig=os.path.join(resfig_dir, f"sliced_raw_{full_mesh_name}-split_maxproj.pdf"),
             meshes=[inner_mesh, outer_mesh], max_proj=True,
             mesh_colors=["red", "blue"], hidefig=hidefig)

    mesh_names = ["full_mesh_smooth", "inner_mesh_smooth", "outer_mesh_smooth"]
    raw_meshes = [full_mesh, inner_mesh, outer_mesh]
    smooth_meshes = []
    smooth_subset_meshes = []
    for i, mesh_i in enumerate(raw_meshes):
        # ==== Smooth Mesh(es) ====
        mesh_i_smooth = analysis.taubin_smooth_mesh(mesh=mesh_i, n_iter=smooth_iterations,
                                                    pass_band=smooth_factor)
        # ==== Save Mesh(es) ====
        save_mesh(mesh_i_smooth, os.path.join(resdata_dir, f"{mesh_names[i]}.ply"))
        smooth_meshes.append(mesh_i_smooth)

        # ==== Plot Image Slices with Mesh Overlay ====
        plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[mesh_i_smooth],
                 show_mesh_normals=True, cmap="Greys_r",
                 savefig=os.path.join(resfig_dir, f"sliced_raw_{mesh_names[i]}.pdf"), hidefig=hidefig)
        plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[mesh_i_smooth], cmap="Greys",
                 max_proj=True, savefig=os.path.join(resfig_dir,
                                                     f"sliced_raw_{mesh_names[i]}_maxproj.pdf"),
                 hidefig=hidefig)

        mesh_i_smooth_subsets = analysis.find_connected_meshes(mesh=mesh_i_smooth)
        sizes = [mesh.vertices.shape[0] for mesh in mesh_i_smooth_subsets]
        mesh_i_smooth_subsets_largest = mesh_i_smooth_subsets[np.argmax(sizes)]

        # ==== Save Mesh(es) ====
        save_mesh(mesh_i_smooth_subsets_largest,
                  os.path.join(resdata_dir, f"{mesh_names[i]}_subset.ply"))
        smooth_subset_meshes.append(mesh_i_smooth_subsets_largest)

        # ==== Plot Image Slices with Mesh Overlay ====
        plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[mesh_i_smooth_subsets_largest],
                 show_mesh_normals=True, cmap="Greys_r",
                 savefig=os.path.join(resfig_dir, f"sliced_raw_{mesh_names[i]}_subset.pdf"),
                 hidefig=hidefig)
        plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=[mesh_i_smooth_subsets_largest], cmap="Greys",
                 max_proj=True, savefig=os.path.join(resfig_dir,
                                                     f"sliced_raw_{mesh_names[i]}_subset_maxproj.pdf"),
                 hidefig=hidefig)

    if render:
        view_mesh(mesh_list=raw_meshes, mesh_colors=["white", "red", "blue"], mesh_titles=["FULL", "INNER", "OUTER"],
                  img=img_raw, scale=img_scale, vec_freq=100, hide_vectors=False, vec_length=10,
                  vec_edge_width=0.2)
        view_mesh(mesh_list=smooth_meshes, mesh_colors=["white", "red", "blue"],
                  mesh_titles=["FULL SMOOTH", "INNER SMOOTH", "OUTER SMOOTH"], img=img_raw, scale=img_scale,
                  vec_freq=100, hide_vectors=False, vec_length=10, vec_edge_width=0.2)
        view_mesh(mesh_list=smooth_subset_meshes, mesh_colors=["white", "red", "blue"],
                  mesh_titles=["FULL SMOOTH SUBSET", "INNER SMOOTH SUBSET", "OUTER SMOOTH SUBSET"], img=img_raw,
                  scale=img_scale, vec_freq=100, hide_vectors=False, vec_length=10, vec_edge_width=0.2)
    save_array(np.column_stack(([box_size], [smooth_factor], [smooth_iterations])),
               name="meshing_parameters",
               header="box_size,smooth_factor,smooth_iterations", folderpath=resdata_dir)
    return raw_meshes, smooth_meshes, smooth_subset_meshes
