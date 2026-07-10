"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Mesh image
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts import analysis
from module_scripts.visuals import plot_img, view_mesh
from module_scripts.datahandler import create_resdirs, save_array, save_mesh

# Import Python essentials
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def _process_and_save(mesh_or_meshes, name, resdata_dir, resfig_dir, img_raw, img_scale, img_unit, hidefig,
                      mesh_colors=None):
    meshes = mesh_or_meshes if isinstance(mesh_or_meshes, list) else [mesh_or_meshes]
    meshes = [m for m in meshes if m is not None]
    if not meshes:
        return

    for m in meshes:
        m.remove_degenerate_faces()
        m.remove_duplicate_faces()

    if not isinstance(mesh_or_meshes, list) and meshes:
        save_mesh(meshes[0], os.path.join(resdata_dir, f"{name}.ply"))

    plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=meshes,
             show_mesh_normals=False, cmap="Greys_r", mesh_colors=mesh_colors,
             savefig=os.path.join(resfig_dir, f"sliced_raw_{name}.png"), hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=meshes,
             show_mesh_normals=True, cmap="Greys_r", mesh_colors=mesh_colors,
             savefig=os.path.join(resfig_dir, f"sliced_raw_{name}_normals.png"), hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, meshes=meshes, cmap="Greys",
             max_proj=True, mesh_colors=mesh_colors,
             savefig=os.path.join(resfig_dir, f"sliced_raw_{name}_maxproj.png"), hidefig=hidefig)


def _get_largest_submesh(mesh):
    try:
        if mesh is None:
            return None
        submeshes = analysis.find_connected_meshes(mesh=mesh)
        if not submeshes:
            return None
        return submeshes[np.argmax([m.vertices.shape[0] for m in submeshes])]
    except Exception as e:
        print(f"[!] Issue finding submeshes: {e}")
        return None


def main(img_path, t_select, c_select, box_size, taubin_smooth_passband, taubin_smooth_iterations, show_figures=True,
         split_mode="inner/out", render=False):
    print(f">> Attempting to mesh image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Load image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    img_load = analysis.load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load

    # ==== Create folder structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Load preprocessed image ====
    img_thresh_load = analysis.load_img_virtual(path=os.path.join(resfig_dir, "img_thresholded.tiff"), t_sel_idx=0,
                                                c_sel_idx=0)
    img_thresh_raw, img_thresh_dim, img_thresh_scale, img_thresh_unit = img_thresh_load

    # ==== Segment surface mesh(es) ====
    full_mesh = analysis.marching_cubes(img=img_thresh_raw, scale=img_thresh_scale, level=0.5, step_size=box_size)
    full_mesh_name = "full_mesh"

    raw_meshes = [{"mesh": full_mesh, "color": "white", "title": "FULL"}]
    smooth_meshes = []
    smooth_subset_meshes = []

    # ==== Process Full Mesh ====
    _process_and_save(full_mesh, full_mesh_name, resdata_dir, resfig_dir, img_raw, img_scale, img_unit, hidefig)
    print(f"MESH {full_mesh_name}: # VERTICES = {len(full_mesh.vertices)} !")

    # ==== Smooth Full Mesh ====
    full_mesh_smooth = analysis.taubin_smooth_mesh(mesh=full_mesh, n_iter=taubin_smooth_iterations,
                                                   pass_band=taubin_smooth_passband)
    if full_mesh_smooth is not None:
        smooth_meshes.append({"mesh": full_mesh_smooth, "color": "white", "title": "FULL SMOOTH"})
        _process_and_save(full_mesh_smooth, f"{full_mesh_name}_smooth", resdata_dir, resfig_dir, img_raw, img_scale,
                          img_unit, hidefig)

    # ==== Subset Full Mesh ====
    mesh_smooth_subset_largest = _get_largest_submesh(full_mesh_smooth)
    if mesh_smooth_subset_largest is not None:
        smooth_subset_meshes.append(
            {"mesh": mesh_smooth_subset_largest, "color": "white", "title": "FULL SMOOTH SUBSET"})
        _process_and_save(mesh_smooth_subset_largest, f"{full_mesh_name}_smooth_subset", resdata_dir, resfig_dir,
                          img_raw, img_scale, img_unit, hidefig)

    # ==== Evaluate Split Mode ====
    if split_mode == "along_z":
        mesh_sel_mask = full_mesh.vertex_normals[:, 0] < 0
    elif split_mode == "radial_spherical":
        mesh_sel_mask = np.einsum('ij,ij->i', full_mesh.vertices - np.mean(full_mesh.vertices, axis=0),
                                  full_mesh.vertex_normals) > 0
    elif split_mode == "radial_cylindrical":
        radial_vecs = full_mesh.vertices - np.mean(full_mesh.vertices, axis=0)
        radial_vecs[:, 0] = 0
        mesh_sel_mask = np.einsum('ij,ij->i', radial_vecs, full_mesh.vertex_normals) > 0
    else:
        print(f"[!] Unrecognised splitting mode {split_mode} !")
        return None
    print(f">> Applying splitting mode {split_mode}...")
    # ==== Apply sub-mesh selection ====
    try:
        inner_mesh = analysis.sel_submesh(mesh=full_mesh, mask=mesh_sel_mask)
        outer_mesh = analysis.sel_submesh(mesh=full_mesh, mask=~mesh_sel_mask)
        print(inner_mesh, outer_mesh)
        valid_split_meshes = []
        valid_split_colors = []
        split_items = []

        if inner_mesh is not None:
            inner_mesh.remove_degenerate_faces()
            inner_mesh.remove_duplicate_faces()
            save_mesh(inner_mesh, os.path.join(resdata_dir, "inner_mesh.ply"))
            valid_split_meshes.append(inner_mesh)
            valid_split_colors.append("red")
            raw_meshes.append({"mesh": inner_mesh, "color": "red", "title": "INNER"})
            split_items.append((inner_mesh, "inner_mesh_smooth", "INNER SMOOTH", "red", "INNER SMOOTH SUBSET"))

        if outer_mesh is not None:
            outer_mesh.remove_degenerate_faces()
            outer_mesh.remove_duplicate_faces()
            save_mesh(outer_mesh, os.path.join(resdata_dir, "outer_mesh.ply"))
            valid_split_meshes.append(outer_mesh)
            valid_split_colors.append("blue")
            raw_meshes.append({"mesh": outer_mesh, "color": "blue", "title": "OUTER"})
            split_items.append((outer_mesh, "outer_mesh_smooth", "OUTER SMOOTH", "blue", "OUTER SMOOTH SUBSET"))

        if valid_split_meshes:
            _process_and_save(valid_split_meshes, f"{full_mesh_name}-split", resdata_dir, resfig_dir, img_raw,
                              img_scale, img_unit, hidefig, mesh_colors=valid_split_colors)

        for mesh_i, name_smooth, title_smooth, color, title_subset in split_items:
            mesh_i_smooth = analysis.taubin_smooth_mesh(mesh=mesh_i, n_iter=taubin_smooth_iterations,
                                                        pass_band=taubin_smooth_passband)
            if mesh_i_smooth is not None:
                smooth_meshes.append({"mesh": mesh_i_smooth, "color": color, "title": title_smooth})
                _process_and_save(mesh_i_smooth, name_smooth, resdata_dir, resfig_dir, img_raw, img_scale, img_unit,
                                  hidefig)

                mesh_i_subset = _get_largest_submesh(mesh_i_smooth)
                if mesh_i_subset is not None:
                    smooth_subset_meshes.append({"mesh": mesh_i_subset, "color": color, "title": title_subset})
                    _process_and_save(mesh_i_subset, f"{name_smooth}_subset", resdata_dir, resfig_dir, img_raw,
                                      img_scale, img_unit, hidefig)

    except Exception as e:
        print(f"[!] Could not split mesh: {e}")

    if render:
        if raw_meshes:
            view_mesh(mesh_list=[m["mesh"] for m in raw_meshes], mesh_colors=[m["color"] for m in raw_meshes],
                      mesh_titles=[m["title"] for m in raw_meshes], img=img_raw, scale=img_scale, vec_freq=100,
                      hide_vectors=False, vec_length=10, vec_edge_width=0.2)
        if smooth_meshes:
            view_mesh(mesh_list=[m["mesh"] for m in smooth_meshes], mesh_colors=[m["color"] for m in smooth_meshes],
                      mesh_titles=[m["title"] for m in smooth_meshes], img=img_raw, scale=img_scale, vec_freq=100,
                      hide_vectors=False, vec_length=10, vec_edge_width=0.2)
        if smooth_subset_meshes:
            view_mesh(mesh_list=[m["mesh"] for m in smooth_subset_meshes],
                      mesh_colors=[m["color"] for m in smooth_subset_meshes],
                      mesh_titles=[m["title"] for m in smooth_subset_meshes], img=img_raw, scale=img_scale,
                      vec_freq=100, hide_vectors=False, vec_length=10, vec_edge_width=0.2)

    save_array(np.array([[box_size, taubin_smooth_passband, taubin_smooth_iterations]]),
               name="meshing_parameters", header="box_size,taubin_smooth_passband,taubin_smooth_iterations",
               folderpath=resdata_dir)
    return None
