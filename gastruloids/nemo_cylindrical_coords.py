"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Convert gastruloid mesh to cylindrical coordinate system
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import load_img_scaling, load_img_unit
from module_scripts.datahandler import create_resdirs, load_mesh, save_array, load_array
from gastruloids.extra_gastruloids import (
    skeletonise_mesh,
    order_line_points,
    pca_axis_line_extend_inside_mesh,
    spline_fit_curve_3d_extend_inside_mesh,
    planarise_curve,
    reparametrize_curve_by_curvature,
    cylindrical_along_curve,
    create_s_phi_basis
)

# Import python essentials
import numpy as np
import trimesh


def main(img_path, t_select, c_select, spline_num_pts=2000, voxel_size=2.0, spline_order_k=2,
         spline_smooth_factor=100.0, curve_sampling_interval=2, spline_step_u=0.01, show_figures=True, overwrite=False,
         force_pca_use=False, render=False):
    print(f">> Attempting to perform cylindrical analysis of gastruloid {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    # ==== Load Image ====
    img_scale = load_img_scaling(path=img_path)
    img_unit = load_img_unit(path=img_path)
    if img_scale is None:
        return None
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")
    mesh_name = "sampling_mesh.ply"
    mesh_path = os.path.join(resdata_dir, mesh_name)
    if os.path.exists(mesh_path):
        gastr_mesh = load_mesh(os.path.join(resdata_dir, "sampling_mesh.ply"))
        print(f"Number of sampling vertices: {len(gastr_mesh.vertices)} !")
    else:
        print(f"[!] No sampling mesh found for {img_path}!")
        return None

    if not os.path.exists(os.path.join(resdata_dir, "3d_midline_curve.csv")) or overwrite:

        centerline, voxelised_mesh = skeletonise_mesh(mesh=gastr_mesh, voxel_size=voxel_size)
        centerline = order_line_points(points=centerline)
        centerline = centerline[::curve_sampling_interval]
        print(f"Reduced curve to {len(centerline)} points!")
        if len(centerline) <= 3 or force_pca_use:
            print(f"Not possible to fit curve, reverting to PCA...")
            centerline_fitted = pca_axis_line_extend_inside_mesh(mesh=gastr_mesh,
                                                                 num_points=spline_num_pts,
                                                                 oversample=100)
        else:
            try:
                centerline_fitted = spline_fit_curve_3d_extend_inside_mesh(
                    curve=centerline,
                    mesh=gastr_mesh,
                    order_k=spline_order_k,
                    num_pts=spline_num_pts,
                    smooth=spline_smooth_factor,
                    step_u=spline_step_u
                )
            except Exception as e:
                print(f"[!] Error finding curve: {e}")
                centerline_fitted = None
            centerline_fitted = planarise_curve(centerline_fitted)
            centerline_fitted = reparametrize_curve_by_curvature(centerline_fitted)
        save_array(centerline_fitted, name="3d_midline_curve", header="x,y,z",
                   folderpath=resdata_dir)
    else:
        centerline_fitted = load_array(name="3d_midline_curve", folderpath=resdata_dir)

    mesh_s, mesh_rho, mesh_phi = cylindrical_along_curve(points=gastr_mesh.vertices,
                                                         curve=centerline_fitted)

    save_array(np.column_stack((mesh_s, mesh_rho, mesh_phi)), name="mesh_s-rho-phi", header="s,rho,phi",
               folderpath=resdata_dir)
    e_s, e_phi, e_rho = create_s_phi_basis(points=gastr_mesh.vertices, curve=centerline_fitted,
                                           normals=gastr_mesh.vertex_normals)
    save_array(e_s, name="e_s", header="x,y,z", folderpath=resdata_dir)
    save_array(e_phi, name="e_phi", header="x,y,z", folderpath=resdata_dir)
    save_array(e_rho, name="e_rho", header="x,y,z", folderpath=resdata_dir)
    mesh_extent = 10 * np.ptp(gastr_mesh.vertices, axis=0)
    gastr_mesh_simplified = gastr_mesh.simplify_quadric_decimation(face_count=10000)

    plot_img(img=np.zeros(shape=(1, 1, 1)), scale=(1, 1, 1), unit=img_unit,
             meshes=[gastr_mesh_simplified, trimesh.Trimesh(vertices=centerline_fitted)],
             mesh_alpha=0.5,
             slice_depth=mesh_extent.max(),
             mesh_colors=["grey",
                          color_scalar(np.linspace(0, 1, len(centerline_fitted)), cmap="Spectral")],
             max_proj=True, hidefig=hidefig, mesh_thick=0.5,
             savefig=os.path.join(resfig_dir, "3d_midline_curve.pdf"))

    plot_rho_profile(mesh_s=mesh_s, mesh_rho=mesh_rho, mesh_phi=mesh_phi, img_unit=img_unit, hidefig=hidefig,
                     savefig=os.path.join(resfig_dir, f"rho-profile.pdf"))

    if render:
        view_colored_mesh_multiple([gastr_mesh, gastr_mesh, gastr_mesh, gastr_mesh],
                                   vert_colors_list=["white",
                                                     color_scalar(mesh_s,
                                                                  manual_vminmax=[0, mesh_s.max()],
                                                                  cmap="inferno"),
                                                     color_scalar(mesh_rho,
                                                                  manual_vminmax=[0, mesh_rho.max()],
                                                                  cmap="Spectral"),
                                                     color_scalar(mesh_phi,
                                                                  manual_vminmax=[-np.pi, np.pi],
                                                                  cmap="hsv")],
                                   markers=centerline_fitted, mesh_shading="flat",
                                   mesh_opacity_list=[0.6, 1.0, 1.0, 1.0],
                                   marker_colors=color_scalar(np.linspace(0, 1, len(centerline_fitted)),
                                                              cmap="Blues"))
        view_colored_mesh(gastr_mesh, vert_colors="white", markers=centerline_fitted, mesh_shading="flat",
                          mesh_opacity=0.6,
                          mesh_blending="translucent_no_depth",
                          marker_colors=color_scalar(np.linspace(0, 1, len(centerline_fitted)),
                                                     cmap="Blues"))

    return centerline_fitted, mesh_s, mesh_rho, mesh_phi, e_s, e_rho, e_phi
