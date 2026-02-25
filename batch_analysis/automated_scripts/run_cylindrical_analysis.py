"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Convert gastruloid mesh to cylindrical coordinate system
Requires Python 3.9.6 and packages from requirements.txt.
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO scripts
import os
import sys

import numpy as np
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts import analysis, datahandler, visuals, extra_gastruloids

# Import python essentials
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="", required=True, type=str, help="Path to the input TIFF image")
    parser.add_argument("--show_figures", action=argparse.BooleanOptionalAction, help="Show figures")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, help="Render result")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, help="Overwrite analysis")
    parser.add_argument("--force_pca_use", action=argparse.BooleanOptionalAction, help="Force use of PCA")
    parser.add_argument("--voxel_size", type=float, default=5.0,
                        help="Voxel size")
    parser.add_argument("--spline_num_pts", type=int, default=2000,
                        help="Number of points to interpolate along the spline")
    parser.add_argument("--spline_order_k", type=int, default=2,
                        help="Order of the spline polynomial")
    parser.add_argument("--spline_smooth_factor", type=float, default=200.0,
                        help="Smoothing factor for spline fitting")
    parser.add_argument("--curve_sampling_interval", type=int, default=3,
                        help="Interval for sampling points from the curve")
    parser.add_argument("--spline_step_u", type=float, default=0.01,
                        help="Step size for spline parameterization")
    return parser.parse_args()


def main(img_path, spline_num_pts=2000, voxel_size=2.0, spline_order_k=2, spline_smooth_factor=100.0,
         curve_sampling_interval=2, spline_step_u=0.01, show_figures=False, overwrite=False, force_pca_use=False,
         render=False):
    print(f">> Attempting to perform cylindrical analysis of gastruloid {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    # ==== Load Image ====
    img_scale = analysis.load_img_scaling(path=img_path)
    img_unit = "um"
    if img_scale is None:
        return None
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = datahandler.create_resdirs(img_path)
    mesh_name = "sampling_mesh.ply"
    mesh_path = os.path.join(resdata_dir, mesh_name)
    if os.path.exists(mesh_path):
        gastr_mesh = datahandler.load_mesh(os.path.join(resdata_dir, "sampling_mesh.ply"), recalc_normals=False,
                                           clean=False)
        print(f"Number of sampling vertices: {len(gastr_mesh.vertices)} !")
        gastr_mesh = datahandler.load_mesh(os.path.join(resdata_dir, "sampling_mesh.ply"), recalc_normals=False,
                                           clean=False)
    else:
        print(f"[!] No sampling mesh found for {img_path}!")
        return None

    if not os.path.exists(os.path.join(resdata_dir, "3d_midline_curve.csv")) or overwrite:

        centerline, voxelised_mesh = extra_gastruloids.skeletonise_mesh(mesh=gastr_mesh, voxel_size=voxel_size)
        centerline = extra_gastruloids.order_line_points(points=centerline)
        centerline = centerline[::curve_sampling_interval]
        print(f"Reduced curve to {len(centerline)} points!")
        if len(centerline) <= 3 or force_pca_use:
            print(f"Not possible to fit curve, reverting to PCA...")
            centerline_fitted = extra_gastruloids.pca_axis_line_extend_inside_mesh(mesh=gastr_mesh,
                                                                                   num_points=spline_num_pts,
                                                                                   oversample=100)
        else:
            try:
                centerline_fitted = extra_gastruloids.spline_fit_curve_3d_extend_inside_mesh(
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
            centerline_fitted = extra_gastruloids.planarise_curve(centerline_fitted)
            centerline_fitted = extra_gastruloids.reparametrize_curve_by_curvature(centerline_fitted)
        datahandler.save_array(centerline_fitted, name="3d_midline_curve", header="x,y,z",
                               folderpath=resdata_dir)
    else:
        centerline_fitted = datahandler.load_array(name="3d_midline_curve", folderpath=resdata_dir)

    mesh_s, mesh_rho, mesh_phi = extra_gastruloids.cylindrical_along_curve(points=gastr_mesh.vertices,
                                                                           curve=centerline_fitted)

    datahandler.save_array(np.column_stack((mesh_s, mesh_rho, mesh_phi)), name="mesh_s-rho-phi", header="s,rho,phi",
                           folderpath=resdata_dir)

    mesh_extent = np.ptp(gastr_mesh.vertices, axis=0)
    gastr_mesh_simplified = gastr_mesh.simplify_quadric_decimation(face_count=10000)
    visuals.plot_img(img=np.zeros(shape=(1, 1, 1)), scale=(1, 1, 1), unit="um",
                     meshes=[gastr_mesh_simplified, trimesh.Trimesh(vertices=centerline_fitted)], mesh_alpha=[0.1, 1.0],
                     slice_depth=mesh_extent.max(), mesh_colors=["grey", "blue"],
                     max_proj=True, hidefig=hidefig, mesh_thick=0.2,
                     savefig=os.path.join(resfig_dir, "3d_midline_curve.png"))

    visuals.plot_rho_profile(mesh_s=mesh_s, mesh_rho=mesh_rho, mesh_phi=mesh_phi, img_unit=img_unit, hidefig=hidefig,
                             savefig=os.path.join(resfig_dir, f"rho-profile.png"))
    # visuals.view_colored_mesh(gastr_mesh, vert_colors="white", markers=centerline_fitted, mesh_shading="flat",
    #                           mesh_opacity=0.6,
    #                           mesh_blending="translucent_no_depth",
    #                           marker_colors=visuals.color_scalar(np.linspace(0, 1, len(centerline_fitted)),
    #                                                              cmap="Blues"))

    if render:
        visuals.view_colored_mesh_multiple([gastr_mesh, gastr_mesh, gastr_mesh, gastr_mesh],
                                           vert_colors_list=["white",
                                                             visuals.color_scalar(mesh_s,
                                                                                  manual_vminmax=[0, mesh_s.max()],
                                                                                  cmap="inferno"),
                                                             visuals.color_scalar(mesh_rho,
                                                                                  manual_vminmax=[0, mesh_rho.max()],
                                                                                  cmap="Spectral"),
                                                             visuals.color_scalar(mesh_phi,
                                                                                  manual_vminmax=[-np.pi, np.pi],
                                                                                  cmap="hsv")],
                                           markers=centerline_fitted, mesh_shading="flat",
                                           mesh_opacity_list=[0.6, 1.0, 1.0, 1.0],
                                           marker_colors=visuals.color_scalar(np.linspace(0, 1, len(centerline_fitted)),
                                                                              cmap="Blues"))

    return centerline_fitted, mesh_s, mesh_rho, mesh_phi


if __name__ == "__main__":
    print("======== START NEMO ========")
    main(img_path=args.img_path, show_figures=args.show_figures, spline_num_pts=args.spline_num_pts,
         voxel_size=args.voxel_size, spline_order_k=args.spline_order_k, spline_smooth_factor=args.spline_smooth_factor,
         curve_sampling_interval=args.curve_sampling_interval, spline_step_u=args.spline_step_u,
         overwrite=args.overwrite, force_pca_use=args.force_pca_use, render=args.render)
    print("======== END NEMO ========")
