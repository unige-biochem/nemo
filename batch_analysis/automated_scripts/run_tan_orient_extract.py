"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Extract the tangential orientation of features on a curved mesh projection.
Requires Python 3.9.6 and packages from requirements.txt.
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts import analysis, datahandler, visuals

# Import python essentials
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, default="", type=str, help="Path to the input TIFF image")
    parser.add_argument("--layer_label", required=True, default="", type=str, help="Name of layer to be analysed")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, help="Render in 3D")
    parser.add_argument("--patch_orientation", default=1500, type=int,
                        help="Orientation extraction patch size (nneighbours)")
    # new arguments
    parser.add_argument("--normal_validity_k", type=int, default=20,
                        help="Number of nearest neighbors used for normal validity computation")
    parser.add_argument("--normal_validity_thresh", type=float, default=0.99,
                        help="Threshold for normal vector validity")

    parser.add_argument("--compute_num", type=int, default=3000,
                        help="Number of directors")

    parser.add_argument("--grid_n_2dcurve_analysis", type=int, default=30,
                        help="Grid resolution for 2D curved analysis")
    parser.add_argument("--debug_2dcurve_analysis", action=argparse.BooleanOptionalAction,
                        help="Enable debug output for 2D curved analysis")

    parser.add_argument("--patch_mode", type=str, default="radius", choices=["radius", "nearest"],
                        help="Averaging mode")
    parser.add_argument("--patch_size", type=float, default=20.0, help="Averaging size")
    parser.add_argument("--show_plots", default=False, action="store_true", help="Show plots")
    return parser.parse_args()


def main(img_path, layer_label, render, patch_mode, patch_size, normal_validity_k,
         normal_validity_thresh, compute_num, grid_n_2dcurve_analysis, debug_2dcurve_analysis, show_figures):
    print(f">> Attempting to analysing projected layer {layer_label} image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None
    hidefig = not show_figures
    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    # ==== Choose Time Step and Channel ====
    t_select = 0
    c_select = 0
    # ==== [Optional] Reduce Resolution ====
    z_reduce_factor = 1  # 1 means no reduction
    xy_reduce_factor = 1  # 1 means no reduction
    # ==== [Optional] Normalise Intensities to [0, 1] ====
    normalise_intensities = False  # can be set to False
    # ==== [Optional] Overwrite Scaling with FIJI Values ====
    custom_scaling = None  # please use (z, y, x)
    # ==== [Optional] Overwrite unit with FIJI Values ====
    custom_unit = "um"  # as string
    # ==== Load Image ====
    img_load = analysis.load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select,
                                         reduce_xy=xy_reduce_factor,
                                         reduce_z=z_reduce_factor, norm_vals=normalise_intensities,
                                         custom_scaling=custom_scaling, custom_unit=custom_unit)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = datahandler.create_resdirs(img_path)

    # ==== Load Projected Result ====
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    proj_layer = datahandler.load_array("intensities", folderpath=resdata_dir_layer)
    proj_layer /= proj_layer.max()
    layer_mesh = datahandler.load_mesh(os.path.join(resdata_dir_layer, "layer_mesh.ply"), recalc_normals=False,
                                       clean=False)

    # ==== Preselect Vertices for 2D+ Orientation Analysis ====
    idxs_sel = np.arange(layer_mesh.vertices.shape[0])
    idxs_sel = analysis.filter_normal_validity(mesh=layer_mesh, idxs_sel=idxs_sel, k=normal_validity_k,
                                               threshold=normal_validity_thresh)

    # ==== Compute only points at Interval ====
    idxs_sel = np.random.choice(idxs_sel, size=int(compute_num))
    if len(idxs_sel) == 0:
        print("!! ERROR: No vertices were selected for analysis !!")
        return None
    else:
        print(f">> Selected initially {len(idxs_sel)} for analysis !")

    # ==== Tune Plotting parameters ====
    if patch_mode == "radius":
        idxs_neigh = analysis.coord_search_radius(layer_mesh.vertices, custom_probes=layer_mesh.vertices[idxs_sel],
                                                  r=patch_size)
        title_render = f"Extracted directors w.r.t {patch_size}{img_unit}"
    elif patch_mode == "nearest":
        idxs_neigh = analysis.coord_search_neighbours(layer_mesh.vertices, custom_probes=layer_mesh.vertices[idxs_sel],
                                                      k=patch_size, n_process=8)
        title_render = f"Extracted directors w.r.t {patch_size - 1} neighbours"
    else:
        print(f"[!] Unknown patch type: {patch_mode}")
        return None

    # ==== Find Nearest Intensities ====
    proj_layer_neigh = [proj_layer[patch_idxs] for patch_idxs in idxs_neigh]

    # ==== Save Vertices for 2D+ Orientation Analysis ====
    print(f"Num of directors to be calculated: {len(idxs_sel)} !")
    datahandler.save_array(idxs_sel, "calcindeces", header="idx", folderpath=resdata_dir_layer)

    # ==== Create the Tangential Bases ====
    neighbors_coords = [layer_mesh.vertices[patch] for patch in idxs_neigh]
    central_normals = layer_mesh.vertex_normals[idxs_sel]

    tan_cords, tan_x, tan_y = analysis.tan_proj(neighbors_coords, central_normals)

    # ==== Save the Tangential Bases ====
    datahandler.save_array(tan_x, "tan_x", header="t1x,t1y,t1z", folderpath=resdata_dir_layer)
    datahandler.save_array(tan_y, "tan_y", header="t2x,t2y,t2z", folderpath=resdata_dir_layer)

    # ==== Tune Local Orientation Extraction Accuracy ====
    box_size = grid_n_2dcurve_analysis // 3
    debug_vert_idx, debug_grid_x, debug_grid_y, debug_grid_z = None, None, None, None

    print(f">> Using local grid of NxN: {grid_n_2dcurve_analysis} and box size: {box_size}...")

    if debug_2dcurve_analysis:
        # ==== Pick single vertex for debug ====
        debug_vert_idx = np.random.choice(np.arange(idxs_sel.shape[0]))
        big_grid = analysis.tan_interp_batch(
            coords=[tan_cords[debug_vert_idx]],
            intensities=[proj_layer_neigh[debug_vert_idx]],
            grid_size=grid_n_2dcurve_analysis
        )[2][0]
        print(big_grid.shape)
    else:
        big_grid = np.vstack([grid.T for grid in analysis.tan_interp_batch(
            coords=tan_cords,
            intensities=proj_layer_neigh,
            grid_size=grid_n_2dcurve_analysis
        )[2]])

    # ==== Extract Directors ====
    directors_2dcurved = analysis.batch_2d_orientation(
        big_grid=big_grid, box_size=box_size,
        vertices=layer_mesh.vertices[idxs_sel],
        tan_x=tan_x, tan_y=tan_y,
        debug=debug_2dcurve_analysis,
        debug_idx=debug_vert_idx,
        debug_line_length=1
    )

    if not debug_2dcurve_analysis:
        # ==== Save Directors ====
        datahandler.save_array(directors_2dcurved, "directors_2dcurved", header="x,y,z,vx,vy,vz",
                               folderpath=resdata_dir_layer)

        # ==== Plot Directors ====
        visuals.plot_dir_field(directors=directors_2dcurved, title=title_render,
                               savefig=os.path.join(resfig_dir_layer, "directors_2dcurved.png"), veclength=10,
                               hidefig=hidefig)
    else:
        print(f"This was a debug run, not saving results...")
        return None

    sph_proj_phi, sph_proj_theta = analysis.spherical_project(pts=layer_mesh.vertices)
    vec_dir_phi, vec_dir_theta = analysis.spherical_project_vectors(directors_2dcurved[:, :3],
                                                                    directors_2dcurved[:, 3:])

    visuals.plot_spherical_projection(
        phi=sph_proj_phi,
        theta=sph_proj_theta,
        intensities=proj_layer,
        vec_pos_phi=sph_proj_phi[idxs_sel],
        vec_pos_theta=sph_proj_theta[idxs_sel],
        vec_dir_phi=vec_dir_phi,
        vec_dir_theta=vec_dir_theta,
        hexgridsize=400,
        scale_factor=5,
        cmap="Greens",
        arrow_alpha=0.7, figsize=(13, 10), hidefig=hidefig,
        savefig=os.path.join(resfig_dir_layer, "spherical_projection_extracted-directors.png")
    )
    if render:
        # ==== 3D Render Vertices for 2D+ Orientation Analysis ====
        visuals.view_colored_verts(verts=layer_mesh.vertices[idxs_sel],
                                   colors=visuals.color_scalar(proj_layer[idxs_sel], cmap="Greens_r"), scale=img_scale)

        # ==== 3D Render Directors ====
        veclength = 10
        vecwidth = 0.5
        visuals.view_3d_vector_field(vec_pos=directors_2dcurved[:, :3], vec_dir=directors_2dcurved[:, 3:],
                                     vec_colors="red",
                                     verts=layer_mesh.vertices,
                                     verts_colors=visuals.color_scalar(analysis.normalise_range(proj_layer),
                                                                       cmap="Greens_r"),
                                     edge_width=veclength / 6, length=veclength, vec_opacity=0.5, pts_size=1,
                                     pts_opacity=0.8, img=None, scale=img_scale)
        visuals.view_colored_mesh_dir_field(mesh=layer_mesh, directors=directors_2dcurved,
                                            mesh_vert_colors=visuals.color_scalar(analysis.normalise_range(proj_layer),
                                                                                  cmap="Greys_r"), vec_length=veclength,
                                            vec_edge_width=vecwidth)
    return None


if __name__ == "__main__":
    print("======== START NEMO ========")
    args = parse_args()
    main(img_path=args.img_path, layer_label=args.layer_label, render=args.render,
         normal_validity_k=args.normal_validity_k,
         normal_validity_thresh=args.normal_validity_thresh,
         compute_num=args.compute_num,
         grid_n_2dcurve_analysis=args.grid_n_2dcurve_analysis,
         debug_2dcurve_analysis=args.debug_2dcurve_analysis, patch_mode=args.patch_mode, patch_size=args.patch_size,
         show_figures=args.show_figures, )
    print("======== END NEMO ========")
