"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Calculate the tangential nematic order for a director field on a curved mesh projection.
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
    parser.add_argument("--avg_mode", type=str, default="radius", choices=["radius", "nearest"], help="Averaging mode")
    parser.add_argument("--avg_size", type=float, default=20.0, help="Averaging size")
    return parser.parse_args()


def main(img_path, layer_label, render, avg_mode, avg_size):
    print(f">> Attempting to analysing the tangential field on projected layer {layer_label} image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

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

    # ==== Load 2D+ Directors ====
    idxs_sel = datahandler.load_array("calcindeces", folderpath=resdata_dir_layer).astype(int)
    tan_x = datahandler.load_array("tan_x", folderpath=resdata_dir_layer)
    tan_y = datahandler.load_array("tan_y", folderpath=resdata_dir_layer)
    directors_2dcurved = datahandler.load_array("directors_2dcurved", folderpath=resdata_dir_layer)

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
        neigh_idxs = analysis.coord_search_radius(veccoords, r=patch_size)
        patch_label = f"r-{patch_size}{img_unit}"
        title_hist = f"Avg over {patch_size}{img_unit}: Order Scalar $S$"
        title_render = f"Avg over {patch_size}{img_unit}: Average Directors"
    elif patch_type == "nearest":
        neigh_idxs = analysis.coord_search_neighbours(veccoords, k=patch_size, n_process=8)
        patch_label = f"k-{patch_size}"
        title_hist = f"Avg over {patch_size - 1} neighbours: Order Scalar $S$"
        title_render = f"Avg over {patch_size - 1} neighbours: Average Directors"
    else:
        print(f"[!] Unknown patch type: {patch_type}")
        return None

    # ==== Calculate Curved Nematic Order ====
    s_2dcurv, n_avg_2dcurv = analysis.avg_tan_nem_tens(t1_cov=tan_x, t2_cov=tan_y, directors=directors_2dcurved,
                                                       neigh_idxs=neigh_idxs)

    # ==== Save Curved Nematic Order ====
    datahandler.save_array(s_2dcurv, name=f"S-order_2dcurved_{patch_label}", header="S",
                           folderpath=resdata_dir_layer)
    datahandler.save_array(np.column_stack((veccoords, n_avg_2dcurv)),
                           name=f"directors-avg_2dcurved_{patch_label}",
                           header="x,y,z,vx,vy,vz", folderpath=resdata_dir_layer)

    # ==== Plot Curved Nematic Order ====
    directors_2dcurved_avg = directors_2dcurved.copy()
    directors_2dcurved_avg[:, 3:] = n_avg_2dcurv
    savefig_render = os.path.join(resfig_dir_layer, f"field_intial-avg-nematic_{patch_label}.png")
    savefig_hist = os.path.join(resfig_dir_layer, f"hist_intial-order-s_{patch_label}.png")

    visuals.plot_dir_field(directors=directors_2dcurved_avg, veclength=vec_length, view_init=plot2d_view,
                           veccolor=s_2dcurv, cmap_label="order scalar $S$", title=title_render, manual_vminmax=[0, 1],
                           savefig=savefig_render, figsize=renderfigsize, show_axes=False, hidefig=True)
    visuals.plot_hist(array=s_2dcurv, title=title_hist, savefig=savefig_hist,
                      figsize=histfigsize, xlim=[0, 1], hidefig=True)

    sph_proj_phi, sph_proj_theta = analysis.spherical_project(pts=layer_mesh.vertices)
    vec_dir_phi, vec_dir_theta = analysis.spherical_project_vectors(directors_2dcurved_avg[:, :3],
                                                                    directors_2dcurved_avg[:, 3:])

    visuals.plot_spherical_projection(
        phi=sph_proj_phi,
        theta=sph_proj_theta,
        intensities=proj_layer,
        vec_pos_phi=sph_proj_phi[idxs_sel],
        vec_pos_theta=sph_proj_theta[idxs_sel],
        vec_dir_phi=vec_dir_phi,
        vec_dir_theta=vec_dir_theta,
        vec_cmap_label="order scalar $S$",
        hexgridsize=400,
        scale_factor=5,
        cmap="Greys_r", vec_width=0.0015,
        veccolor=s_2dcurv,
        vec_manual_vminmax=[0, 1],
        arrow_alpha=1.0, figsize=(20, 13), hidefig=True,
        savefig=os.path.join(resfig_dir_layer, f"spherical_projection_field_avg-nematic_{patch_label}.png")
    )

    if render:
        # ==== Visualise Averaging Patch ====
        patch_sel_idx = np.random.choice(range(len(neigh_idxs)))
        patch_color = np.array(["#FF0000" for _ in range(len(veccoords))])
        patch_color[neigh_idxs[patch_sel_idx]] = "#FFFF00"
        patch_color[neigh_idxs[patch_sel_idx][0]] = "#0000FF"
        visuals.view_colored_verts(verts=veccoords, colors=list(patch_color), use_orig_color=True)
        # ==== 3D Render Curved Nematic Order ====
        visuals.view_colored_mesh_dir_field(mesh=layer_mesh, directors=directors_2dcurved_avg,
                                            vec_colors=visuals.color_scalar(s_2dcurv, manual_vminmax=[0, 1]),
                                            mesh_vert_colors=visuals.color_scalar(analysis.normalise_range(proj_layer),
                                                                                  cmap="Greys_r"),
                                            vec_length=vec_length, vec_edge_width=vec_edge_width)
        visuals.view_3d_vector_field(vec_pos=directors_2dcurved_avg[:, :3], vec_dir=directors_2dcurved_avg[:, 3:],
                                     vec_colors=visuals.color_scalar(s_2dcurv, cmap="Spectral", manual_vminmax=[0, 1]),
                                     length=vec_length, pts_size=1,
                                     edge_width=vec_edge_width)

        visuals.view_3d_vector_field(vec_pos=directors_2dcurved_avg[:, :3], vec_dir=directors_2dcurved_avg[:, 3:],
                                     vec_colors=visuals.color_scalar(s_2dcurv, cmap="Spectral"),
                                     length=vec_length, edge_width=vec_edge_width,
                                     verts=layer_mesh.vertices,
                                     verts_colors=visuals.color_scalar(analysis.normalise_range(proj_layer),
                                                                       cmap="Greens_r"))

    return None


if __name__ == "__main__":
    print("======== START NEMO ========")
    args = parse_args()
    main(img_path=args.img_path, layer_label=args.layer_label, render=args.render, avg_mode=args.avg_mode,
         avg_size=args.avg_size)
    print("======== END NEMO ========")
