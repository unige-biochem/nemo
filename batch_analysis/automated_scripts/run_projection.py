"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Project a .tiff file onto a .ply mesh
Requires Python 3.9.6 and packages from requirements.txt.
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO scripts
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts import analysis, datahandler, visuals

# Import python essentials
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, default="", type=str, help="Path to the input TIFF image")
    parser.add_argument("--dist_min", required=True, type=float, help="Minimum distance to start projection")
    parser.add_argument("--dist_max", required=True, type=float, help="Maximum distance to end projection")
    parser.add_argument("--dist_num", default=30, type=int, help="Number projection points")
    parser.add_argument("--proj_mode", default="max", choices=["max", "mean"], help="Projection mode (max/mean)")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, help="Render in 3D")
    args = parser.parse_args()
    return args


def main(img_path, dist_min, dist_max, dist_num, proj_mode, render):
    print(f">> Attempting to project image {img_path}!")
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

    try:
        sampl_mesh = datahandler.load_mesh(os.path.join(resdata_dir, "sampling_mesh.ply"), recalc_normals=False,
                                           clean=False)
    except:
        print(f">> Failed to load sampling mesh from {img_path}!")
        return None

    # ==== Projection Logic ====
    layer_label = f"proj_{dist_min}_to_{dist_max}_{img_unit}"

    # ==== Specify Custom Minimum ====
    dist_min_custom = None
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    if not os.path.exists(resdata_dir_layer):
        os.makedirs(resdata_dir_layer)
    if not os.path.exists(resfig_dir_layer):
        os.makedirs(resfig_dir_layer)

    # ==== Save Sampling Vertices and Normals ====
    datahandler.save_array(sampl_mesh.vertices, "verts", header="x,y,z", folderpath=resdata_dir_layer)
    datahandler.save_array(sampl_mesh.vertex_normals, "normals", header="nx,ny,nz", folderpath=resdata_dir_layer)
    # layer_mesh = analysis.scale_mesh(mesh=sampl_mesh, distance=dist_middle)
    layer_mesh = sampl_mesh.copy()
    datahandler.save_mesh(layer_mesh, filepath=os.path.join(resdata_dir_layer, "layer_mesh.ply"))

    # ==== Project onto Mesh ====
    proj_layer = analysis.proj2mesh(img=img_raw, mesh=layer_mesh, min_dist_per_vert=dist_min_custom,
                                    scale=img_scale, min_dist=dist_min, max_dist=dist_max,
                                    num_dist=dist_num, mode=proj_mode, show_proj=True,
                                    savefig=os.path.join(resfig_dir_layer, f"distgraph.png"), unit=img_unit,
                                    normalise=False, hidefig=True)

    # ==== Save Projection ====
    datahandler.save_array(proj_layer, "intensities", header="I", folderpath=resdata_dir_layer)

    # ==== Plot Projected Result ====
    mercator_x, mercator_y = analysis.spherical_project(pts=layer_mesh.vertices)
    visuals.plot_spherical_projection(phi=mercator_x, theta=mercator_y, intensities=proj_layer, hexgridsize=400,
                                      savefig=os.path.join(resfig_dir_layer, "mercator.png"), hidefig=True)
    visuals.plot_maxproj_pts(verts=layer_mesh.vertices, colors=proj_layer, cmap="Greens", hexsize=200, unit=img_unit,
                             figsize=(18, 8), savefig=os.path.join(resfig_dir_layer, "maxproj.png"), hidefig=True)

    if render:
        visuals.view_colored_mesh(mesh=layer_mesh, mesh_blending="opaque",
                                  vert_colors=visuals.color_scalar(analysis.normalise_range(proj_layer),
                                                                   cmap="Greens"), img=img_raw, scale=img_scale)
    return None


if __name__ == "__main__":
    print("======== START NEMO ========")
    args = parse_args()
    main(img_path=args.img_path, dist_min=args.dist_min, dist_max=args.dist_max, dist_num=args.dist_num,
         proj_mode=args.proj_mode, render=args.render)
    print("======== END NEMO ========")
