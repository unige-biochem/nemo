"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Embryo Flattening for Magdalena Schindler (EMBL Heidelberg, Petridou Lab).
Requires Python 3.9.6 and packages from requirements.txt.
Author: Konstantinos Andreadis
"""
# Import custom scripts
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts import analysis, datahandler, visuals, simulation
from importlib import reload

for module in (analysis, datahandler, visuals, simulation):
    reload(module)

# Import python essentials
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import tifffile
import re, glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Patth to the input TIFF image")
    parser.add_argument("--result_path", type=str, required=True, help="Path to result folder")

    # parser.add_argument("--img_path", type=str,
    #                     default="/Users/andreadi/Downloads/CTL_TG_His_lyn_E25_20230926_block1_New-01And-2To7-TP10To13.tif",
    #                     help="Path to the input TIFF image")
    # parser.add_argument("--result_path", type=str, default="/Users/andreadi/Downloads/cluster_result/",
    #                     help="Path to result folder")
    parser.add_argument("--t_segmentation", type=int, default=0, help="Time index for segmentation")
    parser.add_argument("--c_segmentation", type=int, default=0, help="Channel index for segmentation")
    parser.add_argument("--min_proj_dist", type=float, default=0.0, help="Minimum projection distance")
    parser.add_argument("--max_proj_dist", type=float, default=120.0, help="Maximum projection distance")
    parser.add_argument("--proj_thickness", type=float, default=60.0, help="Projection thickness")
    parser.add_argument("--num_proj_samples", type=int, default=10, help="Number of projection samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_path = args.img_path
    result_path = args.result_path
    t_segmentation = args.t_segmentation
    c_segmentation = args.c_segmentation
    min_proj_dist = args.min_proj_dist
    max_proj_dist = args.max_proj_dist
    proj_thickness = args.proj_thickness
    num_proj_samples = args.num_proj_samples
    print(">> Parameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("======== START script ========")
    # Additional Projection Parameters
    rotate_proj_angles = [0, 0, 0]
    flip_proj_direction = True
    proj_mode = "mean"

    # Specify Segmentation Paths
    print(f"Selected image path: {img_path}")
    ct_segmentation_label = f"t-{t_segmentation}_c-{c_segmentation}"
    tifname_split = os.path.splitext(os.path.basename(img_path))
    img_path_ct_segmentation = os.path.join(result_path, f"{tifname_split[0]}", ct_segmentation_label)
    print(f"Segmentation at t={t_segmentation} c={c_segmentation} !")

    # Load
    hyperstack_dimensions = analysis.load_img_dimensions(path=img_path)
    c_projection_range = np.arange(hyperstack_dimensions["C"])
    t_projection_range = np.arange(hyperstack_dimensions["T"])
    print(f"Full Time Range: {t_projection_range} | Channel Range: {c_projection_range}")

    # Initialise Script Run(s)
    hide_fig_output = True
    segment = 0
    project = 0
    savetiff = True

    if not segment and not project and not savetiff:
        print("[!] Please choose either segmentation, projection or result saving !")
        sys.exit()

    if segment:
        print("======== START segmentation ========")
        # ==== Load Image ====
        img_load = analysis.load_img_virtual(path=img_path, t_sel_idx=t_segmentation,
                                             c_sel_idx=c_segmentation, reduce_xy=1,
                                             reduce_z=1, norm_vals=False,
                                             custom_scaling=None, custom_unit="um")
        if img_load is None:
            print(f"Image load failed, exiting script...")
            sys.exit()

        img_raw, img_dim, img_scale, img_unit = img_load

        # ==== Create Folder Structure ====
        resdata_dir, resfig_dir = datahandler.create_resdirs(img_path_ct_segmentation)

        # ==== Optional: Pad Image above & below Z/Y/X with 0s ====
        # z_pad = 5
        # img_raw = np.pad(img_raw, pad_width=((z_pad, z_pad), (0, 0), (0, 0)), mode='constant')
        # img_dim = img_raw.shape

        # ==== Plot Image Slices and Max Projections ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, max_proj=True, cmap='Greens',
                         savefig=os.path.join(resfig_dir, f"sliced_maxproj_raw.png"), hidefig=hide_fig_output)
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                         savefig=os.path.join(resfig_dir, f"sliced_raw.png"),
                         cmap="Greens_r", hidefig=hide_fig_output)

        # ==== Blur Image ====
        sigma = 8
        sigma_ = analysis.rescale_val_xyz(val=sigma, scale=img_scale)
        img_blur = analysis.gaussian_blur(img=img_raw, sigma=sigma_, renorm=False)

        # ==== Plot Image Slices ====
        visuals.plot_img(img=img_blur, scale=img_scale, unit=img_unit, cmap="inferno",
                         savefig=os.path.join(resfig_dir, "sliced_blur.png"), hidefig=hide_fig_output)

        # ==== Yen Threshold Image ====
        # img_thresh_val = np.min([analysis.yen_thresh(img_blur[:, :, img_dim[2] // i]) for i in np.arange(2, 6)])
        img_thresh_val = np.min(
            [analysis.yen_thresh(img_blur[:, :, img_dim[2] // 2]),
             analysis.yen_thresh(img_blur[:, img_dim[1] // 2, :]),
             analysis.yen_thresh(img_blur[img_dim[0] // 2, :, :])])
        img_thresh_val *= 0.1

        # img_thresh_val = np.min([analysis.yen_thresh(img_blur[:, :, int(img_dim[2] * i)]) for i in np.linspace(0.4, 0.6, 10)])

        # ==== Binarise Image using Threshold ====
        img_thresh = analysis.thresh_img(img_blur, img_thresh_val)

        # ==== Plot Image Slices ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                         savefig=os.path.join(resfig_dir, "sliced_thresh.png"),
                         cmap="inferno", thresh_mask=img_thresh, hidefig=hide_fig_output)

        # ==== Segment Surface Mesh(es) ====
        mcub_res = 2
        full_mesh = analysis.marching_cubes(img=img_thresh, scale=img_scale, level=0.5, step_size=mcub_res)

        # ==== Select TOP / BOTTOM Mesh ====
        mesh_sel_mask = np.einsum('ij,ij->i', np.array([[1, 0, 0] for i in range(len(full_mesh.vertices))]),
                                  full_mesh.vertex_normals) < 0

        # ==== Apply Sub-Mesh Selection ====
        upper_surface = analysis.sel_submesh(mesh=full_mesh, mask=mesh_sel_mask)

        # ==== Save Mesh(es) ====
        datahandler.save_mesh(full_mesh, os.path.join(resdata_dir, "full_mesh.ply"))
        datahandler.save_mesh(upper_surface, os.path.join(resdata_dir, "upper_surface.ply"))

        # ==== Plot Image Slices with Mesh Overlay ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                         savefig=os.path.join(resfig_dir, "sliced_raw_full-mesh.png"),
                         meshes=[full_mesh, upper_surface],
                         mesh_colors=["grey", "red"], hidefig=hide_fig_output)

        # ==== Smooth Mesh(es) ====
        smooth_factor = 0.001
        smooth_iterations = 200
        upper_surface_smooth = analysis.taubin_smooth_mesh(mesh=upper_surface, n_iter=smooth_iterations,
                                                           pass_band=smooth_factor)

        # ==== Save Mesh(es) ====
        datahandler.save_mesh(upper_surface_smooth, os.path.join(resdata_dir, "upper_surface_smooth.ply"))

        # ==== Plot Image Slices with Mesh Overlay ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                         savefig=os.path.join(resfig_dir, "sliced_smooth_upper-surface.png"),
                         meshes=[upper_surface, upper_surface_smooth], mesh_colors=["black", "red"],
                         hidefig=hide_fig_output)

        # ==== Fit Sphere ====
        sphere_params = analysis.fit_sphere(points=upper_surface_smooth.vertices)
        sphere_x0, sphere_y0, sphere_z0, sphere_radius = sphere_params
        sphere_mesh = trimesh.creation.icosphere(radius=sphere_radius, subdivisions=9)
        sphere_mesh.vertices += [sphere_x0, sphere_y0, sphere_z0]
        datahandler.save_array(np.array(sphere_params)[:, np.newaxis].T, "sphere_fit", header="x,y,z,radius",
                               folderpath=resdata_dir)

        # ==== Crop Sphere ====
        zmax, zmin = sphere_mesh.vertices[:, 0].max(), sphere_mesh.vertices[:, 0].min()
        sphere_crop_mask = sphere_mesh.vertices[:, 0] > (zmax - zmin) / 1.3 + zmin
        sphere_mesh_cropped = analysis.sel_submesh(mesh=sphere_mesh, mask=sphere_crop_mask)
        # sphere_mesh_cropped = analysis.subdivide_mesh(sphere_mesh_cropped, 1.5)
        print(f"Num of sphere vertices: {sphere_mesh_cropped.vertices.shape[0]}")

        # ==== Plot Image Slices with Mesh Overlay ====
        visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                         savefig=os.path.join(resfig_dir, "sliced_raw_sphere-fit.png"), cmap="Greens_r",
                         meshes=[sphere_mesh_cropped], mesh_alpha=1.0, hidefig=hide_fig_output, slice_depth=5)

        # ==== Select Sampling Mesh ====
        sampl_mesh = sphere_mesh_cropped.copy()

        # ==== Save Sampling Mesh ====
        datahandler.save_mesh(sampl_mesh, os.path.join(resdata_dir, "sampling_mesh.ply"))
        print("======== END segmentation ========")

    if project:
        print("======== START PROJECTION ========")
        for timepoint in t_projection_range:
            for channel in c_projection_range:
                print(f"======== TIMEPOINT = {timepoint} | CHANNEL = {channel} ========")
                # ==== Choose Time Step and Channel ====
                t_select = timepoint
                c_select = channel
                ct_label = f"t-{t_select}_c-{c_select}"

                # ==== Load Image ====
                img_load = analysis.load_img_virtual(path=img_path, t_sel_idx=t_select,
                                                     c_sel_idx=c_select, reduce_xy=1,
                                                     reduce_z=1, norm_vals=False,
                                                     custom_scaling=None, custom_unit="um")
                if img_load is None:
                    print(f"Image load failed for {ct_label}, skipping...")
                    continue

                img_raw, img_dim, img_scale, img_unit = img_load

                # ==== Create Folder Structure ====
                tifname_split = os.path.splitext(os.path.basename(img_path))
                img_path_ct = os.path.join(result_path, f"{tifname_split[0]}", ct_label)
                resdata_dir, resfig_dir = datahandler.create_resdirs(img_path_ct)

                # ==== Plot Image Slices and Max Projections ====
                visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, max_proj=True, cmap='Greens',
                                 savefig=os.path.join(resfig_dir, f"sliced_maxproj_raw.png"), hidefig=hide_fig_output)
                visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                                 savefig=os.path.join(resfig_dir, f"sliced_raw.png"),
                                 cmap="Greens_r", hidefig=hide_fig_output)

                # ----- Load Sphere Fit -----
                resdata_dir_seg, resfig_dir_seg = datahandler.create_resdirs(img_path_ct_segmentation)
                print(f"Loading sphere fit from {resdata_dir_seg} ...")
                sampl_mesh = datahandler.load_mesh(os.path.join(resdata_dir_seg, "sampling_mesh.ply"),
                                                   recalc_normals=False, clean=False)
                print(f"Number of sampling vertices: {len(sampl_mesh.vertices)} !")
                sphere_fit_params = datahandler.load_array("sphere_fit", folderpath=resdata_dir_seg)[0, :]
                sphere_x0, sphere_y0, sphere_z0, sphere_radius = sphere_fit_params
                print(f"sphere_x0 = {sphere_x0} {img_unit}")
                print(f"sphere_y0 = {sphere_y0} {img_unit}")
                print(f"sphere_z0 = {sphere_z0} {img_unit}")
                print(f"sphere_radius = {sphere_radius} {img_unit}")
                min_all = np.arange(min_proj_dist, max_proj_dist, proj_thickness)
                max_all = min_all + proj_thickness
                mask = max_all <= max_proj_dist
                min_all, max_all = min_all[mask], max_all[mask]
                proj_tasks = np.stack([
                    min_all,
                    max_all,
                    np.full(min_all.shape, num_proj_samples)
                ], axis=1)

                proj_radii = sphere_radius - (min_all + 0.5 * proj_thickness)
                print(f">> Projecting at radii {proj_radii} {img_unit} ...")

                # Flip direction of projection for going "inwards"
                if flip_proj_direction:
                    proj_tasks[:, :2] *= -1

                all_projections = np.empty((len(proj_tasks), len(sampl_mesh.vertices)))
                for i in range(len(proj_tasks)):
                    dist_min, dist_max, dist_num = proj_tasks[i]
                    all_projections[i] = analysis.proj2mesh(img=img_raw, mesh=sampl_mesh, unit=img_unit,
                                                            min_dist_per_vert=None,
                                                            scale=img_scale, min_dist=dist_min, max_dist=dist_max,
                                                            num_dist=dist_num, mode=proj_mode, show_proj=False,
                                                            savefig="", normalise=False)
                print("=======")
                print(f"Projected {len(all_projections)} layers !")

                proj_xcords, proj_ycords = analysis.mercator_project(pts=sampl_mesh.vertices,
                                                                     ref_point=[sphere_x0, sphere_y0, sphere_z0],
                                                                     rotate=rotate_proj_angles)
                radial_projection = analysis.create_radial_stack(values=all_projections,
                                                                 phi_coords=proj_xcords,
                                                                 theta_cords=proj_ycords,
                                                                 grid_n=np.mean(img_dim[1:]),
                                                                 projection_radii=proj_radii)
                radial_stack, stack_cords = radial_projection
                np.savez_compressed(os.path.join(resdata_dir, f"radial_projection_{ct_label}.npz"),
                                    radial_stack=radial_stack,
                                    stack_cords=stack_cords)

                mid_radial_stack_i = radial_stack.shape[0] // 2
                visuals.plot_matrix(radial_stack[mid_radial_stack_i], figsize=(14, 8), origin="upper",
                                    title=f"R = {proj_radii[mid_radial_stack_i]} {img_unit}",
                                    savefig=os.path.join(resfig_dir, f"radial-stack_{mid_radial_stack_i}.png"),
                                    unit="px", colorbar=True, cmap="inferno", hidefig=hide_fig_output)
                # datahandler.save_tiff(radial_stack, filepath=os.path.join(resfig_dir, f"radial-stack_{ct_label}.tiff"))
        print("======== END PROJECTION ========")

    if savetiff:
        print("======== START SAVING FINAL TIFF ========")
        file_list = sorted(
            glob.glob(os.path.join(result_path, f"{tifname_split[0]}", "**", "radial_projection_t-*_c-*.npz"),
                      recursive=True)
        )
        pattern = re.compile(r't-(\d+)_c-(\d+)\.npz')
        metadata = [pattern.search(f).groups() for f in file_list]
        metadata = [(int(t), int(c)) for t, c in metadata]
        times = sorted(set(t for t, c in metadata))
        channels = sorted(set(c for t, c in metadata))
        T, C = len(times), len(channels)
        sample = np.load(file_list[0])
        Z, Y, X = sample["radial_stack"].shape
        hyperstack = np.zeros((T, Z, C, Y, X), dtype=np.float32)
        coordstack = np.zeros((T, Z, 3, Y, X), dtype=np.float32)
        for f, (t, c) in zip(file_list, metadata):
            radial_projection_load = np.load(f)
            hyperstack[times.index(t), :, channels.index(c)] = radial_projection_load["radial_stack"]
            coordstack[times.index(t)] = np.moveaxis(radial_projection_load["stack_cords"], -1, 1)

        hyperstack_uint8 = np.zeros_like(hyperstack, dtype=np.uint8)
        for t in range(T):
            for z in range(Z):
                for c in range(C):
                    slice_ = hyperstack[t, z, c]
                    norm = (slice_ - slice_.min()) / (slice_.ptp() + 1e-8)
                    hyperstack_uint8[t, z, c] = (norm * 255).astype(np.uint8)
        hyperstack_save_path = os.path.join(result_path, f"{tifname_split[0]}", f"radial_hyperstack.tiff")
        coordstack_save_path = os.path.join(result_path, f"{tifname_split[0]}", f"coordinate_metadata.tiff")
        tifffile.imwrite(
            hyperstack_save_path,
            hyperstack_uint8,
            imagej=True,
            metadata={"axes": "TZCYX"},
        )
        print(f"Saved tiff to {hyperstack_save_path} !")
        tifffile.imwrite(
            coordstack_save_path,
            coordstack,
            imagej=True,
            metadata={"axes": "TZCYX"},
        )
        print(f"Saved tiff to {coordstack_save_path} !")
        print("======== END SAVING FINAL TIFF ========")

    print("======== END script ========")
