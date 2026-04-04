"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Import and visualise 2d segmentation of z slices
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    load_img_virtual,
    z_range_2d_max,
    contour_masks,
    coord_search_radius,
    coord_search_neighbours,
    avg_2d_nem_tens
)
from module_scripts.datahandler import (
    create_resdirs,
    create_dir,
    load_array,
    load_png,
    save_video_multiple
)
from module_scripts.visuals import (
    plot_matrix,
    plot_img_2d_masks,
    plot_polar_hist,
    plot_hist,
    plot_matrix_vectors,
    plot_slice_heatmap
)
# Import python essentials
import numpy as np


def main(img_path, t_select, c_select):
    print(f">> Attempting to import 2d segmentation for image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None
    # ==== Choose Image ====
    print(f"Selected image path: {img_path}")
    # ==== Load Image ====
    img_load = load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Import, Visualise and Analysie 2D segmentation ====
    patch_avg = ["radius", 20]
    # patch_avg = ["nearest", 6]
    patch_type = patch_avg[0]
    patch_size = patch_avg[1]
    if not patch_type in ["radius", "nearest"]:
        print(f"[!] Unknown patch type: {patch_type}")
        return None

    crop_max_ar = 8.0  # Max for colorbar !
    local_s_weighted_max = 3.0  # Max for colorbar !
    xy_pixel_scale = np.mean(img_scale[1:])
    file_ending = "png"

    z_range_2d_analysis = np.arange(img_dim[0])

    hide_all_figs = True
    dpi_all_figs = 100
    figsize = (22, 10)
    resfig_dir_2dsliced = os.path.join(resfig_dir, "2d-sliced_analysis")
    resdata_dir_2dsliced = os.path.join(resdata_dir, "2d-sliced_analysis")
    create_dir(resfig_dir_2dsliced)
    create_dir(resdata_dir_2dsliced)
    img_name = os.path.basename(img_path).split(".tif")[0]
    directors_2d_all = []
    aspect_ratio_all = []
    valid_2dseg_idxs = []
    masks = mask_outlines = None
    only_load_ellip = False

    for z_sel in z_range_2d_analysis:
        print(f">> Z-slice selected: {z_sel} / {z_range_2d_max()}")

        # ==== Load Slice Segmentation Results ====
        # seg_ellips_results = load_array(name=f"{z_sel + 1:04d}_seg-ellipses", folderpath=resdata_dir_2dsliced,
        #                                             return_df=True)
        # seg_ellips_results = load_array(name=f"{img_name}_z{z_sel + 1}_cp_masks_Ellipsoids",
        #                                             folderpath=resdata_dir_2dsliced,
        #                                             return_df=True)

        seg_ellips_results = load_array(name=f"{img_name}_z{z_sel + 1}_cp_masks_ellipses",
                                        folderpath=os.path.join(os.path.dirname(img_path),
                                                                "z_slice_segmentation", img_name),
                                        return_df=True)
        # ==== Process (or skip empty) Slice Segmentations ====
        if seg_ellips_results is not None and len(seg_ellips_results) > 1:
            seg_ellips_results["Ellipse.Orientation"] *= -1
            seg_ellips_results["Ellipse.Radius1"] *= xy_pixel_scale
            seg_ellips_results["Ellipse.Radius2"] *= xy_pixel_scale
            seg_ellips_results["aspect_ratio"] = seg_ellips_results["Ellipse.Radius1"] / seg_ellips_results[
                "Ellipse.Radius2"]
            valid_2dseg_idxs.append(z_sel)

            # ==== [!!] ONLY DEBUG [!] ====
            # seg_ellips_results["Ellipse.Orientation"] = np.ones(len(seg_ellips_results["Ellipse.Orientation"])) * 0  # !!!!!!
            # seg_ellips_results["Ellipse.Orientation"] = np.random.uniform(-180, 180, size=len(seg_ellips_results["Ellipse.Orientation"]))
            # seg_ellips_results["aspect_ratio"] = np.ones(len(seg_ellips_results["aspect_ratio"])) * 2
            seg_ellips_results = seg_ellips_results[seg_ellips_results["aspect_ratio"] < crop_max_ar]
            if len(seg_ellips_results) <= 1:
                print(f"No segmentation found for {z_sel}, continuing to next z slice...")
                continue
            # ==== [!!] ONLY DEBUG [!] ====

            if not only_load_ellip:
                # ==== Load Segmentation Masks ====
                masks = load_png(os.path.join(os.path.dirname(img_path), "z_slice_segmentation", img_name,
                                              f"{img_name}_z{z_sel + 1}_cp_masks.pdf"))

                if masks is not None:
                    mask_outlines = contour_masks(masks)
                else:
                    print(f"No masks found for {z_sel}, continuing to next z slice...")
                    continue
        else:
            print(f"No segmentation found for {z_sel}, continuing to next z slice...")
            continue

        # ==== Saving Path Initialisation ====
        resfig_dir_2dsliced = os.path.join(resfig_dir, "2d-sliced_analysis")
        resdata_dir_2dsliced = os.path.join(resdata_dir, "2d-sliced_analysis")
        create_dir(resfig_dir_2dsliced)
        create_dir(resdata_dir_2dsliced)

        # ==== Convert Orientations to Director Field ====
        seg_directors_2d = np.column_stack((seg_ellips_results["Ellipse.Center.X"],
                                            seg_ellips_results["Ellipse.Center.Y"],
                                            np.cos(np.radians(seg_ellips_results["Ellipse.Orientation"])),
                                            np.sin(np.radians(seg_ellips_results["Ellipse.Orientation"]))))
        print("Max aspect ratio =", np.max(seg_ellips_results["aspect_ratio"]))
        seg_directors_2d[:, :2] *= img_scale[1:]
        directors_2d = seg_directors_2d.copy()
        directors_2d_all.append(directors_2d)
        aspect_ratio_all.append(np.array(seg_ellips_results["aspect_ratio"]))

        if not only_load_ellip:
            # ==== Local (Weighted) Nematic Order ====
            seg_ellips_results["shape_scalar"] = seg_ellips_results["aspect_ratio"] - 1

            if patch_type == "radius":
                local_idxs = coord_search_radius(directors_2d[:, :2], r=patch_size)
                patch_label = f"r-{patch_size}{img_unit}"
                # title_hist = f"Avg over {patch_size}{img_unit}: Order Scalar $S$"
                # title_render = f"Avg over {patch_size}{img_unit}: Average Directors"
            elif patch_type == "nearest":
                if len(directors_2d) <= patch_size:
                    print(f"Only {len(directors_2d)} directors, using k=2 ...")
                    patch_size = 2
                local_idxs = coord_search_neighbours(directors_2d[:, :2], k=patch_size, n_process=8)
                patch_label = f"k-{patch_size - 1}"
                # title_hist = f"Avg over {patch_size - 1} neighbours: Order Scalar $S$"
                # title_render = f"Avg over {patch_size - 1} neighbours: Average Directors"
            else:
                local_idxs = patch_label = None
                # title_hist = title_render = None
                print(f"[!] Unknown patch type: {patch_type}")

            s_2d_seg_local, n_2d_seg_local = avg_2d_nem_tens(directors_2d, weights=None,
                                                             neigh_idxs=local_idxs)
            s_2d_seg_local_weighted, n_2d_seg_local = avg_2d_nem_tens(directors_2d, local_idxs,
                                                                      weights=np.array(
                                                                          seg_ellips_results["shape_scalar"]))
            print(f"Max weighted order S: {s_2d_seg_local_weighted.max()}")
            patch_size = patch_avg[1]

            # ==== Plot Results ====
            z_sel_label = f"| Z = {z_sel} / {img_dim[0] - 1}"
            plot_matrix(matrix=img_raw[z_sel],
                        scale=img_scale, unit=img_unit, figsize=figsize,
                        cmap_label="Intensity Signal (a.u.)",
                        title=f"Raw Image {z_sel_label}", cmap="gray", colorbar=True,
                        savefig=os.path.join(resfig_dir_2dsliced, f"z-{z_sel}_raw.{file_ending}"),
                        hidefig=hide_all_figs, dpi=dpi_all_figs)
            plot_img_2d_masks(matrix=img_raw[z_sel], unit=img_unit, scale=img_scale,
                              segmentation_mask=(masks, mask_outlines),
                              figsize=figsize, cmap="tab20", alpha=0.8,
                              title=f"Segmentation Masks {z_sel_label}",
                              savefig=os.path.join(resfig_dir_2dsliced,
                                                   f"z-{z_sel}_seg_raw_masks.{file_ending}"),
                              hidefig=hide_all_figs, dpi=dpi_all_figs)
            plot_polar_hist(seg_ellips_results["Ellipse.Orientation"],
                            savefig=os.path.join(resfig_dir_2dsliced,
                                                 f"z-{z_sel}_seg_thetas_polar-hist.{file_ending}"),
                            hidefig=hide_all_figs, dpi=dpi_all_figs, title=f"Orientation Angles {z_sel_label}")

            long_axis_mean = np.mean(seg_ellips_results["Ellipse.Radius1"])
            plot_hist(seg_ellips_results["Ellipse.Radius1"],
                      title=fr"Long Axis with mean$\approx${long_axis_mean:.2f} {img_unit} {z_sel_label}",
                      savefig=os.path.join(resfig_dir_2dsliced, f"z-{z_sel}_seg_long-axis_hist.{file_ending}"),
                      hidefig=hide_all_figs, dpi=dpi_all_figs)
            short_axis_mean = np.mean(seg_ellips_results["Ellipse.Radius2"])
            plot_hist(seg_ellips_results["Ellipse.Radius2"],
                      title=fr"Short Axis with mean$\approx${short_axis_mean:.2f} {img_unit} {z_sel_label}",
                      savefig=os.path.join(resfig_dir_2dsliced, f"z-{z_sel}_seg_short-axis_hist.{file_ending}"),
                      hidefig=hide_all_figs, dpi=dpi_all_figs)
            if crop_max_ar is None:
                ar_plot_range = None
            else:
                ar_plot_range = [0, crop_max_ar]

            plot_hist(seg_ellips_results["aspect_ratio"], title=f"Aspect Ratio {z_sel_label}",
                      savefig=os.path.join(resfig_dir_2dsliced, f"z-{z_sel}_seg_ar_hist.{file_ending}"),
                      hidefig=hide_all_figs, xlim=ar_plot_range, dpi=dpi_all_figs)
            plot_matrix_vectors(x=seg_ellips_results["Ellipse.Center.X"],
                                y=seg_ellips_results["Ellipse.Center.Y"],
                                angle_field=np.radians(seg_ellips_results["Ellipse.Orientation"]),
                                matrix=img_raw[z_sel],
                                veclength=15 * np.array(seg_ellips_results["aspect_ratio"]),
                                scale=img_scale, unit=img_unit, figsize=figsize,
                                vec_colors=seg_ellips_results["aspect_ratio"],
                                cbar_matrix_label="Intensity Signal (a.u.)",
                                cbar_vector_label="Aspect Ratio",
                                title=f"Segmented Ellipse Long Axes {z_sel_label}",
                                savefig=os.path.join(resfig_dir_2dsliced,
                                                     f"z-{z_sel}_seg_raw_ar.{file_ending}"),
                                hidefig=hide_all_figs, vec_cmap_limits=ar_plot_range, dpi=dpi_all_figs)
            # NORMALISED
            plot_matrix_vectors(x=seg_ellips_results["Ellipse.Center.X"],
                                y=seg_ellips_results["Ellipse.Center.Y"],
                                angle_field=np.radians(seg_ellips_results["Ellipse.Orientation"]),
                                matrix=img_raw[z_sel],
                                veclength=15 * np.array(seg_ellips_results["aspect_ratio"]),
                                scale=img_scale, unit=img_unit, figsize=figsize, vec_colors=s_2d_seg_local,
                                vec_cmap_limits=[0, 1], cbar_matrix_label="Intensity Signal (a.u.)",
                                cbar_vector_label="Local Nematic Order $S$",
                                title=f"Local Nematic Order {patch_label} {z_sel_label}",
                                savefig=os.path.join(resfig_dir_2dsliced,
                                                     f"z-{z_sel}_seg_raw_local_S_{patch_label}.{file_ending}"),
                                hidefig=hide_all_figs, dpi=dpi_all_figs)
            if len(seg_ellips_results) > 4:
                plot_slice_heatmap(coords=directors_2d[:, :2], values=s_2d_seg_local, pt_size=5,
                                   cmap_label="Local Nematic Order",
                                   title=f"Local Nematic Order {patch_label} {z_sel_label}",
                                   img_dim=img_dim, img_scale=img_scale, manual_vminvmax=[0, 1],
                                   savefig=os.path.join(resfig_dir_2dsliced,
                                                        f"z-{z_sel}_seg_local_S_heatmap_{patch_label}.{file_ending}"),
                                   hidefig=hide_all_figs, dpi=dpi_all_figs)
            plot_hist(s_2d_seg_local, title=f"Local Nematic Order $S$ {patch_label} {z_sel_label}",
                      savefig=os.path.join(resfig_dir_2dsliced,
                                           f"z-{z_sel}_seg_local_S_hist_{patch_label}.{file_ending}"),
                      hidefig=hide_all_figs, xlim=[0, 1], dpi=dpi_all_figs)
            # WEIGHTED
            plot_matrix_vectors(x=seg_ellips_results["Ellipse.Center.X"],
                                y=seg_ellips_results["Ellipse.Center.Y"],
                                angle_field=np.radians(seg_ellips_results["Ellipse.Orientation"]),
                                matrix=img_raw[z_sel],
                                veclength=15 * np.array(seg_ellips_results["aspect_ratio"]),
                                scale=img_scale, unit=img_unit, figsize=figsize,
                                vec_colors=s_2d_seg_local_weighted,
                                cbar_matrix_label="Intensity Signal (a.u.)",
                                vec_cmap_limits=[0, local_s_weighted_max],
                                cbar_vector_label="Local Weighted Nematic Order $S$",
                                title=f"Local Weighted Nematic Order {patch_label} {z_sel_label}",
                                savefig=os.path.join(resfig_dir_2dsliced,
                                                     f"z-{z_sel}_seg_raw_local_S_weighted_{patch_label}.{file_ending}"),
                                hidefig=hide_all_figs, dpi=dpi_all_figs)
            if len(seg_ellips_results) > 4:
                plot_slice_heatmap(coords=directors_2d[:, :2], values=s_2d_seg_local_weighted,
                                   cmap_label="Local Weighted Nematic Order", pt_size=5,
                                   title=f"Local Weighted Nematic Order {patch_label} {z_sel_label}",
                                   img_dim=img_dim, img_scale=img_scale,
                                   manual_vminvmax=[0, local_s_weighted_max],
                                   savefig=os.path.join(resfig_dir_2dsliced,
                                                        f"z-{z_sel}_seg_local_S_weighted_heatmap_{patch_label}.{file_ending}"),
                                   hidefig=hide_all_figs, dpi=dpi_all_figs)
            plot_hist(s_2d_seg_local_weighted,
                      title=f"Local Weighted Nematic Order $S$ {patch_label} {z_sel_label}",
                      savefig=os.path.join(resfig_dir_2dsliced,
                                           f"z-{z_sel}_seg_local_S_weighted_hist_{patch_label}.{file_ending}"),
                      hidefig=hide_all_figs, xlim=[0, local_s_weighted_max], dpi=dpi_all_figs)
    save_video_multiple(folderpath=os.path.join(resfig_dir, "2d-sliced_analysis"), ext="mp4")
    return None
