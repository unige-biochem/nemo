"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Preprocess image for meshing
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import load_img_virtual, gaussian_blur, yen_thresh, thresh_img, rescale_val_xyz
from module_scripts.datahandler import create_resdirs, save_array, save_tiff
from module_scripts.visuals import plot_img, view_img

# Import Python essentials
import numpy as np


def main(img_path, t_select, c_select, img_blur_val, img_thresh_val, show_figures=True, render=False):
    print(f">> Attempting to preprocess image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Load image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    img_load = load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load

    # ==== Create folder structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Blur image ====
    sigma_ = rescale_val_xyz(val=img_blur_val, scale=img_scale)
    img_blur = gaussian_blur(img=img_raw, sigma=sigma_, renorm=False)

    # ==== Plot image slices ====
    plot_img(img=img_blur, scale=img_scale, unit=img_unit, cmap="inferno",
             savefig=os.path.join(resfig_dir, "sliced_blurred.pdf"), hidefig=hidefig)
    plot_img(img=img_blur, scale=img_scale, unit=img_unit, cmap="inferno_r",
             savefig=os.path.join(resfig_dir, "sliced_blurred_maxproj.pdf"), hidefig=hidefig, max_proj=True)

    # ==== Yen threshold image if no custom value is given ====
    if img_thresh_val is None:
        img_thresh_val = np.min(
            [yen_thresh(img_blur[img_dim[0] // 2, :, :]),
             yen_thresh(img_blur[:, img_dim[1] // 2, :]),
             yen_thresh(img_blur[:, :, img_dim[2] // 2])])

    # ==== Binarise image using threshold ====
    img_thresh = thresh_img(img_blur, img_thresh_val)

    # ==== Plot image slices ====
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, savefig=os.path.join(resfig_dir, "sliced_thresh.pdf"),
             cmap="inferno", thresh_mask=img_thresh, hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit,
             savefig=os.path.join(resfig_dir, "sliced_thresh_maxproj.pdf"),
             cmap="inferno_r", thresh_mask=img_thresh, hidefig=hidefig, max_proj=True)
    save_tiff(img_blur, filepath=os.path.join(resfig_dir, "img_blurred.tiff"), img_unit=img_unit, img_scale=img_scale)
    save_tiff(img_thresh, filepath=os.path.join(resfig_dir, "img_thresholded.tiff"), img_unit=img_unit,
              img_scale=img_scale)
    save_array(np.column_stack(([img_blur_val], [img_thresh_val])), name="blur_thresh_parameters",
               header="blur,thresh", folderpath=resdata_dir)
    # ==== 3D Render Images ====
    if render:
        view_img(img_list=[img_raw, img_blur, img_thresh], scale=img_scale,
                 title_list=["Raw Image", "Blurred Image", "Thresholded Image"],
                 color_list=["Greens_r", "inferno", "Blues_r"], opacity_list=[1.0, 0.7, 0.7])
    return img_blur, img_thresh
