"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Load image
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module_scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import load_img_virtual
from module_scripts.visuals import plot_img, view_img
from module_scripts.datahandler import create_resdirs


def main(img_path, t_select, c_select, img_slice_colour="Greens_r", img_maxproj_colour="Greens",
         render_colour="green", show_figures=True, render=False):
    print(f">> Attempting to quickly visualise image {img_path}!")
    if not os.path.exists(img_path):
        print(f">> Image {img_path} does not exist!")
        return None

    # ==== Load Image ====
    print(f"Selected image path: {img_path}")
    hidefig = not show_figures
    img_load = load_img_virtual(path=img_path, t_sel_idx=t_select, c_sel_idx=c_select)
    if img_load is None:
        return None
    img_raw, img_dim, img_scale, img_unit = img_load

    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Plot Image Slices and Max Projections ====
    z_i, y_i, x_i = int(img_dim[0] // 2), int(img_dim[1] // 2), int(img_dim[2] // 2)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, x_i=x_i, y_i=y_i, z_i=z_i,
             savefig=os.path.join(resfig_dir, "sliced_raw.pdf"), cmap=img_slice_colour, hidefig=hidefig)
    plot_img(img=img_raw, scale=img_scale, unit=img_unit, max_proj=True, cmap=img_maxproj_colour,
             savefig=os.path.join(resfig_dir, "sliced_raw_maxproj.pdf"), hidefig=hidefig)
    if render:
        view_img(img_list=[img_raw], color_list=[render_colour], title_list=["Raw Image"], scale=img_scale)
    return img_load
