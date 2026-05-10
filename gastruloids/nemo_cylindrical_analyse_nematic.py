"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Decompose obtained tangential nematic field on s-phi basis
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO module scripts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import load_img_scaling, load_img_unit
from gastruloids.extra_gastruloids import proj_nem_on_sphi


def main(img_path, t_select, c_select, layer_label, q_decomp_radius, low_cutoff_phi, high_cutoff_phi, profile_bins,
         show_figures=False):
    print(f">> Attempting to perform decomposition analysis of gastruloid {img_path} for layer {layer_label}!")
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

    # ==== Run Decomposition ====
    proj_nem_on_sphi(img_path=img_path, t_select=t_select, c_select=c_select, layer_label=layer_label,
                     q_decomp_radius=q_decomp_radius,
                     low_cutoff_phi=low_cutoff_phi, high_cutoff_phi=high_cutoff_phi,
                     profile_bins=profile_bins, hidefig=hidefig, img_unit=img_unit)
    return None
