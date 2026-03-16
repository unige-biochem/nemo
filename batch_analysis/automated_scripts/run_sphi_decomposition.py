"""
Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Purpose: Decompose obtained tangential nematic field on s-phi basis
Requires Python 3.9.6 and packages from requirements.txt.
Author: Konstantinos Andreadis (Roux Lab & Salbreux Lab @UNIGE)
"""

# Import NEMO scripts
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts import analysis, extra_gastruloids

# Import python essentials
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="", required=True, type=str, help="Path to the input TIFF image")
    parser.add_argument("--show_figures", action=argparse.BooleanOptionalAction, help="Show figures")
    parser.add_argument("--layer_label", default="", type=str, help="Name of layer to be analysed")
    parser.add_argument("--q_decomp_radius", default=40.0, type=float, help="Q decomposition radius")
    parser.add_argument("--low_cutoff_phi", default=-180, type=float, help="Low cutoff phi")
    parser.add_argument("--high_cutoff_phi", default=180, type=float, help="High cutoff phi")
    parser.add_argument("--profile_bins", default=20, type=int, help="Number of bins for profile")
    return parser.parse_args()


def main(img_path, layer_label, q_decomp_radius, low_cutoff_phi, high_cutoff_phi, profile_bins, show_figures=False):
    print(f">> Attempting to perform decomposition analysis of gastruloid {img_path} for layer {layer_label}!")
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

    # ==== Run Decomposition ====
    extra_gastruloids.proj_nem_on_sphi(img_path=img_path, layer_label=layer_label, q_decomp_radius=q_decomp_radius,
                                       low_cutoff_phi=low_cutoff_phi, high_cutoff_phi=high_cutoff_phi,
                                       profile_bins=profile_bins, hidefig=hidefig, img_unit=img_unit)
    return None


if __name__ == "__main__":
    print("======== START NEMO ========")
    main(img_path=args.img_path, show_figures=args.show_figures, layer_label=args.layer_label,
         q_decomp_radius=args.q_decomp_radius, low_cutoff_phi=args.low_cutoff_phi, high_cutoff_phi=args.high_cutoff_phi,
         profile_bins=args.profile_bins, )
    print("======== END NEMO ========")
