"""
Visualisation Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################

import os.path
import shutil

import imageio
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib import colors as pltcolors
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

#############################################
# [!!!] Dark/Light Mode for all Plots [!!!] #
#############################################
plt.style.use('default')  # LIGHT, for DARK use 'dark_background'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}"
})


#################
# BASIC MODULES #
#################
def normalise_range(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


def rgb2hex_fast(colors):
    colors = (colors[:, :3] * 255).astype(np.uint32)
    hex_ints = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    hex_strings = np.char.zfill(np.char.mod('%06x', hex_ints), 6)
    return np.char.add('#', hex_strings)


def color_scalar(scalar, cmap="Spectral", normalise=False, manual_vminmax=None):
    if manual_vminmax is not None:
        norm = plt.Normalize(vmin=manual_vminmax[0], vmax=manual_vminmax[1])
        scalar = norm(scalar)
    if normalise:
        scalar = normalise_range(scalar)
    return rgb2hex_fast(colormaps[cmap](scalar))


def create_figdir(directory):
    if not os.path.exists(directory):
        print(f">> Creating directory {directory} ...")
        os.makedirs(directory)


def sharp_blue_red(min_val, intermediate_val, max_val):
    color_below_mid = (0, 1, 1)
    color_above_mid = (1, 0, 0)
    colors = []
    for value in np.linspace(min_val, max_val, 256):
        if value < intermediate_val:
            colors.append(color_below_mid)
        else:
            colors.append(color_above_mid)
    return ListedColormap(colors)


def colour_dist(distances, middle_val, radial_points, mesh, radial_intensities):
    print(">> Colouring based on distance...")
    norm_distances = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    cmap = sharp_blue_red(min_val=np.min(distances), intermediate_val=middle_val,
                          max_val=np.max(distances))
    intensities_flat = normalise_range(radial_intensities.ravel())
    dist_color = cmap(norm_distances(np.tile(distances, radial_points.shape[0])))
    dist_color[:, :3] = dist_color[:, :3] * intensities_flat[:, None]
    dist_color[:, 3] = np.ones_like(intensities_flat) * 0.5
    dist_color_reshaped = dist_color.reshape(len(mesh.vertices), len(distances), 4)
    dist_color = np.max(dist_color_reshaped, axis=1)
    return dist_color, cmap


#######################
# 2D PLOTTING MODULES #
#######################

def plot_img(img, scale, unit, x_i=None, y_i=None, z_i=None, figsize=(20, 3), slice_line_alpha=0.8,
             cmap="Greens_r", dpi=200, thresh_mask=None, max_proj=False, meshes=None, mesh_normal_alpha=0.8,
             slice_depth=10, mesh_thick=0.1, mesh_alpha=0.3, thresh_alpha=0.8, mesh_colors=None,
             show_mesh_normals=False, normal_scale=0.03, normal_vecfreq=20,
             cmap_label="Fluorescence Intensity (a.u.)", title_digit_precision=2,
             manual_vminvmax=None, savefig="", hidefig=False):
    print(">> Plotting slices...")
    if z_i is None:
        z_i = img.shape[0] // 2
    if y_i is None:
        y_i = img.shape[1] // 2
    if x_i is None:
        x_i = img.shape[2] // 2

    if max_proj:
        z_title = "Z Max Projection"
        y_title = "Y Max Projection "
        x_title = "X Max Projection"
        z_slice = np.max(img, axis=0)
        y_slice = np.max(img, axis=1)
        x_slice = np.max(img, axis=2)
    else:
        z_title = f'Z={np.round(z_i * scale[0], decimals=title_digit_precision)} {unit} Slice'
        y_title = f'Y={np.round(y_i * scale[1], decimals=title_digit_precision)} {unit} Slice'
        x_title = f'X={np.round(x_i * scale[2], decimals=title_digit_precision)} {unit} Slice'
        z_slice = img[z_i, :, :]
        y_slice = img[:, y_i, :]
        x_slice = img[:, :, x_i]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if manual_vminvmax is None:
        vmin, vmax = np.array([z_slice.min(), y_slice.min(), x_slice.min()]).min(), np.array(
            [z_slice.max(), y_slice.max(),
             x_slice.max()]).max()
    else:
        vmin, vmax = manual_vminvmax
    axes[0].imshow(z_slice, cmap=cmap, aspect=scale[1] / scale[2], origin='upper', vmin=vmin, vmax=vmax)
    axes[0].set_title(z_title)
    axes[1].imshow(y_slice, cmap=cmap, aspect=scale[0] / scale[2], origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(y_title)
    axes[2].imshow(x_slice, cmap=cmap, aspect=scale[0] / scale[1], origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title(x_title)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                        ax=axes, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label(cmap_label)

    if thresh_mask is not None:
        axes[0].imshow(thresh_mask[z_i, :, :] > 0, cmap="Reds", aspect=scale[1] / scale[2], origin='upper',
                       alpha=thresh_alpha)
        axes[1].imshow(thresh_mask[:, y_i, :] > 0, cmap="Reds", aspect=scale[0] / scale[2], origin='lower',
                       alpha=thresh_alpha)
        axes[2].imshow(thresh_mask[:, :, x_i] > 0, cmap="Reds", aspect=scale[0] / scale[1], origin='lower',
                       alpha=thresh_alpha)
    if meshes is not None:
        if mesh_colors is None:
            mesh_colors = ["red" for _ in range(len(meshes))]
        for i, mesh in enumerate(meshes):
            pts = mesh.vertices.copy()
            pts_z, pts_y, pts_x = pts[:, 0], pts[:, 1], pts[:, 2]
            zmask = abs(pts_z - z_i * scale[0]) < slice_depth
            ymask = abs(pts_y - y_i * scale[1]) < slice_depth
            xmask = abs(pts_x - x_i * scale[2]) < slice_depth

            axes[0].scatter(pts_x[zmask] / scale[2], pts_y[zmask] / scale[1], c=mesh_colors[i], s=mesh_thick,
                            alpha=mesh_alpha)
            axes[1].scatter(pts_x[ymask] / scale[2], pts_z[ymask] / scale[0], c=mesh_colors[i], s=mesh_thick,
                            alpha=mesh_alpha)
            axes[2].scatter(pts_y[xmask] / scale[1], pts_z[xmask] / scale[0], c=mesh_colors[i], s=mesh_thick,
                            alpha=mesh_alpha)
            if show_mesh_normals:
                norms = mesh.vertex_normals.copy()
                norms_z, norms_y, norms_x = norms[:, 0], norms[:, 1], norms[:, 2]

                # Z-Slice
                axes[0].quiver((pts_x[zmask] / scale[2])[::normal_vecfreq],
                               (pts_y[zmask] / scale[1])[::normal_vecfreq],
                               (norms_x[zmask] / scale[2])[::normal_vecfreq],
                               (norms_y[zmask] / scale[1])[::normal_vecfreq],
                               color=mesh_colors[i], alpha=mesh_normal_alpha, scale=normal_scale,
                               angles='xy', scale_units='xy')

                # Y-Slice
                axes[1].quiver((pts_x[ymask] / scale[2])[::normal_vecfreq],
                               (pts_z[ymask] / scale[0])[::normal_vecfreq],
                               (norms_x[ymask] / scale[2])[::normal_vecfreq],
                               (norms_z[ymask] / scale[0])[::normal_vecfreq],
                               color=mesh_colors[i], alpha=mesh_normal_alpha, scale=normal_scale,
                               angles='xy', scale_units='xy')

                # X-Slice
                axes[2].quiver((pts_y[xmask] / scale[1])[::normal_vecfreq],
                               (pts_z[xmask] / scale[0])[::normal_vecfreq],
                               (norms_y[xmask] / scale[1])[::normal_vecfreq],
                               (norms_z[xmask] / scale[0])[::normal_vecfreq],
                               color=mesh_colors[i], alpha=mesh_normal_alpha, scale=normal_scale,
                               angles='xy', scale_units='xy')

    if not max_proj:
        axes[0].axhline(y=y_i, linestyle="--", alpha=slice_line_alpha)
        axes[0].axvline(x=x_i, linestyle="--", alpha=slice_line_alpha)
        axes[1].axhline(y=z_i, linestyle="--", alpha=slice_line_alpha)
        axes[1].axvline(x=x_i, linestyle="--", alpha=slice_line_alpha)
        axes[2].axhline(y=z_i, linestyle="--", alpha=slice_line_alpha)
        axes[2].axvline(x=y_i, linestyle="--", alpha=slice_line_alpha)

    axes[0].set_xlabel(f'X ({unit})')
    axes[0].set_xticks(np.linspace(0, img.shape[2], 5))
    axes[0].set_xticklabels(np.round(axes[0].get_xticks() * scale[2], 2))
    axes[0].set_ylabel(f'Y ({unit})')
    axes[0].set_yticks(np.linspace(0, img.shape[1], 5))
    axes[0].set_yticklabels(np.round(axes[0].get_yticks() * scale[1], 2))

    axes[1].set_xlabel(f'X ({unit})')
    axes[1].set_xticks(np.linspace(0, img.shape[2], 5))
    axes[1].set_xticklabels(np.round(axes[1].get_xticks() * scale[2], 2))
    axes[1].set_ylabel(f'Z ({unit})')
    axes[1].set_yticks(np.linspace(0, img.shape[0], 5))
    axes[1].set_yticklabels(np.round(axes[1].get_yticks() * scale[0], 2))

    axes[2].set_xlabel(f'Y ({unit})')
    axes[2].set_xticks(np.linspace(0, img.shape[1], 5))
    axes[2].set_xticklabels(np.round(axes[2].get_xticks() * scale[1], 2))
    axes[2].set_ylabel(f'Z ({unit})')
    axes[2].set_yticks(np.linspace(0, img.shape[0], 5))
    axes[2].set_yticklabels(np.round(axes[2].get_yticks() * scale[0], 2))

    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_maxproj_pts(verts, unit, cmap="Spectral", colors=None, hexsize=600, figsize=(15, 8), savefig="", dpi=200,
                     cmap_label="Signal (a.u.)", manual_vminmax=None, hidefig=False):
    print(">> Plotting max projection of coloured points...")
    if colors is None:
        print(f"No colors given, assuming uniform color!")
        colors = np.zeros_like(verts)
    if manual_vminmax is None:
        vmin, vmax = colors.min(), colors.max()
    else:
        vmin, vmax = manual_vminmax
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    panels = [
        (0, 0, verts[:, 0] >= verts[:, 0].max() / 2, verts[:, 2], verts[:, 1], 'X', 'Y'),
        (0, 1, verts[:, 1] >= verts[:, 1].max() / 2, verts[:, 2], verts[:, 0], 'X', 'Z'),
        (0, 2, verts[:, 2] >= verts[:, 2].max() / 2, verts[:, 1], verts[:, 0], 'Y', 'Z'),
        (1, 0, verts[:, 0] < verts[:, 0].max() / 2, verts[:, 2], verts[:, 1], 'X', 'Y'),
        (1, 1, verts[:, 1] < verts[:, 1].max() / 2, verts[:, 2], verts[:, 0], 'X', 'Z'),
        (1, 2, verts[:, 2] < verts[:, 2].max() / 2, verts[:, 1], verts[:, 0], 'Y', 'Z'),
    ]
    for i, j, mask, x, y, xlabel, ylabel in panels:
        ax = axes[i, j]
        if np.any(mask):
            ax.hexbin(x[mask], y[mask], C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel(f'{xlabel} ({unit})')
            ax.set_ylabel(f'{ylabel} ({unit})')
            ax.set_xticks(np.linspace(x[mask].min(), x[mask].max(), 3))
            ax.set_yticks(np.linspace(y[mask].min(), y[mask].max(), 3))
            ax.set_xticklabels(np.round(ax.get_xticks(), 2))
            ax.set_yticklabels(np.round(ax.get_yticks(), 2))
            ax.set_aspect('equal')
        else:
            ax.set_visible(False)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes, orientation="vertical", fraction=0.02, pad=0.05
    )
    cbar.set_label(cmap_label)
    fig.subplots_adjust(right=0.85)

    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_hist(array, ylabel="Frequency", title="", figsize=(4, 3), xlim=None, savefig="", dpi=200, density=True,
              bins=None, hidefig=False):
    plt.figure(figsize=figsize)
    plt.title(title)
    if bins is not None:
        plt.hist(array, density=density, bins=bins)
    else:
        plt.hist(array, density=density)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.ylabel(ylabel)
    plt.grid(True)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_scatter(x, y, xlabel, ylabel, title, vert_line=None, xlim=None, ylim=None, figsize=(8, 5), savefig="",
                 hidefig=False, dpi=200):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.scatter(x, y)
    if vert_line is not None:
        if type(vert_line) != list:
            plt.axvline(x=vert_line, linestyle="--", color="k")
        else:
            for vline in vert_line:
                plt.axvline(x=vline, linestyle="--", color="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_matrix(matrix, unit="px", colorbar=False, cmap="twilight", figsize=(4, 3), savefig="", dpi=200,
                title=None, origin="upper", cmap_limits=None, cmap_label="", remove_axes=False, scale=(1, 1, 1),
                hidefig=False):
    fig, ax = plt.subplots(figsize=figsize)
    if cmap_limits is not None:
        vmin, vmax = cmap_limits
    else:
        vmin, vmax = matrix.min(), matrix.max()
    ax.imshow(matrix, aspect=scale[1] / scale[2], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        cmapable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cbar = fig.colorbar(cmapable, ax=ax, orientation="vertical", fraction=0.02, pad=0.05)
        cbar.set_label(cmap_label)
    if remove_axes:
        ax.axis('off')
    else:
        ax.set_xlabel(f"X ({unit})")
        ax.set_ylabel(f"Y ({unit})")
        ax.set_xticks(np.linspace(0, matrix.shape[1], 5))
        ax.set_xticklabels(np.round(ax.get_xticks() * scale[2], 2))
        ax.set_yticks(np.linspace(0, matrix.shape[0], 5))
        ax.set_yticklabels(np.round(ax.get_yticks() * scale[1], 2))
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def get_shuffled_cmap(mask, cmap_name="nipy_spectral"):
    labels = np.unique(mask)
    base_cmap = plt.cm.get_cmap(cmap_name, len(labels))
    colors = base_cmap(np.arange(len(labels)))
    np.random.seed(0)
    np.random.shuffle(colors)
    return plt.cm.colors.ListedColormap(colors)


def plot_img_2d_masks(matrix, segmentation_mask, unit="px", figsize=(4, 3), savefig="", dpi=200,
                      title="", origin="upper", cmap_label="", remove_axes=False, scale=(1, 1, 1),
                      hidefig=False, alpha=0.7, cmap="nipy_spectral", mask_outline="black"):
    masks, outlines = segmentation_mask
    cmap = get_shuffled_cmap(masks, cmap_name=cmap)
    masks = masks.astype(float)
    masks[masks == 0] = np.nan
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(matrix, aspect=scale[1] / scale[2], cmap="gray", origin=origin)
    plt.imshow(masks, cmap=cmap, alpha=alpha, origin=origin)
    for contour in outlines:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=0.7, color=mask_outline)
    plt.colorbar(label=cmap_label)
    if remove_axes:
        plt.axis('off')
    else:
        plt.xlabel(f"X ({unit})")
        plt.ylabel(f"Y ({unit})")
        ax = plt.gca()
        xticks = np.linspace(0, matrix.shape[1], 5)
        yticks = np.linspace(0, matrix.shape[0], 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.round(xticks * scale[2], 2))
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.round(yticks * scale[1], 2))
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_matrix_vectors(x, y, angle_field, matrix, veclength=1, title=None, figsize=(4, 3),
                        savefig="", dpi=200, vec_colors=None, matrix_cmap="Greys_r", vec_cmap="Spectral",
                        cbar_vector_label="", cbar_matrix_label="", matrix_origin="upper", unit="px",
                        remove_axes=False, scale=(1, 1, 1), img_cmap_limits=None, vec_cmap_limits=None, hidefig=False):
    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        plt.title(title)
    vx = np.cos(angle_field)
    vy = np.sin(angle_field)
    if img_cmap_limits is not None:
        vmin, vmax = img_cmap_limits
    else:
        vmin, vmax = matrix.min(), matrix.max()
    ax.imshow(matrix, cmap=matrix_cmap, origin=matrix_origin, aspect=scale[1] / scale[2], vmin=vmin, vmax=vmax)

    cmapable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=matrix_cmap)
    cbar = fig.colorbar(cmapable, ax=ax, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label(cbar_matrix_label)

    if vec_colors is not None:
        if vec_cmap_limits is not None:
            vmin, vmax = vec_cmap_limits
        else:
            vmin, vmax = vec_colors.min(), vec_colors.max()
        cmapable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=vec_cmap)
        cbar = fig.colorbar(cmapable, ax=ax, orientation="vertical", fraction=0.02, pad=0.05)
        cbar.set_label(cbar_vector_label)
        cmapable.set_array([])
        colors = cmapable.to_rgba(vec_colors)
    else:
        colors = "red"
    ax.quiver(x, y, vx, vy, color=colors, scale=1 / veclength, pivot='middle',
              scale_units="xy", headlength=0, headwidth=0, headaxislength=0)
    if remove_axes:
        ax.axis('off')
    else:
        ax.set_xlabel(f"X ({unit})")
        ax.set_ylabel(f"Y ({unit})")
        ax.set_xticks(np.linspace(0, matrix.shape[1], 5))
        ax.set_xticklabels(np.round(ax.get_xticks() * scale[2], 2))
        ax.set_yticks(np.linspace(0, matrix.shape[0], 5))
        ax.set_yticklabels(np.round(ax.get_yticks() * scale[1], 2))

    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_slice_heatmap(coords, values, img_dim, img_scale, pt_size=20, grid_n=400, cmap_label="cmap_label", title="",
                       cmap="Spectral", manual_vminvmax=None, interp_method="cubic", savefig="", dpi=200,
                       figsize=(9, 5), hidefig=False):
    x, y = coords[:, 0], coords[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_n)
    yi = np.linspace(y.min(), y.max(), grid_n)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), values, (xi, yi), method=interp_method)
    plt.figure(figsize=figsize)
    if manual_vminvmax is None:
        manual_vminvmax = [values.min(), values.max()]
    plt.imshow(zi, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
               cmap=cmap, aspect="auto", vmin=manual_vminvmax[0], vmax=manual_vminvmax[1])
    plt.colorbar(label=cmap_label)
    plt.title(title)
    plt.scatter(x, y, c=values, cmap=cmap, edgecolor='k', vmin=manual_vminvmax[0], vmax=manual_vminvmax[1], s=pt_size)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.xlim(0, img_dim[2] * img_scale[2])
    plt.ylim(img_dim[1] * img_scale[1], 0)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_director_bins(ap_par_binned_dirs, ap_orth_binned_dirs, ap_par_binned_idxs, ap_orth_binned_idxs,
                       cmap="tab20", unit="px", savefig="", dpi=200, figsize=(14, 6), pt_size=0.2, hidefig=False,
                       vmin_vmax_par=None, vmin_vmax_orth=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    configs = [
        (ap_par_binned_dirs, ap_par_binned_idxs, "Parallel Bins", vmin_vmax_par),
        (ap_orth_binned_dirs, ap_orth_binned_idxs, "Orthogonal Bins", vmin_vmax_orth),
    ]

    for ax, (dirs, idxs, title, vlim) in zip(axes, configs):
        all_dirs = np.concatenate(dirs)
        all_bins = np.concatenate([np.full(len(group), i) for i, group in enumerate(idxs)])

        if vlim is None:
            vmin, vmax = all_bins.min(), all_bins.max()
        else:
            vmin, vmax = vlim

        sc = ax.scatter(
            all_dirs[:, 0], all_dirs[:, 1],
            s=pt_size, c=all_bins, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_aspect("equal")
        ax.set_xlabel(f"X ({unit})")
        ax.set_ylabel(f"Y ({unit})")
        ax.invert_yaxis()
        cb = fig.colorbar(sc, ax=ax, label="Bin index", shrink=0.7, ticks=np.arange(vmin, vmax + 1))
        cb.ax.set_yticklabels([str(i) for i in range(vmin, vmax + 1)])
        ax.set_title(title)

    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_curve_projections(img, curve, pts, s_parallel, s_orthogonal, scale=(1, 1, 1), unit="px", savefig="",
                           dpi=200, figsize=(14, 6), curvewidth=3, pt_size=10, im_alpha=0.6, hidefig=False):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, s_values, title, label in zip(
            axes,
            [s_parallel, s_orthogonal],
            [f"A-P || coordinate ({unit})", f"A-P ⊥ coordinate ({unit})"],
            [f"A-P || coordinate ({unit})", f"A-P ⊥ coordinate ({unit})"]
    ):
        ax.imshow(img, cmap="gray", alpha=im_alpha, aspect=scale[1] / scale[2], origin="upper")
        ax.plot(curve[:, 0] / scale[2], curve[:, 1] / scale[1], "k-", linewidth=curvewidth, label="Centerline")
        sc = ax.scatter(pts[:, 0] / scale[2], pts[:, 1] / scale[1], c=s_values, cmap="Spectral",
                        label=label, s=pt_size)
        ax.set_xlabel(f"X ({unit})")
        ax.set_ylabel(f"Y ({unit})")
        ax.set_xticks(np.linspace(0, img.shape[1], 5))
        ax.set_xticklabels(np.round(ax.get_xticks() * scale[2], 2))
        ax.set_yticks(np.linspace(0, img.shape[0], 5))
        ax.set_yticklabels(np.round(ax.get_yticks() * scale[1], 2))
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_binned_ap_results_horizontal(img, curve, ap_par_binned_s_2d_weighted, ap_orth_binned_s_2d_weighted,
                                      s_parallel_bin_centers, s_orthogonal_bin_centers,
                                      s_par_orthogonality_weighted, s_orth_orthogonality_weighted,
                                      figsize=(16, 8), scale=(1, 1, 1), unit="px", savefig="", dpi=200, hidefig=False):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 6], height_ratios=[1, 1], wspace=0.2, hspace=0.3)

    # Left: Full height image
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img, cmap='gray', origin='upper', aspect=scale[1] / scale[2])
    ax_img.scatter(curve[:, 0] / scale[2], curve[:, 1] / scale[1], s=2, c=np.linspace(0, 1, len(curve)),
                   cmap="Spectral", label="Central line of Elongation")
    ax_img.set_xlabel(f"X ({unit})")
    ax_img.set_ylabel(f"Y ({unit})")
    ax_img.set_xticks(np.linspace(0, img.shape[1], 5))
    ax_img.set_xticklabels(np.round(ax_img.get_xticks() * scale[2], 2))
    ax_img.set_yticks(np.linspace(0, img.shape[0], 5))
    ax_img.set_yticklabels(np.round(ax_img.get_yticks() * scale[1], 2))

    # Top: parallel profile
    ax_parallel = fig.add_subplot(gs[0, 1])
    ax_parallel.grid(color="grey", linestyle="--", alpha=0.5)
    ax_br_ax2 = ax_parallel.twinx()
    ax_parallel.axhline(1, c="darkblue", linestyle="--", alpha=0.5)
    ax_parallel.plot(s_parallel_bin_centers, ap_par_binned_s_2d_weighted, "s-", label='Weighted by $AR-1$',
                     color='darkblue', alpha=0.7)
    ax_br_ax2.plot(s_parallel_bin_centers, s_par_orthogonality_weighted, "s-", label='Weighted by $AR-1$',
                   color='brown', alpha=0.7)

    ax_parallel.set_xlabel(f"A-P || coordinate ({unit})")
    ax_parallel.set_ylabel('Nematic order $S$ weighted by $AR-1$', color='darkblue')
    ax_parallel.tick_params(axis='y', colors='darkblue')
    ax_br_ax2.set_ylabel('Orthogonality', color='brown')
    ax_br_ax2.tick_params(axis='y', colors='brown')
    ax_br_ax2.set_ylim(-0.05, 1.05)

    # Bottom: orthogonal profile
    ax_orthogonal = fig.add_subplot(gs[1, 1])
    ax_orthogonal.grid(color="darkblue", linestyle="--", alpha=0.5)
    ax_tr_ax2 = ax_orthogonal.twinx()
    ax_orthogonal.axhline(1, c="darkblue", linestyle="--", alpha=0.5)
    ax_orthogonal.plot(s_orthogonal_bin_centers, ap_orth_binned_s_2d_weighted, "s-",
                       label='Weighted by $AR-1$', color='darkblue', alpha=0.7)
    ax_tr_ax2.plot(s_orthogonal_bin_centers, s_orth_orthogonality_weighted, "o-", label='Weighted by $AR-1$',
                   color='brown', alpha=0.7)
    ax_orthogonal.set_xlabel(f"A-P ⊥ coordinate ({unit})")
    ax_orthogonal.set_ylabel('Nematic order $S$ weighted by $AR-1$', color='darkblue')
    ax_orthogonal.tick_params(axis='y', colors='darkblue')
    ax_tr_ax2.set_ylabel('Orthogonality', color='brown')
    ax_tr_ax2.tick_params(axis='y', colors='brown')
    ax_tr_ax2.set_ylim(-0.05, 1.05)

    ax_parallel.set_ylim(bottom=0.0)
    ax_orthogonal.set_ylim(bottom=0.0)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_spherical_projection(phi, theta, intensities,
                              hexview=True, ptview=False, hexgridsize=200,
                              cmap="Greens", vec_pos_phi=None, vec_pos_theta=None,
                              vec_dir_phi=None, vec_dir_theta=None, veccolor="red",
                              vec_manual_vminmax=None, vec_cmap="Spectral",
                              scale_factor=1.0, figsize=(5, 4), arrow_alpha=1.0,
                              vec_cmap_label="", vec_width=0.001, ptsize=2, alpha=1.0,
                              invert_y_axis=False, aspect=2,
                              cmap_label="Intensity Signal (a.u.)", manual_vminmax=None, savefig="", dpi=200,
                              hidefig=False, marker_idxs=None, marker_color="yellow",
                              marker_size=500, marker_alpha=0.9, marker_vec=None, marker_vec_scale=1.0,
                              marker_vec_width=0.002, marker_vec_color="red"):
    draw_vectors = all(x is not None for x in [vec_pos_phi, vec_pos_theta, vec_dir_phi, vec_dir_theta])
    vmin, vmax = (intensities.min(), intensities.max()) if manual_vminmax is None else manual_vminmax
    if draw_vectors and not isinstance(veccolor, str):
        vec_norm = plt.Normalize(vmin=np.min(veccolor) if vec_manual_vminmax is None else vec_manual_vminmax[0],
                                 vmax=np.max(veccolor) if vec_manual_vminmax is None else vec_manual_vminmax[1])
    else:
        vec_norm = None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.05, 0.1, 0.75, 0.8])
    if hexview:
        ax.hexbin(phi, theta, C=intensities, cmap=cmap, gridsize=hexgridsize, alpha=alpha)
    if ptview:
        ax.scatter(phi, theta, c=intensities, cmap=cmap, s=ptsize, alpha=alpha)
    if draw_vectors:
        if isinstance(veccolor, str):
            ax.quiver(vec_pos_phi, vec_pos_theta, vec_dir_phi * scale_factor, vec_dir_theta * scale_factor,
                      pivot="middle", headlength=0, headwidth=0, headaxislength=0,
                      angles='xy', scale_units='xy', scale=1, color=veccolor,
                      alpha=arrow_alpha, width=vec_width)
        else:
            ax.quiver(vec_pos_phi, vec_pos_theta, vec_dir_phi * scale_factor, vec_dir_theta * scale_factor,
                      pivot="middle", headlength=0, headwidth=0, headaxislength=0,
                      angles='xy', scale_units='xy', scale=1,
                      color=plt.cm.get_cmap(vec_cmap)(vec_norm(veccolor)),
                      alpha=arrow_alpha, width=vec_width)

    if marker_idxs is not None:
        if type(marker_color) == str:
            ax.scatter(phi[marker_idxs], theta[marker_idxs], color=marker_color, s=marker_size, alpha=marker_alpha)
        else:
            ax.scatter(phi[marker_idxs], theta[marker_idxs], c=marker_color, s=marker_size,
                       alpha=marker_alpha)
        if marker_vec is not None:
            marker_vec_phi, marker_vec_theta, marker_vec_idxs = marker_vec
            ax.quiver(phi[marker_vec_idxs], theta[marker_vec_idxs],
                      marker_vec_phi * marker_vec_scale,
                      marker_vec_theta * marker_vec_scale,
                      angles='xy', scale_units='xy', scale=1,
                      color=marker_vec_color, alpha=marker_alpha, width=marker_vec_width)
    if invert_y_axis:
        ax.invert_yaxis()
    ax.set_aspect(aspect)
    ax.set_xlabel("Phi (deg)")
    ax.set_ylabel("Theta (deg)")
    cax_int = fig.add_axes([0.75, 0.1, 0.03, 0.8])
    sm_int = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm_int.set_array([])
    cbar_int = fig.colorbar(sm_int, cax=cax_int)
    cbar_int.set_label(cmap_label)
    if draw_vectors and not isinstance(veccolor, str):
        cax_vec = fig.add_axes([0.82, 0.1, 0.03, 0.8])
        sm_vec = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(vec_cmap), norm=vec_norm)
        sm_vec.set_array([])
        cbar_vec = fig.colorbar(sm_vec, cax=cax_vec)
        cbar_vec.set_label(vec_cmap_label)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_cylindrical_projection(phi, rho, s, colors, cbar_label="Fluorescence Intensity (a.u.)", hexsize=400,
                                cmap="magma", manual_vminmax=None, shift_angle_rad=None, title="", savefig="",
                                hidefig=False, xlabel=fr'Arc length $s$ (um)', ylabel=r'$\rho\,\phi$  (um)',
                                aspect="equal", dpi=300, figsize=(10, 10)):
    phi_proj = phi.copy()
    if shift_angle_rad is not None:
        phi_proj = (phi - shift_angle_rad + np.pi) % (2 * np.pi) - np.pi
    y = rho * phi_proj
    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    if manual_vminmax is None:
        vmin, vmax = colors.min(), colors.max()
    else:
        vmin, vmax = manual_vminmax
    img = ax.hexbin(s, y, C=colors, gridsize=hexsize, cmap=cmap,
                    vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax, label=cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect(aspect)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_interp_grid(grid_x, grid_y, grid_z, theta=None, linelength=2, figsize=(4, 3), hidefig=False):
    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap='Greys_r')
    if theta is not None:
        x_center = (grid_x.min() + grid_x.max()) / 2
        y_center = (grid_y.min() + grid_y.max()) / 2
        x_vec = x_center + np.array([-np.cos(theta), np.cos(theta)]) * linelength / 2
        y_vec = y_center + np.array([-np.sin(theta), np.sin(theta)]) * linelength / 2
        ax.plot(x_vec, y_vec, color="red", linewidth=2, label="Orientation")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Local X')
    ax.set_ylabel('Local Y')
    ax.set_title('Interpolated Neighborhood')
    fig.colorbar(contour, ax=ax, orientation='vertical', label='Fluorescence Intensity (a.u.)')

    if not hidefig:
        plt.show()
    else:
        plt.close(fig)


def plot_vector_field(x, y, u, v, defect, savefigpath="", figsize=(3, 3), dpi=200, hidefig=False):
    plt.figure(figsize=figsize)
    print(">> Plotting the vector field with(out) defect...")
    plt.quiver(x, y, u, v, color='white', scale=20, pivot='middle', headaxislength=0)
    plt.title(f"{defect} Defect")
    plt.tight_layout()
    if savefigpath != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefigpath, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_dist_kymograph(distances, intensities, cmap, figsize, unit, savefig="", dpi=200, hidefig=False):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    intensities = normalise_range(intensities)
    axes[0].imshow(intensities, aspect="auto", cmap=cmap, extent=[0, distances.max(), len(intensities), 0])
    axes[0].set_ylabel("Sampling Point Index")
    axes[0].set_xlabel(f"Distance from min Distance ({unit})")
    axes[1].plot(distances, np.average(intensities, axis=0), "o-")
    axes[1].set_ylabel("Average Intensity (a.u.)")
    axes[1].set_xlabel(f"Distance from min Distance ({unit})")
    mean_intensity = distances[np.argmax(np.average(intensities, axis=0))]
    axes[1].axvline(x=mean_intensity, label=f"ARGMAX = {round(mean_intensity, 1)} {unit}")
    axes[1].legend()
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_dir_field(directors=None, t1_raw=None, t2_raw=None, normals=None, veclength=10, freq=1, figsize=(8, 6),
                   view_init=None, title=None, savefig="", dpi=200, veccolor="red", cmap_label="cmap_label",
                   cmap="Spectral", show_axes=True, xlim=None, ylim=None, zlim=None, aspect="equal", hide_cmap=False,
                   marker=None, pt_label="marker_label", pt_size=100, pt_style="o", pt_color="yellow", pt_alpha=1.0,
                   manual_vminmax=None, hidefig=False, vec_alpha=1.0):
    print(">> Plotting director field...")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = directors[::freq][:, 0], directors[::freq][:, 1], directors[::freq][:, 2]
    vx, vy, vz = directors[::freq][:, 3], directors[::freq][:, 4], directors[::freq][:, 5]
    if show_axes:
        ax.quiver(0, 0, 0, 1, 0, 0, color="k", length=10 * veclength, arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, 1, 0, color="k", length=10 * veclength, arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, 0, 1, color="k", length=10 * veclength, arrow_length_ratio=0.2)

    if t1_raw is not None:
        t1x, t1y, t1z = t1_raw[::freq][:, 0], t1_raw[::freq][:, 1], t1_raw[::freq][:, 2]
        ax.quiver(x, y, z, t1x, t1y, t1z, color="b", length=veclength, arrow_length_ratio=0.5, alpha=0.5,
                  label="Tangent 1")
    if t2_raw is not None:
        t2x, t2y, t2z = t2_raw[::freq][:, 0], t2_raw[::freq][:, 1], t2_raw[::freq][:, 2]
        ax.quiver(x, y, z, t2x, t2y, t2z, color="g", length=veclength, arrow_length_ratio=0.5, alpha=0.5,
                  label="Tangent 2")
    if normals is not None:
        nx, ny, nz = normals[::freq][:, 0], normals[::freq][:, 1], normals[::freq][:, 2]
        ax.quiver(x, y, z, nx, ny, nz, color="y", length=veclength, arrow_length_ratio=0.5, alpha=0.5,
                  label="Normal")
    centered_x = x - 0.5 * vx * veclength
    centered_y = y - 0.5 * vy * veclength
    centered_z = z - 0.5 * vz * veclength
    if type(veccolor) is not str:
        if manual_vminmax is not None:
            norm = plt.Normalize(vmin=manual_vminmax[0], vmax=manual_vminmax[1])
        else:
            norm = plt.Normalize(vmin=np.min(veccolor), vmax=np.max(veccolor))
        ax.quiver(centered_x, centered_y, centered_z, vx, vy, vz, length=veclength,
                  arrow_length_ratio=0.0, label="Director", alpha=vec_alpha,
                  color=colormaps[cmap](norm(veccolor)), cmap=cmap)
        if not hide_cmap:
            sm = plt.cm.ScalarMappable(cmap=colormaps[cmap], norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=cmap_label)
    else:
        ax.quiver(centered_x, centered_y, centered_z, vx, vy, vz, color=veccolor, length=veclength,
                  arrow_length_ratio=0.0, label="Director", alpha=vec_alpha)
    if marker is not None:
        ax.scatter(marker[:, 0], marker[:, 1], marker[:, 2], marker=pt_style, color=pt_color, label=pt_label, s=pt_size,
                   alpha=pt_alpha)
    ax.set_aspect(aspect)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_zlabel('Z')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if view_init is not None:
        ax.view_init(view_init[0], view_init[1])
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_polar_hist(angles_deg, title="Polar Histogram of Theta Angles", figsize=(6, 6), savefig="", dpi=200, bins=30,
                    hidefig=False):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    counts, bin_edges = np.histogram(angles_deg, bins=bins, range=(-90, 90), density=True)
    bin_centers_rad = np.radians((bin_edges[:-1] + bin_edges[1:]) / 2)
    ax.bar(bin_centers_rad, counts, width=np.radians(np.diff(bin_edges)), bottom=0, align='center')
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks(np.radians([-90, -45, 0, 45, 90]))
    ax.set_xticklabels(["-90°", "-45°", "0°", "45°", "90°"])
    ax.set_title(title, va='bottom')
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def animate_img_slices(img, scale, unit, gifpath, gifsuffix, time_spacing=0.1, cmap="Greens", dpi=100,
                       cmap_label="Fluorescence Intensity (a.u.)", cleanup_frames=True, figsize=(5, 4), temp_path=None):
    print(">> Animating slices...")
    if temp_path is None:
        temp_path = os.path.join(gifpath, "temp")
    create_figdir(temp_path)
    slices = {
        "Z": (img.shape[0], lambda j: img[j, :, :], scale[1] / scale[2]),
        "Y": (img.shape[1], lambda j: img[:, j, :], scale[0] / scale[2]),
        "X": (img.shape[2], lambda j: img[:, :, j], scale[0] / scale[1])
    }
    temp_files = []
    for axis, (num_slices, slice_func, aspect_ratio) in slices.items():
        print(f">> {axis} slices...")
        frames = []

        for i in range(num_slices):
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(f"{axis}={i} Slice")
            im = ax.imshow(slice_func(i), cmap=cmap, aspect=aspect_ratio, origin="lower" if axis != "Z" else "upper")
            fig.colorbar(im, label=cmap_label)

            ax.set_xlabel(f'X ({unit})' if axis != "X" else f'Y ({unit})')
            ax.set_xticks(np.linspace(0, img.shape[2 if axis != "X" else 1], 5))
            ax.set_xticklabels(np.round(ax.get_xticks() * scale[2 if axis != "X" else 1], 2))

            ax.set_ylabel(f'Y ({unit})' if axis == "Z" else f'Z ({unit})')
            ax.set_yticks(np.linspace(0, img.shape[1 if axis == "Z" else 0], 5))
            ax.set_yticklabels(np.round(ax.get_yticks() * scale[1 if axis == "Z" else 0], 2))

            plt.draw()
            frame_path = os.path.join(temp_path, f"frame_{axis}_{i}.png")
            plt.savefig(os.path.join(temp_path, f"frame_{axis}_{i}.png"), dpi=dpi, bbox_inches='tight')
            temp_files.append(frame_path)
            frames.append(imageio.imread_v2(frame_path))
            plt.close(fig)

        max_shape = np.array([frame.shape[:2] for frame in frames]).max(axis=0)

        padded_frames = []
        for frame in frames:
            pad_h = max_shape[0] - frame.shape[0]
            pad_w = max_shape[1] - frame.shape[1]
            pad_h = max(0, pad_h)
            pad_w = max(0, pad_w)

            if frame.ndim == 2:
                padded_frame = np.pad(frame, [(0, pad_h), (0, pad_w)], mode='edge')
            else:
                padded_frame = np.pad(frame, [(0, pad_h), (0, pad_w), (0, 0)], mode='edge')

            padded_frames.append(padded_frame)

        imageio.mimsave(os.path.join(gifpath, f"{axis}_{gifsuffix}.gif"), padded_frames, duration=time_spacing)
        print(f"Saved animation for {axis} slice to {gifpath}!")
    if cleanup_frames:
        for file in temp_files:
            os.remove(file)
        shutil.rmtree(temp_path, ignore_errors=True)
    print("Animation complete !")


def plot_violinplot_comparative(sequence_data, sequence_labels, title, sequence_label="Time", group_labels=None,
                                group_name=None, figsize=(8, 5), savefig="", hidefig=False, dpi=200):
    values = []
    times = []
    groups = []
    for t_idx, time_point in enumerate(sequence_data):
        for g_idx, group_data in enumerate(time_point):
            values.extend(group_data)
            times.extend([sequence_labels[t_idx]] * len(group_data))
            groups.extend([group_labels[g_idx] if group_labels else f"Group {g_idx + 1}"] * len(group_data))

    df = pd.DataFrame({title: values, sequence_label: times, "Group": groups})

    plt.figure(figsize=figsize)
    sns.violinplot(x=sequence_label, y=title, hue="Group", data=df, palette="Set2", inner="quartile")
    sns.swarmplot(x=sequence_label, y=title, hue="Group", data=df, dodge=True, color="k", alpha=0.6, size=1.5,
                  legend=False)
    plt.title(title)
    plt.legend(title=group_name)
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_violinplot(sequence_data, sequence_labels, title, sequence_label="Time", figsize=(8, 5), savefig="",
                    hidefig=False, dpi=200, ):
    values = []
    groups = []
    for label, seq in zip(sequence_labels, sequence_data):
        values.extend(seq)
        groups.extend([label] * len(seq))

    df = pd.DataFrame({title: values, sequence_label: groups})

    plt.figure(figsize=figsize)
    sns.violinplot(x=sequence_label, y=title, data=df, inner="quartile")
    sns.swarmplot(x=sequence_label, y=title, data=df, color="k", alpha=0.6, size=2, legend=False)

    plt.title(title)
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_qsphi_profiles(dir_s, s_bin_centers, Q_ss, Q_ss_mean, Q_phiphi, Q_phiphi_mean, Q_sphi, Q_sphi_mean,
                        y_limits=None, figsize=(10, 5), savefig="", hidefig=False, dpi=200):
    plt.figure(figsize=figsize)
    plt.scatter(dir_s, Q_ss, color='tab:blue', alpha=0.2, s=5)
    plt.scatter(dir_s, Q_phiphi, color='tab:orange', alpha=0.2, s=5)
    plt.scatter(dir_s, Q_sphi, color='tab:green', alpha=0.2, s=5)
    line_ss, = plt.plot(s_bin_centers, Q_ss_mean, color='tab:blue', lw=2)
    line_pp, = plt.plot(s_bin_centers, Q_phiphi_mean, color='tab:orange', lw=2)
    line_sp, = plt.plot(s_bin_centers, Q_sphi_mean, color='tab:green', lw=2)
    plt.legend([line_ss, line_pp, line_sp], [r'$Q_{ss}$', r'$Q_{\phi\phi}$', r'$Q_{s\phi}$'], loc='upper right')
    plt.xlabel("Arc length $s$ (µm)", fontsize=12)
    plt.ylabel("Component magnitude", fontsize=12)
    plt.title(r"Tangential Nematic Order in $s$-$\phi$ basis", fontsize=14)
    plt.grid(True)
    if y_limits is not None:
        plt.ylim(y_limits)
    plt.yticks(ticks=np.arange(-0.6, 0.6, 0.1))
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_qsphi_profiles_separated_phi(dir_s, dir_phi, s_bin_centers, Q_ss, Q_ss_mean, Q_phiphi, Q_phiphi_mean, Q_sphi,
                                      Q_sphi_mean, y_limits=None,
                                      figsize=(10, 10), savefig="", hidefig=False, dpi=200):
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={'hspace': 0.15}
    )

    Q_components = [
        (Q_phiphi, Q_phiphi_mean, r"$Q_{\phi\phi}$"),
        (Q_ss, Q_ss_mean, r"$Q_{ss}$"),
        (Q_sphi, Q_sphi_mean, r"$Q_{s\phi}$"),
    ]
    cmap = "hsv"
    phi_min, phi_max = -np.pi, np.pi
    sc = None
    for ax, (Q_raw, Q_mean, label) in zip(axes, Q_components):
        sc = ax.scatter(
            dir_s, Q_raw, c=dir_phi,
            cmap=cmap, vmin=phi_min, vmax=phi_max,
            s=8, alpha=1.0
        )
        ax.plot(
            s_bin_centers, Q_mean,
            color="k", lw=2, label="mean"
        )
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel(r"Arc length $s$ (µm)", fontsize=12)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="3%", pad=0.35)
    cbar = plt.colorbar(sc, cax=cax, orientation='vertical',
                        ticks=[-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    cbar.set_label(r'$\phi$ (rad)')

    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_rho_profile(mesh_s, mesh_rho, mesh_phi, img_unit, figsize=(10, 4), savefig="", hidefig=False, dpi=200):
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        mesh_s, mesh_rho,
        c=mesh_phi,
        cmap='hsv',
        s=5,
        alpha=0.7,
        vmin=-np.pi,
        vmax=np.pi
    )
    ax.set_xlabel(f'Arc length $s$ ({img_unit})')
    ax.set_ylabel(fr'Radial distance $\rho$ ({img_unit})')
    ax.set_title(fr'Radial distance $\rho$ to curve ({img_unit})')
    ax.grid(True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(sc, cax=cax, orientation='vertical', ticks=[-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    cbar.set_label(r'$\phi$ (rad)')
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_rho_profile_evolution(time_points_all, mesh_s_all, mesh_rho_all, figsize=(10, 4), xlabel=None, ylabel=None,
                               cbarlabel=None, title=None, savefig="", hidefig=False, dpi=200):
    time_points_all = np.array(time_points_all)
    time_unique = np.sort(np.unique(time_points_all))
    cmap = cm.coolwarm(np.linspace(0, 1, len(time_unique)))
    colors = [cmap[np.where(time_unique == tp)[0][0]] for tp in time_points_all]

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(right=0.88, top=0.9, bottom=0.15)
    for i, color in enumerate(colors):
        s_vals = mesh_s_all[i]
        rho_vals = mesh_rho_all[i]
        ax.scatter(s_vals, rho_vals, s=1, alpha=0.007, color=color, label=f"{time_points_all[i]}h")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    cmap_listed = ListedColormap(cmap)
    bounds = np.arange(len(time_unique) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap_listed.N)
    sm = cm.ScalarMappable(cmap=cmap_listed, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=np.arange(len(time_unique)))
    if cbarlabel is not None:
        cbar.set_label(cbarlabel, fontsize=12)
    cbar.ax.set_yticklabels([str(int(tp)) for tp in time_unique])
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_qsphi_profile_evolution(time_points_all, s_bin_centers_all, q_mean_all_dict, figsize=(10, 10), ylim=None,
                                 xlabel=None,
                                 cbarlabel=None, title=None, savefig="", hidefig=False, dpi=200):
    time_points_all = np.array(time_points_all)
    time_unique = np.sort(np.unique(time_points_all))
    cmap = cm.coolwarm(np.linspace(0, 1, len(time_unique)))
    colors = [cmap[np.where(time_unique == tp)[0][0]] for tp in time_points_all]
    line_styles = {'Q_ss': '-', 'Q_phiphi': '-', 'Q_sphi': '-'}
    line_widths = {'Q_ss': 3, 'Q_phiphi': 3, 'Q_sphi': 3}
    component_labels = {'Q_ss': r'$Q_{ss}$', 'Q_phiphi': r'$Q_{\phi\phi}$', 'Q_sphi': r'$Q_{s\phi}$'}
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)
    plt.subplots_adjust(hspace=0.15, right=0.88, top=0.93)
    for ax in axes:
        ax.axhline(0, c="grey", linestyle="--")
    for ax, comp in zip(axes, ['Q_phiphi', 'Q_ss', 'Q_sphi']):
        for i, color in enumerate(colors):
            s_centers_norm = s_bin_centers_all[i]
            Q_mean = q_mean_all_dict[comp][i]
            ax.plot(s_centers_norm, Q_mean, color=color,
                    lw=line_widths[comp], linestyle=line_styles[comp])
        ax.set_ylabel(f"{component_labels[comp]}", fontsize=13)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)

    if xlabel is not None:
        axes[-1].set_xlabel(xlabel, fontsize=13)
    if title is not None:
        axes[0].set_title(title, fontsize=15, pad=20)
    cmap_listed = ListedColormap(cmap)
    bounds = np.arange(len(time_unique) + 1) - 0.5  # centre ticks
    norm = BoundaryNorm(bounds, cmap_listed.N)
    sm = cm.ScalarMappable(cmap=cmap_listed, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.04, pad=0.03, ticks=np.arange(len(time_unique)))
    if cbarlabel is not None:
        cbar.set_label(cbarlabel, fontsize=12)
    cbar.ax.set_yticklabels([str(int(tp)) for tp in time_unique])

    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


#####################
# 3D RENDER MODULES #
#####################
def view_img(img_list, scale, color_list=None, title_list=None, opacity_list=None):
    print(">> Rendering image...")
    if opacity_list is None:
        opacity_list = np.ones(len(img_list))
    if color_list is None:
        color_list = ["green" for _ in range(len(img_list))]
    if title_list is None:
        title_list = ["layer" for _ in range(len(img_list))]
    if type(scale) == tuple:
        scale_list = [scale for _ in range(len(img_list))]
    elif type(scale) == list:
        scale_list = scale
    else:
        scale_list = [(1, 1, 1) for _ in range(len(img_list))]
    viewer = napari.Viewer()
    for i, img in enumerate(img_list):
        viewer.add_image(img, name=title_list[i], colormap=color_list[i], rendering="mip", scale=scale_list[i], opacity=
        opacity_list[i])
    viewer.dims.ndisplay = 3
    napari.run()


def view_mesh(mesh_list, mesh_colors=None, mesh_titles=None, mesh_opacities=None, vec_freq=15, img=None,
              img_cmap="green", scale=(1, 1, 1), vec_edge_width=None, vec_length=1, img_opacity=0.4,
              hide_vectors=True):
    print(">> Rendering mesh...")
    viewer = napari.Viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=img_cmap, rendering="mip", scale=scale, opacity=img_opacity)

    if mesh_colors is None:
        mesh_colors = ["white" for _ in range(len(mesh_list))]
    if mesh_titles is None:
        mesh_titles = ["Mesh" for _ in range(len(mesh_list))]
    if mesh_opacities is None:
        mesh_opacities = [0.4 for _ in range(len(mesh_list))]
    if vec_edge_width is None:
        vec_edge_width = vec_length / 6

    for i, mesh in enumerate(mesh_list):
        viewer.add_surface((mesh.vertices, mesh.faces), name=mesh_titles[i], shading='none', opacity=mesh_opacities[i],
                           blending="translucent", colormap=mesh_colors[i])
        if not hide_vectors:
            viewer.add_vectors(
                data=np.stack((mesh.vertices[::vec_freq], mesh.vertex_normals[::vec_freq]), axis=1),
                edge_color=mesh_colors[i], edge_width=vec_edge_width,
                length=vec_length, opacity=1.0, name=f"{mesh_titles[i]}_normals"
            )
    viewer.dims.ndisplay = 3
    napari.run()


def view_colored_verts(verts, colors, scale=None, img=None, ptsize=2, cmap_img="green", blending="opaque",
                       shading="none", opacity=0.2, use_orig_color=True):
    print(">> Rendering colored vertices...")
    viewer = napari.Viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale)
    if use_orig_color:
        colors_mapped = colors.copy()
    else:
        colors_mapped = np.zeros((colors.shape[0], 3))
        colors_mapped[:, 1] = colors
    viewer.add_points(verts, name='Mesh', shading=shading, opacity=opacity, blending=blending,
                      face_color=colors_mapped, border_color=colors_mapped, size=ptsize)
    viewer.dims.ndisplay = 3
    napari.run()


def view_colored_mesh(mesh, vert_colors="red", mesh_shading="none", mesh_opacity=1.0, img=None, scale=None,
                      img_opacity=1.0, cmap_img="green", color_override=None, mesh_blending="opaque", markers=None,
                      marker_colors="yellow", marker_size=5, add_hidden_vector=False):
    print(">> Rendering colored mesh...")

    if type(vert_colors) == str:
        vert_colors = np.tile(np.array(pltcolors.to_rgb(vert_colors)), (mesh.vertices.shape[0], 1))
    else:
        if type(vert_colors[0]) == str:
            vert_colors = np.array([pltcolors.to_rgb(c) for c in vert_colors])
        else:
            vert_colors = np.array(
                [tuple(int(hex_code.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)) for hex_code in vert_colors])
    if color_override is not None:
        vert_colors = color_override.copy()
        vert_colors[:, 3] = 1.0
    viewer = napari.Viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale, opacity=img_opacity)
        mesh_opacity *= 0.8
    viewer.add_surface((mesh.vertices, mesh.faces), vertex_colors=vert_colors, shading=mesh_shading,
                       opacity=mesh_opacity, blending=mesh_blending)
    if markers is not None:
        viewer.add_points(markers, name='Defects', shading="none", opacity=0.6, blending="opaque",
                          face_color=marker_colors, border_color=marker_colors, size=marker_size)
    if add_hidden_vector:
        viewer.add_vectors(np.array([[[0, 0, 0], [0, 0, 1]]]), name="dummy", visible=False)
    viewer.dims.ndisplay = 3
    napari.run()


def view_colored_mesh_multiple(mesh_list, vert_colors_list=None, mesh_shading="none", mesh_opacity_list=None, img=None,
                               scale=None, img_opacity=1.0, cmap_img="green", color_override_list=None,
                               mesh_blending_list=None, markers=None, add_hidden_vector=False,
                               marker_colors="yellow", marker_size=5, name_list=None):
    if color_override_list is None:
        color_override_list = [None for _ in range(len(mesh_list))]
    if name_list is None:
        name_list = ["Mesh" for _ in range(len(mesh_list))]
    if vert_colors_list is None:
        vert_colors_list = ["red" for _ in range(len(mesh_list))]
    if mesh_opacity_list is None:
        mesh_opacity_list = [1.0 for _ in range(len(mesh_list))]
    if mesh_blending_list is None:
        mesh_blending_list = ["opaque" for _ in range(len(mesh_list))]
    print(">> Rendering colored mesh...")
    viewer = napari.Viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale, opacity=img_opacity)
    for mesh, vert_colors, color_override, name, opacity, blending in zip(mesh_list, vert_colors_list,
                                                                          color_override_list, name_list,
                                                                          mesh_opacity_list, mesh_blending_list):
        if type(vert_colors) == str:
            vert_colors = np.tile(np.array(pltcolors.to_rgb(vert_colors)), (mesh.vertices.shape[0], 1))
        else:
            if type(vert_colors[0]) == str:
                vert_colors = np.array([pltcolors.to_rgb(c) for c in vert_colors])
            else:
                vert_colors = np.array(
                    [tuple(int(hex_code.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)) for hex_code in
                     vert_colors])
        if color_override is not None:
            vert_colors = color_override.copy()
            vert_colors[:, 3] = 1.0

        viewer.add_surface((mesh.vertices, mesh.faces), vertex_colors=vert_colors, shading=mesh_shading,
                           opacity=opacity, blending=blending, name=name)
    if markers is not None:
        viewer.add_points(markers, name='Defects', shading="none", opacity=0.6, blending="opaque",
                          face_color=marker_colors, border_color=marker_colors, size=marker_size)
    if add_hidden_vector:
        viewer.add_vectors(np.array([[[0, 0, 0], [0, 0, 1]]]), name="dummy", visible=False)
    viewer.dims.ndisplay = 3
    napari.run()


def view_3d_vector_field(vec_pos, vec_dir, vec_colors, verts=None, verts_colors=None, edge_width=0.3, length=10,
                         vec_opacity=0.9, pts_size=1, pts_opacity=0.9, pts_blending="opaque",
                         vector_style="line", img=None, scale=None, img_opacity=0.5):
    print(">> Rendering 3D vector field...")
    viewer = napari.Viewer()
    centered_x = vec_pos[:, 0] - 0.5 * vec_dir[:, 0] * length
    centered_y = vec_pos[:, 1] - 0.5 * vec_dir[:, 1] * length
    centered_z = vec_pos[:, 2] - 0.5 * vec_dir[:, 2] * length
    centered_pos = np.column_stack((centered_x, centered_y, centered_z))
    if img is not None and scale is not None:
        viewer.add_image(img, opacity=img_opacity, scale=scale, rendering="mip")

    if verts is not None and verts_colors is not None:
        viewer.add_points(
            data=verts,
            border_color=verts_colors,
            face_color=verts_colors,
            size=pts_size,
            opacity=pts_opacity,
            blending=pts_blending
        )
    viewer.add_vectors(
        data=np.stack((centered_pos, vec_dir), axis=1),
        edge_color=vec_colors,
        edge_width=edge_width,
        length=length,
        opacity=vec_opacity,
        vector_style=vector_style
    )
    viewer.dims.ndisplay = 3
    napari.run()


def view_3d_vector_field_multiple(vec_pos, vec_dir, vec_colors, verts=None, verts_colors=None, edge_width=0.3,
                                  vec_length=10, vec_opacity=0.9, pts_size=1, pts_opacity=0.9, pts_blending="opaque",
                                  vector_style="line", img=None, scale=None, img_opacity=0.5, mesh=None,
                                  mesh_shading="flat", mesh_blending="opaque", centered_directors=True, vec_names=None):
    print(">> Rendering 3D vector field...")
    viewer = napari.Viewer()
    if img is not None and scale is not None:
        viewer.add_image(img, opacity=img_opacity, scale=scale, rendering="mip")
    if mesh is not None:
        viewer.add_surface((mesh.vertices, mesh.faces), shading=mesh_shading, opacity=1.0,
                           blending=mesh_blending)
    if verts is not None and verts_colors is not None:
        viewer.add_points(
            data=verts,
            border_color=verts_colors,
            face_color=verts_colors,
            size=pts_size,
            opacity=pts_opacity,
            blending=pts_blending
        )
    if vec_names is None:
        vec_names = ["Layer" for _ in range(len(vec_pos))]
    for i in range(len(vec_pos)):
        if centered_directors:
            centered_x = vec_pos[i][:, 0] - 0.5 * vec_dir[i][:, 0] * vec_length
            centered_y = vec_pos[i][:, 1] - 0.5 * vec_dir[i][:, 1] * vec_length
            centered_z = vec_pos[i][:, 2] - 0.5 * vec_dir[i][:, 2] * vec_length
            centered_pos = np.column_stack((centered_x, centered_y, centered_z))
        else:
            centered_pos = vec_pos[i].copy()
        viewer.add_vectors(
            data=np.stack((centered_pos, vec_dir[i]), axis=1),
            edge_color=vec_colors[i],
            edge_width=edge_width,
            length=vec_length,
            opacity=vec_opacity,
            vector_style=vector_style,
            name=vec_names[i]
        )
    viewer.dims.ndisplay = 3
    napari.run()


def view_colored_mesh_dir_field(mesh, directors, vec_colors="red", vec_length=20, vec_edge_width=None,
                                mesh_vert_colors="red", vec_opacity=1.0, vector_style="line",
                                mesh_blending="opaque", mesh_color_override=None, mesh_opacity=1.0,
                                mesh_shading="none", markers=None, marker_colors="yellow", marker_size=5, img=None,
                                scale=None, img_opacity=0.5, marker_vectors=None, marker_vectors_color="red",
                                marker_vectors_length=100, marker_vector_width=2):
    print(">> Rendering mesh and colored vertices...")
    vec_pos, vec_dir = directors[:, :3], directors[:, 3:]
    if type(mesh_vert_colors) == str:
        mesh_vert_colors = np.tile(np.array(pltcolors.to_rgb(mesh_vert_colors)), (mesh.vertices.shape[0], 1))
    else:
        if type(mesh_vert_colors[0]) == str:
            mesh_vert_colors = np.array([pltcolors.to_rgb(c) for c in mesh_vert_colors])
        else:
            mesh_vert_colors = np.array(
                [tuple(int(hex_code.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)) for hex_code in
                 mesh_vert_colors])
    if mesh_color_override is not None:
        vert_colors = color_override.copy()
        vert_colors[:, 3] = 1.0

    if vec_edge_width is None:
        vec_edge_width = vec_length / 8
    viewer = napari.Viewer()
    if img is not None and scale is not None:
        viewer.add_image(img, opacity=img_opacity, scale=scale, rendering="mip", colormap="green")
    viewer.add_surface((mesh.vertices, mesh.faces), vertex_colors=mesh_vert_colors, shading=mesh_shading,
                       opacity=mesh_opacity, blending=mesh_blending)
    centered_x = vec_pos[:, 0] - 0.5 * vec_dir[:, 0] * vec_length
    centered_y = vec_pos[:, 1] - 0.5 * vec_dir[:, 1] * vec_length
    centered_z = vec_pos[:, 2] - 0.5 * vec_dir[:, 2] * vec_length
    centered_pos = np.column_stack((centered_x, centered_y, centered_z))
    viewer.add_vectors(
        data=np.stack((centered_pos, vec_dir), axis=1),
        edge_color=vec_colors,
        edge_width=vec_edge_width,
        length=vec_length,
        opacity=vec_opacity,
        vector_style=vector_style
    )
    if markers is not None:
        if type(marker_colors) != str:
            marker_colors = np.array(
                [tuple(int(hex_code.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)) for hex_code in
                 marker_colors])
        viewer.add_points(markers, name='Defects', shading="none", opacity=0.6, blending="opaque",
                          face_color=marker_colors, border_color=marker_colors, size=marker_size)

    if marker_vectors is not None:
        viewer.add_vectors(
            data=np.stack((marker_vectors[:, :3], marker_vectors[:, 3:]), axis=1),
            edge_color=marker_vectors_color, edge_width=marker_vector_width,
            length=marker_vectors_length, vector_style="arrow"
        )
    viewer.dims.ndisplay = 3
    napari.run()


def view_colored_verts_multiple(verts1, colors1, verts2, colors2, scale=None, img=None, ptsize=2, cmap_img="green",
                                blending="opaque", shading="none", opacity=0.2, use_orig_color=True):
    print(">> Rendering colored vertices...")
    viewer = napari.Viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale)
    if use_orig_color:
        colors_mapped1 = colors1.copy()
        colors_mapped2 = colors2.copy()
    else:
        colors_mapped1 = np.zeros((colors1.shape[0], 3))
        colors_mapped1[:, 1] = colors1
        colors_mapped2 = np.zeros((colors2.shape[0], 3))
        colors_mapped2[:, 1] = colors2
    viewer.add_points(verts1, name='Mesh1', shading=shading, opacity=opacity, blending=blending,
                      face_color=colors_mapped1, border_color=colors_mapped1, size=ptsize)
    viewer.add_points(verts2, name='Mesh2', shading=shading, opacity=opacity, blending=blending,
                      face_color=colors_mapped2, border_color=colors_mapped2, size=ptsize)
    viewer.dims.ndisplay = 3
    napari.run()


def view_colored_labels_3d(segmentation_3d, scale, img=None, img_opacity=0.5):
    viewer = napari.Viewer()
    if img is not None:
        viewer.add_image(img, scale=scale, opacity=img_opacity)
    viewer.add_labels(segmentation_3d, name='segmentation', scale=scale)
    viewer.dims.ndisplay = 3
    napari.run()
