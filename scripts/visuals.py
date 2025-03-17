"""
Visualisation Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import numpy as np
import os.path
import shutil
import matplotlib.pyplot as plt
import napari
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib import colormaps
import imageio

#################################
# Dark/Light Mode for all Plots #
#################################
plt.style.use('dark_background')


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


def colour_dist(distances, middle_val, radial_points, mesh, radial_intensities, cut_z):
    print(">> Colouring based on distance...")
    norm_distances = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    cmap = sharp_blue_red(min_val=np.min(distances), intermediate_val=middle_val,
                          max_val=np.max(distances))
    # reshaped_radial_points = radial_points.reshape(-1, 3)
    intensities_flat = radial_intensities.filled(0).ravel()
    intensities_flat = normalise_range(intensities_flat)
    dist_color = cmap(norm_distances(np.tile(distances, radial_points.shape[0])))
    dist_color[:, :3] = dist_color[:, :3] * intensities_flat[:, None]
    dist_color[:, 3] = np.ones_like(intensities_flat) * 0.5

    dist_color_reshaped = dist_color.reshape(len(mesh.vertices), len(distances), 4)
    # reshaped_radial_points = reshaped_radial_points.reshape(len(mesh.vertices), len(distances), 3)

    dist_points = mesh.vertices

    dist_color = np.max(dist_color_reshaped, axis=1)
    if cut_z:
        cut_mask = dist_points[:, 0] >= np.max(dist_points[:, 0]) // 2
        cut_dist_points = dist_points[cut_mask]
        cut_dist_color = dist_color[cut_mask]
    else:
        cut_dist_points = dist_points
        cut_dist_color = dist_color
    return cut_dist_points, cut_dist_color, cmap


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


#######################
# 2D PLOTTING MODULES #
#######################

def plot_img(img, scale, unit, x_i=None, y_i=None, z_i=None, figsize=(18, 3), slice_line_alpha=0.8,
             cmap="Greens_r", dpi=200, thresh_mask=None, max_proj=False, meshes=None,
             slice_depth=10, mesh_thick=0.1, mesh_alpha=0.3, thresh_alpha=0.8, mesh_colors=None,
             cmap_label="Fluorescence Intensity (a.u.)", hide_img=False, title_digit_precision=2,
             manual_vminvmax=None, savefig=""):
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
    if not hide_img:
        plt.show()
    else:
        plt.close()


def plot_maxproj_pts(verts, unit, cmap="Spectral", colors=None, hexsize=600, figsize=(15, 8), savefig="", dpi=200,
                     cmap_label="Signal (a.u.)", manual_vminmax=None):
    print("Plotting max projection of coloured points...")
    if colors is None:
        print(f"No colors given, assuming uniform color!")
        colors = np.zeros_like(verts)
    if manual_vminmax is None:
        vmin, vmax = colors.min(), colors.max()
    else:
        vmin, vmax = manual_vminmax
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    mask = verts[:, 0] >= verts[:, 0].max() / 2
    axes[0, 0].hexbin(verts[:, 2][mask], verts[:, 1][mask],
                      C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_xlabel(f'X ({unit})')
    axes[0, 0].set_xticks(np.linspace(verts[:, 2][mask].min(), verts[:, 2][mask].max(), 3))
    axes[0, 0].set_ylabel(f'Y ({unit})')
    axes[0, 0].set_yticks(np.linspace(verts[:, 1][mask].min(), verts[:, 1][mask].max(), 3))

    mask = verts[:, 1] >= verts[:, 1].max() / 2
    axes[0, 1].hexbin(verts[:, 2][mask], verts[:, 0][mask],
                      C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_xlabel(f'X ({unit})')
    axes[0, 1].set_xticks(np.linspace(verts[:, 2][mask].min(), verts[:, 2][mask].max(), 3))
    axes[0, 1].set_ylabel(f'Z ({unit})')
    axes[0, 1].set_yticks(np.linspace(verts[:, 0][mask].min(), verts[:, 0][mask].max(), 3))

    mask = verts[:, 2] >= verts[:, 2].max() / 2
    axes[0, 2].hexbin(verts[:, 1][mask], verts[:, 0][mask],
                      C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 2].set_xlabel(f'Y ({unit})')
    axes[0, 2].set_xticks(np.linspace(verts[:, 1][mask].min(), verts[:, 1][mask].max(), 3))
    axes[0, 2].set_ylabel(f'Z ({unit})')
    axes[0, 2].set_yticks(np.linspace(verts[:, 0][mask].min(), verts[:, 0][mask].max(), 3))

    mask = verts[:, 0] < verts[:, 0].max() / 2
    axes[1, 0].hexbin(verts[:, 2][mask], verts[:, 1][mask],
                      C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_xlabel(f'X ({unit})')
    axes[1, 0].set_xticks(np.linspace(verts[:, 2][mask].min(), verts[:, 2][mask].max(), 3))
    axes[1, 0].set_ylabel(f'Y ({unit})')
    axes[1, 0].set_yticks(np.linspace(verts[:, 1][mask].min(), verts[:, 1][mask].max(), 3))

    mask = verts[:, 1] < verts[:, 1].max() / 2
    axes[1, 1].hexbin(verts[:, 2][mask], verts[:, 0][mask],
                      C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_xlabel(f'X ({unit})')
    axes[1, 1].set_xticks(np.linspace(verts[:, 2][mask].min(), verts[:, 2][mask].max(), 3))
    axes[1, 1].set_ylabel(f'Z ({unit})')
    axes[1, 1].set_yticks(np.linspace(verts[:, 0][mask].min(), verts[:, 0][mask].max(), 3))

    mask = verts[:, 2] < verts[:, 2].max() / 2
    axes[1, 2].hexbin(verts[:, 1][mask], verts[:, 0][mask],
                      C=colors[mask], gridsize=hexsize, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].set_xlabel(f'Y ({unit})')
    axes[1, 2].set_xticks(np.linspace(verts[:, 1][mask].min(), verts[:, 1][mask].max(), 3))
    axes[1, 2].set_ylabel(f'Z ({unit})')
    axes[1, 2].set_yticks(np.linspace(verts[:, 0][mask].min(), verts[:, 0][mask].max(), 3))
    axes[0, 0].set_aspect("equal")
    axes[0, 1].set_aspect("equal")
    axes[0, 2].set_aspect("equal")
    axes[1, 0].set_aspect("equal")
    axes[1, 1].set_aspect("equal")
    axes[1, 2].set_aspect("equal")
    axes[0, 0].set_xticklabels(np.round(axes[0, 0].get_xticks(), 2))
    axes[0, 1].set_xticklabels(np.round(axes[0, 0].get_xticks(), 2))
    axes[0, 2].set_xticklabels(np.round(axes[0, 0].get_xticks(), 2))
    axes[1, 0].set_xticklabels(np.round(axes[0, 0].get_xticks(), 2))
    axes[1, 1].set_xticklabels(np.round(axes[0, 0].get_xticks(), 2))
    axes[1, 2].set_xticklabels(np.round(axes[0, 0].get_xticks(), 2))

    axes[0, 0].set_yticklabels(np.round(axes[0, 0].get_yticks(), 2))
    axes[0, 1].set_yticklabels(np.round(axes[0, 0].get_yticks(), 2))
    axes[0, 2].set_yticklabels(np.round(axes[0, 0].get_yticks(), 2))
    axes[1, 0].set_yticklabels(np.round(axes[0, 0].get_yticks(), 2))
    axes[1, 1].set_yticklabels(np.round(axes[0, 0].get_yticks(), 2))
    axes[1, 2].set_yticklabels(np.round(axes[0, 0].get_yticks(), 2))

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                        ax=axes, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label(cmap_label)

    fig.subplots_adjust(right=0.85)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_hist(array, ylabel="Frequency", title="", figsize=(4, 3), xlim=None, savefig="", dpi=200, density=True,
              bins=None):
    print("Plotting histogram...")
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
    plt.show()


def plot_matrix(matrix, aspect=1, colorbar=False, cmap="twilight", figsize=(4, 3), savefig="", dpi=200, title=None,
                origin="lower", cmap_limits=None, cmap_label="", remove_axes=False):
    print(">> Plotting a matrix...")
    fig, ax = plt.subplots(figsize=figsize)
    if cmap_limits is not None:
        vmin, vmax = cmap_limits
    else:
        vmin, vmax = matrix.min(), matrix.max()
    ax.imshow(matrix, aspect=aspect, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        cmapable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cbar = fig.colorbar(cmapable, ax=ax, orientation="vertical", fraction=0.02, pad=0.05)
        cbar.set_label(cmap_label)
    if remove_axes:
        ax.axis('off')
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_mercator_project(mercator_x, mercator_y, intensities, hexview=True, ptview=False, hexgridsize=200, savefig="",
                          cmap="Greens", mercator_x_or=None, mercator_y_or=None, scale_factor=1.0, vec_freq=1,
                          figsize=(8, 4), arrow_alpha=1.0, ptsize=2, alpha=1.0, dpi=200, invert_y_axis=False,
                          aspect=2, cmap_label="Intensity Signal (a.u.)", manual_vminmax=None):
    print(">> Plotting the mercator projection...")
    if manual_vminmax is None:
        vmin, vmax = intensities.min(), intensities.max()
    else:
        vmin, vmax = manual_vminmax
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if hexview:
        ax.hexbin(mercator_x, mercator_y, C=intensities, cmap=cmap, gridsize=hexgridsize, alpha=alpha)
    if ptview:
        ax.scatter(mercator_x, mercator_y, c=intensities, cmap=cmap, s=ptsize, alpha=alpha)

    if mercator_x_or is not None and mercator_y_or is not None:
        ax.quiver(mercator_x[::vec_freq], mercator_y[::vec_freq], mercator_x_or[::vec_freq] * scale_factor,
                  mercator_y_or[::vec_freq] * scale_factor,
                  angles='xy', scale_units='xy', scale=1, color="red", alpha=arrow_alpha)
    if invert_y_axis:
        print("INVERTING Y AXIS !!")
        ax.invert_yaxis()
    ax.set_aspect(aspect)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                        ax=ax, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label(cmap_label)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_flattened_neighborhood(local_2d_projection, neighbors_intensities, figsize=(3, 3)):
    print(">> Plotting the flattened neighborhood...")
    plt.figure(figsize=figsize)
    plt.scatter(local_2d_projection[:, 0], local_2d_projection[:, 1], c=neighbors_intensities, cmap='Greys_r', s=50)
    plt.colorbar(label='Fluorescence Intensity (a.u.)')
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('Local X')
    plt.ylabel('Local y')
    plt.gca().set_aspect("equal")
    plt.title('Flattened Neighborhood')
    plt.show()


def plot_interp_grid(grid_x, grid_y, grid_z, points, theta=None, hide_pts=False, figsize=(4, 3),
                     linelength=2):
    print(">> Plotting the interpolated flattened neighborhood...")
    plt.figure(figsize=figsize)
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='Greys_r')
    if not hide_pts:
        plt.scatter(points[:, 0], points[:, 1], c="k", marker="x", s=1, alpha=0.2)

    plt.scatter(points[0, 0], points[0, 1], c="r", edgecolor='r', s=50)
    if theta is not None:
        x = np.array([-np.cos(theta), np.cos(theta)])
        y = np.array([-np.sin(theta), np.sin(theta)])
        plt.plot(linelength * x, linelength * y, color="red", linewidth=2)
        # plt.quiver(0, 0, np.cos(theta), np.sin(theta), color="red", scale=3)
    plt.colorbar(contour, orientation='vertical', label='Fluorescence Intensity (a.u.)')
    plt.title('Interp. Neighbourhood')
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Local X')
    plt.ylabel('Local y')
    plt.show()


def plot_vector_field_styles(x, y, u, v, defect, savefigpath="", figsize=(15, 5), dpi=200):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    print(">> Plotting the vector field with(out) defect...")
    axes[0].quiver(x, y, u, v, color='white', scale=20, pivot='middle', headaxislength=0)
    axes[0].set_title(f"Headless Vectors for {defect} Defect")
    axes[0].axis('equal')
    axes[0].set_facecolor('black')
    axes[1].quiver(x, y, u, v, color='white', scale=20, pivot='middle')
    axes[1].set_title(f"Vectors with Heads for {defect} Defect")
    axes[1].axis('equal')
    axes[1].set_facecolor('black')
    axes[2].streamplot(x, y, u, v, color='white', density=1, arrowstyle="-")
    axes[2].set_title(f"Streamlines for {defect} Defect")
    axes[2].axis('equal')
    axes[2].set_facecolor('black')
    plt.tight_layout()
    if savefigpath != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefigpath, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_vector_field(x, y, u, v, defect, savefigpath="", figsize=(3, 3), dpi=200):
    plt.figure(figsize=figsize)
    print(">> Plotting the vector field with(out) defect...")
    plt.quiver(x, y, u, v, color='white', scale=20, pivot='middle', headaxislength=0)
    plt.title(f"{defect} Defect")
    plt.tight_layout()
    if savefigpath != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefigpath, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_dist_kymograph(distances, plot_mask, plot_all, cmap, figsize, unit, savefig="", dpi=200):
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    data_min = np.nanmin(plot_mask)
    data_max = 0.9
    plot_mask = (plot_mask - data_min) / (data_max - data_min)
    plot_mask[plot_mask > 1.0] = 1.0
    axes[0].imshow(plot_mask, aspect="auto", cmap=cmap)
    avg_masked = np.average(plot_all, axis=0)
    axes[1].plot(distances, avg_masked, "o-")
    axes[0].set_ylabel("Sampling Point Index")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].set_ylabel("Average Intensity (a.u.)")
    axes[1].set_xlabel(f"Distance from Mesh ({unit})")
    avg_masked_mean = distances[np.argmax(np.average(plot_all, axis=0))]
    axes[1].axvline(x=avg_masked_mean, label=f"ARGMAX = {round(avg_masked_mean, 1)}")
    axes[1].legend()
    plt.tight_layout()
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_dir_field(directors=None, t1_raw=None, t2_raw=None, normals=None, veclength=10, freq=1, figsize=(8, 6),
                   view_init=None, title=None, savefig="", dpi=200, veccolor="red", cmap_label="cmap_label",
                   cmap="Spectral", show_axes=True, xlim=None, ylim=None, zlim=None, aspect="equal", hide_cmap=False,
                   marker=None, pt_label="marker_label", pt_size=100, pt_style="o", pt_color="yellow", pt_alpha=1.0,
                   manual_vminmax=None):
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
                  arrow_length_ratio=0.0, label="Director",
                  color=colormaps[cmap](norm(veccolor)), cmap=cmap)
        if not hide_cmap:
            sm = plt.cm.ScalarMappable(cmap=colormaps[cmap], norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=cmap_label)
    else:
        ax.quiver(centered_x, centered_y, centered_z, vx, vy, vz, color=veccolor, length=veclength,
                  arrow_length_ratio=0.0, label="Director")
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
    plt.show()


def plot_matrix_vectors(x, y, angle_field, matrix, veclength=1, vescale=50, title=None, figsize=(4, 3),
                        savefig="", dpi=200, vec_colors=None, matrix_cmap="Greys_r", vec_cmap="Spectral",
                        cbar_vector_label="", cbar_matrix_label="", matrix_origin="lower"):
    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        plt.title(title)
    vx = np.cos(angle_field)
    vy = np.sin(angle_field)
    centered_x = x - 0.5 * vx * veclength
    centered_y = y - 0.5 * vy * veclength
    img = ax.imshow(matrix, cmap=matrix_cmap, origin=matrix_origin)
    fig.colorbar(img, ax=ax, label=cbar_matrix_label)
    if vec_colors is not None:
        norm = plt.Normalize(vec_colors.min(), vec_colors.max())
        sm = plt.cm.ScalarMappable(cmap=vec_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=cbar_vector_label, fraction=0.046, pad=0.04)
        colors = sm.to_rgba(vec_colors)
    else:
        colors = "red"
    ax.quiver(centered_x, centered_y, vx, vy, color=colors, scale=vescale, pivot='middle', headaxislength=0)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_polar_hist(angles_deg, title="Polar Histogram of Theta Angles", figsize=(6, 6), savefig="", dpi=200, bins=30):
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
    plt.show()


def animate_img_slices(img, scale, unit, gifpath, gifsuffix, time_spacing=0.1, cmap="Greens", dpi=100,
                       cmap_label="Fluorescence Intensity (a.u.)", cleanup_frames=True, figsize=(5, 4), temp_path=None):
    print(">> Animating slices...")
    if temp_path is None:
        temp_path = os.path.join(gifpath, "temp")
    create_figdir(temp_path)
    slices = {
        "Z": (img.shape[0], lambda i: img[i, :, :], scale[1] / scale[2]),
        "Y": (img.shape[1], lambda i: img[:, i, :], scale[0] / scale[2]),
        "X": (img.shape[2], lambda i: img[:, :, i], scale[0] / scale[1])
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


def plot_cmap(cmap_custom, distances, midpoint, savefigpath="", dpi=200):
    print(">> Visualising the cmap...")
    min_val = np.min(distances)
    max_val = np.max(distances)
    norm_distances = plt.Normalize(vmin=np.min(distances), vmax=np.max(distances))
    cmap_custom = cmap_custom(min_val, midpoint, max_val)
    dist_color = cmap_custom(norm_distances(distances))
    fig, ax = plt.subplots(figsize=(3, 0.4))
    for i, color in enumerate(dist_color):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(distances))
    ax.set_ylim(0, 1)
    if savefig != "":
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefigpath, dpi=dpi, bbox_inches='tight')
    plt.show()


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

    viewer = napari.Viewer()
    for i, img in enumerate(img_list):
        viewer.add_image(img, name=title_list[i], colormap=color_list[i], rendering="mip", scale=scale, opacity=
        opacity_list[i])
    viewer.dims.ndisplay = 3
    napari.run()


def view_mesh(mesh_list, mesh_colors=None, mesh_titles=None, mesh_opacities=None, vec_freq=15, img=None,
              img_cmap="green", scale=(1, 1, 1), vec_edge_width=None, vec_length=1, img_opacity=0.4,
              hide_vectors=True):
    print(">> Rendering mesh...")
    viewer = napari.Viewer()
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
                           blending="translucent_no_depth", colormap=mesh_colors[i])
        if not hide_vectors:
            viewer.add_vectors(
                data=np.stack((mesh.vertices[::vec_freq], mesh.vertex_normals[::vec_freq]), axis=1),
                edge_color=mesh_colors[i], edge_width=vec_edge_width,
                length=vec_length, opacity=1.0, name=f"{mesh_titles[i]}_normals"
            )
    if img is not None:
        viewer.add_image(img, name="Image", colormap=img_cmap, rendering="mip", scale=scale, opacity=img_opacity)
    viewer.dims.ndisplay = 3
    napari.run()


def view_verts(verts, colors="white", opacity=1.0, shading="none", point_size=2, blending="opaque", ):
    print(">> Rendering vertices...")
    viewer = napari.Viewer()
    viewer.add_points(verts, name='Mesh', shading=shading, opacity=opacity, blending=blending,
                      face_color=colors, border_color=colors, size=point_size)
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


def view_mesh_and_verts(mesh_list, mesh_colors, verts, verts_colors, vec_freq=15, img=None, img_cmap="green",
                        scale=(1, 1, 1), vec_edge_width=1, vec_length=20, img_opacity=0.4, hide_vectors=True,
                        use_orig_color=True, verts_size=3):
    print(">> Rendering mesh and colored vertices...")
    viewer = napari.Viewer()

    for i, mesh in enumerate(mesh_list):
        viewer.add_surface((mesh.vertices, mesh.faces), shading='none', opacity=0.3,
                           blending="translucent_no_depth", colormap=mesh_colors[i])
        if not hide_vectors:
            viewer.add_vectors(
                data=np.stack((mesh.vertices[::vec_freq], mesh.vertex_normals[::vec_freq]), axis=1),
                edge_color=mesh_colors[i], edge_width=vec_edge_width,
                length=vec_length, opacity=1.0
            )
    if img is not None:
        viewer.add_image(img, name="Image", colormap=img_cmap, rendering="mip", scale=scale, opacity=img_opacity)

    if use_orig_color:
        colors_mapped = verts_colors.copy()
    else:
        colors_mapped = np.zeros((verts_colors.shape[0], 3))
        colors_mapped[:, 1] = colors
    viewer.add_points(verts, name='Mesh', shading="none", opacity=0.6, blending="opaque",
                      face_color=colors_mapped, border_color=colors_mapped, size=verts_size)
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


def view_mesh_dir_field(mesh_list, directors, vec_colors, verts=None, verts_colors=None, vec_length=20,
                        mesh_colors=None, vec_edge_width=None, vec_opacity=1.0, vector_style="line", pts_size=1,
                        pts_opacity=1.0, pts_blending="opaque", ):
    print(">> Rendering mesh and colored vertices...")
    vec_pos, vec_dir = directors[:, :3], directors[:, 3:]
    if mesh_colors is None:
        mesh_colors = ["white" for _ in range(len(mesh_list))]
    if vec_edge_width is None:
        vec_edge_width = vec_length / 8
    viewer = napari.Viewer()
    for i, mesh in enumerate(mesh_list):
        viewer.add_surface((mesh.vertices, mesh.faces), shading='none', opacity=0.3,
                           blending="translucent_no_depth", colormap=mesh_colors[i])
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
    if verts is not None and verts_colors is not None:
        viewer.add_points(
            data=verts,
            border_color=verts_colors,
            face_color=verts_colors,
            size=pts_size,
            opacity=pts_opacity,
            blending=pts_blending
        )
    viewer.dims.ndisplay = 3
    napari.run()


def view_neighbourhood(points, intensities, neighbours, sel_idx, mesh_opacity=0.3, neighbour_opacity=1.0, pt_size=5,
                       mesh_size=2, plot_coord_system=False, mesh_blending="opaque"):
    print(">> Rendering the neighbourhood of a particle...")
    viewer = napari.Viewer()
    intensities_mapped = np.zeros((intensities.shape[0], 3))
    intensities_mapped[:, 1] = intensities
    viewer.add_points(points, name='Mesh', shading='none', opacity=mesh_opacity, blending=mesh_blending,
                      face_color=intensities_mapped, border_color=intensities_mapped, size=mesh_size)
    viewer.add_points(neighbours[sel_idx], face_color="yellow", border_color="yellow", size=pt_size,
                      opacity=neighbour_opacity, blending="opaque")

    viewer.add_points(points[sel_idx], face_color="red", border_color="red", size=pt_size)
    if plot_coord_system:
        viewer.add_vectors(
            np.stack((np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),
                     axis=1), edge_color=["red", "green", "blue"], length=100, edge_width=1)
    viewer.dims.ndisplay = 3
    napari.run()
