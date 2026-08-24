"""
Visualisation Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################

import os.path
import matplotlib.pyplot as plt
import napari
import numpy as np
from matplotlib import colormaps
from matplotlib import colors as pltcolors
from matplotlib.colors import Normalize
import matplotlib.tri as tri
from matplotlib.transforms import offset_copy

plt.style.use("default")
font_size = 10.0
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "axes.grid": False,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out"
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
    # print(f">> Creating directory {directory} ...")
    os.makedirs(directory, exist_ok=True)


#######################
# 2D PLOTTING MODULES #
#######################


def plot_img(img, scale, unit, x_i=None, y_i=None, z_i=None, figsize=(8.5, 2.5), slice_line_alpha=0.5,
             cmap="Greens_r", dpi=300, thresh_mask=None, max_proj=False, meshes=None, mesh_normal_alpha=0.8,
             slice_depth=10, mesh_thick=0.2, mesh_alpha=0.2, thresh_alpha=0.8, mesh_colors=None,
             show_mesh_normals=False, mesh_interval=5, normal_scale=0.05, normal_interval=50,
             cmap_label="Fluorescence Intensity (a.u.)", title_digit_precision=2,
             manual_vminvmax=None, savefig="", hidefig=False,
             show_scalebar=True, scalebar_fontsize=5, slice_line_color="white"):
    print(f">> Plotting {'Max Projections' if max_proj else 'Slices'} ...")
    if z_i is None: z_i = img.shape[0] // 2
    if y_i is None: y_i = img.shape[1] // 2
    if x_i is None: x_i = img.shape[2] // 2

    if max_proj:
        show_mesh_normals = False
        scalebar_colour = "black"
        z_slice, y_slice, x_slice = np.max(img, axis=0), np.max(img, axis=1), np.max(img, axis=2)
        z_title, y_title, x_title = "Z Max Projection", "Y Max Projection", "X Max Projection"
    else:
        scalebar_colour = "white"
        z_slice, y_slice, x_slice = img[z_i, :, :], img[:, y_i, :], img[:, :, x_i]
        z_title = rf'Z$\approx${np.round(z_i * scale[0], decimals=title_digit_precision)}{unit}'
        y_title = rf'Y$\approx${np.round(y_i * scale[1], decimals=title_digit_precision)}{unit}'
        x_title = rf'X$\approx${np.round(x_i * scale[2], decimals=title_digit_precision)}{unit}'

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if manual_vminvmax is None:
        vmin, vmax = z_slice.min(), z_slice.max()
    else:
        vmin, vmax = manual_vminvmax
    axes[0].imshow(z_slice, cmap=cmap, aspect=scale[1] / scale[2], origin='upper', vmin=vmin, vmax=vmax)
    axes[1].imshow(y_slice, cmap=cmap, aspect=scale[0] / scale[2], origin='lower', vmin=vmin, vmax=vmax)
    axes[2].imshow(x_slice, cmap=cmap, aspect=scale[0] / scale[1], origin='lower', vmin=vmin, vmax=vmax)
    for ax, title in zip(axes, [z_title, y_title, x_title]):
        ax.set_title(title)

    if thresh_mask is not None:
        if max_proj:
            t_z, t_y, t_x = np.max(thresh_mask, axis=0), np.max(thresh_mask, axis=1), np.max(thresh_mask, axis=2)
        else:
            t_z, t_y, t_x = thresh_mask[z_i, :, :], thresh_mask[:, y_i, :], thresh_mask[:, :, x_i]

        axes[0].imshow(t_z > 0, cmap="Reds", aspect=scale[1] / scale[2], origin='upper', alpha=thresh_alpha)
        axes[1].imshow(t_y > 0, cmap="Reds", aspect=scale[0] / scale[2], origin='lower', alpha=thresh_alpha)
        axes[2].imshow(t_x > 0, cmap="Reds", aspect=scale[0] / scale[1], origin='lower', alpha=thresh_alpha)

    if meshes is not None:
        if mesh_colors is None:
            mesh_colors = ["red" for _ in range(len(meshes))]

        for i, mesh in enumerate(meshes):
            pts = mesh.vertices.copy()
            c_i = mesh_colors[i]

            def get_c(mask, idx, color_data):
                if isinstance(color_data, np.ndarray) and len(color_data) == len(pts):
                    return color_data[mask][idx]
                return color_data

            if max_proj:
                zmask = ymask = xmask = np.ones(len(pts), dtype=bool)
                idx_z = idx_y = idx_x = slice(None, None, mesh_interval)
            else:
                zmask = abs(pts[:, 0] - z_i * scale[0]) < slice_depth
                ymask = abs(pts[:, 1] - y_i * scale[1]) < slice_depth
                xmask = abs(pts[:, 2] - x_i * scale[2]) < slice_depth
                idx_z = idx_y = idx_x = slice(None)

            axes[0].scatter(pts[zmask, 2][idx_z] / scale[2], pts[zmask, 1][idx_z] / scale[1],
                            c=get_c(zmask, idx_z, c_i), s=mesh_thick, alpha=mesh_alpha)
            axes[1].scatter(pts[ymask, 2][idx_y] / scale[2], pts[ymask, 0][idx_y] / scale[0],
                            c=get_c(ymask, idx_y, c_i), s=mesh_thick, alpha=mesh_alpha)
            axes[2].scatter(pts[xmask, 1][idx_x] / scale[1], pts[xmask, 0][idx_x] / scale[0],
                            c=get_c(xmask, idx_x, c_i), s=mesh_thick, alpha=mesh_alpha)

            if show_mesh_normals:
                norms = mesh.vertex_normals.copy()
                q_idx = slice(None, None, normal_interval)

                for ax_idx, mask, ix, iy, sx, sy in [(0, zmask, 2, 1, scale[2], scale[1]),
                                                     (1, ymask, 2, 0, scale[2], scale[0]),
                                                     (2, xmask, 1, 0, scale[1], scale[0])]:
                    axes[ax_idx].quiver(pts[mask, ix][q_idx] / sx, pts[mask, iy][q_idx] / sy,
                                        norms[mask, ix][q_idx] / sx,
                                        norms[mask, iy][q_idx] / sy,
                                        color=get_c(mask, q_idx, c_i), alpha=mesh_normal_alpha, scale=normal_scale,
                                        angles='xy', scale_units='xy')

    if not max_proj:
        axes[0].axhline(y=y_i, linestyle="--", alpha=slice_line_alpha, color=slice_line_color)
        axes[0].axvline(x=x_i, linestyle="--", alpha=slice_line_alpha, color=slice_line_color)
        axes[1].axhline(y=z_i, linestyle="--", alpha=slice_line_alpha, color=slice_line_color)
        axes[1].axvline(x=x_i, linestyle="--", alpha=slice_line_alpha, color=slice_line_color)
        axes[2].axhline(y=z_i, linestyle="--", alpha=slice_line_alpha, color=slice_line_color)
        axes[2].axvline(x=y_i, linestyle="--", alpha=slice_line_alpha, color=slice_line_color)

    for i, (ax, xlab, ylab) in enumerate(zip(axes, ['X', 'X', 'Y'], ['Y', 'Z', 'Z'])):

        if show_scalebar:
            ax.set_xlabel(f"{xlab}")
            ax.set_ylabel(f"{ylab}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel(f"{xlab} ({unit})")
            ax.set_ylabel(f"{ylab} ({unit})")
            ax.set_xticks(np.linspace(0, img.shape[2] if i < 2 else img.shape[1], 5))
            ax.set_xticklabels(np.round(ax.get_xticks() * (scale[2] if i < 2 else scale[1]), 1))
            ax.set_yticks(np.linspace(0, img.shape[1] if i == 0 else img.shape[0], 5))
            ax.set_yticklabels(np.round(ax.get_yticks() * (scale[1] if i == 0 else scale[0]), 1))

    if show_scalebar:
        r = 0.2 * max(np.array(img.shape) * scale)
        auto_len = int(
            min([10 ** n * m for n in [np.floor(np.log10(r))] for m in [1, 2, 5, 10]], key=lambda x: abs(x - r)))
        axes_configs = [(axes[0], z_slice, scale[2], True, 'X', 'Y'),
                        (axes[1], y_slice, scale[2], False, 'X', 'Z'),
                        (axes[2], x_slice, scale[1], False, 'Y', 'Z')]

        for ax, s_data, x_sc, is_up, xlab, ylab in axes_configs:
            h, w = s_data.shape
            auto_len_px = auto_len / x_sc
            x_s = 0.98 * w - auto_len_px
            y_b = 0.97 * h if is_up else 0.05 * h
            ax.plot([x_s, x_s + auto_len_px], [y_b, y_b], color=scalebar_colour, lw=1.5)
            offset = 4
            trans = offset_copy(ax.transData, fig=fig, y=offset, units='points')
            ax.text(x_s + auto_len_px / 2, y_b, f"{auto_len} {unit}",
                    transform=trans, color=scalebar_colour, fontsize=scalebar_fontsize,
                    ha='center', va='center', fontweight='bold')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                        ax=axes, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label(cmap_label)

    if savefig != "":
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_maxproj_pts(verts, unit, cmap="Spectral", colors=None, interp_grid_n=100, figsize=(8.5, 5), savefig="",
                     dpi=300,
                     cmap_label="Signal (a.u.)", manual_vminmax=None, hidefig=False):
    print(">> Plotting max projection with fast triangulation...")

    if colors is None:
        colors = np.zeros(len(verts))

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

    for i, j, mask, x_data, y_data, xlabel, ylabel in panels:
        ax = axes[i, j]
        if np.any(mask):
            x, y, c = x_data[mask], y_data[mask], colors[mask]
            triang = tri.Triangulation(x, y)
            interp = tri.LinearTriInterpolator(triang, c)
            xi = np.linspace(x.min(), x.max(), interp_grid_n)
            yi = np.linspace(y.min(), y.max(), interp_grid_n)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interp(Xi, Yi)
            ax.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()),
                      origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_xlabel(f'{xlabel} ({unit})')
            ax.set_ylabel(f'{ylabel} ({unit})')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_visible(False)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes, orientation="vertical", fraction=0.02, pad=0.05
    )
    cbar.set_label(cmap_label)

    if savefig != "":
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_hist(array, ylabel="Frequency", title="", xlabel="", figsize=(3.5, 2.8), xlim=None, savefig="", dpi=300,
              density=True, bins=None, hidefig=False):
    plt.figure(figsize=figsize)
    plt.title(title)
    hist_kwargs = {
        "density": density,
        "align": "mid",
        "color": "black",
        "rwidth": 0.92,
    }
    if bins is not None:
        hist_kwargs["bins"] = bins
    plt.hist(array, **hist_kwargs)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig != "":
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_matrix(matrix, unit, colorbar=False, cmap="twilight", figsize=(3.5, 2.8), savefig="", dpi=300,
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
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_spherical_projection(phi, theta, intensities,
                              hexview=True, ptview=False, interp_grid_n=200,
                              cmap="Greens", vec_pos_phi=None, vec_pos_theta=None,
                              vec_dir_phi=None, vec_dir_theta=None, veccolor="red",
                              vec_manual_vminmax=None, vec_cmap="Spectral",
                              scale_factor=10.0, figsize=(8.2, 3.2), arrow_alpha=0.7,
                              vec_cmap_label="", vec_width=0.001, ptsize=2, alpha=1.0,
                              invert_y_axis=False,
                              cmap_label="Intensity Signal (a.u.)", manual_vminmax=None, savefig="", dpi=300,
                              hidefig=False, marker_idxs=None, marker_color="yellow",
                              marker_size=200, marker_alpha=0.9, marker_vec=None, marker_vec_scale=20,
                              marker_vec_width=0.005, marker_vec_color="red", xlabel="Phi (deg)", ylabel="Theta (deg)",
                              disable_all_colorbars=False, ):
    draw_vectors = all(x is not None for x in [vec_pos_phi, vec_pos_theta, vec_dir_phi, vec_dir_theta])
    vmin, vmax = (intensities.min(), intensities.max()) if manual_vminmax is None else manual_vminmax

    if draw_vectors and not isinstance(veccolor, str):
        v_min_val = np.min(veccolor) if vec_manual_vminmax is None else vec_manual_vminmax[0]
        v_max_val = np.max(veccolor) if vec_manual_vminmax is None else vec_manual_vminmax[1]
        vec_norm = Normalize(vmin=v_min_val, vmax=v_max_val)
    else:
        vec_norm = None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.10, 0.15, 0.50, 0.70])

    if hexview:
        triang = tri.Triangulation(phi, theta)
        interp = tri.LinearTriInterpolator(triang, intensities)
        xi = np.linspace(phi.min(), phi.max(), interp_grid_n)
        yi = np.linspace(theta.min(), theta.max(), interp_grid_n)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interp(Xi, Yi)
        ax.imshow(zi, extent=(phi.min(), phi.max(), theta.min(), theta.max()),
                  origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect='equal', alpha=alpha, interpolation='bilinear')
    if ptview:
        ax.scatter(phi, theta, c=intensities, cmap=cmap, s=ptsize, alpha=alpha, vmin=vmin, vmax=vmax)

    if draw_vectors:
        q_color = veccolor if isinstance(veccolor, str) else plt.cm.get_cmap(vec_cmap)(vec_norm(veccolor))
        ax.quiver(vec_pos_phi, vec_pos_theta, vec_dir_phi * scale_factor, vec_dir_theta * scale_factor,
                  pivot="middle", headlength=0, headwidth=0, headaxislength=0,
                  angles='xy', scale_units='xy', scale=1, color=q_color,
                  alpha=arrow_alpha, width=vec_width)

    if marker_idxs is not None:
        m_c = marker_color if isinstance(marker_color, str) else marker_color
        ax.scatter(phi[marker_idxs], theta[marker_idxs], c=m_c, s=marker_size, alpha=marker_alpha, edgecolors='none')

        if marker_vec is not None:
            mv_phi, mv_theta, mv_idxs = marker_vec
            ax.quiver(phi[mv_idxs], theta[mv_idxs], mv_phi * marker_vec_scale, mv_theta * marker_vec_scale,
                      angles='xy', scale_units='xy', scale=1,
                      color=marker_vec_color, alpha=marker_alpha, width=marker_vec_width)

    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(0, 180.0)

    if not disable_all_colorbars:
        cax_int = fig.add_axes([0.65, 0.15, 0.02, 0.70])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                     cax=cax_int, label=cmap_label)
        if draw_vectors and not isinstance(veccolor, str):
            cax_vec = fig.add_axes([0.77, 0.15, 0.02, 0.70])
            fig.colorbar(plt.cm.ScalarMappable(norm=vec_norm, cmap=plt.cm.get_cmap(vec_cmap)),
                         cax=cax_vec, label=vec_cmap_label)

    if savefig != "":
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_interp_grid(grid_x, grid_y, grid_z, theta=None, linelength=2, figsize=(3.5, 2.8), hidefig=False, savefig="",
                     dpi=300):
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

    if savefig != "":
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_dist_kymograph(distances, intensities, cmap, figsize, unit, savefig="", dpi=300, hidefig=False):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(normalise_range(intensities), aspect="auto", cmap=cmap,
                   extent=[0, distances.max(), len(intensities), 0])
    axes[0].set_ylabel("Sampling Point Index")
    axes[0].set_xlabel(f"Distance from min distance ({unit})")
    intensities_avg = np.average(intensities, axis=0)
    axes[1].plot(distances, intensities_avg / intensities_avg.max(), "o-")
    axes[1].set_ylabel("Average Intensity (a.u.)")
    axes[1].set_xlabel(f"Distance from min distance ({unit})")
    plt.tight_layout()
    if savefig != "":
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


def plot_dir_field(directors=None, t1_raw=None, t2_raw=None, normals=None, veclength=10, freq=1, figsize=(8, 6),
                   view_init=None, title=None, savefig="", dpi=300, veccolor="red", cmap_label="cmap_label",
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
        print(f">> Saving figure to {savefig} ...")
        create_figdir(os.path.dirname(savefig))
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
    if not hidefig:
        plt.show()
    else:
        plt.close()


#####################
# 3D RENDER MODULES #
#####################

def initialise_viewer():
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    viewer.scale_bar.visible = True
    viewer.axes.visible = True
    return viewer


def view_img(img_list, scale, color_list=None, title_list=None, opacity_list=None, gamma_list=None):
    print(">> Rendering image...")
    if opacity_list is None:
        opacity_list = np.ones(len(img_list))
    if gamma_list is None:
        gamma_list = np.ones(len(img_list))
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
    viewer = initialise_viewer()

    for i, img in enumerate(img_list):
        viewer.add_image(img, name=title_list[i], colormap=color_list[i], rendering="mip", scale=scale_list[i], opacity=
        opacity_list[i], gamma=gamma_list[i])

    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()


def view_mesh(mesh_list, mesh_colors=None, mesh_titles=None, mesh_opacities=None, mesh_shadings=None, vec_freq=15,
              img=None,
              img_cmap="green", scale=(1, 1, 1), vec_edge_width=None, vec_length=1, img_opacity=0.4,
              hide_vectors=True):
    print(">> Rendering mesh...")
    viewer = initialise_viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=img_cmap, rendering="mip", scale=scale, opacity=img_opacity)

    if mesh_colors is None:
        mesh_colors = ["white" for _ in range(len(mesh_list))]
    if mesh_titles is None:
        mesh_titles = ["Mesh" for _ in range(len(mesh_list))]
    if mesh_shadings is None:
        mesh_shadings = ["none" for _ in range(len(mesh_list))]

    if mesh_opacities is None:
        mesh_opacities = [0.4 for _ in range(len(mesh_list))]
    if vec_edge_width is None:
        vec_edge_width = vec_length / 6

    for i, mesh in enumerate(mesh_list):
        viewer.add_surface((mesh.vertices, mesh.faces), name=mesh_titles[i], shading=mesh_shadings[i],
                           opacity=mesh_opacities[i],
                           blending="translucent", colormap=mesh_colors[i])
        if not hide_vectors:
            viewer.add_vectors(
                data=np.stack((mesh.vertices[::vec_freq], mesh.vertex_normals[::vec_freq]), axis=1),
                edge_color=mesh_colors[i], edge_width=vec_edge_width, blending="translucent_no_depth",
                length=vec_length, opacity=1.0, name=f"{mesh_titles[i]}_normals"
            )
    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()


def view_colored_verts(verts, colors, scale=None, img=None, ptsize=2, cmap_img="green", blending="translucent",
                       shading="none", opacity=0.8, use_orig_color=True):
    print(">> Rendering coloured vertices...")
    viewer = initialise_viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale)
    if use_orig_color:
        colors_mapped = colors.copy()
    else:
        colors_mapped = np.zeros((colors.shape[0], 3))
        colors_mapped[:, 1] = colors
    viewer.add_points(verts, name='Mesh', shading=shading, opacity=opacity, blending=blending,
                      face_color=colors_mapped, border_color=colors_mapped, size=ptsize)
    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()


def view_colored_mesh(mesh, vert_colors="red", mesh_shading="none", mesh_opacity=1.0, img=None, scale=None,
                      img_opacity=1.0, cmap_img="green", color_override=None, mesh_blending="opaque", markers=None,
                      marker_colors="yellow", marker_size=5, add_hidden_vector=False):
    print(">> Rendering coloured mesh...")

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
    viewer = initialise_viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale, opacity=img_opacity,
                         blending="translucent")
    viewer.add_surface((mesh.vertices, mesh.faces), vertex_colors=vert_colors, shading=mesh_shading,
                       opacity=mesh_opacity, blending=mesh_blending)
    if markers is not None:
        viewer.add_points(markers, name='Defects', shading="none", opacity=0.6, blending="opaque",
                          face_color=marker_colors, border_color=marker_colors, size=marker_size)
    if add_hidden_vector:
        viewer.add_vectors(np.array([[[0, 0, 0], [0, 0, 1]]]), name="dummy", visible=False)
    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()


def view_colored_mesh_multiple(mesh_list, vert_colors_list=None, mesh_shading="none", mesh_opacity_list=None, img=None,
                               scale=None, img_opacity=1.0, cmap_img="green", color_override_list=None,
                               mesh_blending_list=None, markers=None, add_hidden_vector=False,
                               marker_colors="yellow", marker_size=5, name_list=None):
    print(">> Rendering multiple coloured meshes...")
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
    viewer = initialise_viewer()
    if img is not None:
        viewer.add_image(img, name="Image", colormap=cmap_img, rendering="mip", scale=scale, opacity=img_opacity,
                         blending="translucent")
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
    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()


def view_3d_vector_field_multiple(vec_pos, vec_dir, vec_colors, verts=None, verts_colors=None, edge_width=0.3,
                                  vec_length=10, vec_opacity=0.9, pts_size=1, pts_opacity=0.9, pts_blending="opaque",
                                  vector_style="line", img=None, scale=None, img_opacity=0.5, mesh=None,
                                  mesh_shading="flat", mesh_blending="opaque", centered_directors=True, vec_names=None):
    print(">> Rendering 3D vector/director field...")
    viewer = initialise_viewer()
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
    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()


def view_colored_mesh_dir_field(mesh, directors, vec_colors="red", vec_length=20, vec_edge_width=None,
                                mesh_vert_colors="red", vec_opacity=1.0, vector_style="line",
                                mesh_blending="opaque", mesh_color_override=None, mesh_opacity=1.0,
                                mesh_shading="none", markers=None, marker_colors="yellow", marker_size=5, img=None,
                                scale=None, img_opacity=0.5, marker_vectors=None, marker_vectors_color="red",
                                marker_vectors_length=100, marker_vector_width=2, center_vectors=True,
                                center_marker_vector=False, marker_vectors_style="arrow"):
    print(">> Rendering mesh and coloured vectors/directors...")
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
    viewer = initialise_viewer()
    if img is not None and scale is not None:
        viewer.add_image(img, opacity=img_opacity, scale=scale, rendering="mip", colormap="green")
    viewer.add_surface((mesh.vertices, mesh.faces), vertex_colors=mesh_vert_colors, shading=mesh_shading,
                       opacity=mesh_opacity, blending=mesh_blending)
    if center_vectors:
        centered_x = vec_pos[:, 0] - 0.5 * vec_dir[:, 0] * vec_length
        centered_y = vec_pos[:, 1] - 0.5 * vec_dir[:, 1] * vec_length
        centered_z = vec_pos[:, 2] - 0.5 * vec_dir[:, 2] * vec_length
        centered_pos = np.column_stack((centered_x, centered_y, centered_z))
    else:
        centered_pos = vec_pos.copy()

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
        if center_marker_vector:
            centered_marker_x = marker_vectors[:, 0] - 0.5 * marker_vectors[:, 3] * marker_vectors_length
            centered_marker_y = marker_vectors[:, 1] - 0.5 * marker_vectors[:, 4] * marker_vectors_length
            centered_marker_z = marker_vectors[:, 2] - 0.5 * marker_vectors[:, 5] * marker_vectors_length
            centered_marker_pos = np.column_stack((centered_marker_x, centered_marker_y, centered_marker_z))
        else:
            centered_marker_pos = marker_vectors[:, :3].copy()
        viewer.add_vectors(
            data=np.stack((centered_marker_pos, marker_vectors[:, 3:]), axis=1),
            edge_color=marker_vectors_color, edge_width=marker_vector_width,
            length=marker_vectors_length, vector_style=marker_vectors_style
        )
    viewer.dims.axis_labels = ("X (Z)", "Y (Y)", "Z (X)")
    napari.run()
