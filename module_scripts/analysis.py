"""
Analysis Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import orientationpy as op
import pyvista as pv
import scipy.sparse as sp
import trimesh
import zarr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, binary_fill_holes
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from skimage import measure
from skimage.filters import threshold_yen
from sklearn.neighbors import KDTree as KDTreeSklearn
from tifffile import TiffFile

from module_scripts.datahandler import load_array
from module_scripts.visuals import plot_dist_kymograph, plot_interp_grid, plot_matrix


#################
# BASIC MODULES #
#################
def normalise_range(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


def coord_search_neighbours(verts, k, custom_probes=None, n_process=8, debug=False, return_dists=False):
    if debug:
        print(f">> Searching {k} neighbours ...")
    tree = KDTree(verts)
    if custom_probes is not None:
        dists, idxs = tree.query(custom_probes, k=k, workers=n_process)
    else:
        dists, idxs = tree.query(verts, k=k)
    if return_dists:
        return idxs, dists
    else:
        return idxs


def coord_search_radius(verts, r, custom_probes=None, return_dists=False, debug=False):
    if debug:
        print(f">> Searching neighbours within radius {r} ...")
    tree = KDTreeSklearn(verts)
    if custom_probes is not None:
        idxs, dists = tree.query_radius(custom_probes, r=r, return_distance=True, sort_results=True)
    else:
        idxs, dists = tree.query_radius(verts, r=r, return_distance=True, sort_results=True)
    if return_dists:
        return list(idxs), list(dists)
    else:
        return list(idxs)


def rot3dmatrix(alpha, beta, gamma):
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]])
    rot_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])
    rot_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                      [np.sin(gamma), np.cos(gamma), 0],
                      [0, 0, 1]])
    return rot_x @ rot_y @ rot_z


def otimes(a, b):
    return np.stack([
        np.outer(a, a),
        np.outer(a, b),
        np.outer(b, a),
        np.outer(b, b)]).reshape(2, 2, 3, 3)


def gmetric(e1, e2):
    return np.array([[np.dot(e1, e1), np.dot(e1, e2)],
                     [np.dot(e2, e1), np.dot(e2, e2)]])


def covariant2contravariant(e1_cov, e2_cov):
    g = gmetric(e1_cov, e2_cov)
    g_inv = np.linalg.inv(g)
    e1e2_contr = np.matmul(g_inv, np.row_stack((e1_cov, e2_cov)))
    return e1e2_contr[0], e1e2_contr[1]


def tensprod(a, b):
    return np.tensordot(a, b, axes=2)


def create_tangential_basis(normals, first_choice_axis=None, second_choice_axis=None, hide_output=True):
    if first_choice_axis is None:
        first_choice_axis = np.array([1, 0, 0])
    if second_choice_axis is None:
        second_choice_axis = np.array([0, 1, 0])
    if not hide_output:
        print(
            f">> Creating tangential basis using cross product of normals with axis {first_choice_axis} or {second_choice_axis} if parallel...")
    cross_basis_vector = np.where(
        np.all(np.isclose(np.cross(normals, first_choice_axis), 0), axis=1)[:, None],
        second_choice_axis,
        first_choice_axis
    )
    t1_raw = np.cross(normals, cross_basis_vector)
    t1_raw /= np.linalg.norm(t1_raw, axis=1, keepdims=True)
    t2_raw = np.cross(normals, t1_raw)
    t2_raw /= np.linalg.norm(t2_raw, axis=1, keepdims=True)
    return t1_raw, t2_raw


def rescale_val_xyz(val, scale, debug=False):
    zscale, xscale, yscale = scale
    xyscale = (xscale + yscale) / 2
    if zscale > xyscale:
        if debug:
            print(">> Adjusting due to better xy resolution than z resolution...")
        return val * xyscale / zscale, val, val
    else:
        if debug:
            print(">> Adjusting due to better z resolution than xy resolution...")
        return val, val * zscale / xyscale, val * zscale / xyscale


def filter_valid_patches(verts, idxs_neigh, factor=0.1):
    if isinstance(idxs_neigh, list):
        valid_patches = []
        valid_idxs = []
        for i, neigh in enumerate(idxs_neigh):
            if len(neigh) == 0:
                continue
            verts_patch = verts[neigh]
            centroid = np.mean(verts_patch, axis=0, keepdims=True)
            reldists = np.linalg.norm(verts_patch - centroid, axis=1)
            if reldists[0] < factor * np.max(reldists):
                valid_patches.append(neigh)
                valid_idxs.append(i)
        print(f"{len(valid_patches)} valid, {len(idxs_neigh) - len(valid_patches)} invalid patches")
        return valid_patches, np.array(valid_idxs)
    centroids_patches = np.mean(verts[idxs_neigh], axis=1)[:, np.newaxis]
    reldists = np.linalg.norm(verts[idxs_neigh] - centroids_patches, axis=2)
    center_reldist = reldists[:, 0]
    keep_mask = center_reldist < factor * np.max(reldists, axis=1)
    valid_patch_idxs = idxs_neigh[keep_mask]
    valid_idxs = np.argwhere(keep_mask)[:, 0]
    print(f"{len(valid_patch_idxs)} valid, {len(centroids_patches) - len(valid_patch_idxs)} invalid ! ")
    return valid_patch_idxs, valid_idxs


def filter_normal_validity(mesh, idxs_sel, k=10, threshold=0.2):
    normals_all = mesh.vertex_normals
    normals = normals_all[idxs_sel]
    valid_norm_mask = np.linalg.norm(normals, axis=1) > 0
    print(f"{np.sum(~valid_norm_mask)} zero-norm normals added to exclusion list !")
    normals_valid = normals[valid_norm_mask]
    normals_valid /= np.linalg.norm(normals_valid, axis=1, keepdims=True)
    neigh_idxs = coord_search_neighbours(verts=mesh.vertices, k=k, n_process=8,
                                         custom_probes=mesh.vertices[idxs_sel[valid_norm_mask]])
    normals_valid_neigh_avg = np.mean(normals_all[neigh_idxs], axis=1)
    normals_valid_neigh_avg /= np.linalg.norm(normals_valid_neigh_avg, axis=1, keepdims=True)
    valid_orth_mask = np.sum(normals_valid * normals_valid_neigh_avg, axis=1) > threshold
    print(f"{np.sum(~valid_orth_mask)} tangential normals added to exclusion list !")
    return idxs_sel[valid_norm_mask][valid_orth_mask]


def expand_2d_array(array, num):
    if type(num) is not int:
        return np.repeat(np.repeat(array, num[0], axis=0), num[1], axis=1)
    else:
        return np.repeat(np.repeat(array, num, axis=0), num, axis=1)


############################
# IMAGE PROCESSING MODULES #
############################


def get_tiff_scaling(tif):
    # ======== XY resolution ========
    try:
        xres = tif.pages[0].tags['XResolution'].value
        yres = tif.pages[0].tags['YResolution'].value
        xscale = 1 / float(xres[0] / xres[1])
        yscale = 1 / float(yres[0] / yres[1])
        print(f"Found xscale = {xscale}, yscale = {yscale} !")
    except:
        xscale, yscale = 1.0, 1.0
        print(f"XY scaling absent... using default {xscale, yscale} instead !")

    # ======== Z resolution ========
    try:
        zscale = float(next(line for line in tif.pages[0].tags.get('ImageDescription', None).value.splitlines() if
                            line.startswith("spacing=")).split('=')[1])
        print(f"Found zscale = {zscale} !")
    except:
        zscale = 1.0
        print(f"Z spacing absent... using default {zscale} instead !")
    img_scale = (zscale, yscale, xscale)
    return img_scale


def load_img_unit(path, default_unit="um"):
    if not os.path.exists(path):
        print(f"[!] Image does not exist, aborting !")
        return None
    with TiffFile(path) as tif:
        try:
            unit = next(line for line in tif.pages[0].tags.get('ImageDescription', None).value.splitlines() if
                        line.startswith("unit=")).split('=')[1]
            if unit == "micron":
                unit = "um"
            print(f"Found unit = {unit} !")
            return unit
        except:
            print(f"[!] Could not find unit, resorting to standard {default_unit}...")
            return default_unit


def load_img_dimensions(path):
    print(f">> Loading {path}...")
    if not os.path.exists(path):
        print(f"[!] Image does not exist, aborting !")
        return None

    with TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        axis_map = {ax: i for i, ax in enumerate(axes)}
        dims = {ax: shape[axis_map[ax]] if ax in axis_map else 1 for ax in ['T', 'Z', 'C', 'Y', 'X']}
        print(', '.join(f"{k} = {v}" for k, v in dims.items()))
        return dims


def load_img_scaling(path):
    print(f">> Loading {path}...")
    if not os.path.exists(path):
        print(f"[!] Image does not exist, aborting !")
        return None

    with TiffFile(path) as tif:
        img_scale = get_tiff_scaling(tif)
        return img_scale


def load_img_virtual(path, norm_vals=False, t_sel_idx=0, c_sel_idx=0, custom_scaling=None,
                     reduce_xy=1, reduce_z=1):
    print(f">> Loading {path}...")
    if not os.path.exists(path):
        print(f"[!] Image does not exist, aborting !")
        return None
    if os.path.isdir(path):
        print(f"[!] You gave the NEMO path, not the .tif path !")
        return None
    with TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        axis_map = {ax: i for i, ax in enumerate(axes)}
        print(f"Found {series.ndim}D {axes} Hyperstack | {shape} | {round(series.nbytes / 1e6, 2)} MB !")

        # ======== Load Scaling ========
        img_scale = get_tiff_scaling(tif)
        print(f"Found image scale = {img_scale} !")

        # ======== Load Unit ========
        img_unit = load_img_unit(path=path)

        # ======== Extract single z-stack ========
        T = shape[axis_map['T']] if 'T' in axis_map else 1
        C = shape[axis_map['C']] if 'C' in axis_map else 1

        if t_sel_idx >= T or c_sel_idx >= C or t_sel_idx < 0 or c_sel_idx < 0:
            print(
                f"Invalid selected time point ({t_sel_idx} / max {T - 1}) or channel ({c_sel_idx} / max {C - 1}), aborting !")
            return None
        print(f">> Selecting Z-Stack at T={t_sel_idx} C={c_sel_idx} ...")

        store = series.aszarr(level=0)
        arr = zarr.open(store, mode='r')
        key = [slice(None)] * arr.ndim
        for ax, idx in axis_map.items():
            if ax == 'T':
                key[idx] = t_sel_idx
            elif ax == 'C':
                key[idx] = c_sel_idx
            elif ax == 'Z':
                key[idx] = slice(None)

        img_raw = np.asarray(arr[tuple(key)])
        img_dim = img_raw.shape

        if 'Z' not in axis_map:
            print(f"[!] 2D Image, expanding to 3D...")
            img_raw = img_raw[np.newaxis, ...]
            img_dim = img_raw.shape

    # ======== Optional: Scaling Override ========
    if custom_scaling is not None:
        print(f">> Overwriting scaling with {custom_scaling}...")
        img_scale = custom_scaling

    # ======== Reduce resolution if needed by averaging ========
    if reduce_z > 1 or reduce_xy > 1:
        # Dynamically get current shape (now guaranteed 3D: Z, Y, X)
        curr_z, curr_y, curr_x = img_raw.shape

        if reduce_xy > 1:
            print(f">> Reducing XY resolution by averaging {reduce_xy} pixels...")
            trimmed_y = (curr_y // reduce_xy) * reduce_xy
            trimmed_x = (curr_x // reduce_xy) * reduce_xy
            img_raw = img_raw[:, :trimmed_y, :trimmed_x]

            new_shape = (
                curr_z,
                trimmed_y // reduce_xy,
                reduce_xy,
                trimmed_x // reduce_xy,
                reduce_xy,
            )
            img_raw = img_raw.reshape(new_shape).mean(axis=(2, 4))
            img_scale = (img_scale[0], reduce_xy * img_scale[1], reduce_xy * img_scale[2])

        # Refresh shape info for Z-reduction step
        curr_z, curr_y, curr_x = img_raw.shape

        if reduce_z > 1 and curr_z > 1:
            print(f">> Reducing Z resolution by averaging {reduce_z} pixels...")
            trimmed_z = (curr_z // reduce_z) * reduce_z
            img_raw = img_raw[:trimmed_z, :, :]
            new_shape = (
                trimmed_z // reduce_z,
                reduce_z,
                curr_y,
                curr_x,
            )
            img_raw = img_raw.reshape(new_shape).mean(axis=1)
            img_scale = (reduce_z * img_scale[0], img_scale[1], img_scale[2])

        img_dim = img_raw.shape
        print(f"Reduced shape = {img_dim} | scale = {img_scale} | size = {round(img_raw.nbytes / 1e6, 2)} MB !")

    # ======== Norm values if needed ========
    if norm_vals:
        print(f">> Norming intensities...")
        img_raw = normalise_range(img_raw)

    print(f"Loaded T={t_sel_idx} C={c_sel_idx} Z-stack ({round(img_raw.nbytes / 1e6, 2)} MB) !")
    return img_raw, img_dim, img_scale, img_unit


def gaussian_blur(img, sigma, renorm):
    if sigma is None:
        img_blur = img
    else:
        print(f">> Applying 3D Gaussian blur with sigma = {sigma}...")
        img_blur = gaussian_filter(img, sigma=sigma)

    if renorm:
        print(f">> Re-Norming intensities...")
        return normalise_range(img_blur)
    else:
        return img_blur


def yen_thresh(img):
    thresh_val = threshold_yen(img)
    return thresh_val


def thresh_img(img, thresh, inverse=True, keep_values_above=False):
    print(f">> Applying threshold {thresh}...")
    img_thresh = np.copy(img)
    if inverse:
        mask = img < thresh
    else:
        mask = img > thresh
    img_thresh[mask] = 0
    if not keep_values_above:
        img_thresh[~mask] = 1
    return img_thresh


def fill_holes_img(img):
    print(f">> Filling holes in img...")
    return binary_fill_holes(img).astype(float)


#############################
# MESH SEGMENTATION MODULES #
#############################
def marching_cubes(img, scale, level, step_size, allow_degenerate=False):
    print(f">> Extracting mesh(es) using marching cubes algorithm with box size {step_size}...")
    verts, faces, normals, _ = measure.marching_cubes(volume=img, level=level, spacing=scale,
                                                      step_size=step_size, allow_degenerate=allow_degenerate)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=-normals)
    return mesh


def sel_submesh(mesh, mask):
    print(f">> Identifying surfaces using mask...")
    verts, faces, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
    original_indices = np.where(mask)[0]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(original_indices)}
    masked_verts = verts[mask]
    masked_faces = np.vectorize(index_map.get)(faces[np.all(mask[faces], axis=1)])
    masked_normals = normals[mask]
    masked_mesh = trimesh.Trimesh(vertices=masked_verts, faces=masked_faces, vertex_normals=masked_normals)
    return masked_mesh


def find_connected_meshes(mesh):
    print(f">> Searching for connected component meshes...")
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
    components = []
    for i in range(labels.max() + 1):
        face_indices = (labels == i)
        submesh = mesh.submesh([face_indices], append=True)
        components.append(submesh)
    print(f">> Found {len(components)} connected components!")
    return components


###########################
# MESH PROCESSING MODULES #
###########################

def taubin_smooth_mesh(mesh, n_iter=100, pass_band=0.01, recalc_normals=True):
    print(f">> Taubin smoothing mesh with {n_iter} iterations and pass band {pass_band}...")
    pvmesh = pv.PolyData()
    pvmesh.points = mesh.vertices
    pvmesh.faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).astype(int)
    smooth_pvmesh = pvmesh.smooth_taubin(n_iter=n_iter, pass_band=pass_band)
    smooth_trimesh = trimesh.Trimesh(vertices=smooth_pvmesh.points, faces=smooth_pvmesh.faces.reshape((-1, 4))[:, 1:],
                                     face_normals=smooth_pvmesh.face_normals)
    if recalc_normals:
        smooth_trimesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
            vertex_count=len(smooth_trimesh.vertices), faces=smooth_trimesh.faces,
            face_normals=smooth_trimesh.face_normals,
            face_angles=smooth_trimesh.face_angles)
    return smooth_trimesh


def scale_mesh(mesh, distance):
    print(f">> Scaling mesh ...")
    mesh_scaled = mesh.copy()
    mesh_scaled.vertices += mesh.vertex_normals * distance
    mesh_scaled.vertex_normals = trimesh.geometry.weighted_vertex_normals(vertex_count=len(mesh_scaled.vertices),
                                                                          faces=mesh_scaled.faces,
                                                                          face_normals=mesh_scaled.face_normals,
                                                                          face_angles=mesh_scaled.face_angles)
    return mesh_scaled


########################
# MESH FITTING MODULES #
########################
def sphere_func(params, points):
    x0, y0, z0, r = params
    return np.linalg.norm(points - [x0, y0, z0], axis=1) - r


def fit_sphere(points):
    center = np.mean(points, axis=0)
    r0 = np.mean(np.linalg.norm(points - center, axis=1))
    return least_squares(sphere_func, x0=[*center, r0], args=(points,)).x


def ellipsoid_func(params, points):
    x0, y0, z0, a, b, c, alpha, beta, gamma = params
    shifted_points = points - np.array([x0, y0, z0])
    rotmatrix = rot3dmatrix(alpha=alpha, beta=beta, gamma=gamma)
    rotated_points = shifted_points @ rotmatrix.T
    rho = (rotated_points[:, 0] / a) ** 2 + (rotated_points[:, 1] / b) ** 2 + (rotated_points[:, 2] / c) ** 2
    residuals = (rho - 1)
    return residuals


def fit_ellipsoid(points, fit_rotation=True, fixed_angles=None):
    if fixed_angles is None:
        fixed_angles = [0, 0, 0]
    print(f">> Fitting ellipsoid to {len(points)} pts...")
    x0, y0, z0 = np.mean(points, axis=0)
    a, b, c = np.ptp(points, axis=0) / 2
    print(f"GUESS = x0 {x0}, y0 {y0}, z0 {z0}")
    print(f"GUESS = a {a}, b = {b}, c = {c}")
    if fit_rotation:
        initial_params = [x0, y0, z0, np.abs(a), np.abs(b), np.abs(c), 0, 0, 0]
        bounds = (
            [-np.inf, -np.inf, -np.inf, 0, 0, 0, -np.pi / 2, -np.pi / 2, -np.pi / 2],  # lower bound
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi / 2, np.pi / 2, np.pi / 2]  # upper bound
        )
    else:
        print(f"Fixing the angle to angles {fixed_angles} !")
        initial_params = [x0, y0, z0, np.abs(a), np.abs(b), np.abs(c)]
        bounds = (
            [-np.inf, -np.inf, -np.inf, 0, 0, 0],  # lower bound
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]  # upper bound
        )

    def ellipsoid_func_reduced(params, pts):
        if fit_rotation:
            return ellipsoid_func(params, pts)
        else:
            return ellipsoid_func(np.concatenate([params, fixed_angles]), pts)

    fit_result = least_squares(fun=ellipsoid_func_reduced, x0=initial_params, args=(points,), bounds=bounds)
    fit_params = fit_result.x
    if not fit_result.success:
        print("WARNING: Least squares optimization did not converge!", fit_result.message)
    if not fit_rotation:
        fit_params = np.concatenate([fit_params, fixed_angles])
    x0, y0, z0, a, b, c, alpha, beta, gamma = fit_params
    print(f"x0 = {round(x0, 1)}, y0 = {round(y0, 1)}, z0 = {round(z0, 1)}")
    print(f"a = {round(a, 1)}, b = {round(b, 1)}, c = {round(c, 1)}")
    if fit_rotation:
        print(
            f"alpha = {round(np.degrees(alpha), 1)}, beta = {round(np.degrees(beta), 1)}, gamma = {round(np.degrees(gamma), 1)}")
    return fit_params


def generate_ellipsoid(params, num_points):
    print(f">> Creating ellipsoid mesh....")
    x0, y0, z0, a, b, c, alpha, beta, gamma = params

    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)
    ellipsoid_points = np.column_stack((x, y, z))

    rotmatrix = rot3dmatrix(alpha=alpha, beta=beta, gamma=gamma)
    rotated_ellipsoid_points = ellipsoid_points @ rotmatrix

    local_normals = np.column_stack((x / (a ** 2), y / (b ** 2), z / (c ** 2)))
    normals = local_normals @ rotmatrix
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    x_shifted = rotated_ellipsoid_points[:, 0] + x0
    y_shifted = rotated_ellipsoid_points[:, 1] + y0
    z_shifted = rotated_ellipsoid_points[:, 2] + z0
    vertices = np.column_stack((x_shifted, y_shifted, z_shifted))
    print(f"Created ellipsoid with {vertices.shape[0]} unique points!")
    return vertices, normals


#########################
# MESH ANALYSIS MODULES #
#########################
def curvature_by_srf_fit(mesh, num_sample, patch_size=20, patch_mode="nearest", filter_boundary=False, debug=False,
                         gauss_crop_range=None, mean_crop_range=None, boundary_excl_factor=0.1,
                         use_original_vertices=False, custom_basis=None):
    print(f">> Calculating curvature for {num_sample}/{len(mesh.vertices)} vertices of mesh...")
    random_idxs = np.random.choice(np.arange(mesh.vertices.shape[0]), size=num_sample)
    if use_original_vertices:
        random_idxs = np.arange(mesh.vertices.shape[0])
    verts = mesh.vertices[random_idxs]
    N = len(verts)
    normals = mesh.vertex_normals[random_idxs]
    if patch_mode == "radius":
        neigh_idxs = coord_search_radius(verts=verts, r=patch_size, debug=False)
        neigh_verts = [verts[idxs] for idxs in neigh_idxs]
        neigh_normals = [normals[idxs] for idxs in neigh_idxs]
    elif patch_mode == "nearest":
        neigh_idxs = coord_search_neighbours(verts=verts, k=patch_size, debug=False)
        neigh_verts = verts[neigh_idxs]
        neigh_normals = normals[neigh_idxs]
    else:
        print(f"[!] Unknown patch type: {patch_mode}")
        return None

    if filter_boundary:
        neigh_idxs, valid_idxs = filter_valid_patches(verts=verts, idxs_neigh=neigh_idxs, factor=boundary_excl_factor)
        N = len(neigh_idxs)
        random_idxs = random_idxs[valid_idxs]
    if custom_basis is not None:
        print("[!] Using custom basis for curvature...")
        tan_x_cov, tan_y_cov = custom_basis[0][random_idxs], custom_basis[1][random_idxs]
    else:
        print("[!] Using arbitrary tangential basis for curvature...")
        tan_x_cov, tan_y_cov = create_tangential_basis(normals=normals, hide_output=True)
    g = np.array([gmetric(tan_x_cov[i], tan_y_cov[i]) for i in range(N)])
    g_inv = np.array([np.linalg.inv(g[i]) for i in range(N)])
    inter_calc = np.array(
        [covariant2contravariant(tan_x_cov[i], tan_y_cov[i]) for i in range(N)])
    tan_x_contr, tan_y_contr = inter_calc[:, 0, :], inter_calc[:, 1, :]
    C_gauss, C_mean, C_tensors, C_tensors_mixed = [], [], [], []
    for i in range(N):
        C = np.zeros((2, 2))
        z_nbs, xi_nbs, eta_nbs = [], [], []
        n_neigh = len(neigh_verts[i]) if isinstance(neigh_verts, list) else neigh_verts.shape[1]
        for j in range(1, n_neigh):
            if isinstance(neigh_verts, list):
                drvec = neigh_verts[i][j] - neigh_verts[i][0]
                n0 = neigh_normals[i][0]
            else:
                drvec = neigh_verts[i, j] - neigh_verts[i, 0]
                n0 = neigh_normals[i, 0]
            z_nb = np.dot(drvec, n0)
            z_nbs.append(z_nb)
            xi_nb = np.dot(drvec, tan_x_contr[i])
            xi_nbs.append(xi_nb)
            eta_nb = np.dot(drvec, tan_y_contr[i])
            eta_nbs.append(eta_nb)
        if len(xi_nbs) < 3:
            C_gauss.append(np.nan)
            C_mean.append(np.nan)
            C_tensors.append(np.full((2, 2), np.nan))
            C_tensors_mixed.append(np.full((2, 2), np.nan))
            continue
        S = np.array([[xi ** 2, xi * eta, eta ** 2] for xi, eta in zip(xi_nbs, eta_nbs)])
        b = np.array(z_nbs)
        v, residuals, rank, s_lstq = np.linalg.lstsq(S, b, rcond=None)
        C[0, 0] = -2 * v[0]
        C[0, 1] = -v[1]
        C[1, 0] = -v[1]
        C[1, 1] = -2 * v[2]
        C_tensors.append(C.copy())
        C_mixed = np.dot(C, g_inv[i])
        C_tensors_mixed.append(C_mixed.copy())
        mean_curvature = np.trace(C_mixed) / 2
        gauss_curvature = np.linalg.det(C) / np.linalg.det(g[i])
        C_gauss.append(gauss_curvature)
        C_mean.append(mean_curvature)
    C_gauss, C_mean, C_tensors, C_tensors_mixed = np.asarray(C_gauss), np.asarray(C_mean), np.asarray(
        C_tensors), np.asarray(C_tensors_mixed)
    if debug:
        print(
            f"Gauss: AVG = {np.nanmean(C_gauss)} | MIN = {np.nanmin(C_gauss)} | MAX = {np.nanmax(C_gauss)} | None? = {np.isnan(C_gauss).sum()}")
        print(
            f"Mean: AVG = {np.nanmean(C_mean)} | MIN = {np.nanmin(C_mean)} | MAX = {np.nanmax(C_mean)} | None? = {np.isnan(C_mean).sum()}")

    Gauss_idxs = random_idxs.copy()
    if gauss_crop_range is not None:
        print(f">> Cropping Gauss to min {gauss_crop_range[0]}, max {gauss_crop_range[1]}...")
        crop_mask = (gauss_crop_range[0] < C_gauss) & (C_gauss < gauss_crop_range[1]) & (~np.isnan(C_gauss))
        C_gauss, Gauss_idxs = C_gauss[crop_mask], Gauss_idxs[crop_mask]
        if debug:
            print(
                f"Gauss: AVG = {np.nanmean(C_gauss)} | MIN = {np.nanmin(C_gauss)} | MAX = {np.nanmax(C_gauss)} | None? = {np.isnan(C_gauss).sum()}")

    mean_idxs = random_idxs.copy()
    if mean_crop_range is not None:
        print(f">> Cropping Mean to min {mean_crop_range[0]}, max {mean_crop_range[1]}...")
        crop_mask = (mean_crop_range[0] < C_mean) & (C_mean < mean_crop_range[1]) & (~np.isnan(C_mean))
        C_mean, mean_idxs = C_mean[crop_mask], mean_idxs[crop_mask]
        if debug:
            print(
                f"Mean: AVG = {np.nanmean(C_mean)} | MIN = {np.nanmin(C_mean)} | MAX = {np.nanmax(C_mean)} | None? = {np.isnan(C_mean).sum()}")
    tensor_idxs = random_idxs.copy()
    return C_gauss, C_mean, Gauss_idxs, mean_idxs, C_tensors, C_tensors_mixed, tensor_idxs


def density_estimate(mesh, k=30, debug=False, crop_range=None):
    print(f">> Density estimation for {len(mesh.vertices)} vertices with k = {k}...")
    idxs, dists = coord_search_neighbours(mesh.vertices, k=k, debug=False, return_dists=True)
    densities = k / np.average(np.square(dists[:, 1:]), axis=1)
    if debug:
        print(
            f"Densities: AVG = {np.average(densities)} | MIN = {np.min(densities)} | MAX = {np.max(densities)} | None? = {np.isnan(densities).sum()}")

    idxs = np.arange(0, mesh.vertices.shape[0])
    if crop_range is not None:
        print(f">> Cropping to min {crop_range[0]}, max {crop_range[1]}...")
        crop_mask = (crop_range[0] < densities) & (densities < crop_range[1])
        densities, idxs = densities[crop_mask], idxs[crop_mask]
        if debug:
            print(
                f"Densities: AVG = {np.average(densities)} | MIN = {np.min(densities)} | MAX = {np.max(densities)} | None? = {np.isnan(densities).sum()}")

    return densities, idxs


def inter_dist_mesh(mesh_1, mesh_2, num_sample, debug=False, crop_range=None, allow_multiple_hits=False):
    print(
        f">> Calculating distance between two meshes ({mesh_1.vertices.shape[0]}, {mesh_2.vertices.shape[0]}) using {num_sample} samples...")
    random_idxs = np.random.choice(np.arange(mesh_1.vertices.shape[0]), size=num_sample)
    vertices_1, normals_1 = mesh_1.vertices[random_idxs], mesh_1.vertex_normals[random_idxs]
    locations, index_ray, index_tri = mesh_2.ray.intersects_location(vertices_1, normals_1,
                                                                     multiple_hits=allow_multiple_hits)
    distances = np.full(len(vertices_1), np.nan)
    dist_vector = locations - vertices_1[index_ray]
    distances[index_ray] = np.einsum('ij,ij->i', dist_vector, normals_1[index_ray])
    valid_mask = ~np.isnan(distances)
    valid_indeces = random_idxs[valid_mask]
    valid_distances = distances[valid_mask]
    if debug:
        print(
            f"AVG = {np.average(valid_distances)} | MIN = {np.min(valid_distances)} | MAX = {np.max(valid_distances)} | None? = {np.isnan(distances).sum()}")
    if crop_range is None:
        return valid_distances, valid_indeces
    else:
        print(f">> Cropping to min {crop_range[0]}, max {crop_range[1]}...")
        crop_mask = (crop_range[0] < valid_distances) & (valid_distances < crop_range[1])
        cropped_distances, cropped_indeces = valid_distances[crop_mask], valid_indeces[crop_mask]
        if debug:
            print(
                f"AVG = {np.average(cropped_distances)} | MIN = {np.min(cropped_distances)} | MAX = {np.max(cropped_distances)} | None? = {np.isnan(cropped_distances).sum()}")
        return cropped_distances, cropped_indeces


def interpolate_on_mesh(mesh, value_idxs, values, k=10, eps=1e-8):
    print(f">> Interpolating {len(values)} values on mesh of {len(mesh.vertices)} vertices with k = {k}...")
    known_pts = mesh.vertices[value_idxs]
    all_pts = mesh.vertices
    dists, idxs = KDTree(known_pts).query(all_pts, k=k)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    dists += eps

    weights = 1.0 / dists
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights /= weight_sums
    gathered_values = values[idxs]
    interpolated = np.einsum("ij,ij->i", gathered_values, weights)
    return interpolated


def geodesic_distmesh(mesh, index1, index2, debug=False):
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length
    num_vertices = len(mesh.vertices)
    adj_matrix = sp.csr_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(num_vertices, num_vertices))
    adj_matrix = adj_matrix + adj_matrix.T
    distances = sp.csgraph.dijkstra(adj_matrix, indices=index1)
    geodesic_distance = distances[index2]
    if debug:
        print(f"Geodesic distance between vertex #{index1} and #{index2}: {geodesic_distance}")
    return geodesic_distance


def select_geodesic_defects(order, mesh, idxs_sel, unit, dist_cutoff=50, max_candidates=10):
    candidate_order = np.argsort(order)[:max_candidates]
    selected = []

    for i in candidate_order:
        mesh_idx = idxs_sel[i]
        if not selected:
            selected.append(i)
            continue
        dists = [geodesic_distmesh(mesh, mesh_idx, idxs_sel[j], debug=False) for j in selected]
        if all(d > dist_cutoff for d in dists):
            selected.append(i)

    print(f"Found {len(selected)} defects!")
    print("Relative distances between selected defects:")
    rel_dists = []
    for i, j in combinations(selected, 2):
        d = geodesic_distmesh(mesh, idxs_sel[i], idxs_sel[j], debug=False)
        rel_dists.append([i, j, d])
        print(f"#{i} <-> #{j} = {d:.2f} {unit}")
    rel_dists = np.array(rel_dists)
    return selected, rel_dists


######################
# PROJECTION MODULES #
######################


def proj2mesh(img, mesh, scale, unit, min_dist, max_dist, num_dist, mode, min_dist_per_vert=None, show_proj=False,
              figsize=(12, 5), interp_method="linear", return_full=False, cmap="inferno", savefig="", normalise=False,
              hidefig=False):
    num_dist = int(num_dist)
    print(
        f">> Projecting {mode} image intensities on {len(mesh.vertices)} verts using {interp_method} interpolation method...")
    print(f"Range: MIN = {min_dist}{unit}, MAX = {max_dist}{unit}, NUM = {num_dist}")
    verts, normals = mesh.vertices, mesh.vertex_normals
    if min_dist_per_vert is None:
        min_dist_per_vert = np.full(len(verts), min_dist)
    else:
        print(">> Overriding MIN distance !")
    distances = np.linspace(0, max_dist - min_dist, num_dist)
    distances_reshaped = distances.reshape(1, num_dist, 1)
    img_grid = (
        np.arange(img.shape[0]) * scale[0], np.arange(img.shape[1]) * scale[1], np.arange(img.shape[2]) * scale[2])
    interp_function = RegularGridInterpolator(points=img_grid, values=img, method=interp_method, bounds_error=False,
                                              fill_value=-1)
    outward_samples = verts[:, None, :] + normals[:, None, :] * (distances_reshaped + min_dist_per_vert[:, None, None])
    sampling_points = outward_samples.reshape(-1, 3)

    intensities = interp_function(sampling_points).reshape(len(verts), -1)
    outside_mask = intensities == -1
    if np.any(outside_mask):
        print(
            f"[!] {np.any(outside_mask, axis=1).sum()} / {len(verts)} sampling lines reach outside FOV, defaulting to 0.")
    intensities[outside_mask] = 0
    if show_proj:
        plot_dist_kymograph(distances=distances, intensities=intensities, cmap=cmap, figsize=figsize, unit=unit,
                            savefig=savefig, hidefig=hidefig)
    if mode == "max":
        proj_intensity_values = np.max(intensities, axis=1)
    elif mode == "mean":
        proj_intensity_values = np.mean(intensities, axis=1)
    else:
        return None

    if return_full:
        return distances, outward_samples, intensities

    if normalise:
        proj_intensity_values = normalise_range(proj_intensity_values)
    return proj_intensity_values


def spherical_project(pts, ref_point=None, rotate=None, debug=False):
    print(">> Applying Spherical Projection...")
    if ref_point is not None:
        centered_inner_verts = pts - ref_point
    else:
        centered_inner_verts = pts - np.mean(pts, axis=0)
    if rotate is not None:
        centered_inner_verts @= rot3dmatrix(alpha=rotate[0], beta=rotate[1], gamma=rotate[2])
    x, y, z = centered_inner_verts[:, 0], centered_inner_verts[:, 1], centered_inner_verts[:, 2]

    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.degrees(np.arctan2(y, x))
    theta = np.degrees(np.arccos(z / rho))
    if debug:
        print(f"Min phi = {np.min(phi)} | MAX phi = {np.max(phi)}")
        print(f"Min theta = {np.min(theta)} | MAX theta = {np.max(theta)}")
    return phi, theta


def spherical_project_vectors(pts, vecs, ref_point=None, rotate=None):
    print(">> Applying Spherical Projection to vector field...")
    pts = pts.copy()
    vecs = vecs.copy()
    if ref_point is not None:
        pts = pts - ref_point
    else:
        pts = pts - np.mean(pts, axis=0)
    if rotate is not None:
        R = rot3dmatrix(alpha=rotate[0], beta=rotate[1], gamma=rotate[2])
        pts = pts @ R
        vecs = vecs @ R
    x, y, z = pts.T
    r = np.linalg.norm(pts, axis=1)
    r[r == 0] = np.finfo(float).eps
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    e_theta = np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ], axis=1)
    vphi = np.einsum('ij,ij->i', vecs, e_phi)
    vtheta = np.einsum('ij,ij->i', vecs, e_theta)
    vphi /= np.sin(theta)
    vphi[np.isnan(vphi)] = 0
    norm = np.sqrt(vphi ** 2 + vtheta ** 2)
    vphi /= norm
    vtheta /= norm
    return vphi, vtheta


def create_radial_stack(values, phi_coords, theta_cords, projection_radii, grid_n=None):
    if grid_n is None:
        grid_n = int(np.sqrt(len(phi_coords)))
    else:
        grid_n = int(grid_n)
    print(f">> Creating z stack of {grid_n} x {grid_n} grid points...")
    phi_coords_fine = np.linspace(phi_coords.min(), phi_coords.max(), grid_n)
    theta_cords_fine = np.linspace(theta_cords.min(), theta_cords.max(), grid_n)
    grid_phi, grid_theta = np.meshgrid(phi_coords_fine, theta_cords_fine)
    proj_points = np.column_stack((phi_coords.ravel(), theta_cords.ravel()))
    grid_points = np.column_stack((grid_phi.ravel(), grid_theta.ravel()))
    tree = KDTree(proj_points)
    _, nearest_idx = tree.query(grid_points, k=1)

    radial_stack = np.empty((values.shape[0], grid_n, grid_n))
    stack_coords = np.empty((values.shape[0], grid_n, grid_n, 3))
    phi_grid = np.fliplr(grid_phi.T)
    theta_grid = np.fliplr(grid_theta.T)

    for i in range(values.shape[0]):
        radial_stack[i] = np.fliplr(values[i].ravel()[nearest_idx].reshape(grid_phi.shape).T)
        stack_coords[i, ..., 0] = phi_grid
        stack_coords[i, ..., 1] = theta_grid
        stack_coords[i, ..., 2] = projection_radii[i]
    return radial_stack, stack_coords


#############################################
# 2D ORIENTATION & NEMATIC ANALYSIS MODULES #
#############################################
def compute_orientation_with_intensity(img, mode, box_size, dimension=3, calc_energy_coherency=True):
    if type(box_size) is not int:
        structureTensorBoxes = op.computeGradientStructureTensorBoxes(img, box_size)
    else:
        structureTensorBoxes = op.computeGradientStructureTensorBoxes(img, [box_size] * dimension)
    if calc_energy_coherency:
        orientation = op.computeOrientation(
            structureTensorBoxes,
            mode=mode,
            computeEnergy=True,
            computeCoherency=True
        )
    else:
        orientation = op.computeOrientation(
            structureTensorBoxes,
            mode=mode,
            computeEnergy=False,
            computeCoherency=False
        )
    orientation["vector"] = np.moveaxis(op.anglesToVectors(orientation), 0, -1)
    return orientation


def compute_2d_orientation(mode, img, sampling_box_size, onlytheta=False, debug=True):
    if onlytheta:
        sampled_orientation = compute_orientation_with_intensity(img=img, mode=mode, box_size=sampling_box_size,
                                                                 dimension=2, calc_energy_coherency=False)
        return sampled_orientation["theta"]
    else:
        sampled_orientation = compute_orientation_with_intensity(img=img, mode=mode, box_size=sampling_box_size,
                                                                 dimension=2, calc_energy_coherency=True)
        if debug:
            print(f">> Upscaling orientation result...")
        theta = expand_2d_array(array=sampled_orientation["theta"], num=sampling_box_size)
        energy = expand_2d_array(array=sampled_orientation["energy"], num=sampling_box_size)
        coherency = expand_2d_array(array=sampled_orientation["coherency"], num=sampling_box_size)
        vectors = expand_2d_array(array=sampled_orientation["vector"], num=sampling_box_size)
        if debug:
            print(f"Total of {len(vectors)} orientation vectors !")
        return theta, energy, coherency, vectors


##############################################
# 2D+ ORIENTATION & NEMATIC ANALYSIS MODULES #
##############################################

def tan_proj(neighbors_coords, central_normal):
    print(f">> Locally flattening coords ...")
    if isinstance(neighbors_coords, list):
        local_2d_coords_list = []
        tangent_x_list = []
        tangent_y_list = []
        for i, coords in enumerate(neighbors_coords):
            c_norm = central_normal[i]
            central_coord = coords[0]

            cross_basis_vector = np.array([1, 0, 0]) if not np.allclose(np.cross(c_norm, [1, 0, 0]), 0) else np.array(
                [0, 1, 0])
            t_x = np.cross(c_norm, cross_basis_vector)
            t_x /= np.linalg.norm(t_x)
            t_y = np.cross(c_norm, t_x)
            t_y /= np.linalg.norm(t_y)

            translated_points = coords - central_coord
            local_x = translated_points @ t_x
            local_y = translated_points @ t_y
            local_2d_coords_list.append(np.stack((local_x, local_y), axis=-1))
            tangent_x_list.append(t_x)
            tangent_y_list.append(t_y)

        tangent_x_axes = np.stack(tangent_x_list)
        tangent_y_axes = np.stack(tangent_y_list)
        return local_2d_coords_list, tangent_x_axes, tangent_y_axes

    else:
        central_coord = neighbors_coords[:, 0, :]
        cross_basis_vector = np.where(np.all(np.isclose(np.cross(central_normal, [1, 0, 0]), 0), axis=1)[:, None],
                                      [0, 1, 0],
                                      [1, 0, 0])
        tangent_x_axes = np.cross(central_normal, cross_basis_vector)
        tangent_x_axes /= np.linalg.norm(tangent_x_axes, axis=1, keepdims=True)
        tangent_y_axes = np.cross(central_normal, tangent_x_axes)
        tangent_y_axes /= np.linalg.norm(tangent_y_axes, axis=1, keepdims=True)

        translated_points = neighbors_coords - central_coord[:, np.newaxis, :]
        local_x = np.einsum('nij,nj->ni', translated_points, tangent_x_axes)
        local_y = np.einsum('nij,nj->ni', translated_points, tangent_y_axes)
        local_2d_coordinates = np.stack((local_x, local_y), axis=-1)
        return local_2d_coordinates, tangent_x_axes, tangent_y_axes


def tan_interp_batch(coords, intensities, grid_size):
    if isinstance(coords, list):
        batch_size = len(coords)
    else:
        batch_size, _, _ = coords.shape

    lin_u = np.linspace(0, 1, grid_size)
    lin_v = np.linspace(0, 1, grid_size)
    grid_x, grid_y = np.meshgrid(lin_u, lin_v, indexing="ij")

    grid_z = []
    for i in range(batch_size):
        pts = coords[i] if isinstance(coords, list) else coords[i]
        inten = intensities[i] if isinstance(coords, list) else intensities[i]
        u_min, u_max = pts[:, 0].min(), pts[:, 0].max()
        v_min, v_max = pts[:, 1].min(), pts[:, 1].max()
        gx = u_min + (u_max - u_min) * grid_x
        gy = v_min + (v_max - v_min) * grid_y
        grid_pts = np.stack([gx, gy], axis=-1).reshape(-1, 2)
        tree = KDTree(pts)
        _, idx = tree.query(grid_pts, k=1)
        gz = inten[idx].reshape(grid_size, grid_size)
        grid_z.append(gz)
    return grid_x, grid_y, grid_z


def batch_2d_orientation(big_grid, box_size, vertices, tan_x, tan_y, unit, debug=False, debug_idx=None,
                         debug_line_length=5):
    dir_vec = np.zeros(shape=(len(tan_x), 3))
    theta_all = compute_2d_orientation(mode="fiber", img=big_grid, sampling_box_size=box_size, onlytheta=True) * -1

    if debug:
        theta_center = np.radians(theta_all[theta_all.shape[0] // 2, theta_all.shape[1] // 2])
        dir_vec[debug_idx] = np.cos(theta_center) * tan_x[debug_idx] + np.sin(theta_center) * tan_y[debug_idx]
        print(f"=== # {debug_idx} | theta = {np.round(np.degrees(theta_center), 2)} degrees ===")
        debug_grid_z = big_grid[0] if isinstance(big_grid, list) else big_grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, debug_grid_z.shape[0]),
            np.linspace(0, 1, debug_grid_z.shape[1]),
            indexing="ij"
        )
        plot_interp_grid(grid_x, grid_y, debug_grid_z, theta=theta_center, linelength=debug_line_length)
        plot_matrix(theta_all, title="Theta", colorbar=True, origin="lower", cmap_limits=[-90, 90],
                    remove_axes=True, unit=unit)
    else:
        theta_mid_idx = theta_all.shape[1] // 2
        center_indices = theta_mid_idx + np.arange(0, len(theta_all), theta_all.shape[1])
        theta_center = np.radians(theta_all[center_indices, theta_mid_idx])
        dir_vec = np.cos(theta_center)[:, None] * tan_x + np.sin(theta_center)[:, None] * tan_y

    directors = np.column_stack((vertices, dir_vec))
    print(f"Finished! {len(directors)} director(s)!")
    return directors


def avg_tan_nem_tens(t1_cov, t2_cov, directors, neigh_idxs, debug=False, return_qij_bar=False):
    print(f">> Averaging the nematic tensor in basis t1 t2 ...")
    N = len(t1_cov)

    res = np.array([covariant2contravariant(t1_cov[i], t2_cov[i]) for i in range(N)])
    t1_contra, t2_contra = res[:, 0, :], res[:, 1, :]

    t_outer = np.array([otimes(t1_contra[i], t2_contra[i]) for i in range(N)])
    n_vecs = directors[:, 3:]
    n_locals = np.hstack([
        np.sum(n_vecs * t1_cov, axis=1, keepdims=True),
        np.sum(n_vecs * t2_cov, axis=1, keepdims=True)
    ])
    n_locals = n_locals / np.linalg.norm(n_locals, axis=1, keepdims=True)
    q = n_locals[:, :, np.newaxis] * n_locals[:, np.newaxis, :] - (1 / 2) * np.eye(2)[np.newaxis, :, :]

    q_tilde = np.array([tensprod(q[i], t_outer[i]) for i in range(N)])

    if isinstance(neigh_idxs, np.ndarray):
        q_tilde_avg = np.average(q_tilde[neigh_idxs], axis=1)
    else:
        q_tilde_avg = []
        for i, neigh in enumerate(neigh_idxs):
            if len(neigh) == 0:
                q_tilde_avg.append(np.zeros((2, 2)))
            else:
                q_tilde_avg.append(np.average(q_tilde[neigh], axis=0))
        q_tilde_avg = np.stack(q_tilde_avg)
    qij_bar = np.array([np.array([[tensprod(q_tilde_avg[i], t_outer[i, 0, 0]),
                                   tensprod(q_tilde_avg[i], t_outer[i, 0, 1])],
                                  [tensprod(q_tilde_avg[i], t_outer[i, 1, 0]),
                                   tensprod(q_tilde_avg[i], t_outer[i, 1, 1])]]) for i in range(N)])

    eigvals, eigvecs = np.linalg.eigh(qij_bar)
    max_indeces = np.argmax(eigvals, axis=1)
    max_eigvals = eigvals[np.arange(len(max_indeces)), max_indeces]
    max_eigvecs = eigvecs[np.arange(len(max_indeces)), :, max_indeces]
    max_eigvecs = max_eigvecs / np.linalg.norm(max_eigvecs, axis=1, keepdims=True)
    S_order = max_eigvals * 2
    n_avg = (max_eigvecs[:, 0, np.newaxis] * t1_cov) + (max_eigvecs[:, 1, np.newaxis] * t2_cov)
    n_avg = n_avg / np.linalg.norm(n_avg, axis=1, keepdims=True)
    if debug:
        print(f"directors = \n {directors[:, 3:]}")
        print(f"n_locals with shape {n_locals.shape} = \n {n_locals}")
        print(f"t_outer with shape {t_outer.shape}")
        print(f"q with shape {q.shape} = \n {q}")
        print(f"q_tilde with shape {q_tilde.shape} = \n {q_tilde}")
        print(f"q_tilde_avg with shape {q_tilde_avg.shape} = \n {q_tilde_avg}")
        print(f"q_bar with shape {qij_bar.shape} = \n {qij_bar}")
        print(f"S_order with shape {S_order.shape} = \n {S_order}")
        print(f"n_avg with shape {n_avg.shape} = \n {n_avg}")
    if return_qij_bar:
        return S_order, n_avg, qij_bar
    else:
        return S_order, n_avg


def unique_neighborhoods(arr):
    if isinstance(arr, list):
        M = len(arr)
        neighborhoods = [set(neigh) for neigh in arr]
        conflict = np.zeros((M, M), dtype=bool)
        for i in range(M):
            for j in range(i + 1, M):
                if neighborhoods[i] & neighborhoods[j]:
                    conflict[i, j] = True
                    conflict[j, i] = True

        selected = []
        remaining = np.ones(M, dtype=bool)
        while remaining.any():
            degrees = conflict[remaining][:, remaining].sum(axis=1)
            idx_in_remaining = np.argmin(degrees)
            idx = np.flatnonzero(remaining)[idx_in_remaining]
            selected.append(idx)
            to_remove = conflict[idx] | (np.arange(M) == idx)
            remaining[to_remove] = False
        print(f"{len(selected)} unique out of {len(arr)}!")
        return [arr[i] for i in selected]
    else:
        arr = np.asarray(arr)
        M, k = arr.shape
        max_index = arr.max() + 1
        membership = np.zeros((M, max_index), dtype=bool)
        membership[np.arange(M)[:, None], arr] = True
        conflict = membership @ membership.T > 0
        np.fill_diagonal(conflict, 0)
        selected = []
        remaining = np.ones(M, dtype=bool)

        while remaining.any():
            degrees = conflict[remaining][:, remaining].sum(axis=1)
            idx_in_remaining = np.argmin(degrees)
            idx = np.flatnonzero(remaining)[idx_in_remaining]
            selected.append(idx)
            to_remove = conflict[idx] | (np.arange(M) == idx)
            remaining[to_remove] = False
        print(f"{len(selected)} unique out of {len(arr)}!")
        return arr[selected]


def patch_surface_integral(mesh, value, patch_idxs, debug=False):
    patch_integrals = []
    for i in range(patch_idxs.shape[0]):
        patch_faces = mesh.faces[np.isin(mesh.faces, patch_idxs[i]).all(axis=1)]
        v0, v1, v2 = patch_faces.T
        p0, p1, p2 = mesh.vertices[v0], mesh.vertices[v1], mesh.vertices[v2]
        areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / 2
        avg_val = np.nanmean([value[v0], value[v1], value[v2]])
        patch_integrals.append(np.nansum(avg_val * areas))
    patch_integrals = np.array(patch_integrals) / (2 * np.pi)
    if debug:
        print(f"-- surface integrals = {patch_integrals}")
    return patch_integrals


def find_boundary_indeces(mesh, patch_idxs, tan_x, tan_y, angle_precision=1):
    boundary_indeces = []
    for i in range(len(patch_idxs)):
        patch_vertices = mesh.vertices[patch_idxs[i]]
        idx_rel_center = np.argmin(np.linalg.norm(patch_vertices - np.mean(patch_vertices, axis=0), axis=1))
        patch_tan_x = tan_x[[patch_idxs[i][idx_rel_center]]]
        patch_tan_y = tan_y[[patch_idxs[i][idx_rel_center]]]
        patch_rel_pos = patch_vertices - patch_vertices[idx_rel_center, :][np.newaxis, :]
        rel_x = np.einsum("ij,kj->i", patch_rel_pos, patch_tan_x)
        rel_y = np.einsum("ij,kj->i", patch_rel_pos, patch_tan_y)
        patch_rel_angles = np.arctan2(rel_y, rel_x)
        distances = np.linalg.norm(patch_rel_pos, axis=1)
        rounded_angles = np.round(patch_rel_angles, angle_precision)
        angle_to_idx = {}
        for j, angle in enumerate(rounded_angles):
            if angle not in angle_to_idx or distances[j] > distances[angle_to_idx[angle]]:
                angle_to_idx[angle] = j
        unique_sorted_indices = np.array([patch_idxs[i][angle_to_idx[angle]] for angle in sorted(angle_to_idx.keys())])
        boundary_indeces.append(unique_sorted_indices[::-1])
    return boundary_indeces


def top_charge_loop_integral(loop_idxs, directors, director_indeces, normals, correct_orientation=True, debug=False):
    topological_charges = []
    index_map = {idx: i for i, idx in enumerate(director_indeces)}

    for patch_indices in loop_idxs:
        normal_sel = normals[patch_indices]
        dir_indices = np.array([index_map[idx] for idx in patch_indices])

        p_current = directors[dir_indices][:, 3:]
        p_next = np.roll(p_current, -1, axis=0)

        if correct_orientation:
            signs = np.sign(np.einsum('ij,ij->i', p_current, p_next))
            p_next *= signs[:, None]

        pdiff = p_next - p_current

        normal_sel /= np.linalg.norm(normal_sel, axis=1, keepdims=True)

        dot = np.einsum('ij,ij->i', pdiff, normal_sel)
        pdiff_proj = pdiff - dot[:, None] * normal_sel

        cross = np.cross(pdiff_proj, p_current)
        winding_contrib = np.einsum('ij,ij->i', cross, normal_sel)

        m = np.sum(winding_contrib) / (2 * np.pi)
        topological_charges.append(m)

    if debug:
        print(f"-- line charge m = {topological_charges}")
    return topological_charges


def curved_nem_charge(mesh, directors, calc_idxs, director_indeces, tan_x, tan_y, c_gauss, loop_angle_precision=1,
                      patch_mode="nearest", patch_size=25, debug=False, return_all_contributions=False,
                      correct_orientation=True):
    print(f">> Calculating topological charge for {len(directors)} directors...")
    if patch_mode == "radius":
        charge_patch_broad = coord_search_radius(mesh.vertices,
                                                 custom_probes=mesh.vertices[director_indeces[calc_idxs]], r=patch_size,
                                                 debug=debug)
        charge_patch_broad = np.array(charge_patch_broad, dtype=object)
    elif patch_mode == "nearest":
        charge_patch_broad = coord_search_neighbours(mesh.vertices,
                                                     custom_probes=mesh.vertices[director_indeces[calc_idxs]],
                                                     k=patch_size,
                                                     debug=debug)
    else:
        print(f"[!] Unknown patch type: {patch_mode}")
        return None
    m_gauss_contribution = patch_surface_integral(mesh=mesh, value=c_gauss,
                                                  patch_idxs=charge_patch_broad,
                                                  debug=debug)
    calc_charge_loop_idxs = find_boundary_indeces(mesh, [np.intersect1d(loop, director_indeces) for loop in
                                                         charge_patch_broad],
                                                  tan_x=tan_x, tan_y=tan_y,
                                                  angle_precision=loop_angle_precision)
    m_line_charge = top_charge_loop_integral(loop_idxs=calc_charge_loop_idxs, directors=directors,
                                             director_indeces=director_indeces,
                                             normals=mesh.vertex_normals, debug=debug,
                                             correct_orientation=correct_orientation)
    m_charge = m_line_charge + m_gauss_contribution
    if debug:
        print(f"Charges calculated : {m_charge}")
        total_charge = np.nansum(m_charge)
        print(f"Total Charge: {total_charge}")
    if not return_all_contributions:
        return m_charge, calc_charge_loop_idxs
    else:
        return m_charge, calc_charge_loop_idxs, m_line_charge, m_gauss_contribution


def layers_crisscross(layer_name_1, layer_name_2, patch_label_1, patch_label_2, resdata_dir,
                      director_name_prefix="directors-avg_2dcurved_"):
    print(f"Loading layer 1 {layer_name_1}...")
    resdata_dir_layer_1 = os.path.join(resdata_dir, layer_name_1)
    directors_2dcurved_avg_1 = load_array(f"{director_name_prefix}{patch_label_1}",
                                          folderpath=resdata_dir_layer_1)
    tan_x_1 = load_array("tan_x", folderpath=resdata_dir_layer_1)
    tan_y_1 = load_array("tan_y", folderpath=resdata_dir_layer_1)
    print(f"Loading layer 2 {layer_name_2}...")

    resdata_dir_layer_2 = os.path.join(resdata_dir, layer_name_2)
    directors_2dcurved_avg_2 = load_array(f"{director_name_prefix}{patch_label_2}",
                                          folderpath=resdata_dir_layer_2)
    tan_x_2 = load_array("tan_x", folderpath=resdata_dir_layer_2)
    tan_y_2 = load_array("tan_y", folderpath=resdata_dir_layer_2)

    joint_directors_2dcurved_avg = np.concatenate((directors_2dcurved_avg_1, directors_2dcurved_avg_2), axis=0)
    joint_tan_x = np.concatenate((tan_x_1, tan_x_2), axis=0)
    joint_tan_y = np.concatenate((tan_y_1, tan_y_2), axis=0)

    coords1 = directors_2dcurved_avg_1[:, :3]
    coords2 = directors_2dcurved_avg_2[:, :3]
    neigh_in_set2 = coord_search_neighbours(coords2, custom_probes=coords1, k=1, n_process=8).ravel()
    N1 = len(coords1)
    N2 = len(coords2)
    pair_1 = np.column_stack([
        np.arange(N1),
        neigh_in_set2 + N1
    ])
    pair_2 = np.column_stack([
        np.arange(N1, N1 + N2),
        np.arange(N1, N1 + N2)
    ])
    joint_neigh_idxs = np.vstack([pair_1, pair_2])
    S_2dcurv_interlayer, n_avg_2dcurv_interlayer = avg_tan_nem_tens(t1_cov=joint_tan_x, t2_cov=joint_tan_y,
                                                                    directors=joint_directors_2dcurved_avg,
                                                                    neigh_idxs=joint_neigh_idxs)
    crisscross_mag = 1 - S_2dcurv_interlayer[:len(directors_2dcurved_avg_1)]
    return crisscross_mag, directors_2dcurved_avg_1, directors_2dcurved_avg_2


def compute_defect_polarisations(mesh, idxs_sel, directors, vertex_normals,
                                 defect_idxs_calc, m_charge, patch_type, patch_size,
                                 show_profile=False, hidefig=True):
    pol_positions, pol_vectors, pol_idxs = [], [], []
    directors_v = directors[:, 3:6]
    directors_v /= np.linalg.norm(directors_v, axis=1, keepdims=True) + 1e-12

    for d_idx, charge in zip(defect_idxs_calc, m_charge):
        s = np.round(charge * 2) / 2
        if not (abs(s) == 0.5):
            continue

        core_idx_sel = idxs_sel[d_idx]
        core_vertex = mesh.vertices[core_idx_sel]

        # --- Neighborhood Retrieval ---
        if patch_type == "radius":
            neigh_idxs_sel = coord_search_radius(mesh.vertices[idxs_sel],
                                                 custom_probes=[core_vertex],
                                                 r=patch_size)[0]
        elif patch_type == "nearest":
            neigh_idxs_sel = coord_search_neighbours(mesh.vertices[idxs_sel],
                                                     custom_probes=[core_vertex],
                                                     k=patch_size)[0]
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")

        if len(neigh_idxs_sel) < 10:
            continue

        # --- Local Basis (t1, t2) ---
        normal = vertex_normals[core_idx_sel]
        normal /= np.linalg.norm(normal)
        t1 = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(normal, [0, 1, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)

        # --- Project Positions and Directors ---
        rel_pos = mesh.vertices[idxs_sel[neigh_idxs_sel]] - core_vertex
        # Spatial angle (phi in Function 2)
        phi_spatial = np.arctan2(rel_pos @ t2, rel_pos @ t1)

        # Director angle (theta in Function 2)
        neigh_dirs = directors_v[neigh_idxs_sel]
        d_proj_x = neigh_dirs @ t1
        d_proj_y = neigh_dirs @ t2
        theta_dir = np.arctan2(d_proj_y, d_proj_x)

        # --- MATCHING MATH: Complex Phase Extraction ---
        # Calculation: Z = mean( exp( i * (2*theta - 2*s*phi) ) )
        # This finds the intrinsic phase phi_0
        Z = np.mean(np.exp(1j * (2 * theta_dir - 2 * s * phi_spatial)))
        phi_0 = np.angle(Z) / 2

        current_pols_3d = []

        if s > 0:  # +1/2 Comet
            alpha = 2 * phi_0
            pol = np.cos(alpha) * t1 + np.sin(alpha) * t2
            current_pols_3d.append(pol)
        else:  # -1/2 Trefoil
            for m in range(3):
                alpha = (2.0 / 3.0) * (phi_0 + m * np.pi)
                pol = np.cos(alpha) * t1 + np.sin(alpha) * t2
                current_pols_3d.append(pol)
        for p in current_pols_3d:
            pol_positions.append(core_vertex)
            pol_vectors.append(p)
            pol_idxs.append(d_idx)

        if show_profile:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_aspect('equal')
            ax.set_title(f"Defect {d_idx}, s={s}")

            x_2d = rel_pos @ t1
            y_2d = rel_pos @ t2
            scale = 0.15 * patch_size

            ax.quiver(x_2d, y_2d, d_proj_x * scale, d_proj_y * scale,
                      color='blue', alpha=0.3)

            for p in current_pols_3d:
                p2d = [p @ t1, p @ t2]
                ax.quiver(0, 0, p2d[0] * scale * 3, p2d[1] * scale * 3,
                          color='red', width=0.02, pivot='tail')

            ax.scatter(0, 0, color='k', marker='x')
            if not hidefig:
                plt.show()
            else:
                plt.close()

    return np.column_stack((np.array(pol_positions), np.array(pol_vectors))), np.array(pol_idxs)
