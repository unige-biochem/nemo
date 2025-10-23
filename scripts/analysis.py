"""
Analysis Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import os
import numpy as np
from tifffile import imread, TiffFile
import zarr
from scipy.spatial import KDTree
from skimage import measure
from scipy.interpolate import RegularGridInterpolator, splprep, splev
from scipy.ndimage import gaussian_filter, binary_fill_holes
from scipy.optimize import least_squares
import orientationpy as op
import scipy.sparse as sp
from skimage.filters import threshold_yen
import trimesh
from scripts.visuals import plot_dist_kymograph, plot_interp_grid, plot_matrix
from sklearn.neighbors import KDTree as KDTreeSklearn
import pyvista as pv
from sklearn.decomposition import PCA
from skimage.morphology import medial_axis
from itertools import combinations


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


def multi_coord_search_neighbours(verts_tree, verts_query, k, n_process=8, debug=False, return_dists=False):
    if debug:
        print(f">> Searching {k} neighbours with different tree ...")
    tree = KDTree(verts_tree)
    dists, idxs = tree.query(verts_query, k=k, workers=n_process)
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


def rot_x(alpha):
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]])


def rot_y(beta):
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                     [0, 1, 0],
                     [-np.sin(beta), 0, np.cos(beta)]])


def rot_z(gamma):
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])


def rot3dmatrix(alpha, beta, gamma):
    Rx = rot_x(alpha)
    Ry = rot_y(beta)
    Rz = rot_z(gamma)
    return Rx @ Ry @ Rz


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


# def filter_valid_patches(verts, idxs_neigh, factor=0.1):
#     centroids_patches = np.mean(verts[idxs_neigh], axis=1)[:, np.newaxis]
#     reldists = np.linalg.norm(verts[idxs_neigh] - centroids_patches, axis=2)
#     center_reldist = reldists[:, 0]
#     keep_mask = center_reldist < factor * np.max(reldists, axis=1)
#     valid_patch_idxs = idxs_neigh[keep_mask]
#     valid_idxs = np.argwhere(keep_mask)[:, 0]
#     print(f"{len(valid_patch_idxs)} valid, {len(centroids_patches) - len(valid_patch_idxs)} invalid ! ")
#     return valid_patch_idxs, valid_idxs

def filter_valid_patches(verts, idxs_neigh, factor=0.1):
    # Handle list-of-lists (radius-search)
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

    # --- existing vectorised code for ndarray (nearest k) ---
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


def expand_3d_array(array, num):
    if type(num) is not int:
        return np.repeat(np.repeat(np.repeat(array, num[0], axis=0), num[1], axis=1), num[2], axis=2)
    else:
        return np.repeat(np.repeat(np.repeat(array, num, axis=0), num, axis=1), num, axis=2)


############################
# IMAGE PROCESSING MODULES #
############################
def load_img(path, norm_vals, reduce_xy=1, reduce_z=1, recalc_z=False, custom_scaling=None, img_unit="um"):
    print(f">> Loading image {path}...")
    try:
        img_raw = imread(path)
    except:
        print(f"[!] Image does not exist, aborting !")
        return None
    img_dim = img_raw.shape
    n_dim = len(img_dim)
    if n_dim != 3:
        print(f"! Not yet adapted to {n_dim}-D images, aborting !")
        return None, None, None
    print(f"{n_dim}D | shape = {img_dim} | size = {round(img_raw.nbytes / 1e6, 2)} MB ")
    with TiffFile(path) as tif:
        # ======== XY resolution ========
        try:
            xres = tif.pages[0].tags['XResolution'].value
            yres = tif.pages[0].tags['YResolution'].value
            xscale = 1 / float(xres[0] / xres[1])
            yscale = 1 / float(yres[0] / yres[1])
            print(f"Found xscale = {xscale}, yscale = {yscale} !")
        except:
            xscale, yscale = 1, 1
            print(f"XY scaling absent... using default {xscale, yscale} instead !")

        # ======== Z resolution ========
        try:
            zscale = float(next(line for line in tif.pages[0].tags.get('ImageDescription', None).value.splitlines() if
                                line.startswith("spacing=")).split('=')[1])
            print(f"Found zscale = {zscale} !")
        except:
            zscale = 1.0
            print(f"Z spacing absent... using default {zscale} instead !")
        if recalc_z:
            zscale = np.average([yscale, xscale]) * np.average(img_dim[1:]) / img_dim[0]
            print(f"Recalculated z spacing as {zscale} !")
        img_scale = (zscale, yscale, xscale)
        print(f"Found image scale = {img_scale} !")

    # ======== Overwrite scaling if not found ========
    if custom_scaling is not None:
        print(f">> Overwriting scaling with {custom_scaling}...")
        img_scale = custom_scaling

    # ======== Reduce resolution if needed by averaging ========
    if reduce_z > 1 or reduce_xy > 1:
        if reduce_xy > 1:
            print(f">> Reducing XY resolution by averaging {reduce_xy} pixels...")
            trimmed_y = (img_dim[1] // reduce_xy) * reduce_xy
            trimmed_x = (img_dim[2] // reduce_xy) * reduce_xy
            img_raw = img_raw[:, :trimmed_y, :trimmed_x]
            new_shape = (
                img_dim[0],
                trimmed_y // reduce_xy,
                reduce_xy,
                trimmed_x // reduce_xy,
                reduce_xy,
            )
            img_raw = img_raw.reshape(new_shape).mean(axis=(2, 4))
            img_scale = (img_scale[0], reduce_xy * img_scale[1], reduce_xy * img_scale[2])
        img_dim = img_raw.shape
        if reduce_z > 1:
            print(f">> Reducing Z resolution by averaging {reduce_z} pixels...")
            trimmed_z = (img_dim[0] // reduce_z) * reduce_z
            img_raw = img_raw[:trimmed_z, :, :]
            new_shape = (
                trimmed_z // reduce_z,
                reduce_z,
                img_dim[1],
                img_dim[2],
            )
            img_raw = img_raw.reshape(new_shape).mean(axis=1)
            img_scale = (reduce_z * img_scale[0], img_scale[1], img_scale[2])
        img_dim = img_raw.shape
        print(f"Reduced shape = {img_dim} | scale = {img_scale} | size = {round(img_raw.nbytes / 1e6, 2)} MB !")

    # ======== Norm values if needed ========
    if norm_vals:
        print(f">> Norming intensities...")
        img_raw = normalise_range(img_raw)
    print(f"Using unit {img_unit}...")
    return img_raw, img_dim, img_scale, img_unit


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


def load_img_virtual(path, norm_vals=False, t_sel_idx=0, c_sel_idx=0, custom_unit=None, custom_scaling=None,
                     reduce_xy=1, reduce_z=1):
    print(f">> Loading {path}...")
    if not os.path.exists(path):
        print(f"[!] Image does not exist, aborting !")
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
        img_unit = "px"
        if custom_unit is not None:
            print(f">> Overwriting unit with {custom_unit}...")
            img_unit = custom_unit

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

    # ======== Optional: Scaling Override ========
    if custom_scaling is not None:
        print(f">> Overwriting scaling with {custom_scaling}...")
        img_scale = custom_scaling

    # ======== Reduce resolution if needed by averaging ========
    if reduce_z > 1 or reduce_xy > 1:
        if reduce_xy > 1:
            print(f">> Reducing XY resolution by averaging {reduce_xy} pixels...")
            trimmed_y = (img_dim[1] // reduce_xy) * reduce_xy
            trimmed_x = (img_dim[2] // reduce_xy) * reduce_xy
            img_raw = img_raw[:, :trimmed_y, :trimmed_x]
            new_shape = (
                img_dim[0],
                trimmed_y // reduce_xy,
                reduce_xy,
                trimmed_x // reduce_xy,
                reduce_xy,
            )
            img_raw = img_raw.reshape(new_shape).mean(axis=(2, 4))
            img_scale = (img_scale[0], reduce_xy * img_scale[1], reduce_xy * img_scale[2])
        img_dim = img_raw.shape
        if reduce_z > 1:
            print(f">> Reducing Z resolution by averaging {reduce_z} pixels...")
            trimmed_z = (img_dim[0] // reduce_z) * reduce_z
            img_raw = img_raw[:trimmed_z, :, :]
            new_shape = (
                trimmed_z // reduce_z,
                reduce_z,
                img_dim[1],
                img_dim[2],
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
def marching_cubes(img, scale, level, step_size, allow_degenerate=False, clean=True):
    print(f">> Extracting mesh(es) using marching cubes algorithm with box size {step_size}...")
    verts, faces, normals, _ = measure.marching_cubes(volume=img, level=level, spacing=scale,
                                                      step_size=step_size, allow_degenerate=allow_degenerate)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=-normals)
    if clean:
        return clean_mesh(mesh)
    else:
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
def clean_mesh(mesh):
    mesh = mesh.copy()
    verts, faces, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
    invalid_normals = np.linalg.norm(normals, axis=1) < 0.9
    valid_verts = verts[~invalid_normals]
    valid_normals = normals[~invalid_normals]
    vertex_map = np.cumsum(~invalid_normals) - 1
    invalid_face_mask = np.any(invalid_normals[faces], axis=1)
    valid_faces = vertex_map[faces[~invalid_face_mask]]
    if np.sum(invalid_normals) > 0:
        print(f"[!] Cleaned {np.sum(invalid_normals)} out of {len(mesh.vertex_normals)} !")
    mesh_cleaned = trimesh.Trimesh(vertices=valid_verts, faces=valid_faces, vertex_normals=valid_normals)
    return mesh_cleaned


def mesh_properties(mesh, unit):
    print(
        f"-#vertices: {len(mesh.vertices)} | #faces = {len(mesh.faces)} | #edges = {len(mesh.edges)} | #vertex_normals = {len(mesh.vertex_normals)}")
    volume = abs(mesh.volume)
    area = abs(mesh.area)
    print(f"-XYZ PTP = {np.ptp(mesh.vertices, axis=0)}")
    print(f"-XYZ RADIUS = {np.ptp(mesh.vertices, axis=0).mean() / 2}")
    if volume > 0 and area > 0:
        print(f"-AREA = {round(area, 2)} {unit}^2 => Exp. radius = {round(np.sqrt(area / (4 * np.pi)), 2)} {unit}")
        print(
            f"-VOLUME = {round(volume, 2)} {unit}^3 => Exp. radius = {round((volume / (4 / 3 * np.pi)) ** (1 / 3), 2)} {unit}")

    else:
        print(f"!Weird mesh, vol={round(volume, 2)} {unit}^3, area={round(area, 2)} {unit}^2")


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


def laplacian_smooth_mesh(mesh, num_neighbors_verts):
    print(f">> Smoothing mesh ...")
    mesh = mesh.copy()
    neigh_idxs = coord_search_neighbours(mesh.vertices, num_neighbors_verts)
    mesh.vertices = np.average(mesh.vertices[neigh_idxs], axis=1)
    mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(vertex_count=len(mesh.vertices), faces=mesh.faces,
                                                                   face_normals=mesh.face_normals,
                                                                   face_angles=mesh.face_angles)
    return clean_mesh(mesh=mesh)


def subdivide_mesh(mesh, max_edge=8):
    mesh = mesh.copy()
    print(f">> Subdividing mesh using max length of {max_edge} for the edges...")
    subdiv_mesh = mesh.subdivide_to_size(max_edge=max_edge)
    print(f"New nr. of vertices = {len(subdiv_mesh.vertices)} !")
    return clean_mesh(mesh=subdiv_mesh)


def simplify_mesh(mesh, target_face_count):
    print(f">> Simplifying {len(mesh.vertices)} vertices to {target_face_count} faces ...")
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
    reduced_mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)
    print(f"New number of vertices = {len(reduced_mesh.vertices)} !")
    return reduced_mesh


def advanced_simplify_mesh(mesh, director_indices, target_face_count=20000):
    director_faces_mask = np.any(np.isin(mesh.faces, director_indices), axis=1)
    director_faces = mesh.faces[director_faces_mask]
    locked_vertices = np.unique(director_faces.flatten())
    other_faces_mask = ~director_faces_mask
    other_mesh = mesh.submesh([other_faces_mask], append=True)
    simplified_other_mesh = other_mesh.simplify_quadric_decimation(face_count=target_face_count)
    combined_vertices = np.vstack((mesh.vertices[locked_vertices], simplified_other_mesh.vertices))
    old_to_new = -np.ones(len(mesh.vertices), dtype=int)
    old_to_new[locked_vertices] = np.arange(len(locked_vertices))
    adjusted_locked_faces = old_to_new[director_faces]
    simplified_faces = simplified_other_mesh.faces + len(locked_vertices)
    combined_faces = np.vstack((adjusted_locked_faces, simplified_faces))
    simplified_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    simplified_mesh.merge_vertices()
    simplified_mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
        vertex_count=len(simplified_mesh.vertices), faces=simplified_mesh.faces,
        face_normals=simplified_mesh.face_normals,
        face_angles=simplified_mesh.face_angles)
    return simplified_mesh


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
def ellipsoid_func(params, points):
    x0, y0, z0, a, b, c, alpha, beta, gamma = params
    shifted_points = points - np.array([x0, y0, z0])
    rotmatrix = rot3dmatrix(alpha=alpha, beta=beta, gamma=gamma)
    rotated_points = shifted_points @ rotmatrix.T
    rho = (rotated_points[:, 0] / a) ** 2 + (rotated_points[:, 1] / b) ** 2 + (rotated_points[:, 2] / c) ** 2
    residuals = (rho - 1)
    return residuals


def sphere_func(params, points):
    x0, y0, z0, r = params
    return np.linalg.norm(points - [x0, y0, z0], axis=1) - r


def fit_sphere(points):
    center = np.mean(points, axis=0)
    r0 = np.mean(np.linalg.norm(points - center, axis=1))
    return least_squares(sphere_func, x0=[*center, r0], args=(points,)).x


def fit_ellipsoid(points, fit_rotation=True, fixed_angles=None):
    if fixed_angles is None:
        fixed_angles = [0, 0, 0]
    print(f">> Fitting ellipsoid to {len(points)} points...")
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

    def ellipsoid_func_reduced(params, points):
        if fit_rotation:
            return ellipsoid_func(params, points)
        else:
            return ellipsoid_func(np.concatenate([params, fixed_angles]), points)

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


def generate_sliced_mesh(img, img_scale, xres=1, yres=1, zres=1, normal_vec=None):
    if normal_vec is None:
        normal_vec = [1, 0, 0]
    else:
        normal_vec /= np.linalg.norm(normal_vec)
    print(f">> Generating sliced mesh...")
    img_shape = img.shape
    x = np.linspace(1, (img_shape[0] - 1) * img_scale[0], xres)
    y = np.linspace(1, (img_shape[1] - 1) * img_scale[1], yres)
    z = np.linspace(1, (img_shape[2] - 1) * img_scale[2], zres)
    xx, yy = np.meshgrid(x, y)
    verts_ = np.column_stack([np.tile(xx.ravel(), len(z)), np.tile(yy.ravel(), len(z)), np.repeat(z, xx.size)])
    normals = np.repeat([normal_vec], len(verts_), axis=0)
    mesh = trimesh.Trimesh(vertices=verts_, vertex_normals=normals)
    print(f"Number of slices: {np.min([len(x), len(y), len(z)])} !")
    print(f"Number of vertices: {len(verts_)} !")
    return mesh


#########################
# MESH ANALYSIS MODULES #
#########################
def curvature_by_srf_fit(mesh, num_sample, patch_size=20, patch_mode="nearest", filter_boundary=False, debug=False,
                         gauss_crop_range=None, mean_crop_range=None, boundary_excl_factor=0.1,
                         use_original_vertices=False):
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
        print(f"[!] Unknown patch type: {patch_type}")
        return None

    if filter_boundary:
        neigh_idxs, valid_idxs = filter_valid_patches(verts=verts, idxs_neigh=neigh_idxs, factor=boundary_excl_factor)
        N = len(neigh_idxs)
        random_idxs = random_idxs[valid_idxs]
    tan_x_cov, tan_y_cov = create_tangential_basis(normals=normals, hide_output=True)
    g = np.array([gmetric(tan_x_cov[i], tan_y_cov[i]) for i in range(N)])
    g_inv = np.array([np.linalg.inv(g[i]) for i in range(N)])
    inter_calc = np.array(
        [covariant2contravariant(tan_x_cov[i], tan_y_cov[i]) for i in range(N)])
    tan_x_contr, tan_y_contr = inter_calc[:, 0, :], inter_calc[:, 1, :]
    C_gauss, C_mean = [], []
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
            continue
        S = np.array([[xi ** 2, xi * eta, eta ** 2] for xi, eta in zip(xi_nbs, eta_nbs)])
        b = np.array(z_nbs)
        v, residuals, rank, s_lstq = np.linalg.lstsq(S, b, rcond=None)
        C[0, 0] = -2 * v[0]
        C[0, 1] = -v[1]
        C[1, 0] = -v[1]
        C[1, 1] = -2 * v[2]
        C_mixed = np.dot(C, g_inv[i])
        mean_curvature = np.trace(C_mixed) / 2
        gauss_curvature = np.linalg.det(C) / np.linalg.det(g[i])
        C_gauss.append(gauss_curvature)
        C_mean.append(mean_curvature)
    C_gauss, C_mean = np.asarray(C_gauss), np.asarray(C_mean)
    if debug:
        print(
            f"Gauss: AVG = {np.nanmean(C_gauss)} | MIN = {np.nanmin(C_gauss)} | MAX = {np.nanmax(C_gauss)} | None? = {np.isnan(C_gauss).sum()}")
        print(
            f"Mean: AVG = {np.nanmean(C_mean)} | MIN = {np.nanmin(C_mean)} | MAX = {np.nanmax(C_mean)} | None? = {np.isnan(C_mean).sum()}")

    Gauss_idxs = random_idxs
    if gauss_crop_range is not None:
        print(f">> Cropping Gauss to min {gauss_crop_range[0]}, max {gauss_crop_range[1]}...")
        crop_mask = (gauss_crop_range[0] < C_gauss) & (C_gauss < gauss_crop_range[1]) & (~np.isnan(C_gauss))
        C_gauss, Gauss_idxs = C_gauss[crop_mask], Gauss_idxs[crop_mask]
        if debug:
            print(
                f"Gauss: AVG = {np.nanmean(C_gauss)} | MIN = {np.nanmin(C_gauss)} | MAX = {np.nanmax(C_gauss)} | None? = {np.isnan(C_gauss).sum()}")

    mean_idxs = random_idxs
    if mean_crop_range is not None:
        print(f">> Cropping Mean to min {mean_crop_range[0]}, max {mean_crop_range[1]}...")
        crop_mask = (mean_crop_range[0] < C_mean) & (C_mean < mean_crop_range[1]) & (~np.isnan(C_mean))
        C_mean, mean_idxs = C_mean[crop_mask], mean_idxs[crop_mask]
        if debug:
            print(
                f"Mean: AVG = {np.nanmean(C_mean)} | MIN = {np.nanmin(C_mean)} | MAX = {np.nanmax(C_mean)} | None? = {np.isnan(C_mean).sum()}")

    return C_gauss, C_mean, Gauss_idxs, mean_idxs


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
    dists = dists + eps

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


def select_geodesic_defects(order, mesh, idxs_sel, dist_cutoff=50, max_candidates=10, unit="px"):
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
    # vphi /= np.sin(theta)
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


def contour_masks(masks):
    masks = masks.copy().astype(float)
    masks[masks == 0] = np.nan
    contours_all = []
    for label in np.unique(masks[~np.isnan(masks)]):
        contours = measure.find_contours(masks == label, 0.5)
        contours_all.extend(contours)
    return contours_all


def find_medial_axis(img, scale=(1, 1, 1)):
    medial_axis_raw = np.argwhere(medial_axis(img, mask=img > 0))
    medial_axis_scaled = medial_axis_raw[:, [1, 0]] * scale[1:]
    return medial_axis_scaled


def spline_fit_curve(curve, order_k, num_pts, smooth, sample_interv, start_u, end_u):
    x = curve[::sample_interv, 0]
    y = curve[::sample_interv, 1]
    tck = splprep([x, y], s=smooth, k=order_k)[0]
    curve_fitted = np.array(splev(np.linspace(start_u, end_u, num_pts), tck)).T
    return curve_fitted


def find_pca_axes(img):
    non_zero_indices = np.column_stack(np.where(img > 0))
    center = np.mean(non_zero_indices, axis=0)
    normalized_points = non_zero_indices - center
    pca = PCA(n_components=2)
    pca.fit(normalized_points)
    axes = pca.components_
    return axes, center


def draw_pca_curve(pca_center, pca_axes, dimensions=(1, 1, 1), scale=(1, 1, 1)):
    start_point_curve = pca_center - np.max(dimensions[1:]) / 2 * pca_axes[0]
    end_point_curve = pca_center + np.max(dimensions[1:]) / 2 * pca_axes[0]
    start_point_curve = start_point_curve[[1, 0]]
    end_point_curve = end_point_curve[[1, 0]]
    initial_curve_n = int(
        np.max([end_point_curve[1] - start_point_curve[1], end_point_curve[0] - start_point_curve[0]]))
    initial_curve_xpts = np.linspace(start_point_curve[0], end_point_curve[0], initial_curve_n)
    initial_curve_ypts = np.linspace(start_point_curve[1], end_point_curve[1], initial_curve_n)
    curve = np.column_stack((initial_curve_ypts, initial_curve_xpts))
    curve = curve * scale[1:]
    return curve


def proj2curve(points, curve):
    tree = KDTree(curve)
    dists, idxs = tree.query(points)
    closest_points = curve[idxs]
    s_parallel = np.cumsum(np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1)))
    s_parallel = np.insert(s_parallel, 0, 0)
    s_proj = s_parallel[idxs]
    s_orthogonal = np.linalg.norm(points - closest_points, axis=1)
    return s_proj, s_orthogonal


def filter_curve_inside_shape(curve, image, thresh=0.5, scale=(1, 1)):
    x_int = np.clip(np.round(curve[:, 0] / scale[0]).astype(int), 0, image.shape[1] - 1)
    y_int = np.clip(np.round(curve[:, 1] / scale[1]).astype(int), 0, image.shape[0] - 1)
    mask = image[y_int, x_int] > thresh
    return curve[mask]


def bin_array_with_indices(values, n_bins):
    sorted_indices = np.argsort(values)
    binned_indices = np.array_split(sorted_indices, n_bins)
    return binned_indices


def bin_indices(values, edges):
    bin_ids = np.digitize(values, edges) - 1
    valid = (bin_ids >= 0) & (bin_ids < len(edges) - 1)
    binned = [[] for _ in range(len(edges) - 1)]
    for idx, valid_flag in zip(np.arange(len(values)), valid):
        if valid_flag:
            binned[bin_ids[idx]].append(idx)
    return [np.array(b, dtype=int) for b in binned]


def bin_directors(directors, ap_par_binned_idxs, ap_orth_binned_idxs, curve,
                  nematic_weights):
    ap_par_binned_S_2d, ap_par_binned_n_2d = avg_2d_nem_tens(directors=directors,
                                                             neigh_idxs=ap_par_binned_idxs,
                                                             weights=nematic_weights)
    ap_orth_binned_S_2d, ap_orth_binned_n_2d = avg_2d_nem_tens(directors=directors,
                                                               neigh_idxs=ap_orth_binned_idxs,
                                                               weights=nematic_weights)
    ap_par_binned_S_2d_unw, ap_par_binned_n_2d_unw = avg_2d_nem_tens(directors=directors,
                                                                     neigh_idxs=ap_par_binned_idxs,
                                                                     weights=None)
    ap_orth_binned_S_2d_unw, ap_orth_binned_n_2d_unw = avg_2d_nem_tens(directors=directors,
                                                                       neigh_idxs=ap_orth_binned_idxs,
                                                                       weights=None)

    nematic_results = (ap_par_binned_S_2d, ap_par_binned_n_2d,
                       ap_orth_binned_S_2d, ap_orth_binned_n_2d)
    nematic_results_unweighted = (ap_par_binned_S_2d_unw, ap_par_binned_n_2d_unw,
                                  ap_orth_binned_S_2d_unw, ap_orth_binned_n_2d_unw)
    ap_par_binned_dirs = []
    ap_orth_binned_dirs = []
    s_par_orthogonality = []
    s_orth_orthogonality = []
    s_par_orthogonality_unw = []
    s_orth_orthogonality_unw = []

    curve_tangents = np.diff(curve, axis=0)
    curve_tangents /= np.linalg.norm(curve_tangents, axis=1, keepdims=True)
    curve_pt_tree = KDTree(curve[:-1])

    for i_, parallel_bin_idxs in enumerate(ap_par_binned_idxs):
        ap_par_binned_dirs.append(directors[parallel_bin_idxs])
        _, indices = curve_pt_tree.query(directors[:, :2][parallel_bin_idxs])
        local_normals = np.stack((-curve_tangents[indices][:, 1], curve_tangents[indices][:, 0]), axis=-1)
        local_normals = np.mean(local_normals, axis=0)
        local_normals = local_normals / np.linalg.norm(local_normals, keepdims=True)
        s_par_orthogonality.append(np.abs(np.dot(ap_par_binned_n_2d[i_], local_normals)))
        s_par_orthogonality_unw.append(np.abs(np.dot(ap_par_binned_n_2d_unw[i_], local_normals)))

    for i_, orthogonal_bin_idxs in enumerate(ap_orth_binned_idxs):
        ap_orth_binned_dirs.append(directors[orthogonal_bin_idxs])
        _, indices = curve_pt_tree.query(directors[:, :2][orthogonal_bin_idxs])
        local_normals = np.stack((-curve_tangents[indices][:, 1], curve_tangents[indices][:, 0]), axis=-1)
        local_normals = np.mean(local_normals, axis=0)
        local_normals = local_normals / np.linalg.norm(local_normals, keepdims=True)
        s_orth_orthogonality.append(np.abs(np.dot(ap_orth_binned_n_2d[i_], local_normals)))
        s_orth_orthogonality_unw.append(np.abs(np.dot(ap_orth_binned_n_2d_unw[i_], local_normals)))
    s_par_orthogonality = np.array(s_par_orthogonality)
    s_orth_orthogonality = np.array(s_orth_orthogonality)
    s_par_orthogonality_unw = np.array(s_par_orthogonality_unw)
    s_orth_orthogonality_unw = np.array(s_orth_orthogonality_unw)
    parallel_results = ap_par_binned_dirs, s_par_orthogonality, s_par_orthogonality_unw
    orthogonal_results = ap_orth_binned_dirs, s_orth_orthogonality, s_orth_orthogonality_unw
    return parallel_results, orthogonal_results, nematic_results, nematic_results_unweighted


#############################################
# 2D ORIENTATION & NEMATIC ANALYSIS MODULES #
#############################################
def compute_orientation_with_intensity(img, mode, box_size, dimension=3, calc_energy_coherency=True):
    # print(f">> Computing orientation {dimension}D with window {box_size} in mode {mode}")
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


def avg_2d_nem_tens(directors, neigh_idxs, debug=False, weights=None):
    if debug:
        print("Averaging 2D nematic tensor...")
    n_vecs = directors[:, 2:]
    n_vecs = n_vecs / np.linalg.norm(n_vecs, axis=1, keepdims=True)
    q = n_vecs[:, :, np.newaxis] * n_vecs[:, np.newaxis, :] - (1 / 2) * np.eye(2)[np.newaxis, :, :]

    # --- handle nearest neighbours ---
    if isinstance(neigh_idxs, np.ndarray):
        if weights is None:
            q_avg = np.average(q[neigh_idxs], axis=1)
        else:
            w = weights[neigh_idxs][..., np.newaxis, np.newaxis]
            q_avg = np.average(q[neigh_idxs] * w, axis=1)
    else:
        q_avg = []
        for neigh in neigh_idxs:
            if len(neigh) == 0:
                q_avg.append(np.zeros((2, 2)))
            else:
                if weights is None:
                    q_avg.append(np.mean(q[neigh], axis=0))
                else:
                    w = weights[neigh][..., np.newaxis, np.newaxis]
                    q_avg.append(np.average(q[neigh] * w, axis=0))
        q_avg = np.stack(q_avg)
    eigvals, eigvecs = np.linalg.eigh(q_avg)
    max_indeces = np.argmax(eigvals, axis=1)
    max_eigvals = eigvals[np.arange(len(max_indeces)), max_indeces]
    max_eigvecs = eigvecs[np.arange(len(max_indeces)), :, max_indeces]
    max_eigvecs = max_eigvecs / np.linalg.norm(max_eigvecs, axis=1, keepdims=True)
    n_avg = max_eigvecs
    S_order = max_eigvals * 2
    if debug:
        print(f"director components = \n {n_vecs}")
        print(f"q with shape {q.shape} = \n {q}")
        print(f"q_avg with shape {q_avg.shape} = \n {q_avg}")
        print(f"S_order with shape {S_order.shape} = \n {S_order}")
        print(f"n_avg with shape {n_avg.shape} = \n {n_avg}")
    return S_order, n_avg


def orient2d(img, boxsize, thresh_val, num_neigh_nem=3 ** 2, debug=True):
    if debug:
        print(
            f">> Performing 2D Orientation Analysis for window {boxsize}x{boxsize}, mask threshold {thresh_val} and num_neigh_nem {num_neigh_nem}...")
    img = img.copy().astype(np.float64)
    theta_all_deg = -compute_2d_orientation(mode="fiber", img=img, sampling_box_size=boxsize, onlytheta=True,
                                            debug=debug)
    theta_all_deg = expand_2d_array(theta_all_deg, num=boxsize)
    keepmask = img > thresh_val
    X, Y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    X, Y = X[keepmask.T], Y[keepmask.T]
    theta_all_deg[~keepmask] = None
    theta_masked_rad = np.radians(theta_all_deg[keepmask])
    directors_2d = np.column_stack(
        (X, Y, np.cos(theta_masked_rad), np.sin(theta_masked_rad)))
    neigh_idxs = coord_search_neighbours(directors_2d[:, :2], k=num_neigh_nem, debug=debug)
    S_2d, n_2d = avg_2d_nem_tens(directors=directors_2d, neigh_idxs=neigh_idxs, debug=debug)
    return theta_all_deg, theta_masked_rad, S_2d, n_2d, X, Y, directors_2d


##############################################
# 2D+ ORIENTATION & NEMATIC ANALYSIS MODULES #
##############################################
# def tan_proj(neighbors_coords, central_normal):
#     print(f">> Locally flattening coords ...")
#     central_coord = neighbors_coords[:, 0, :]
#     cross_basis_vector = np.where(np.all(np.isclose(np.cross(central_normal, [1, 0, 0]), 0), axis=1)[:, None],
#                                   [0, 1, 0],
#                                   [1, 0, 0])
#     tangent_x_axes = np.cross(central_normal, cross_basis_vector)
#     tangent_x_axes /= np.linalg.norm(tangent_x_axes, axis=1, keepdims=True)
#     tangent_y_axes = np.cross(central_normal, tangent_x_axes)
#     tangent_y_axes /= np.linalg.norm(tangent_y_axes, axis=1, keepdims=True)
#     translated_points = neighbors_coords - central_coord[:, np.newaxis, :]
#     local_x = np.einsum('nij,nj->ni', translated_points, tangent_x_axes)
#     local_y = np.einsum('nij,nj->ni', translated_points, tangent_y_axes)
#     local_2d_coordinates = np.stack((local_x, local_y), axis=-1)
#     return local_2d_coordinates, tangent_x_axes, tangent_y_axes

def tan_proj(neighbors_coords, central_normal):
    print(f">> Locally flattening coords ...")

    # Detect if input is a list (variable-length neighbours)
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
        # Regular uniform array path (vectorized)
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


# def tan_interp_batch(coords, intensities, grid_size):
#     batch_size, num_points, _ = coords.shape
#     u_min = coords[:, :, 0].min(axis=1)[:, None, None]
#     u_max = coords[:, :, 0].max(axis=1)[:, None, None]
#     v_min = coords[:, :, 1].min(axis=1)[:, None, None]
#     v_max = coords[:, :, 1].max(axis=1)[:, None, None]
#     lin_u = np.linspace(0, 1, grid_size)
#     lin_v = np.linspace(0, 1, grid_size)
#     grid_x, grid_y = np.meshgrid(lin_u, lin_v, indexing="ij")
#     grid_x = u_min + (u_max - u_min) * grid_x
#     grid_y = v_min + (v_max - v_min) * grid_y
#     grid_points = np.stack([grid_x, grid_y], axis=-1).reshape(batch_size, -1, 2)
#     trees = [KDTree(coords[i]) for i in range(batch_size)]
#     idxs = np.array([tree.query(grid_points[i], k=1)[1] for i, tree in enumerate(trees)])
#     grid_z = np.take_along_axis(intensities, idxs, axis=1).reshape(batch_size, grid_size, grid_size)
#     return grid_x, grid_y, grid_z

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


# def batch_2d_orientation(big_grid, box_size, vertices, tan_x, tan_y, debug=False, debug_idx=None,
#                          debug_grid_x=None, debug_grid_y=None, debug_grid_z=None, tan_cords=None, debug_line_length=5):
#     dir_vec = np.zeros(shape=(len(tan_x), 3))
#     if debug:
#         theta_all = compute_2d_orientation(mode="fiber", img=big_grid, sampling_box_size=box_size,
#                                            onlytheta=True) * -1
#         theta_center = np.radians(theta_all[theta_all.shape[0] // 2, theta_all.shape[1] // 2])
#         dir_vec[debug_idx] = np.cos(theta_center) * tan_x[debug_idx] + np.sin(theta_center) * \
#                              tan_y[debug_idx]
#         print(f"=== # {debug_idx} | theta = {np.round(np.degrees(theta_center), 2)} degrees ===")
#         plot_interp_grid(grid_x=debug_grid_x, grid_y=debug_grid_y, grid_z=debug_grid_z,
#                          points=tan_cords[debug_idx], theta=theta_center, linelength=debug_line_length)
#         plot_matrix(theta_all, title="Theta", colorbar=True, origin="lower", cmap_limits=[-90, 90],
#                     remove_axes=True)
#     else:
#         print(f"Running batch analysis for {big_grid.shape} and tensor box size: {box_size} ")
#         theta_all = compute_2d_orientation(mode="fiber", img=big_grid, sampling_box_size=box_size, onlytheta=True) * -1
#         theta_mid_idx = theta_all.shape[1] // 2
#         center_indices = theta_mid_idx + np.arange(0, len(theta_all), theta_all.shape[1])
#         theta_center = np.radians(theta_all[center_indices, theta_mid_idx])
#
#         dir_vec = np.cos(theta_center)[:, None] * tan_x + \
#                   np.sin(theta_center)[:, None] * tan_y
#     directors = np.column_stack((vertices, dir_vec))
#     print(f"Finished! {len(directors)} director(s)!")
#     return directors


def batch_2d_orientation(big_grid, box_size, vertices, tan_x, tan_y, debug=False, debug_idx=None, debug_line_length=5):
    dir_vec = np.zeros(shape=(len(tan_x), 3))
    theta_all = compute_2d_orientation(mode="fiber", img=big_grid, sampling_box_size=box_size, onlytheta=True) * -1

    if debug:
        theta_center = np.radians(theta_all[theta_all.shape[0] // 2, theta_all.shape[1] // 2])
        dir_vec[debug_idx] = np.cos(theta_center) * tan_x[debug_idx] + np.sin(theta_center) * tan_y[debug_idx]
        print(f"=== # {debug_idx} | theta = {np.round(np.degrees(theta_center), 2)} degrees ===")
        # Extract the debug grid for plotting
        debug_grid_z = big_grid[0] if isinstance(big_grid, list) else big_grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, debug_grid_z.shape[0]),
            np.linspace(0, 1, debug_grid_z.shape[1]),
            indexing="ij"
        )
        # Plot the debug grid
        plot_interp_grid(grid_x, grid_y, debug_grid_z, theta=theta_center, linelength=debug_line_length)
        plot_matrix(theta_all, title="Theta", colorbar=True, origin="lower", cmap_limits=[-90, 90],
                    remove_axes=True)
    else:
        # Batch mode for all vertices
        theta_mid_idx = theta_all.shape[1] // 2
        center_indices = theta_mid_idx + np.arange(0, len(theta_all), theta_all.shape[1])
        theta_center = np.radians(theta_all[center_indices, theta_mid_idx])
        dir_vec = np.cos(theta_center)[:, None] * tan_x + np.sin(theta_center)[:, None] * tan_y

    directors = np.column_stack((vertices, dir_vec))
    print(f"Finished! {len(directors)} director(s)!")
    return directors


def avg_tan_nem_tens(t1_cov, t2_cov, directors, neigh_idxs, debug=False):
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

    # --- Handle neighbor averaging ---
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
    return S_order, n_avg


def unique_neighborhoods(arr):
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

    # Ensure director_indeces is a dict for fast lookup
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
        print(f"[!] Unknown patch type: {patch_type}")
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


#############################################
# 3D ORIENTATION & NEMATIC ANALYSIS MODULES #
#############################################
def compute_3d_orientation(mode, img, sampling_box_size, onlydirec=True, upscale_vec=False):
    if onlydirec:
        sampled_orientation = compute_orientation_with_intensity(img=img, mode=mode, box_size=sampling_box_size,
                                                                 dimension=3, calc_energy_coherency=False)
        if upscale_vec:
            return expand_3d_array(sampled_orientation["vector"], num=sampling_box_size)
        else:
            return sampled_orientation["vector"]
    else:
        sampled_orientation = compute_orientation_with_intensity(img=img, mode=mode, box_size=sampling_box_size,
                                                                 dimension=3)
        theta = expand_3d_array(array=sampled_orientation["theta"], num=sampling_box_size)
        phi = expand_3d_array(array=sampled_orientation["phi"], num=sampling_box_size)
        energy = expand_3d_array(array=sampled_orientation["energy"], num=sampling_box_size)
        coherency = expand_3d_array(array=sampled_orientation["coherency"], num=sampling_box_size)
        vectors = expand_3d_array(sampled_orientation["vector"], num=sampling_box_size)
        return theta, phi, energy, coherency, vectors


def avg_3d_nem_tens(directors, neigh_idxs, debug=False):
    print("Averaging full 3D nematic tensor...")
    n_vecs = directors[:, 3:]
    n_vecs = n_vecs / np.linalg.norm(n_vecs, axis=1, keepdims=True)
    q = n_vecs[:, :, np.newaxis] * n_vecs[:, np.newaxis, :] - (1 / 3) * np.eye(3)[np.newaxis, :, :]

    # --- Handle neighbor averaging ---
    if isinstance(neigh_idxs, np.ndarray):
        q_avg = np.average(q[neigh_idxs], axis=1)
    else:  # ragged list
        q_avg = []
        for neigh in neigh_idxs:
            if len(neigh) == 0:
                q_avg.append(np.zeros((3, 3)))
            else:
                q_avg.append(np.mean(q[neigh], axis=0))
        q_avg = np.stack(q_avg)

    eigvals, eigvecs = np.linalg.eigh(q_avg)
    max_indeces = np.argmax(eigvals, axis=1)
    max_eigvals = eigvals[np.arange(len(max_indeces)), max_indeces]
    max_eigvecs = eigvecs[np.arange(len(max_indeces)), :, max_indeces]
    max_eigvecs = max_eigvecs / np.linalg.norm(max_eigvecs, axis=1, keepdims=True)
    n_avg = max_eigvecs
    S_order = max_eigvals * (3 / 2)
    if debug:
        print(f"director components = \n {n_vecs}")
        print(f"q with shape {q.shape} = \n {q}")
        print(f"q_avg with shape {q_avg.shape} = \n {q_avg}")
        print(f"S_order with shape {S_order.shape} = \n {S_order}")
        print(f"n_avg with shape {n_avg.shape} = \n {n_avg}")
    return S_order, n_avg
