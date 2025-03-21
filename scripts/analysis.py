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
from scipy.spatial import KDTree
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
import orientationpy as op
from scipy.interpolate import griddata
import scipy.sparse as sp
from skimage.filters import threshold_yen
import trimesh
from trimesh import Trimesh
from scipy.ndimage import binary_fill_holes

from scripts.visuals import plot_dist_kymograph, plot_flattened_neighborhood, plot_interp_grid, plot_matrix
from sklearn.neighbors import KDTree as KDTreeSklearn
import pyvista as pv


#################
# BASIC MODULES #
#################
def normalise_range(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


def coord_search_neighbours(verts, k, n_process=8, debug=False, return_dists=False):
    if debug:
        print(f">> Searching {k} neighbours ...")
    tree = KDTree(verts)
    dists, idxs = tree.query(verts, k=k, workers=n_process)
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


def coord_search_radius(verts, r):
    print(f">> Searching neighbours within radius {r} ...")
    tree = KDTreeSklearn(verts)
    idxs = tree.query_radius(verts, r=r)
    return idxs


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


def create_tangential_basis(normals, hide_output=True):
    if not hide_output:
        print(f">> Creating tangential basis ...")
    cross_basis_vector = np.where(np.all(np.isclose(np.cross(normals, [1, 0, 0]), 0), axis=1)[:, None],
                                  [0, 1, 0], [1, 0, 0])
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


def find_valid_patches(mesh, idxs_neigh, factor=1.5):
    centroids_patches = np.mean(mesh.vertices[idxs_neigh], axis=1)
    reldist = np.linalg.norm(mesh.vertices - centroids_patches, axis=1)
    patchreldistcutoff = factor * np.average(reldist)
    outside_mask = reldist > patchreldistcutoff
    valid_patch_idxs = np.where(~outside_mask)[0]
    print(f"{len(valid_patch_idxs)} valid, {len(centroids_patches) - len(valid_patch_idxs)} invalid ! ")
    return valid_patch_idxs


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
    print(f">> Importing image {path}...")
    img_raw = imread(path)
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
            xyres_found = True
        except:
            xscale, yscale = 1, 1
            xyres_found = False
            print(f"XY scaling absent... using default {xscale, yscale} instead !")

        # ======== Z resolution ========
        try:
            zscale = float(next(line for line in tif.pages[0].tags.get('ImageDescription', None).value.splitlines() if
                                line.startswith("spacing=")).split('=')[1])
            print(f"Found zscale = {zscale} !")
        except:
            if xyres_found:
                zscale = np.average([yscale, xscale]) * np.average(img_dim[1:]) / img_dim[0]
            else:
                zscale = np.average(img_dim[1:]) / img_dim[0]
            print(f"Z spacing absent... estimating {zscale} !")
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


def thresh_img(img, thresh, inverse=True):
    print(f">> Applying threshold {thresh}...")
    img_thresh = np.copy(img)
    if inverse:
        mask = img < thresh
    else:
        mask = img > thresh
    img_thresh[mask] = 0
    img_thresh[~mask] = 1
    return img_thresh


def fill_holes_img(img):
    print(f">> Filling holes in img...")
    return binary_fill_holes(img)


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


###########################
# MESH PROCESSING MODULES #
###########################
def clean_mesh(mesh, debug=False):
    mesh = mesh.copy()
    verts, faces, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
    invalid_normals = np.linalg.norm(normals, axis=1) < 0.9
    valid_verts = verts[~invalid_normals]
    valid_normals = normals[~invalid_normals]
    vertex_map = np.cumsum(~invalid_normals) - 1
    invalid_face_mask = np.any(invalid_normals[faces], axis=1)
    valid_faces = vertex_map[faces[~invalid_face_mask]]
    if debug:
        print(f"Cleaned {np.sum(invalid_normals)} out of {len(mesh.vertex_normals)} !")
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
def curvature_by_srf_fit(mesh, k=4 ** 2, debug=False, gauss_crop_range=None, mean_crop_range=None):
    print(f">> Calculating curvature for mesh with k={k} and {len(mesh.vertices)} vertices...")
    verts = mesh.vertices
    N = len(verts)
    normals = mesh.vertex_normals
    neigh_idxs, neigh_dists = coord_search_neighbours(verts=verts, k=k, debug=False, return_dists=True)
    neigh_verts = verts[neigh_idxs]
    neigh_normals = normals[neigh_idxs]
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
        for j in range(1, k):
            drvec = neigh_verts[i, j] - neigh_verts[i, 0]
            z_nb = np.dot(drvec, neigh_normals[i, 0])
            z_nbs.append(z_nb)
            xi_nb = np.dot(drvec, tan_x_contr[i])
            xi_nbs.append(xi_nb)
            eta_nb = np.dot(drvec, tan_y_contr[i])
            eta_nbs.append(eta_nb)

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
            f"Gauss: AVG = {np.average(C_gauss)} | MIN = {np.min(C_gauss)} | MAX = {np.max(C_gauss)} | None? = {np.isnan(C_gauss).sum()}")
        print(
            f"Mean: AVG = {np.average(C_mean)} | MIN = {np.min(C_mean)} | MAX = {np.max(C_mean)} | None? = {np.isnan(C_mean).sum()}")

    Gauss_idxs = np.arange(0, verts.shape[0])
    if gauss_crop_range is not None:
        print(f">> Cropping Gauss to min {gauss_crop_range[0]}, max {gauss_crop_range[1]}...")
        crop_mask = (gauss_crop_range[0] < C_gauss) & (C_gauss < gauss_crop_range[1])
        C_gauss, Gauss_idxs = C_gauss[crop_mask], Gauss_idxs[crop_mask]
        if debug:
            print(
                f"Gauss: AVG = {np.average(C_gauss)} | MIN = {np.min(C_gauss)} | MAX = {np.max(C_gauss)} | None? = {np.isnan(C_gauss).sum()}")

    mean_idxs = np.arange(0, verts.shape[0])
    if mean_crop_range is not None:
        print(f">> Cropping Mean to min {mean_crop_range[0]}, max {mean_crop_range[1]}...")
        crop_mask = (mean_crop_range[0] < C_mean) & (C_mean < mean_crop_range[1])
        C_mean, mean_idxs = C_mean[crop_mask], mean_idxs[crop_mask]
        if debug:
            print(
                f"Mean: AVG = {np.average(C_mean)} | MIN = {np.min(C_mean)} | MAX = {np.max(C_mean)} | None? = {np.isnan(C_mean).sum()}")

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


def inter_dist_mesh(mesh_1, mesh_2, debug=False, crop_range=None):
    print(f">> Calculating distance between two meshes ({mesh_1.vertices.shape[0]}, {mesh_2.vertices.shape[0]}) ...")
    vertices_1, normals_1 = mesh_1.vertices, mesh_1.vertex_normals
    ray_origins, ray_directions = vertices_1, normals_1
    locations, index_ray, index_tri = mesh_2.ray.intersects_location(ray_origins, ray_directions)
    distances = np.full(len(vertices_1), np.nan)
    dist_vector = locations - vertices_1[index_ray]
    distances[index_ray] = np.einsum('ij,ij->i', dist_vector, normals_1[index_ray])
    valid_mask = ~np.isnan(distances)
    valid_indeces = np.arange(0, vertices_1.shape[0])[valid_mask]
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


######################
# PROJECTION MODULES #
######################
def proj2mesh(img, mesh, scale, unit, min_dist, max_dist, num_dist, mode, show_proj=False,
              figsize=(7, 5), interp_method="linear", return_full=False, cmap="inferno", savefig=""):
    print(f">> Projecting {mode} image intensities on verts using {interp_method} interpolation method...")
    print(f"Range: MIN = {min_dist}{unit}, MAX = {max_dist}{unit}, NUM = {num_dist}")
    verts, normals = mesh.vertices, mesh.vertex_normals
    distances = np.linspace(min_dist, max_dist, num_dist)
    distances_reshaped = distances.reshape(1, num_dist, 1)
    img_grid = (
        np.arange(img.shape[0]) * scale[0], np.arange(img.shape[1]) * scale[1], np.arange(img.shape[2]) * scale[2])
    interp_function = RegularGridInterpolator(points=img_grid, values=img, method=interp_method, bounds_error=False,
                                              fill_value=-1)
    outward_samples = verts[:, None, :] + normals[:, None, :] * distances_reshaped
    sampling_points = outward_samples.reshape(-1, 3)

    intensities = interp_function(sampling_points)
    intensities = intensities.reshape(len(verts), -1)
    masked_intensities = np.ma.masked_where(intensities == -1, intensities)

    plot_mask = np.copy(intensities)
    plot_mask[plot_mask == -1] = None
    if show_proj:
        plot_dist_kymograph(distances=distances, plot_mask=plot_mask, plot_all=masked_intensities, cmap=cmap,
                            figsize=figsize, unit=unit, savefig=savefig)
    if mode == "max":
        max_intensity_values = np.max(masked_intensities, axis=1)
    elif mode == "mean":
        max_intensity_values = np.mean(masked_intensities, axis=1)
    else:
        return None

    if return_full:
        return distances, outward_samples, masked_intensities

    max_intensity_values = normalise_range(max_intensity_values)
    return max_intensity_values


def mercator_project(pts, ref_point=None, rotate=None, debug=False):
    print(">> Applying Mercator Projection...")
    if ref_point is not None:
        centered_inner_verts = pts - ref_point
    else:
        centered_inner_verts = pts - np.mean(pts, axis=0)
    if rotate is not None:
        centered_inner_verts @= rot3dmatrix(alpha=rotate[0], beta=rotate[1], gamma=rotate[2])
    x, y, z = centered_inner_verts[:, 0], centered_inner_verts[:, 1], centered_inner_verts[:, 2]

    mercator_rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    mercator_x = np.degrees(np.arctan2(y, x))
    mercator_y = np.degrees(np.arccos(z / mercator_rho))
    if debug:
        print(f"Min X = {np.min(mercator_x)} | MAX X = {np.max(mercator_x)}")
        print(f"Min Y = {np.min(mercator_y)} | MAX Y = {np.max(mercator_y)}")
    return mercator_x, mercator_y


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


def avg_2d_nem_tens(directors, neigh_idxs, debug=False):
    if debug:
        print("Averaging 2D nematic tensor...")
    n_vecs = directors[:, 2:]
    n_vecs = n_vecs / np.linalg.norm(n_vecs, axis=1, keepdims=True)
    q = n_vecs[:, :, np.newaxis] * n_vecs[:, np.newaxis, :] - (1 / 2) * np.eye(2)[np.newaxis, :, :]
    q_avg = np.average(q[neigh_idxs], axis=1)
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


##############################################
# 2D+ ORIENTATION & NEMATIC ANALYSIS MODULES #
##############################################
def tan_proj(neighbors_coords, central_normal):
    print(f">> Locally flattening coords ...")
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


def tan_interp(coords, intensities, grid_size):
    u_min, u_max = coords[:, 0].min(), coords[:, 0].max()
    v_min, v_max = coords[:, 1].min(), coords[:, 1].max()
    grid_x, grid_y = np.meshgrid(np.linspace(u_min, u_max, grid_size), np.linspace(v_min, v_max, grid_size))
    grid_z = griddata(coords, intensities, (grid_x, grid_y), method='nearest', fill_value=np.nan)
    return grid_x, grid_y, grid_z


def tan_interp_batch(coords, intensities, grid_size):
    batch_size, num_points, _ = coords.shape
    u_min = coords[:, :, 0].min(axis=1)[:, None, None]
    u_max = coords[:, :, 0].max(axis=1)[:, None, None]
    v_min = coords[:, :, 1].min(axis=1)[:, None, None]
    v_max = coords[:, :, 1].max(axis=1)[:, None, None]
    lin_u = np.linspace(0, 1, grid_size)
    lin_v = np.linspace(0, 1, grid_size)
    grid_x, grid_y = np.meshgrid(lin_u, lin_v, indexing="ij")
    grid_x = u_min + (u_max - u_min) * grid_x
    grid_y = v_min + (v_max - v_min) * grid_y
    grid_points = np.stack([grid_x, grid_y], axis=-1).reshape(batch_size, -1, 2)
    trees = [KDTree(coords[i]) for i in range(batch_size)]
    idxs = np.array([tree.query(grid_points[i], k=1)[1] for i, tree in enumerate(trees)])
    grid_z = np.take_along_axis(intensities, idxs, axis=1).reshape(batch_size, grid_size, grid_size)
    return grid_x, grid_y, grid_z


def batch_2d_orientation(big_grid, box_size, vertices, tan_x, tan_y, idxs_sel, debug=False, debug_idx=None,
                         debug_grid_x=None, debug_grid_y=None, debug_grid_z=None, tan_cords=None):
    dir_vec = np.zeros(shape=(len(vertices), 3))
    if debug:
        theta_all = compute_2d_orientation(mode="fiber", img=big_grid, sampling_box_size=box_size,
                                           onlytheta=True) * -1
        theta_center = np.radians(theta_all[theta_all.shape[0] // 2, theta_all.shape[1] // 2])
        dir_vec[debug_idx] = np.cos(theta_center) * tan_x[debug_idx] + np.sin(theta_center) * \
                             tan_y[debug_idx]
        print(f"=== # {debug_idx} | theta = {np.round(np.degrees(theta_center), 2)} degrees ===")
        plot_interp_grid(grid_x=debug_grid_x, grid_y=debug_grid_y, grid_z=debug_grid_z,
                         points=tan_cords[debug_idx], theta=theta_center)
        plot_matrix(theta_all, title="Theta", aspect=1, colorbar=True, origin="lower", cmap_limits=[-90, 90],
                    remove_axes=True)
    else:
        print(f"Running batch analysis for {big_grid.shape} and tensor box size: {box_size} ")
        theta_all = compute_2d_orientation(mode="fiber", img=big_grid, sampling_box_size=box_size, onlytheta=True) * -1
        theta_mid_idx = theta_all.shape[1] // 2
        center_indices = theta_mid_idx + np.arange(0, len(theta_all), theta_all.shape[1])
        theta_center = np.radians(theta_all[center_indices, theta_mid_idx])

        dir_vec[idxs_sel] = np.cos(theta_center)[:, None] * tan_x[idxs_sel] + \
                            np.sin(theta_center)[:, None] * tan_y[idxs_sel]
    directors = np.column_stack((vertices[idxs_sel], dir_vec[idxs_sel]))
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

    q_tilde_avg = np.average(q_tilde[neigh_idxs], axis=1)
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


def patch_surface_integral(mesh, value, patch_idxs, debug=False):
    patch_integrals = []
    for i in range(patch_idxs.shape[0]):
        patch_faces = mesh.faces[np.isin(mesh.faces, patch_idxs[i]).all(axis=1)]
        v0, v1, v2 = patch_faces.T
        p0, p1, p2 = mesh.vertices[v0], mesh.vertices[v1], mesh.vertices[v2]
        areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / 2
        avg_val = np.nanmean([value[v0], value[v1], value[v2]])
        patch_integrals.append(np.nansum(avg_val * areas))
    patch_integrals = np.array(patch_integrals)
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


def top_charge_loop_integral(loop_idxs, directors, director_indeces, normals, debug=False):
    topological_charges = []
    for i in range(len(loop_idxs)):
        patch_indeces = loop_idxs[i]
        normal_sel = normals[patch_indeces]
        dir_indices = np.searchsorted(director_indeces, patch_indeces)
        directors_vec_sel = directors[dir_indices][:, 3:]
        p_current = directors_vec_sel
        p_next = np.roll(directors_vec_sel, -1, axis=0)
        pdiff = p_next - p_current
        dot_prod_pdiff_normal = np.einsum('ij,ij->i', pdiff, normal_sel)
        dot_prod_normal_normal = np.einsum('ij,ij->i', normal_sel, normal_sel)
        pdiff_proj = pdiff - (dot_prod_pdiff_normal[:, None] / dot_prod_normal_normal[:, None]) * normal_sel
        cross_pdiff_proj_pcurrent = np.cross(pdiff_proj, p_current)
        dot_prod_pdiff_proj_normal = np.einsum('ij,ij->i', cross_pdiff_proj_pcurrent, normal_sel)
        print(np.sum(dot_prod_pdiff_proj_normal))
        m = np.sum(dot_prod_pdiff_proj_normal) / (2 * np.pi)
        topological_charges.append(m)
    if debug:
        print(f"-- line charge m = {topological_charges}")
    return topological_charges


def curved_nem_charge(mesh, directors, calc_idxs, director_indeces, tan_x, tan_y, c_gauss, loop_angle_precision=1,
                      k_charge=5 ** 2, debug=False, return_all_contributions=False):
    print(f">> Calculating topological charge for {len(directors)} directors...")
    charge_patch_broad = coord_search_neighbours(mesh.vertices, k=k_charge, debug=debug)[director_indeces][calc_idxs]
    m_gauss_contribution = patch_surface_integral(mesh=mesh, value=c_gauss,
                                                  patch_idxs=charge_patch_broad,
                                                  debug=debug) / (2 * np.pi)
    calc_charge_loop_idxs = find_boundary_indeces(mesh, [np.intersect1d(loop, director_indeces) for loop in
                                                         charge_patch_broad],
                                                  tan_x=tan_x, tan_y=tan_y,
                                                  angle_precision=loop_angle_precision)
    m_line_charge = top_charge_loop_integral(loop_idxs=calc_charge_loop_idxs, directors=directors,
                                             director_indeces=director_indeces,
                                             normals=mesh.vertex_normals, debug=debug)
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
    q_avg = np.average(q[neigh_idxs], axis=1)
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
