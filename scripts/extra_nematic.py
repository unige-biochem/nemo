"""
Extra Nematic Analysis Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

import matplotlib.pyplot as plt

from scripts.analysis import (
    coord_search_radius,
    compute_orientation_with_intensity,
    coord_search_neighbours,
    compute_2d_orientation,
    expand_2d_array
)


def expand_3d_array(array, num):
    if type(num) is not int:
        return np.repeat(np.repeat(np.repeat(array, num[0], axis=0), num[1], axis=1), num[2], axis=2)
    else:
        return np.repeat(np.repeat(np.repeat(array, num, axis=0), num, axis=1), num, axis=2)


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
    if isinstance(neigh_idxs, np.ndarray):
        q_avg = np.average(q[neigh_idxs], axis=1)
    else:
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


# def compute_defect_polarisations(mesh, idxs_sel, directors, vertex_normals,
#                                  defect_idxs_calc, m_charge, patch_type, patch_size,
#                                  show_profile=False):
#     pol_vec_2d = None
#     pol_positions, pol_vectors, pol_idxs = [], [], []
#
#     directors_v = directors[:, 3:6]
#     directors_v /= np.linalg.norm(directors_v, axis=1, keepdims=True) + 1e-12
#
#     for d_idx, charge in zip(defect_idxs_calc, m_charge):
#         if not (0.4 < abs(charge) < 0.6):
#             continue
#
#         core_idx_sel = idxs_sel[d_idx]
#         core_vertex = mesh.vertices[core_idx_sel]
#         if patch_type == "radius":
#             neigh_idxs_sel = coord_search_radius(mesh.vertices[idxs_sel],
#                                                  custom_probes=[core_vertex],
#                                                  r=patch_size)[0]
#         elif patch_type == "nearest":
#             neigh_idxs_sel = coord_search_neighbours(mesh.vertices[idxs_sel],
#                                                      custom_probes=[core_vertex],
#                                                      k=patch_size)[0]
#         else:
#             raise ValueError(f"Unknown patch type: {patch_type}")
#
#         if len(neigh_idxs_sel) < 10:
#             continue
#
#         neigh_dirs = directors_v[neigh_idxs_sel]
#         neigh_pos = mesh.vertices[idxs_sel[neigh_idxs_sel]]
#         normal = vertex_normals[core_idx_sel]
#         normal /= np.linalg.norm(normal)
#         t1 = np.cross(normal, [1, 0, 0])
#         if np.linalg.norm(t1) < 1e-6:
#             t1 = np.cross(normal, [0, 1, 0])
#         t1 /= np.linalg.norm(t1)
#         t2 = np.cross(normal, t1)
#
#         rel_pos = neigh_pos - core_vertex
#         x = rel_pos @ t1
#         y = rel_pos @ t2
#         theta = np.arctan2(y, x)
#
#         d_proj = np.stack([neigh_dirs @ t1, neigh_dirs @ t2], axis=1)
#         phi = np.arctan2(d_proj[:, 1], d_proj[:, 0])
#         q = np.exp(1j * 2 * phi)
#         base_angle = None
#
#         if charge > 0:  # +1/2 defect
#             ex = np.cos(theta)
#             ey = np.sin(theta)
#             w = np.cos(2 * (phi - theta))
#             pol_vec_2d = np.array([np.sum(w * ex), np.sum(w * ey)])
#             pol_vec_2d /= np.linalg.norm(pol_vec_2d) + 1e-12
#             pol = pol_vec_2d[0] * t1 + pol_vec_2d[1] * t2
#             pol_positions.append(core_vertex)
#             pol_vectors.append(pol)
#             pol_idxs.append(d_idx)
#         else:  # -1/2 defect (threefold symmetric)
#             base_angle = 2 / 3 * np.angle(np.sum(q))
#             for k in range(3):
#                 angle = base_angle + k * 2 * np.pi / 3
#                 pol = np.cos(angle) * t1 + np.sin(angle) * t2
#                 pol_positions.append(core_vertex)
#                 pol_vectors.append(pol)
#                 pol_idxs.append(d_idx)
#         if show_profile:
#             fig, ax = plt.subplots(figsize=(5, 5))
#             ax.set_aspect('equal')
#             ax.set_title(f"Defect {d_idx}, charge {charge:.2f}")
#
#             scale = 0.15 * patch_size
#             ax.quiver(x, y,
#                       np.cos(phi) * scale, np.sin(phi) * scale,
#                       color='blue', alpha=0.5, label='Directors')
#             if charge > 0:
#                 ax.quiver(0, 0,
#                           pol_vec_2d[0] * scale * 3,
#                           pol_vec_2d[1] * scale * 3,
#                           color='red', width=0.02, label='Polarisation')
#             else:
#                 for k in range(3):
#                     angle = base_angle + k * 2 * np.pi / 3
#                     ax.quiver(0, 0,
#                               np.cos(angle) * scale * 3,
#                               np.sin(angle) * scale * 3,
#                               color='red', width=0.02)
#
#             ax.scatter(0, 0, color='k', s=50, marker='x', label='Core')
#             ax.legend()
#             plt.show()
#
#     print(f"Found {len(pol_positions)} polarisation vectors!")
#     return np.column_stack((np.array(pol_positions), np.array(pol_vectors))), np.array(pol_idxs)

import numpy as np


def compute_defect_polarisations(mesh, idxs_sel, directors, vertex_normals,
                                 defect_idxs_calc, m_charge, patch_type, patch_size,
                                 show_profile=False):
    pol_positions, pol_vectors, pol_idxs = [], [], []

    directors_v = directors[:, 3:6]
    directors_v /= np.linalg.norm(directors_v, axis=1, keepdims=True) + 1e-12

    for d_idx, charge in zip(defect_idxs_calc, m_charge):
        # Identify charge sign (s)
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
            plt.show()

    return np.column_stack((np.array(pol_positions), np.array(pol_vectors))), np.array(pol_idxs)


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


def avg_2d_nem_tens(directors, neigh_idxs, debug=False, weights=None):
    if debug:
        print("Averaging 2D nematic tensor...")
    n_vecs = directors[:, 2:]
    n_vecs = n_vecs / np.linalg.norm(n_vecs, axis=1, keepdims=True)
    q = n_vecs[:, :, np.newaxis] * n_vecs[:, np.newaxis, :] - (1 / 2) * np.eye(2)[np.newaxis, :, :]
    if isinstance(neigh_idxs, np.ndarray):
        if weights is None:
            q_avg = np.average(q[neigh_idxs], axis=1)
        else:
            w = weights[neigh_idxs][..., np.newaxis, np.newaxis]
            q_avg = np.average(q[neigh_idxs] * w, axis=1)
    else:
        q_avg = []
        for neigh in neigh_idxs:
            if len(neigh) <= 1:
                q_avg.append(np.full(fill_value=np.nan, shape=(2, 2)))
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
