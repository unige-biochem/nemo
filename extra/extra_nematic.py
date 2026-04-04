"""
Extra Nematic Analysis Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from module_scripts.analysis import (
    compute_orientation_with_intensity,
    coord_search_neighbours,
    compute_2d_orientation,
    expand_2d_array
)

# Import python essentials
import numpy as np


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
