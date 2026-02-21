"""
Data Handling Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import os
import re

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tifffile
import trimesh


#####################
# ARRAY I/O MODULES #
#####################
def load_array(name, folderpath, return_df=False, debug=True):
    filepath = os.path.join(folderpath, f"{name}.csv")
    if not os.path.exists(filepath):
        print(f"{filepath} does not exist, exiting...")
        return None
    df = pd.read_csv(filepath, delimiter=",", header=0)
    df.columns = [col.lstrip('# ').strip() for col in df.columns]
    df_cols = list(df.columns)
    raw_data = df.to_numpy()
    raw_data_shape = raw_data.shape
    if len(raw_data_shape) == 2 and raw_data_shape[1] == 1:
        raw_data = raw_data.ravel()
    if debug:
        print(f">> Loaded .../{name}.csv | cols {df_cols} | shape {raw_data.shape}")
    if return_df:
        return df
    else:
        return raw_data


def save_array(array, name, header, folderpath, delimiter=","):
    create_dir(folderpath)
    filepath = os.path.join(folderpath, f"{name}.csv")
    np.savetxt(filepath, array, delimiter=delimiter, header=header)
    print(f">> Saved {filepath} !")


#####################
# IMAGE I/O MODULES #
#####################


def save_tiff(array, filepath):
    print(f">> Saving tiff {filepath}...")
    create_dir(os.path.dirname(filepath))
    tifffile.imwrite(filepath, array.astype(np.float32))


def save_video_multiple(folderpath, frame_len_ms=100, ext="mp4"):
    files = np.array(sorted(f for f in os.listdir(folderpath) if re.match(r"z-\d+_.+\.png", f)))
    suffixes = np.unique([re.match(r"z-\d+(_.+)\.png", f).group(1) for f in files])
    fps = 1000 / frame_len_ms

    def pad_to_size(img, shape_, fill=0):
        h, w = img.shape[:2]
        th, tw = shape_[:2]
        pad_top, pad_bottom = (th - h) // 2, th - h - (th - h) // 2
        pad_left, pad_right = (tw - w) // 2, tw - w - (tw - w) // 2
        if img.ndim == 2:
            return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=fill)
        else:
            return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=fill)

    def pad_to_multiple_of_16(img):
        h, w = img.shape[:2]
        th = ((h + 15) // 16) * 16
        tw = ((w + 15) // 16) * 16
        return pad_to_size(img, (th, tw, img.shape[2] if img.ndim == 3 else 1))

    for suf in suffixes:
        print(f">> Saving {ext} for {suf}...")
        matched = sorted([f for f in files if f.endswith(suf + ".png")],
                         key=lambda x: int(re.search(r"z-(\d+)_", x).group(1)))
        imgs = [imageio.imread(os.path.join(folderpath, f)) for f in matched]
        max_h = max(img.shape[0] for img in imgs)
        max_w = max(img.shape[1] for img in imgs)
        target_shape = (max_h, max_w, imgs[0].shape[2] if imgs[0].ndim == 3 else 1)
        imgs = [pad_to_size(img, target_shape) for img in imgs]
        imgs = [pad_to_multiple_of_16(img) for img in imgs]
        outpath = os.path.join(folderpath, suf[1:] + f".{ext}")
        imageio.mimsave(outpath, imgs, fps=fps)

    print(f">> Saved {len(suffixes)} {ext} files in {folderpath}!")


####################
# MESH I/O MODULES #
####################

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


def load_mesh(filepath, recalc_normals, clean=True):
    print(f">> Loading mesh {filepath}...")
    try:
        mesh = trimesh.load(filepath)
    except:
        print(f"[!] Could not load mesh {filepath}!")
        return None
    if recalc_normals:
        print(f">> Recomputing normals ...")
        mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(vertex_count=len(mesh.vertices),
                                                                       faces=mesh.faces,
                                                                       face_normals=mesh.face_normals,
                                                                       face_angles=mesh.face_angles)
    if clean:
        return clean_mesh(mesh=mesh)
    else:
        return mesh


def save_mesh(mesh, filepath):
    print(f">> Saving mesh to {filepath}...")
    create_dir(os.path.dirname(filepath))
    _ = mesh.export(filepath)


################################
# DIRECTORY MANAGEMENT MODULES #
################################
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_resdirs(path, ct_label=None):
    if ct_label is not None:
        resfig_dir = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0],
                                  ct_label, "figures")
        resdata_dir = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0],
                                   ct_label, "data")
    else:
        resfig_dir = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], "figures")
        resdata_dir = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], "data")
    create_dir(resfig_dir)
    create_dir(resdata_dir)
    return resdata_dir, resfig_dir
