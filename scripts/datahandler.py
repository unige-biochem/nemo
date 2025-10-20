"""
Data Handling Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import pandas as pd
import os, re
import numpy as np
import tifffile
import trimesh
from scripts.analysis import clean_mesh
from PIL import Image
import imageio.v2 as imageio


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
def load_tiff(filepath):
    print(f">> Loading tif {filepath}...")
    return tifffile.imread(filepath)


def save_tiff(array, filepath):
    print(f">> Saving tiff {filepath}...")
    create_dir(os.path.dirname(filepath))
    tifffile.imwrite(filepath, array.astype(np.float32))


def load_png(filepath):
    if not os.path.exists(filepath):
        print(f"{filepath} does not exist, exiting...")
        return None
    else:
        return np.array(Image.open(filepath))


def save_gif_multiple(folderpath, frame_len_ms=100):
    files = np.array(sorted(f for f in os.listdir(folderpath) if re.match(r"z-\d+_.+\.png", f)))
    suffixes = np.unique([re.match(r"z-\d+(_.+)\.png", f).group(1) for f in files])
    for suf in suffixes:
        print(f">> Saving gif for {suf}...")
        matched = sorted([f for f in files if f.endswith(suf + ".png")],
                         key=lambda x: int(re.search(r"z-(\d+)_", x).group(1)))
        imgs = list(map(lambda f: Image.open(os.path.join(folderpath, f)), matched))
        imgs[0].save(os.path.join(folderpath, suf[1:] + ".gif"), save_all=True, append_images=imgs[1:],
                     duration=frame_len_ms, loop=0)
    print(f">> Saved {len(suffixes)} gif files in {folderpath}!")


def save_video_multiple(folderpath, frame_len_ms=100, ext="mp4"):
    files = np.array(sorted(f for f in os.listdir(folderpath) if re.match(r"z-\d+_.+\.png", f)))
    suffixes = np.unique([re.match(r"z-\d+(_.+)\.png", f).group(1) for f in files])
    fps = 1000 / frame_len_ms

    def pad_to_size(img, target_shape, fill=0):
        h, w = img.shape[:2]
        th, tw = target_shape[:2]
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

        # pad images to same size
        max_h = max(img.shape[0] for img in imgs)
        max_w = max(img.shape[1] for img in imgs)
        target_shape = (max_h, max_w, imgs[0].shape[2] if imgs[0].ndim == 3 else 1)
        imgs = [pad_to_size(img, target_shape) for img in imgs]

        # pad to multiple of 16 for ffmpeg
        imgs = [pad_to_multiple_of_16(img) for img in imgs]

        outpath = os.path.join(folderpath, suf[1:] + f".{ext}")
        imageio.mimsave(outpath, imgs, fps=fps)

    print(f">> Saved {len(suffixes)} {ext} files in {folderpath}!")


####################
# MESH I/O MODULES #
####################
def load_mesh(filepath, recalc_normals, clean=True):
    print(f">> Loading mesh {filepath}...")
    mesh = trimesh.load(filepath)
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
        # print(f">> Creating directory {directory} ...")
        os.makedirs(directory)


def create_resdirs(path):
    # print(f">> Creating result paths for {path}...")
    resfig_dir = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], "figures")
    resdata_dir = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], "data")
    create_dir(resfig_dir)
    create_dir(resdata_dir)
    return resdata_dir, resfig_dir
