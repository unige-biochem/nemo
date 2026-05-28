"""
Data Handling Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import os
import numpy as np
import pandas as pd
import trimesh
from tifffile import tifffile


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

def save_tiff(array, filepath, img_unit, img_scale, scalar_type=np.float32):
    z_pix, y_pix, x_pix = img_scale
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    clean_unit = 'micron' if img_unit.lower() in ['um', 'µm', 'micron', 'microns'] else img_unit
    meta = {
        'unit': clean_unit,
        'spacing': z_pix,
    }
    if array.ndim == 5:
        n_t, n_z, n_c, n_y, n_x = array.shape
        meta.update({'axes': 'TZCYX', 'frames': n_t, 'slices': n_z, 'channels': n_c})
    elif array.ndim == 4:
        n_z, n_c, n_y, n_x = array.shape
        meta.update({'axes': 'ZCYX', 'slices': n_z, 'channels': n_c})
    elif array.ndim == 3:
        n_z, n_y, n_x = array.shape
        meta.update({'axes': 'ZYX', 'slices': n_z})
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")
    tifffile.imwrite(
        filepath,
        array.astype(scalar_type),
        imagej=True,
        resolution=(1.0 / x_pix, 1.0 / y_pix),
        metadata=meta
    )

    print(f">> Saved {meta['axes']} to {filepath}")
    print(f">> Scaling: {x_pix:.4f}x{y_pix:.4f}x{z_pix:.4f} {clean_unit}")


####################
# MESH I/O MODULES #
####################


def load_mesh(filepath):
    print(f">> Loading mesh {filepath}...")
    try:
        mesh = trimesh.load(filepath)
    except:
        print(f"[!] Could not load mesh {filepath}!")
        return None
    mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(vertex_count=len(mesh.vertices),
                                                                   faces=mesh.faces,
                                                                   face_normals=mesh.face_normals,
                                                                   face_angles=mesh.face_angles)
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
