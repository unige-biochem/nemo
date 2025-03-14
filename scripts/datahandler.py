"""
Data Handling Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import pandas as pd
import os
import numpy as np
import tifffile
import trimesh
from scripts.analysis import clean_mesh


#####################
# ARRAY I/O MODULES #
#####################
def load_array(name, folderpath, return_df=False):
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
    print(f">> Loaded .../{name}.csv | cols {df_cols} | shape {raw_data.shape}")
    if return_df:
        return df_cols, df, raw_data
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
