# Import NEMO scripts
import os, re, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts import analysis, datahandler, visuals

# Import python essentials
import numpy as np
import trimesh


def find_meshes(folder):
    pattern = re.compile(r"(\d+)h_(\d+)_([A-Za-z0-9]+)_sampling_mesh\.ply")
    mesh_paths = []

    for f in os.listdir(folder):
        if pattern.match(f):
            mesh_paths.append(os.path.join(folder, f))

    if not mesh_paths:
        print(">> No matching PLYs found.")
        return None
    else:
        return mesh_paths


mesh_dirs = find_meshes(folder="/Users/andreadi/Desktop/all_meshes")
all_meshes = [datahandler.load_mesh(filepath=mesh_path, recalc_normals=False, clean=False) for mesh_path in mesh_dirs]
all_meshes_joined = trimesh.util.concatenate(all_meshes)
visuals.view_mesh([all_meshes_joined])