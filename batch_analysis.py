"""
Example Batch Analysis Script for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

# Import custom scripts
from scripts import analysis, datahandler, visuals, simulation
from importlib import reload

for module in (analysis, datahandler, visuals, simulation):
    reload(module)

# Import python essentials
import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

magdalena_root = "/Users/andreadi/Library/CloudStorage/OneDrive-UniversitédeGenève/Academic/Data/4_collaborations/magdalena_embl"
# magdalena_rel_path = "TP1/C2-CTL_TG_His_lyn_E29_20240423_block4_New-02-0To7-TP1.tif"
magdalena_rel_path = "TP80/C2-CTL_TG_His_lyn_E29_20240423_block4_New-02-0To7-TP80.tif"
img_path = os.path.join(magdalena_root, magdalena_rel_path)

print(f"Selected image path: {img_path}")

# ==== Create Folder Structure ====
resdata_dir, resfig_dir = datahandler.create_resdirs(img_path)

# ==== Optional: Reduce Resolution ====
z_reduce_factor = 1
xy_reduce_factor = 1

# ==== Optional: Normalise Intensities to [0, 1] ====
normalise_intensities = True

# ==== Load Image ====
img_raw, img_dim, img_scale, img_unit = analysis.load_img(path=img_path, reduce_xy=xy_reduce_factor,
                                                          reduce_z=z_reduce_factor,
                                                          norm_vals=normalise_intensities,
                                                          custom_scaling=None)

# ==== Plot Image Slices and Max Projections ====
visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, max_proj=True, cmap='Greens',
                 savefig=os.path.join(resfig_dir, "sliced_maxproj_raw.png"))
z_i, y_i, x_i = int(img_dim[0] // 2), int(img_dim[1] // 2), int(img_dim[2] // 2)
visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, x_i=x_i, y_i=y_i, z_i=z_i,
                 savefig=os.path.join(resfig_dir, "sliced_raw.png"), cmap="Greens_r")

extract_mesh = False
if extract_mesh:
    # ==== Blur Image ====
    sigma = 5
    sigma_ = analysis.rescale_val_xyz(val=sigma, scale=img_scale)
    img_blur = analysis.gaussian_blur(img=img_raw, sigma=sigma_, renorm=True)

    # ==== Plot Image Slices ====
    visuals.plot_img(img=img_blur, scale=img_scale, unit=img_unit, cmap="inferno",
                     savefig=os.path.join(resfig_dir, "sliced_blur.png"))

    # ==== Yen Threshold Image ====
    # img_thresh_val = np.min([analysis.yen_thresh(img_blur[:, :, img_dim[2] // i]) for i in np.arange(2, 6)])
    img_thresh_val = np.min(
        [analysis.yen_thresh(img_blur[:, :, img_dim[2] // 2]),
         analysis.yen_thresh(img_blur[:, img_dim[1] // 2, :]),
         analysis.yen_thresh(img_blur[img_dim[0] // 2, :, :])])
    img_thresh_val *= 0.11

    # ==== Binarise Image using Threshold ====
    img_thresh = analysis.thresh_img(img_blur, img_thresh_val, inverse=True)

    # ==== Plot Image Slices ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit, savefig=os.path.join(resfig_dir, "sliced_thresh.png"),
                     cmap="inferno", thresh_mask=img_thresh)

    # ==== Segment Surface Mesh(es) ====
    mcub_res = 2
    full_mesh = analysis.marching_cubes(img=img_thresh, scale=img_scale, level=0.5, step_size=mcub_res)

    # ==== Select TOP / BOTTOM Mesh ====
    mesh_sel_mask = np.einsum('ij,ij->i', np.array([[1, 0, 0] for i in range(len(full_mesh.vertices))]),
                              full_mesh.vertex_normals) < 0

    # ==== Apply Sub-Mesh Selection ====
    inner_mesh = analysis.sel_submesh(mesh=full_mesh, mask=mesh_sel_mask)
    outer_mesh = analysis.sel_submesh(mesh=full_mesh, mask=~mesh_sel_mask)

    # ==== Save Mesh(es) ====
    datahandler.save_mesh(inner_mesh, os.path.join(resdata_dir, "inner_mesh.ply"))
    print(
        f"Number of vertices: #INNER = {inner_mesh.vertices.shape[0]} + #OUTER = {outer_mesh.vertices.shape[0]} == #FULL = {full_mesh.vertices.shape[0]}!")

    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_raw_full-mesh.png"), meshes=[inner_mesh, outer_mesh],
                     mesh_colors=["red", "blue"])

    # ==== Smooth Mesh(es) ====
    smooth_factor = 0.001
    smooth_iterations = 200
    inner_mesh_smooth = analysis.taubin_smooth_mesh(mesh=inner_mesh, n_iter=smooth_iterations, pass_band=smooth_factor)
    # ==== Save Mesh(es) ====
    datahandler.save_mesh(inner_mesh_smooth, os.path.join(resdata_dir, "inner_mesh_smooth.ply"))

    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_smooth_inner-mesh.png"),
                     meshes=[inner_mesh, inner_mesh_smooth], mesh_colors=["blue", "red"])
    ################################################################################
    # mesh_to_fit = inner_mesh_smooth.copy()
    mesh_to_fit = datahandler.load_mesh(os.path.join(resdata_dir, "inner_mesh_smooth.ply"), recalc_normals=True)
    # ==== Fit Sphere ====
    sphere_params = analysis.fit_sphere(points=mesh_to_fit.vertices)
    sphere_x0, sphere_y0, sphere_z0, sphere_radius = sphere_params
    sphere_mesh = trimesh.creation.icosphere(radius=sphere_radius, subdivisions=7)
    sphere_mesh.vertices += [sphere_x0, sphere_y0, sphere_z0]
    datahandler.save_array(np.array(sphere_params)[:, np.newaxis].T, "sphere_fit", header="x,y,z,radius",
                           folderpath=resdata_dir)
    # ==== Crop Sphere ====
    zmax, zmin = sphere_mesh.vertices[:, 0].max(), sphere_mesh.vertices[:, 0].min()
    sphere_crop_mask = sphere_mesh.vertices[:, 0] > (zmax - zmin) / 1.4 + zmin
    # sphere_crop_mask = np.ones_like(sphere_crop_mask)
    sphere_mesh_cropped = analysis.sel_submesh(mesh=sphere_mesh, mask=sphere_crop_mask)
    sphere_mesh_cropped = analysis.subdivide_mesh(sphere_mesh_cropped, 2)
    print(f"Num of sphere vertices: {sphere_mesh_cropped.vertices.shape[0]}")

    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_raw_sphere-fit.png"), cmap="Greens_r",
                     meshes=[sphere_mesh_cropped], mesh_alpha=1.0)

    ################################################################################
    # ==== Select Sampling Mesh ====
    # sampl_mesh = full_mesh_smooth.copy()
    sampl_mesh = sphere_mesh_cropped.copy()
    # sampl_mesh = inner_mesh_smooth.copy()

    # ==== Save Sampling Mesh ====
    datahandler.save_mesh(sampl_mesh, os.path.join(resdata_dir, "sampling_mesh.ply"))

    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_raw_sampling-mesh.png"), meshes=[sampl_mesh])
    print(f"Number of sampling points: {len(sampl_mesh.vertices)} !")
else:
    # ==== Load Sampling Mesh ====
    sampl_mesh = datahandler.load_mesh(os.path.join(resdata_dir, "sampling_mesh.ply"), recalc_normals=True, clean=True)
    print(f"Number of sampling vertices: {len(sampl_mesh.vertices)} !")
    # sampl_mesh.show()
    # ==== Plot Image Slices with Mesh Overlay ====
    visuals.plot_img(img=img_raw, scale=img_scale, unit=img_unit,
                     savefig=os.path.join(resfig_dir, "sliced_raw_sampling-mesh.png"), meshes=[sampl_mesh])
################################################################################
# ==== Define Projection Range ====
dist_min = -140
dist_max = -120
dist_num = 20

# ==== Project onto Mesh ====
proj_broad = analysis.proj2mesh(img=img_raw, mesh=sampl_mesh, unit=img_unit, min_dist_per_vert=None,
                                scale=img_scale, min_dist=dist_min, max_dist=dist_max, num_dist=dist_num, mode="mean",
                                show_proj=True,
                                savefig=os.path.join(resfig_dir, f"{dist_min}-{dist_max}_distgraph_broad-scan.png"))

################################################################################
# ==== Experimental: Mercator Projection relative to Sphere Fit ====
sphere_params_load = datahandler.load_array("sphere_fit", folderpath=resdata_dir)
sphere_x0, sphere_y0, sphere_z0, sphere_radius = sphere_params_load[0, :]
mercator_x, mercator_y = analysis.mercator_project(
    pts=sampl_mesh.vertices, ref_point=[sphere_x0, sphere_y0, sphere_z0])
visuals.plot_mercator_project(mercator_x=mercator_x, mercator_y=mercator_y,
                              intensities=proj_broad, figsize=(14, 8), ptview=True, aspect="equal",
                              savefig=os.path.join(resfig_dir,
                                                   f"{dist_min}-{dist_max}_distgraph_broad-scan_mercator.png"))
