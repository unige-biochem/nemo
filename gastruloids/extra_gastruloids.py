"""
Extra Gastruloid Analysis Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################

import os

import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_fill_holes
from scipy.spatial import KDTree
from skimage import measure
from skimage.morphology import skeletonize_3d, medial_axis
from sklearn.decomposition import PCA

from module_scripts.analysis import coord_search_radius, avg_tan_nem_tens
from module_scripts.datahandler import create_resdirs, load_mesh, load_array, save_array
from module_scripts.visuals import plot_qsphi_profiles, plot_qsphi_profiles_separated_phi, plot_scatter, \
    plot_rho_profile, \
    plot_cylindrical_projection, plot_s_sphi_profile


def load_png(filepath):
    if not os.path.exists(filepath):
        print(f"{filepath} does not exist, exiting...")
        return None
    else:
        return np.array(Image.open(filepath))


def skeletonise_mesh(mesh, voxel_size):
    voxelized = mesh.voxelized(pitch=voxel_size)
    voxelized_grid = voxelized.matrix.astype(np.uint8)
    voxelized_grid = binary_fill_holes(voxelized_grid)
    skel = skeletonize_3d(voxelized_grid)
    skel_coords = np.ceil(np.argwhere(skel > 0) * voxelized.pitch + voxelized.translation).astype(int)
    print(f"Found {len(skel_coords)} skeleton points!")
    return skel_coords, voxelized_grid


def order_line_points(points, start_idx=None, neigh_k=6):
    p = np.asarray(points)
    n = len(p)
    tree = KDTree(p)

    if start_idx is None:
        start_idx = np.argmin(p[:, 2])

    ordered = [start_idx]
    used = np.zeros(n, dtype=bool)
    used[start_idx] = True

    curr = start_idx

    for _ in range(n - 1):
        dists, idxs = tree.query(p[curr], k=neigh_k)
        next_idx = next((i for i in idxs if not used[i]), None)
        if next_idx is None:
            break
        ordered.append(next_idx)
        used[next_idx] = True
        prev, curr = curr, next_idx
    if not all(used):
        unused = np.where(~used)[0]
        for u in unused:
            dists = np.linalg.norm(p[ordered] - p[u], axis=1)
            insert_at = np.argmin(dists)
            ordered.insert(insert_at, u)

    return p[ordered]


def reparametrize_curve_by_curvature(curve, smooth=0.1):
    tck, _ = splprep(curve.T, s=smooth, k=3)
    u_vals = np.linspace(0, 1, len(curve))
    curve_fit = np.array(splev(u_vals, tck)).T
    tangents = np.gradient(curve_fit, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    ds = np.cumsum(np.r_[0, np.sqrt((np.diff(curve_fit, axis=0) ** 2).sum(1))])
    new_s = np.linspace(ds.min(), ds.max(), len(curve))
    curve_uniform = np.array([np.interp(new_s, ds, curve_fit[:, i]) for i in range(3)]).T
    return curve_uniform


def spline_fit_curve_3d_extend_inside_mesh(curve, mesh, order_k=2, num_pts=200, smooth=100, step_u=0.01):
    print(
        f">> Fitting spline to {len(curve)} points with order {order_k}, smooth {smooth}, step_u {step_u} and {num_pts} points...")
    x, y, z = curve.T
    tck, u = splprep([x, y, z], s=smooth, k=order_k)
    u_max = 1.0
    while True:
        u_max += step_u
        pt = np.array(splev(u_max, tck))
        if not mesh.contains(pt.reshape(1, 3))[0]:
            u_max -= step_u
            break
    u_min = 0.0
    while True:
        u_min -= step_u
        pt = np.array(splev(u_min, tck))
        if not mesh.contains(pt.reshape(1, 3))[0]:
            u_min += step_u
            break
    u_vals = np.linspace(u_min, u_max, num_pts)
    curve_fitted = np.array(splev(u_vals, tck)).T
    print(f"Fitted curve of length {len(curve_fitted)}!")
    return curve_fitted


def pca_axis_line_extend_inside_mesh(mesh, num_points=100, oversample=1000):
    print(f">> Fitting PCA line to mesh with oversample {oversample} and {num_points} points...")
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(vertices)
    axis = pca.components_[0]
    axis[0] = 0.0
    axis /= np.linalg.norm(axis)
    projections = (vertices - centroid) @ axis
    t_min = projections.min()
    t_max = projections.max()
    t_dense = np.linspace(t_min, t_max, oversample)
    dense_line = centroid[None, :] + t_dense[:, None] * axis[None, :]
    inside_mask = mesh.contains(dense_line)
    if not np.any(inside_mask):
        raise RuntimeError("No points along PCA axis are inside the mesh!")
    line_start = dense_line[np.argmax(inside_mask)]
    line_end = dense_line[np.where(inside_mask)[0][-1]]
    t = np.linspace(0, 1, num_points)[:, None]
    line = line_start + t * (line_end - line_start)
    print(f"Created PCA line extended inside mesh with {len(line)} points!")
    return line


def cylindrical_along_curve(points, curve):
    print(f">> Performing cylindrical projection along curve for {len(points)} vertices!")
    points = np.asarray(points)
    curve = np.asarray(curve)

    # 1. Compute tangents
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents[0] = tangents[1]
    tangents[-1] = tangents[-2]

    # 2. Create a "Z-up" reference frame instead of a curvature-based one
    z_up = np.array([-1.0, 0.0, 0.0])

    # Project global Z onto the plane perpendicular to the tangent: u = Z - (Z.t)*t
    # Z.t is just the z-component of the tangent
    dot_z_t = tangents[:, 2:3]
    u = z_up - dot_z_t * tangents
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)

    # Fallback: if the curve goes straight up/down, Z-projection is zero.
    # # We use +Y as a fallback "top" for those specific points.
    # vertical_mask = (u_norm[:, 0] < 1e-6)
    # if np.any(vertical_mask):
    #     print(f"[!] Warning: {np.sum(vertical_mask)} point(s) are perfectly vertical; using +y as local 'top'.")
    #     Y = np.array([0.0, 1.0, 0.0])
    #     fallback_u = Y - tangents[vertical_mask, 1:2] * tangents[vertical_mask]
    #     fallback_u /= np.linalg.norm(fallback_u, axis=1, keepdims=True)
    #     u[vertical_mask] = fallback_u
    #     u_norm[vertical_mask] = 1.0

    u /= u_norm  # 'u' is now our normal vector, ALWAYS pointing as close to +Z as possible
    v = np.cross(tangents, u)  # 'v' completes the orthonormal basis (our new binormal)

    # 3. Curve path length (s)
    ds = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s_curve = np.concatenate([[0], np.cumsum(ds)])

    # 4. Map points to the closest point on the curve
    tree = KDTree(curve)
    _, idx = tree.query(points)
    closest = curve[idx]

    vec = points - closest
    rho = np.linalg.norm(vec, axis=1)

    # 5. Compute phi using the new Z-up frame
    u_mapped = u[idx]
    v_mapped = v[idx]

    # phi is 0 exactly when 'vec' aligns with 'u' (the Z-projected vector)
    phi = np.arctan2(np.sum(vec * v_mapped, axis=1), np.sum(vec * u_mapped, axis=1))
    s = s_curve[idx]

    return s, rho, phi


def create_s_phi_basis(points, curve, normals):
    points = np.asarray(points)
    curve = np.asarray(curve)
    tree = KDTree(curve)
    _, idx = tree.query(points)
    closest = curve[idx]
    r_vec = points - closest
    r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r_norm[r_norm == 0] = 1.0
    e_rho = r_vec / r_norm
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents[0] = tangents[1]
    tangents[-1] = tangents[-2]
    t_closest = tangents[idx]
    e_phi = np.cross(t_closest, e_rho)
    e_phi /= np.linalg.norm(e_phi, axis=1, keepdims=True)
    e_s = np.cross(e_phi, normals)
    e_s /= np.linalg.norm(e_s, axis=1, keepdims=True)
    return e_s, e_phi, e_rho


def decompose_q_sphi(q_sphi, s_coords, num_bins=50):
    q_ss = q_sphi[:, 0, 0]
    q_phiphi = q_sphi[:, 1, 1]
    q_sphi = q_sphi[:, 0, 1]
    s_bins = np.linspace(s_coords.min(), s_coords.max(), num_bins)
    s_bin_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
    q_ss_mean = np.zeros(num_bins - 1)
    q_phiphi_mean = np.zeros(num_bins - 1)
    q_sphi_mean = np.zeros(num_bins - 1)

    for i in range(num_bins - 1):
        mask = (s_coords >= s_bins[i]) & (s_coords < s_bins[i + 1])
        if mask.any():
            q_ss_mean[i] = q_ss[mask].mean()
            q_phiphi_mean[i] = q_phiphi[mask].mean()
            q_sphi_mean[i] = q_sphi[mask].mean()
    return s_bin_centers, q_ss, q_ss_mean, q_phiphi, q_phiphi_mean, q_sphi, q_sphi_mean


def shift_angle_periodic(angle, angle_zerobase):
    return np.angle(np.exp(1j * (angle - angle_zerobase)))


def find_phi_intensity_max(phi_vals, intensity_vals):
    phi_vals_sorted = np.sort(phi_vals)
    dphi = np.diff(np.r_[phi_vals_sorted, phi_vals_sorted[0] + 2 * np.pi])
    adaptive_w = intensity_vals * np.interp(phi_vals, phi_vals_sorted, dphi)
    return np.angle(np.sum(adaptive_w * np.exp(1j * phi_vals)) / np.sum(adaptive_w))


def planarise_curve(curve):
    points_centered = curve - curve.mean(axis=0)
    _, _, vh = np.linalg.svd(points_centered)
    normal = vh[-1]
    curve_planar = curve - np.outer(points_centered @ normal, normal)
    distances = np.abs((curve_planar - curve_planar.mean(axis=0)) @ normal)
    print("Max deviation after projection:", distances.max())
    return curve_planar


def crop_by_angles(values, angles, angle_low_cutoff, angle_high_cutoff):
    keep_mask = (angles > angle_low_cutoff) & (angles < angle_high_cutoff)
    return values[keep_mask]


def proj_nem_on_sphi(img_path, layer_label, img_unit, q_decomp_radius=40.0, low_cutoff_phi=-np.pi / 3,
                     high_cutoff_phi=np.pi / 3,
                     profile_bins=30, hidefig=True):
    print(f"Selected image path: {img_path}")
    if not os.path.exists(img_path):
        print(f"Image path does not exist: {img_path} !")
        return None
    # ==== Create Folder Structure ====
    resdata_dir, resfig_dir = create_resdirs(img_path, ct_label=f"t={t_select}_c={c_select}")

    # ==== Load Projected Result ====
    resdata_dir_layer = os.path.join(resdata_dir, layer_label)
    resfig_dir_layer = os.path.join(resfig_dir, layer_label)
    proj_layer = load_array("intensities", folderpath=resdata_dir_layer)
    proj_layer /= proj_layer.max()
    layer_mesh = load_mesh(os.path.join(resdata_dir_layer, "layer_mesh.ply"))

    # ==== Load 2D+ Directors ====
    idxs_sel = load_array("calcindeces", folderpath=resdata_dir_layer).astype(int)
    directors_2dcurved = load_array("directors_2dcurved", folderpath=resdata_dir_layer)

    centerline_fitted = load_array(name="3d_midline_curve", folderpath=os.path.join(resdata_dir))
    dir_coords = cylindrical_along_curve(points=directors_2dcurved[:, :3], curve=centerline_fitted)
    dir_s, dir_rho, dir_phi = dir_coords
    save_array(np.column_stack((dir_s, dir_rho, dir_phi)), name="dir_s-rho-phi",
               header="s,rho,phi", folderpath=resdata_dir_layer)
    layer_mesh_coords = cylindrical_along_curve(points=layer_mesh.vertices, curve=centerline_fitted)
    layer_mesh_s, layer_mesh_rho, layer_mesh_phi = layer_mesh_coords
    save_array(np.column_stack((layer_mesh_s, layer_mesh_rho, layer_mesh_phi)), name="layer_mesh_s-rho-phi",
               header="s,rho,phi", folderpath=resdata_dir_layer)

    # phi_new_zero = find_phi_intensity_max(phi_vals=layer_mesh_phi, intensity_vals=proj_layer)
    # if phi_new_zero_shift is not None:
    #     phi_new_zero += phi_new_zero_shift
    # layer_mesh_phi = shift_angle_periodic(angle=layer_mesh_phi, angle_zerobase=phi_new_zero)
    # dir_phi = shift_angle_periodic(angle=dir_phi, angle_zerobase=phi_new_zero)
    plot_scatter(x=layer_mesh_phi, y=proj_layer, title="Projection Intensity vs. Angle", xlabel=r"$\phi$ (rad)",
                 ylabel="Projection Intensity (a.u.)", xlim=[-np.pi, np.pi], hidefig=hidefig,
                 vert_line=[low_cutoff_phi, high_cutoff_phi],
                 savefig=os.path.join(resfig_dir_layer, "proj-intensity_vs_phi.pdf"))

    plot_cylindrical_projection(phi=layer_mesh_phi, rho=layer_mesh_rho, s=layer_mesh_s, colors=proj_layer, interp_grid_n=200,
                                cmap="inferno", title=f"Projected Intensities \n{layer_label}",
                                savefig=os.path.join(resfig_dir_layer, f"cylindrical_projection.pdf"), hidefig=hidefig)
    plot_rho_profile(mesh_s=layer_mesh_s, mesh_rho=layer_mesh_rho, mesh_phi=layer_mesh_phi,
                     img_unit=img_unit, hidefig=hidefig,
                     savefig=os.path.join(resfig_dir_layer, f"rho-profile-{img_unit}.pdf"))

    e_s, e_phi, e_rho = create_s_phi_basis(points=directors_2dcurved[:, :3], curve=centerline_fitted,
                                           normals=layer_mesh.vertex_normals[idxs_sel])
    save_array(e_s, name="dir_e_s",
               header="x,y,z", folderpath=resdata_dir_layer)
    save_array(e_phi, name="dir_e_phi",
               header="x,y,z", folderpath=resdata_dir_layer)
    save_array(e_rho, name="dir_e_rho",
               header="x,y,z", folderpath=resdata_dir_layer)

    layer_mesh_s_cropped = crop_by_angles(values=layer_mesh_s, angles=layer_mesh_phi, angle_low_cutoff=low_cutoff_phi,
                                          angle_high_cutoff=high_cutoff_phi)
    layer_mesh_rho_cropped = crop_by_angles(values=layer_mesh_rho, angles=layer_mesh_phi,
                                            angle_low_cutoff=low_cutoff_phi,
                                            angle_high_cutoff=high_cutoff_phi)
    layer_mesh_phi_cropped = crop_by_angles(values=layer_mesh_phi, angles=layer_mesh_phi,
                                            angle_low_cutoff=low_cutoff_phi,
                                            angle_high_cutoff=high_cutoff_phi)
    save_array(np.column_stack((layer_mesh_s_cropped, layer_mesh_rho_cropped, layer_mesh_phi_cropped)),
               name="layer_mesh_s-rho-phi_cropped",
               header="s,rho,phi", folderpath=resdata_dir_layer)
    proj_layer_cropped = crop_by_angles(values=proj_layer, angles=layer_mesh_phi, angle_low_cutoff=low_cutoff_phi,
                                        angle_high_cutoff=high_cutoff_phi)
    save_array(proj_layer_cropped, name="intensities_cropped", header="I", folderpath=resdata_dir_layer)
    dir_s_cropped = crop_by_angles(values=dir_s, angles=dir_phi, angle_low_cutoff=low_cutoff_phi,
                                   angle_high_cutoff=high_cutoff_phi)
    dir_rho_cropped = crop_by_angles(values=dir_rho, angles=dir_phi, angle_low_cutoff=low_cutoff_phi,
                                     angle_high_cutoff=high_cutoff_phi)
    dir_phi_cropped = crop_by_angles(values=dir_phi, angles=dir_phi, angle_low_cutoff=low_cutoff_phi,
                                     angle_high_cutoff=high_cutoff_phi)
    save_array(np.column_stack((dir_s_cropped, dir_rho_cropped, dir_phi_cropped)),
               name="dir_s-rho-phi_cropped",
               header="s,rho,phi", folderpath=resdata_dir_layer)

    e_s_cropped = crop_by_angles(values=e_s, angles=dir_phi, angle_low_cutoff=low_cutoff_phi,
                                 angle_high_cutoff=high_cutoff_phi)
    e_rho_cropped = crop_by_angles(values=e_rho, angles=dir_phi, angle_low_cutoff=low_cutoff_phi,
                                   angle_high_cutoff=high_cutoff_phi)
    e_phi_cropped = crop_by_angles(values=e_phi, angles=dir_phi, angle_low_cutoff=low_cutoff_phi,
                                   angle_high_cutoff=high_cutoff_phi)
    save_array(e_s_cropped, name="dir_e_s_cropped",
               header="x,y,z", folderpath=resdata_dir_layer)
    save_array(e_phi_cropped, name="dir_e_phi_cropped",
               header="x,y,z", folderpath=resdata_dir_layer)
    save_array(e_rho_cropped, name="dir_e_rho_cropped",
               header="x,y,z", folderpath=resdata_dir_layer)
    directors_2dcurved_cropped = crop_by_angles(values=directors_2dcurved, angles=dir_phi,
                                                angle_low_cutoff=low_cutoff_phi,
                                                angle_high_cutoff=high_cutoff_phi)
    save_array(directors_2dcurved_cropped, "directors_2dcurved_cropped", header="x,y,z,vx,vy,vz",
               folderpath=resdata_dir_layer)
    plot_cylindrical_projection(phi=layer_mesh_phi_cropped, rho=layer_mesh_rho_cropped, s=layer_mesh_s_cropped,
                                colors=proj_layer_cropped, interp_grid_n=200, cmap="inferno",
                                title=f"Cropped Projected Intensities \n{layer_label}",
                                savefig=os.path.join(resfig_dir_layer, f"cylindrical_projection_cropped.pdf"),
                                hidefig=hidefig, figsize=(10, 5))
    plot_rho_profile(mesh_s=layer_mesh_s_cropped, mesh_rho=layer_mesh_rho_cropped, mesh_phi=layer_mesh_phi_cropped,
                     img_unit=img_unit, hidefig=hidefig,
                     savefig=os.path.join(resfig_dir_layer, f"rho-profile-{img_unit}_cropped.pdf"))

    # ==== Tune Curved Nematic Analysis Number of Neighbours ====
    patch_avg = ["radius", q_decomp_radius]
    patch_type = patch_avg[0]
    patch_size = patch_avg[1]

    # ==== Tune Plotting parameters ====
    if patch_type == "radius":
        patch_label = f"r-{patch_size}{img_unit}"
        neigh_idxs = coord_search_radius(directors_2dcurved_cropped[:, :3], r=patch_size)
    elif patch_type == "nearest":
        patch_label = f"k-{patch_size}"
        neigh_idxs = coord_search_neighbours(directors_2dcurved_cropped[:, :3], k=patch_size, n_process=8)
    else:
        neigh_idxs = None
        patch_label = None
        print(f"[!] Unknown patch type: {patch_type}")

    # ==== Calculate Curved Nematic Order ====
    s_2dcurv_sphi_cropped, n_avg_2dcurv_sphi_cropped, q_sphi_full_cropped = avg_tan_nem_tens(t1_cov=e_s_cropped,
                                                                                             t2_cov=e_phi_cropped,
                                                                                             directors=directors_2dcurved_cropped,
                                                                                             neigh_idxs=neigh_idxs,
                                                                                             return_qij_bar=True)

    save_array(s_2dcurv_sphi_cropped, name=f"s_2dcurv_sphi_cropped_{patch_label}", header="S",
               folderpath=resdata_dir_layer)
    save_array(np.column_stack((directors_2dcurved_cropped[:, :3], n_avg_2dcurv_sphi_cropped)),
               name=f"n_avg_2dcurv_sphi_cropped_{patch_label}",
               header="x,y,z,vx,vy,vz", folderpath=resdata_dir_layer)
    np.savez_compressed(os.path.join(resdata_dir_layer, f"q_sphi_full_cropped_{patch_label}"), q_sphi_cropped=q_sphi_full_cropped)

    # ==== Plot Curved Nematic Order ====
    q_sphi_decomposition = decompose_q_sphi(q_sphi=q_sphi_full_cropped, s_coords=dir_s_cropped, num_bins=profile_bins)
    s_bin_centers_cropped, q_ss_cropped, q_ss_mean_cropped, q_phiphi_cropped, q_phiphi_mean_cropped, q_sphi_cropped, q_sphi_mean_cropped = q_sphi_decomposition
    save_array(s_bin_centers_cropped, name=f"s_bin_centers_cropped_{patch_label}", header="s",
               folderpath=resdata_dir_layer)
    save_array(q_ss_cropped, name=f"q_ss_cropped_{patch_label}", header="q_ss",
               folderpath=resdata_dir_layer)
    save_array(q_ss_mean_cropped, name=f"q_ss_mean_cropped_{patch_label}", header="q_ss",
               folderpath=resdata_dir_layer)

    save_array(q_phiphi_cropped, name=f"q_phiphi_cropped_{patch_label}", header="q_phiphi",
               folderpath=resdata_dir_layer)

    save_array(q_phiphi_mean_cropped, name=f"q_phiphi_mean_cropped_{patch_label}", header="q_phiphi",
               folderpath=resdata_dir_layer)

    save_array(q_sphi_cropped, name=f"q_sphi_cropped_{patch_label}", header="q_sphi",
               folderpath=resdata_dir_layer)
    save_array(q_sphi_mean_cropped, name=f"q_sphi_mean_cropped_{patch_label}", header="q_sphi",
               folderpath=resdata_dir_layer)

    plot_qsphi_profiles(dir_s=dir_s_cropped, s_bin_centers=s_bin_centers_cropped, Q_ss=q_ss_cropped,
                        Q_ss_mean=q_ss_mean_cropped, Q_phiphi=q_phiphi_cropped,
                        Q_phiphi_mean=q_phiphi_mean_cropped, Q_sphi=q_sphi_cropped, Q_sphi_mean=q_sphi_mean_cropped,
                        y_limits=[-0.5, 0.5], hidefig=hidefig,
                        savefig=os.path.join(resfig_dir_layer, f"q-sphi-profile-{img_unit}_{patch_label}_cropped.pdf"))

    plot_qsphi_profiles_separated_phi(dir_s=dir_s_cropped, dir_phi=dir_phi_cropped, s_bin_centers=s_bin_centers_cropped,
                                      Q_ss=q_ss_cropped,
                                      Q_ss_mean=q_ss_mean_cropped, Q_phiphi=q_phiphi_cropped, hidefig=hidefig,
                                      Q_phiphi_mean=q_phiphi_mean_cropped, Q_sphi=q_sphi_cropped,
                                      Q_sphi_mean=q_sphi_mean_cropped,
                                      y_limits=[-0.5, 0.5], savefig=os.path.join(resfig_dir_layer,
                                                                                 f"q-sphi-by-phi-profile-{img_unit}_{patch_label}_cropped.pdf"))

    plot_s_sphi_profile(dir_s=dir_s_cropped, s_global=s_2dcurv_sphi_cropped, y_limits=[0, 1], hidefig=hidefig,
                        savefig=os.path.join(resfig_dir_layer, f"S-profile-{img_unit}_{patch_label}_cropped.pdf"))
    return None


def bin_directors(directors, ap_par_binned_idxs, ap_orth_binned_idxs, curve,
                  nematic_weights):
    ap_par_binned_s_2d, ap_par_binned_n_2d = avg_2d_nem_tens(directors=directors,
                                                             neigh_idxs=ap_par_binned_idxs,
                                                             weights=nematic_weights)
    ap_orth_binned_s_2d, ap_orth_binned_n_2d = avg_2d_nem_tens(directors=directors,
                                                               neigh_idxs=ap_orth_binned_idxs,
                                                               weights=nematic_weights)
    ap_par_binned_s_2d_unw, ap_par_binned_n_2d_unw = avg_2d_nem_tens(directors=directors,
                                                                     neigh_idxs=ap_par_binned_idxs,
                                                                     weights=None)
    ap_orth_binned_s_2d_unw, ap_orth_binned_n_2d_unw = avg_2d_nem_tens(directors=directors,
                                                                       neigh_idxs=ap_orth_binned_idxs,
                                                                       weights=None)

    nematic_results = (ap_par_binned_s_2d, ap_par_binned_n_2d,
                       ap_orth_binned_s_2d, ap_orth_binned_n_2d)
    nematic_results_unweighted = (ap_par_binned_s_2d_unw, ap_par_binned_n_2d_unw,
                                  ap_orth_binned_s_2d_unw, ap_orth_binned_n_2d_unw)
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
    curve *= scale[1:]
    return curve


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
