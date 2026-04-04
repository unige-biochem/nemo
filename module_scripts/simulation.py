"""
Simulation Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


#################
# BASIC MODULES #
#################
def random_tangential(t1, t2, seed=None):
    print(">> Randomising tangential vector field...")
    if seed is not None:
        np.random.seed(seed)
    return t1 * np.random.uniform(-1, 1, size=(len(t1), 1)) + t2 * np.random.uniform(-1, 1, size=(
        len(t2), 1))


def generate_surface(mode, x_, y_):
    print(f">> Generating {mode} surface..")
    if mode == "parabolic_up":
        z = x_ ** 2 + y_ ** 2
        dzdx = 2 * x_
        dzdy = 2 * y_
    elif mode == "saddle":
        z = x_ ** 2 - y_ ** 2
        dzdx = 2 * x_
        dzdy = -2 * y_
    elif mode == "parabolic_down":
        z = -x_ ** 2 - y_ ** 2
        dzdx = -2 * x_
        dzdy = -2 * y_
    elif mode == "flat":
        z = np.zeros_like(x_)
        dzdx = np.zeros_like(x_)
        dzdy = np.zeros_like(x_)
    else:
        print(f"Unknown mode: {mode} !")
        z, dzdx, dzdy = None, None, None
    return z, dzdx, dzdy


def surface_warp(x, y, z, dz_dx, dz_dy, u, v):
    print(">> Projection vector field onto surface...")
    w = (dz_dx * u + dz_dy * v)
    magnitude = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    vx, vy, vz = u / magnitude, v / magnitude, w / magnitude
    vecdata = np.column_stack((x.flatten(), y.flatten(), z.flatten(), vx.flatten(), vy.flatten(), vz.flatten()))
    normals = np.array([dz_dx, dz_dy, -np.ones_like(z)])
    normals /= np.sqrt(dz_dx ** 2 + dz_dy ** 2 + np.ones_like(z) ** 2)
    normals *= -1
    normal_x, normal_y, normal_z = normals
    normals = np.column_stack((normal_x.ravel(), normal_y.ravel(), normal_z.ravel()))
    return vecdata, normals


######################
# GENERATION MODULES #
######################
def spherical_shell(l, r, dr, sigma=0, center=None, ring_intensity=8, num_rings=50, shell_intensity=1, phi_width=0.01,
                    theta_width=0.01):
    print(
        f">> Creating shell in {l} x {l} x {l}, inner radius {r}, thickness {dr}, ring with intensity {ring_intensity} and Gaussian blur {sigma}...")
    if center is None: center = (l // 2, l // 2, l // 2)
    x, y, z = np.indices((l, l, l))
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    shell_ = np.logical_and(dist >= r - dr, dist <= r + dr).astype(float) * shell_intensity

    x -= center[0]
    y -= center[1]
    z -= center[2]

    phi = np.arctan2(y, x)
    phi_vals = np.linspace(-np.pi, np.pi, num_rings)
    print("Constant phi rings made !")
    for phi_ in phi_vals:
        ring_mask = (
                (np.abs(phi - phi_) <= phi_width) & (dist >= r - dr) & (dist <= r - dr / 2)
        )
        shell_[ring_mask] = ring_intensity

    theta = np.arccos(z / dist)
    theta_vals = np.linspace(0, np.pi, num_rings)
    print("Constant theta rings made !")
    for theta_ in theta_vals:
        ring_mask = (np.abs(theta - theta_) <= theta_width) & (dist >= r + dr / 2) & (dist <= r + dr)
        shell_[ring_mask] = ring_intensity

    return gaussian_filter(shell_, sigma=sigma) if sigma > 0 else shell_


def uniform_shell(l, r, dr, sigma=0, center=None):
    print(
        f">> Creating shell in {l} x {l} x {l}, inner radius {r}, thickness {dr} and Gaussian blur {sigma}...")
    if center is None: center = (l // 2, l // 2, l // 2)
    x, y, z = np.indices((l, l, l))
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    shell_ = np.logical_and(dist >= r - dr, dist <= r + dr).astype(float)
    return gaussian_filter(shell_, sigma=sigma) if sigma > 0 else shell_


def point_defect_2d(l, n, defect_type, defect_center=(0, 0)):
    print(f">> Creating {defect_type} defect at {defect_center}...")
    x = np.linspace(-l, l, n)
    y = np.linspace(-l, l, n)
    X, Y = np.meshgrid(x, y)
    dx = X - defect_center[0]
    dy = Y - defect_center[1]
    Theta = np.arctan2(dy, dx)

    if defect_type == "0":
        m = 0  # Uniform field
    elif defect_type == "+1  aster":
        m = 1  # +1 defect (radial aster)
    elif defect_type == "-1":
        m = -1  # -1 defect (hyperbolic)
    elif defect_type == "+2":
        m = 2  # +2 defect (double winding)
    elif defect_type == "+1.5":
        m = 1.5  # +3 /2 defect
    elif defect_type == "-0.5":
        m = -0.5  # -1/2 defect
    elif defect_type == "+0.5":
        m = 0.5  # +1/2 defect
    elif defect_type == "+1  ring":
        m = 0  # FAKE, see next if
    elif defect_type == "+1  vortex":
        m = 0  # FAKE, see next if
    else:
        raise ValueError("Unsupported defect type.")
    angle = m * Theta
    U = np.cos(angle)  # Radial component
    V = np.sin(angle)  # Tangential component

    if defect_type == "+1  ring":
        angle = Theta + np.pi / 2  # Tangential (perpendicular to radial)
        U = np.cos(angle)  # Radial component
        V = np.sin(angle)  # Tangential component
    elif defect_type == "+1  vortex":
        radial_component = -1
        tangential_component = 1
        U = radial_component * np.cos(Theta) - tangential_component * np.sin(Theta)
        V = radial_component * np.sin(Theta) + tangential_component * np.cos(Theta)

    magnitude = np.sqrt(U ** 2 + V ** 2)
    U /= magnitude
    V /= magnitude
    return X, Y, U, V


def point_defect_2d_charge(l, n, defect_charge, defect_center=(0, 0)):
    print(f">> Creating {defect_charge} defect at {defect_center}...")
    x = np.linspace(-l, l, n)
    y = np.linspace(-l, l, n)
    X, Y = np.meshgrid(x, y)
    dx = X - defect_center[0]
    dy = Y - defect_center[1]
    Theta = np.arctan2(dy, dx)
    m = float(defect_charge)
    angle = m * Theta
    U = np.cos(angle)
    V = np.sin(angle)
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U /= magnitude
    V /= magnitude
    return X, Y, U, V


def point_defect_2d_charge_multiple(l, n, defects):
    print(f">> Creating defects...")
    x = np.linspace(-l, l, n)
    y = np.linspace(-l, l, n)
    X, Y = np.meshgrid(x, y)
    Theta_total = np.zeros(X.shape)

    for i, defect in enumerate(defects):
        x_d, y_d, charge = defect
        dx = X - x_d
        dy = Y - y_d
        Theta = np.arctan2(dy, dx)
        m = float(charge)
        Theta_total += m * Theta
    U = np.cos(Theta_total)
    V = np.sin(Theta_total)
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U /= magnitude
    V /= magnitude
    return X, Y, U, V


def isotropic_2d_field(l, n):
    print(f">> Creating isotropic 2D field of {l}x{l}...")
    x = np.linspace(-l, l, n)
    y = np.linspace(-l, l, n)
    X, Y = np.meshgrid(x, y)
    theta = np.random.uniform(0, 2 * np.pi, size=(n, n))
    U = np.cos(theta)
    V = np.sin(theta)
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U /= magnitude
    V /= magnitude
    return X, Y, U, V


def spherical_defect(n, mode):
    print(f">> Creating spherical defect of {mode}...")
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n // 2)
    theta, phi = np.meshgrid(theta, phi)
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)

    if mode == "ring":
        vx = -np.sin(phi)
        vy = np.cos(phi)
        vz = np.zeros_like(vx)
    elif mode == "aster":
        vx = np.cos(phi) * np.cos(theta)
        vy = np.sin(phi) * np.cos(theta)
        vz = -np.sin(theta)
    elif mode == "isotropic":
        random_theta = np.arccos(1 - 2 * np.random.rand(*theta.shape))
        random_phi = 2 * np.pi * np.random.rand(*phi.shape)
        vx_random = np.sin(random_theta) * np.cos(random_phi)
        vy_random = np.sin(random_theta) * np.sin(random_phi)
        vz_random = np.cos(random_theta)
        rx, ry, rz = X, Y, Z
        dot_product = vx_random * rx + vy_random * ry + vz_random * rz
        vx = vx_random - dot_product * rx
        vy = vy_random - dot_product * ry
        vz = vz_random - dot_product * rz
    else:
        vx = np.ones_like(phi)
        vy, vz = vx, vx
    mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    vx /= mag
    vy /= mag
    vz /= mag

    T1_x = np.cos(theta) * np.cos(phi)
    T1_y = np.cos(theta) * np.sin(phi)
    T1_z = -np.sin(theta)

    T2_x = -np.sin(phi)
    T2_y = np.cos(phi)
    T2_z = np.zeros_like(T2_x)

    t1_raw = np.column_stack((T1_x.flatten(), T1_y.flatten(), T1_z.flatten()))
    t2_raw = np.column_stack((T2_x.flatten(), T2_y.flatten(), T2_z.flatten()))
    vecdata = np.column_stack((X.flatten(), Y.flatten(), Z.flatten(), vx.flatten(), vy.flatten(), vz.flatten()))
    return vecdata, t1_raw, t2_raw


def generate_nemo_master_bench(l=256, do_inner=True, do_outer=True, r_inner=100, r_outer=130, dr=1.5, steps=40,
                               seed_density=0.012):
    print(f">> Initializing 3D manifold simulation...")
    vol = np.zeros((l, l, l), dtype=np.float32)
    center = np.array([l // 2, l // 2, l // 2])
    z, y, x = np.indices((l, l, l))
    dx = (x - center[0]).astype(np.float32)
    dy = (y - center[1]).astype(np.float32)
    dz = (z - center[2]).astype(np.float32)
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 1e-9
    nx, ny, nz = dx / dist, dy / dist, dz / dist
    phi = np.arctan2(dy, dx)
    theta = np.arccos(np.clip(nz, -1, 1))
    et = [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), -np.sin(theta)]
    ep = [-np.sin(phi), np.cos(phi), np.zeros_like(phi)]
    seeds = (np.random.uniform(0, 1, (l, l, l)) > (1 - seed_density)).astype(np.float32)

    def advect_fibers(flow, mask, n_steps):
        active_seeds = seeds * mask
        acc = np.copy(active_seeds)
        dt = 0.8
        for d in [1, -1]:
            cz, cy, cx = z.astype(float), y.astype(float), x.astype(float)
            for _ in range(n_steps):
                cx += d * dt * flow[0]
                cy += d * dt * flow[1]
                cz += d * dt * flow[2]
                acc += map_coordinates(active_seeds, [cz, cy, cx], order=1, mode='constant', cval=0)
        res = np.zeros_like(acc)
        res[mask] = acc[mask]
        return res

    if do_inner:
        print(">> Calculating Tetrahedral Ground State (Tennis Ball)...")
        s = 1 / np.sqrt(3)
        pts = [
            np.array([s, s, s]),
            np.array([s, -s, -s]),
            np.array([-s, s, -s]),
            np.array([-s, -s, s])
        ]
        v_sum_x = np.zeros_like(dx)
        v_sum_y = np.zeros_like(dx)
        for p in pts:
            v_vec = [nx - p[0], ny - p[1], nz - p[2]]
            v_t = v_vec[0] * et[0] + v_vec[1] * et[1] + v_vec[2] * et[2]
            v_p = v_vec[0] * ep[0] + v_vec[1] * ep[1] + v_vec[2] * ep[2]
            v_mag = np.sqrt(v_t ** 2 + v_p ** 2 + 1e-9)
            v_sum_x += v_t / v_mag
            v_sum_y += v_p / v_mag
        inner_angle = 0.5 * np.arctan2(v_sum_y, v_sum_x)
        flow_in = [et[0] * np.cos(inner_angle) + ep[0] * np.sin(inner_angle),
                   et[1] * np.cos(inner_angle) + ep[1] * np.sin(inner_angle),
                   et[2] * np.cos(inner_angle) + ep[2] * np.sin(inner_angle)]
        inner_mask = (dist >= r_inner - dr) & (dist <= r_inner + dr)
        vol += advect_fibers(flow_in, inner_mask, steps) * 180
        vol[inner_mask] += 40

    if do_outer:
        print(">> Calculating Outer Asters...")
        flow_out = [et[0], et[1], et[2]]
        outer_mask = (dist >= r_outer - dr) & (dist <= r_outer + dr)
        vol += advect_fibers(flow_out, outer_mask, steps) * 180
        vol[outer_mask] += 40
    vol = gaussian_filter(vol, sigma=0.5)
    print(f">> Benchmark saved successfully.")
    return vol
