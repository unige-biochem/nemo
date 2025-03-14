"""
Simulation Modules for NEMO, the Nematics & Morphology Toolkit.
Author: Konstantinos Andreadis
"""

####################
# IMPORT LIBRARIES #
####################
import numpy as np
from scipy.ndimage import gaussian_filter


#################
# BASIC MODULES #
#################
def random_tangential(t1, t2):
    print(">> Randomising tangential vector field...")
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
        # Spiral inward: radial + tangential components
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
    Theta_total = np.zeros((len(defects), *X.shape))

    for i, defect in enumerate(defects):
        x_d, y_d, charge = defect
        dx = X - x_d
        dy = Y - y_d
        Theta = np.arctan2(dy, dx)
        m = float(charge)
        Theta_total[i] = m * Theta

    Theta_total = np.average(np.cos(2 * Theta_total), axis=0)
    Theta_total = np.arccos(Theta_total)
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
