#!/usr/bin/env python
from .common import list2arr, get_logger
import numpy as np
import re

logger = get_logger(__name__)


def calLattice(arr):
    """
    Calculate lattice, angle, zeropoint from lammstrj's matrix

    args
    ;; arr -> np.nadarry(2, 3) or (3, 3)

    return
    ;; lattice -> list
    ;; angle -> list
    ;; zeropoint -> list
    """
    if arr.shape[1] == 3:
        xlo, xhi, xy = map(float, arr[0, :])
        ylo, yhi, xz = map(float, arr[1, :])
        zlo, zhi, yz = map(float, arr[2, :])
    elif arr.shape[1] == 2:
        xlo, xhi = map(float, arr[0, :])
        ylo, yhi = map(float, arr[1, :])
        zlo, zhi = map(float, arr[2, :])
        (
            xy,
            xz,
            yz,
        ) = 0, 0, 0
    xhi -= np.max([0, xy, xz, xy + xz])
    xlo -= np.min([0, xy, xz, xy + xz])
    ylo -= np.min([0, yz])
    yhi -= np.max([0, yz])
    x_length = xhi - xlo
    y_length = yhi - ylo
    z_length = zhi - zlo
    a_vec = np.array([x_length, 0, 0])
    b_vec = np.array([xy, y_length, 0])
    c_vec = np.array([xz, yz, z_length])
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
    lattice = [a, b, c]
    angle = [alpha, beta, gamma]
    zeropoint = [xlo, ylo, zlo]
    return lattice, angle, zeropoint


def setMatrix(lattice, angle, return_lammps_matrix=False):
    """
    It returns matrix suit for MD simulations
    Be careful the matrix shapes like
    [[a1, 0, 0], [b1, b2, 0], [c1, c2, c3]]

    args
    ;;lattice -> list or np.ndarray
    ;;angle -> list or np.ndarray

    return
    ;;matrix, V
    """
    alpha, beta, gamma = np.radians(angle)
    x, y, z = lattice
    V = np.sqrt(
        1
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )
    matrix = np.array(
        [
            [x, y * np.cos(gamma), z * np.cos(beta)],
            [
                0,
                y * np.sin(gamma),
                z * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
            ],
            [0, 0, z * V / np.sin(gamma)],
        ]
    ).T
    if return_lammps_matrix:
        a = lattice[0]
        b = lattice[1]
        c = lattice[2]
        lx = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        ly = np.sqrt(b**2 - xy**2)
        yz = (b * c * np.cos(alpha) - xy * xz) / ly
        lz = np.sqrt(c**2 - xz**2 - yz**2)
        lammps_matrix = [[0, lx, xy], [0, ly, xz], [0, lz, yz]]
        lammps_matrix = np.round(np.array(lammps_matrix), 6)
        return matrix, V, lammps_matrix
    else:
        return matrix, V


def lammpsLattice(self, matrix):
    lx = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    ly = np.sqrt(b**2 - xy**2)
    yz = (b * c * np.cos(alpha) - xy * xz) / ly
    lz = np.sqrt(c**2 - xz**2 - yz**2)
    lammps_lattice = [[0, lx, xy], [0, ly, xz], [0, lz, yz]]
    self.lammps_lattice = np.round(np.array(lammps_lattice), 6)


def cal_distance(pos1, pos2, matrix, cartesian=False):
    """
    Calculate distance fraction coordinations of between two groups
    it also consider periodic boundary conditions

    args:
    ;; pos1 -> list or np.nadarry(shape(n, 3))
    ;; pos2 -> list or np.ndarray(shape(m, 3))
    ;; matrix -> list or np.ndarray(shape=(3. 3))
    ;;; cartesian -> bool, type of coordinations

    return
    ;; table -> np.nadarry(shape(n, m))
    """
    pos1 = list2arr(pos1)
    pos2 = list2arr(pos2)
    matrix = list2arr(matrix)
    r = (pos1[:, None, :] - pos2[None, :, :]).astype(float)
    if cartesian:
        r = r @ np.linalg.inv(matrix)
    r = (r - np.round(r)) @ matrix
    r = np.sqrt(np.sum(r**2, axis=2))
    distance = np.where(r <= 0.01, np.inf, r)
    return distance


def cal_distance_with_block(pos1, pos2, matrix, block=200, cartesian=False):
    n = pos1.shape[0]
    m = pos2.shape[0]
    table = np.empty((n, m))
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        pos1i = pos1[i0:i1]
        r = (pos1i[:, None, :] - pos2[None, :, :]).astype(float)
        if cartesian:
            r = r @ np.linalg.inv(matrix)
        r = (r - np.round(r)) @ matrix
        r = np.sqrt(np.sum(r**2, axis=2))
        r = np.where(r <= 0.01, np.inf, r)
        table[i0:i1] = r
    return table


def cal_uniform(matrix=None, mole_sum=None, spacing=3.5, **kwargs):
    """
    makes grid from matrix, each grid seperated from the value of spacing
    in default, it do not include start-point and includes end-point
    Be careful both start-point, end-point included along the axis, (0 == 1, mod 1)

    args
    ;; matrix -> np.nadarry(3, 3)
    ;; mole_sum -> int,  the number of whole molecules in system
    ;;; spacing -> float, distance between grids, default=3.5
    ;;;; include_start -> list(bool, bool, bool), if True, include startpoint
    ;;;; include_end -> list(bool, bool, bool), if True, include zeropoint
    ;;;; weight_x,  weight_y, weight_z -> int,  multiply number of grid along axis

    return
    np.ndarray(-1, 3)
    """
    for val in (matrix, mole_sum):
        if val is None:
            logger.critical("Wrong value in 'cal_uniform'")
            return False
    while True:
        ngrid = np.ceil(np.linalg.norm(matrix, axis=1) / spacing).astype(int)
        for i, axis in enumerate(["x", "y", "z"]):
            weight = kwargs.get(f"weight_{axis}", None)
            if weight is not None:
                ngrid[i] *= weight
                ngrid[i] = np.ceil(ngrid[i])
        nx, ny, nz = ngrid
        if (nx) * (ny) * (nz) < mole_sum:
            spacing -= 0.2
        else:
            break
    logger.info("Spacing: ", f"{spacing:.2f}")
    start = kwargs.get("include_start", [False, False, False])
    start_ = [1.01, 1.01, 1.3]
    s = [s_ - int(s) for s_, s in zip(start_, start)]
    end_ = [0.01, -0.01, -0.3]
    end = kwargs.get("include_end", [True, True, True])
    e = [s_ + int(s) for s_, s in zip(end_, end)]
    if not any(s and e for s, e in zip(start, end)):
        logger.warning("Warning: Both start & end point are actiavted")
    # start = 0 if include_start else 1
    # end = 1 if include_end else 0
    grid = (
        np.mgrid[s[0] : nx + e[0], s[1] : ny + e[1], s[2] : nz + e[2]].reshape(3, -1).T
    )
    grid = grid / [nx, ny, nz]
    #    grid = np.where(grid == 0, grid + 0.001, grid)
    return grid


def cal_uniform_by_space(
    matrix=None, key_enum=None, mole_sum=None, spacing_list=None, **kwargs
):
    for val in (matrix, key_enum, mole_sum):
        if val is None:
            logger.warning("Wrong value in 'cal_uniform_by_space'")
            return False
    spacing_list = (
        [3.5, 3.5, 1.8]
        if (spacing_list is None) or (not isinstance(spacing_list, list))
        else spacing_list
    )
    not_enum = [i for i in (0, 1, 2) if i != key_enum]
    lattice = matrix
    lattice = np.linalg.norm(matrix, axis=1)
    fix_lattice = lattice[not_enum]
    ngrid = list(
        np.ceil(
            [fix_lattice[not_enum[i]] / spacing_list[not_enum[i]] for i in range(2)]
        ).astype(int)
    )
    mol_per_layer = int(ngrid[0] * ngrid[1])
    num_layer = np.ceil(mole_sum / mol_per_layer).astype(int)
    logger.info(f"Number of molecules to add : {mole_sum}")
    logger.info(f"Number of atoms per surface : {mol_per_layer}")
    logger.info(f"Number of layer : {num_layer}")
    ngrid.insert(key_enum, num_layer)
    linspace = list(map(lambda x: np.linspace(0, 1, x + 1)[0:-1], ngrid))
    shrink_ratio = spacing_list[key_enum] * num_layer / matrix[key_enum][key_enum]
    matrix[key_enum, :] = matrix[key_enum, :] * shrink_ratio
    mesh_grid = np.meshgrid(*linspace)
    grid = np.vstack(mesh_grid).reshape(3, -1).T
    return grid, matrix


def extract_number(string, dtype=float):
    """
    extract list of floats(int) from string

    args
    ;; string
    ;;; dtype, float or int, default=float

    return
    ;; list
    """
    ptn = r"[0-9\.\-]+"
    num = re.findall(ptn, string)
    num = [dtype(n) for n in num if n != "."]
    return num
