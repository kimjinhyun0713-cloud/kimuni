#!/usr/bin/env python
import numpy as np
import pandas as pd
import re
import os
from .functions import setMatrix
from .functions import calLattice
from .common import overwritePrint
from .common import list2arr

def poscar2Dic(infile):
    with open(infile) as o:
        base = o.readline().strip()
        dic = {"base": base}
        o.readline()
        lst = []
        for _ in range(3):
            lst.append(o.readline())
        matrix = [list(map(float, l.split())) for l in lst]
        matrix = list2arr(matrix)
        dic["matrix"] = matrix
        abc = np.linalg.norm(matrix, axis=1)
        dic["lattice"] = abc
        a, b, c = abc
        angle = []
        angle.append(np.arccos(matrix[1, :] @ matrix[2, :] / (b*c)) * 180/np.pi)
        angle.append(np.arccos(matrix[0, :] @ matrix[2, :] / (a*c)) * 180/np.pi)
        angle.append(np.arccos(matrix[0, :] @ matrix[1, :] / (a*b)) * 180/np.pi)
        dic["angle"] = angle
        elem = o.readline().split()
        dic["elem"] = elem
        n_elem = [int(v) for v in o.readline().split()]
        o.readline()
        data = []
        for i, n in enumerate(n_elem):
            lst = []
            arr = np.empty((n, 4), dtype=object)
            arr[:, 0] = elem[i]
            for _ in range(n):
                lst.append([float(v) for v in o.readline().split()])
            arr[:, 1:4] = lst
            data.append(arr)
        dic["data"] = np.vstack(data)
        return dic

        
def lmp2Dic(lammpstrj, to_fract=False):
    """
    convert lammpstrj to dictionary of which havs or local values

    args
    ;; lammpstrj

    return
    ;; locals()
    """
    with open(lammpstrj) as o:
        readlines = o.readlines()
        natoms = int(readlines[3])
        nlines = natoms + 9
        nstep = len(readlines) / nlines
        assert nstep == int(nstep), f"Please check the file is completely written nstep={nstep}"
        nstep = int(nstep)
        ldata = np.zeros((nstep, natoms, 4), dtype=object)
        lattice = np.zeros((nstep, 3), dtype=float)
        angle = np.zeros((nstep, 3), dtype=float)
        print(f"Loading {lammpstrj}")
        for s in range(nstep):
            skip = s * nlines
            lattice_ = np.array(list(map(lambda l: l.split(), readlines[skip + 5:skip + 8])), dtype=float)
            lattice[s, :], angle[s, :], zeropoint = calLattice(lattice_)
            data_ = readlines[skip + 9:skip + nlines]
            data_ = " ".join(data_).strip().split()
            try:
                data_ = np.array(data_, dtype=object).reshape(-1, 6)
                ncol = 6
            except ValueError:
                data_ = np.array(data_, dtype=object).reshape(-1, 7)
                ncol = 7
            data_[:, 3:6] = data_[:, 3:6].astype(float)
            data_[:, 3] -= zeropoint[0]
            data_[:, 4] -= zeropoint[1]
            data_[:, 5] -= zeropoint[2]
            ldata[s, :, :] = data_[:, 2:6]
            overwritePrint(f"Store step: {s}")
        elem = np.unique(data_[:, 2])
        type_ = np.unique(data_[:, 1])
        if to_fract:
            matrix_list = []
            for i in range(ldata.shape[0]):
                matrix_, _ = setMatrix(lattice[i, :], angle[i, :])
                matrix_list.append(matrix_)
                ldata[i, :, 1:4] = ldata[i, :, 1:4] @ np.linalg.inv(matrix_)
            matrix = np.vstack(matrix_list).reshape(-1, 3, 3)
            del matrix_list
            
        dic = {k: v for k, v in locals().items() if k in ["nlines", "ldata", "natoms", "nstep", "lattice", "angle", "elem", "ldata", "matrix"]}
        return dic
            

def excel2data(infile):
    """
    convert excel(.xlsx) data to Pd.dataFrame
    args:
    ;; infile -> .xlsx

    return
    pd.DataFrame
    """
    with pd.ExcelFile(infile) as o:
        sheet_names = o.sheet_names
        for sheet in sheet_names:
            df = o.parse(sheet)
    return df
    
def cif2data(infile, cartesian=False):
    """
    convert cif data to pd.DateFrame includes whole columns in cif.
    For example symbol, occupancy
    
    args
    ;; infile -> str, infile
    ;;; cartesian  -> bool, convert fractional coordinations to cartesian(default: False)
    
    returns
    lattice -> list
    angle -> list
    data -> pd.DateFrame
    """
    with open(infile) as o:
        assert os.path.splitext(infile)[1] == ".cif", "Make sure to put 'CIF'"
        read = o.read()
        lattice_ptn  = r"_cell_length_[abc]+ +([0-9\.]+)"
        angle_ptn = r"_cell_angle_[a-zA-Z]+ +([0-9\.]+)"
        lattice = [float(v) for v in re.findall(lattice_ptn, read)]
        angle = [float(v) for v in re.findall(angle_ptn, read)]
        ptn_col = r"_atom_site_(.*)"
        col = re.findall(ptn_col, read)
        col = [c.strip() for c in col]
        data = read.split(col[-1])[-1]
        data = np.array([s.split() for s in data.strip().split("\n")], dtype=object).reshape(-1, len(col))
        data = pd.DataFrame(data, columns=col)
        data["fract_x"] = data["fract_x"].astype(float)
        data["fract_y"] = data["fract_y"].astype(float)
        data["fract_z"] = data["fract_z"].astype(float)
        if cartesian:
            matrix, _ = setMatrix(lattice, angle)
            data[["fract_x", "fract_y", "fract_z"]] = data[["fract_x", "fract_y", "fract_z"]] @ matrix
        return lattice, angle, data
        
    
