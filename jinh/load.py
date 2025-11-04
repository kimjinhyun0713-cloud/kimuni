#!/usr/bin/env python
import numpy as np
import pandas as pd
import re
import os
from .functions import setMatrix
from .functions import calLattice
from .common import overwritePrint

def lmp2Dic(lammpstrj, to_fract=False):
    """
    convert lammpstrj to dictionary of which havs or local values

    args
    ;; lammpstrj

    return
    ;; locals()
    """
    with open(lammpstrj) as o:
        for i in range(4):
            line = o.readline()
        natoms = int(line)
        nlines = natoms + 9
    with open(lammpstrj) as o:
        readlines = o.readlines()
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
        return locals()
            
            
def cif2data(fname, cartesian=False):
    """
    convert cif data to pd.DateFrame includes whole columns in cif.
    For example symbol, occupancy
    
    args
    ;; fname -> str, infile
    ;;; cartesian  -> bool, convert fractional coordinations to cartesian(default: False)
    
    returns
    lattice -> list
    angle -> list
    data -> pd.DateFrame
    """
    with open(fname) as o:
        assert os.path.splitext(fname)[1] == ".cif", "Make sure to put 'CIF'"
        read = o.read()
        lattice_ptn  = r"_cell_length_[abc]+ +([0-9\.]+)"
        angle_ptn = r"_cell_angle_[a-zA-Z]+ +([0-9\.]+)"
        lattice = [float(v) for v in re.findall(lattice_ptn, read)]
        angle = [float(v) for v in re.findall(angle_ptn, read)]
        ptn_col = r"_atom_site_(.*)"
        col = re.findall(ptn_col, read)
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
        
    
