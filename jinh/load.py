#!/usr/bin/env python
import numpy as np
import pandas as pd
import re
import os
from .functions import setMatrix
from .functions import calLattice

    
class LAMMPSTRJ():
    def __init__(self, fname):
        self.lammpstrj = fname
        assert os.path.splitext(self.lammpstrj)[1] == ".lammpstrj", "Make sure to put lammpstrj"
        self.dname = os.path.dirname(os.path.abspath(self.lammpstrj))
        self.loadLammptsrj()
        self.matrix = []
        self.V = []
        for lattice, angle in zip(self.lattice, self.angle):
            matrix, V = setMatrix(lattice, angle)
            self.matrix.append(matrix)
            self.V.append(V)
        self.matrix = np.vstack(self.matrix).reshape(-1, 3, 3)
        self.V = np.vstack(self.V)

    def __str__(self):
        string = "\n"
        string += f"Number of atoms: {self.natoms}"
        string += f"\nNumber of steps: {self.nstep}"
        return string
    
    def loadLammptsrj(self):
        with open(self.lammpstrj) as o:
            for i in range(4):
                line = o.readline()
            self.natoms = int(line)
            nlines = self.natoms + 9
        with open(self.lammpstrj) as o:
            readlines = o.readlines()
            nstep = len(readlines) / nlines  # nstep: whole step of lammpstrj
            assert nstep == int(nstep), f"Please check the file is completely written nstep={nstep}"
            self.nstep = int(nstep)
            self.data = np.zeros((self.nstep, self.natoms, 4), dtype=object)
            self.lattice = np.zeros((self.nstep, 3), dtype=float)
            self.angle = np.zeros((self.nstep, 3), dtype=float)
            print(f"Loading {self.lammpstrj}")
            for s in range(self.nstep):
                skip = s * nlines
                lattice = np.array(list(map(lambda l: l.split(), readlines[skip + 5:skip + 8])), dtype=float)
                self.lattice[s, :], self.angle[s, :], zeropoint = calLattice(lattice)
                data = readlines[skip + 9:skip + nlines]
                data = " ".join(data).strip().split()
                try:
                    data = np.array(data, dtype=object).reshape(-1, 6)
                except ValueError:
                    data = np.array(data, dtype=object).reshape(-1, 7)
                data[:, 3:6] = data[:, 3:6].astype(float)
                data[:, 3] -= zeropoint[0]
                data[:, 4] -= zeropoint[1]
                data[:, 5] -= zeropoint[2]
                self.data[s, :, :] = data[:, 2:6]
            self.elem = np.unique(data[:, 2])
            
    def to_fract(self):
        for i in range(self.data.shape[0]):
            self.data[i, :, 1:4] = self.data[i, :, 1:4] @ np.linalg.inv(self.matrix[i, :, :])
            
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
        
    
