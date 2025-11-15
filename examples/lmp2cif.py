#!/usr/bin/env python
import numpy as np
import re
import os
from jinh import cif_head, lmp2Dic, setMatrix


fmt = cif_head
fmt_ = "{:>5s}" + "{:12f}".format(1.0) + 3*"{:20.16f}"
fmt_ += "{:>8s}".format("Biso") + "{:10f}".format(1.0) + "{:>5s}\n"


class LAMMPSTRJ():

    @classmethod
    def setLammpstrj(cls, infile):
        if os.path.splitext(infile)[1] != ".lammpstrj":
            raise ValueError("Input file must have a .lammpstrj extension.")
        cls.infile = infile
    
    def __init__(self):
        if LAMMPSTRJ.infile is None:
            ValueError("File does not exist.")
        self.loadLammptsrj()

        
    def loadLammptsrj(self):
        dataDic = lmp2Dic(self.infile)
        for key in ["elem", "nstep", "natoms"]:
            setattr(self, key, dataDic[key])
        for key in ["ldata", "lattice", "angle"]:
            setattr(self, f"lmp{key}", dataDic[key])
        print(f"Number of atoms: {self.natoms}")
        print(f"Number of Frame: {self.nstep}")
        matrix_list = []
        for i in range(self.lmpldata.shape[0]):
            matrix_, _ = setMatrix(self.lmplattice[i, :], self.lmpangle[i, :])
            matrix_list.append(matrix_.T)
        self.lmpmatrix = np.vstack(matrix_list).reshape(-1, 3, 3)
        del matrix_list
        if LAMMPSTRJ.index is None:
            if LAMMPSTRJ.len_index == 0:
                LAMMPSTRJ.index = [0, self.nstep - 1]
            else:
                LAMMPSTRJ.index = [int(100 + n *(self.nstep - 100) / (LAMMPSTRJ.len_index - 1)) - 1 for n in range(LAMMPSTRJ.len_index)]
        if max(LAMMPSTRJ.index) > self.nstep - 1:
            raise ValueError(f"\nRange 0 - {self.nstep - 1}\n")
        self.ndata = len(LAMMPSTRJ.index)
        fmt = "Convert Frame {} to CIF"
        frame_num = ", ".join(str(s) for s in LAMMPSTRJ.index)
        print(fmt.format(frame_num))
        print()

            
class CIF(LAMMPSTRJ):
    def __init__(self):
        super().__init__()
        self.base = os.path.splitext(self.infile)[0].split("/")[-1]
        self.pardir = os.path.dirname(os.path.abspath(self.infile))
        self.outputCif()
    
            
    def labeling_clayff(self):
        def calDist(r):
            assert r.ndim == 3
            r = (r @ np.linalg.inv(self.matrix).T).astype(float)
            r = (r - np.round(r)) @ self.matrix.T
            r = np.sqrt(np.sum(r**2, axis=2))
            return r
        elem = ["C", "Ca", "O", "H", "Si"]
        self.label = np.zeros((self.data.shape[0], ),dtype=object)
        for e in elem:
            idx = np.nonzero(self.data[:, 0] == e)[0]
            if idx.shape[0] != 0:
                setattr(self, e, idx)
                setattr(self, f"pos_{e}", self.data[idx])
        pos_O = self.pos_O[:, None, 1:]
        pos_H = self.pos_H[None, :, 1:]
        r = pos_O - pos_H
        r = calDist(r)
        bond_OH = np.where(r < 1.2, True, False)
        idx_bond = np.nonzero(bond_OH)
        idx_bond = np.vstack(idx_bond).T
        unique, index, count = np.unique(idx_bond[:, 0], return_counts=True, return_index =True)
        idx_water = np.nonzero(count==2)
        self.label[self.O[unique[idx_water]]] = "o*"
        for idx_ in unique[idx_water]:
            idx_water_h = idx_bond[idx_bond[:, 0] == idx_][:, 1]
            self.label[self.H[idx_water_h]] = "h*"

            
        if hasattr(self, "C"):
            pos_C = self.pos_C[:, None, 1:]
            pos_O = self.pos_O[None, :, 1:]
            r = pos_C - pos_O
            r = calDist(r)
            r = np.where(r < 1.5, True, False)
            indices = np.vstack(np.nonzero(r)).T
            idx_c, count = np.unique(indices[:, 0], return_counts=True)
            for i in range(idx_c.shape[0]):
                if count[i] == 3:
                    idx_o = indices[np.nonzero(indices[:, 0] == idx_c[i])][:, 1]
                    self.label[self.C[idx_c[i]]] = "co"
                    self.label[self.O[idx_o]] = "oc"
                else:
                    print("??"*10)
                    print(idx_c)
#                    raise ValueError("bond")

        if hasattr(self, "Ca"):
            flag = self.data[:, 0] == "Ca"
            offset = 5
            max_si = np.max(self.data[flag][:, 3])
            max_si += offset
            min_si = np.min(self.data[flag][:, 3])
            min_si -= offset
            layerCa = np.where((self.pos_Ca[:, 3] < max_si) & (self.pos_Ca[:, 3] > min_si), True, False)
            self.label[self.Ca[layerCa]] = "cah"
            self.label[self.Ca[~layerCa]] = "Ca"
            
            
        if hasattr(self, "Si"):
            pos_Si = self.pos_Si[:, None, 1:]
            pos_O = self.pos_O[None, :, 1:]
            r = pos_Si - pos_O
            r = calDist(r)
            r = np.where(r < 1.9, True, False)
            indices = np.vstack(np.nonzero(r)).T
            idx_si, count = np.unique(indices[:, 0], return_counts=True)
            self.label[self.Si[idx_si]] = "st"
            for i in range(idx_si.shape[0]):
                if count[i] == 4:
                    idx_o = indices[np.nonzero(indices[:, 0] == idx_si[i])][:, 1]
                    for i in range(idx_o.shape[0]):
                        idx = idx_o[i]
                        num_bond = np.sum(bond_OH[idx, :])
                        if num_bond == 0:
                            self.label[self.O[idx_o[i]]] = "ob"
                        elif num_bond == 1 or num_bond == 2:
                            self.label[self.O[idx_o[i]]] = "oh"
                            idx_h = np.nonzero(bond_OH[idx, :])[0]
                            self.label[self.H[idx_h]] = "ho"
                        else:
                            raise ValueError("bond")
                # else:
                    #raise ValueError(f"{count[i]}")
                
        not_label = np.nonzero(self.label == 0)[0]
        if len(not_label) != 0:
            not_label_elem = self.data[not_label][:, 0]
            self.label[not_label] = not_label_elem
            print("Unlabeled atoms\n", " ".join(f"idx {z1}:{z2}" for z1, z2 in zip(not_label, not_label_elem)))
        else:
            print("Every atom is labeled")

    def labeling(self):
        if self.labeltype == "default":
            return
        if self.labeltype == "clayff":
            self.labeling_clayff()
            
    
    def makeZshift(self):
        flag = self.data[:, 0] == "Si"
        if not np.any(flag):
            print("Do not have to shift\n")
            return
        self.data[:, 3] += self.zshift
        while True:
            self.data[flag, 1:] = np.mod(self.data[flag, 1:], 1.0)
            if np.all(self.data[flag, 3] < 0.6):
                break
            else:
                self.data[:, 3] += 0.001
                self.zshift += 0.01
        print(f"Executed z-shift ... : {self.zshift:.2f}")
        
        
    def makeSnapshot(self):
        for n in range(self.ndata):
            self.data = self.lmpldata[n]
            self.lattice = self.lmplattice[n]
            self.angle = self.lmpangle[n]
            self.matrix = self.lmpmatrix[n]
            self.labeling()
            self.data[:, 1:4] = self.data[:, 1:4] @ np.linalg.inv(self.matrix).T # -> fract
            fmt_head = fmt.format(self.base, *self.lattice, *self.angle).strip() + "\n"
            label = self.data[:, 0] if not hasattr(self, "label") else self.label
            symbol = self.data[:, 0] if not hasattr(self, "symbol") else self.symbol
            for i in range(self.data.shape[0]):
                fmt_tail = fmt_.format(label[i], *self.data[i, 1:], symbol[i])
                fmt_head += fmt_tail
            yield fmt_head

            
    def outputCif(self):
        snapshot = self.makeSnapshot()
        for idx in LAMMPSTRJ.index:
            timestep = idx
            filebase = f"{self.base}.{timestep}step"
            path = os.path.join(self.pardir, LAMMPSTRJ.out)
            os.makedirs(path, exist_ok=True)
            filepath = os.path.join(path,  f"{filebase}.cif")
            ciftxt = snapshot.__next__()
            with open(filepath, "w") as o:
                o.write(ciftxt)
                print(f"{filepath.replace('./', '')} was created")
                print()
                

def main():
    import argparse
    import glob
    description = """
    lammpstrj -> ciffile
    
    """.lstrip()
    par = argparse.ArgumentParser(description=description, prog="CIF")
    par.add_argument('infile', nargs="?", help="入力しなければFolderの一番前のファイルをとる")
    par.add_argument('-i', '--index', nargs="+", type=int, help="入力しなければ最後のStepを変換")
    par.add_argument('-l', '--len_index',  type=int, default=0, help="no default")
    par.add_argument('-t', '--labeltype', choices=["clayff", "default"], default="default", help="")
    par.add_argument('-z', '--zshift', type=float, default=0, help="axis-C, 'fraction coord'")
    par.add_argument('-s', '--symbol', default=False, action="store_true", help="axis-C, 'fraction coord'")
    par.add_argument('-o', '--out', default="cif")
    args = par.parse_args()

    infile = args.infile if args.infile else glob.glob("./*.lammpstrj")[0]
    LAMMPSTRJ.setLammpstrj(infile)
    LAMMPSTRJ.index = args.index
    LAMMPSTRJ.len_index = args.len_index
    LAMMPSTRJ.out = args.out
    if args.symbol:
        CIF.symbol = np.load("symbol.npy", allow_pickle=True)
    for w in ["labeltype", "zshift"]:
        setattr(CIF, w, getattr(args, w))
        
    CIF()

if __name__ == "__main__":
    main()
