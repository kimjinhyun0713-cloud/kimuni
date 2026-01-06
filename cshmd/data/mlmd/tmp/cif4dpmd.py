#!/usr/bin/env python
from jinh import cif2data, setMatrix
from jinh.util import mass
import numpy as np

class RUN:
    def __init__(self, infile):
        self.infile = infile
        self.lattice, self.angle, self.df = cif2data(self.infile)
        self.matrix, _, self.lmatrix = setMatrix(self.lattice, self.angle, return_lammps_matrix=True)
        self.natom = len(self.df)
        self.label_unique = np.unique(self.df["label"])
        self.setData()
        
    def setData(self):
        convert = {'cah': "Ca",
                   'co': "C",
                   'h*': "H",
                   'ho': "H",
                   'o*': "O",
                   'ob': "O",
                   'oc': "O",
                   'oh': "O",
                   'st': "Si"}
        head = "molecular system: {}\n".format(self.infile)
        head += "{} atoms\n".format(self.natom)
        head += "{} atom types\n".format(len(self.label_unique))
        head += "\n"
        L = self.lmatrix[:, 0:2]
        head += "{:>10.6f} {:>10.6f} xlo xhi\n".format(*L[0])
        head += "{:>10.6f} {:>10.6f} ylo yhi\n".format(*L[1])
        head += "{:>10.6f} {:>10.6f} zlo zhi\n".format(*L[2])
        Lz = self.lmatrix[:, 2]
        head += "{:>10.6f} {:>10.6f} {:>10.6f} xy xz yz\n".format(*Lz)
        head += "\nMasses\n\n"
        label = {}
        for i, atoms in enumerate(self.label_unique):
            label[atoms] = i + 1
            head += "{} {:10.6f} # {:4s}\n".format(label[atoms],
                                                   mass[convert[atoms]], atoms)
        xyz = self.df.loc[:, ["fract_x", "fract_y", "fract_z"]] @ self.matrix
        fmt = "{:6d} {:6d} {:10.6f} {:10.6f} {:10.6f} # {:4s}\n"
        body = "\nAtoms\n\n"
        for i, df in self.df.iterrows():
            body += fmt.format(i + 1, label[df["label"]],
                                *xyz.loc[i, 0:3], df["label"])
        dat = head + body
        with open(self.infile.replace(".cif", ".data"), "w") as o:
            o.write(dat)
    
def main():
    import argparse
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*", help="#Infile")
    args = par.parse_args()
    if len(args.infiles) == 0:
        import glob
        infiles = glob.glob("*cif")
    else:
        infiles = args.infiles
    for f in infiles:
        RUN(f)

if __name__ == "__main__":
    main()
