#!/usr/bin/env python
#from jinh_csh._structure import STRUCTURE
from tmp_structure import STRUCTURE
from jinh_csh.functions import data_lib
import sys
import numpy as np
import argparse
import yaml
from dataclasses import dataclass

@dataclass
class BUILDER():
    
    model: dict
    base: str
    out: str

    def set_model(self):
        self.base = self.base.replace(".cif", "") + ".cif"
        self.infile = data_lib(f"{self.base}")
        self.csh = STRUCTURE(self.infile)
        self.csh.outfile = self.out.replace(".cif", "") + ".cif"
        self.csh.model = self.model
        self.csh.bond_info()
        self.csh.set_model()
        
    def validation(self):
        self.csh.verbose = False
        self.csh.validate_chain()
        self.csh.validate_tetra()
               
            
    def supercell(self, supercell=None):
        if supercell is None or len(supercell) == 0:
            return False
        self.csh.makeSupercell(supercell)
        self.csh.init_setting()
        self.csh.update_matrix()


    def defect_BT(self, scope="all", seed=10, termination=False):
        self.validation()
        BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
        for i in range(2):
            if scope == "all":
                target = BT[i]
            elif scope == "half" or isinstance(scope, int):
                n_tetra = BT[i].shape[0]
                n_del = n_tetra // 2 if scope == "half" else scope
                if n_del > n_tetra:
                    warning = f"ERROR: scope must be smaller than {n_tetra}"
                    raise ValueError(warning)
                print(f"[Structure] Remove {n_del} Bridging Teta")
                np.random.seed(seed)
                mask = np.random.choice(n_tetra, size=n_del, replace=False)
                target = BT[i][mask]
            else:
                raise ValueError("ERROR: Wrong Datatype")
            location = "top" if i == 0 else "bottom"
            self.csh.remove_BT(target, termination, location=location)

    def remove(self):
        if len(self.csh.rm_list) != 0:
            arr = np.hstack(self.csh.rm_list)
            target_list = sorted(arr, reverse=True)
            for t in target_list:
                self.csh.delete(index=t)
        self.csh.init_setting()
        self.csh.update_matrix()

    def out_model(self):
        self.remove()
        self.csh.bond_info()
        self.validation()
        self.csh.writeCif()
        charge = self.csh.return_charge()
        print(f"Total Charge: {charge}")

def main():
    config = yaml.safe_load(sys.stdin.read())
    build = BUILDER(**config["info"])
    build.set_model()
    for c in config["command"]:
        getattr(build, c["func"])(**c["args"])
    build.out_model()
        
if __name__ == "__main__":
    main()
