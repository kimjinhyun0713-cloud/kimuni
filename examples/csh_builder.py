#!/usr/bin/env python
from jinh_csh.structure import STRUCTURE
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
        self.infile = data_lib(f"{self.base}")
        self.csh = STRUCTURE(self.infile)
        self.csh.model = self.model
        self.csh.bond_info()
            
    def supercell(self, supercell=None):
        if supercell is None or len(supercell) == 0:
            return False
        self.csh.makeSupercell(supercell)
        self.csh.init_setting()
        self.csh.update_matrix()
        self.csh.outfile = self.out.replace(".cif", "") + ".cif"

    def validation(self):
        self.csh.verbose = True
        self.csh.validate_chain()
        self.csh.validate_tetra()

    def defect_BT(self, scope=None, termination=False):
        scope = "half"
        termination = False
        if scope is None:
            raise ValueError("Missing keyword 'scope' in config beyond 'defect'")
        self.validation()
        BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
        if scope == "all":
            target = BT
        elif scope == "half":
            n_tetra = BT.shape[0]
            n_del = n_tetra // 2
            mask = np.random.choice(n_tetra, size=n_del, replace=False)
            target = BT[mask]
        self.csh.remove_BT(target, termination)

    def write(self):
        self.csh.writeCif()

def main():
    config = yaml.safe_load(sys.stdin.read())
    build = BUILDER(**config["info"])
    build.set_model()
    for c in config["command"]:
        getattr(build, c["func"])(**c["args"])
        
if __name__ == "__main__":
    main()
