import numpy as np
import pandas as pd
import os, sys
from jinh import df2xyz, type2list
from jinh.editor import DATA
from .analysis import BOND, cal_Qn, cal_CS

class STRUCTURE(DATA):
    
    mole_need_check = ["H2O", "SiO4", "OH", "CO3"]
    mole_dont_check = ["HCO3", "SiOH", "CaOH"]
    mole_all = mole_need_check + mole_dont_check + ["Ca"]
    verbose = False
    info = []

    @classmethod
    def out2info(cls):
        df = pd.concat(cls.info, ignore_index=True)
#        df = df.set_index("Structure")
        print(df)

    def __init__(self, infile):
        super().__init__(infile)
        for v in ["mole_all", "mole_need_check", "verbose"]:
            setattr(self, v, getattr(self.__class__, v))
        self.init_setting()
        
    def __repr__(self):
        string = " ".join(k for k in self.__dict__.keys())
        return string

    def vprint(self, string: str):
        if self.verbose:
            print(string)    
        
    def init_setting(self):
        self.data = df2xyz(self.df)
        self.elem = np.unique(self.data[:, 0])
        self.elem_index = {}
        self.is_csh = True if "Si" in self.elem else False
        for e in self.elem:
            self.elem_index[e] = np.nonzero(self.data[:, 0] == e)[0]

    def update_matrix(self):
        super().update_matrix()
        
        
    def bond_info(self):
        self.bond = BOND(self.data, self.matrix)
        self.bond.verbose = self.verbose
        for m in self.mole_need_check:
            self.bond(m)
        if self.is_csh:
            range_Si = self.data[self.elem_index["Si"]][:, 3]
            self.layer_range = (min(range_Si), max(range_Si))
            self.bond.check_where_OH(range_CaOH=self.layer_range)
        else:
            self.bond.check_where_OH()
        for m in self.mole_all:
            if hasattr(self.bond, m):
                setattr(self, m, getattr(self.bond, m))

                
    def validate_chain(self):
        if not self.is_csh:
            return
        isin_tetra = np.isin(self.elem_index["Si"], self.SiO4[:, 0])
        if not isin_tetra.all():
            broken = self.elem_index["Si"][~isin_tetra]
            broken_str = ' '.join(str(v) for v in broken)
            print(f"\nWarning: Some Silicate Chain broken: {broken_str}")
            sys.exit(1)
        self.Qn = cal_Qn(self.SiO4, verbose=self.verbose)
        self.CS = cal_CS(self.data, layer=self.layer_range, verbose=self.verbose)

        
    def validate_tetra(self):
        if not hasattr(self, "model"):
            return
        model = getattr(self, "model")
        self.tetra = {"BT": {}, "PT": {}}
        self.tetra["BT"] = {"surface": [], "bulk": []}
        self.tetra["PT"] = {"surface": [], "bulk": []}
        BTcount = 0
        PTcount = 0
        for key1 in model.keys():
            for key2 in model[key1].keys():
                ranged = model[key1][key2]
                for s in ranged:
                    mask1 = self.data[self.elem_index["Si"], 3] > s[0]
                    mask2 = self.data[self.elem_index["Si"], 3] < s[1]
                    mask = mask1 & mask2 
                    self.tetra[key1][key2].append(self.elem_index["Si"][mask][0])
                    if key1 == "BT":
                        BTcount += np.sum(mask)
                    else:
                        PTcount += np.sum(mask) 
        assert (BTcount + PTcount) == self.elem_index["Si"].shape[0], "Check model file"
        self.vprint(f"[Structure] BT: {BTcount}, PT: {PTcount}")
        
        
    def terminate(self, target: int | list):
        target = type2list(target)
        for t in target:
            self.protonation(t)

    def decorate_Ca(self, target: int | list):
        target = type2list(target)
        for t in target:
            cartesian = self.data[t, 1:] @ self.matrix
            fract = (cartesian + [0, 0, 1.7]) @ np.linalg.inv(self.matrix)
            self.insert(fract, symbol="Ca", label="Ca")
            
    def remove_BT(self, target: int | list, termination: bool):
        if not hasattr(self, "model"):
            raise ValueError("Missing keyword' model' in config")
        mask = np.isin(self.SiO4[:, 0], target)
        target_list = type2list(target)
        target_T = self.SiO4[mask]
        candidate_O = np.unique(target_T[:, 1:])
        NBO = self.bond.find_NBO()
        target_O = candidate_O[np.isin(candidate_O, NBO)]
        if termination:
            self.terminate(target_O)
        else:
            self.decorate_Ca(target)
        target_OH = self.SiOH[np.isin(self.SiOH[:, 0], target_O)].ravel()
        target_list = np.append(target_OH, target_list)
        target_list = sorted(target_list, reverse=True)
        for t in target_list:
            self.delete(index=t)
            
            
    def set_info(self):
        df = pd.DataFrame()
        df["Structure"] = [self.base]
        for m in self.mole_all:
            if hasattr(self, m):
                val = getattr(self, m)
                num = val.shape[0]
                df[m] = [int(num)]
        self.__class__.info.append(df)

        
def main():
    from pathlib import Path
    import argparse
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infile', nargs="*",
                     help="Input CIF file (if not provided, it will be selected automatically)")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                     help="set verbosity level by True 'or' 'False'")
    args = par.parse_args()
    STRUCTURE.verbose = args.verbose
    if args.infile is not None:
        infile = args.infile
    else:
        infile = list(Path(".").glob("*cif"))
    for f in infile:
        csh = STRUCTURE(f)
        csh.bond_info()
        csh.validate_chain()
        csh.set_info()
    STRUCTURE.out2info()
    print(csh)

if __name__ == "__main__":
    main()
