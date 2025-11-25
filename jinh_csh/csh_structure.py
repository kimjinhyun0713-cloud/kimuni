import numpy as np
import pandas as pd
import os, sys
from jinh import df2xyz, load_yaml
from jinh.editor import DATA
from .analysis import BOND, cal_Qn, cal_CS
import yaml

class STRUCTURE(DATA):
    
    mole_need_check = ["H2O", "SiO4", "OH", "CO3"]
    mole_dont_check = ["HCO3", "SiOH", "CaOH"]
    mole_all = mole_need_check + mole_dont_check + ["Ca"]
    info = []

    @classmethod
    def out2info(cls):
        df = pd.concat(cls.info, ignore_index=True)
#        df = df.set_index("Structure")
        print(df)

    def __init__(self, infile):
        super().__init__(infile)
#        print(load_yaml("csh_config.yml"))
        for v in ["mole_all", "mole_need_check", "verbose"]:
            setattr(self, v, getattr(self.__class__, v))
        self.data = df2xyz(self.df)
        self.init_setting()
        
    def __str__(self):
        string = " ".join(k for k in self.__dict__.keys())
        return string 
        
    def init_setting(self):
        self.elem = np.unique(self.data[:, 0])
        self.elem_index = {}
        self.is_csh = True if "Si" in self.elem else False
        for e in self.elem:
            self.elem_index[e] = np.nonzero(self.data[:, 0] == e)[0]

    def bond_info(self):
        bond = BOND(self.data, self.matrix)
        bond.verbose = self.verbose
        for m in self.mole_need_check:
            bond(m)
        if self.is_csh:
            range_Si = self.data[self.elem_index["Si"]][:, 3]
            self.layer_range = (min(range_Si), max(range_Si))
            bond.check_where_OH(range_CaOH=self.layer_range)
        else:
            bond.check_where_OH()
        for m in self.mole_all:
            if hasattr(bond, m):
                setattr(self, m, getattr(bond, m))
        
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
    STRUCTURE.out2info()
    print(csh)

if __name__ == "__main__":
    main()
