#!/usr/bin/env python
from .csh_structure import STRUCTURE
from .functions import config_base
from jinh import load_yaml
from pathlib import Path
import argparse
import importlib.resources as res

def builder(base, out, supercell=[3, 3, 1], config=None):
    d = res.files("jinh_csh.data") 
    infile = d.joinpath(f"{base}.cif")
    config = config if config is not None else base
    csh = STRUCTURE(infile)
    csh.config = load_yaml(d.joinpath(f"{base}.yaml"))
    config_base()
    csh.makeSupercell(supercell)
    csh.init_setting()
    csh.update_matrix()
    csh.bond_info()
    base = base.replace(".cif", "") + ".cif"
    csh.outfile = base
    csh.verbose = True
    csh.validate_chain()
    csh.config_chain()
    csh.writeCif()
    return

def main():
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infile', nargs="*",
                 help="Input CIF file (if not provided, it will be selected automatically)")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                 help="set verbosity level by True 'or' 'False'")
    args = par.parse_args()
    STRUCTURE.verbose = args.verbose
    builder("T14_base", "T14_normal", supercell=[3, 3, 1])

if __name__ == "__main__":
    main()
