#!/usr/bin/env python
from jinh_csh import STRUCTURE
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
