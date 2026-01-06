#!/usr/bin/env python
from cshmd.trj_analysis import ANALYZER
from cshmd.analysis import UnitCell
import argparse
from pathlib import Path
import os

class Run_trj_analysis():
    verbose = False
    
    def __init__(self, infile):
        ANALYZER.verbose = Run_trj_analysis.verbose
        self.analyzer = ANALYZER(infile, UnitCell(infile))
        self.analyzer.trace_step()

def main():
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*",
                     help="'.lammpstrj'")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                     help="set verbosity level by True 'or' 'False'")
    par.add_argument('-g', '--glob', nargs="?", default="*e.lammpstrj",
                     help="set trjfile to glob when 'args.infiles' are not setted")
    
    args = par.parse_args()
    Run_trj_analysis.verbose = args.verbose
    infiles = args.infiles if len(args.infiles) != 0 else list(Path(".").rglob(args.glob))
    for infile in infiles:
        if not os.path.exists(infile):
            print(f"[INFO] File not found '{infile}'")
        else:
            Run_trj_analysis(infile)
    return 0
    
if __name__ == "__main__":
    main()
