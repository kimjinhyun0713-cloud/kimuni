#!/usr/bin/env python
import os
from cshmd.trj_analysis import ANALYZER
from cshmd.analysis import UnitCell

class Run_trj_analysis():
    verbose = False
    crystal_centered = True
    
    def __init__(self, infile):
        ANALYZER.verbose = Run_trj_analysis.verbose
        ANALYZER.crystal_centered = Run_trj_analysis.crystal_centered
        self.analyzer = ANALYZER(infile, UnitCell(infile))
        self.analyzer.center_step_()
        
def main():
    import argparse
    from pathlib import Path
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*",
                     help="'.lammpstrj'")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                     help="set verbosity level by True 'or' 'False'")
    par.add_argument('-c', '--crystal_centered', default=True, action="store_false",
                     help="set C-S-H in middle of the system")
    par.add_argument('-g', '--glob', nargs="?",
                     help="set trjfile to glob when 'args.infiles' are not setted")
    
    args = par.parse_args()
    Run_trj_analysis.verbose = args.verbose
    Run_trj_analysis.crystal_centered = args.crystal_centered
    if args.glob is None: args.glob = "e.lammpstrj"
    infiles = args.infiles if len(args.infiles) != 0 else list(Path(".").rglob(args.glob))
    for infile in infiles:
        if not os.path.exists(infile):
            print(f"[INFO] File not found '{infile}'")
        else:
            Run_trj_analysis(infile)
    return 0
    
if __name__ == "__main__":
    main()
