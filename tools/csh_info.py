#!/usr/bin/env python

def main():
    from cshmd.structure import STRUCTURE
    from pathlib import Path
    import argparse
    description = """ Result ->
                         Structure  Natoms  Charge  Ca  H2O  SiO4  OH  CO3  HCO3  SiOH  CaOH
    0    CaOH2_0900K_10_vasp.0step     180       0  36    0     0  72    0     0     0     0
    1  CaOH2_0900K_10_vasp.599step     180       0  36    0     0  72    0     0     0     0
    """
    par = argparse.ArgumentParser(description=description,
                                  prog="[csh_info]",
                                  formatter_class=argparse.RawTextHelpFormatter)
    par.add_argument('infile', nargs="*",
                     help="Input CIF file (if not provided, it will be selected automatically)")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                     help="set verbosity level by True 'or' 'False'")
    args = par.parse_args()
    STRUCTURE.verbose = args.verbose
    if len(args.infile) != 0:
        infile = args.infile
    else:
        infile = list(Path(".").glob("*.cif"))
    if len(infile) == 0:
        print("[INFO] No trj file exists")
    for f in infile:
        csh = STRUCTURE(f)
        csh.set_info()
    STRUCTURE.out2info()
        
if __name__ == "__main__":
    main()
