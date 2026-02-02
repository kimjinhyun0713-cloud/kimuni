#!/usr/bin/env python
from cshmd.surface_map import Map_Data
from cshmd.analysis import UnitCell
from pathlib import Path

class Run_map_maker():
    def __init__(self, path):
        for p in path:
            Map_Data(p, UnitCell(p, Si_key_upper_lower=True))

def main():
    import argparse
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*", help="# lammpstrj")
    par.add_argument('-o', '--outfiles', help="Please set a outfile")
    args = par.parse_args()
    path = [Path(f) for f in args.infiles]
    if len(path) == 0:
        path = list(Path(".").rglob("e.center.lammpstrj"))
    if len(path) == 0:
        print("[INFO] No trj file exists")
        return False
    Run_map_maker(path)
    return True

if __name__ == "__main__":
    main()
