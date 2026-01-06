#!/usr/bin/env python
from cshmd.surface_map import Map_Data
from cshmd.analysis import UnitCell
from pathlib import Path

class Run_map_maker():
    def __init__(self, path):
        for p in path:
            Map_Data(p, UnitCell(Si_key_upper_lower=True))

def main():
    path = list(Path(".").rglob("e.center.lammpstrj"))
    if len(path) == 0:
        print("[INFO] No trj file exists")
        return False
    Run_map_maker(path)
    return True

if __name__ == "__main__":
    main()
