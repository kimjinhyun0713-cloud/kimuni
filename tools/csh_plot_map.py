#!/usr/bin/env python
from cshmd.surface_map import Mapper, Plotter
from pathlib import Path
import os

class Run_map_plotter():
    def __init__(self, paths, cutoff_size=None):
        for path in paths:
            Mapper(path)
        Plotter.cwd = Path(os.getcwd())
        print(f"[INFO] CWD: {Plotter.cwd}")
        self.plot = Plotter(path, Mapper())
        if cutoff_size is not None:
            self.plot.cutoff_size = cutoff_size

    def __call__(self, plot="contour", **setup):
        getattr(self, plot)(**setup)

    def adp(self, **setup):
        self.plot.set_adp(**setup)

    def contour(self, **setup):
        self.plot.set_contour(**setup)

    def cutoff(self, val):
        self.plot.cutoff_size = val
    

def main():
    import argparse
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*", help="# lammpstrj")
    par.add_argument('-o', '--outfiles', help="Please set a outfile")
    args = par.parse_args()
    path = [Path(f) for f in args.infiles]
    if len(path) == 0:
        path = list(Path(".").rglob("*map.npz"))
    if len(path) == 0:
        print("[INFO] No trj file exists")
        return False
    plotter = Run_map_plotter(path)
    plotter(plot="adp", plot_Ca=True)
    plotter(plot="adp", plot_I=True)
    plotter.cutoff(15)
    plotter(plot="contour", plot_I=True, plot_Si=True)
    plotter(plot="contour", plot_Ca=True, plot_Si=True)
    plotter(plot="contour", plot_Ca=True, plot_I=True, plot_Si=False)
    # plotter(plot="contour", plot_Ca=True, colorbar=True)

    return True

if __name__ == "__main__":
    main()
