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
        self.plot = Plotter(Mapper())
        if cutoff_size is not None:
            self.plot.cutoff_size = cutoff_size

    def __call__(self, plot="contour", **setup):
        getattr(self, plot)(**setup)

    def adp(self, **setup):
        self.plot.set_adp(**setup)

    def contour(self, **setup):
        self.plot.set_contour(**setup)

def main():
    globed_path = list(Path(".").rglob("map.npz"))
    plotter = Run_map_plotter(globed_path)
    plotter(plot="adp", plot_Ca=True)
    plotter(plot="contour", plot_Si=True)
    # plotter(plot="contour", plot_Ca=True, colorbar=False)

    return True

if __name__ == "__main__":
    main()
