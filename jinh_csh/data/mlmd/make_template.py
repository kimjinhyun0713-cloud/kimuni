#!/usr/bin/env python
from pathlib import Path
import os

data = ["BT.h.00.00.data", "BT.h.03.00.data",  "BT.h.06.00.data", "BT.h.09.00.data"]


class RUN:
    
    required = ["mlmd_eq.mod", "mlmd_run.mod"]
    n_data = 20
    fname = [f"{i:>03}" for i in range(n_data)]
    eq_fmt = "variable        base string {base}\n"
    eq_fmt += "variable        seed string {seed}\n"
    eq_fmt += "variable        elem string 'Ca C H H O O O O Si'\n"
    eq_fmt += "include         ../../mlmd_eq.mod"
    rst_fmt = "variable        base string {base}\n"
    rst_fmt += "variable        elem string 'Ca C H H O O O O Si'\n"
    rst_fmt += "include         ../../mlmd_run.mod"
            
    def __init__(self, infile):
        self.base = os.path.splitext(infile)[0]
        self.makedirs()
        self.setfiles()
                

    def makedirs(self):
        Path(self.base).mkdir(exist_ok=True)
        self.path_list = []
        for f in self.fname:
            pname = f"{self.base}/{f}"
            path = Path(pname)
            self.path_list.append(path)
            path.mkdir(exist_ok=True)

    def setfiles(self):
        for i, path in enumerate(self.path_list):
            seed =  str(i * 713 + 2)
            out = self.eq_fmt.format(base=self.base, seed=seed)
            fname = path.joinpath("eq.in")
            with open(fname, "w") as o:
                o.write(out)
            out = self.rst_fmt.format(base=self.base)
            fname2 = path.joinpath("rst.in")
            with open(fname2, "w") as o:
                o.write(out)
        print(f"{self.base}, done")
for d in data:
    RUN(d)
