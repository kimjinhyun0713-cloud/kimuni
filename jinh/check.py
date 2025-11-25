import numpy as np
import re
import os, sys
from pathlib import Path

class JOB():
    
    col = ["loop", "loop_time", "energy", "virial", "temp", "step"]
    verbose = False
    
    @classmethod
    def is_pandas(cls, yes):
        if yes:
            cls.table = pd.DataFrame()
        else:
            cls.lst = []

    @classmethod
    def result(cls):
        if hasattr(cls, "table"):
            return cls.table.T
        else:
            fmt1 = "{:<25s}"
            fmt2 = "{:>10s}"
            string = fmt1.format("Job")
            string += " ".join(fmt2.format(c) for c in cls.col)
            string += "\n"
            string += "-" * (len(string) - 1)
            string += "\n"
            for val in cls.lst:
                if not all(np.isnan(v) for v in val[1]):
                    string += fmt1.format(val[0])
                    string += " ".join(fmt2.format(str(round(v, 3))) for v in val[1])
                    string += "\n"
            return string

    def __init__(self, path):
        self.path = path
        self.is_valid = self.setVaspFiles()
        if self.is_valid:
            self.load_Vasp()
        
    def vprint(self, string):
        if self.verbose:
            print(string)
    
    def setVaspFiles(self):
        keys = ["outcar", "oszicar"]
        vals = [("*outcar", "OUTCAR"),
                ("*oszicar", "OSZICAR")]
        fileDic = {k: v for k, v in zip(keys, vals)}
        for key, val in fileDic.items():
            for v in val:
                vaspfile = list(self.path.glob(v))
                if len(vaspfile) == 1:
                    self.vprint(f"[info]: {str(self.path):30s}: loaded {v}")
                    break
            else:
                self.vprint(f"[info]: {str(self.path):30s}: No {v}")
                vaspfile = None
            if vaspfile is not None:
                setattr(self, key, vaspfile[0])
        if not any(hasattr(self, key) for key in keys):
            return False
        return True
    
        
    def load_Vasp(self):
        self.max_length = - float("inf")
        if hasattr(self, "outcar"):
            self.load_OUTCAR()
        if hasattr(self, "oszicar"):
            self.load_OSZICAR()

            
    def load_OUTCAR(self):
        with open(self.outcar) as o:
            read = o.read()
        ptnDic = {}
        ptnDic["loop_time"] = re.compile(r"LOOP\+:.+?([0-9\.]+):")
        ptnDic["energy"] = re.compile(r"% ion-electron   TOTEN  = +([0-9\.\-]+)")
        ptnDic["virial"] = re.compile(r"external pressure = +?([0-9\.\-]+) kB")
        ptnDic["temp"] = re.compile(r"temperature  ([0-9\-\.]+) K")
        for k, ptn in ptnDic.items():
            found = re.findall(ptn, read)
            found = [float(v) for v in found]
            setattr(self, k, found)
            self.max_length = max(self.max_length, len(found))

            
    def load_OSZICAR(self):
        with open(self.oszicar) as o:
            read = o.read()
        ptn = re.compile(r"N +E.+?\n(.+?)(\d+ T=.+?)\n", re.DOTALL)
        data = re.findall(ptn, read)
        self.loop = []
        if not hasattr(self, "temp"):
            self.temp = []
            store_temp = True
        else:
            store_temp = False
        for d in data:
            scf = d[0].strip().split("\n")[-1].split()[1]
            t = d[1].split()[2]
            self.loop.append(int(scf))
            if store_temp:
                self.temp.append(float(t))
        self.max_length = max(self.max_length, len(data))

        
    def setData(self):
        if hasattr(self.__class__, "table"):
            self.setDataFrame()
        else:
            self.setArr()

    def setDataFrame(self):
        self.df = pd.DataFrame()
        for col in self.__class__.col:
            if hasattr(self, col):
                val = getattr(self, col)
                self.df[col] = list(val) + [np.nan] * (self.max_length - len(val))
        range_ = self.__class__.range_
        if range_ is not None:
            self.mean = self.df[int(range_[0]):int(range_[1])].mean()
        else:
            self.mean = self.df.mean()
            self.mean["step"] = len(self.df)
        self.__class__.table[str(self.path)] = np.round(self.mean, decimals=3)
            
    def setArr(self):
        lst = []
        for col in self.__class__.col[:-1]:
            val = getattr(self, col)
            lst.append(list(val) + [np.nan] * (self.max_length - len(val)))
        self.arr = np.vstack(lst).T
        range_ = self.__class__.range_
        step = self.arr.shape[0] if self.arr.shape[0] != 0 else np.nan
        if range_ is not None:
            self.mean = np.mean(self.arr[int(range_[0]):int(range_[1])], axis=0)
        else:
            self.mean = np.mean(self.arr, axis=0)
        self.mean = np.append(self.mean, step)
        self.__class__.lst.append([str(self.path),  self.mean])
        
if __name__ == "__main__":
    import argparse
    par = argparse.ArgumentParser(description="require 'outcar' or 'oszicar'", prog="Vasp Check")
    par.add_argument('paths', nargs="*",
                     help="Path of diretory where 'outcar or oszicar'")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                     help="if 'True' print length of Data and infomration")
    par.add_argument('-r', '--range',  nargs=2, type=int, default=[0, 0], 
                     help="Range of steps which you use in this program")
    par.add_argument('-pd', '--pandas',  default=False, action="store_true",
                     help="install of pandas")
    
    args = par.parse_args()
    try:
        import pandas as pd
        JOB.is_pandas(True)
    except ModuleNotFoundError:
        JOB.is_pandas(False)
        print("Pandas is NOT installed")
    if len(args.paths) == 0:
        paths = Path(".").glob("*/")
    else:
        paths = [Path(p) for p in args.paths]
    JOB.verbose = args.verbose
    JOB.range_ = None if all(r == 0 for r in args.range) else args.range
    for path in paths:
        job = JOB(path)
        if job.is_valid:
            job.setData()
    print(JOB.result())
