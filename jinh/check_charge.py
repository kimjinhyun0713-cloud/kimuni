#!/usr/bin/env python
import os, sys
from jinh import cif2data
from jinh.functions import cal_distance
import argparse
import numpy as np
import pandas as pd

par = argparse.ArgumentParser(description="", prog="")
par.add_argument('infile', nargs="*", help="")
args = par.parse_args()

if len(args.infile) == 0:
    import glob
    args.infile = glob.glob("./*cif")

charge = {}
charge["O"] = -2
charge["H"] = 1
charge["Ca"] = 2
charge["Si"] = 4
charge["C"] = 4

for infile in args.infile:
    _ , _ , df = cif2data(infile)
    charge_total = 0
    for w in charge.keys():
        mask = df.loc[:, ["type_symbol"]] == w
        num = np.sum(mask.to_numpy())
        charge_total += num * charge[w]
    print(f"\nFilename : {infile}\n", "charge  : ", charge_total, "\n")
    
