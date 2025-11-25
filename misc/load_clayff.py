#!/usr/bin/env python
from jinh.load import excel2Dic
import pandas as pd 
import os
import pickle

ff = f"{os.environ['HOME']}/bin/clayff.xlsx"
dfs = excel2Dic(ff)
df = dfs["VDW"]
charge = df[["type", "charge"]]
charge.to_pickle("clayff.pkl")
print("generate 'clayff.pkl'")
