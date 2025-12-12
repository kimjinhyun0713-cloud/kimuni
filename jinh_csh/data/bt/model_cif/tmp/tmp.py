#!/usr/bin/env python
from jinh import cif2data
import pandas as pd
df_lst = [cif2data("w.bt.00.h.200step.cif"),
          cif2data("w.bt.03.h.200step.cif"),
          cif2data("w.bt.06.h.200step.cif"),
          cif2data("w.bt.09.h.200step.cif")]

df_lst = [d[-1] for d in df_lst]

for df in df_lst:
    mask = df["type_symbol"] == "Ca"
    val = df.loc[mask, 'fract_z'].mean() + 0.5
    print(f"0.5-0.5-{val}-CO3")
    
