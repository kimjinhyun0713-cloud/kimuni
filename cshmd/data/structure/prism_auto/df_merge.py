#!/usr/bin/env python
from pathlib import Path
import pandas as pd
from cshmd.load import Pkl


class Pkl_mcl(Pkl):
    def __init__(self, path):
        super().__init__(path)

    @property
    def MCL(self):
        return self.value["MCL"][0]


path = []
path.extend(list(Path(".").glob("L???_w2750.tmp.pkl")))
path.extend(list(Path(".").glob("L???_w2750_c.tmp.pkl")))
data = [Pkl_mcl(p) for p in path if "L120" not in str(p) and "L070" not in str(p)]
data = sorted(data, key=lambda x: float(x.value["MCL"][0]))
dfs = pd.concat([d.value for d in data])
print(dfs)
dfs.to_pickle("merge.pkl")
