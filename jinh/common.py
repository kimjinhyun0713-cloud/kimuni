import numpy as np
import pandas as pd
from pathlib import Path
import sys, os
import subprocess as sp
import time

def type2list(data):
    """
    convert int, float or str to list
    if only in case the data can converted to numerical value
    
    args:
    ;; data

    return
    ;; data
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, (int, float)):
        return [data]
    elif isinstance(data, str):
        try:
            return [float(data)]
        except ValueError:
            raise TypeError("Error: cannot convert to a numerical value")
    
def list2arr(data, dtype=float):
    """
    convert int, float, str or list to array
    if only in case the data can converted to  numerical value
    convert to arr if only 
    if data's type is neither of that, then raise TypeError
    """
    if isinstance(data, np.ndarray):
        pass
    else:
        data = type2list(data)
        data = np.array(data, dtype=dtype)
    return data

def sum_traindata(path, return_df=False):
    """
    print name of traindata, nstep, natom recursively
    and if return_df, return DataFrame of that.

    stdout: name of traindata, nstep, natom
    
    args
    ;; path -> path
    
    return
    None if return_df df.DataFrame
    """
    total_nstep = 0
    generator = Path(path).rglob("coord.npy")
    if return_df:
        df_list = []
    for dat in generator:
        tname = str(dat).split("set.000")[0]
        print(f"{str(tname):<40s}", end="")
        loaded = np.load(dat)
        if loaded.ndim == 1:
            loaded.reshape(1, -1)
        try:
            nstep, natom = np.load(dat).shape
        except ValueError:
            print("no data")
            continue
        if return_df:
            df = pd.Series([tname, nstep, natom])
            df_list.append(df)
        natom /= 3
        total_nstep += nstep

        print(f"Steps: {int(nstep):<10}", f"Atoms: {int(natom):<10}")
    print()
    print("Total steps: ", total_nstep)
    return pd.concat(df_list, ignore_index=True) if return_df else None

def find2pipe():
    """
    convert pipelines stdout to list
    
    input
    ;;stdout
    ;;e.g.) find . -name ".ext" |
    
    return
    list,  result of stdout
    """
    files = [Path(line.strip()) for line in sys.stdin if line.strip()]
    return files

def string_filter(ptn, lst):
    """
    return list of string which have speific pattern
    
    args
    ;;pattern which you want to filter, lst
    
    return
    ;; list
    """
    return list(filter(lambda s: ptn in s, lst))

def sp_run(command):
    """
    implement command
    args
    ;; command -> list
    """
    return sp.run(command, capture_output=True, text=True)

def overwritePrint(string):
    """
    overwrite the stdout

    args
    ;; string -> str
    """
    print(f"\r{string}", end="", flush=True)


def checkTime(func):
    """
    decoration to check processing time
    args
    ;; func -> func
    
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"processing time : {func.__name__} in {end_time - start_time:.5f} seconds")
        return result
    return wrapper
