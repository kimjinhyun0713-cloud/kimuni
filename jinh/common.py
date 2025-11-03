import numpy as np
from pathlib import Path
import sys, os
import subprocess as sp 
def list2arr(data):
    """
    if data's type is list, convert to arr
    if data's type is neither list nor arr
    raise TypeError
    """
    if isinstance(data, list):
        data = np.array(data, dtype=float)
    if not isinstance(data, np.ndarray):
        raise TypeError("Error: input must to be list or arr")
    return data

    
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
