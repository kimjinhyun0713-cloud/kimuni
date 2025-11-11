from .functions import cal_distance
from .common import list2arr
from .util.molecule import molCharge
import numpy as np
import pandas as pd



def find_bond(data, matrix, elem1="O", elem2="H", rcut=1.2, cartesian=False):
    """
    In dataset with coordinations, elements, matrix,
    find bonds and return index of bond pairs

    args:
    ;; data -> list or np.nadarry(-1, 4), [elem, x, y, z]
    ;; matrix -> np.nadarry(3, 3)
    ;;; elem1 -> str, default "O"
    ;;; elem2 -> str, default "H"
    ;;; rcut -> float, default=1.2
    ;;; cartesian -> bool, type of coordinations
    
    return
    ;; arr of bonds -> np.nadarry(-1, 2)
    """
    data = list2arr(data)
    matrix = list2arr(matrix)
    mask1 = data[:, 0] == elem1
    mask2 = data[:, 0] == elem2
    which_e1 = np.nonzero(mask1)[0]
    which_e2 = np.nonzero(mask2)[0]
    pos1 = data[mask1, 1:]
    pos2 = data[mask2, 1:]
    r = cal_distance(pos1, pos2, matrix, cartesian=cartesian)
    indexs = np.where(r < rcut)
    bond = np.vstack([which_e1[indexs[0]], which_e2[indexs[1]]]).T
    return bond

def find_mole(arr, elem1="O", elem2="H", nbond=2):
    """
    find molecules from indexs which can calculate through "find_bond"
    args:
    ;; arr -> np.ndarray, indexs if bonds (-1, 2)
    ;;; elem1 -> str, default "O"
    ;;; elem2 -> str, default "H"
    ;;; nbond -> int, default 2, the number of bonds to identify molecules

    return
    ;; arr of molecules -> np.nadarry(-1, nbond)
    """
    arr = list2arr(arr)
    idx, count = np.unique(arr[:, 0], return_counts=True)
    center = idx[count == nbond]
    if len(center) == 0:
        return np.array([], dtype=int)
    center_list, boundary_list  = list(), list()
    for c in center:
        mask = c == arr[:, 0]
        boundary = arr[mask, 1]
        center_list.append(c)
        boundary_list.append(boundary)
        
    c_arr = np.vstack(center_list)
    b_arr = np.vstack(boundary_list)
    mole = np.hstack([c_arr, b_arr])
    return mole
        

def unwrap_mole(pos, matrix):
    """
    convert fraction coordinations, into unwrapped coordinations,
    which means do not consider periodic boundary condition
    key is coordination of index 0
    
    args
    ;; coordinations -> list or nd.ndarray(nmole, 3)
    ;; matrix -> np.nadarry(3, 3)

    return
    ;; coordinations, nd.ndarray(3, ), which unwrapped
    """
    pos = list2arr(pos)
    matrix = list2arr(matrix)
    r = pos[1:, :] - pos[0, :]
    boundary = pos[1:, :]
    new_boundary = np.where(np.abs(r) < 0.5, boundary, boundary - np.sign(r))
    pos[1:, :] = new_boundary
    return pos

def df2charge(df):
    """
    calculate total charge of the system
    args
    ;; pd.DataFrame with have column "tyep_symbol"

    return charge -> int
    """
    charge_total = 0
    unique = np.unique(df.loc[:, ["type_symbol"]].to_numpy())
    for w in unique:
        mask = df.loc[:, ["type_symbol"]] == w
        num = np.sum(mask.to_numpy())
        charge_total += num * molCharge[w]
    return charge_total
    

if __name__ == "__main__":
    pos = [[0, 0, 1], [-0.1, -0.1, 1.2], [0.98, 0.98, 0.02]]
    print(unwrap_mole(pos))
