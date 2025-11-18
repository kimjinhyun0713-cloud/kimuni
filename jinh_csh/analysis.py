import numpy as np
from jinh import find_bond, find_mole, list2arr


class BOND():
    """
    input -> np.ndarray shape(-1, 4) [[elem, x, y, z], []...]
    Return Bond or Molecule index by call the instance
    """
    def __init__(self, data, matrix):
        self.data = data
        self.matrix = matrix
        self.elem = np.unique(self.data[:, 0])

    def __call__(self, keyword):
        if not hasattr(self, keyword):
            getattr(self, f"find_{keyword}")()
        val = getattr(self, keyword)
        return val

    def find_Si_O(self):
        self.Si_O = find_bond(self.data, self.matrix, "Si", "O", rcut=2.1, cartesian=False)

    def find_SiO4(self):
        if not hasattr(self, "Si_O"):
            getattr(self, "find_Si_O")()
        self.SiO4 = find_mole(self.Si_O, "Si", "O", nbond=4)
        
    def find_O_H(self):
        self.O_H = find_bond(self.data, self.matrix, "O", "H", rcut=1.2, cartesian=False)
        
    def find_OH(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.OH = find_mole(self.O_H, "O", "H", nbond=1)

    def find_H2O(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.H2O = find_mole(self.O_H, "O", "H", nbond=2)

    def find_C_O(self):
        self.C_O = find_bond(self.data, self.matrix, "C", "O", rcut=1.4, cartesian=False)
        
    def find_CO2(self):
        if not hasattr(self, "C_O"):
            getattr(self, "find_C_O")()
        self.CO3 = find_mole(self.C_O, "C", "O", nbond=2)

    def find_CO3(self):
        if not hasattr(self, "C_O"):
            getattr(self, "find_C_O")()
        self.CO3 = find_mole(self.C_O, "C", "O", nbond=3)

def cal_Qn(SiO4: np.ndarray, verbose=False):
    """
    found Qn from index of SiO4 
    input -> np.ndarray shape(-1, 5) [[Si, O1, O2, O3, O4], []...]
    """
    index_O = SiO4[:, 1:]
    Qn = []
    for i in range(SiO4.shape[0]):
        mask = np.arange(0, SiO4.shape[0])
        mask = np.ones((SiO4.shape[0], ), dtype=bool)
        mask[i] = False
        other_chains_O = index_O[mask]
        Qn.append(np.sum(np.isin(other_chains_O, index_O[i])))
    Qn = list2arr(Qn, dtype=int)
    if verbose:
        length = Qn.shape[0]
        for i in range(5):
            count = np.sum(Qn == i)
            if count != 0:
                print(f"[Info] Q{str(i)}: {count/length:.4f}\n")
    return Qn

        
