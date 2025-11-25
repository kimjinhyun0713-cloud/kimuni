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
        self.Ca = np.nonzero(self.data[:, 0] == "Ca")[0]
        
    def __call__(self, keyword):
        if not hasattr(self, keyword):
            getattr(self, f"find_{keyword}")()
        val = getattr(self, keyword)
        return val

    def vprint(self, string):
        if self.verbose:
            print(string)
    
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

    def check_where_OH(self, range_CaOH=None):
        name_convert = {"SiO4": "SiOH",
                        "CO3": "HCO3",
                        "Ca": "CaOH"
                        }
        if not hasattr(self, "OH"):
            getattr(self, "find_OH")()
        if self.OH.size == 0:
            return
        O = self.OH[:, 0]
        for m in ["CO3", "SiO4"]:
            if hasattr(self, m):
                self.vprint(f"[Info]: Searching 'OH' in {m}")
                val_m = getattr(self, m)
                if val_m.size == 0:
                    continue
                molecular_O = val_m[:, 1:].ravel()
                not_in_mole = self.OH[~np.isin(O, molecular_O)]
                in_mole = self.OH[np.isin(O, molecular_O)]
                setattr(self, "OH", not_in_mole)
                setattr(self, name_convert[m], in_mole)
                O = self.OH[:, 0]
        if range_CaOH is not None:
            condition1 = self.data[O][:, 3] < range_CaOH[1]
            condition2 = self.data[O][:, 3] > range_CaOH[0]
            condition = condition1 & condition2
            if condition.any():
                not_in_caoh = self.OH[~condition]
                in_caoh = self.OH[condition]
                setattr(self, "OH", not_in_caoh)
                setattr(self, name_convert["Ca"], in_caoh)

def cal_Qn(SiO4: np.ndarray, verbose=False):
    """
    Cal Qn from index of SiO4 
    SiO4 -> np.ndarray shape(-1, 5) [[Si, O1, O2, O3, O4], []...]
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
            if count != 0 and verbose:
                print(f"[Stucture] Q{str(i)}: {count/length:.3f}")
    return Qn

        
def cal_CS(data: np.ndarray, layer=None, verbose=False):
    """
    cal Ca/Si from data
    data -> np.ndarray shape(-1, 5) [[elem, O1, O2, O3, O4], []...]
    """
    n_Si = np.sum(data[:, 0] == "Si")
    if layer is None:
        n_Ca = np.sum(data[:, 0] == "Ca")
    else:
        mask1 = data[:, 3] > layer[0]
        mask2 = data[:, 3] < layer[1]
        mask = mask1 & mask2
        n_Ca = np.sum(data[mask][:, 0] == "Ca")
        CS = n_Ca/n_Si
        if verbose:
            print(f"[Stucture] C/S: {CS:.3f}")
    return CS
        
