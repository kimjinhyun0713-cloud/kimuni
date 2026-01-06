import numpy as np
import pandas as pd
from pathlib import Path
import os
import yaml
from importlib import resources
from .functions import cal_distance, cal_distance_with_block, setMatrix
from .common import list2arr, df2xyz
from .util.molecule import molCharge
from .load import cif2data


def find_bond(data, matrix, elem1="O", elem2="H",
              rcut=1.2, cartesian=False, use_block=True,
              reverse=False, reverse_rcut=None):
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
    if reverse and reverse_rcut is None:
        msg = "[Error]: Set Both 'reverse' and  'reverse_rcut'"
        raise ValueError(msg)
    data = list2arr(data)
    matrix = list2arr(matrix)
    mask1 = data[:, 0] == elem1
    mask2 = data[:, 0] == elem2
    which_e1 = np.nonzero(mask1)[0]
    which_e2 = np.nonzero(mask2)[0]
    pos1 = data[mask1, 1:]
    pos2 = data[mask2, 1:]
    func = cal_distance_with_block if use_block else cal_distance
    r = func(pos1, pos2, matrix, cartesian=cartesian)
    indexs = np.where(r < rcut)
    bond = np.vstack([which_e1[indexs[0]], which_e2[indexs[1]]]).T
    if reverse:
        r_ = func(pos2, pos1, matrix, cartesian=cartesian)
        indexs_ = np.where((r_ < reverse_rcut) & (r_ > rcut))
        bond_ = np.vstack([which_e2[indexs_[0]], which_e1[indexs_[1]]]).T
        return bond, bond_
    else:
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
    center_list, boundary_list = list(), list()
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

def df2charge4clayff(df, return_charge=False):
    base = Path(__file__).resolve().parent
    charge_df = pd.read_pickle(f"{base}/data/clayff.pkl")
    charge_total = 0
    unique = np.unique(df.loc[:, ["label"]].to_numpy())
    for w in unique:
        if w in ["co", "oc"]:
            continue
        mask = df.loc[:, ["label"]] == w
        num = np.sum(mask.to_numpy())
        where = charge_df["type"] == w
        c = charge_df.loc[where, "charge"].iloc[0]
        charge_total += num * c
    if return_charge:
        return charge_total, charge_df
    else:
        return charge_total


class BOND():
    """
    input -> np.ndarray shape(-1, 4) [[elem, x, y, z], []...]
    Return Bond or Molecule index by call the instance
    """
    mole = ["H2O", "SiO4", "OH", "CO3", "H3O"]
    mole_extra = ["SiOH", "HCO3", "CaOH", "HCO3", "H2CO3", "H3CO3"]
    mole_hco3_family = ["HCO3", "H2CO3", "H3CO3"]
    verbose = False
    
    def __init__(self, data, matrix, cartesian=False):
        self.data = data
        self.matrix = matrix
        self.cartesian = cartesian
        self.elem = np.unique(self.data[:, 0])
        self._initialize_mole(*self.__class__.mole,
                              *self.__class__.mole_extra)
        
    def __call__(self, keyword):
        val = getattr(self, keyword)
        return val

    def _initialize_mole(self, *args):
        for arg in args:
            setattr(self, arg, np.empty(0))

    def find(self, *args):
        for arg in args:
            getattr(self, f"find_{arg}")()

    def vprint(self, string):
        if self.verbose:
            print(string)
            
    def find_Si_O(self):
        self.Si_O = find_bond(self.data, self.matrix,
                              "Si", "O",
                              rcut=2.1, cartesian=self.cartesian)

    def find_SiO4(self):
        if not hasattr(self, "Si_O"):
            getattr(self, "find_Si_O")()
        self.SiO4 = find_mole(self.Si_O, "Si", "O", nbond=4)
        
        
    def find_O_H(self, reverse=False, reverse_rcut=3.5):
        returned = find_bond(self.data,
                             self.matrix,
                             "O", "H",
                             rcut=1.2,
                             cartesian=self.cartesian,
                             reverse=reverse,
                             reverse_rcut=reverse_rcut)
        if not reverse:
            self.O_H = returned
        else:
            self.O_H, self.H_O = returned
            
        
    def find_OH(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.OH = find_mole(self.O_H, "O", "H", nbond=1)

    def find_H2O(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.H2O = find_mole(self.O_H, "O", "H", nbond=2)


    def find_H3O(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.H2O = find_mole(self.O_H, "O", "H", nbond=3)

    def find_C_O(self):
        self.C_O = find_bond(self.data, self.matrix,
                             "C", "O",
                             rcut=1.6, cartesian=self.cartesian)
        
    def find_CO2(self):
        if not hasattr(self, "C_O"):
            getattr(self, "find_C_O")()
        self.CO2 = find_mole(self.C_O, "C", "O", nbond=2)

    def find_CO3(self):
        if not hasattr(self, "C_O"):
            getattr(self, "find_C_O")()
        self.CO3 = find_mole(self.C_O, "C", "O", nbond=3)

    def find_Ca_O(self):
        self.Ca_O = find_bond(self.data, self.matrix,
                              "Ca", "O",
                              rcut=3.0, cartesian=self.cartesian)
        
    def cal_BO(self):
        if not hasattr(self, "SiO4"):
            getattr(self, "SiO4")()
        unique = np.unique(self.SiO4[:, 1:], return_counts=True)
        self.BO = np.vstack(unique).T

        
    def find_NBO(self):
        if not hasattr(self, "BO"):
            getattr(self, "cal_BO")()
        mask = self.BO[:, 1] == 1
        return self.BO[mask][:, 0]

    def _check_and_find_co3s(self):
        if hasattr(self, "CO3"):
            O = self.OH[:, 0]
            self.vprint("[Info]: Searching 'OH' in 'CO3'")
            val_m = getattr(self, "CO3")
            if val_m.size == 0:
                return
            molecular_O = val_m[:, 1:].ravel()
            isin = np.isin(O, molecular_O)
            not_in_mole = self.OH[~isin]
            in_mole = self.OH[isin]
            if in_mole.size != 0:
                for co3s in self.__class__.mole_hco3_family:
                    setattr(self, co3s, list())
                co3_table = np.isin(val_m, in_mole)
                co3_h_series = np.sum(co3_table, axis=1)
                site_true = np.vstack(np.nonzero(co3_table)).T
                is_co3 = np.where(co3_h_series == 0)[0]
                isnt_co3 = np.where(co3_h_series != 0)[0]
                H_index_dic = {}
                for index in isnt_co3:
                    H_index_dic[index] = []
                for axis0, axis1 in site_true:
                    O_index = val_m[axis0, axis1]
                    H_index = in_mole[in_mole[:, 0] == O_index][0, 1]
                    H_index_dic[axis0].append(H_index)
                for co3_index, H_index_list in H_index_dic.items():
                    len_H = len(H_index_list)
                    co3s_name = f'H{str(len_H) if len_H != 1 else ""}CO3'
                    species_arr = np.hstack([val_m[co3_index], H_index_list])
                    getattr(self, co3s_name).append(species_arr)
                for co3s in self.__class__.mole_hco3_family:
                    val = getattr(self, co3s)
                    if len(val) != 0:
                        new_val = np.vstack(val)
                        setattr(self, co3s, new_val)
                    else:
                        delattr(self, co3s)
                setattr(self, "CO3", val_m[is_co3])
            setattr(self, "OH", not_in_mole)

    def _check_and_find_sioh(self):
        if hasattr(self, "SiO4"):
            self.vprint("[Info]: Searching 'OH' in 'SiO4")
            val_m = getattr(self, "SiO4")
            if val_m.size == 0:
                return
            O = self.OH[:, 0]
            molecular_O = val_m[:, 1:].ravel()
            isin = np.isin(O, molecular_O)
            not_in_mole = self.OH[~isin]
            in_mole = self.OH[isin]
            setattr(self, "OH", not_in_mole)
            setattr(self, "SiOH", in_mole)
            
    
    def check_where_OH(self, range_CaOH=None):
        if not hasattr(self, "OH"):
            getattr(self, "find_OH")()
        if self.OH.size == 0:
            return
        self._check_and_find_co3s()
        self._check_and_find_sioh()
        if range_CaOH is not None:
            O = self.OH[:, 0]
            condition1 = self.data[O][:, 3] < range_CaOH[1]
            condition2 = self.data[O][:, 3] > range_CaOH[0]
            condition = condition1 & condition2
            if condition.any():
                not_in_caoh = self.OH[~condition]
                in_caoh = self.OH[condition]
                setattr(self, "OH", not_in_caoh)
                setattr(self, "CaOH", in_caoh)

                
def cal_Qn(SiO4: np.ndarray, verbose=False, ratio=False):
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
    QnDic = {}
    if verbose:
        length = Qn.shape[0]
        for i in range(5):
            count = np.sum(Qn == i)
            if count != 0 and verbose:
                val = count/length
                QnDic[i] = val if ratio else count
                print(f"[Structure] Q{str(i)}: {val:.3f}")
    return QnDic



class UnitCell():       
    csh_category = ["bridgeCa000",
                    "bridgeCa00",
                    "bridgeCa01",
                    "bridgeCa02",
                    "crystal00",
                    "crystal01",
                    "tobermorite"]
    bulk_category = ["w50", "w40"]

    @classmethod
    def _set_config_(cls, unit_name):
        if unit_name is None:
            cwd = os.getcwd()
        else:
            cwd = unit_name
        cwd = os.path.abspath(cwd)
        unit_list = []
        category = [cls.csh_category, cls.bulk_category]
        default_category = ["model", "w50"]
        for categories, default in zip(category, default_category):
            unit_ = default
            for category in categories:
                if category in cwd:
                    unit_ = category
            unit_list.append(unit_)
        cls.unit_name = f"kim_{unit_list[0]}.{unit_list[1]}"
        if unit_list[1] == "w40":
            cls.config = "w40" if unit_list[0] != "bridgeCa000" else "w400"
        else:
            cls.config = "config00" if unit_list[0] != "bridgeCa000" else "config000"
        print(f"[INFO] Set {cls.unit_name} as a unit cell")
        print(f"[INFO] Set {cls.config} as a config")
    
    def __init__(self, unit_name=None, Si_key_upper_lower=False):
        UnitCell._set_config_(unit_name)
        unit = resources.files("cshmd.data.unitcell") / f"{UnitCell.unit_name}.cif"
        o = resources.files("cshmd.data.unitcell") / f"{UnitCell.config}.yaml"
        self.set_data(cif2data(unit))
        self.rangeDic = yaml.safe_load(o.read_text())
        self.label_keys = self.rangeDic.keys()
        self.indexDic = {key: [] for key in self.rangeDic.keys()}
        self.Si_key_upper_lower = Si_key_upper_lower
        if Si_key_upper_lower:
            self.where = "upper"
            for key in ("BT_Si", "PT_Si"):
                if key in self.label_keys:
                    self.indexDic[key] = {"upper": [], "lower": []}
        self.get_Si_index()
        self.get_O_index()
        self.get_Ca_index()
        
    def __getitem__(self, key):
        return self.indexDic[key]

    def set_data(self, cif_data):
        lattice, angle = cif_data[0:2]
        self.data = df2xyz(cif_data[-1])
        self.matrix, _ = setMatrix(lattice, angle)
        self.bond = BOND(self.data, self.matrix)
        self.bond.find("SiO4")
        self.O_of_SiO4 = np.unique(self.bond.SiO4[:, 1:])
        
    def get_Si_index(self):
        mask = self.data[:, 0] == "Si"
        data_ = self.data[mask]
        index_Si = np.nonzero(mask)[0]
        for key in ("BT_Si", "PT_Si"):
            if self.rangeDic[key] is None:
                continue
            range_ = self.rangeDic[key]
            for min_, max_ in range_:
                min_mask = data_[:, 3] > min_
                max_mask = data_[:, 3] < max_
                sum_mask = min_mask & max_mask
                index = index_Si[sum_mask]
                if self.Si_key_upper_lower:
                    self.indexDic[key][self.where].extend(index)
                    self.where = "lower" if self.where == "upper" else "upper"
                else:
                    self.indexDic[key].extend(index)
                
    def get_O_index(self):
        mask_SiO4 = np.zeros((self.data.shape[0],), dtype=bool)
        mask_SiO4[self.O_of_SiO4] = True
        mask = (self.data[:, 0] == "O") & mask_SiO4
        data_ = self.data[mask]
        index_O_ = np.nonzero(mask)[0]
        index_O = index_O_[np.isin(index_O_, self.O_of_SiO4)]
        keys = [key for key in  self.label_keys if "O" in key]
        for key in keys:
            if self.rangeDic[key] is None:
                continue
            for min_, max_ in self.rangeDic[key]:
                min_mask = data_[:, 3] > min_
                max_mask = data_[:, 3] < max_
                sum_mask = min_mask & max_mask
                index = index_O[sum_mask]
                self.indexDic[key].extend(index)
                
    def get_Ca_index(self):
        mask = self.data[:, 0] == "Ca"
        data_ = self.data[mask]
        index_Ca = np.nonzero(mask)[0]
        all_mask = np.zeros((data_.shape[0], ), dtype=bool)
        for key in ["layer_Ca"]:
            if self.rangeDic[key] is None:
                continue
            for min_, max_ in self.rangeDic[key]:
                min_mask = data_[:, 3] > min_
                max_mask = data_[:, 3] < max_
                sum_mask = min_mask & max_mask
                all_mask = all_mask | sum_mask
                index = index_Ca[sum_mask]
                self.indexDic[key].extend(index)
        bulk_index = index_Ca[~all_mask]
        self.indexDic["bulk_Ca"].extend(bulk_index)        
        
def cal_CS(data: np.ndarray, layer=None, verbose=False):
    """
    cal Ca/Si from data
    data -> np.ndarray shape(-1, 4) [[elem, x, y, z], []...]
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
            print(f"[Structure] C/S: {CS:.3f}")
    return CS
        
def cal_MCL(Q1: int, Q2: int) -> float:
    if Q1 == 0:
        val = float("inf")
    else:
        val = 2 * (1 + Q2 / Q1)
    print(f"[Structure] MCL: {val:.3f}")
    return val


if __name__ == "__main__":
    pos = [[0, 0, 1], [-0.1, -0.1, 1.2], [0.98, 0.98, 0.02]]
    print(unwrap_mole(pos))
