import numpy as np
import pandas as pd
from pathlib import Path
import os
import yaml
from importlib import resources
import itertools
import inspect
import networkx as nx
from collections import defaultdict, Counter
import random
from dataclasses import dataclass
from functools import reduce, wraps
from .common import checkTime, get_logger
from .functions import cal_distance, cal_distance_with_block, setMatrix
from .common import list2arr, df2xyz
from .util.molecule import molCharge
from .load import cif2data


logger = get_logger(__name__)


def stdout_search_level(func):
    @wraps(func)
    def search_level(*args, **kwargs):
        msg = f"- Search level: {func.__name__.split('_')[1]}"
        if len(args) >= 2:
            msg += f" -> {''.join(str(a) for a in args[1:])}"
        if len(kwargs) >= 1:
            msg += f", {''.join(f'{str(k)} - {str(v)}' for k, v in kwargs.items())}"
        return func(*args, **kwargs)

    return search_level


def stdout_analysis_level(func):
    @wraps(func)
    def analysis_level(*args, **kwargs):
        msg = f"- Analysis level: {func.__name__.split('_')[1]}"
        if len(args) >= 2:
            msg += f" -> {''.join(str(a) for a in args[1:])}"
        if len(kwargs) >= 1:
            msg += f", {''.join(f'{str(k)} - {str(v)}' for k, v in kwargs.items())}"
        logger.info(msg)
        return func(*args, **kwargs)

    return analysis_level


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
        where = charge_df["type"] == w
        if not np.any(where):
            continue
        mask = df.loc[:, ["label"]] == w
        num = np.sum(mask.to_numpy())
        c = charge_df.loc[where, "charge"].iloc[0]
        charge_total += num * c
    if return_charge:
        return charge_total, charge_df
    else:
        return charge_total
@dataclass    
class CarbonStatus:
    
    



class Bond:
    """
    input -> np.ndarray shape(-1, 4) [[elem, x, y, z], []...]
    Return Bond or Molecule index by call the instance
    """

    all_bond = ("Si_O", "O_H", "H_O", "Ca_O", "C_O", "BO", "NBO")
    mole_basic_mole = ("H2O", "SiO4", "OH", "CO3", "CO2", "H3O")
    mole_search_basic = ("H2O", "SiO4", "OH", "CO3", "H3O", "Ca_O")
    order = {"Si": 0, "Ca": 1, "C": 2, "O": 3, "H": 4}
    rcut = {
        ("O", "H"): 1.2,
        ("C", "O"): 1.6,
        ("Si", "O"): 2.3,
        ("Ca", "O"): 3.0,
        ("Si", "C"): 5.3,
    }

    def __getattribute__(self, val):
        attr = object.__getattribute__(self, val)
        # if val == "OH":
        #     print()
        #     print(attr.size)
        #     print()
        return attr

    @classmethod
    def _get_find_methods(cls):
        return [
            name
            for name, obj in inspect.getmembers(cls, inspect.isfunction)
            if name.startswith("find")
        ]

    def __init__(self, data=None, matrix=None, cartesian=False, **kwargs):
        self.data = data if data is not None else self.data
        self.matrix = matrix if matrix is not None else self.matrix
        self.cartesian = cartesian if cartesian is not None else self.cartesian
        self.elem = np.unique(self.data[:, 0])
        self.elem_index = {}
        self.dist_param = {}
        for e in self.elem:
            self.elem_index[e] = np.nonzero(self.data[:, 0] == e)[0]
        logger.debug("[StructureManager] Now new data setted for get Bonds")
        self.angle_pair = {}
        self.__del_bond__(*self.__class__.all_bond)
        self.__initialize_mole__(*self.__class__.mole_basic_mole)

    def __call__(self, keyword):
        val = getattr(self, keyword)
        return val

    def __del_bond__(self, *args):
        _args = " ".join(str(a) for a in args)
        logger.debug(f"del -> {_args}")
        for arg in args:
            setattr(self, arg, [])
            delattr(self, arg)

    def __initialize_mole__(self, *args):
        _args = " ".join(str(a) for a in args)
        logger.debug(f"initalize -> {_args}")
        for arg in args:
            setattr(self, arg, np.array([], dtype=int))

    @staticmethod
    def get_mole(arr, elem1="O", elem2="H", nbond=2):
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
        logger.debug(f"get_mole {elem1} - {elem2} -> nbonds: {nbond}")
        if arr.size == 0:
            return np.array([], dtype=int)
        arr = list2arr(arr)
        idx, count = np.unique(arr[:, 0], return_counts=True)
        center = idx[count == nbond]
        center_list, boundary_list = list(), list()
        for c in center:
            mask = c == arr[:, 0]
            boundary = arr[mask, 1]
            center_list.append(c)
            boundary_list.append(boundary)
        if len(center_list) != 0:
            c_arr = np.vstack(center_list)
            b_arr = np.vstack(boundary_list)
            mole = np.hstack([c_arr, b_arr])
        else:
            mole = np.array([], dtype=int)
        return mole

    def find_bond(
        self,
        elem1="O",
        elem2="H",
        rcut=1.2,
        use_block=True,
        reverse=False,
        reverse_rcut=None,
    ):
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
        logger.debug(f"find_bond {elem1} - {elem2} rcut -> {rcut}")
        if reverse and (reverse_rcut is None):
            msg = "[Error]: Set Both 'reverse' and  'reverse_rcut'"
            raise ValueError(msg)
        if not np.all([e in self.elem for e in (elem1, elem2)]):
            return np.array([], dtype=int)
        if (elem1, elem2) not in self.dist_param.keys():
            self.get_distance_table_by_elem(
                elem1=elem1, elem2=elem2, use_block=use_block
            )
        r = self.dist_param[(elem1, elem2)]
        which_e1 = self.elem_index[elem1]
        which_e2 = self.elem_index[elem2]
        indexs = np.where(r < rcut)
        bond = np.vstack([which_e1[indexs[0]], which_e2[indexs[1]]]).T
        if reverse:
            r_ = r.T
            indexs_ = np.where((r_ < reverse_rcut) & (r_ > rcut))
            bond_ = np.vstack([which_e2[indexs_[0]], which_e1[indexs_[1]]]).T
            return bond, bond_
        else:
            return bond

    @checkTime
    def get_distance_table_by_elem(self, elem1="O", elem2="H", use_block=True):
        data = list2arr(self.data)
        matrix = list2arr(self.matrix)
        mask1 = data[:, 0] == elem1
        mask2 = data[:, 0] == elem2
        pos1 = data[mask1, 1:]
        pos2 = data[mask2, 1:]
        func = cal_distance_with_block if use_block else cal_distance
        dist = func(pos1, pos2, matrix, cartesian=self.cartesian)
        self.dist_param[(elem1, elem2)] = dist

    @checkTime
    def get_distance_table_by_index(self, index1=None, index2=None, use_block=True):
        if index1 is None or index2 is None:
            raise ValueError("No index inputted")
        data = list2arr(self.data)
        matrix = list2arr(self.matrix)
        pos1 = self.data[index1, 1:]
        pos2 = self.data[index2, 1:]
        func = cal_distance_with_block if use_block else cal_distance
        dist = func(pos1, pos2, matrix, cartesian=self.cartesian)
        return dist

    def find(self, *args, **rcut):
        for arg in args:
            if "_" in arg:
                rcut_ = rcut.get(arg, None)
                getattr(self, f"find_{arg}")(rcut_)
            else:
                getattr(self, f"find_{arg}")()

    def find_Si_O(self, rcut=None):
        rcut = Bond.rcut[("Si", "O")] if rcut is None else rcut
        self.Si_O = self.find_bond("Si", "O", rcut=rcut)

    def find_Si_C(self, rcut=None, return_only=True):
        rcut = Bond.rcut[("Si", "C")] if rcut is None else rcut
        bond = self.find_bond("Si", "C", rcut=rcut)
        if return_only:
            return bond
        self.Si_C = bond

    def find_SiO4(self):
        if not hasattr(self, "Si_O"):
            getattr(self, "find_Si_O")()
        self.SiO4 = Bond.get_mole(self.Si_O, "Si", "O", nbond=4)

    def find_O_H(self, reverse=False, reverse_rcut=3.5, rcut=None):
        rcut = Bond.rcut[("O", "H")] if rcut is None else rcut
        returned = self.find_bond(
            "O", "H", reverse=reverse, reverse_rcut=reverse_rcut, rcut=rcut
        )
        if not reverse:
            self.O_H = returned
        else:
            self.O_H, self.H_O = returned

    def find_OH(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.OH = Bond.get_mole(self.O_H, "O", "H", nbond=1)

    def find_H2O(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.H2O = Bond.get_mole(self.O_H, "O", "H", nbond=2)

    def find_H3O(self):
        if not hasattr(self, "O_H"):
            getattr(self, "find_O_H")()
        self.H3O = Bond.get_mole(self.O_H, "O", "H", nbond=3)

    def find_C_O(self, rcut=None, return_only=False):
        rcut = Bond.rcut[("C", "O")] if rcut is None else rcut
        bond = self.find_bond("C", "O", rcut=rcut)
        if return_only:
            return bond
        self.C_O = bond

    def find_CO2(self):
        if not hasattr(self, "C_O"):
            getattr(self, "find_C_O")()
        self.CO2 = Bond.get_mole(self.C_O, "C", "O", nbond=2)

    def find_CO3(self):
        if not hasattr(self, "C_O"):
            getattr(self, "find_C_O")()
        self.CO3 = Bond.get_mole(self.C_O, "C", "O", nbond=3)

    def find_Ca_O(self, rcut=None, return_only=False):
        rcut = Bond.rcut[("Ca", "O")] if rcut is None else rcut
        bond = self.find_bond("Ca", "O", rcut=rcut)
        if return_only:
            return bond
        self.Ca_O = bond

    def find_BO(self):
        unique = np.unique(self.SiO4[:, 1:], return_counts=True)
        self.BO = np.vstack(unique).T

    def find_NBO(self, return_only=False):
        getattr(self, "find_BO")()
        mask = self.BO[:, 1] == 1
        NBO = self.BO[mask][:, 0]
        if not return_only:
            self.NBO = NBO
        return NBO

    def find_angle_pair(self, elem1, elem2, elem3, return_only=False, **kwargs):
        flag = kwargs.get("flag", None)
        index = kwargs.get("index", None)
        bond_index_df_list = list()
        same_elem = True if elem1 == elem3 else False
        prev_key = None
        key_list = []
        result_list = list()
        for e in (elem1, elem3):
            key = tuple(sorted([e, elem2], key=lambda o: self.__class__.order[o]))
            key_list.append(key)
            if key == prev_key:
                break
            prev_key = key
            bond_index_df = pd.DataFrame()
            rcut = self.__class__.rcut[key]
            bond_attr_name = f"{key[0]}_{key[1]}"
            bond_func_name = f"find_{bond_attr_name}"
            if not hasattr(self, bond_attr_name):
                getattr(self, bond_func_name)(rcut)
            indexs = getattr(self, bond_attr_name)
            for i in range(2):
                if key[i] not in bond_index_df.columns:
                    bond_index_df[key[i]] = indexs[:, i]
            if index is not None:
                mask1 = np.isin(bond_index_df[key[0]], index[key[0]])
                mask2 = np.isin(bond_index_df[key[1]], index[key[1]])
                bond_index_df = bond_index_df[mask1 & mask2]
            bond_index_df_list.append(bond_index_df)
        if same_elem:
            dist = self.dist_param[(key)]
            all_idx_e1 = self.elem_index[elem1]
            all_idx_e2 = self.elem_index[elem2]
            df = bond_index_df_list[0]
            unique_idx = np.unique(df[elem2])
            for idx in unique_idx:
                mask = df[elem2] == idx
                matched = df.loc[mask, elem1]
                n_pair = len(matched)
                if n_pair > 1:
                    for m in itertools.combinations(matched, 2):
                        _pair = list(m)
                        _pair.insert(1, idx)
                        mask1 = all_idx_e2 == idx
                        mask2 = np.isin(all_idx_e1, m)
                        _dist = dist if key_list[0][0] == elem2 else dist.T
                        d1, d2 = _dist[mask1, mask2]
                        cart_e1, cart_e3 = (
                            self.data[m, 1:4]
                            if not self.cartesian
                            else self.data[m, 1:4] @ np.linalg.inv(self.matrix)
                        )
                        cart_e2 = (
                            self.data[idx, 1:4]
                            if not self.cartesian
                            else self.data[idx, 1:4] @ np.linalg.inv(self.matrix)
                        )
                        vec1_ = (cart_e1 - cart_e2).astype(float)
                        vec1 = (vec1_ - np.round(vec1_)) @ self.matrix
                        vec2_ = (cart_e3 - cart_e2).astype(float)
                        vec2 = (vec2_ - np.round(vec2_)) @ self.matrix
                        angle = np.degrees(np.arccos(np.dot(vec1, vec2) / (d1 * d2)))
                        _pair.append(angle)
                        result_list.append(_pair)
            total_angle_pair = pd.DataFrame(
                result_list, columns=[elem1, elem2, elem3, "angle"]
            )
        else:
            total_angle_pair = bond_index_df_list[0].merge(
                bond_index_df_list[1], on="O", how="inner"
            )[[elem1, elem2, elem3]]
            angle_pair = []
            for i in range(len(total_angle_pair)):
                idx_e1, idx_e2, idx_e3 = total_angle_pair.iloc[i, :]
                vec_list = []
                dist_multy = 1
                for i in range(2):
                    idx_other = idx_e1 if i == 0 else idx_e3
                    elem_other = elem1 if i == 0 else elem3
                    if elem2 == key_list[i][0]:
                        dist = self.dist_param[key_list[i]]
                    else:
                        dist = self.dist_param[key_list[i]].T
                    match_e2 = idx_e2 == self.elem_index[elem2]
                    match_other = idx_other == self.elem_index[elem_other]
                    dist_match = float(dist[match_e2, match_other][0])
                    dist_multy *= dist_match
                    cart_other = (
                        self.data[idx_other, 1:4]
                        if not self.cartesian
                        else self.data[m, 1:4] @ self.matrix
                    )
                    cart_elem2 = (
                        self.data[idx_e2, 1:4]
                        if not self.cartesian
                        else self.data[m, 1:4] @ self.matrix
                    )
                    vec = (cart_other - cart_elem2).astype(float)
                    vec = (vec - np.round(vec)) @ self.matrix
                    vec_list.append(vec)
                angle_pair.append(np.degrees(np.arccos(np.dot(*vec_list) / dist_multy)))
            total_angle_pair["angle"] = angle_pair
        if return_only:
            return total_angle_pair
        else:
            self.angle_pair[(elem1, elem2, elem3)] = total_angle_pair


class CshParam(Bond):
    mole_detail_search_oh = ("SiOH", "CaOH", "HCO3", "H2CO3", "H3CO3")
    group_co3 = ("HCO3", "H2CO3", "H3CO3")
    species_co3 = ("CO2", "CO3", "HCO3", "H2CO3", "H3CO3")
    result = dict()
    result["orientational_order_parameter"] = list()
    result["density_profile"] = defaultdict(list)
    # result["Ca_adsorption"] = defaultdict(int)
    # result["CO3_adsorption"] = defaultdict(int)
    result["Ca_adsorption_by_time"] = defaultdict(list)
    result["CO3_adsorption_by_time"] = defaultdict(list)
    result["cluster"] = list()
    flag_layer_searched = False

    @classmethod
    def initialize_result(cls):
        logger.info("Initialize all result parameters.")
        cls.result = dict()
        cls.result["orientational_order_parameter"] = list()
        cls.result["density_profile"] = defaultdict(list)
        # cls.result["Ca_adsorption"] = defaultdict(int)
        # cls.result["CO3_adsorption"] = defaultdict(int)
        cls.result["Ca_adsorption_by_time"] = defaultdict(list)
        cls.result["CO3_adsorption_by_time"] = defaultdict(list)
        cls.result["cluster"] = list()
        cls.flag_layer_searched = False

    def __init__(self, data=None, matrix=None, cartesian=False, **kwargs):
        self.search_layer_once = (
            True if kwargs.pop("search_layer_once", None) else False
        )
        super().__init__(data=data, matrix=matrix, cartesian=cartesian, **kwargs)
        self.__initialize_mole__(*self.__class__.mole_detail_search_oh)
        self.is_csh = True if "Si" in self.elem_index.keys() else False
        if self.is_csh:
            self.Si_range = []
            self.Si_range.append(np.min(self.data[self.elem_index["Si"], 3]))
            self.Si_range.append(np.max(self.data[self.elem_index["Si"], 3]))

    @stdout_search_level
    def search_basic(self, **kwargs):
        self._search_basic(**kwargs)

    @stdout_search_level
    def search_chain(self, **kwargs):
        self._search_basic(**kwargs)
        self._detail_search_chain()

    @stdout_search_level
    def search_full(self, **kwargs):
        self._search_basic(**kwargs)
        self._detail_search_chain()
        self._detail_search_ca_adsorption()
        self._detail_search_oh()
        # print(np.intersect1d(self.surface_BT_O, self.H2O[:, 1:].ravel()), "sss")

    def _search_basic(self, **rcut_dict):
        self.find(*self.__class__.mole_search_basic, **rcut_dict)
        logger.debug(
            f"Basic search result -> H2O: {self.H2O.shape[0]}, "
            f"SiO4: {self.SiO4.shape[0]}, "
            f"OH: {self.OH.shape[0]}, "
            f"CO3: {self.CO3.shape[0]}, "
            f"H3O: {self.H3O.shape[0]}, "
            f"Ca_O: {self.Ca_O.shape[0]}"
        )

    def _detail_search_chain(self):
        if not self.search_layer_once or not CshParam.flag_layer_searched:
            logger.debug("Search layer details")
            self._detail_search_layer()
            self.setattr_chain()
            if self.search_layer_once:
                CshParam.flag_layer_searched = True
            # self.find_Ca_O(rcut=2.6)
        else:
            # self.search_basic(**{"Ca_O": 2.6})
            self.set_layer_boundary()
            logger.debug("Skip searching layer details")

    def set_layer_boundary(self):
        if not self.is_csh:
            logger.warning("Doesn't have C-S-H layer. RETURN")
            return
        self.layer_center_z = np.mean(self.data[self.elem_index["Si"], 3])
        upper_intra = self.data[self.intra_layer_ca, 3] > self.layer_center_z
        lower_intra = self.data[self.intra_layer_ca, 3] < self.layer_center_z
        layer_upper_cao = np.mean(self.data[self.intra_layer_ca[upper_intra], 3])
        layer_lower_cao = np.mean(self.data[self.intra_layer_ca[lower_intra], 3])
        self.cao_range = [layer_lower_cao, layer_upper_cao]
        logger.debug(f"Set layer boundary -> {self.cao_range[0]} - {self.cao_range[1]}")

    def _detail_search_layer(self):
        if not self.is_csh:
            logger.warning("Doesn't have C-S-H layer. RETURN")
            return
        self.find_angle_pair("Ca", "O", "Si")
        df_Ca_O_Si = self.angle_pair[("Ca", "O", "Si")]
        unique_si, counts_si = np.unique(df_Ca_O_Si["Si"], return_counts=True)
        mask_PT = np.where(counts_si >= 4, True, False)
        self.PT_si = unique_si[mask_PT]
        isin_BT = np.isin(self.elem_index["Si"], self.PT_si)
        self.BT_si = self.elem_index["Si"][~isin_BT]

        unique_ca, counts_ca = np.unique(df_Ca_O_Si["Ca"], return_counts=True)
        mask_intra_layer_ca = np.where(counts_ca >= 6, True, False)
        self.intra_layer_ca = unique_ca[mask_intra_layer_ca]
        self.set_layer_boundary()
        self.cao_range = [
            np.min(self.data[self.intra_layer_ca, 3]),
            np.max(self.data[self.intra_layer_ca, 3]),
        ]
        mask_layer_ca = self._get_csh_mask(self.elem_index["Ca"])
        layer_ca = self.elem_index["Ca"][mask_layer_ca]

        # unique_Caのうち、intra_layer_caでないものと、elem_index["Ca"]のうち、unique_caでないものを合わせる
        # unique_CaがすべてのCaをカバーしているわけではないため、unique_Caのうち、intra_layer_caでないものと、elem_index["Ca"]のうち、unique_caでないものを合わせる必要がある
        # 後ほど、コードの修正が必要になるかもしれない
        not_intra_layer_ca = np.hstack(
            [
                unique_ca[~mask_intra_layer_ca],
                self.elem_index["Ca"][~np.isin(self.elem_index["Ca"], unique_ca)],
            ]
        )

        self.inter_layer_ca = layer_ca[np.isin(layer_ca, not_intra_layer_ca)]
        not_layer_ca = self.elem_index["Ca"][~mask_layer_ca]
        self.bulk_ca = not_layer_ca[~np.isin(not_layer_ca, self.intra_layer_ca)]
        layer_threshold = np.mean(self.data[self.inter_layer_ca, 3])
        mask_intra_upper = self.data[self.intra_layer_ca, 3] > layer_threshold
        intra_layer_ca_upper = self.intra_layer_ca[mask_intra_upper]
        intra_layer_ca_lower = self.intra_layer_ca[~mask_intra_upper]
        upper_intra_threshold = np.mean(self.data[intra_layer_ca_upper, 3])
        lower_intra_threshold = np.mean(self.data[intra_layer_ca_lower, 3])
        _mask_surface_BT1 = self.data[self.BT_si, 3] > upper_intra_threshold
        _mask_surface_BT2 = self.data[self.BT_si, 3] < lower_intra_threshold
        self.surface_BT_si = self.BT_si[_mask_surface_BT1 | _mask_surface_BT2]
        self.bulk_BT_si = self.BT_si[~_mask_surface_BT1 & ~_mask_surface_BT2]
        _mask_surface_PT1 = self.data[self.PT_si, 3] > upper_intra_threshold
        _mask_surface_PT2 = self.data[self.PT_si, 3] < lower_intra_threshold
        self.surface_PT_si = self.PT_si[_mask_surface_PT1 | _mask_surface_PT2]
        self.bulk_PT_si = self.PT_si[~_mask_surface_PT1 & ~_mask_surface_PT2]
        assert (
            self.bulk_ca.shape[0]
            + self.inter_layer_ca.shape[0]
            + self.intra_layer_ca.shape[0]
            == self.elem_index["Ca"].shape[0]
        ), "Mismatch in Ca atom counts"
        _surface_BT_O = np.unique(
            self.SiO4[np.isin(self.SiO4[:, 0], self.surface_BT_si)][:, 1:]
        )
        self.surface_PT_O, s_PT_count = np.unique(
            self.SiO4[np.isin(self.SiO4[:, 0], self.surface_PT_si)][:, 1:],
            return_counts=True,
        )
        self.surface_PT_BO = self.surface_PT_O[np.where(s_PT_count == 2)]
        self.surface_BT_O = _surface_BT_O[~np.isin(_surface_BT_O, self.surface_PT_O)]
        _bulk_BT_O = np.unique(
            self.SiO4[np.isin(self.SiO4[:, 0], self.bulk_BT_si)][:, 1:]
        )
        self.bulk_PT_O, b_PT_count = np.unique(
            self.SiO4[np.isin(self.SiO4[:, 0], self.bulk_PT_si)][:, 1:],
            return_counts=True,
        )
        self.bulk_PT_BO = self.bulk_PT_O[np.where(b_PT_count == 2)]
        self.bulk_BT_O = _bulk_BT_O[~np.isin(_bulk_BT_O, self.bulk_PT_O)]

    def setattr_chain(self):
        if not self.is_csh:
            logger.debug("Skip finding silicate chain. No C-S-H system.")
            return
        # 要修正、z軸が高いchainが低いindex順にくるようにしているが、必ずしもそうなっているとは限らない
        # また、コードの可読性もあまり良くないため、修正が必要になるかもしれない
        center_z = np.average(self.cao_range)
        self.tetra = defaultdict(lambda: defaultdict(list))
        for chain, region, arr in (
            ("BT", "surface", self.surface_BT_si),
            ("BT", "bulk", self.bulk_BT_si),
            ("PT", "surface", self.surface_PT_si),
            ("PT", "bulk", self.bulk_PT_si),
        ):
            upper = arr[self.data[arr, 3] > center_z]
            lower = arr[self.data[arr, 3] < center_z]
            if upper.size:
                self.tetra[chain][region].append(upper)

            if lower.size:
                self.tetra[chain][region].append(lower)

    def _detail_search_oh(self):
        if self.OH.size == 0:
            logger.warning("No OH in basic search")
            return
        self.OH_all = self.OH.copy()
        self.__detail_search_oh_co3()
        self.__detail_search_oh_sioh()
        self.__detail_search_oh_caoh()

    def __detail_search_oh_co3(self):
        if self.CO3.size == 0:
            logger.debug("No CO3 in basic search")
            return
        else:
            logger.debug("Searching 'OH' in 'CO3'")
            val_m = getattr(self, "CO3")
            molecular_O = val_m[:, 1:].ravel()
            isin = np.isin(self.OH[:, 0], molecular_O)
            in_mole = self.OH[isin]
            # self.OHの初期化
            self.OH = self.OH[~isin]
            if in_mole.size != 0:
                for co3s in self.__class__.group_co3:
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
                    co3s_name = f"H{str(len_H) if len_H != 1 else ''}CO3"
                    species_arr = np.hstack([val_m[co3_index], H_index_list])
                    getattr(self, co3s_name).append(species_arr)
                for co3s in self.__class__.group_co3:
                    val = getattr(self, co3s)
                    if len(val) != 0:
                        new_val = np.vstack(val)
                        setattr(self, co3s, new_val)
                    else:
                        setattr(self, co3s, np.array([], dtype=int))
                setattr(self, "CO3", val_m[is_co3])

    def __detail_search_oh_sioh(self):
        if self.SiO4.size == 0:
            logger.debug("No SiO4 in basic search")
            return
        else:
            logger.debug("Searching 'OH' in 'SiO4")
            val_m = getattr(self, "SiO4")
            molecular_O = val_m[:, 1:].ravel()
            isin = np.isin(self.OH[:, 0], molecular_O)
            self.SiOH = self.OH[isin]
            # self.OHの初期化
            self.OH = self.OH[~isin]
            self.surface_BT_oh = self.SiOH[np.isin(self.SiOH[:, 0], self.surface_BT_O)][
                :, 1
            ]
            self.surface_PT_oh = self.SiOH[np.isin(self.SiOH[:, 0], self.surface_PT_O)][
                :, 1
            ]
            self.bulk_BT_oh = self.SiOH[np.isin(self.SiOH[:, 0], self.bulk_BT_O)][:, 1]
            self.bulk_PT_oh = self.SiOH[np.isin(self.SiOH[:, 0], self.bulk_PT_O)][:, 1]

    def __detail_search_oh_caoh(self):
        if self.elem_index["Ca"].size == 0:
            logger.debug("No 'Ca' in system. RETURN")
            return
        logger.debug("Searching 'OH' in 'CaOH")
        mask_layer = self._get_csh_mask(self.OH[:, 0])
        mask_CaOH = np.isin(self.OH[:, 0], self.Ca_O[:, 1])
        # self.ca_stat["inner"]
        ca_adsorbed_list = []
        for stat in ("pair", "bridge", "defect"):
            ca_adsorbed_list.extend(
                [
                    idx
                    for idx, vdic in self.ca_status.items()
                    if vdic.get("adsorbed") == stat
                ]
            )
        # ca_adsorbed = np.array()
        index_adsorbed_O = self.Ca_O[np.isin(self.Ca_O[:, 0], ca_adsorbed_list)][:, 1]
        mask_adsorbed = np.isin(self.OH[:, 0], index_adsorbed_O)
        self.bulk_oh = self.OH[~mask_CaOH & mask_layer]
        self.surface_caoh = self.OH[mask_CaOH & ~mask_layer & mask_adsorbed]
        self.free_caoh = self.OH[mask_CaOH & ~mask_layer & ~mask_adsorbed]
        self.bulk_caoh = self.OH[mask_CaOH & mask_layer]
        self.OH = self.OH[~mask_CaOH & ~mask_layer]

    @checkTime
    def _detail_search_ca_adsorption(self):
        self.ca_status = defaultdict(lambda: {"CN": 0, "adsorption": "non-adsorbed"})
        if self.bulk_ca.size == 0:
            logger.debug("No 'Ca' in solution")
            return
        index_ca = self.bulk_ca
        index_o = self.Si_O[:, 1]
        bond_inner = self.find_Ca_O(rcut=3.1, return_only=True)
        bonded_ca = set(np.unique(bond_inner[:, 0]))
        bond_outer = self.find_Ca_O(rcut=5.2, return_only=True)
        outer_ca = set(bond_outer[np.isin(bond_outer[:, 1], index_o)][:, 0])
        ca_to_o = defaultdict(list)
        for ca, o in bond_inner:
            ca_to_o[ca].append(o)
        for idx_ca in index_ca:
            if idx_ca not in bonded_ca:
                continue
            adsorbed_O = np.array(ca_to_o[idx_ca])
            self.ca_status[idx_ca]["CN"] = adsorbed_O.size
            stat = self._get_ca_adsorbed_stat(adsorbed_O)
            if stat != "not_inner":
                self.ca_status[idx_ca]["adsorption"] = stat
            elif idx_ca in outer_ca:
                self.ca_status[idx_ca]["adsorption"] = "outer"

    

    @stdout_analysis_level
    def analyze_dynamics(self, **kwargs):
        timeseries = kwargs.get("timeseries", True)
        self._anal_ca_adsorption_site(timeseries=timeseries)
        self._anal_co3_adsorption_site(timeseries=timeseries)
        self._anal_caco3_cluster()
        self._anal_density_profile("Ca", "C", "Si", "H2O")

    @stdout_analysis_level
    def analyze_final_structure(self):
        self._anal_orientational_order_parameter()

    @stdout_analysis_level
    def analyze_full(self, **kwargs):
        timeseries = kwargs.get("timeseries", True)
        self._anal_ca_adsorption_site(timeseries=timeseries)
        self._anal_co3_adsorption_site(timeseries=timeseries)
        self._anal_caco3_cluster()
        self._anal_orientational_order_parameter()
        self._anal_density_profile("Ca", "C", "Si", "H2O")

    @stdout_analysis_level
    def analyze_choice(self, *params):
        for param in params:
            match param:
                case "density":
                    self._anal_density_profile("Ca", "C", "Si", "H2O")
                    
    def _anal_ca_adsorption_site(self, timeseries=True):
        adsorption_types = (
            "pair",
            "bridge",
            "defect",
            "outer",
            "non-adsorbed",
        )
        counter = Counter(dic["adsorption"] for dic in self.ca_status.values())
        if timeseries:
            for adsorption_type in adsorption_types:
                CshParam.result["Ca_adsorption_by_time"][adsorption_type].append(
                    counter[adsorption_type]
                    )                   
                    
                    
    def _anal_co3_adsorption_site(self, timeseries=True):
        self.co3_status = defaultdict(
            lambda: {
                "Si_adsorption": "non-adsorbed",
                "Si_adsorption_dimension": 0,
                "Ca_adsorption": "non-adsorbed",
                "Ca_adsorption_dimension": 0,
            }
        )
        if "C" not in self.elem_index:
            logger.debug("No 'C' in system")
            return
        index_O_in_SiO4 = np.hstack([self.surface_BT_O, self.surface_PT_O])
        index_H_in_SiO4 = np.hstack([self.surface_BT_oh, self.surface_PT_oh])
        species_O, species_H = [], []
        o_to_c, h_to_c = {}, {}
        for mole in self.species_co3:
            arr = getattr(self, mole)
            if not arr.size:
                continue
            match mole:
                case "CO2":
                    O = arr[:, 1:3]
                    C = arr[:, 0]
                case "CO3":
                    O = arr[:, 1:4]
                    C = arr[:, 0]
                case _:
                    O = arr[:, 1:4]
                    H = arr[:, 4:]
                    species_H.append(H.ravel())
                    C = arr[:, 0]
                    for c, hydro in zip(C, H):
                        for h in hydro:
                            h_to_c[h] = c
            species_O.append(O.ravel())
            for c, oxygens in zip(C, O):
                for o in oxygens:
                    o_to_c[o] = c
        species_O = (
            np.hstack(species_O) if len(species_O) != 0 else np.array([], dtype=int)
        )
        species_H = (
            np.hstack(species_H) if len(species_H) != 0 else np.array([], dtype=int)
        )
        bonded_C = []
        for pairs, map_to_c in (
            ([index_O_in_SiO4, species_H], False),
            ([species_O, index_H_in_SiO4], True),
        ):
            mask_O, mask_H = np.where(self.get_distance_table_by_index(*pairs) < 2.4)
            if map_to_c:
                bonded_C.append(np.unique([o_to_c[o] for o in pairs[0][mask_O]]))
            else:
                bonded_C.append([h_to_c[h] for h in pairs[1][mask_H]])
        bonded_C_surface_Si, dimensions_Si = np.unique(
            np.hstack(bonded_C), return_counts=True
        )
        for index, dimension in zip(bonded_C_surface_Si, dimensions_Si):
            self.co3_status[index]["Si_adsorption"] = "adsorbed"
            self.co3_status[index]["Si_adsorption_dimension"] = dimension
        bonded_C = {key: [] for key in ("pair", "bridge", "defect")}
        pair_Ca_Oc = self.Ca_O[np.isin(self.Ca_O[:, 1], species_O)]
        #ckpt
        for Ca, Oc in pair_Ca_Oc:
            match self.ca_status[Ca]["adsorption"]:
                case "pair":
                    bonded_C["pair-site Ca"].append(o_to_c[Oc])
                case "defect":
                    bonded_C["defect-site Ca"].append(o_to_c[Oc])
                case "bridge":
                    bonded_C["bridging-site Ca"].append(o_to_c[Oc])
        
        for key, ca_ad_list in bonded_C.items():
            arr, dimension = np.unique(
                ca_ad_list, return_counts=True
            )
            for index, dimension in zip(arr, dimension):
                self.co3_status[index]["Ca_adsorption"] = key
                self.co3_status[index]["Ca_adsorption_dimension"] = dimension                    

        if timeseries:
            stats = ("Si-Ca-adsorbed", "Ca-adsorbed", "Si-adsorbed", "non-adsorbed")
            countDic = {s: 0 for s in stats}
            for index in self.elem_index["C"]:
                si = self.co3_status[index]["Si_adsorption"] == "adsorbed"
                ca = self.co3_status[index]["Ca_adsorption"] == "adsorbed"
                match (si, ca):
                    case (True, True):
                        countDic["Si-Ca-adsorbed"] += 1
                    case (False, True):
                        countDic["Ca-adsorbed"] += 1
                    case (True, False):
                        countDic["Si-adsorbed"] += 1
                    case _:
                        countDic["non-adsorbed"] += 1
            for s in stats:
                CshParam.result["CO3_adsorption_by_time"][s].append(countDic[s])

    def _anal_orientational_order_parameter(self):
        if self.H2O.size == 0:
            logger.warning("No 'H2O' found in basic search")
            return
        else:
            logger.debug("Calculate orientational order parameter for 'H2O'")
            for indexs_O_H_H in self.H2O[self._get_bulk_mask(self.H2O[:, 0])]:
                fract = self.data[indexs_O_H_H, 1:].astype(float)
                vec_O_H = fract[1:, :] - fract[0, :]
                vec_O_H = 0.5 * (vec_O_H[0] + vec_O_H[1])
                vec_O_H = (vec_O_H - np.round(vec_O_H)) @ self.matrix
                vec_O_H /= np.linalg.norm(vec_O_H)
                cos = vec_O_H[2]
                CshParam.result["orientational_order_parameter"].append(
                    (0.5 * (3 * cos**2 - 1))
                )

    def _anal_density_profile(self, *moles):
        max_c_vec = self.matrix[2, 2]
        threshold = self.layer_center_z * self.matrix[2, 2]
        for mole in moles:
            match mole:
                case "Ca":
                    indexs = self.bulk_ca
                case "H2O":
                    indexs = self.H2O[self._get_bulk_mask(self.H2O[:, 0]), 0]
                case _:
                    try:
                        indexs = self.elem_index[mole]
                    except KeyError:
                        logger.warning(
                            f"No '{mole}' found in system. Skip density profile analysis for '{mole}'"
                        )
                        continue
            val = (self.data[indexs, 1:4] @ self.matrix)[:, 2]
            val = np.where(
                val < threshold, max_c_vec + (val - threshold), val - threshold
            )
            CshParam.result["density_profile"][mole].append(val)

    def _get_ca_adsorbed_stat(self, arr_adsorbed_O):
        if np.isin(self.surface_PT_BO, arr_adsorbed_O).any():
            return "pair"
        elif np.isin(self.surface_BT_O, arr_adsorbed_O).any():
            return "bridge"
        elif np.isin(self.Si_O[:, 1], arr_adsorbed_O).any():
            return "defect"
        else:
            return "not_inner"

    def evaluate_chain(self, save_chain_info=False):
        """
        Initialize self.tetra,
        calculate MCL
        """
        if not self.is_csh:
            logger.debug("No C-S-H")
            return
        isin_tetra = np.isin(self.elem_index["Si"], self.SiO4[:, 0])
        self.Qn = CALC.Qn(self.SiO4, verbose=True)
        if not isin_tetra.all():
            broken = self.elem_index["Si"][~isin_tetra]
            broken_str = " ".join(str(v) for v in broken)
            logger.warning(f"Some Silicate Chain broken: {broken_str}")
            return
        BTcount = 0
        PTcount = 0
        for key, value in self.tetra.items():
            for k, v in value.items():
                if len(v) != 1:
                    if key == "BT":
                        BTcount += reduce(lambda x, y: x.size + y.size, v)
                    elif key == "PT":
                        PTcount += reduce(lambda x, y: x.size + y.size, v)
                else:
                    if key == "BT":
                        BTcount += v[0].size
                    elif key == "PT":
                        PTcount += v[0].size
        if BTcount + PTcount != self.elem_index["Si"].shape[0]:
            logger.warning(
                f"Mismatch in Si atom counts for tetrahedral classification: {BTcount + PTcount} != {self.elem_index['Si'].shape[0]}"
            )
        logger.info(f"[Silicate Chain] BT: {BTcount}, PT: {PTcount}")
        Q = []
        for i in range(1, 5):
            Q.append(self.Qn.get(i, 0))
        sum_Q = reduce(lambda x, y: x + y, Q)
        self.Q_factor = Q[0] / sum_Q if sum_Q != 0 else 0
        self.chain = {"surface": [], "bulk": []}
        for where in ("surface", "bulk"):
            # Sth wrong when number of layer bigger than 2, but currently we have only 2 layers,
            # so it is not a problem. Need to check when more layers exist.
            index_max = max(len(self.tetra["BT"][where]), len(self.tetra["PT"][where]))
            # index_max = max(len(self.tetra["BT"]), len(self.tetra["PT"]))
            BT_list = self.tetra["BT"][where]
            PT_list = self.tetra["PT"][where]
            for index in range(index_max):
                all_tetra = list()
                if len(BT_list) > index:
                    all_tetra.append(self.tetra["BT"][where][index])
                if len(PT_list) > index:
                    all_tetra.append(self.tetra["PT"][where][index])
                self.chain[where].append(
                    Chain(
                        self.Si_O[
                            np.isin(self.Si_O[:, 0], np.hstack(all_tetra).ravel())
                        ],
                        deletable_Si=BT_list[index] if BT_list else None,
                    )
                )
        mcl_count = Counter(self.chain["surface"][0].sort_chain_length)
        logger.info(
            f"[Silicate Chain] CL : {' '.join(f'Length {k!r}: {v!r}' for k, v in mcl_count.items())}"
        )
        self.MCL = self.chain["surface"][0].mean_chain_length
        logger.info(f"[Silicate Chain] MCL : {self.MCL:.3f}")
        if save_chain_info:
            self.chain_info = {}
            self.chain_info["CL"] = mcl_count
            self.chain_info["BT"] = BTcount
            self.chain_info["PT"] = PTcount
        else:
            self.chain_info = None

    def __get_index_in_CO3species(self, elem):
        index = []
        for mole in self.species_co3:
            arr = getattr(self, mole)
            if not arr.size:
                continue
            match mole, elem:
                case "CO2", "O":
                    index.append(arr[:, 1:3].ravel())
                case _, "O":
                    index.append(arr[:, 1:4].ravel())
                case _, "H":
                    index.append(arr[:, 4:].ravel())
        return np.hstack(index) if index else np.array([], dtype=int)

    def _get_index_O_in_CO3species(self):
        return self.__get_index_in_CO3species("O")

    def _get_index_H_in_CO3species(self):
        return self.__get_index_in_CO3species("H")

    def _get_bulk_mask(self, index):
        mask1 = self.data[index, 3] > self.cao_range[1]
        mask2 = self.data[index, 3] < self.cao_range[0]
        return mask1 | mask2

    def _get_csh_mask(self, index):
        mask1 = self.data[index, 3] < self.cao_range[1]
        mask2 = self.data[index, 3] > self.cao_range[0]
        return mask1 & mask2

    def _get_Si_to_tetra_O(self, si_index):
        if not self.is_csh:
            logger.debug("Doesn't have C-S-H layer. Return")
            return
        if not hasattr(self, "SiO4"):
            logger.error(
                "SiO4 bond not found. Please run search_basic or find_SiO4 first."
            )
            raise AttributeError(
                "SiO4 bond not found. Please run search_basic or find_SiO4 first."
            )
        mask = self.SiO4[:, 0] == si_index
        return self.SiO4[mask][:, 1:]

    def _make_co3_species_map(self):
        species_map = {}
        for key in CshParam.species_co3:
            arr = getattr(self, key)
            if arr.size == 0:
                continue
            for c in arr[:, 0]:
                species_map[c] = key
        return species_map

    def _add_Ca_n_C_nodes(self, G, species_map):
        for ca in self.bulk_ca:
            G.add_node(
                ca,
                elem="Ca",
                species="Ca",
                adsorption=self.ca_status[ca]["adsorption"],
                CN=self.ca_status[ca]["CN"],
            )
        for c in self.elem_index["C"]:
            G.add_node(
                c,
                elem="C",
                species=species_map.get(c),
                Ca_adsorption=self.co3_status[c]["Ca_adsorption"],
                Ca_adsorption_dimension=self.co3_status[c]["Ca_adsorption_dimension"],
                Si_adsorption=self.co3_status[c]["Ca_adsorption"],
                Si_adsorption_dimension=self.co3_status[c]["Ca_adsorption_dimension"],
            )
        return G

    def _anal_caco3_cluster(self):
        if "C" not in self.elem_index or "Ca" not in self.elem_index:
            logger.debug("No Ca or C")
            return
        G = self._add_Ca_n_C_nodes(nx.Graph(), self._make_co3_species_map())
        co3_species_list = []
        for attr in CshParam.species_co3:
            val = getattr(self, attr)
            if val.size != 0:
                co3_species_list.append(val[:, :4])
        co3_species = np.vstack(co3_species_list)
        oc = co3_species[:, 1:]
        ca_oc = self.Ca_O[np.isin(self.Ca_O[:, 1], oc)]
        for ca, o in ca_oc:
            c = co3_species[np.where(co3_species == o)[0][0], 0]
            G.add_edge(ca, c)
        CshParam.result["cluster"].append(G)


class Chain:
    def __init__(self, Si_O, deletable_Si=None):
        self.G = nx.Graph()
        for si, o in Si_O:
            self.G.add_edge(si, o)
        self.Si_graph = nx.Graph()
        self.deletable_Si = deletable_Si if deletable_Si is not None else None
        logger.debug(f"{self.__class__!r} loaded")
        for o in Si_O[:, 1]:
            si_neighbors = list(self.G.neighbors(o))
            for i in range(len(si_neighbors)):
                for j in range(i + 1, len(si_neighbors)):
                    self.Si_graph.add_edge(si_neighbors[i], si_neighbors[j])

    @property
    def chain_length(self):
        return [len(c) for c in nx.connected_components(self.Si_graph)]

    @property
    def sort_chain_length(self):
        return np.sort(self.chain_length)

    @property
    def mean_chain_length(self):
        return np.mean(self.chain_length)

    def prune_until_lenght(self, target):
        nodes = list()
        while True:
            avg = self.mean_chain_length
            if avg <= target:
                break
            ccs = list(nx.connected_components(self.Si_graph))
            max_size = max(len(c) for c in ccs)
            candidates = [c for c in ccs if len(c) == max_size]
            largest_cc = random.choice(candidates)
            candidates = [n for n in largest_cc if n in self.deletable_Si]
            best_node_canidates = list()
            best_var = float("inf")
            for n in candidates:
                G_tmp = self.Si_graph.copy()
                G_tmp.remove_node(n)
                sizes = [len(c) for c in nx.connected_components(G_tmp)]
                v = np.var(sizes)
                if v <= best_var:
                    best_var = v
                    #                    best_node = n
                    best_node_canidates.append(n)
            node = np.random.choice(best_node_canidates)
            #            node = best_node
            self.Si_graph.remove_node(node)
            nodes.append(node)
        return np.array(nodes, dtype=int)


class UnitCell:
    csh_category = [
        "bridgeCa000",
        "bridgeCa00",
        "bridgeCa01",
        "bridgeCa02",
        "crystal00",
        "crystal01",
        "tobermorite",
    ]
    bulk_category = ["w50", "w40"]

    @classmethod
    def _set_config_(cls, unit_name):
        if unit_name is None:
            cwd = os.getcwd()
        else:
            cwd = unit_name
        cwd = os.path.abspath(cwd)
        unit_list = []
        for tk_key in ("b00.w40", "b00.w50", "b00.w60"):
            if tk_key in cwd:
                cls.config = tk_key
                cls.unit_name = tk_key
                return
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

    def __init__(self, unit_name=None, Si_key_upper_lower=False):
        UnitCell._set_config_(unit_name)
        logger.info(f"Set {UnitCell.unit_name} as a unit cell")
        logger.info(f"Set {UnitCell.config} as a config")
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
        for k, v in self.indexDic.items():
            if isinstance(v, dict):
                for (
                    k_,
                    v_,
                ) in v.items():
                    logger.info(f"{k} -> {k_}: {len(v_)}")
            else:
                logger.info(f"{k}: {len(v)}")

    def __getitem__(self, key):
        return self.indexDic[key]

    def set_data(self, cif_data):
        lattice, angle = cif_data[0:2]
        self.data = df2xyz(cif_data[-1])
        self.matrix, _ = setMatrix(lattice, angle)
        self.bond = Bond(self.data, self.matrix)
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
        keys = [key for key in self.label_keys if "O" in key]
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
        all_mask = np.zeros((data_.shape[0],), dtype=bool)
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


class CALC:
    @staticmethod
    def CS(data: np.ndarray, layer=None, verbose=False):
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
            CS = n_Ca / n_Si
            if verbose:
                logger.info(f"[StructureManager] C/S: {CS:.3f}")
        return CS

    @staticmethod
    def MCL(Q1: int, Q2: int) -> float:
        if Q1 == 0:
            val = float("inf")
        else:
            val = 2 * (1 + Q2 / Q1)
        logger.info(f"[StructureManager] MCL: {val:.3f}")
        return val

    @staticmethod
    def Qn(SiO4: np.ndarray, verbose=False, ratio=False):
        """
        Cal Qn from index of SiO4
        SiO4 -> np.ndarray shape(-1, 5) [[Si, O1, O2, O3, O4], []...]
        """
        index_O = SiO4[:, 1:]
        Qn = []
        for i in range(SiO4.shape[0]):
            mask = np.arange(0, SiO4.shape[0])
            mask = np.ones((SiO4.shape[0],), dtype=bool)
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
                    val = count / length
                    QnDic[i] = val if ratio else count
                    logger.info(f"[Silicate Chain] Q{str(i)}: {val:.3f}")
        return QnDic
