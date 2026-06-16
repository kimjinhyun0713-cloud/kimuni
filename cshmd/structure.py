import numpy as np
import pandas as pd
import os
from scipy.spatial import cKDTree
from .analysis import df2charge, df2charge4clayff
from .functions import setMatrix
from .load import poscar2Dic, cif2data
from .util import cif_head, cif_tail, molDic
from .common import df2xyz, type2list, get_logger
from .analysis import CshParam

logger = get_logger(__name__)
pd.options.display.float_format = "{:.3f}".format


class Coordmanager(CshParam):
    molecule = molDic
    verbose = False

    def __init__(self, infile=None, ftype="cif", **kwargs):
        self.rm_list = []
        if infile is not None:
            self._load_file(infile)
        else:
            self._load_direct(**kwargs)
        self.tmpl_df = self.df.iloc[0].copy()
        self.natom = self.natom_init = len(self.df)
        self.update_data(**kwargs)

    def _load_file(self, infile):
        columns = ["label", "type_symbol", "fract_x", "fract_y", "fract_z"]
        logger.info(f"{infile} was loaded")
        self.infile = infile
        self.base, ext = os.path.splitext(infile)
        if self.base == "POSCAR" or ext == ".vasp":
            dic = poscar2Dic(infile)
            self.lattice = dic["lattice"]
            self.angle = dic["angle"]
            self.matrix = dic["matrix"]
            data = dic["data"]
            self.df = pd.DataFrame(data[:, :5], columns=columns)
            self.ftype = "vasp"
        elif ext == ".cif":
            self.lattice, self.angle, self.df = cif2data(infile)
            self.matrix, self.V = setMatrix(self.lattice, self.angle)

    def _load_direct(self, **kwargs):
        self.base = kwargs.pop("base", "df_direct")
        self.df = kwargs.pop("df", None)
        if self.df is None:
            if hasattr(self, "data"):
                self.df = self._data_to_df()
            else:
                raise ValueError("No df or data supplied")
        self.lattice = kwargs.pop("lattice", getattr(self, "lattice", None))
        self.angle = kwargs.pop("angle", getattr(self, "angle", None))
        self.matrix = kwargs.pop("matrix", getattr(self, "matrix", None))
        logger.debug("DataFrame loaded directly")

    def _data_to_df(self):
        columns = ["label", "type_symbol", "fract_x", "fract_y", "fract_z"]
        df = pd.DataFrame(columns=columns)
        df.iloc[:, 0] = self.data[:, 0]
        for i in range(4):
            df.iloc[:, i + 1] = self.data[:, i]
        return df

    def update_data(
        self,
        calculate_matrix=False,
        evaluate_chain=True,
        search_level="full",
        save_chain_info=False,
        **kwargs,
    ):
        if calculate_matrix:
            self.matrix, self.V = setMatrix(self.lattice, self.angle)
        self.data = df2xyz(self.df)
        super(Coordmanager, self).__init__(**kwargs)
        match search_level:
            case "full":
                self.search_full()
            case "chain":
                self.search_chain()
            case "basic":
                self.search_basic()
                evaluate_chain = False
            case "none":
                evaluate_chain = False
            case _:
                raise ValueError("No search_level exists")
        if evaluate_chain:
            self.evaluate_chain(save_chain_info=save_chain_info)

    def __str__(self):
        string = f"\nNumber of atoms: {self.natom_init}\n"
        string += f"Lattice: {' '.join(str(v) for v in self.lattice)}\n"
        string += f"Angle: {' '.join(str(v) for v in self.angle)}\n"
        return string

    def return_charge(self):
        return df2charge(self.df)

    def min_query(self, npoint=100, rcut=2, return_coord=False, **kwargs):
        """
        to provide random coordinations in cutoff "rcut"

        args
        ;; npoint -> int, number of npoint for query
        ;; rcut -> radius cutoff
        return
        ;; None, logger.info fraction to stdout
        ;; if return_coord, return single coord -> np.ndarray(3, )
        """
        fract = self.df[["fract_x", "fract_y", "fract_z"]]
        fract = fract.to_numpy().reshape(-1, 3)
        shifts = np.array(
            [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]
        )
        all_ = fract[:, None, :] + shifts[None, :, :]
        all_ = all_.reshape(-1, 3) @ self.matrix
        tree = cKDTree(all_)
        fract_points = np.random.rand(npoint, 3)
        for i, axis_range in enumerate(["range_x", "range_y", "range_z"]):
            range_ = kwargs.get(axis_range, None)
            if range_ is not None:
                if len(range_) == 1:
                    fract_points[:, i] = range_[0]
                else:
                    fract_points[:, i] = (
                        fract_points[:, i] * (range_[1] - range_[0]) + range_[0]
                    )
        cart_points = fract_points @ self.matrix
        r_min, _ = tree.query(cart_points)
        isolated_points = cart_points[r_min >= rcut] @ np.linalg.inv(self.matrix)
        if return_coord:
            return isolated_points[0] if len(isolated_points) != 0 else None
        if self.verbose:
            logger.info(
                f"{len(isolated_points)} coordinations for insert element, rcut={rcut}"
            )
            for point in isolated_points:
                logger.info("-".join(str(round(s, 3)) for s in point))

    def extendLattice(self, abc=None, **kwargs):  # ckpt_extend
        """
        extend lattice constant

        args
        ;; abc -> list, cartesian value to extend
        ;;;; key -> dict, {axis: fractional value}, start point of extending lattice

        return
        ;;None, update pd.DateFrame
        """
        assert abc is not None
        keyDic = kwargs.get("key", None)
        fmt = "{:.3f} {:.3f} {:.3f}"
        stdout1 = fmt.format(*self.lattice)
        stdout_extra = ""
        if keyDic is not None:
            index = {"x": 0, "y": 1, "z": 2}
            for axis, key in keyDic.items():
                stdout_extra += ", along axis {}: {}".format(axis, key)
                shift = abc[index[axis]] / self.lattice[index[axis]]
                flag_shift = self.df[f"fract_{axis}"] > key
                self.df.loc[flag_shift, f"fract_{axis}"] += shift
        self.lattice = [z1 + z2 for z1, z2 in zip(self.lattice, abc)]
        cartesian_df = self.df[["fract_x", "fract_y", "fract_z"]] @ self.matrix
        self.matrix, self.V = setMatrix(self.lattice, self.angle)
        self.df[["fract_x", "fract_y", "fract_z"]] = cartesian_df @ np.linalg.inv(
            self.matrix
        )
        stdout2 = fmt.format(*self.lattice)
        logger.info(f"Extend lattice {stdout1} -> {stdout2} {stdout_extra}")

    def check_neigbor(self, x, y, z, rcut=2.5):  # ckpt_neighbor
        """
        check neighbor stat and return infomration of nearest atoms

        args
        ;; x, y, z -> float, cartesianal value
        ;;; rcut -> float, raidus cutoff
        ;;; matrix - > np.nadarry, matrix of system

        return
        ;; np.nadarry(-1, 3), [elem, index, distance ], stats of those within rcut
        """
        matrix = matrix if matrix is not None else self.matrix

        def set_mask(coord, symbol, idx):
            r = np.abs(self.df[symbol] - coord)
            r = np.where(r > 0.5, 1 - r, r)
            mask = r < (rcut / matrix[idx, idx]) * 1.2
            return mask

        mask_x = set_mask(x, "fract_x", 0)
        mask_y = set_mask(y, "fract_y", 1)
        mask_z = set_mask(z, "fract_z", 2)
        mask = mask_x & mask_y & mask_z
        if np.nonzero(mask)[0].size == 0:
            return np.empty(0, dtype=bool)
        fract_r = self.df.loc[mask, ["fract_x", "fract_y", "fract_z"]] - [x, y, z]
        fract_r = fract_r - np.round(fract_r)
        if fract_r.ndim == 0:
            fract_r = fract_r.to_numpy().reshape(-1, 3)
        norm = np.linalg.norm(fract_r.astype(float) @ matrix, axis=1)
        idx_true = np.nonzero(norm < rcut)[0]
        norm_true = norm[norm < rcut]
        mask_ = np.nonzero(mask)[0][idx_true]
        symbol = (
            self.df.loc[mask_, ["type_symbol"]]
            .to_numpy()
            .reshape(
                -1,
            )
        )
        index = (
            self.df.loc[mask_]
            .index.to_numpy()
            .reshape(
                -1,
            )
        )
        check = np.vstack([symbol, index, norm_true]).T
        return check

    def deformation(self, deform):
        """
        execute deformation of all system along axis

        args
        ;; deform -> list(len=3), deformation values along axis 'a', 'b', 'c'

        return
        ;; update pd.Dataframe inplace
        """
        self.lattice = [z1 * z2 for z1, z2 in zip(self.lattice, deform)]
        self.matrix, self.V = setMatrix(self.lattice, self.angle)
        logger.info("Deformation lattice {} {} {}".format(*deform))

    def cut(self, shrink=False, delete_all=False, **kwargs):  # ckpt_cut
        """
        delete atoms in range

        args

        ;;; shrink -> bool(defalut False), cut the area where cutted down
        ;;; delete_all -> bool(defalut False), perform a logical AND operation over the ranges of 'x', 'y', 'z'
        ;;;; range_{x or y or z}, cut atoms over the range. e.g. x: 0.3-0.5

        return
        ;; update pd.Dataframe inplace
        """
        self.flag_x = self.flag_y = self.flag_z = pd.Series(False, index=self.df.index)
        for axis in ["x", "y", "z"]:
            setattr(self, f"range_{axis}", kwargs.get(axis, None))
            cut_range = getattr(self, f"range_{axis}")
            if cut_range is not None:
                start = cut_range[0]
                end = cut_range[1]
                flag = (self.df[f"fract_{axis}"] >= start) & (
                    self.df[f"fract_{axis}"] <= end
                )
                setattr(self, f"flag_{axis}", flag)
                logger.info(f"Cutting range axis {axis}: {start:.5f} to {end:.5f}")
        flag_xyz = (
            self.flag_x & self.flag_y & self.flag_z
            if delete_all
            else self.flag_x | self.flag_y | self.flag_z
        )
        self.df = self.df[~flag_xyz]
        self.cut_natom = self.natom - len(self.df)
        if self.cut_natom != 0:
            logger.info(f"Number of atoms: {self.natom} -> {len(self.df)}")
        else:
            logger.info(f"Number of atoms: {self.natom}, Did not cutted")
        self.natom = len(self.df)
        self.df.reset_index(drop=True, inplace=True)
        if shrink:
            axis_idx = 0
            for axis in ["x", "y", "z"]:
                cut_range = getattr(self, f"range_{axis}")
                if cut_range is not None:
                    cut_len = cut_range[1] - cut_range[0]
                    flag = self.df[f"fract_{axis}"] > cut_range[1]
                    self.df.loc[flag, f"fract_{axis}"] -= cut_len
                    self.df.loc[:, f"fract_{axis}"] /= 1 - cut_len
                    self.lattice[axis_idx] *= 1 - cut_len
                axis_idx += 1
            logger.info("remove empty space")

    def adjust(self, idx, x=None, y=None, z=None):
        """
        adjust atoms position

        args

        ;; idx -> int, index of atom
        ;;; x, y, z -> float, value for adjust position

        return
        ;; update pd.Dataframe inplace
        """
        elem = self.df.loc[idx, "type_symbol"]
        if x is not None:
            bf = self.df.loc[idx, "fract_x"]
            self.df.at[idx, "fract_x"] = bf + x
            logger.info(f"Updated Value in {elem} a: {bf} -> {bf - x}")
        if y is not None:
            bf = self.df.loc[idx, "fract_y"]
            self.df.at[idx, "fract_y"] = bf + y
            logger.info(f"Updated Value in {elem} b: {bf} -> {bf - y}")
        if z is not None:
            bf = self.df.loc[idx, "fract_z"]
            self.df.at[idx, "fract_z"] = bf + z
            logger.info(f"Updated Value in {elem} c: {bf} -> {bf - z}")

    def delete(self, **kwargs):
        """
        kwargs: idx, label, symbol
        !! 'idx' starts from 0, which differ from id of  'VESTA' starts from 1 !!
        """
        idx = kwargs.get("index", None)
        label = kwargs.get("label", None)
        symbol = kwargs.get("symbol", None)
        if all(param is None for param in [idx, label, symbol]):
            raise ValueError(
                "At least one of 'index', 'label', or 'symbol' must be provided."
            )
        idx = [idx] if idx is not None and not isinstance(idx, list) else idx
        label = [label] if label is not None and not isinstance(label, list) else label
        symbol = (
            [symbol] if symbol is not None and not isinstance(symbol, list) else symbol
        )
        if idx:
            for i in idx:
                if self.verbose:
                    logger.info(
                        f"Delete symbol: {self.df.at[i, 'type_symbol']}, idx: {i}"
                    )
                self.df.drop(i, axis=0, inplace=True)
        if label:
            for l in label:
                flag = self.df["label"].str.lower() == l.lower()
                matching_idx = self.df[flag].index
                for i in matching_idx:
                    if self.verbose:
                        logger.info(
                            f"Delete symbol: {self.df.at[i, 'type_symbol']}, label: {l}"
                        )
                self.df.drop(matching_idx, axis=0, inplace=True)
        if symbol:
            for s in symbol:
                flag = self.df["symbol"].str.lower() == s.lower()
                matching_idx = self.df[flag].index
                for i in matching_idx:
                    if self.verbose:
                        logger.info(
                            f"Delete symbol: {self.df.at[i, 'type_symbol']}, symbol: {s}"
                        )
                self.df.drop(matching_idx, axis=0, inplace=True)
        self.natom = len(self.df)
        self.df.reset_index(drop=True, inplace=True)

    def add_rm_list_water(self):
        logger.info("Now eliminate all H2O but OH or H3O")
        self.rm_list.append(self.H2O.ravel())

    def add_rm_list_water_free(self):
        logger.info("Now eliminate all unadsorbed H2O but OH or H3O")
        index_O = self.H2O[:, 0]
        index_Ca = self.bulk_ca
        bool_Ca = np.isin(self.Ca_O[:, 0], index_Ca)
        bool_O = np.isin(self.Ca_O[:, 1], index_O)
        adsorbed_index_O = []
        for i, (isin_Ca, isin_O) in enumerate(zip(bool_Ca, bool_O)):
            if isin_Ca and isin_O:
                adsorbed_index_O.append(self.Ca_O[i, 1])

        # bond_C_O = self.find_C_O(rcut=4, return_only=True)
        # index_C = self.elem_index["C"]
        # bool_C = np.isin(bond_C_O[:, 0], index_C)
        # bool_Oc = np.isin(bond_C_O[:, 1], index_O)
        # for i, (isin_C, isin_Oc) in enumerate(zip(bool_C, bool_Oc)):
        #     if isin_C and isin_Oc:
        #         adsorbed_index_O.append(bond_C_O[i, 1])
        self.rm_list.append(
            self.H2O[~np.isin(self.H2O[:, 0], np.unique(adsorbed_index_O))]
        )

    def add_rm_list_free_Ca(self):
        logger.info("Now eliminate free Ca")
        Ca_not_free = []
        Ca_not_free.append(
            np.unique(self.Ca_O[np.isin(self.Ca_O[:, 1], self.Si_O[:, 1])][:, 0])
        )
        Ca_not_free.append(
            np.unique(self.Ca_O[np.isin(self.Ca_O[:, 1], self.OH_all[:, 0])][:, 0])
        )
        Ca_not_free = np.unique(np.hstack(Ca_not_free))
        Ca_free = self.elem_index["Ca"][~np.isin(self.elem_index["Ca"], Ca_not_free)]
        self.rm_list.append(Ca_free)

    def insert(
        self, xyz=(0.5, 0.5, 0.5), symbol=None, label=None, rotation=None
    ):  # ckpt_insert
        """
        insert atoms or molecule

        args

        ;; xyz -> float, value of position to insert atoms or molecule
        ;;; label, symbol, columns -> string or None, columns of DataFrame of inserted atom

        return
        ;; update pd.Dataframe inplace
        """
        x, y, z = xyz
        if symbol is None:
            raise Exception("[Error] No symbol")
        update_param_list, update_df_list = [], []
        if symbol in self.molecule.keys():
            cartesian = self.molecule[symbol][:, 1:4]
            if rotation is not None:
                theta = np.deg2rad(rotation)
                R = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1],
                    ]
                )

                cartesian = cartesian @ R.T
            fraction = (cartesian @ np.linalg.inv(self.matrix)).astype(float)
            symbols = self.molecule[symbol][:, 0]
            labels = self.molecule[symbol][:, 4]
            for symbol, add_xyz, label in zip(symbols, fraction, labels):
                x, y, z = xyz + add_xyz
                update_param_list.append([label, symbol, x, y, z])
        else:
            if label is None:
                label = symbol.replace("@", "*")
            update_param_list.append([label, symbol, x, y, z])
        columns = ["label", "type_symbol", "fract_x", "fract_y", "fract_z"]
        for param in update_param_list:
            tmpl_df = self.tmpl_df.copy()
            msg = "[StructureManager] Add mole, "
            msg += " ".join(
                f"{p:.3f}" if isinstance(p, float) else str(p) for p in param
            )
            for col, val in zip(columns, param):
                tmpl_df[col] = val
            update_df_list.append(tmpl_df)
        self.natom = len(self.df)
        new_df = pd.DataFrame(update_df_list)
        bf_len = len(self.df)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        af_len = len(self.df)
        if self.verbose:
            logger.info(msg)
            logger.info(f"[StructureManager] Number of atoms: {bf_len} -> {af_len} ")
        self.natom = len(self.df)

    def protonation(self, idx: int):
        fract = self.df.loc[idx, ["fract_x", "fract_y", "fract_z"]]
        cartesian = fract @ self.matrix
        if fract["fract_z"] > 0.25:
            add_iter = iter([[0, 0, 1], [0, 0, -1], [0, 1, 0]])
            add_iter = iter(
                [[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], [0, 0, -1], [0, 1, 0]]
            )
        else:
            add_iter = iter([[0, 0, -1], [0, 0, 1], [0, 1, 0]])
            add_iter = iter(
                [[-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2], [0, 0, -1], [0, 1, 0]]
            )
        add = next(add_iter)
        new_fraction = (cartesian + add) @ np.linalg.inv(self.matrix)
        self.insert(new_fraction, symbol="H", label="ho")
        self.natom = len(self.df)

    def substitute(self, before=None, after=None, **kwargs):
        """
        substitute atoms or molecule

        args
        ;;; before, after : elem, type -> list or str
        ;;;; kwargs        : column, condition, index

        return
        ;; update pd.Dataframe inplace
        """
        if not any((before, after, kwargs.values())):
            raise KeyError("Wrong argument")
        if isinstance(before, tuple):
            before = list(before)
        condition = kwargs.get("condition", None)
        column = kwargs.get("column", None)
        index = kwargs.get("index", None)
        label = kwargs.get("label", None)
        if index:
            before_change = self.df.at[index, "type_symbol"]
            self.df.at[index, "type_symbol"] = after
            self.df.at[index, "label"] = after if label is None else label
            logger.info(
                f"Updated values in column symbol with idx {index}: {before_change}->{after}"
            )
            if label:
                logger.info(
                    f"Updated values in column label with idx {index}: {before_change}->{label}"
                )
            return
        if condition is not None:
            mask = self.df.eval(condition)
            logger.info(mask)
            if column:
                self.df.loc[mask, column] = self.df.loc[mask, column].replace(
                    before, after
                )
                logger.info(
                    f"Updated values in column where condition is true {column}: {condition}"
                )
            else:
                self.df.loc[mask, :] = self.df.loc[mask, :].replace(before, after)
                logger.info(
                    f"Updated entire row values where condition is true: {condition}"
                )
        if isinstance(before, list) and isinstance(after, list):
            mapping = dict(zip(before, after))
            if column:
                self.df[column] = self.df[column].replace(mapping)
                logger.info(
                    f"Updated column {column} after replacement: {before}->{after}"
                )
            else:
                self.df = self.df.replace(mapping)
                logger.info(f"Updated  after replacement: {before} {after}")
        else:
            if column:
                self.df[column] = self.df[column].replace(before, after)
                logger.info(
                    f"Updated column {column} after replacement: {before}->{after}"
                )
            else:
                self.df = self.df.replace(before, after)
                logger.info(f"Updated entire DataFrame: \n{self.df}")

    def proper_location(self, fract_xyz, rcut, **kwargs):
        """
        check whether it is in the correct position

        args
        ;; fract_xyz -> list
        ;; rcut

        return
        ;; bool, whether the position is within rcut
        """
        exception = kwargs.get("exception")
        if exception is not None:
            exception = [exception] if not isinstance(exception, int) else exception
        if not isinstance(fract_xyz, np.ndarray):
            fract_xyz = fract_xyz.to_numpy()
        df_fract_xyz = self.df[["fract_x", "fract_y", "fract_z"]].to_numpy()
        r_fract = df_fract_xyz - fract_xyz
        r_cartesian = (r_fract @ self.matrix).astype(float)
        norm = np.linalg.norm(r_cartesian, axis=1)
        flag = np.ones_like(norm, dtype=bool)
        flag[exception] = False
        norm = norm[flag]
        if np.all(norm > rcut):
            return True
        else:
            return False

    def makeSupercell(self, scell):
        """
        make supercell along rhe axis

        args
        ;; scell -> list(int, int ,int),

        return
        ;; update pd.Dataframe inplace
        """
        lx = int(scell[0])
        ly = int(scell[1])
        lz = int(scell[2])
        self.df["fract_x"] -= np.floor(self.df["fract_x"])
        self.df["fract_y"] -= np.floor(self.df["fract_y"])
        self.df["fract_z"] -= np.floor(self.df["fract_z"])
        if not ((self.df["fract_x"] >= 0).all() and (self.df["fract_x"] < 1).all()):
            raise AssertionError("Error in fract_x values!")
        if not ((self.df["fract_y"] >= 0).all() and (self.df["fract_y"] < 1).all()):
            raise AssertionError("Error in fract_y values!")
        if not ((self.df["fract_z"] >= 0).all() and (self.df["fract_z"] < 1).all()):
            raise AssertionError("Error in fract_z values!")
        self.lattice = [int(z1) * z2 for z1, z2 in zip(scell, self.lattice)]
        scellList = []
        for i in range(lx):
            for j in range(ly):
                for k in range(lz):
                    data = self.df.copy()
                    data["fract_x"] += i
                    data["fract_y"] += j
                    data["fract_z"] += k
                    scellList.append(data)
        self.df = pd.concat(scellList, ignore_index=True)
        self.df["fract_x"] /= lx
        self.df["fract_y"] /= ly
        self.df["fract_z"] /= lz
        af_len = len(self.df)
        miss_atom = self.natom * lx * ly * lz - af_len
        assert_str = f"{miss_atom} Atom Missing while creating Supercell"
        assert miss_atom == 0, assert_str
        if self.verbose:
            logger.info(f"[StructureManager] Number of atoms: {self.natom} -> {af_len}")
        self.natom = af_len
        logger.info(f"[StructureManager] Supercell created {lx} * {ly} * {lz}")

    def writeCif(self, template=None):
        lines = ""
        for c in set(self.df.columns):
            if "iso" in c:
                iso = c
                break
        else:
            iso = "U_iso_or_equiv"
        for _, row in self.df.iterrows():
            line = cif_tail.format(
                str(row["label"]),
                float(row["occupancy"]) if "occupancy" in row.keys() else 1.0,
                float(row["fract_x"]),
                float(row["fract_y"]),
                float(row["fract_z"]),
                str(row["adp_type"]) if "adp_type" in row.keys() else "Uiso",
                float(row[iso]) if iso in row.keys() else 0.0,
                str(row["type_symbol"]),
            )
            lines += f"{line}\n"

        template = cif_head.format(self.infile, *self.lattice, *self.angle)
        template += "\n"
        template += lines
        version = 0
        if hasattr(self, "outfile"):
            _outfile = self.outfile.replace(".cif", "") + ".cif"
        else:
            while True:
                _outfile = f"{self.base}.{version:02d}.cif"
                if not os.path.exists(_outfile):
                    break
                version += 1

        with open(_outfile, "w") as o:
            o.write(template)
            logger.info(f"{_outfile} was created")

    def check_rm_list(self):
        if len(self.rm_list) != 0:
            target_list = np.unique(np.hstack(self.rm_list))
            logger.info(f"Delete {len(target_list)} atoms in rm_list")
            self.delete(index=target_list)
            self.rm_list = []

    def labeling_by_clayff(self):
        logger.info("Now get label according to 'Clayff' style")
        self.find("OH", "H2O", "SiO4", "CO3")
        if len(self.H2O) != 0:
            oo, hh = self.H2O[:, 0], np.unique(self.H2O[:, 1:])
            self.df.loc[oo, ["label"]] = "o*"
            self.df.loc[hh, ["label"]] = "h*"
        if len(self.CO3) != 0:
            co, oc = self.CO3[:, 0], np.unique(self.CO3[:, 1:])
            self.df.loc[co, ["label"]] = "co"
            self.df.loc[oc, ["label"]] = "oc"
        if len(self.SiO4) != 0:
            st, ob = self.SiO4[:, 0], np.unique(self.SiO4[:, 1:])
            self.df.loc[st, ["label"]] = "st"
            self.df.loc[ob, ["label"]] = "ob"
        if len(self.OH) != 0:
            oh, ho = self.OH[:, 0], self.OH[:, 1]
            self.df.loc[oh, ["label"]] = "oh"
            self.df.loc[ho, ["label"]] = "ho"
        mask_Ca = self.df["type_symbol"] == "Ca"
        layerCa = (self.df.loc[mask_Ca, ["fract_z"]] < 0.54) & (
            self.df.loc[mask_Ca, ["fract_z"]] > 0.02
        )
        cah = layerCa[layerCa["fract_z"]].index
        Ca = layerCa[~layerCa["fract_z"]].index
        self.df.loc[cah, ["label"]] = "cah"
        self.df.loc[Ca, ["label"]] = "Ca"
        self.df.loc[Ca, ["label"]] = "cah"
        clayff_label = ["Ca", "cah", "co", "h*", "ho", "o*", "ob", "oc", "st", "oh"]
        unique = np.unique(self.df["label"])
        isin = np.isin(unique, clayff_label)
        if not isin.all():
            elem_not_labeled = unique[~isin]
            sample = ""
            for e in elem_not_labeled:
                sample += f"{e!s} ->  "
                sample += " ".join(
                    str(n) for n in np.nonzero(self.df["label"] == e)[:10][0]
                )
            logger.warning(f"Unlabeled atoms in clayff, {sample} ...")
        else:
            logger.info("All atoms labeled by clayff")
        charge, _ = df2charge4clayff(self.df, return_charge=True)
        logger.info(f"Clayff charge: {charge:.3f}")


class StructureManager(Coordmanager):
    mole_csh_detailed = (
        "bulk_ca",
        "inter_layer_ca",
        "intra_layer_ca",
        "surface_BT_si",
        "surface_PT_si",
        "surface_BT_oh",
        "surface_PT_oh",
        "bulk_ca_oh",
        "intra_layer_ca_oh",
        "inter_layer_ca_oh",
        "both_intra_n_bulk_oh",
    )
    verbose = False
    info = []

    @classmethod
    def out2info(cls):
        cls.df = pd.concat(cls.info, ignore_index=True)
        fname = f"{cls.base if hasattr(cls, 'base') else 'info'}.tmp.pkl"
        cls.df.to_pickle(fname)
        logger.info(f"'{fname}' was created")
        print(cls.df)
        cls.info = []
        return cls.df

    def __init__(self, infile, **kwargs):
        super().__init__(infile, **kwargs)
        mole_all = (
            self.__class__.mole_search_basic + self.__class__.mole_detail_search_oh
        )
        self.mole_all = (
            mole_all if not self.is_csh else mole_all + self.__class__.mole_csh_detailed
        )

    def __str__(self):
        string = " ".join(k for k in self.__dict__.keys())
        return string

    def vprint(self, string: str):
        if self.verbose:
            logger.info(string)

    def set_lammps_data(self):
        self.tetra_o = np.unique(self.SiO4[:, 1:])
        _H2O_family = [
            self.H2O[:, 0] if self.H2O.size != 0 else [],
            self.OH[:, 0] if self.OH.size != 0 else [],
            self.H3O[:, 0] if self.H3O.size != 0 else [],
        ]
        total_h2o = np.hstack(_H2O_family).astype(int)
        _mask_in_layer_h2o1 = self.data[total_h2o, 3] < self.cao_range[1]
        _mask_in_layer_h2o2 = self.data[total_h2o, 3] > self.cao_range[0]
        mask_in_layer_h2o = _mask_in_layer_h2o1 & _mask_in_layer_h2o2
        self.inter_layer_o = total_h2o[mask_in_layer_h2o]
        self.bulk_o = total_h2o[~mask_in_layer_h2o]
        self.carbonate_o = self.C_O[:, 1:].ravel() if self.C_O.size != 0 else None
        # lammpsの.data
        mole_lammps = (
            "bulk_ca",
            "inter_layer_ca",
            "intra_layer_ca",
            "BT_si",
            "PT_si",
            "tetra_o",
            "inter_layer_o",
            "bulk_o",
            "carbonate_o",
        )
        mole_real = ("Ca", "Ca", "Ca", "Si", "Si", "O", "O", "O", "O")
        lammps_mole = np.zeros((len(self.df),), dtype=int)
        lammps_mole_num = 1
        self.lammps_mole = {}
        for name, real in zip(mole_lammps, mole_real):
            #            val = locals().get(f"index_{name}", None)
            val = getattr(self, name)
            if val is None or val.size == 0:
                continue
            lammps_mole[val] = lammps_mole_num
            self.lammps_mole[lammps_mole_num] = {"lammps": name, "real": real}
            lammps_mole_num += 1
        for e in self.elem:
            mask_not_input = np.where(lammps_mole == 0)[0]
            common = np.intersect1d(mask_not_input, self.elem_index[e])
            if common.size == 0:
                continue
            lammps_mole[common] = lammps_mole_num
            self.lammps_mole[lammps_mole_num] = {"lammps": e, "real": e}
            lammps_mole_num += 1
        self.df["lmp_mole"] = lammps_mole
        if np.where(lammps_mole == 0)[0].size != 0:
            raise ValueError

    def terminate(self, target: int | list):
        target = type2list(target)
        for t in target:
            self.protonation(t)

    def add_Ca(
        self,
        target: int | list,
        direction: str,
        param: list = [[0, 0, 1], [0, 0, 1]],
        param_OH: list = [[1.50, -1, 2], [-1.50, 1, 2]],
        param_H2O: list = [[0, 0, 2], [0, 0, -2]],
        n_OH: int = 0,
        n_H2O: int = 0,
        **kwargs,
    ):
        target = type2list(target)
        param_1 = param[0]
        param_2 = param[1]
        param_OH_1 = param_OH[0]
        param_OH_2 = param_OH[1]
        param_H2O_1 = param_H2O[0]
        param_H2O_2 = param_H2O[1]
        if direction == "down":
            param_1 = [-a for a in param_1]
            param_2 = [-a for a in param_2]
            param_OH_1 = [-a for a in param_OH_1]
            param_OH_2 = [-a for a in param_OH_2]
            param_H2O_1 = [-a for a in param_H2O_1]
            param_H2O_2 = [-a for a in param_H2O_2]
        msg = "[StructureManager] Add Ca"
        if n_OH != 0:
            msg += f" with {n_OH} OH"
        if n_H2O != 0:
            msg += f" and with {n_H2O} H2O"
        logger.info(msg)
        for t in target:
            cart = self.data[t, 1:] @ self.matrix
            cart_Ca = (cart + param_1) if direction == "up" else (cart + param_2)
            fract_Ca = (cart_Ca) @ np.linalg.inv(self.matrix)
            self.insert(fract_Ca, symbol="Ca", label="Ca")
            for i in range(n_OH):
                cart_OH = (cart_Ca + param_OH_1) if i == 0 else (cart_Ca + param_OH_2)
                OH = self.molecule["OH"].copy()
                OH[:, 1:4] = (OH[:, 1:4] + cart_OH) @ np.linalg.inv(self.matrix)
                for oh in OH:
                    self.insert(oh[1:4], symbol=oh[0], label=oh[4])
            for i in range(n_H2O):
                cart_H2O = cart_Ca + param_H2O_1 if i == 0 else cart_Ca + param_H2O_2
                H2O = self.molecule["H2O_CSH"].copy()
                H2O[:, 1:4] = (H2O[:, 1:4] + cart_H2O) @ np.linalg.inv(self.matrix)
                for h2o in H2O:
                    self.insert(h2o[1:4], symbol=h2o[0], label=h2o[4])

    def deprotonate_BT(self, target: int | list, scope: str, site: str, direction: str):
        target = type2list(target)
        mask = np.isin(self.SiO4[:, 0], target)
        target_T = self.SiO4[mask]
        self.find_NBO()
        for T in target_T:
            NBO_in_target = T[1:][np.isin(T[1:], self.NBO)]
            if site == "all":
                target_Pr = NBO_in_target
            else:
                func = (
                    np.argmax if (direction == "up") == (site == "upper") else np.argmin
                )
                arg = func(self.data[NBO_in_target][:, 3])
                target_Pr = NBO_in_target[arg]
            mask_O = np.isin(self.SiOH[:, 0], target_Pr)
            proton = self.SiOH[mask_O][:, 1]
            self.rm_list.append(proton)

    def remove_BT(self, target: int | list, termination: bool | str, *add_options):
        """
        Remove BT from the specified target. Optionally retain OH groups.
        Parameters
        target : int or list
        Index or list of indices specifying the target sites.
        termination : bool
        Whether to apply termination after removal or add Ca.
        direction : str
        Direction identifier for the removal operation.
        If True, OH groups are retained; if False (default), they are removed.
        """
        if target is None:
            return
        target = type2list(target)
        mask = np.isin(self.SiO4[:, 0], target)
        self.rm_list.append(target)
        target_T = self.SiO4[mask]
        self.find_NBO()
        target_O_list = []
        term_O_list = []
        for T in target_T:
            T_of_O = T[1:]
            NBO_in_target = T_of_O[np.isin(T_of_O, self.NBO)]
            target_O_list.append(NBO_in_target)
            if termination == "half":
                NBO_in_term = T_of_O[~np.isin(T_of_O, self.NBO)][0]
            elif termination:
                NBO_in_term = T_of_O[~np.isin(T_of_O, self.NBO)]
            else:
                NBO_in_term = []
            term_O_list.append(NBO_in_term)
        target_O = np.hstack(target_O_list)
        self.rm_list.append(target_O)
        term_O = np.hstack(term_O_list)
        if term_O.size != 0:
            self.terminate(term_O)
        for add_option in add_options:
            add = add_option.get("mole", None)
            if add is not None:
                match add:
                    case "Ca":
                        add_option["n_OH"] = 0
                    case "CaOH":
                        add_option["n_OH"] = 1
                    case "CaOH2":
                        add_option["n_OH"] = 2
                self.add_Ca(target, **add_option)
        target_H = self.SiOH[np.isin(self.SiOH[:, 0], target_O)][:, 1]
        self.rm_list.append(target_H)

    def set_info(self, **kwargs):
        df = pd.DataFrame(index=range(1))
        df["Structure"] = [self.base.replace(".00", "")]
        df["Natoms"] = len(self.df)
        for elem, indexs in self.elem_index.items():
            df[elem] = indexs.shape[0]
        df["Charge"] = df2charge(self.df)
        if self.is_csh:
            df["H2O(solution)"] = np.sum(self._get_bulk_mask(self.H2O[:, 0]))
            df["H2O(C-S-H)"] = np.sum(self._get_csh_mask(self.H2O[:, 0]))
            if hasattr(self, "MCL"):
                df["MCL"] = self.MCL
            Ca_size = 0
            for stat in ("pair", "bridge", "defect", "outer"):
                Ca_size += sum(
                    vdic.get("adsorbed") == stat for vdic in self.ca_status.values()
                )
            Ca_size += self.inter_layer_ca.shape[0] + self.intra_layer_ca.shape[0]
            df["Ca/Si"] = Ca_size / self.elem_index["Si"].shape[0]
            df["Ca(solution)"] = self.bulk_ca.shape[0]
            df["CO3(solution)"] = self.CO3.shape[0]
        if kwargs.get("basic", False):
            if self.is_csh:
                clDic = self.chain_info.pop("CL", None)
                for k, v in self.chain_info.items():
                    df[k] = v
                df["Q"] = self.Q_factor
                for i in range(1, 5):
                    df[f"Q{str(i)}"] = self.Qn.get(i, 0)
                for cl, count in clDic.items():
                    df[f"CL: {cl}"] = count

        if not kwargs.get("no_analysis", False) and self.is_csh:
            self.find_NBO()
            mask_BT = np.isin(self.NBO, self.surface_BT_O)
            intra_O = self.Ca_O[np.isin(self.Ca_O[:, 0], self.intra_layer_ca)][:, 1]
            surface_PT_O_not_in_intra = self.surface_PT_O[
                ~np.isin(self.surface_PT_O, intra_O)
            ]
            mask_PT = np.isin(self.NBO, surface_PT_O_not_in_intra)
            mask = mask_BT | mask_PT
            surface_NBO = self.NBO[mask]
            n_NBO = surface_NBO.shape[0]
            # SiOH = self.surface_BT_oh.shape[0] + self.surface_PT_oh.shape[]0
            # df["SiOH(ratio)"] = SiOH / (n_NBO)
            df["NBO"] = n_NBO
            df["free OH"] = self.OH.shape[0]
            df["OH[mol/L]"] = self.OH.shape[0] / (2750 / 55.5)
            df["SiOH/Si(all)"] = self.SiOH.shape[0] / self.elem_index["Si"].shape[0]
            df["CaOH/Ca(all)"] = (
                self.bulk_caoh.shape[0] + self.surface_caoh.shape[0]
            ) / self.elem_index["Ca"].shape[0]
            df["SiOH/Si(C-S-H)"] = (
                self.bulk_BT_oh.shape[0] + self.bulk_PT_oh.shape[0]
            ) / (self.bulk_BT_si.shape[0] + self.bulk_PT_si.shape[0])
            df["CaOH/Ca(C-S-H)"] = self.bulk_caoh.shape[0] / (
                self.intra_layer_ca.shape[0] / 2 + self.inter_layer_ca.shape[0]
            )
            H2O_mask1 = self.data[self.H2O[:, 0], 3] < self.cao_range[1]
            H2O_mask2 = self.data[self.H2O[:, 0], 3] > self.cao_range[0]
            len_H2O_mask = np.sum(H2O_mask1 & H2O_mask2)
            df["H2O/Si"] = len_H2O_mask / self.elem_index["Si"].shape[0]
            len_H2O_all = (
                len_H2O_mask
                + self.bulk_oh.shape[0]
                + self.bulk_caoh.shape[0]
                + self.bulk_PT_oh.shape[0]
                + self.bulk_BT_oh.shape[0]
                + self.bulk_PT_oh.shape[0]
            )
            df["H2O/Si(Qomi, et al.)"] = len_H2O_all / self.elem_index["Si"].shape[0]
            df["H2O(interlayer)"] = len_H2O_mask
            # df["H2O(interlayer, include OH)"] = len_H2O_all
        df["H3O"] = self.H3O.shape[0]
        if kwargs.get("specific", False):
            for m in self.mole_all:
                if hasattr(self, m):
                    val = getattr(self, m)
                    num = val.shape[0]
                else:
                    num = 0
                df[m] = [int(num)]
        self.__class__.info.append(df)
