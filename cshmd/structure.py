import numpy as np
import pandas as pd
import os
import subprocess as sp
from scipy.spatial import cKDTree
from .analysis import find_mole, unwrap_mole, df2charge
from .functions import setMatrix
from .load import poscar2Dic, cif2data
from .util import cif_head, cif_tail, molDic 
from .common import df2xyz, type2list
from .analysis import BOND, cal_Qn, cal_CS, cal_MCL


class DATA():
    
    infile = None
    rcut = None
    molecule = molDic
        
    def __init__(self, infile):        
        self.infile = infile
        print(f"[INFO] {infile} was loaded")
        self.base, ext = os.path.splitext(infile)
        if self.base == "POSCAR" or ext == ".vasp":
            dic = poscar2Dic(infile)
            self.lattice = dic["lattice"]
            self.angle = dic["angle"]
            self.matrix = dic["matrix"]
            columns = ["label", "type_symbol", "fract_x", "fract_y", "fract_z"]
            data = dic["data"]
            self.df = pd.DataFrame(columns=columns)
            self.df.iloc[:, 0] = data[:, 0]
            for i in range(4):
                self.df.iloc[:, i + 1] = data[:, i]
            self.ftype = "vasp"
        else:
            self.lattice, self.angle, self.df = cif2data(self.infile)
            self.columns = self.df.columns
            self.matrix, self.V = setMatrix(self.lattice, self.angle)
            self.ftype = "cif"
        self.natom, self.natom_init = len(self.df), len(self.df)

        
        
    def __str__(self):
        string = f"\nNumber of atoms: {self.natom_init}\n"
        string += f"Lattice: {' '.join(str(v) for v in self.lattice)}\n"
        string += f"Angle: {' '.join(str(v) for v in self.angle)}\n"
        return string
            
    def update_matrix(self):
        self.matrix, self.V = setMatrix(self.lattice, self.angle)

    def return_charge(self):
        return df2charge(self.df)
        
    def min_query(self, npoint=100, rcut=2, return_coord=False, **kwargs):
        """
        to provide random coordinations in cutoff "rcut"

        args
        ;; npoint -> int, number of npoint for query
        ;; rcut -> radius cutoff
        return
        ;; None, print fraction to stdout
        ;; if return_coord, return single coord -> np.ndarray(3, )
        """
        fract = self.df[["fract_x", "fract_y", "fract_z"]]
        fract = fract.to_numpy().reshape(-1, 3)
        shifts = np.array([[i, j, k] for i in [-1, 0, 1]
                           for j in [-1, 0, 1]
                           for k in [-1, 0, 1]])
        all_ = (fract[:, None, :] + shifts[None, :, :])
        all_ = all_.reshape(-1, 3) @ self.matrix
        tree = cKDTree(all_)
        fract_points = np.random.rand(npoint, 3)
        for i, axis_range in enumerate(["range_x", "range_y", "range_z"]):
            range_ = kwargs.get(axis_range, None)
            if range_ is not None:
                if len(range_) == 1:
                    fract_points[:, i] = range_[0]
                else:
                    fract_points[:, i] = fract_points[:, i] * (range_[1] - range_[0]) + range_[0]
        cart_points = fract_points @ self.matrix
        r_min, _ = tree.query(cart_points)
        isolated_points = cart_points[r_min >= rcut] @ np.linalg.inv(self.matrix)
        if return_coord:
            return isolated_points[0] if len(isolated_points) != 0 else None
        if self.verbose:
            print()
            print(f"{len(isolated_points)} coordinations for insert element, rcut={rcut}")
            for point in isolated_points:
                print("-".join(str(round(s, 3)) for s in point))
                

                    
    def extendLattice(self, abc=None, **kwargs):#ckpt_extend
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
        self.df[["fract_x", "fract_y", "fract_z"]] = cartesian_df @ np.linalg.inv(self.matrix)
        stdout2 = fmt.format(*self.lattice)
        print(f"Extend lattice {stdout1} -> {stdout2} {stdout_extra}")


    def check_neigbor(self, x, y, z, rcut=3, matrix=None):#ckpt_neighbor
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
        fract_r = (fract_r - np.round(fract_r))
        if fract_r.ndim == 0:
            fract_r = fract_r.to_numpy().reshape(-1, 3)
        norm = np.linalg.norm(fract_r.astype(float) @ matrix, axis=1)
        idx_true = np.nonzero(norm < rcut)[0]
        norm_true = norm[norm < rcut]
        mask_ = np.nonzero(mask)[0][idx_true]
        symbol = self.df.loc[mask_, ["type_symbol"]].to_numpy().reshape(-1, )
        index = self.df.loc[mask_].index.to_numpy().reshape(-1, )
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
        print("Deformation lattice {} {} {}".format(*deform))

        
    def cut(self, shrink=False, delete_all=False, **kwargs):#ckpt_cut
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
                flag = (self.df[f"fract_{axis}"] >= start) & (self.df[f"fract_{axis}"] <= end)
                setattr(self, f"flag_{axis}", flag)
                print(f"Cutting range axis {axis}: {start:.5f} to {end:.5f}")
        flag_xyz = self.flag_x & self.flag_y & self.flag_z if delete_all else self.flag_x | self.flag_y | self.flag_z
        self.df = self.df[~flag_xyz]
        self.cut_natom = self.natom - len(self.df)
        if self.cut_natom != 0:
            print(f"Number of atoms: {self.natom} -> {len(self.df)}")
        else:
            print(f"Number of atoms: {self.natom}, Did not cutted")
        self.natom = len(self.df)
        if shrink:
            axis_idx = 0
            for axis in ["x", "y", "z"]:
                cut_range = getattr(self, f"range_{axis}")
                if cut_range is not None:
                    cut_len = cut_range[1] - cut_range[0]
                    flag = self.df[f"fract_{axis}"] > cut_range[1]
                    self.df.loc[flag, f"fract_{axis}"] -= cut_len
                    self.df.loc[:, f"fract_{axis}"] /= (1 - cut_len)
                    self.lattice[axis_idx] *= (1 - cut_len)
                axis_idx += 1
            print("remove empty space")
        
        
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
            print(f"Updated Value in {elem} a: {bf} -> {bf - x}")
        if y is not None:
            bf = self.df.loc[idx, "fract_y"]
            self.df.at[idx, "fract_y"] = bf + y
            print(f"Updated Value in {elem} b: {bf} -> {bf - y}")
        if z is not None:
            bf = self.df.loc[idx, "fract_z"]
            self.df.at[idx, "fract_z"] = bf + z
            print(f"Updated Value in {elem} c: {bf} -> {bf - z}")

    def delete(self, **kwargs):
        """
        kwargs: idx, label, symbol
        !! 'idx' starts from 0, which differ from id of  'VESTA' starts from 1 !!
        """
        idx = kwargs.get("index", None)
        label = kwargs.get("label", None)
        symbol = kwargs.get("symbol", None)
        if all(param is None for param in [idx, label, symbol]):
            raise ValueError("At least one of 'index', 'label', or 'symbol' must be provided.")
        idx = [idx] if idx is not None and not isinstance(idx, list) else idx
        label = [label] if label is not None and not isinstance(label, list) else label
        symbol = [symbol] if symbol is not None and not isinstance(symbol, list) else symbol
        if idx:
            for i in idx:
                if self.verbose:
                    print(f"Delete symbol: {self.df.at[i, 'type_symbol']}, idx: {i}")
                self.df.drop(i, axis=0, inplace=True)
        if label:
            for l in label:
                flag = self.df["label"].str.lower() == l.lower()
                matching_idx = self.df[flag].index
                for i in matching_idx:
                    if self.verbose:
                        print(f"Delete symbol: {self.df.at[i, 'type_symbol']}, label: {l}")
                self.df.drop(matching_idx, axis=0, inplace=True)
        if symbol:
            for s in symbol:
                flag = self.df["symbol"].str.lower() == s.lower()
                matching_idx = self.df[flag].index
                for i in matching_idx:
                    if self.verbose:
                        print(f"Delete symbol: {self.df.at[i, 'type_symbol']}, symbol: {s}")
                self.df.drop(matching_idx, axis=0, inplace=True)
                
        self.natom = len(self.df)
                
    def insert(self, xyz=(0.5, 0.5, 0.5), label=None, symbol=None, columns=None, matrix=None):#ckpt_insert
        """
        insert atoms or molecule

        args

        ;; xyz -> float, value of position to insert atoms or molecule
        ;;; label, symbol, columns -> string or None, columns of DataFrame of inserted atom
        
        return
        ;; update pd.Dataframe inplace
        """
        def updateMole(molecule=None):
            output = ""
            check = self.check_neigbor(x, y, z, rcut=2.5, matrix=matrix) if matrix is not None else self.check_neigbor(x, y, z, rcut=2.5)
            if check.size != 0:
                if self.verbose:
                    output += f"Warning! ({x:.3f}, {y:.3f}, {z:.3f}) label: {label} symbol: {symbol}\n"
                    output += "Has some near neighbor atoms\n"
                for arr in check:
                    if self.verbose:
                        output += "{} id: {} {:.2f}\n".format(*arr)
                        if arr[2] < 1.5:
                        # print(x, y, z)
                        # print(check)
                            output += "Update failed\n"
                        # return False, output
            if molecule is None:
                template_df['label'] = label
                template_df['type_symbol'] = symbol
                template_df['fract_x'] = x
                template_df['fract_y'] = y
                template_df['fract_z'] = z
            else:
                template_df.loc[molecule, 'label'] = label
                template_df.loc[molecule, 'type_symbol'] = symbol
                template_df.loc[molecule, 'fract_x'] = x
                template_df.loc[molecule, 'fract_y'] = y
                template_df.loc[molecule, 'fract_z'] = z
            if self.verbose:
                output += f"Updated new molecule: ({x:.3f}, {y:.3f}, {z:.3f}) label: {label} symbol: {symbol}"
            return True, output
        x, y, z = xyz
        columns = self.columns if columns is None else columns
        output_ = ""
        if symbol is None:
            raise Exception("No elem")
        if symbol in self.molecule.keys():
            mole_num = self.molecule[symbol].shape[0]
            template_df = self.df.loc[0:mole_num - 1, :].copy()
            output_ += f"Add {symbol}"
            cartesian = self.molecule[symbol][:, 1:4]
            fraction = (cartesian @ np.linalg.inv(self.matrix)).astype(float)
            elems = self.molecule[symbol][:, 0]
            labels = self.molecule[symbol][:, 4]
            range_ = range(mole_num)
            for e, add_xyz, l, i in zip(elems, fraction, labels, range_):
                label, symbol = l, e
                x, y, z = xyz + add_xyz
                updated, output = updateMole(molecule=i)
                output_ += output
                if not updated:
                    return False
        else:
            template_df = self.df.loc[0, :].copy()
            label = label.replace("@", "*") if label is not None else symbol
            updated, output = updateMole()
            output_ += output
        if self.verbose:
            print(output_)
        if updated:
            self.natom = len(self.df)
            try:
                self.df = pd.concat([self.df, template_df.to_frame().T], ignore_index=True)
            except AttributeError:
                self.df = pd.concat([self.df, template_df], ignore_index=True)
            return True
        else:
            return False
 
    def protonation(self, idx: int):
        before_natom = len(self.df)
        fract = self.df.loc[idx, ["fract_x", "fract_y", "fract_z"]]
        cartesian = fract @ self.matrix
        if fract["fract_z"] > 0.25:
            add_iter = iter([[0, 0, 1], [0, 0, -1], [0, 1, 0]])
            add_iter = iter([[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], [0, 0, -1], [0, 1, 0]])
        else:
            add_iter = iter([[0, 0, -1], [0, 0, 1], [0, 1, 0]])
            add_iter = iter([[-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2], [0, 0, -1], [0, 1, 0]])
        add = next(add_iter)
        new_fraction = (cartesian + add) @ np.linalg.inv(self.matrix)
        while True:
            proper = True
            if proper:
                self.insert(new_fraction, symbol="H", label="ho")
                break
            else:
                try:
                    add = next(add_iter)
                except StopIteration:
                    print("improper Site", f"Index: {idx}")
                    break
                new_fraction = (cartesian + add) @ np.linalg.inv(self.matrix)
        if self.verbose:
            print(f"Number of atoms: {before_natom} -> {len(self.df)}")
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
            print(f"Updated values in column symbol with idx {index}: {before_change}->{after}")
            if label:
                print(f"Updated values in column label with idx {index}: {before_change}->{label}")
            return 
        if condition is not None:
            mask = self.df.eval(condition)
            print(mask)
            if column:
                self.df.loc[mask, column] = self.df.loc[mask, column].replace(before, after)
                print(f"Updated values in column where condition is true {column}: {condition}")
            else:
                self.df.loc[mask, :] = self.df.loc[mask, :].replace(before, after)
                print(f"Updated entire row values where condition is true: {condition}")
        if isinstance(before, list) and isinstance(after, list):
            mapping = dict(zip(before, after))
            if column:
                self.df[column] = self.df[column].replace(mapping)
                print(f"Updated column {column} after replacement: {before}->{after}")
            else:
                self.df = self.df.replace(mapping)
                print(f"Updated  after replacement: {before} {after}")
        else:
            if column:
                self.df[column] = self.df[column].replace(before, after)
                print(f"Updated column {column} after replacement: {before}->{after}")
            else:
                self.df = self.df.replace(before, after)
                print(f"Updated entire DataFrame: \n{self.df}")
                
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
        r_fract = (df_fract_xyz - fract_xyz)
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
        print(f"Number of atoms: {self.natom} -> {af_len}")
        self.natom = af_len
        print(f"Supercell created {lx}*{ly}*{lz}")
        
        
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
                str(row['label']),
                float(row['occupancy']) if 'occupancy' in row.keys() else 1.0,
                float(row['fract_x']),          
                float(row['fract_y']),          
                float(row['fract_z']),
                str(row['adp_type']) if 'adp_type' in row.keys() else "Uiso",
                float(row[iso]) if iso in row.keys() else 0.0,     
                str(row['type_symbol']),
            )
            lines += f"{line}\n"

        template = cif_head.format(self.infile, *self.lattice, *self.angle)
        template += "\n"
        template += lines
        version = 0
        if hasattr(self, "outfile"):
            _outfile = self.outfile
        else:
            while True:
                _outfile = f"{self.base}.{version:02d}.cif"
                if not os.path.exists(_outfile):
                    break
                version += 1
                
        with open(_outfile, "w") as o:
            o.write(template)
            print(f"{_outfile} was created")



class STRUCTURE(DATA):
    
    mole_need_check = ["H2O", "SiO4", "OH", "CO3"]
    mole_dont_check = ["HCO3", "SiOH", "CaOH"]
    mole_all = mole_need_check + mole_dont_check
    verbose = False
    info = []

    @classmethod
    def out2info(cls):
        df = pd.concat(cls.info, ignore_index=True)
        print(df)

    def __init__(self, infile, model_config=None):
        super().__init__(infile)
        if model_config is not None:
            self.model = model_config
        for v in ["mole_all", "mole_need_check", "verbose"]:
            setattr(self, v, getattr(self.__class__, v))
        self.update_data(updata_matrix=False)
        self.rm_list = list()

        
    def __str__(self):
        string = " ".join(k for k in self.__dict__.keys())
        return string

    def vprint(self, string: str):
        if self.verbose:
            print(string)
            
    def update_data(self, updata_matrix=True):
        self.set_data(updata_matrix=updata_matrix)
        self.set_bond()
        self.validate_chain()
        self.validate_tetra()
        
    def set_data(self, updata_matrix=True):
        self.data = df2xyz(self.df)
        self.elem = np.unique(self.data[:, 0])
        self.elem_index = {}
        self.is_csh = True if "Si" in self.elem else False
        for e in self.elem:
            self.elem_index[e] = np.nonzero(self.data[:, 0] == e)[0]
        if updata_matrix:
            super().update_matrix()
        
    def set_bond(self):
        self.bond = BOND(self.data, self.matrix)
        self.bond.verbose = self.verbose
        self.bond.find(*self.mole_need_check)
        if self.is_csh:
            range_Si = self.data[self.elem_index["Si"]][:, 3]
            self.layer_range = (min(range_Si), max(range_Si))
            self.bond.check_where_OH(range_CaOH=self.layer_range)
        else:
            self.bond.check_where_OH()
        for m in self.mole_all:
            if hasattr(self.bond, m):
                setattr(self, m, getattr(self.bond, m))
        
    def validate_chain(self):
        """
        Check whether the silicate chain is broken, and
        calculate QN, CS
        """
        if not self.is_csh:
            return
        isin_tetra = np.isin(self.elem_index["Si"], self.SiO4[:, 0])
        if not isin_tetra.all():
            broken = self.elem_index["Si"][~isin_tetra]
            broken_str = ' '.join(str(v) for v in broken)
            print(f"\nWarning: Some Silicate Chain broken: {broken_str}")
            return
        self.Qn = cal_Qn(self.SiO4, verbose=True)
        self.CS = cal_CS(self.data, layer=self.layer_range, verbose=True)
        
        
    def validate_tetra(self):
        """
        Initialize self.tetra,
        calculate MCL

        required: config which type of 'yaml' (set self.model) 
        """
        if not hasattr(self, "model"):
            return
        self.tetra = {"BT": {}, "PT": {}}
        self.tetra["BT"] = {"surface": [], "bulk": []}
        self.tetra["PT"] = {"surface": [], "bulk": []}
        model = getattr(self, "model")
        BTcount = 0
        PTcount = 0
        for key1 in model.keys():
            for key2 in model[key1].keys():
                ranged = model[key1][key2]
                for s in ranged:
                    mask1 = self.data[self.elem_index["Si"], 3] > s[0]
                    mask2 = self.data[self.elem_index["Si"], 3] < s[1]
                    mask = mask1 & mask2
                    self.tetra[key1][key2].append(self.elem_index["Si"][mask])
                    if key1 == "BT":
                        BTcount += np.sum(mask)
                    else:
                        PTcount += np.sum(mask) 
        assert (BTcount + PTcount) == self.elem_index["Si"].shape[0], "Check model file"
        print(f"[Structure] BT: {BTcount}, PT: {PTcount}")
        termQ2 = len(self.tetra["PT"]["bulk"][0]) + len(self.tetra["PT"]["bulk"][1])
        termQ2 += len(self.tetra["BT"]["bulk"][0]) + len(self.tetra["BT"]["bulk"][1])
        Q1 = self.Qn.get(1, 0)
        Q2 = self.Qn.get(2, 0) - termQ2
        self.MCL = cal_MCL(Q1, Q2)
        
    def terminate(self, target: int | list):
        target = type2list(target)
        for t in target:
            self.protonation(t)

    def decorate_Ca(self, target: int | list, location: str):
        target = type2list(target)
        if location == "top":
            add = [0, 0, 1]
        elif location == "bottom":
            add = [0, 0, -1]
        else:
            raise ValueError("ERROR: set 'location'")
        for t in target:
            cartesian = self.data[t, 1:] @ self.matrix
            fract = (cartesian + add) @ np.linalg.inv(self.matrix)
            self.insert(fract, symbol="Ca", label="Ca")
            
    def remove_BT(self,
                  target: int | list,
                  termination: bool = True,
                  location: str = "top",
                  keep_OH:  bool = False):
        """
        Remove BT from the specified target. Optionally retain OH groups.
        Parameters
        target : int or list
        Index or list of indices specifying the target sites.
        termination : bool
        Whether to apply termination after removal or add Ca.
        location : str
        Location identifier for the removal operation.
        keep_OH : bool, optional
        If True, OH groups are retained; if False (default), they are removed.
        """
        if not hasattr(self, "model"):
            raise ValueError("Missing keyword' model' in config")
        mask = np.isin(self.SiO4[:, 0], target)
        self.rm_list.append(target)
        target_T = self.SiO4[mask]
        candidate_O = np.unique(target_T[:, 1:])
        NBO = self.bond.find_NBO()
        target_O = candidate_O[np.isin(candidate_O, NBO)]
        if termination:
            term_O = candidate_O[~np.isin(candidate_O, NBO)]
            self.terminate(term_O)
        else:
            self.decorate_Ca(target, location=location)
        if keep_OH:
            maska = self.data[target_O, 3] > self.model["BT"]["surface"][0][1]
            maskb = self.data[target_O, 3] < self.model["BT"]["surface"][1][1]
            maskab = maska | maskb
            target_O = target_O[maskab]
        target_OH = self.SiOH[np.isin(self.SiOH[:, 0], target_O)].ravel()
        self.rm_list.append(target_OH)
            
    def set_info(self):
        df = pd.DataFrame()
        df["Structure"] = [self.base.replace(".00", "")]
        df["Natoms"] = len(self.df)
        df["Charge"] = df2charge(self.df)
        if "Ca" in self.elem:
            df["Ca"] = np.sum(self.data[:, 0] == "Ca")
        for m in self.mole_all:
            if hasattr(self, m):
                val = getattr(self, m)
                num = val.shape[0]
            else:
                num = 0
            df[m] = [int(num)]
        self.__class__.info.append(df)


    def remove(self):
        if len(self.rm_list) != 0:
            arr = np.hstack(self.rm_list)
            target_list = sorted(arr, reverse=True)
            for t in target_list:
                self.delete(index=t)
            self.rm_list = []

            
def main():
    import argparse
    import glob
    description = """
    Modify a CIF or POSCAR
    """.strip()
    
    par = argparse.ArgumentParser(description=description, prog="CIF editor")
    par.add_argument(
        'infile', nargs="?",
        help="Input CIF file (if not provided, it will be selected automatically)")
    par.add_argument(
        '--verbose', action="store_false", default=False,
        help="set verbosity level by bool")
    par.add_argument(
        '-o', '--outfile',
        help="Name of outfile")
    par.add_argument(
        '-r', '--run', default=True, action="store_false",
        help="Open after created CIF file")
    par.add_argument(
        '-i', '--insert', nargs="+",
        help="Insert atoms or molecule(H2O, CO3...) : <coord-coord-coord-symbol> or <coord-coord-coord-symbol-label>\n"
        "e.g. 0.7-0.5-0.4-I 0.2-0.1-0.6-O-ob => ")
    par.add_argument(
        '-pr', '--protonation', nargs="+", type=int,
        help="Perform protonation: <indices>...\n"
        "e.g. 2 13 20 30")
    par.add_argument(
        '-apr', '--auto_protonation', default=False, action="store_true",
        help="Perform protonation automatically (default: False)")
    par.add_argument(
        '-c', '--cut', nargs=3,
        help="Delete atoms over the range: (if --shrink, shell is shrinked) <coord-coord> <coord-coord> <coord-coord>\n"
        "e.g. 0.1-0.5 0.6-0.9 0.1-0.3")
    par.add_argument(
        '-ca', '--cut_axis', nargs="+",
        help="Delete atoms along the axis. you can input multiple values: (if --shrink, shell is shrinked) <axis coord-coord> [axis coord-coord] \n"
        "e.g. z 0.1-0.3 0.6-0.9 x 0.1-0.5")
    par.add_argument(
        '-ac', '--auto_cut_axis', 
        help="Remove voids within rcut along the axis: <axis-cutoff>")
    par.add_argument(
        '-e', '--extend_lattice', nargs=3, default=[0, 0, 0], type=float,
        help="Extend lattice constants: <val> <val> <val>\n"
        "e.g. 6.5 5 0")
    par.add_argument(
        '-ev', '--extend_vaccum', nargs="+", default=None,
        help="Extend one of the lattice constants along a chosen axis; you can designate the coordinate along the axis where the extension is applied.\n"
        "<axis-coord-val> [axis-coord-val]\n"
        "z-0.5-5 x-0.1-10")
    par.add_argument(
        '-as', '--assemble', nargs=2,
        help="Assemble another CIF file, when lattice of axis-a axis-b is same <hoge.cif> <axis>\n"
        "e.g. hoge.pb x")
    par.add_argument(
        '-mi', '--make_interface', nargs="*", default=None,
        help="Create an interface along the c-axis. You can choose the element or molecule (e.g., w=H2O, c=CO3) and specify the number of each.: <(mole or element)-int> [(mole or element)-int]\n"
        "e.g. w-100 c-1 Ca-2 I-3")
    par.add_argument(
        '-d', '--delete', nargs="+",
        help="Remove atoms corresponding to the given indices. You can use slice expressions such as 125-129\n"
        "e.g. 1-12 20 31")
    par.add_argument(
        '-s', '--supercell', nargs=3, default=[1, 1, 1],
        help="Make supercell  along three scale factors (copy the system): <int int int>\n"
        "e.g. 2 3 1")
    par.add_argument(
        '-df', '--deformation', nargs="+", default=[1], type=float,
        help="Deform the system along three scale factors: <float float float> or <float> (<1.1> equivalent to <1.1 1.1 1.1>)\n"
        "e.g. 1.2 1.2 1.2 or 1.1")
    par.add_argument(
        '-a', '--adjust', nargs="+",
        help="Adjust the positions of atom with specific index atom: <coord-coord-coord-index\n"
        "e.g. @0.1-@0.1-0-12, '@'means minus")
    par.add_argument(
        '-b', '--sub', nargs="+",
        help="Substitute the atom with a specific index by another element: <index-element> or <index-element-label>\n"
        "e.g. 378-I 2312-O-ob")
    par.add_argument(
        '-sc', '--sub_column', nargs="+",
        help="Substitute the word in specific column of pd.Dataframe by another word: <str-str-column>\n"
        "e.g. Ca-Cah-label")
    par.add_argument(
        '-rc', '--random_coord', nargs="*",
        help="Print random coordinates that are at least 2.5 angstrom away from other atoms within the given range: <axis-coord-coord> [axis-coord-coord]\n"
        "e.g. z-0.4-0.6 x-0.1-0.6")
    par.add_argument(
        '-ri', '--random_insert', nargs="+",
        help="Insert random coordinates that are at least 2.5 angstrom away from other atoms within the given range: <mole-natoms>... [axis-coord-coord]... \n"
        "e.g. z-0.4-0.6 w-5")
    par.add_argument(
        '--shrink', default=False, action="store_true",
        help="(This argument is used only when the 'cut or cut_axis' is enabled)\n"
        "The system is shrunk after performing 'cut' or 'cut_axis'.")
    par.add_argument(
        '--search', nargs="+",
        help="Search for atoms that satisfy the specified condition (element type or location or index or label): <'e'-element or axis-coord-coord or index or l>...\n"
        "e.g. e-Ca x-0.1-0.5 z-0.5-0.8, or 82 or l-oh")
    par.add_argument(
        '--clayff', default=False, action="store_true",
        help="Make label in accordance with the ClayFF style")
    par.add_argument(
        '--spacing', default=3.5, type=float,
        help="(This argument is used only when the 'make_interface' is enabled)\n"
        "make grids that are at least 2.5 angstrom away from other atoms")
    par.add_argument(
        '--mode', default=0, type=int,
        help="(This argument is used only when the 'make_interface' is enabled)\n"
        "Set the condition for a specific element or the location of a molecule")
    args = par.parse_args()

    Modifier.verbose = args.verbose
    Modifier.spacing = args.spacing
    Modifier.shrink = args.shrink
    if args.mode != 0:
        setattr(Modifier, f"mode{args.mode}", True)

    infile = args.infile if args.infile else glob.glob("*cif")[0]
    if args.outfile:
        base = args.outfile.replace(".cif", "")
        Modifier.outfile = base + ".cif"
        
    Modifier.order = ["search", "make_interface", "assemble", "random_coord"]
    Modifier.order += ['sub', 'insert', 'protonation', 'auto_protonation', 'random_insert',
                  'adjust', 'auto_cut_axis', 'extend_vaccum', 'extend_lattice',
                  'delete', 'deformation', 'sub_column', 'cut', 'cut_axis', 'supercell']
    Modifier.order += ['clayff']
    for attr in Modifier.order:
        setattr(Modifier, attr, getattr(args, attr))

    Modifier(infile)
    
    if args.run:
        sp.run(["open", Modifier.outfile])
    
if __name__  == "__main__":
    main()
 
