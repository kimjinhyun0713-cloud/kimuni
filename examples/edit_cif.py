#!/usr/bin/env python
import argparse
import numpy as np
import glob
import re
import pandas as pd
import os, sys
from jinh.analysis import find_bond, find_mole, unwrap_mole
from jinh.functions import setMatrix, cal_uniform
from jinh import cif2data, cif_head, cif_tail
from jinh import molDic, molWeight
import subprocess as sp
from scipy.spatial import KDTree, cKDTree

class CIF():
    
    infile = None
    rcut = None
    molecule = molDic
        
    def __init__(self, infile):        
        self.infile = infile
        print(f"{infile} was loaded ... ")
        self.lattice, self.angle, self.df = cif2data(self.infile)
        self.columns = self.df.columns
        self.natom, self.natom_init = len(self.df), len(self.df)
        print("Number of atoms", self.natom_init)
        self.matrix, self.V = setMatrix(self.lattice, self.angle)
            
                    
    def min_query(self, npoint=100, rcut=2, **kwargs):
        """
        to provide random coordinations in cutoff "rcut"

        args
        ;; npoint -> int, number of npoint for query
        ;; rcut -> radius cutoff
        return
        ;; None, print fraction to stdout
        """
        fract = self.df[["fract_x", "fract_y", "fract_z"]]
        fract = fract.to_numpy().reshape(-1, 3)
        shifts = np.array([[i, j, k] for i in [-1,0, 1]
                           for j in [-1,0, 1]
                           for k in [-1,0, 1]])
        all_ = (fract[:, None, :] + shifts[None, :, :])
        all_ = all_.reshape(-1, 3) @ self.matrix
        tree = cKDTree(all_)
        fract_points = np.random.rand(npoint, 3)
        for i, axis_range in enumerate(["range_x", "range_y", "range_z"]):
            range_ = kwargs.get(axis_range, None)
            if range_ is not None:
                fract_points[:, i] = fract_points[:, i] * (range_[1] - range_[0]) + range_[0]
        cart_points = fract_points @ self.matrix
        r_min, _ = tree.query(cart_points)
        isolated_points = cart_points[r_min >= rcut] @ np.linalg.inv(self.matrix)
        if self.verbose == 1:
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
            # if np.max(r) > 1.0:
            #     print(np.max(r))
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
                print(f"Delete symbol: {self.df.at[i, 'type_symbol']}, idx: {i}")
                self.df.drop(i, axis=0, inplace=True)
        if label:
            for l in label:
                flag = self.df["label"].str.lower() == l.lower()
                matching_idx = self.df[flag].index
                for i in matching_idx:
                    print(f"Delete symbol: {self.df.at[i, 'type_symbol']}, label: {l}")
                self.df.drop(matching_idx, axis=0, inplace=True)
        if symbol:
            for s in symbol:
                flag = self.df["symbol"].str.lower() == s.lower()
                matching_idx = self.df[flag].index
                for i in matching_idx:
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
                if self.verbose == 0:
                    output += f"Warning! ({x:.3f}, {y:.3f}, {z:.3f}) label: {label} symbol: {symbol}\n"
                    output += "Has some near neighbor atoms\n"
                for arr in check:
                    if self.verbose == 0:
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
            if self.verbose >= 1:
                output += f"Updated new molecule: ({x:.3f}, {y:.3f}, {z:.3f}) label: {label} symbol: {symbol}\n"
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
        if self.verbose == 1:
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
        iso = [r for r in self.df.columns if "iso" in r][0]
        for _, row in self.df.iterrows():
            line = cif_tail.format(
                str(row['label']),
                float(row['occupancy']),
                float(row['fract_x']),          
                float(row['fract_y']),          
                float(row['fract_z']),
                str(row['adp_type']),
                float(row[iso]),     
                str(row['type_symbol']),     
            )
            lines += f"{line}\n"

        template = cif_head.format(self.infile, *self.lattice, *self.angle)
        template += "\n"
        template += lines
        base, ext = os.path.splitext(self.infile)
        version = 0
        if hasattr(self, "outfile"):
            RUN.outfile = self.outfile
        else:
            while True:
                RUN.outfile = f"{base}.{version:02d}{ext}"
                if not os.path.exists(RUN.outfile):
                    break
                version += 1
                
        with open(RUN.outfile, "w") as o:
            o.write(template)
            print(f"{RUN.outfile} was created")



class RUN():
    
    order = []
    convert = {}
    convert["c"] = "CO3"
    convert["w"] = "H2O"
    convert["ca"] = "Ca"
    avo = 6.0221408e+23

    
    def __init__(self, infile):
        self.c = CIF(infile)
        self.c.verbose = RUN.verbose
        lst = []
        lst.extend(np.nonzero(self.c.df["label"] == "co"))
        lst.extend(np.nonzero(self.c.df["label"] == "oc"))
        lst = list(np.hstack(lst))
        if hasattr(self, "outfile"):
            self.c.outfile = getattr(self, "outfile")
            
        for o in RUN.order:
            if getattr(RUN, o) is not None:
                getattr(self, f"exec_{o}")()

        self.c.writeCif()

    def exec_assemble(self):
        c2 = CIF(self.assemble)
        print(f"assemble {self.assemble}")
        if all(np.round(self.c.lattice[0:2], decimals=3) != np.round(c2.lattice[0:2], decimals=3)):
            print(f"Error: lattice constant Mismatch, {self.c.lattice[0:2]} and {c2.lattice[0:2]}")       
        matrix_branch, _ = setMatrix(c2.lattice, c2.angle)
        data = c2.df[["type_symbol", "fract_x", "fract_y", "fract_z"]].to_numpy()
        OH = find_bond(data, matrix_branch)
        H2O = find_mole(OH)
        CO = find_bond(data, matrix_branch, elem1="C", elem2="O", rcut=1.4)
        CO3 = find_mole(CO, nbond=3)
        for mole in [H2O, CO3]:
            for m in mole:
                xyz = c2.df.loc[m, ["fract_x", "fract_y", "fract_z"]].to_numpy()
                xyz = unwrap_mole(xyz, matrix_branch)
                c2.df.loc[m, ["fract_x", "fract_y", "fract_z"]] = xyz
        self.c.df[["fract_x", "fract_y", "fract_z"]] @= self.c.matrix
        c2.df["fract_z"] += (self.c.lattice[2] / c2.lattice[2])
        c2.df[["fract_x", "fract_y", "fract_z"]] @= matrix_branch
        self.c.lattice[2] += (c2.lattice[2] * 1)
        assembled_matrix, _ = setMatrix(self.c.lattice, self.c.angle)
        self.c.df = pd.concat([self.c.df, c2.df], ignore_index=True)
        self.c.df[["fract_x", "fract_y", "fract_z"]] @= np.linalg.inv(assembled_matrix)
        print("Number of atoms: ", end="")
        print(f"{self.c.natom_init} + {c2.natom_init} = {self.c.natom_init + c2.natom_init}")
        
    def exec_sub(self):
        for s in self.sub:
            largs = s.count("-")
            assert largs == 1 or largs == 2, "Wrong argument, Please check '-'"
            if largs == 1:
                bf, af = s.split("-", 1)
                xyz = self.c.df.iloc[int(bf), 2:5].to_numpy()
                # self.c.substitute(index=int(bf), after=af)
                self.c.insert(xyz, symbol=af)
                if hasattr(self, "delete"):
                    RUN.delete = [bf]
                else:
                    RUN.delete.append(bf)
            elif largs == 2:
                bf, af, label = s.split("-", 2)
                self.c.substitute(index=int(bf), after=af, label=label)

    def exec_sub_column(self):
        for s in self.sub_column:
            assert s.count("-") == 2, "Wrong argument, Please check '-'"
            bf, af, col = s.split("-", 3)
            self.c.substitute(before=bf, after=af, column=col)
            
    def exec_insert(self):
        for s in self.insert:
            largs = s.count("-")
            assert largs == 3 or largs == 4, "Wrong argument, Please check '-'"
            if largs == 3:
                x, y, z, elem = s.split("-", 4)
                updated = self.c.insert((float(x), float(y), float(z)), symbol=elem)
            elif largs == 4:
                x, y, z, symbol, label = s.split("-", 5)
                updated = self.c.insert((float(x), float(y), float(z)), symbol=symbol, label=label)
        if not updated:
            sys.exit(0)

    def exec_search(self):
        flag = np.ones(len(self.c.df), dtype=bool)
        for s in self.search:
            if s[0] in ["x", "y", "z"]:
                axis, *range_ = s.split("-", 2)
                flag = flag & (self.c.df[f"fract_{axis}"] > float(range_[0]))
                flag = flag & (self.c.df[f"fract_{axis}"] < float(range_[1]))
            elif s[0] == "e":
                elem, mole = s.split("-")
                flag = flag & (self.c.df["type_symbol"] == mole)
        print("Search Result : ", self.c.df[flag])
        print("Index", " ".join(str(s) for s in self.c.df[flag].index))
        sys.exit(0)
    
    def exec_random_coord(self):
        kwargs = {}
        for coord in self.random_coord:
            axis, *range_ = coord.split("-", 2)
            kwargs[f"range_{axis}"] = [float(s) for s in range_]
        self.c.min_query(rcut=2.5, **kwargs)
        print()
        sys.exit(0)
        

    def exec_protonation(self, direct=False):
        before_natom = len(self.c.df)
        for s in self.protonation:
            if direct:
                cartesian = s
            else:
                fract = self.c.df.loc[s, ["fract_x", "fract_y", "fract_z"]]
                cartesian = fract @ self.c.matrix

            if fract["fract_z"] > 0.4:
                add_iter = iter([[0, 0, 1], [0 , 0, -1], [0, 1, 0]])
            else:
                add_iter = iter([[0, 0, -1], [0 , 0, 1], [0, 1, 0]])
            # add_iter = iter([[0, 0, -1], [0 , 0, 1], [0, 1, 0]])
            add = next(add_iter)
            new_fraction = (cartesian + add) @ np.linalg.inv(self.c.matrix)
            while True:
#                proper = self.c.proper_location(new_fraction, 1.1, exception=s)
                proper = True
                if proper:
                    self.c.insert(new_fraction, symbol="H", label="ho")
                    break
                else:
                    try:
                        add = next(add_iter)
                    except StopIteration:
                        print("improper Site", f"Index: {s}")
                        break
                    new_fraction = (cartesian + add) @ np.linalg.inv(self.c.matrix)
        print(f"\nNumber of atoms: {before_natom} -> {len(self.c.df)}")
        self.c.natom = len(self.c.df)
                    
    def exec_auto_protonation(self):
        if not self.auto_protonation:
            return
        flag_Si = self.c.df["type_symbol"] == "Si"
        fract_Si = self.c.df.loc[flag_Si, ["fract_z"]]
        flag_O = self.c.df["type_symbol"] == "O"
        fract_O = self.c.df.loc[flag_O, ["fract_z"]]
        flag_O = fract_O < 0.32
        nonzero = np.nonzero(flag_O)[0]
        fract_O.index[nonzero]
#        nonzero = np.nonzero(flag_O)[0]
        self.protonation = fract_O.index[nonzero]
        print("Number of Auto Protonation Site: ", self.protonation.shape[0])
        self.exec_protonation()
        
                
    def exec_delete(self):
        index_list = []
        for d in self.delete:
            largs = d.count("-")
            assert largs == 0 or largs == 1, "Wrong argument, Please check '-'"
            if largs == 0:
                index_list.append(int(d))
            if largs == 1:
                start_num, end_num = [int(s) for s in d.split("-")]
                index_list.extend(list(range(end_num, start_num - 1, -1)))
        index_list = sorted(index_list, reverse=True)
        if len(index_list) != len(list(set(index_list))):
            doubled = []
            for d in index_list:
                if index_list.count(d) != 1:
                    doubled.append(str(d))
            doubled = list(set(doubled))
            string = " ".join(doubled)
            raise ValueError(f"Same index {string}")
        self.c.delete(index=index_list)
        
    def exec_extend_lattice(self):
        if self.extend_lattice == [0, 0, 0]:
            return
        assert all(x >= 0 for x in self.extend_lattice)
        self.c.extendLattice(abc=self.extend_lattice)
        
        
    def exec_adjust(self):
        for ad in self.adjust:
            assert ad.count("-") == 3, "Wrong argument, Please check '-'"
            x, y, z, idx = ad.split("-")
            x = x.replace("@", "-")
            y = y.replace("@", "-")
            z = z.replace("@", "-")
            idx = int(idx)
            x = None if x == "0" else float(x)
            y = None if y == "0" else float(y)
            z = None if z == "0" else float(z)
            self.c.adjust(idx=idx, x=x, y=y, z=z)

            
    def exec_supercell(self):
        if self.supercell == [1, 1, 1]:
            return
        self.c.makeSupercell(self.supercell)

        
    def exec_deformation(self):
        if self.deformation == [1] or self.deformation == [1, 1, 1]:
            return
        if len(self.deformation) == 1:
            self.deformation *= 3
        elif len(self.deformation) != 3:
            raise ValueError("nargs must to be 1 or 3")
        self.c.deformation(self.deformation)                                               


    def exec_cut(self):
        if len(self.cut) == 3:
            range_x, range_y, range_z = map(lambda x: tuple(map(float, x.split("-"))), self.cut)
        else:
            pass
        range_dic = {}
        for axis in ["x", "y", "z"]:
            key = f"range_{axis}"
            if len(locals()[key]) != 1:
                range_dic[axis] = locals()[key]
        self.c.cut(shrink=RUN.shrink, **range_dic)

    def exec_cut_axis(self):
        dic = dict()
        for c_axis in ["x", "y", "z"]:
            if c_axis in self.cut_axis:
                assert self.cut_axis.count(c_axis) == 1, f"Error: Doubled Axis, {c_axis}"
                c_axis_idx = self.cut_axis.index(c_axis)
                idx = c_axis_idx + 1
                lst = list()
                while True:
                    if idx == len(self.cut_axis) or self.cut_axis[idx] in ["x", "y", "z"]:
                        break
                    else:
                        assert self.cut_axis[idx].count("-") == 1, f"Error: args format mismatch {self.cut_axis[idx]}"
                        lst.extend(sorted(self.cut_axis[idx].split("-")))
                        dic[c_axis] = lst
                        idx += 1
        for axis, vals in dic.items():
            ratio = 1
            vals = sorted(vals, reverse=True)
            for i in range((len(vals) // 2)):
                r_range = [float(v) for v in vals[2 * i: 2 * i + 2]][::-1]
                if RUN.shrink:
                    r_range = [v / ratio for v in r_range]
                    ratio *= 1 - (r_range[1] - r_range[0])
                dic_ = {axis: r_range}
                self.c.cut(shrink=RUN.shrink, **dic_)
                

    def exec_extend_vaccum(self):
        assert len(self.extend_vaccum) <= 3, "Error: nargs must be <= 3"
        dic = dict()
        abc = [0, 0, 0]
        index = {"x": 0, "y": 1, "z": 2}
        for s in self.extend_vaccum:
            axis, key, val = s.split("-")
            dic[axis] = float(key)
            abc[index[axis]] = float(val)
        self.c.extendLattice(abc=abc, key=dic)

    def exec_make_interface(self): #ckpt_interface
        self.c.verbose = 2
        density = 0.5
        offset = 0
        convert = RUN.convert
        mole_dic = {}
        molWeight_sum = 0
        self.mole_sum = 0
        before_natom = self.c.natom
        for s in self.make_interface:
            mole, num = s.split("-") 
            mole = mole if mole not in ["w", "c", "ca"] else convert[mole]
            mole_dic[mole] = int(num)
            if mole == "H2O":
                molWeight_sum += int(num) * molWeight[mole]
            self.mole_sum += int(num)
        molWeight_sum /= RUN.avo # [g]
        V = molWeight_sum / density * 1e24
        extend_c = V / (self.c.V * self.c.lattice[0] * self.c.lattice[1])
        extend_c += offset
        lattice = self.c.lattice.copy()
        lattice[2] = extend_c
        matrix, V = setMatrix(self.c.lattice, self.c.angle)
        grid = cal_uniform(matrix, self.mole_sum, spacing=RUN.spacing, include_end=[True, True, False], weight_z=1.5)
        arange = np.arange(0, grid.shape[0])
        print("MODE: ", end="")
        if hasattr(RUN, "mode1"):
            print("mode1")
            mask_mode1 = (grid[:, 2] > 0.1) & (grid[:, 2] < 0.5)
            mask_mode1 = np.nonzero(mask_mode1)[0]
            size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 4
            setted_indices = mask_mode1[np.random.choice(len(mask_mode1), size=size, replace=False)]
            unsetted_indices = np.setdiff1d(arange, setted_indices) 
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()
            
        if hasattr(RUN, "mode2"):
            print("mode2")
            mask_mode2 = (grid[:, 2] > 0.1) & (grid[:, 2] < 0.9 )
            mask_mode2 = np.nonzero(mask_mode2)[0]
            size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 10
            setted_indices = mask_mode2[np.random.choice(len(mask_mode2), size=size, replace=False)]
            unsetted_indices = np.setdiff1d(arange, setted_indices)
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()

        if hasattr(RUN, "mode3"):
            print("mode3")
            mask_mode_c = (grid[:, 2] > 0.4) & (grid[:, 2] < 0.6)
            # mask_mode_a = (grid[:, 1] > 0.1) & (grid[:, 1] < 0.3)
            # mask_mode_b = (grid[:, 0] > 0.1) & (grid[:, 0] < 0.3)
            # mask_mode_aa = (grid[:, 1] > 0.6) & (grid[:, 1] < 0.8)
            # mask_mode_bb = (grid[:, 0] > 0.6) & (grid[:, 0] < 0.8)
            mask_mode_a = (grid[:, 1] > 0.6) & (grid[:, 1] < 0.8)
            mask_mode_b = (grid[:, 0] > 0.6) & (grid[:, 0] < 0.8)
            mask_mode_aa = (grid[:, 1] > 0.1) & (grid[:, 1] < 0.3)
            mask_mode_bb = (grid[:, 0] > 0.1) & (grid[:, 0] < 0.3)
            mask_mode_Ca = np.nonzero(mask_mode_a & mask_mode_b & mask_mode_c)[0]
            mask_mode_CO3 = np.nonzero(mask_mode_aa & mask_mode_bb & mask_mode_c)[0]
            # size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 5
            Ca_indices = mask_mode_Ca[np.random.choice(len(mask_mode_Ca), size=mole_dic.get("Ca", 0) + 2, replace=False)]
            CO3_indices = mask_mode_CO3[np.random.choice(len(mask_mode_CO3), size=mole_dic.get("CO3", 0) + 2, replace=False)]
            setted_indices = np.concatenate([Ca_indices, CO3_indices])
            unsetted_indices = np.setdiff1d(arange, setted_indices)
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()

        if hasattr(RUN, "mode4"):
            print("mode4")
            mask_mode1 = (grid[:, 2] > 0.1) & (grid[:, 2] < 0.75)
            mask_mode1 = np.nonzero(mask_mode1)[0]
            size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 2
            setted_indices = mask_mode1[np.random.choice(len(mask_mode1), size=size, replace=False)]
            unsetted_indices = np.setdiff1d(arange, setted_indices) 
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()
            
        grid[:, 2] += self.c.lattice[2] / extend_c
        grid[:, 2] /= (1 + self.c.lattice[2] / extend_c)
        self.c.extendLattice(abc=[0, 0, extend_c])
        order = ["CO3", "Ca", "H2O"]
        not_selected = np.arange(0, grid.shape[0]).astype(int)
        for mole in order:
            if mole in mole_dic.keys():
                num = mole_dic[mole]
                for _ in range(num):
                    while True:
                        if mole == "Ca" and "Ca_indices" in locals().keys():
                            flag_random = np.intersect1d(Ca_indices, not_selected)
                        elif mole == "CO3" and  "CO3_indices" in locals().keys():
                            flag_random = np.intersect1d(CO3_indices, not_selected)
                        elif "setted_mole" in locals().keys():
                            flag_random = np.intersect1d(setted_indices, not_selected) if mole in locals().get("setted_mole", list()) else np.intersect1d(unsetted_indices, not_selected)
                        else:
                            flag_random = np.intersect1d(arange, not_selected)
                        index = np.random.choice(flag_random, 1)
                        coord = grid[index]
                        # print(coord, index)
                        inserted = self.c.insert(tuple(coord[0]), symbol=mole) 
                        if inserted:
                            not_selected = not_selected[not_selected != index]
                            break
                    print(f"\r{mole} {_ + 1} / {num}", end="", flush=True)
        print(f"\nNumber of atoms: {before_natom} -> {len(self.c.df)}")

        
        
        
        
    def exec_cut_vaccum(self):
        pos_z = self.c.df["fract_z"]
        cut_len = self.cut_vaccum
        range_z = cut_len / self.c.lattice[2]
        start_z = 0
        was_vaccum = False
        while True:
            end_z = start_z + range_z
            is_vaccum = all((pos_z < start_z) | (pos_z > end_z))
            if is_vaccum:
                was_vaccum = True
                print("Cutting Vaccuum area")
                self.c.cut(shrink=True, z=(start_z, end_z))
                assert self.c.cut_natom == 0
                range_z = cut_len / self.c.lattice[2]
                pos_z = self.c.df["fract_z"]
            if end_z >= 0.99:
                if not was_vaccum:
                    raise ValueError("Not any vaccum")
                else:
                    break
            start_z += 0.0001

            
def main():
    description = """
    To modify a CIF file for system constructino
    """.strip()
    
    par = argparse.ArgumentParser(description=description, prog="CIF editor")
    par.add_argument(
        'infile', nargs="?",
        help="Input CIF file (if not provided, it will be selected automatically)")
    par.add_argument(
        '--verbose', choices=[0, 1, 2], type=int, default=1,
        help="set verbosity level by number")
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
        '-cv', '--cut_vaccum', type=float,
        help="Remove voids within rcut along the c-axis: <rcut>")
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
        '-as', '--assemble', default=None,
        help="Assemble another CIF file, when lattice of axis-a axis-b is same <hoge.cif>")
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
        '--shrink', default=False, action="store_true",
        help="(This argument is used only when the 'cut or cut_axis' is enabled)\n"
        "The system is shrunk after performing 'cut' or 'cut_axis'.")
    par.add_argument(
        '--search', nargs="+",
        help="Search for atoms that satisfy the specified condition (element type or location): <'e'-element or axis-coord-coord>...\n"
        "e.g. e-Ca x-0.1-0.5 z-0.5-0.8")
    par.add_argument(
        '--spacing', default=3.5, type=float,
        help="(This argument is used only when the 'make_interface' is enabled)\n"
        "make grids that are at least 2.5 angstrom away from other atoms")
    par.add_argument(
        '--mode', default=0, type=int,
        help="(This argument is used only when the 'make_interface' is enabled)\n"
        "Set the condition for a specific element or the location of a molecule")
    args = par.parse_args()

    RUN.verbose = args.verbose
    RUN.spacing = args.spacing
    RUN.shrink = args.shrink
    if args.mode != 0:
        setattr(RUN, f"mode{args.mode}", True)

    infile = args.infile if args.infile else glob.glob("*cif")[0]
    if args.outfile:
        base = os.path.splitext(args.outfile)[0]
        RUN.outfile = base + ".cif"
        
    RUN.order = ["search", "make_interface", "assemble", "random_coord"]
    RUN.order += ['sub', 'insert', 'protonation', 'auto_protonation', 'adjust', 'cut_vaccum', 'extend_vaccum', 'extend_lattice', 'delete', 'deformation', 'sub_column', 'cut', 'cut_axis', 'supercell']
    for attr in RUN.order:
        setattr(RUN, attr, getattr(args, attr))
        
    RUN(infile)
    
    if args.run:
        sp.run(["open", RUN.outfile])
    
if __name__  == "__main__":
    main()
