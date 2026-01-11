#!/usr/bin/env python
import numpy as np
import subprocess as sp
import sys
from cshmd.structure import DATA
from cshmd.analysis import find_bond, find_mole, unwrap_mole, df2charge, df2charge4clayff
from cshmd.functions import setMatrix, cal_uniform
from cshmd.util import molWeight

class Modifier():
    
    order = []
    convert = {}
    convert["c"] = "CO3"
    convert["w"] = "H2O"
    convert["ca"] = "Ca"
    avo = 6.0221408e+23
    verbose = False
    
    def __init__(self, infile):
        self.c = DATA(infile)
        self.c.verbose = Modifier.verbose
        lst = []
        lst.extend(np.nonzero(self.c.df["label"] == "co"))
        lst.extend(np.nonzero(self.c.df["label"] == "oc"))
        lst = list(np.hstack(lst))
        if hasattr(self, "outfile"):
            self.c.outfile = getattr(self, "outfile")
        bf_charge = df2charge(self.c.df)
        for o in Modifier.order:
            if getattr(Modifier, o) is not None:
                getattr(self, f"exec_{o}")()
        af_charge = df2charge(self.c.df)
        self.c.writeCif()
        print(f"Total charge: {bf_charge} -> {af_charge}")

    def exec_assemble(self):
        ciffile, axis = self.assemble
        axisDic = {"x": 0, "y": 1, "z": 2}
        index = axisDic[axis]
        c2 = DATA(ciffile)
        for i in range(2):
            if i == index:
                continue
            assert np.round(self.c.lattice[i], decimals=3) == np.round(c2.lattice[i], decimals=3), f"Error: lattice constant Mismatch, {self.c.lattice} and {c2.lattice}"
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
        c2.df[f"fract_{axis}"] += (self.c.lattice[index] / c2.lattice[index])
        c2.df[["fract_x", "fract_y", "fract_z"]] @= matrix_branch
        self.c.lattice[index] += (c2.lattice[index] * 1)
        assembled_matrix, _ = setMatrix(self.c.lattice, self.c.angle)
        self.c.df = pd.concat([self.c.df, c2.df], ignore_index=True)
        self.c.df[["fract_x", "fract_y", "fract_z"]] @= np.linalg.inv(assembled_matrix)
        print("Number of atoms: ", end="")
        print(f"{self.c.natom_init} + {c2.natom_init} = {self.c.natom_init + c2.natom_init}")


    def exec_clayff(self):
        if not self.clayff:
            return
        data = self.c.df[["type_symbol", "fract_x", "fract_y", "fract_z"]].to_numpy()
        OH_bond = find_bond(data, self.c.matrix, rcut=1.15)
        OH = find_mole(OH_bond, nbond=1)
        H2O = find_mole(OH_bond, nbond=2)
        CO3 = find_mole(find_bond(data, self.c.matrix, rcut=1.4, elem1="C", elem2="O"), nbond=3, elem1="C", elem2="O")
        SiO = find_bond(data, self.c.matrix, rcut=1.8, elem1="Si", elem2="O")
        if len(H2O) != 0:
            oo, hh = H2O[:, 0], np.unique(H2O[:, 1:])
            self.c.df.loc[oo, ["label"]] = "o*"
            self.c.df.loc[hh, ["label"]] = "h*"
        if len(CO3) != 0:
            co, oc = CO3[:, 0], np.unique(CO3[:, 1:])
            self.c.df.loc[co, ["label"]] = "co"
            self.c.df.loc[oc, ["label"]] = "oc"
        if len(SiO) != 0:
            st, ob = SiO[:, 0], np.unique(SiO[:, 1])
            self.c.df.loc[st, ["label"]] = "st"
            self.c.df.loc[ob, ["label"]] = "ob"
        if len(OH) != 0:
            oh, ho = OH[:, 0], OH[:, 1]
            self.c.df.loc[oh, ["label"]] = "oh"
            self.c.df.loc[ho, ["label"]] = "ho"
        mask_Ca = self.c.df["type_symbol"] == "Ca"
        layerCa = (self.c.df.loc[mask_Ca, ["fract_z"]] < 0.54) & (self.c.df.loc[mask_Ca, ["fract_z"]] > 0.02)
        cah = layerCa[layerCa["fract_z"]].index
        Ca = layerCa[~layerCa["fract_z"]].index
        self.c.df.loc[cah, ["label"]] = "cah"
        self.c.df.loc[Ca, ["label"]] = "Ca"
        self.c.df.loc[Ca, ["label"]] = "cah"
        clayff_label = ['Ca', 'cah', 'co', 'h*', 'ho', 'o*', 'ob', 'oc', 'st', "oh"]
        unique = np.unique(self.c.df["label"])
        isin = np.isin(unique, clayff_label)
        assert isin.all(), f"Error: Unlabeled atoms exists, {unique[~isin]}, {np.nonzero(self.c.df['label'] == 'O')}"
        charge, charge_df = df2charge4clayff(self.c.df, return_charge=True)
        print(f"Clayff charge: {charge:.3f}")
        if charge != 0:
            print(charge_df)
        del charge_df

        
        
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
                    Modifier.delete = [bf]
                else:
                    Modifier.delete.append(bf)
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
            elif s[0].isdigit():
                index = int(s)
                flag = flag & (self.c.df.index == index)
            elif s[0] == "l":
                _, label = s.split("-")
                flag = flag & (self.c.df["label"] == label)
        print("Search Result :\n", self.c.df[flag])
        print("Index", " ".join(str(s) for s in self.c.df[flag].index))
        sys.exit(0)
    
    def exec_random_coord(self):
        kwargs = {}
        for coord in self.random_coord:
            axis, *range_ = coord.split("-", 2)
            kwargs[f"range_{axis}"] = [float(s) for s in range_]
        self.c.min_query(rcut=2.5, **kwargs)
        sys.exit(0)

    def exec_random_insert(self):
        kwargs = {}
        mole_ = {}
        for s in self.random_insert:
            if s[0] in  ["x", "y", "z"]:
                if s.count("-") == 2:
                    axis, *range_ = s.split("-", 2)
                    kwargs[f"range_{axis}"] = [float(v) for v in range_]
                if s.count("-") == 1:
                    axis, range_ = s.split("-")
                    kwargs[f"range_{axis}"] = [float(range_)]
            elif s.count("-") == 1:
                mole, num = s.split("-")
                mole = mole if mole not in ["w", "c", "ca"] else Modifier.convert[mole]
                mole_[mole] = int(num)
            else:
                raise ValueError(f"Error: Wrong format, {' '.join(str(r) for r in self.random_insert)}")
                
        assert len(mole_.items()) != 0, "Error: No molecule input"
        for m, n in mole_.items():
            add_count = 0
            try_count = 0
            try_max = 100
            while True:
                coord = self.c.min_query(rcut=2.5, **kwargs, return_coord=True)
                if coord is not None:
                    print(coord)
                    self.c.insert(coord, symbol=m)
                    add_count += 1
                else:
                    try_count += 1
                if add_count == n:
                    print(f"Added {n} mol of {m}")
                    break
                if try_count == try_max:
                    print(f"Only can added {add_count}/{n} mol of {m}")
                    break
                
    def exec_protonation(self):
        for s in self.protonation:
            self.c.protonation(s)
                    
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
        self.c.cut(shrink=Modifier.shrink, **range_dic)

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
                if Modifier.shrink:
                    r_range = [v / ratio for v in r_range]
                    ratio *= 1 - (r_range[1] - r_range[0])
                dic_ = {axis: r_range}
                self.c.cut(shrink=Modifier.shrink, **dic_)
                

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
    
    def exec_make_interface(self, **kwargs): #ckpt_interface
        include_start = [True, False, True]
        include_end = [False, True, True]
        weight_z = 0.9
        weight_y = 1.2
        weight_x = 1.2
        density = 1
        # for key in ["density", "include_start", "include_end",
        #             "weight_z", "weight_y", "weight_x"]:
        #     locals()[key] = kwargs.get("density", f"d_{key}")
        axis = "z"
        axisDic = {"x": 0, "y": 1, "z": 2}
        self.c.verbose = False
        offset = 0
        convert = Modifier.convert
        mole_dic = {}
        molWeight_sum = 0
        self.mole_sum = 0
        before_natom = self.c.natom
        data = self.c.df[["type_symbol", "fract_x", "fract_y", "fract_z"]].to_numpy()
        OH = find_bond(data, self.c.matrix)
        H2O = find_mole(OH)
        CO = find_bond(data, self.c.matrix, elem1="C", elem2="O", rcut=1.4)
        CO3 = find_mole(CO, nbond=3)
        SiO = find_bond(data, self.c.matrix, elem1="Si", elem2="O", rcut=1.8)
        SiO4 = find_mole(SiO, nbond=4)
        for mole in [H2O, CO3, SiO4]:
            for m in mole:
                xyz = self.c.df.loc[m, ["fract_x", "fract_y", "fract_z"]].to_numpy()
                xyz = unwrap_mole(xyz, self.c.matrix)
                self.c.df.loc[m, ["fract_x", "fract_y", "fract_z"]] = xyz
                
        for s in self.make_interface:
            if s in ["x", "y", "z"]:
                axis = s
                continue
            mole, num = s.split("-")
            mole = mole if mole not in ["w", "c", "ca"] else convert[mole]
            mole_dic[mole] = int(num)
            if mole == "H2O":
                molWeight_sum += int(num) * molWeight[mole]
            self.mole_sum += int(num)
        molWeight_sum /= Modifier.avo
        V = molWeight_sum / density * 1e24
        index = axisDic[axis]
        extend_c = V / self.c.V
        for i in range(3):
            if i == index:
                continue
            extend_c /= self.c.lattice[index]
        extend_c += offset
        lattice = self.c.lattice.copy()
        lattice[index] = extend_c
        matrix, V = setMatrix(self.c.lattice, self.c.angle)
        grid = cal_uniform(matrix, self.mole_sum,
                           spacing=Modifier.spacing,
                           include_start=include_start,
                           include_end=include_end,
                           weight_z=weight_z, weight_y=weight_y, weight_x=weight_x)
        arange = np.arange(0, grid.shape[0])
        print("MODE: ", end="")
        if hasattr(Modifier, "mode1"):
            print("mode1")
            mask_mode1 = (grid[:, 2] > 0.1) & (grid[:, 2] < 0.5)
            mask_mode1 = np.nonzero(mask_mode1)[0]
            size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 4
            setted_indices = mask_mode1[np.random.choice(len(mask_mode1), size=size, replace=False)]
            unsetted_indices = np.setdiff1d(arange, setted_indices) 
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()
            
        if hasattr(Modifier, "mode2"):
            print("mode2")
            mask_mode2 = (grid[:, 2] > 0.1) & (grid[:, 2] < 0.9 )
            mask_mode2 = np.nonzero(mask_mode2)[0]
            size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 10
            setted_indices = mask_mode2[np.random.choice(len(mask_mode2), size=size, replace=False)]
            unsetted_indices = np.setdiff1d(arange, setted_indices)
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()

        if hasattr(Modifier, "mode3"):
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
            Ca_indices = mask_mode_Ca[np.random.choice(len(mask_mode_Ca), size=mole_dic.get("Ca", 0), replace=False)]
            CO3_indices = mask_mode_CO3[np.random.choice(len(mask_mode_CO3), size=mole_dic.get("CO3", 0), replace=False)]
            setted_indices = np.concatenate([Ca_indices, CO3_indices])
            unsetted_indices = np.setdiff1d(arange, setted_indices)
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()

        if hasattr(Modifier, "mode4"):
            print("mode4")
            mask_mode1 = (grid[:, 2] > 0.1) & (grid[:, 2] < 0.45)
            mask_mode1 = np.nonzero(mask_mode1)[0]
            size = mole_dic.get("CO3", 0) + mole_dic.get("Ca", 0) + 2
            setted_indices = mask_mode1[np.random.choice(len(mask_mode1), size=size, replace=False)]
            unsetted_indices = np.setdiff1d(arange, setted_indices) 
            setted_mole = ["CO3", "Ca"]
            assert not np.isin(setted_indices, unsetted_indices).any()
            
        grid[:, index] += self.c.lattice[index] / extend_c
        grid[:, index] /= (1 + self.c.lattice[index] / extend_c)
        abc = [0, 0, 0]
        abc[index] = extend_c
        self.c.extendLattice(abc=abc)
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
                    if self.c.verbose:
                        print(f"\r{mole} {_ + 1} / {num}", end="", flush=True)
        print(f"\nNumber of atoms: {before_natom} -> {len(self.c.df)}")

    def exec_auto_cut_axis(self):
        axis, cut_len = self.auto_cut_axis.split("-")
        pos_z = self.c.df[f"fract_{axis}"]
        axisDic = {"x": 0, "y": 1, "z": 2}
        index = axisDic[axis]
        cut_len = float(cut_len)
        range_z = cut_len / self.c.lattice[index]
        start_z = 0
        was_vaccum = False
        while True:
            end_z = start_z + range_z
            is_vaccum = all((pos_z < start_z) | (pos_z > end_z))
            if is_vaccum:
                was_vaccum = True
                print("Cutting Vaccuum area")
                kwargs = {axis: (start_z, end_z)}
                self.c.cut(shrink=True, **kwargs)
#                self.c.cut(shrink=True, z=(start_z, end_z))
                assert self.c.cut_natom == 0
                range_z = cut_len / self.c.lattice[index]
                pos_z = self.c.df[f"fract_{axis}"]
            if end_z >= 0.99:
                if not was_vaccum:
                    raise ValueError("Not any vaccum")
                else:
                    break
            start_z += 0.0001


def main():
    import argparse
    import glob
    description = """
    This tool is used to modify CIF files and POSCAR.
    Make sure to file's in the current directory are named with the extension '.cif' or 'POSCAR'.
    """.strip()
    
    par = argparse.ArgumentParser(description=description, prog="Modifier")
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

    infile = args.infile if args.infile else glob.glob("*cif")[0]
    
    if args.outfile:
        base = args.outfile.replace(".cif", "")
        Modifier.outfile = base + ".cif"
    else:
        Modifier.outfile = "modified" + infile
        
    Modifier.order = ["search", "make_interface", "assemble", "random_coord"]
    Modifier.order += ['sub', 'insert', 'protonation', 'auto_protonation',
                       'random_insert', 'adjust', 'auto_cut_axis', 'extend_vaccum',
                       'extend_lattice', 'delete', 'deformation', 'sub_column',
                       'cut', 'cut_axis', 'supercell', 'clayff']

    for attr in Modifier.order:
        setattr(Modifier, attr, getattr(args, attr))
        
    for attr in ("verbose", "spacing", "shrink"):
        setattr(Modifier, attr, getattr(args, attr))

    if args.mode != 0:
        setattr(Modifier, f"mode{args.mode}", True)

    Modifier(infile)
    
    if args.run:
        sp.run(["open", Modifier.outfile])
    
if __name__  == "__main__":
    main()
