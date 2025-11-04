#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import os, sys
import glob
import time
import subprocess as sp
import itertools
import re
import pickle
from jinh import lmp2Dic, calLattice
from jinh.common import overwritePrint, checkTime
from jinh.util import lmp_head_unwrapped


class COORDINATION():
    
    dname = None

    @classmethod
    def initSetting(cls):
        filename = "ca.log"
        filebase = filename
        num = 0
        while True:
            if os.path.exists(filename):
                print(f"{filename} already exists")
                filename = filebase + f"{num:02}"
                num += 1
            else:
                break
        with open(os.path.join(cls.dname, filename), "w") as o:
            ca_id = " ".join(str(a) for a in cls.ca_id)
            o.write(f"Ca id: {ca_id}\n")
            setattr(cls, "ca_log", "")
            print("writing Ca log ...")
        for c in cls.carbonate:
            setattr(cls, f"{c}_num", 0)
            setattr(cls, "CN", 0)
            
    clusters = []
    Ca_C = []
    Ca_O_C = []
    nstep = 0
    clusterLog = ""
    carbonate = ["CO3", "HCO3", "H2CO3", "H3CO3"]
    carbonate.extend(["CO3bi", "CO3tri"])
    carbonate.extend(["HCO3bi", "HCO3tri"])
    carbonate.extend(["H2CO3bi", "H2CO3tri"])
    carbonate.extend(["H3CO3bi", "H3CO3tri"])
    carbonate.extend(["layerSiO4bi", "layerSiO4tri"])
    carbonate.extend(["layerSiO4quad", "layerSiO4penta"])
    cabonate = carbonate.copy()
    hydrate = ["OH", "H2O", "H3O", "layerSiO4", "CaOH", "CaO"]
    carbonate.extend(hydrate)

    
    @classmethod
    def outputCaLog(cls):
        output = f"Step: {COORDINATION.nstep}\n"
        for l in cls.clusters:
            output += f"{l.stat}\n"
        output += "\n"
        with open(os.path.join(cls.dname, "ca.log"), "a") as o:
            o.write(output)

    @classmethod
    def dataFrame2Excel(cls):
        with pd.ExcelWriter("angle.xlsx") as o:
            cls.df.to_excel(o, index=False, sheet_name="ang_dist")
            print("'angle.xlsx' was created")

    @classmethod
    def angle2npz(cls):
        np.savez("trace_angle.npz", Ca_C=cls.Ca_C, Ca_O_C=cls.Ca_O_C)
        print("trace_angle.npz is created")

    def resetSymbol(self):
        for c in self.carbonate:
            setattr(self, c, list())
    
    def __init__(self):
        self.Ca = None
        self.z_Ca = None
        self.CN = 0
        self.resetSymbol()
        for c in self.carbonate:
            setattr(self, f"{c}_num", 0)
        COORDINATION.clusters.append(self)

    def convertToArr(self):
        for c in self.carbonate:
            attrLst = getattr(self, c)
            assert isinstance(attrLst, list)
            if len(attrLst) != 0:
                attrArr = np.vstack(attrLst)
                setattr(self, c, attrArr)
            else:
                setattr(self, c, None)
                
    def outputStat(self):
        self.stat = f"id: {self.Ca}"
        for c in self.carbonate:
            count = getattr(self, f"{c}_num")
            if count != 0:
                self.stat += f", {c}: {count:.3f}"
            setattr(self, f"{c}_num", 0)
        self.stat += f", CN: {self.CN:.3f}"
        # print(self.stat)

    def whichBlock(self, posZ):
        zeropoint = self.__class__.zeropoint
        boundary1 = 12.3 + zeropoint
        boundary2 = 15.3 + zeropoint
        boundary3 = 20 + zeropoint
        if (posZ <= boundary1):
            return "region1"
        elif (posZ <= boundary2):
            return "region2"
        elif (posZ <= boundary3):
            return "region3"
        else:
            return None
        
    def cnStat(self):
        self.convertToArr()
        self.CN = self.idx_O.shape[0]
        COORDINATION.CN += self.idx_O.shape[0]
        for c in self.carbonate:
            attr = getattr(self, c)
            if attr is not None:
                if c == "CO3bi":
                    coeff = 2
                elif c == "CO3tri":
                    coeff = 3
                else:
                    coeff = 1
                count = attr.shape[0] * coeff
                bf_count_cls = getattr(COORDINATION, f"{c}_num")
                bf_count_cls += count
                setattr(self, f"{c}_num", count)
                setattr(COORDINATION, f"{c}_num", bf_count_cls)
        self.outputStat()
        self.resetSymbol()

        
class LAMMPSTRJ():
    
    infile = None
    show_init = False
    output = False
    fmt = lmp_head_unwrapped
    mole_num = {"C": 1, "Ca": 2, "H": 3, "O": 4, "Si": 5, "I": 6}
    
    def __init__(self):
        self.lammpstrj = self.__class__.infile
        assert os.path.splitext(self.lammpstrj)[1] == ".lammpstrj", "Make sure to put lammpstrj"
        self.dname = os.path.dirname(os.path.abspath(self.lammpstrj))
        self.loadLammptsrj()

    def __str__(self):
        string = "\n"
        string += f"Number of atoms: {self.natoms}"
        string += f"\nNumber of steps: {self.nstep}"
        return string
    
    def loadLammptsrj(self):
        dataDic = lmp2Dic(self.lammpstrj)
        for key in ["elem", "nstep", "natoms", "ldata", "lattice", "angle"]:
            setattr(self, key, dataDic[key])
        if dataDic["ncol"] == 6:
            self.mol_lmp = False
        elif dataDic["ncol"] == 7:
            self.mol_lmp = True
        else:
            raise ValueError("Sorry for did not considering of the system which dump values bigger than '7'")

class CALCULATION():
    label = ["HCO3", "CO3", "H2CO3", "H3CO3"]
    label += ["H3O", "H2O", "OH"]
    label += ["layerCa", "layerSiO4", "layerH"]
    label += ["aqueousCa", "H"]
    label += ["CaOH", "CaO"]

    O_label = ["CO3", "HCO3", "H2CO3", "H3CO3", "OH", "H2O", "H3O", "layerSiO4", "CaO", "CaOH"]
    output = True

    def __init__(self, startstep=1):
        self.lmp = LAMMPSTRJ()
        self.infile = self.lmp.lammpstrj
        self.elem = self.lmp.elem
        self.has_layer = False if "Si" not in self.elem else True
        self.natoms = self.lmp.natoms
        self.nstep = self.lmp.nstep
        print(self.lmp)
        self.base = os.path.basename(self.infile)
        self.atomDic = {}
        self.idDic = {}
        self.dDic = {}
        self.symbol = np.zeros((self.natoms, ), dtype=object)
        self.extraSymbol = np.zeros((self.natoms, ), dtype=object)
        self.clusterSymbol = np.zeros((self.natoms, ), dtype=object)
        self.current_step = startstep
        self.process_active = True
        self.rcut = {("O", "H"): 1.2,#ckpt_rcut
                     ("C", "O"): 1.5,
                     ("Si", "O"): 2.1,
                     ("Ca", "O"): 3.0,
                     ("Ca", "Ca"): 5.6,
                     ("Ca", "C"): 3.8,
                     ("I", "O"): 2.0}
        self.setMatrix()
        if self.has_layer:
            self.isShiftNeeded()
            self.updateLayer()
            self.CaO_weight = 9
            self.intraLayer_weight = 6
            self.CaO_weight_elute = 10
        self.setLabel(init=True)
        self.ex = {}
        self.proton_ex = []
        ex_label = CALCULATION.O_label + ["proton"]
        for a, b, c, d in itertools.product(ex_label, repeat=4):  
            self.ex[(a, b, c, d)] = 0
        self.count = {}
        self.count[("CaOH", "OH")] = 0
        if CALCULATION.verbose == 1:
            self.outputStat()
            if self.has_layer:
                print("Silicate Chain: ", " ".join(l for l in self.chain_list))
                print("BT: ", self.BT.shape[0], "PT: ", self.PT.shape[0])
        self.checkUnlabledElem()
        

    def clearLabel(self):
        for l in self.label:
            setattr(self, l, None)

    def makeLabel(self, arr, label, mask, transformed=False, elem_id=False):
        if elem_id:
            new_label = arr
        elif not transformed:
            center = np.unique(arr[:, 0])
            termnimal = arr[:, 1]
            new_label = np.concatenate((center, termnimal))
            assert len(mask) == new_label.shape[0]
            for i in range(new_label.shape[0]):
                new_label[i] = self.idDic[mask[i]][new_label[i]]
        else:
            new_label = np.zeros_like(arr)
            for i in range(new_label.shape[0]):
                new_label[i] = self.idDic[mask[i]][arr[i]]
        new_label = new_label.reshape(1, -1)
        if hasattr(self, label):
            current_label = getattr(self, label)
            if current_label is not None:
                setattr(self, label, np.vstack(
                    (getattr(self, label), new_label)))
            else:
                setattr(self, label, new_label)
        else:
            setattr(self, label, new_label)
        assert getattr(
            self, label).ndim == 2, "dimension of label must to be '2'"
        
    def outputStat(self, *args):
        fmt = "{}  labeled: "
        fmt_id = "id= {} "
        labelname = args if args else self.label
        output = f"{self.base}"
        output_ = ""
        for l in labelname:
            if not hasattr(self, l):
                continue
            else:
                currunt_label = getattr(self, l)
                if currunt_label is not None:
                    output_ += f"{l}: {currunt_label.shape[0]} mol\n"
                    out = fmt.format(l)
                    for molecule in currunt_label:
                        if isinstance(molecule, np.ndarray):
                            id_out = " ".join(fmt_id.format(id_) for id_ in molecule)
                            output += f"\n{out} {id_out}"
                            id_out = ""
                        else:
                            id_out = fmt_id.format(molecule)
                            output += f"\n{out} {id_out}"
                            id_out = ""
        with open(os.path.join(self.lmp.dname, "mole.id"), "w") as o:
            o.write(output + "\n\n" + output_)
        if CALCULATION.verbose != 1:
            return
        
    def setMatrix(self):
        if not hasattr(self, "data"):
            self.data = self.lmp.ldata[0]
            self.lattice = self.lmp.lattice[0]
            self.angle = self.lmp.angle[0]
            
        alpha, beta, gamma = np.radians(self.angle)
        x, y, z = self.lattice
        V = np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
                    2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        self.V = V
        matrix = np.array([
            [x, y * np.cos(gamma), z * np.cos(beta)],
            [0, y * np.sin(gamma), z * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
            [0, 0, z * V / np.sin(gamma)]])
        self.matrix = matrix
        
        
    def makeDistanceDic(self, center, terminal, center_flag=None, terminal_flag=None, output_dic=False,
                        center_id=None, terminal_id=None):#ckpt_dist
        if not hasattr(self, f"num_{center}") or not hasattr(self, f"num_{terminal}"):
            return False
        matrix = self.matrix
        if center_flag is not None:
            pos1 = self.atomDic[center][center_flag]
        else:
            pos1 = self.data[self.idDic[center]][:, 1:] if center_id is None else self.data[center_id][:, 1:]
        if terminal_flag is not None:
            pos2 = self.atomDic[terminal][terminal_flag]
        else:
            pos2 = self.data[self.idDic[terminal]][:, 1:] if terminal_id is None else self.data[terminal_id][:, 1:]
        r = pos1[:, None, :] - pos2[None, :, :]
        r = (r @ np.linalg.inv(matrix).T).astype(float)
        r = (r - np.round(r)) @ matrix.T
        r = np.sqrt(np.sum(r**2, axis=2))
        distance = np.where(r <= 0.01, 2000, r)
        if output_dic:
            return distance
        else:
            self.dDic[(center, terminal)] = distance

            
    def setDistanceDic(self, center, terminal, center_flag=None, terminal_flag=None, output_dic=False,
                        center_id=None, terminal_id=None):#ckpt_dist
        if not hasattr(self, f"num_{center}") or not hasattr(self, f"num_{terminal}"):
            return False
        matrix = self.matrix
        if center_flag is not None:
            pos1 = self.atomDic[center][center_flag]
        else:
            pos1 = self.data[self.idDic[center]][:, 1:] if center_id is None else self.data[center_id][:, 1:]
        if terminal_flag is not None:
            pos2 = self.atomDic[terminal][terminal_flag]
        else:
            pos2 = self.data[self.idDic[terminal]][:, 1:] if terminal_id is None else self.data[terminal_id][:, 1:]
        rcut = self.rcut[(center, terminal)]
        pos1 = pos1.astype(float)
        pos2 = pos2.astype(float)
        LARGE = 1e5
        r = np.full((pos1.shape[0], pos2.shape[0]), LARGE, dtype=float)
        for i, p in enumerate(pos1):
            dr = np.abs((pos2 - p) @ np.linalg.inv(self.matrix).T)
            dr = np.where(dr > 0.5, 1 - dr, dr)
            flagx = dr[:, 0] < 1.2 * rcut / self.lattice[0]
            flagy = dr[:, 1] < 1.2 * rcut / self.lattice[1]
            flagz = dr[:, 2] < 1.2 * rcut / self.lattice[2]
            flag = flagx & flagy & flagz
            if np.nonzero(flag)[0].size != 0:
                for idx in np.nonzero(flag)[0]:
                    cart_dr = (p - pos2[idx, :]).astype(float)
                    fract_dr = (cart_dr @ np.linalg.inv(matrix).T)
                    dr = (fract_dr - np.round(fract_dr)) @ matrix.T
                    r[i, idx] = np.linalg.norm(dr)
                    if np.linalg.norm(dr) < 1.2:
                        pass
        distance = np.where(r <= 0.01, LARGE, r)
        if output_dic:
            return distance
        else:
            self.dDic[(center, terminal)] = distance       
            
    def identifyBond(self, center, terminal, option=None):
        if isinstance(center, str) and isinstance(terminal, str):
            key = (center, terminal, option) if option is not None else (center, terminal)
            indices = np.where(self.dDic[key] < self.rcut[key])
        elif isinstance(center, np.ndarray) and isinstance(terminal, float):
            arr = center
            rcut = terminal
            indices = np.where(arr < rcut)
        else:
            raise TypeError("Type mismatch")
        indices = np.vstack((indices[0], indices[1])).T
        index, count = np.unique(indices[:, 0], return_counts=True)
        return indices, index, count

    def identifyProton(self):
        if not hasattr(self, "num_H") or getattr(self, "num_H") == 0:
            return
        if hasattr(self, "H"):
            del self.H
        bool_bond = np.where(self.dDic[("O", "H")] < self.rcut[("O", "H")], True, False)
        idx_H = np.where(np.sum(bool_bond, axis=0) == 0)[0]
        if idx_H.size != 0:
            self.H = self.idDic["H"][idx_H].copy()
            self.symbol[self.H] = "pr"

            
    def identifyIodine(self):#ckpt_io
        if not hasattr(self, "num_I") or getattr(self, "num_I") == 0:
            return
        r = self.dDic[("I", "O")] < self.rcut[("I", "O")]
        num_bond = np.sum(r, axis=1)
        I_list, IO_list, IO3_list  = [], [], []
        for i, arr in enumerate(r):
            id_I = self.idDic["I"][i]
            if num_bond[i] == 0:
                I_list.append([id_I])
            elif num_bond[i] == 1:
                id_O = self.idDic["O"][np.nonzero(arr)[0]]
                assert id_O.size == 1, "Size must be 1"
                id_IO = np.zeros((2, ), dtype=int)
                id_IO[0] = id_I
                id_IO[1] = id_O
                IO_list.append(id_IO)
                bool_bond = self.bool_bond if hasattr(self, "bool_bond") else np.where(self.dDic[("O", "H")] < self.rcut[("O", "H")], True, False)
                assert all(np.sum(bool_bond[np.nonzero(arr)[0]], axis=1) == 0)
            elif num_bond[i] == 3:
                id_O = self.idDic["O"][np.nonzero(arr)[0]]
                assert id_O.size == 3, "Size must be 3"
                id_IO3 = np.zeros((4, ), dtype=int)
                id_IO3[0] = id_I
                id_IO3[1:] = id_O
                IO3_list.append(id_IO3)
                bool_bond = self.bool_bond if hasattr(self, "bool_bond") else np.where(self.dDic[("O", "H")] < self.rcut[("O", "H")], True, False)
                assert all(np.sum(bool_bond[np.nonzero(arr)[0]], axis=1) == 0)
            else:
                raise ValueError(f"Number of I-O: {num_bond[i]}")
        if len(I_list) != 0:
            self.I = np.concatenate([I_list])
            self.symbol[self.I] = "ii"
        if len(IO_list) != 0:
            self.IO = np.vstack([IO_list])
            self.symbol[self.IO.ravel()] = "io"
        if len(IO3_list) != 0:
            self.IO3 = np.vstack([IO3_list])
            self.symbol[self.IO3.ravel()] = "it"
            
        
    
    def identifyCO3(self):
        if not hasattr(self, "num_C") or getattr(self, "num_C") == 0:
            return
        indices, index_C, count_C = self.identifyBond("C", "O")
        for i in range(len(index_C)):
            idx_CO3 = indices[np.where(index_C[i] == indices[:, 0])]
            idx_O = idx_CO3[:, 1]
            if count_C[i] == 3:
                try:
                    bool_bond = np.where(self.dDic[("O", "H")][idx_O, :] < self.rcut[("O", "H")], True, False)
                    num_bond = np.sum(bool_bond)
                except KeyError:
                    num_bond = 0
                if num_bond == 0:
                    self.makeLabel(idx_CO3, "CO3", mask=("C", "O", "O", "O"))
                    self.symbol[self.idDic["C"][index_C[i]]] = "q"
                    self.symbol[self.idDic["O"][idx_O]] = "q"
                    continue
                idx_H = np.array(np.nonzero(bool_bond)[1], dtype=int)
                num_OH = np.sum(np.where(self.dDic[("O", "H")][:, idx_H] < self.rcut[("O", "H")], True, False))
                if num_bond == 1:
                    if num_OH == 1:
                        self.symbol[self.idDic["C"][index_C[i]]] = "hq"
                        self.symbol[self.idDic["O"][idx_O]] = "hq"
                        self.symbol[self.idDic["H"][idx_H]] = "hq"
                        idx_HCO3 = np.concatenate((idx_CO3[0, 0].reshape(1), idx_CO3[:, 1], idx_H))
                        self.makeLabel(idx_HCO3, "HCO3", mask=("C", "O", "O", "O", "H"), transformed=True)
                    elif num_OH >= 2:
                        self.makeLabel(idx_CO3, "CO3", mask=("C", "O", "O", "O"))
                        self.symbol[self.idDic["C"][index_C[i]]] = "q"
                        self.symbol[self.idDic["O"][idx_O]] = "q"
                        # raise ValueError(f"need to adjust rcut between O-H, elem: 'C', id: {self.idDic['C'][index_C[i]]}")
                    else:
                        raise ValueError(f"sth wrong, elem: 'C', id: {self.idDic['C'][index_C[i]]}")
                elif num_bond == 2:
                    self.symbol[self.idDic["C"][index_C[0]]] = "bq"
                    self.symbol[self.idDic["O"][idx_O]] = "bq"
                    self.symbol[self.idDic["H"][idx_H]] = "bq"
                    idx_H2CO3 = np.concatenate((idx_CO3[0, 0].reshape(1), idx_CO3[:, 1], idx_H))
                    self.makeLabel(idx_H2CO3, "H2CO3", mask=("C", "O", "O", "O", "H", "H"), transformed=True)
                elif num_bond == 3:
                    self.symbol[self.idDic["C"][index_C[0]]] = "zq"
                    self.symbol[self.idDic["O"][idx_O]] = "zq"
                    self.symbol[self.idDic["H"][idx_H]] = "zq"
                    idx_H3CO3 = np.concatenate(
                        (idx_CO3[0, 0].reshape(1), idx_CO3[:, 1], idx_H))
                    self.makeLabel(idx_H3CO3, "H3CO3", mask=(
                        "C", "O", "O", "O", "H", "H", "H"), transformed=True)
                else:
                    string = f" Initial Bond error, id={self.idDic['C'][index_C[i]]}  "
                    string += f"Too many 'O-H' bond around 'C"
                    raise ValueError(string)
            else:
                string = f" Initial Bond error, id={self.idDic['C'][index_C[i]]}  "
                string += f"{count_C[i]} C-O bond exists"
                raise ValueError(string)



    def setLayerRegion(self):#ckpt_region
        if not self.has_layer:
            return
        if self.lmp.mol_lmp:
            z = self.Si_z[self.Si_z < 0.5 * self.matrix[2, 2]]
            id_Si = self.idDic["Si"][self.data[self.idDic["Si"]][:, 3] < 0.5 * self.matrix[2, 2]]
        else:
            z = self.Si_z
            id_Si = self.idDic["Si"]
        z = self.Si_z
        id_Si = self.idDic["Si"]
        sort_z = sorted(z, reverse=True)
        print()
        bt, pt = [], []
        upper_flag = None
        self.chain_list, self.intra_boundary, self.inter_boundary = list(), list(), list()
        for i in range(z.shape[0]):
            try:
                gap = sort_z[i] - sort_z[i + 1]
            except IndexError:
                (pt if self.chain_list[-2] == "BT" else bt).append(id_Si[(threshold + 0.0001 > z)])
                self.chain_list.append("PT" if self.chain_list[-2] == "BT" else "BT")
                break
            if gap > 20:
                if self.chain_list[-2] == "PT":
                    target = id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)]
                    bt.append(target)
                    setattr(self, "bt_low", target)
                    setattr(self, "pt_low", id_Si[(z < self.intra_boundary[-1][0] + 0.0001) & (z > threshold + 0.0001)])
                    self.chain_list.append("BT")
                    upper_flag = None
                elif self.chain_list[-2] == "BT":
                    target = id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)]
                    setattr(self, "pt_low", target)
                    pt.append(target)
                    self.chain_list.append("PT")
                    upper_flag = None
                if hasattr(self, "pt_top"):
                    del self.pt_top
                if hasattr(self, "bt_top"):
                    del self.bt_top
                
            elif gap > 4.1:
                self.inter_boundary.append([sort_z[i + 1], sort_z[i]])
                try:
                    if self.chain_list[-2] == "PT":
                        bt.append(id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)])
                        self.chain_list.append("BT")
                        upper_flag = "IT"
                    elif self.chain_list[-2] == "BT":
                        pt.append(id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)])
                        self.chain_list.append("PT")
                        upper_flag = "IT"
                except IndexError:
                    del self.chain_list, self.intra_boundary, self.inter_boundary
                    self.has_layer = False
                    return 
                
            elif gap > 2:
                self.intra_boundary.append([sort_z[i + 1], sort_z[i]])
                if upper_flag is None:
                    target = id_Si[z > sort_z[i + 1] + 0.0001] if "threshold" not in locals().keys() else id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)]
                    pt.append(target)
                    if not hasattr(self, "pt_top"):
                        setattr(self, "pt_top", target)
                else:
                    pt.append(id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)])
                    if not hasattr(self, "pt_top"):
                        setattr(self, "pt_top", id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)])
                upper_flag = "PT"
                
            elif gap > 0.9:
                if upper_flag is None:
                    target = id_Si[z >= sort_z[i + 1] + 0.0001] if "threshold" not in locals().keys() else id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)]
                    bt.append(target)
                    self.bt_top = target
                    upper_flag = "BT"
                elif upper_flag == "BT" or upper_flag == "IT":
                    bt.append(id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)])
                    upper_flag = "BT"
                elif upper_flag == "PT":
                    pt.append(id_Si[(threshold + 0.0001 > z) & (z > sort_z[i + 1] + 0.0001)])
                    upper_flag = "PT"
                else:
                    raise ValueError()
            else:
                continue
            if "threshold" in locals().keys():
                past_threshold = threshold
            threshold = sort_z[i + 1]
            if upper_flag is not None:
                self.chain_list.append(upper_flag)
        for name, s in zip(["BT", "PT"], [bt, pt]):
            stacked = np.hstack(s)
            setattr(self, name, stacked)
        if self.BT.shape[0] + self.PT.shape[0] != id_Si.shape[0]:
            print(self.BT.shape[0])
            print(self.PT.shape[0])
            print(id_Si.shape[0])
            raise ValueError(self.chain_list)

        
    def identifySiO4(self):
        if not hasattr(self, "num_Si") or getattr(self, "num_Si") == 0:
            return
        indices, index_Si, count_Si = self.identifyBond("Si", "O")
        idx_layerH = np.zeros((0), dtype=int)
        for i in range(len(index_Si)):
            idx_SiO4 = indices[np.where(index_Si[i] == indices[:, 0])]
            if count_Si[i] == 4:
                idx_O4 = idx_SiO4[:, 1]
                self.symbol[self.idDic["O"][idx_O4]] = "ll"
                self.makeLabel(idx_SiO4, "layerSiO4", mask=("Si", "O", "O", "O", "O"))
                indices_OH = np.where(self.dDic[("O", "H")][idx_O4, :] < self.rcut[("O", "H")])
                idx_H = indices_OH[1]
                idx_layerH = np.append(idx_layerH, idx_H)
            else:
                idx = [self.idDic["O"][d] for d in idx_SiO4[:, 1]]
                print("SiO4", self.idDic["Si"][idx_SiO4[0, 0]], idx)
                for d in idx_SiO4[:, 1]:
                    print(self.dDic[("Si", "O")][idx_SiO4[0, 0], d])
                string = f" Initial Bond error, id={self.idDic['Si'][index_Si[i]]}  "
                string += f"Only {count_Si[i]} Si-O bond exists"
                print(string)
                raise ValueError(f"wrong rcut or not good data, \n{string}")
        self.layerH = self.idDic["H"][idx_layerH]
        self.symbol[self.idDic["Si"]] = "ll"
        self.Si_z = self.data[self.idDic["Si"]][:, 3]
        self.setLayerRegion()
        if idx_layerH.size != 0:
            self.extraSymbol[self.layerH] = "lH"
            self.symbol[self.layerH] = "ll"
        
    def identifyCa(self):
        if not hasattr(self, "num_Ca") or getattr(self, "num_Ca") == 0:
            return
        if self.has_layer:
            ca_condition1 = (self.zeropoint + self.CaO_weight > self.data[self.idDic["Ca"]][:, 3])
            ca_condition2 = (self.zeropoint - self.CaO_weight < self.data[self.idDic["Ca"]][:, 3])
            idx_layerCa = np.where(ca_condition1 & ca_condition2)[0]
            idx_bulkCa = np.where(~ca_condition1 | ~ca_condition2)[0]
            self.layerCa = self.idDic["Ca"][idx_layerCa]
            self.aqueousCa = self.idDic["Ca"][idx_bulkCa]
            self.symbol[self.aqueousCa] = "ca"
            self.symbol[self.layerCa] = "lca"
        else:
            self.aqueousCa = self.idDic["Ca"]
            self.symbol[self.aqueousCa] = "ca"

    def identifyCaOH(self):
        if not self.has_layer:
            return
        if not hasattr(self, "num_O") or getattr(self, "num_O") == 0:
            return
        if hasattr(self, "layerSiO4"):
            intra_Ca = []
            for t in self.intra_boundary:
                flag = (self.data[self.idDic["Ca"]][:, 3] < t[1]) & (self.data[self.idDic["Ca"]][:, 3] > t[0])
                intra_Ca.append(self.idDic["Ca"][flag])
            self.intraCa = np.hstack(intra_Ca)
            self.intraCa = np.unique(self.intraCa)
            bool_ = self.dDic["Ca", "O"] < self.rcut[("Ca", "O")]
            idx_O = np.where(np.sum(bool_, axis=0) >= 3)
            id_CaOlayer_O = self.idDic["O"][idx_O]
            if hasattr(self, "intra_boundary"):
                if len(self.intra_boundary) == 2:
                    mask1 = (self.data[self.idDic["O"]][:, 3] < self.intra_boundary[0][1] + 0.5) & (self.data[self.idDic["O"]][:, 3] > self.intra_boundary[0][0] - 0.5)
                    mask2 = (self.data[self.idDic["O"]][:, 3] < self.intra_boundary[1][1] + 0.5) & (self.data[self.idDic["O"]][:, 3] > self.intra_boundary[1][0] - 0.5)
                    mask = mask1 | mask2
                elif len(self.intra_boundary) == 1:
                    mask = (self.data[self.idDic["O"]][:, 3] < self.intra_boundary[0][1] + 0.5) & (self.data[self.idDic["O"]][:, 3] > self.intra_boundary[0][0] - 0.5)
                else:
                    return
            else:
                return
            id_CaOlayer_O = self.idDic["O"][mask]
            layer_O = np.unique(self.layerSiO4[:, 1:])
            id_CaOH_O = id_CaOlayer_O[~np.isin(id_CaOlayer_O, layer_O)]
            self.intraO = id_CaOlayer_O
            self.intraCaO = id_CaOH_O
            if len(id_CaOH_O) == 0:
                return
            idx_CaOH_O = np.where(np.isin(self.idDic["O"], id_CaOH_O))[0]
            bool_bond = self.dDic[("O", "H")][idx_CaOH_O] < self.rcut["O", "H"]
            idx_O_tmp, idx_H_tmp = np.where(self.dDic[("O", "H")][idx_CaOH_O] < self.rcut["O", "H"])
            idx_O, indices = np.unique(idx_O_tmp, return_index=True)
            idx_O = idx_CaOH_O[idx_O]
            idx_H = idx_H_tmp[indices]
            num_OH = np.sum(bool_bond, axis=1)
            assert all(num_OH <= 3), "Wrong Stucture or Parameter, Ca O - H"
            OH_mask = num_OH == 1
            H2O_mask = num_OH == 2
            H3O_mask = num_OH == 3
            self.CaOH = np.vstack([self.idDic["O"][idx_O][OH_mask], self.idDic["H"][idx_H][OH_mask]]).T
            self.num_CaOH = self.CaOH.shape[0]
            if np.sum(H2O_mask) != 0:
                print(f"{np.sum(H2O_mask)} mole of CaOH protonated to H2O")
            if np.sum(H3O_mask) != 0:
                print(f"{np.sum(H3O_mask)} mole of CaOH protonated to H3O")
            raveled = self.CaOH.ravel()
            self.symbol[raveled] = "dd"
        else:
            print("Do not have a Silicate chain")

            
    def identifyH2O(self):
        if not hasattr(self, "num_O") or getattr(self, "num_O") == 0:
            return
        if not hasattr(self, "num_H") or getattr(self, "num_H") == 0:
            return
        non_symbol = np.where(self.symbol[self.idDic["O"]] == 0, True, False)
        non_symbol_index = np.where(self.symbol[self.idDic["O"]] == 0)[0]
        water_arr = self.dDic[("O", "H")][non_symbol]
        indices, index_O, count_O = self.identifyBond(water_arr, self.rcut[("O", "H")])
        for i in range(len(index_O)):
            idx_H2O = indices[np.where(index_O[i] == indices[:, 0])]
            idx_H2O[:, 0] = non_symbol_index[index_O[i]]
            index_O[i] = non_symbol_index[index_O[i]]
            idx_H = idx_H2O[:, 1]
            if count_O[i] == 3:
                bool_bond = np.where(self.dDic[("O", "H")][:, idx_H] < self.rcut[("O", "H")], True, False)
                num_bond = np.sum(bool_bond)
                if num_bond == 3:
                    self.makeLabel(idx_H2O, "H3O", mask=("O", "H", "H", "H"))
                    self.symbol[self.idDic["O"][index_O[i]]] = "tw"
                    self.symbol[self.idDic["H"][idx_H2O[:, 1]]] = "tw"
            elif count_O[i] == 2:
                self.makeLabel(idx_H2O, "H2O", mask=("O", "H", "H"))
                self.symbol[self.idDic["O"][index_O[i]]] = "w"
                self.symbol[self.idDic["H"][idx_H2O[:, 1]]] = "w"
            elif count_O[i] == 1:
                self.makeLabel(idx_H2O, "OH", mask=("O", "H"))
                self.symbol[self.idDic["O"][index_O[i]]] = "sw"
                self.symbol[self.idDic["H"][idx_H2O[:, 1]]] = "sw"
            else:
                string = f" Initial Bond error, id={self.idDic['O'][index_O[i]]}"
                string += f"\nThere are {count_O[i]} O-H bond exists"
                raise ValueError(string)


    def calDist(self, cart1, cart2):
        fract1, fract2 = map(lambda v: v @ np.linalg.inv(self.matrix.T), [cart1, cart2])
        vec = fract1 - fract2
        vec = vec - np.round(vec.astype(float))
        vec_cart = vec @ self.matrix.T
        r = np.linalg.norm(vec_cart)
        return r 

    
    def calAngle(self, cart1, cart2, cart3, return_dist=False):
        fract1, fract2, fract3 = map(lambda v: v @ np.linalg.inv(self.matrix.T), [cart1, cart2, cart3])
        vec1 = fract1 - fract2
        vec2 = fract3 - fract2
        vec1 = vec1 - np.round(vec1.astype(float))
        vec2 = vec2 - np.round(vec2.astype(float))
        vec1_cart, vec2_cart = map(lambda v: v @ self.matrix.T, [vec1, vec2])
        dot = vec1_cart @ vec2_cart
        r1 = np.linalg.norm(vec1_cart)
        r2 = np.linalg.norm(vec2_cart)
        norm = r1 * r2
        angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
        angle = np.degrees(angle)
        if return_dist:
            return angle, r1, r2
        else:
            return angle
        
        
    def isShiftNeeded(self):
        if self.lmp.mol_lmp:
            return
        pos_Si = self.lmp.ldata[:, np.where(self.lmp.ldata[0][:, 0] == "Si")[0], 1:4]
        fraction = pos_Si @ np.linalg.inv(self.matrix.T)
        max_fract_si = np.max(fraction[:, :, 2])
        if max_fract_si > 0.7:
            print("For every step do axis-z shift to adjust layer location .... ")
            self.shift = 1.05 - max_fract_si
            print(f"Shift Value:{self.shift:.3f}")
    
    def layerShift(self):
        # fraction = self.data[np.where(self.data[:, 0] == "Si")[0], 1:4] 
        all_fraction = self.data[:, 1:4] @ np.linalg.inv(self.matrix.T)
        all_fraction += self.shift
        all_fraction -= np.floor(all_fraction)
        self.data[:, 1:4] = all_fraction @ self.matrix.T
        
        
    def updateLayer(self):
        if hasattr(self, "shift"):
            self.layerShift()
        try:
            posZ_Si = self.data[self.idDic["Si"]][:, 3]
            self.zeropoint = np.average(self.data[self.layerCa][:, 3])
        except KeyError:
            posZ_Si = self.data[np.where(self.data[:, 0] == "Si")[0]][:, 3]
            self.zeropoint = np.average(posZ_Si)
        self.layermin = np.min(posZ_Si)
        self.layermax = np.max(posZ_Si)
        if CALCULATION.verbose == 1:
            print(f"layer updated max: {self.layermax:.2f} min: {self.layermin:.2f} avg {self.zeropoint:.2f}")
            

    def setLabel(self, init=False):
        if init:
            pass
        else:
            self.clearLabel()
        for e in self.elem:
            flag = self.data[:, 0] == e
            self.atomDic[e] = self.data[flag][:, 1:]
            self.idDic[e] = np.where(flag)[0]
            setattr(self, f"num_{e}", np.sum(flag))
        self.setDistanceDic("O", "H")
        self.identifyProton()
        self.setDistanceDic("C", "O")
        self.setDistanceDic("I", "O")
        self.setDistanceDic("Ca", "Ca")
        self.setDistanceDic("Ca", "O")
        self.setDistanceDic("Si", "O")
        self.identifySiO4()
        self.identifyCaOH()
        self.identifyCO3()
        self.identifyIodine()
        self.identifyCa()
        self.identifyH2O()

    def checkUnlabledElem(self):
        if np.isin(0, self.symbol):
            unlabeled = np.where(self.symbol == 0)
            unlabeled_e = self.data[unlabeled][:, 0]
            unlabeled_id = np.vstack([unlabeled_e, unlabeled]).T
            print("!!"*11, "Unlabeled atoms", "!!"*11)
            print(" ".join(f'{d[0]}: {d[1]}' for d in unlabeled_id))
            print("!" * 61)
            self.symbol[unlabeled] = "x"
            

    def setStep(self, set_step=None, reset_dic=False, reset_symbol=False):#ckpt_step
        if reset_dic:
            self.atomDic = {}
            self.idDic = {}
            self.dDic = {}
        if reset_symbol:
            self.symbol = np.zeros((self.natoms, ), dtype=object)
        self.extraSymbol = np.zeros((self.natoms, ), dtype=object)
        self.clusterSymbol = np.zeros((self.natoms, ), dtype=object)

        if set_step is not None:
            self.data = self.lmp.ldata[set_step]
            self.lattice = self.lmp.lattice[set_step]
            self.angle = self.lmp.angle[set_step]
            self.setMatrix()
            if self.has_layer:
                self.updateLayer()
            if CALCULATION.verbose == 1:
                print("\n", "*" * 60)
                print(f"Calculating: processing step {set_step}")
            if CALCULATION.verbose == 2:
                overwritePrint(f"Calculating: processing step {set_step}")
            return
        try:
            # self.data, self.lattice, self.angle = next(self.iterdata)
            try:
                self.data = self.lmp.ldata[self.current_step]
            except IndexError:
                self.process_active = False
                return
            self.lattice = self.lmp.lattice[self.current_step]
            self.angle = self.lmp.angle[self.current_step]
            self.setMatrix()
            if self.has_layer:
                self.updateLayer()
            if CALCULATION.verbose == 1:
                print("\n", "*" * 60)
                print(f"\nCalculating: processing step {self.current_step}")
            if CALCULATION.verbose == 2:
                overwritePrint(f"Calculating: processing step {self.current_step}")
            self.current_step += 1
        except StopIteration:
            print("\nStop Iteration\n")
            self.process_active = False

            
    def dump2lmp(self):
        a_vec, b_vec, c_vec = self.matrix.T
        xy, xz, yz = b_vec[0], c_vec[0], c_vec[1]
        box_fmt = "{:.6e} {:.6e} {:.6e}\n"
        tail_fmt = "{:<5} {:<2} {:<3} {:.6f} {:.6f} {:.6f} {}\n"
        box_string = ""
        if hasattr(self, "inter_boundary") and len(getattr(self, "inter_boundary")) != 0:
            std_Ca = self.layerCa[(self.data[self.layerCa][:, 3] > self.inter_boundary[0][0]) & (self.data[self.layerCa][:, 3] < self.inter_boundary[0][1])][0]
            std_fract = (self.data[std_Ca][1:] @ np.linalg.inv(self.matrix.T)).astype(float)
        elif hasattr(self, "layerSiO4"):
            std_Si = self.layerSiO4[0, 0]
            std_fract = (self.data[std_Si][1:] @ np.linalg.inv(self.matrix.T)).astype(float)
        else:
            std_fract = np.zeros((3, ), dtype=float)
        box_string += box_fmt.format(min(0, xy, xz, xy + xz), self.matrix[0, 0] + max(0, xy, xz, xy + xz), xy)
        box_string += box_fmt.format(min(0, yz), self.matrix[1, 1] + max(0, yz), xz)
        box_string += box_fmt.format(0, self.matrix[2, 2], yz)
        head = LAMMPSTRJ.fmt.format((self.current_step - 1), self.natoms, box_string.strip()) + "\n"
        index = 1
        mol_index = 0
        label_list = CALCULATION.label + ["I", "IO", "IO3"]
        lmp_arr = np.zeros((self.natoms, 6), dtype=object)
        fract_data = self.data[:, 1:].astype(float) @ np.linalg.inv(self.matrix.T)
        fract_data = fract_data - std_fract
        fract_data = np.where(fract_data > 0, fract_data, fract_data + 1)
        fract_data = np.hstack([self.data[:, 0].reshape(-1, 1), fract_data])
        for label in label_list:
            if hasattr(self, label) and len(getattr(self, label)) != 0:
                arr = getattr(self, label)
                if label == "layerSiO4":
                    arr = np.unique(arr.reshape(-1, ))
                # dat = self.data[arr]
                dat = fract_data[arr]
                if arr.ndim == 1:
                    elem = dat[:, 0].reshape(-1, 1)
                    fract = dat[:, 1:]
                    fract = fract[:, None, :]
                elif arr.ndim == 2:
                    elem = dat[:, :, 0]
                    fract = dat[:, :, 1:]
                arr = arr.reshape(elem.shape[0], elem.shape[1])
                for f, e, id_ in zip(fract, elem, arr):
                    f = f.astype(float)
                    if f.shape[0] == 1:
                        c  = f @ self.matrix.T
                    else:
                        ref = f[0]
                        delta = f - ref
                        delta -= np.round(delta)
                        adjusted_f = ref + delta
                        c  = adjusted_f @ self.matrix.T
                    for i in range(c.shape[0]):
                        assert (c.shape[0] == e.shape[0]) and (c.shape[0] == id_.shape[0]), f"label: {label}, id: {id_.shape[0]}, cart: {c.shape[0]}"
                        c_ = c[i]
                        e_ = e[i]
                        id__ = id_[i]
                        lmp_arr[id__, :] = LAMMPSTRJ.mole_num[e_], e_, *c_, mol_index
                        index += 1
                    mol_index += 1
        unlabeled_id = np.where((lmp_arr[:, 0].astype(int)) == 0)[0]
        if unlabeled_id.shape[0] != 0:
            for i in range(unlabeled_id.shape[0]):
                elem, *xyz = self.data[unlabeled_id[i]]
                lmp_arr[unlabeled_id[i]] = LAMMPSTRJ.mole_num[elem], elem, *xyz, mol_index
            print("dump2mol: Unlabeled id: ", " ".join(str(d) for d in unlabeled_id), f" -> mol_id: {mol_index}")
        if self.natoms > index - 1:
            print("dump2mol:: Unlabeled atoms exists")
        elif self.natoms < index - 1:
            print(self.natoms, index)
            print("dump2mol: Double labeled atoms exists")
            _, counts = np.unique(np.round(lmp_arr[:, 2:5].astype(float), decimals=2), axis=0, return_counts=True)
            ref = counts[counts == 2]
        lmp_index = 0
        for nd0 in lmp_arr:
            head += tail_fmt.format(lmp_index, *nd0)
            lmp_index += 1
        with open(os.path.join(self.lmp.dname, self.fname), "a") as o:
            o.write(head)
            print("dump2mol ...")
            
    def isinBulk(self, id_):
        if not self.has_layer:
            return id_
        if isinstance(id_, np.ndarray):
            z = self.data[id_][:, 3]
        if isinstance(id_, np.int64):
            z = self.data[id_][3]
        adjust_value = 3
        condition1 = self.layermax - adjust_value < z
        condition2 = self.layermin + adjust_value > z
        isin = condition1 | condition2
        return isin

    def inwhichLabel(self, elem, id_which, return_id=False, pre=False):#ckpt_which
        if isinstance(id_which, np.int64):
            id_which = np.array([id_which], dtype=int)
        if elem == "O":
            for m in CALCULATION.O_label:
                if hasattr(self, f"p_{m}"):
                    label = getattr(self, f"p_{m}")
                    if any(np.isin(id_which, label)):
                        pre_mole = m
                        break
                pre_mole = None
            for m in CALCULATION.O_label:
                if hasattr(self, m):
                    label = getattr(self, m)
                    if label is not None:
                        if any(np.isin(id_which, label)):
                            id_ = label[np.nonzero(np.isin(label, id_which))[0]]
                            now_mole = m
                            break
                now_mole = None
            if now_mole == None:
                print(pre_mole)
                print(id_which)
                print(self.data[id_which])
                exit()
            if not pre:
                if not return_id:
                    return now_mole
                else:
                    return now_mole, id_
            else:
                if not return_id:
                    return pre_mole, now_mole
                else:
                    return pre_mole, now_mole, id_

        
    def anal_proton_exchange(self):
        for i in range(len(self.proton_ex)):
            lst = self.proton_ex[i]
            assert len(lst) == 2 or len(lst) == 3
            if len(lst) == 2:
                donor, idx = lst
            if len(lst) == 3:
                donor, idx, id_donor = lst
            id_H = self.idDic["H"][idx]
            if isinstance(id_H, np.ndarray):
                id_H = id_H[0]
            pre_mole, now_mole, output = self.traceSepertaedProton(idx)
            if "id_donor" in locals().keys():#cckpt
                mole = self.inwhichLabel("O", id_donor)
                self.ex[(donor, mole, pre_mole, now_mole)] += 1
                del id_donor
            else:
                self.ex[(donor, donor, pre_mole, now_mole)] += 1
            print(f"'H': {id_H}", output)
        self.proton_ex = []

    def traceSepertaedProton(self, idx_H):
        output_ = ""
        idx_whichO = np.where(self.bool_bond[:, idx_H])[0]
        id_whichO = self.idDic["O"][idx_whichO] if len(idx_whichO) != 0 else None
        id_H = self.idDic["H"][idx_H]
        if id_whichO is not None:
            preO, whichO, id_ = self.inwhichLabel("O", id_whichO, return_id=True, pre=True)
            if whichO is None:
                output_ += "!!!Unknown 'O'!!!"
            if whichO in ("OH", "H2O", "H3O"):
                self.extraSymbol[id_] = "mc"
                self.extraSymbol[id_H] = "mc"
        else:
            self.symbol[id_H] = "pr"
        whichO = "proton" if "whichO" not in locals().keys() else whichO
        preO = "proton" if "preO" not in locals().keys() else preO
        output_ += f" {preO} ->> {whichO}"
        return preO, whichO, output_

    def traceH3O(self):
        if not hasattr(self, "H3O") or getattr(self, "H3O").size == 0:
            return
        bool_bond = self.bool_bond
        exist_flag = np.ones((self.H3O.shape[0], ), dtype=bool)
        for i in range(self.H3O.shape[0]):
            output_ = ""
            id_O, id_H1, id_H2, id_H3 = self.H3O[i, :]
            idx_O = np.where(self.idDic["O"] == id_O)[0]
            idx_H = np.where(np.isin(self.idDic["H"], self.H3O[i, 1:]))[0]
            bond_num = np.sum(bool_bond[idx_O])
            bond_now = np.nonzero(bool_bond[idx_O, :])[1]
            if bond_num == 3:
                id_bond_now = self.idDic["H"][bond_now]
                if np.sort(id_bond_now).all() != np.sort(self.H3O[i, 1:]).all():
                    change_H = self.H3O[:, i][~np.isin(self.H3O[i, :], id_bond_now)]
                    self.extraSymbol[change_H] = "hc"
                    idx_H = np.where(self.idDic["H"] == change_H)[0]
                    output_ += "id of 'H3O' changed"
                    self.proton_ex.append(["H3O", idx_H, id_O])
                    self.H3O[i, 1:] = bond_now
                    output_ += " id: "
                    output_ += " ".join(str(s) for s in self.H3O[i, 1:])
                    output_ += " -> "
                    output_ += " ".join(str(s) for s in bond_now)
                self.symbol[self.H3O[i, :]] = "tw"
            elif bond_num == 2:
                dis_idx_H = idx_H[~np.isin(idx_H, bond_now)]
                dis_id_H = self.idDic["H"][dis_idx_H[0]]
                id_H2O = np.delete(self.H3O[i], np.where(self.H3O[i] == dis_id_H))
                self.makeLabel(id_H2O, "H2O", mask=None, transformed=True, elem_id=True)
                output_ += "'H3O' broke into 'H2O' id "
                output_ += " ".join(str(d) for d in self.H3O[i])
                output_ += " -> "
                output_ += " ".join(str(d) for d in id_H2O)
                self.proton_ex.append(["H3O", dis_idx_H])
                output_ += f" 'H' :{dis_id_H}"
                exist_flag[i] = False
                self.symbol[id_H2O] = "w"
                
            elif bond_num == 1:
                dis_idx_H = idx_H[~np.isin(idx_H, bond_now)]
                dis_id_H = self.idDic["H"][dis_idx_H]
                id_OH = self.H3O[i][~np.isin(self.H3O[i], dis_id_H)]
                self.makeLabel(id_OH, "OH", mask=None, transformed=True, elem_id=True)
                output_ += "'H3O' broke into 'OH' id "
                output_ += " ".join(str(d) for d in self.H3O[i])
                output_ += " -> "
                for d in idx_H:
                    self.proton_ex.append(["H3O", d, id_OH[0]])
                output_ += " ".join(str(d) for d in id_OH)
                assert dis_idx_H.shape[0] == 2
                for j in range(2):
                    output_ += f" 'H' :{dis_id_H[j]}"
                exist_flag[i] = False
                self.symbol[id_OH] = "sw"
            else:
                raise ValueError("bond number in H3O weird")
            if output_:
                print(output_)
        self.H3O = self.H3O[exist_flag]
        
    def traceOH(self):
        if not hasattr(self, "OH") or getattr(self, "OH").size == 0:
            return
        self.OH = self.OH if self.OH.ndim == 2 else self.OH.reshape(-1, 2)
        exist_flag = np.ones((self.OH.shape[0], ), dtype=bool)
        for i in range(self.OH.shape[0]):
            output_ = ""
            id_O, id_H = self.OH[i, :]
            idx_O, idx_H = np.where(self.idDic["O"] == id_O)[0], np.where(self.idDic["H"] == id_H)[0]
            bond_now = np.nonzero(self.bool_bond[idx_O, :])[1]
            if bond_now.shape[0] == 0:
                print(id_O, id_H)
                print("!!!bond structure broken!!!")
                # raise ValueError("!!!bond structure broken!!!")
            bond_num = np.sum(self.bool_bond[idx_O, :])
            bond_OH_re = self.bool_bond[:, bond_now]
            nOH = np.sum(bond_OH_re)
            if bond_num == 1:
                if nOH == 1:
                    if not np.array_equal(idx_H, bond_now):
                        change_H = self.idDic["H"][bond_now]
                        self.OH[i, 1] = self.idDic['H'][bond_now[0]]
                        self.extraSymbol[change_H] = "hc"
                        output_ += "'OH' id of 'H' changed"
                        self.proton_ex.append(["OH", idx_H, id_O])
                        output_ += f" 'H': {id_H} -> {self.idDic['H'][bond_now][0]}"
                    self.symbol[self.OH.ravel()] = "sw"
                else:
                    whichO = np.unique(np.nonzero(bond_OH_re)[0])
                    whichMole, id_ = self.inwhichLabel("O", self.idDic["O"][whichO], return_id=True)
                    if whichMole in ("OH", "H2O", "H3O"):
                        self.extraSymbol[id_] = "mc"
                    output_ += f'!!!OH absorbed to {whichMole}!!!'

            elif bond_num == 2:
                new_idx_H = bond_now[~np.isin(bond_now, idx_H)]
                new_id_H = self.idDic["H"][new_idx_H]
                if new_idx_H.shape[0] == 2:
                    output_ += "'H' changed and get two new 'H' to be 'H2O'"
                    id_H2O = np.append(self.OH[i, 0], new_id_H)
                    for h in new_idx_H:
                        self.proton_ex.append(["OH", h, id_O])
                elif new_idx_H.shape[0] == 1:
                    id_H2O = np.append(self.OH[i, :], new_id_H)
                else:
                    raise ValueError("sth wrong")
                assert id_H2O.shape[0] == 3, f"{id_H2O}, {id_H2O.shape} {new_idx_H} "
                self.symbol[id_H2O] = "w"
                self.makeLabel(id_H2O, "H2O", mask=None, transformed=True, elem_id=True)
                output_ += "'OH' absorb 'H' to 'H2O' id "
                output_ += " ".join(str(d) for d in self.OH[i])
                output_ += " -> "
                output_ += " ".join(str(d) for d in id_H2O)
                output_ += f" 'H' :{new_id_H[0]}+"
                exist_flag[i] = False
            elif bond_num == 3:
                new_idx_H = bond_now[~np.isin(bond_now, idx_H)]
                new_id_H = self.idDic["H"][new_idx_H]
                id_H3O = np.append(self.OH[i], new_id_H)
                self.symbol[id_H3O] = "tw"
                self.makeLabel(id_H3O, "H3O", mask=None, transformed=True, elem_id=True)
                output_ += "'OH' absorb two 'H' to 'H3O' id "
                output_ += " ".join(str(d) for d in self.OH[i])
                output_ += " ->"
                output_ += " ".join(str(d) for d in id_H3O)
                output_ += f" 'H': {new_id_H[0]}+"
                output_ += f" 'H': {new_id_H[1]}+"
                exist_flag[i] = False
            elif bond_num >= 4:
                raise ValueError(f"bond number in H3O wrong :{bond_num} ")
            if output_:
                print(output_)
        self.OH = self.OH[exist_flag]
        if hasattr(self, "layer") and self.intraCaO.size != 0:
            mask_OH = np.ones((self.OH.shape[0], ), dtype=bool)
            mask_Ca = np.isin(self.idDic["Ca"], self.intraCa)
            for i, id_ in enumerate(self.OH[:, 0]):
                mask_O = self.idDic["O"] == id_
                if np.sum(self.bool_bond_CaO[mask_Ca, mask_O]) != 0:
                    print(f"'OH' move to 'CaOH' id: {self.OH[i, 0]}, {self.OH[i, 1]}")
                    self.CaOH = np.vstack([self.CaOH, self.OH[i]])
                    mask_OH[i] = False
            self.OH = self.OH[mask_OH]
    
        
    def traceH2O(self):
        if not hasattr(self, "H2O") or getattr(self, "H2O").size == 0:
            return
        bool_bond = self.bool_bond
        exist_flag = np.ones((self.H2O.shape[0], ), dtype=bool)
        for i in range(self.H2O.shape[0]):
            output_ = ""
            id_O, id_H1, id_H2 = self.H2O[i]
            idx_O = np.where(self.idDic["O"] == id_O)[0]
            idx_H1 = np.where(self.idDic["H"] == id_H1)[0]
            idx_H2 = np.where(self.idDic["H"] == id_H2)[0]
            bond_OH1 = bool_bond[idx_O, idx_H1]
            bond_OH2 = bool_bond[idx_O, idx_H2]
            bond_num = np.sum(bool_bond[idx_O])
            if bond_num == 3:
                now_idx_H = np.where(bool_bond[idx_O, :])[1]
                OH_num = np.sum(bool_bond[:, now_idx_H])
                if OH_num == 3:
                    idx_H3O = np.append(idx_O, now_idx_H)
                    self.makeLabel(idx_H3O, "H3O", mask=("O", "H", "H", "H"), transformed=True)
                    self.symbol[self.idDic["H"][now_idx_H]] = "tw"
                    self.symbol[self.idDic["O"][idx_O]] = "tw"
                    output_ += f"'H3O' created: {id_O} "
                    output_ += " ".join(str(s) for s in self.H2O[i, 1:])
                    output_ += f"  ->  {id_O} "
                    output_ += " ".join(str(self.idDic["H"][s]) for s in now_idx_H)
                    exist_flag[i] = False
                if OH_num >= 4:
                    self.symbol[self.idDic["H"][now_idx_H]] = "aw"
                    self.symbol[self.idDic["O"][idx_O]] = "aw"
                    idx_ab = np.unique(np.nonzero(bool_bond[:, now_idx_H])[0])
                    idx_newO = idx_ab[~np.isin(idx_ab, idx_O)]
                    id_newO = self.idDic["O"][idx_newO]
                    label, id_ = self.inwhichLabel("O", id_newO, return_id=True)
                    output_ = f"'H2O' absorbed in other {label} molecule"
                    idx_newH = now_idx_H[~np.isin(now_idx_H, np.append(idx_H1, idx_H2))]
                    id_newH = self.idDic["H"][idx_newH]
                    output_ += f"' {id_O} {id_H1} {id_H2} -> {id_newO[0]} "
                    output_ += " ".join(str(s) for s in id_newH)
                print(output_)
                continue

            if bond_num == 2:
                self.symbol[id_O] = "w"
                if bond_OH1 and bond_OH2:
                    self.symbol[[id_H1, id_H2]] = "w"
                else:
                    self.H2O[i, 1:] = self.idDic["H"][np.where(
                        bool_bond[idx_O, :])[1]]
                    self.symbol[self.H2O[i, :]] = "w"
                    output_ += f"'H' in H2O changed: {id_O} {id_H1} {id_H2} -> "
                    output_ += " ".join(str(s) for s in self.H2O[i])
                    output_ += ", "
            elif bond_num == 1:
                self.symbol[id_O] = "sw"
                if bond_OH1:
                    self.symbol[id_H1] = "sw"
                    id_last_H = id_H1
                if bond_OH2:
                    self.symbol[id_H2] = "sw"
                    id_last_H = id_H2
                if (not bond_OH1) & (not bond_OH2):
                    idx_h3 = np.where(bool_bond[idx_O, :])[1]
                    id_h3 = self.idDic["H"][idx_h3]
                    assert idx_h3.shape[0] == 1
                    self.symbol[id_h3] = "sw"
                    id_last_H = id_h3
                id_OH = np.append(id_O, id_last_H)
                if hasattr(self, "OH") and len(getattr(self, "OH")) != 0:
                    self.OH = np.vstack([self.OH, id_OH])
                else:
                    self.OH = np.array(id_OH, dtype=int)
                output_ += f"'H2O' broke to 'OH-': {id_O} {id_H1} {id_H2} -> "
                output_ += " ".join(str(s) for s in id_OH)
                exist_flag[i] = False
            if not bond_OH1:
                output_ += f" 'H': {id_H1}-"
                self.proton_ex.append(["H2O", idx_H1, id_O])
            if not bond_OH2:
                output_ += f" 'H': {id_H2}-"
                self.proton_ex.append(["H2O", idx_H2, id_O])
            if output_:
                print(output_.strip())
        self.H2O = self.H2O[exist_flag]

        
    def traceCO3(self, molecule):
        if not hasattr(self, f"{molecule}") or getattr(self, f"{molecule}").size == 0:
            return
        carbonate = getattr(self, molecule).copy()
        if carbonate is None:
            return
        if not hasattr(self, "num_H"):
            self.symbol[carbonate.ravel()] = "q"
            return
        exist_flag = np.ones((carbonate.shape[0], ), dtype=bool)
        now_mole = molecule
        for i in range(carbonate.shape[0]):
            output_ = ""
            state = ""
            id_O = carbonate[i, 1:4]
            if np.isin(self.except_C_index, carbonate[i, 0]).any():
                continue
            if molecule != "CO3":
                assert carbonate.shape[1] >= 4
                id_H = carbonate[i, 4:].copy()
                idx_H = np.where(np.isin(self.idDic["H"], id_H))[0]
            else:
                id_H, idx_H = None, None
            idx_O = np.where(np.isin(self.idDic["O"], id_O))[0]
            bond_HCO3 = self.bool_bond[idx_O, :]
            num_HCO3 = np.sum(bond_HCO3)
            now_idx_H = np.nonzero(bond_HCO3)[1]
            now_id_H = self.idDic["H"][now_idx_H] if now_idx_H is not None else None
            if now_id_H.size != 0:
                nOH = np.sum(self.bool_bond[:, now_idx_H])
            if idx_H is None:
                dis_idx_H, dis_id_H = [], []
            else:
                dis_idx_H = idx_H[~np.isin(idx_H, now_idx_H)] if now_idx_H.size != 0 else idx_H.copy()
                dis_id_H = self.idDic["H"][dis_idx_H] if dis_idx_H is not None else None
            if num_HCO3 >= 4:
                raise ValueError(f"bond num: {num_HCO3}")
            if num_HCO3 == 0:
                self.symbol[carbonate[i, :4]] = "q"
                now_mole = "CO3"
                if molecule != "CO3":
                    exist_flag[i] = False
                    self.makeLabel(carbonate[i, :4], "CO3", mask=None, transformed=True, elem_id=True)
                    state = "became "
                    
            if num_HCO3 == 1:
                self.symbol[now_id_H] = "hq"
                self.symbol[carbonate[i, :4]] = "hq"
                now_mole = "HCO3"
                if molecule != "HCO3":
                    id_HCO3 = np.append(carbonate[i, :4], now_id_H)
                    assert id_HCO3.shape[0] == 5, f"HCO3, {id_HCO3}"
                    self.except_C_index.append(carbonate[i, 0])
                    if nOH == 1:
                        exist_flag[i] = False
                        self.makeLabel(id_HCO3, "HCO3", mask=None, transformed=True, elem_id=True)
                        state = "became "
                    else:
                        state = "was absorbed as "
                        self.extraSymbol[now_id_H] = "hc"
                        self.extraSymbol[carbonate[i]] = "hc"
                elif dis_id_H is not None:
                    id_HCO3 = np.append(carbonate[i, :4], now_id_H)
                    carbonate[i, :] = id_HCO3
                    state = "index changed "
                    if nOH != 1:
                        state += "and absorbed as "
                        self.extraSymbol[now_id_H] = "hc"
                        self.extraSymbol[carbonate[i]] = "hc"

            if num_HCO3 == 2:
                self.symbol[now_id_H] = "bq"
                self.symbol[carbonate[i, :4]] = "bq"
                now_mole = "H2CO3"
                if molecule != "H2CO3":
                    id_H2CO3 = np.append(carbonate[i, :4], now_id_H)
                    assert id_H2CO3.shape[0] == 6, "H2CO3"
                    self.except_C_index.append(carbonate[i, 0])
                    if nOH == 2:
                        exist_flag[i] = False
                        self.makeLabel(id_H2CO3, "H2CO3", mask=None,
                                       transformed=True, elem_id=True)
                        state = "became "
                    else:
                        state = "was absorbed as "
                        self.extraSymbol[now_id_H] = "hc"
                        self.extraSymbol[carbonate[i]] = "hc"
                elif dis_id_H is not None:
                    id_H2CO3 = np.append(carbonate[i, :4], now_id_H)
                    carbonate[i, :] = id_H2CO3
                    state = "index changed "
                    if nOH != 2:
                        state += "and absorbed as "
                        self.extraSymbol[now_id_H] = "hc"
                        self.extraSymbol[carbonate[i]] = "hc"
                        
            if num_HCO3 == 3:
                self.symbol[now_id_H] = "zq"
                self.symbol[carbonate[i, :4]] = "zq"
                now_mole = "H3CO3"
                if molecule != "H3CO3":
                    id_H3CO3 = np.append(carbonate[i, :4], now_id_H)
                    assert id_H3CO3.shape[0] == 7, "H3CO3"
                    self.except_C_index.append(carbonate[i, 0])
                    if nOH == 3:
                        exist_flag[i] = False
                        self.makeLabel(id_H3CO3, "H3CO3", mask=None,
                                       transformed=True, elem_id=True)
                        state = "became "
                    else:
                        state = "was absorbed as "
                        self.extraSymbol[now_id_H] = "hc"
                        self.extraSymbol[carbonate[i]] = "hc"
                elif dis_id_H is not None:
                    id_H3CO3 = np.append(carbonate[i, :4], now_id_H)
                    carbonate[i, :] = id_H3CO3
                    state = "idx changed "
                    if nOH != 1:
                        state += "and absorbed as "
                        self.extraSymbol[now_id_H] = "hc"
                        self.extraSymbol[carbonate[i]] = "hc"
                    
            if now_mole != molecule or len(dis_id_H) != 0:
                output_ += f"'{molecule}' {state} '{now_mole}' " if now_mole != molecule else f"'{molecule}' {state}"
                output_ += " ".join(str(s) for s in carbonate[i, :4])
                output_ += " 'H': "
                if id_H is not None:
                    output_ += " ".join(str(s) for s in id_H)
                else:
                    output_ += "X"
                output_ += " -> "
                if now_id_H is not None and now_id_H.size != 0:
                    output_ += " ".join(str(s) for s in now_id_H)
                else:
                    output_ += "X"

            if dis_idx_H is not None:
                for idx_H in dis_idx_H:
                    self.proton_ex.append([molecule, idx_H, id_O[0]])
            if output_:
                print(output_)
            now_mole = molecule
        setattr(self, molecule, carbonate[exist_flag])

    def traceLayerCa(self):
        if not hasattr(self, "CaOH"):
            return
        center_flag = np.isin(self.idDic["Ca"], self.layerCa)
        self.setDistanceDic("Ca", "C", center_flag=center_flag)
        idx_CaC = np.where(self.dDic[("Ca", "C")] < self.rcut["Ca", "C"])
        idx_CaC = np.vstack(idx_CaC).T
        if idx_CaC.shape[0] == 0:
            return
        for i in range(idx_CaC.shape[0]):
            id_C = self.idDic["C"][idx_CaC[i, 1]]
            id_Ca = self.layerCa[idx_CaC[i, 0]]
            if hasattr(self, "CO3") and id_C in self.CO3:
                mole = "CO3"
            elif hasattr(self, "HCO3") and id_C in self.HCO3:
                mole = "HCO3"
            elif hasattr(self, "H2CO3") and id_C in self.H2CO3:
                mole = "H2CO3"
            else:
                raise ValueError("?>?>?>?")
            co3_arr = getattr(self, mole)
            id_ = co3_arr[co3_arr[:, 0] == id_C]
            print(f"CaO layer absorbed {mole} id {id_Ca} {' '.join(str(x) for x in id_.flatten())}")
            

    def traceCaOH(self):#ckpt_caoh
        if not hasattr(self, "CaOH") or getattr(self, "CaOH").size == 0:
            return
        output_ = ""
        exist_mask = np.ones((self.CaOH.shape[0], ), dtype=bool)
        for i, d in enumerate(self.CaOH[:, 0]):
            bool_H = self.bool_bond[self.idDic["O"]==d, :].reshape(-1, )
            id_H = self.idDic["H"][bool_H]
            num_OH = np.sum(bool_H)
            if num_OH == 1:
                if self.CaOH[i, 1] != id_H[0]:
                    output_ += f"CaOH: {self.CaOH[i, 0]} {self.CaOH[i, 1]} -> {self.CaOH[i, 0]} {id_H[0]}\n"
                    self.CaOH[i, 1] = id_H[0]
                    idx_H = np.nonzero(self.idDic["H"] == id_H)[0]
                    self.proton_ex.append(["CaOH", idx_H, d])
            elif num_OH == 2:
                mask = self.CaOH[:, 0] == d
                oh = self.CaOH[mask].ravel()
                id_H = id_H.ravel()
                if len(oh) >= 2 and len(id_H) >= 2:
                    output_ += "CaOH protonated to H2O id: {} {} -> {} {} {}\n".format(oh[0], oh[1], d, id_H[0], id_H[1])
                else:
                    output_ += f"Format error: oh={oh}, id_H={id_H}\n"
                id_h2o = np.array([d, *id_H], dtype=int)
                id_h2o = id_h2o.reshape(-1, )
                self.symbol[id_h2o] = "w"
                self.H2O = np.vstack([self.H2O, id_h2o])
                exist_mask[mask] = False
            else:
                raise ValueError("Bond 'O'")
        self.CaOH = self.CaOH[exist_mask]
        exist_mask = np.ones((self.CaOH.shape[0], ), dtype=bool)
        if hasattr(self, "intra_boundary"):
            if len(self.intra_boundary) == 2:
                max_borderline = self.intra_boundary[0][1] + 1
                min_borderline = self.intra_boundary[1][0] - 1
            if len(self.intra_boundary) == 1:
                max_borderline = self.intra_boundary[0][1] + 1
                min_borderline = self.intra_boundary[0][0] - 1
        flag = self.idDic["O"]
        flag1 = self.data[self.idDic["O"]][:, 3] < max_borderline 
        flag2 = self.data[self.idDic["O"]][:, 3] > min_borderline
        flag = flag1 & flag2
        id_O = self.idDic["O"][flag]
        r = self.setDistanceDic("Ca", "O", center_id=self.intraCa, terminal_id=id_O, output_dic=True)
        idx_coorded = np.where(r < self.rcut[("Ca", "O")])[1]
        id_coorded = id_O[idx_coorded]
        eluted_intraO = np.nonzero(~np.isin(self.intraCaO, id_coorded))[0]
        if len(eluted_intraO) != 0:
            id_eluted_O = self.intraCaO[eluted_intraO]
            for d in id_eluted_O:
                bool_H = self.bool_bond[self.idDic["O"] == d, :].reshape(-1, )
                id_H = self.idDic["H"][bool_H]
                if id_H.shape[0] == 2:
                    output_ += f"'H2O' absorbed in CaOH id: {d}, {' '.join(str(H) for H in id_H)}\n"                    
                    continue
                elif id_H.shape[0] == 1:
                    self.symbol[d] = "sw"
                    self.symbol[id_H] = "sw"
                    output_ += f"'CaOH' move to 'OH' id: {d}, {id_H[0]}\n"
                    exist_mask[self.CaOH[:, 0] == d] = False
                    id_OH = np.r_[d, id_H].reshape(1, -1)
                    assert id_OH.size == 2
                    if not hasattr(self, "OH") or self.OH.size == 0:
                        self.OH = id_OH.copy()
                    else:
                        self.OH = np.vstack([self.OH, id_OH])
        self.CaOH = self.CaOH[exist_mask]
        print(output_)
        if len(self.CaOH) != 0:
            self.symbol[self.CaOH.ravel()] = "dd"
        new_num_CaOH = self.CaOH.shape[0]
        if new_num_CaOH != self.num_CaOH:
            print(f"CaO-H: {self.num_CaOH} -> {new_num_CaOH}")
            self.num_CaOH = new_num_CaOH
        else:
            print(f"CaO-H: {self.num_CaOH}")

            
    def traceSiO4(self):
        if not hasattr(self, "layerSiO4") or getattr(self, "layerSiO4").size == 0:
            return
        output_ = ""
        if self.layerH is not None:
            old_layerH = self.layerH.copy()
        self.layerH = self.idDic["H"][np.nonzero(self.bool_bond[self.idx_O_layer, :])[1]]
        unique, counts = np.unique(self.layerH, return_counts=True)
        twice = unique[counts == 2]
        if twice.size > 0:
            print("bridge 'H'", " ".join(str(t) for t in twice))
            self.layerH = unique.copy()
            self.symbol[twice] = "ll"
        del unique
        idx_outlayerH = np.unique(np.nonzero(self.bool_bond[self.idx_O_outlayer, :])[1])
        id_outlayerH = self.idDic["H"][idx_outlayerH]
        bool_ab_idx_H = np.where(np.sum(self.bool_bond[:, idx_outlayerH], axis=0) == 2, True, False)
        self.outlayerH = id_outlayerH[~bool_ab_idx_H]
        ab_id_H = id_outlayerH[bool_ab_idx_H]
        dis_lH = old_layerH[~np.isin(old_layerH, self.layerH)]
        new_lH = self.layerH[~np.isin(self.layerH, old_layerH)]
        self.layerH = self.layerH[~np.isin(self.layerH, ab_id_H)]
        self.symbol[self.layerH] = "ll"
        self.extraSymbol[self.layerH] = "lH"
        if old_layerH.shape[0] == self.layerH.shape[0]:
            output_ += f"SiO4 O-H: {old_layerH.shape[0]}"
        else:
            output_ += f"SiO-H: {old_layerH.shape[0]} -> {self.layerH.shape[0]}"
        output_ += f", surf: {self.outlayerH.shape[0]}"
        output_ += f", absorbed : {ab_id_H.shape[0]}"
        output_ += f", inter: {self.layerH.shape[0] - self.outlayerH.shape[0]}\n"
        if dis_lH.shape[0] != 0:
            for s in dis_lH:
                output_ += f"\nlost 'H' id: {s}"
                idx_H = np.nonzero(self.idDic["H"] == s)[0]
                self.proton_ex.append(["layerSiO4", idx_H])
        if new_lH.shape[0] != 0:
            output_ += "\nget 'H' id: "
            output_ += " ".join(str(s) for s in new_lH)
        if ab_id_H.shape[0] != 0:
            idx_H = np.where(np.isin(self.idDic["H"], ab_id_H))[0]
            for i in range(idx_H.shape[0]):
                output_ += "\nab 'H' id: "
                output_ += f"{str(ab_id_H[i])} "
                idx_whichO = np.where(self.bool_bond[:, idx_H[i]])[0]
                id_whichO = self.idDic["O"][idx_whichO] 
                mole = self.inwhichLabel("O", id_whichO, return_id=False)
                output_ += f" of {mole}"
        
        
    def setVmdSymbol(self, init=False, output=False):
        symbolist = ["pr", "q", "hq", "bq", "zq", "w", "hc", "mc", "mm",
                     "tw", "aw", "sw", "ll", "lH", "lca", "ca", "wc", "dd", "dq"]
        symbolist.extend(["nc", "so"])
        symbolist.extend(["ii", "io", "it"])
        if output:
            for l in symbolist:
                indexfile = f"{l}.index"
                with open(os.path.join(self.lmp.dname, indexfile), "w") as o:
                    o.write(getattr(self, f"{l}_index").strip())
        else:
            for l in symbolist:
                index = getattr(self, f"{l}_index") if hasattr(self, f"{l}_index") else ""
                if l in ("hc", "mm", "mc", "lH"):
                    symbol_step = np.where(self.extraSymbol == l)[0]
                elif l in ("nc", "so"):
                    symbol_step = np.where(self.clusterSymbol == l)[0]
                else:
                    symbol_step = np.where(self.symbol == l)[0]
                index += "index "
                index += " ".join(str(s) for s in symbol_step)
                index += "\n"
                setattr(self, f"{l}_index", index)

    @checkTime
    def bondStat(self):
        self.setStep(reset_dic=True, reset_symbol=True)
        if not self.process_active:
            return
        self.setLabel()
        self.checkUnlabledElem()
        self.outputStat("H", "H3O", "HCO3", "HCO3_ab",
                        "layerH", "H3O_ab", "OH", "H2CO3")

        
    def traceMoleStat(self):
        carbonate = ["CO3", "HCO3", "H2CO3", "H3CO3"]
        unique = ["H", "OH", "H3O"]
        unique += ["I", "IO", "IO3"]
        self.traceH2O()
        self.traceH3O()
        self.traceOH()
        self.identifyIodine()
        self.identifyProton()
        if hasattr(self, "H"):
            print("'H' exists id: ", " ".join(str(h) for h in self.H))
        self.setDistanceDic("C", "O")
        unique.extend(carbonate)
        self.except_C_index = []
        for c in carbonate:
            self.traceCO3(c)
        self.except_C_index = None
        self.traceCaOH()
        self.traceSiO4()
        for u in unique:
            if hasattr(self, u):
                id_u = getattr(self, u)
                if id_u.size != 0:
                    if u in ("OH", "H3O"):
                        if id_u.ndim == 2:
                            id_O = id_u[:, 0]
                        elif id_u.ndim == 1:
                            id_O = id_u[0]
                        isin = self.isinBulk(id_O)
                        num = np.sum(isin)
                    elif u == "H":
                        isin = self.isinBulk(id_u)
                        num = np.sum(isin)
                    else:
                        num = id_u.shape[0]
                    if num != 0:
                        print(f"Number of {u:<6s}: {num}")
                    count = getattr(self, f"num{u}")
                    count += num
                    setattr(self, f"num{u}", count)
        self.checkUnlabledElem()

        
    def traceCluster(self):
        if hasattr(self, "zeropoint"):
            COORDINATION.zeropoint = self.zeropoint
        for i in range(self.bool_cluster.shape[0]):
            idx_O = np.where(self.bool_cluster[i, :])[0]
            if not hasattr(self, f"ca{i}"):
                setattr(self, f"ca{i}", COORDINATION())
                ca = getattr(self, f"ca{i}")
                new_idx_O = None
            else:
                ca = getattr(self, f"ca{i}")
                pre_idx = getattr(ca, "idx_O")
                not_changed = np.isin(idx_O, pre_idx)
                if not_changed.all():
                    new_idx_O = None
                else:
                    new_idx_O = idx_O[~not_changed]
            setattr(ca, "idx_O", idx_O.copy())
            id_Ca = self.aqueousCa[i]
            ca.Ca = id_Ca
            ca.z_Ca = self.data[id_Ca, 3]
            id_O = self.idDic["O"][idx_O]
            ca.tag = np.zeros_like(id_O, dtype=object)
            multy_id = []
            self.clusterSymbol[id_Ca] = "so"
            self.clusterSymbol[id_O] = "so"
            for i in range(id_O.shape[0]):
                id_O_ = id_O[i]
                o = id_O_.reshape(1, )
                if o in multy_id:
                    continue
                symbol, id_ = self.inwhichLabel("O", o, return_id=True)
                # self.clusterSymbol[id_] = "so"
                if id_.shape[0] == 2:
                    id_ = id_[0, :].reshape(1, -1)
                if new_idx_O is not None and id_O_ in self.idDic["O"][new_idx_O]:
                    self.clusterSymbol[id_] = "nc"
                if "CO3" in symbol:
                    id_other = id_O[id_O != o]
                    same_O = np.isin(id_, id_other)
                    if same_O.any():
                        multy_O = id_[same_O]
                        num_cn = np.sum(same_O)
                        assert num_cn == 1 or num_cn == 2, "Coordination Number must to be in between 1-3"
                        multy = "bi" if num_cn == 1 else "tri"
                        symbol += multy
                        ca.tag[np.nonzero(np.isin(id_O, multy_O))] = symbol
                        multy_id.extend(multy_O)
                    id_C_ang = id_[0, 0]
                    id_O_ang = id_[0, 1:4]
                    posCa = self.data[id_Ca][1:]
                    posC = self.data[id_C_ang][1:]
                    posO = self.data[id_O_ang][:, 1:]
                    assert posO.shape == (3, 3) and posC.shape == (3, ) and posCa.shape == (3, )
                    for o in posO:
                        ang_Ca_O_C, dist_Ca_O, _ = self.calAngle(posCa, o, posC, return_dist=True)
                        dist_flag = dist_Ca_O <= 3.0
                        if dist_flag:
                            COORDINATION.Ca_O_C.append(ang_Ca_O_C)
                    ang_Ca_C_O, dist_Ca_C, _ = self.calAngle(posCa, posC, o, return_dist=True)
                    COORDINATION.Ca_C.append(dist_Ca_C)
                if symbol == "H2O":
                    self.symbol[id_] = "wc"
                symList = getattr(ca, symbol)
                symList.append(id_)
                ca.tag[i] = symbol
            layer_id_O_lst = getattr(ca, "layerSiO4")
            layerCN = len(layer_id_O_lst)
            if layerCN == 0 or layerCN == 1:
                pass
            elif layerCN == 2:
                setattr(ca, "layerSiO4", list())
                setattr(ca, "layerSiO4bi", layer_id_O_lst)
            elif layerCN == 3:
                setattr(ca, "layerSiO4", list())
                setattr(ca, "layerSiO4tri", layer_id_O_lst)
            elif layerCN == 4:
                setattr(ca, "layerSiO4", list())
                setattr(ca, "layerSiO4quad", layer_id_O_lst)
            elif layerCN == 5:
                setattr(ca, "layerSiO4", list())
                setattr(ca, "layerSiO4penta", layer_id_O_lst)
            ca.cnStat()
        COORDINATION.outputCaLog()
        
    def outputDistribution(self, label=None, flag=None):
        if label is None:
            label_id = None
        elif hasattr(self, label):
            label_id = getattr(self, label)
        else:
            label_id = getattr(self, "HCO3")
        if label_id is not None:
            dim = label_id.ndim
            if label == "layerSiO4":
                id_ = np.unique(label_id[:, 1:])
            elif dim == 1:
                id_ = label_id
            elif dim == 2:
                id_ = label_id[:, 0]
            else:
                raise ValueError("wrong Dimension of Label")
            if label == "CO3":
                value_z = self.lmp.ldata[self.map_start:, self.idDic["C"], :][:, :, 3].ravel()
            else:
                value_z = self.lmp.ldata[self.map_start:, id_, :][:, :, 3].ravel()
        else:
            value_z = self.lmp.ldata[flag][:, 3].ravel()
        self.zeropoint = np.average(self.data[self.layerCa][:, 3])
        #Tmodel
        zero2bound = 12.69765
        #DBmodel
        zero2bound = 10.54289
        
        if hasattr(self, "shift"):
            zero2bound -= self.shift * self.lattice[2]
        boundary = self.zeropoint + zero2bound
        flag = np.where((value_z > self.zeropoint + 7.2) | (value_z < self.zeropoint - 7.2))[0]
        value_z = value_z[flag]
        value_z -= boundary
        self.value = np.where(value_z < -zero2bound, value_z + self.lattice[2], value_z)
        

        
    def zOverTime(self, z_id, zeroPoint=None):
        dim0 = z_id.shape[0]
        xdat = np.arange(0, self.nstep).astype(float)
        ydat = np.zeros((self.nstep, dim0), dtype=float)
        while self.process_active:
            lz = self.lattice[2]
            posZ = self.data[z_id, 3]
            fractZ = (posZ - zeroPoint) / lz
            fractZ -= np.floor(fractZ)
            assert fractZ.all() >= 0
            posZ = fractZ * lz
            ydat[self.current_step - 1] = posZ
            self.setStep()
        ydat = ydat.T

        self.ylim = np.max(self.lmp.lattice[:, 2])
        return xdat, ydat

    def numLayerOH(self):
        xdat = np.arange(0, self.nstep).astype(float)
        ydat = np.zeros((self.nstep, ), dtype=float)
        while self.process_active:
            self.setStep(reset_dic=True, reset_symbol=True)
            for e in self.elem:
                self.atomDic[e] = self.data[self.idDic[e]][:, 1:]
            self.setDistanceDic("O", "H")
            self.setDistanceDic("Si", "O")
            self.identifySiO4()
            layer_OH = np.where(self.symbol == "lH", True, False)
            num_OH = np.sum(layer_OH)
            ydat[self.current_step - 1] = num_OH
        return xdat, ydat

    def outSpeciesRatio(self):
        # avo = 6.0221408e+23
        avo = 1.136 # Density of water
        mol = 1000 / 18  # g / gpermol
        nH2OperLiter = avo * mol
        carbonate = ["CO3", "HCO3", "H2CO3", "H3CO3"]
        aqua = ["H", "OH", "H3O"]
        Liter = self.num_water_mole / nH2OperLiter
        if "C" in self.elem:
            en_car = self.nstep * self.num_C
            for c in carbonate:
                num = getattr(self, f"num{c}")
                print(c, ": total", num, f"{(num / en_car):.3f}")
                self.resultDic[c] = num / en_car
        for a in aqua:
            num = getattr(self, f"num{a}")
            con = num / self.nstep / Liter
            log_con = -np.log10(con) if num != 0 else 0
            print(a, ": total", num, f"{con:3e}", f"p{a}", f"{log_con:.3f}")
            if a == "OH":
                print(" -> pH", 14 - log_con)
                self.resultDic["pH"] = 14 - log_con
                self.resultDic["OH"] = con


    def adjust_trj(self):#ckpt
        offset_id = self.layerSiO4[0, 0]
        for i in range(self.nstep):
            self.lattice = self.lmp.lattice[i, :]
            self.angle = self.lmp.angle[i, :]
            self.setMatrix()
            self.lmp.ldata[i, :, 1:4] = self.lmp.ldata[i, :, 1:4] @ np.linalg.inv(self.matrix).T
            offset = self.lmp.ldata[i, offset_id, 1:4]
            self.lmp.ldata[i, :, 1:4] -= offset[np.newaxis, :]
            view = self.lmp.ldata[i, :, 1:4]
            view = np.where(view < 0, view + 1, view)
            self.lmp.ldata[i, :, 1:4] = view @ self.matrix.T
            self.lmp.ldata[i, :, 1] = np.mod(self.lmp.ldata[i, :, 1], self.lattice[0])
            
    def map_intralayer_CO3(self):#ckpt_co3
        value_list = list()
        tuple(self.intra_boundary[0])
        co3_species_list = []
        for w in ("CO3", "HCO3", "H2CO3"):
            if hasattr(self, w):
                co3_species_list.append(getattr(self, w)[:, :4])
        all_CO3 = np.vstack(co3_species_list)
        id_O = all_CO3[:, 1:4].ravel()
        for i in range(self.map_start, self.nstep):
            self.setStep(set_step=i)
            r = self.setDistanceDic("Ca", "O", center_id=self.intraCa, terminal_id=id_O, output_dic=True)
            ab_O = id_O[np.nonzero(np.where(r < self.rcut[("Ca", "O")], True, False))[1]]
            for co3 in all_CO3:
                if np.isin(co3, ab_O).any():
                    value_list.append([i, co3[0]])
        try:
            # value = np.vstack(value_list)
            return value_list
        except ValueError:
            print("Intralayer - C : Size 0")
            return None
            
    
class OPERATION():

    path_home = os.environ["HOME"]
    path_bin = os.path.join(path_home, "bin")
    startstep = 1
    infile = None

    fmt_xyz_head = "{}\nVESTA_phase_1                     {}\n"
    fmt_xyz_tail = "{:<2}  {:10.6f}  {:10.6f}  {:10.6f}\n"
    @classmethod
    def setInfile(cls, infile):
        LAMMPSTRJ.infile = infile
        cls.infile = infile

    def __init__(self):
        self.base = os.path.basename(os.getcwd())
        self.dirname = os.path.dirname(OPERATION.infile).replace("./", "")

    def set_interlayer(self, c):
        if not hasattr(c, "num_O") or getattr(c, "num_O") == 0:
            return
        if not hasattr(c, "num_H") or getattr(c, "num_H") == 0:
            return
        c.id_O_layer = np.unique(c.layerSiO4[:, 1:])
        adjust_value = 7
        layer_condition1 = (c.data[c.id_O_layer][:, 3] > c.zeropoint + adjust_value)
        layer_condition2 = (c.data[c.id_O_layer][:, 3] < c.zeropoint - adjust_value)
        flag = np.nonzero(layer_condition1 | layer_condition2)[0]
        c.id_O_outlayer = c.id_O_layer[flag]
        c.idx_O_outlayer = np.where(np.isin(c.idDic["O"], c.id_O_outlayer))[0]
        c.idx_O_layer = np.where(np.isin(c.idDic["O"], c.id_O_layer))[0]
        layer_H2O_condition1 = c.data[c.H2O[:, 0]][:, 3] < c.zeropoint + adjust_value
        layer_H2O_condition2 = c.data[c.H2O[:, 0]][:, 3] > c.zeropoint - adjust_value
        num_layer_H2O = np.sum(layer_H2O_condition1 & layer_H2O_condition2)
        num_water_mole = c.H2O.shape[0] - num_layer_H2O
        if hasattr(c, "H3O"):
            num_water_mole += getattr(c, "H3O").shape[0]
        if hasattr(c, "OH"):
            num_water_mole += getattr(c, "OH").shape[0]
        setattr(c, "num_water_mole", num_water_mole)
        print("Solution H2O: ", num_water_mole)
        print("Interlayer H2O: ",  num_layer_H2O)
        print("------------------------------")


    @checkTime
    def trace_bond(self):
        c = CALCULATION(startstep=self.startstep)
        c.resultDic = {}
        center_flag = np.isin(c.idDic["Ca"], c.aqueousCa)
        c.aqueousCa_idx = np.nonzero(center_flag)[0]
        if OPERATION.cluster:
            COORDINATION.dname = c.lmp.dname
            COORDINATION.ca_id = c.aqueousCa
            COORDINATION.ca_idx = c.aqueousCa_idx
            COORDINATION.initSetting()
        unique = ["CO3", "HCO3", "H2CO3", "H3CO3", "H", "OH", "H3O", "I", "IO", "IO3"]
        carbonate = ["CO3", "HCO3", "H2CO3", "H3CO3"]
        if hasattr(self, "inter_boundary") and len(c.inter_boundary) != 0:
            self.set_interlayer(c)
        elif hasattr(c, "num_O") and hasattr(c, "num_H"):
            setattr(c, "num_water_mole", c.H2O.shape[0])
            if "Si" in c.elem:
                c.id_O_layer = np.unique(c.layerSiO4[:, 1:])
                c.idx_O_layer = np.where(np.isin(c.idDic["O"], c.id_O_layer))[0]
                c.idx_O_outlayer = c.idx_O_layer.copy()
                c.id_O_outlayer = c.id_O_layer.copy()
        for u in unique:
            setattr(c, f"num{u}", 0)
        c.setVmdSymbol()
        if self.dump2mol:
            c.fname = f"{os.path.splitext(c.base)[0]}.mol.lammpstrj"
            if os.path.exists(c.fname):
                sp.run(["rm", c.fname])
                print("Delete", c.fname)
            c.dump2lmp()
        while True:
            for l in CALCULATION.O_label:
                value = getattr(c, l) if hasattr(c, l) and getattr(c, l) is not None else []
                setattr(c, f"p_{l}", value)
            start = time.time()
            if OPERATION.cluster:
                COORDINATION.nstep += 1
            c.setStep(reset_symbol=True)
            if not c.process_active:
                break
            c.symbol[c.aqueousCa] = "ca"
            if c.has_layer:
                c.symbol[c.layerCa] = "lca"
                c.symbol[c.layerSiO4] = "ll"
            for e in c.elem:
                c.atomDic[e] = c.data[c.idDic[e]][:, 1:]
            if hasattr(c, "num_O") and hasattr(c, "num_H"):
                c.setDistanceDic("O", "H")
                c.bool_bond = np.where(c.dDic[("O", "H")] < c.rcut[("O", "H")], True, False)
            if hasattr(c, "num_Ca") and hasattr(c, "num_O"):
                c.setDistanceDic("Ca", "O")
                c.bool_bond_CaO = np.where(c.dDic[("Ca", "O")] < c.rcut[("Ca", "O")], True, False)
            c.traceMoleStat()
            if OPERATION.cluster:
                c.setDistanceDic("Ca", "O", center_flag=center_flag)
                c.bool_cluster = np.where(c.dDic[("Ca", "O")] < c.rcut[("Ca", "O")], True, False)
                c.traceCluster()
            if "C" in c.elem:
                cnum = 0
                for car in carbonate:
                    if hasattr(c, car):
                        cv = getattr(c, car)
                        if cv is not None:
                            cnum += cv.shape[0]
                assert cnum == c.idDic["C"].shape[0], "Not all carbonate labeled"
            c.setVmdSymbol()
            c.anal_proton_exchange()
            end = time.time()
            print("Processing time:", f"{end - start:.2f}s")
            if self.dump2mol:
                c.dump2lmp()
        if OPERATION.cluster:
            COORDINATION.angle2npz()
            # COORDINATION.dataFrame2Excel()
        c.setVmdSymbol(output=True)
        c.outSpeciesRatio()
        c.resultDic["ex"] = c.ex
        with open("reactions.pkl", "wb") as f:
            pickle.dump(c.ex, f)
        np.savez("result.npz", **c.resultDic)
        for key, value in c.ex.items():
            if value != 0:
                print(key, value)

    def output_angle(self):
        CALCULATION.verbose = 2
        startstep = 4000
        # startstep = 1
        c = CALCULATION(startstep=startstep)
        while True:
            inputted = input("Allow chemical reactions? y/n\n").strip().lower()
            if inputted == "y":
                hco3_path = os.path.join(c.lmp.dname, "hq.index")
                if os.path.exists(hco3_path):
                    print("Reading 'hq.index' ...")
                    hco3_index = self.load_index(hco3_path)
                else:
                    raise ValueError("No exists 'hq.index'")
                co3_path = os.path.join(c.lmp.dname, "q.index")
                if os.path.exists(os.path.join(c.lmp.dname, "hq.index")):
                    print("Reading 'q.index' ...")
                    co3_index = self.load_index(co3_path)
                else:
                    raise ValueError("No exists 'q.index'")
                allow_transtrom = True
                break
            elif inputted == "n":
                allow_transtrom = False
                break
        else:
            print("Please enter 'y' or 'n'.")
        center_flag = np.ones(len(c.idDic["Ca"]), dtype=bool)
        c.aqueousCa = c.idDic["Ca"]
        c.aqueousCa_idx = np.nonzero(center_flag)[0]
        Ca_C_Ca, Ca_C, Ca_O_C, Ca_O = ([] for _ in range(4))
        h_Ca_C_Ca, h_Ca_C, h_Ca_O_C, h_Ca_O = ([] for _ in range(4))
        CO3_pair = []
        Ca_C_Ca_lst = list()
        if hasattr(c, "CO3"):
            CO3_pair.append(c.CO3)
        if hasattr(c, "HCO3"):
            if c.HCO3.ndim == 1:
                CO3_pair.append(c.HCO3[:4])
            elif c.HCO3.ndim == 2:
                CO3_pair.append(c.HCO3[:, :4])
        if hasattr(c, "H2CO3"):
            if c.HCO3.ndim == 1:
                CO3_pair.append(c.H2CO3[:4])
            elif c.HCO3.ndim == 2:
                CO3_pair.append(c.H2CO3[:, :4])
        CO3_pair = np.vstack(CO3_pair)
        while True:
            start = time.time()
            c.setStep(reset_symbol=True)
            for e in ["Ca", "C", "O"]:
                c.atomDic[e] = c.data[c.idDic[e]][:, 1:]
            if not c.process_active:
                break
            # c.setDistanceDic("Ca", "C", center_flag=center_flag)
            c.setDistanceDic("Ca", "C")
            c.setDistanceDic("C", "O")
            bond_matrix = np.where(c.dDic["Ca", "C"] < c.rcut["Ca", "C"], True, False)
            for i in range(bond_matrix.shape[1]):
                idx_Ca_aq = np.nonzero(bond_matrix[:, i])[0]
                id_C = c.idDic["C"][i]
                co3 = True
                if allow_transtrom:
                    if id_C in co3_index[c.current_step - 1]:
                        pass
                    elif id_C in hco3_index[c.current_step - 1]:
                        co3 = False
                    else:
                        continue
                pos_c = c.data[id_C, 1:]
                if np.sum(bond_matrix[:, i]) >= 2:
                    combs = list(itertools.combinations(idx_Ca_aq, 2))
                    twice = set()
                    for comb in combs:
                        id_ca1, id_ca2 = c.aqueousCa[comb[0]], c.aqueousCa[comb[1]]
                        pos_ca1, pos_ca2 = c.data[id_ca1, 1:], c.data[id_ca2, 1:]
                        ca_o_ca, r1, r2 = c.calAngle(pos_ca1, pos_c, pos_ca2, return_dist=True)
                        if co3:
                            Ca_C_Ca_lst.append([c.current_step - 1, id_ca1, id_C, id_ca2, ca_o_ca])
                        (Ca_C_Ca if co3 else h_Ca_C_Ca).append(ca_o_ca)
                        target_list = Ca_C if co3 else h_Ca_C
                        for idx, r in zip(comb, [r1, r2]):
                            if idx not in twice:
                                target_list.append(r)
                        twice.update(comb)
                    
                elif np.sum(bond_matrix[:, i]) == 1:
                    for ca in idx_Ca_aq:
                        pos_ca = c.data[c.aqueousCa[ca], 1:]
                        ca_c = c.calDist(pos_c, pos_ca)
                        (Ca_C if co3 else h_Ca_C).append(ca_c) 
                id_O = CO3_pair[CO3_pair[:, 0] == id_C][0, 1:]
                pos_o_ = c.data[id_O][:, 1:]
                for ca in idx_Ca_aq:
                    pos_ca = c.data[c.aqueousCa[ca], 1:]
                    for pos_o in pos_o_:
                        ca_o_c, ca_o, _ = c.calAngle(pos_ca, pos_o, pos_c, return_dist=True)
                        if ca_o < c.rcut["Ca", "O"]:
                            (Ca_O_C if co3 else h_Ca_O_C).append(ca_o_c)
                            (Ca_O if co3 else h_Ca_O).append(ca_o)
            end = time.time()
            overwritePrint(f"Processing time: {end - start:.2f}s")
        np.savez(os.path.join(c.lmp.dname, "angle.npz"), Ca_C_Ca=Ca_C_Ca, Ca_C=Ca_C, Ca_O_C=Ca_O_C, Ca_O=Ca_O,
                 h_Ca_C_Ca=h_Ca_C_Ca, h_Ca_C=h_Ca_C, h_Ca_O_C=h_Ca_O_C, h_Ca_O=h_Ca_O)
        print("angle.npz was created")
        try:
            Ca_C_Ca_arr = np.vstack(Ca_C_Ca_lst)
            Ca_C_Ca_arr = Ca_C_Ca_arr[Ca_C_Ca_arr[:, 4].argsort()]
            Ca_C_Ca_arr[:, :4] = Ca_C_Ca_arr[:, :4].astype(int)
            Ca_C_Ca_log = "step id_Ca id_C id_Ca angle\n" + "\n".join(" ".join(str(int(x)) if isinstance(x, float) and x.is_integer() else f"{x:.3f}"for x in arr) for arr in Ca_C_Ca_arr) 
            with open(os.path.join(c.lmp.dname, "angle.log"), "w") as o:
                o.write(Ca_C_Ca_log.strip())
                print("angle.log was created")
        except ValueError:
            print("no log")


    def output_symbol_label(self):
        CALCULATION.verbose = 2
        c = CALCULATION(startstep=self.startstep)
        elements = c.data[:, 0]
        if hasattr(c, "bt_top"):
            elements[c.bt_top] = "BT"
        if hasattr(c, "pt_top"):
            elements[c.pt_top] = "PT"
        np.save("symbol.npy", elements)
        print("'symbol.npy' was created")


    def output_tetra_xyz(self):#ckpt_xyz
        CALCULATION.verbose = 2
        self.map_start = 4000
        c = CALCULATION(startstep=self.startstep)
        flag = np.zeros((c.natoms, ), dtype=bool)
        # c.adjust_trj()
        elements = c.data[:, 0]
        if hasattr(c, "bt_top"):
            elements[c.bt_top] = "Bt"
            flag[c.bt_top] = True
        if hasattr(c, "pt_top"):
            elements[c.pt_top] = "Pt"
            flag[c.pt_top] = True
        if hasattr(c, "layerCa"):
            flag[c.layerCa] = True
        if hasattr(c, "layerSiO4"):
            flag[np.unique(c.layerSiO4.ravel())] = True
        if hasattr(c, "layerH"):
            flag[c.layerH] = True
        avg = np.average(c.lmp.ldata[self.map_start:, flag, 1:], axis=0)
        data = np.hstack([elements[flag].reshape(-1, 1), avg])
        output = OPERATION.fmt_xyz_head.format(flag.shape[0], self.base)
        for d in data:
            output += OPERATION.fmt_xyz_tail.format(*d)
        with open(os.path.join(c.lmp.dname, f"{c.base}.xyz"), "w") as o:
            o.write(output.strip())
        print(f"'{c.base}.xyz' was created")

        
    def output_map(self):#ckpt_Map
        CALCULATION.verbose = 2
        self.map_start = 4000
        arr_list = list()
        if os.path.exists(os.path.join(self.dirname, "ca.log")):
            with open(os.path.join(self.dirname, "ca.log")) as o:
                read = o.read().strip()
                read = read.split("\n\n")
                for r in read:
                    step = re.findall("Step: (\d+)", r)[0]
                    step = int(step)
                    if step < self.map_start - 1:
                        continue
                    found = re.findall("id: (\d+), .*layerSiO4", r)
                    arr = np.zeros((len(found), 2), dtype=int)
                    arr[:, 0] = step
                    arr[:, 1] = found
                    arr_list.append(arr)
                    del arr
        else:
            raise ValueError("not any ca.log")
        c = CALCULATION(startstep=self.startstep)
        mask_Ca = (c.data[c.idDic["Ca"]][:, 3] > c.layermax + 10)
        if np.sum(mask_Ca) >= 1:
            indices = np.nonzero(mask_Ca)[0]
            argsort = c.data[c.idDic["Ca"][mask_Ca]][:, 3].argsort()
            index_unique_Ca = c.idDic["Ca"][indices[argsort][0]]
        elif np.sum(mask_Ca) == 0:
            print("!ValueError!")
            threshold = c.matrix[2, 2] * 0.3
            mask_Ca = (c.data[c.idDic["Ca"]][:, 3] > threshold)
            if np.sum(mask_Ca) >= 1:
                indices = np.nonzero(mask_Ca)[0]
                argsort = c.data[c.idDic["Ca"][mask_Ca]][:, 3].argsort()
                index_unique_Ca = c.idDic["Ca"][indices[argsort][0]]
            else:
                raise ValueError
        index_C = c.idDic["C"][0]

        
        c.map_start = self.map_start
        hco3_path = os.path.join(c.lmp.dname, "hq.index")
        co3_path = os.path.join(c.lmp.dname, "q.index")
        if os.path.exists(co3_path):
            q_index_list = [[i, lst] for i, lst in enumerate(self.load_index(co3_path)) if i > self.map_start - 1]
        else:
            raise ValueError("not q.index")
        if os.path.exists(hco3_path):
            hq_index_list = [[i, lst] for i, lst in enumerate(self.load_index(hco3_path)) if i > self.map_start - 1]
        else:
            raise ValueError("not hq.index")
        c.b_gamma = np.sin(np.radians(c.angle[2]))
        arr_list_C = c.map_intralayer_CO3()
        # c.adjust_trj()
        flag = np.zeros((c.nstep, c.natoms), dtype=bool)
        flag_unique_Ca = np.zeros((c.nstep, c.natoms), dtype=bool)
        flag_C = np.zeros((c.nstep, c.natoms), dtype=bool)
        flag_HC = np.zeros((c.nstep, c.natoms), dtype=bool)
        for arr in arr_list:
            for a in arr:
                print(a[1], index_unique_Ca)
                if a[1] == index_unique_Ca:
                    flag_unique_Ca[tuple(a)] = True
                else:
                    flag[tuple(a)] = True
        for a in arr_list_C:
            flag_C[tuple(a)] = True
            if any(id_ == a[0] and a[1] in ids for id_, ids in q_index_list):
                flag_C[tuple(a)] = True
            else:
                print("not CO3", end="")
                if any(id_ == a[0] and a[1] in ids for id_, ids in hq_index_list):
                    flag_HC[tuple(a)] = True
                    print("yes HCO3", tuple(a), c.lmp.ldata[*a])
                else:
                    print("????")
                
        Ca = c.lmp.ldata[flag][:, 1:]
        uniqueCa = c.lmp.ldata[flag_unique_Ca][:, 1:]
        intraC = c.lmp.ldata[flag_C][:, 1:]
        intraHC = c.lmp.ldata[flag_HC][:, 1:]
        PT = c.lmp.ldata[self.map_start:, c.pt_top, 1:4] if hasattr(c, "pt_top") else None
        PTlow = c.lmp.ldata[self.map_start:, c.pt_low, 1:4] if hasattr(c, "pt_low") else None
        BT = c.lmp.ldata[self.map_start:, c.bt_top, 1:4] if hasattr(c, "bt_top") else None
        BTlow = c.lmp.ldata[self.map_start:, c.bt_low, 1:4] if hasattr(c, "bt_low") else None
        lattice = np.average(c.lmp.lattice[self.map_start:, :], axis=0)
        print()
        print("cos(gamma): ", np.sin(np.radians(c.angle[2])))
        lattice[1] *= c.b_gamma
        np.savez("map.npz", Ca=Ca, intraC=intraC, intraHC=intraHC, PT=PT, BT=BT, PTlow=PTlow, BTlow=BTlow, lattice=lattice, uniqueCa=uniqueCa)
        print("'map.npz' created")

    def output_adp(self):
        CALCULATION.verbose = 2
        def adp_by_index(species):
            index_file = os.path.join(self.dirname, param_dic[species]["fname"])
            listed = self.load_index(index_file)
            flag = np.zeros((c.nstep, c.natoms), dtype=bool)
            c_index = c.idDic[param_dic[species]["c_elem"]]
            for i in range(c.map_start, len(listed)):
                indexs = listed[i]
                if len(indexs) % param_dic[species]["nmol"] != 0:
                    print(f"num: {len(indexs)} with step {i}")
                    print(" ".join(str(d) for d in indexs))
                    continue
                indexs = np.array(indexs, dtype=int).reshape(-1, param_dic[species]["nmol"])
                indexs = c_index[np.isin(c_index, indexs)]
                flag[i, indexs] = True
            c.outputDistribution(flag=flag)
            return c.value
        c = CALCULATION(startstep=self.startstep)
        c.map_start = 4000
        hco3_path = os.path.join(c.lmp.dname, "hq.index")
        co3_path = os.path.join(c.lmp.dname, "q.index")
        param_dic = {}
        param_dic["CO3"], param_dic["HCO3"], param_dic["OH"]  = {}, {}, {}
        param_dic["CO3"]["fname"], param_dic["HCO3"]["fname"] = co3_path, hco3_path
        param_dic["CO3"]["nmol"], param_dic["HCO3"]["nmol"] = 4, 5
        param_dic["CO3"]["c_elem"], param_dic["HCO3"]["c_elem"] = "C", "C"
        param_dic["OH"]["fname"], param_dic["OH"]["nmol"], param_dic["OH"]["c_elem"] = "sw.index", 2, "O"
        co3 = adp_by_index("CO3")
        hco3 = adp_by_index("HCO3")
        oh = adp_by_index("OH")
        c.outputDistribution(label="aqueousCa")
        ca = c.value
        c.value = None
        h2o = c.outputDistribution(label="H2O")
        h2o = c.value
        c.value = None
        np.savez("adp.npz", co3=co3, hco3=hco3, oh=oh, ca=ca, h2o=h2o)
        print("adp.npz was created")
        
def main():
    description = """
    In basic,
    Prints the chemical reaction status by running

    'bond_identifier.py [infile]'

    or can make output files for analyzing
    """.strip()
    par = argparse.ArgumentParser(description=description, prog="bond")
    par.add_argument(
        '-m', '--mode', choices=["stat", "trace_bond", "symbol_label", "adp", "map", "angle", "tetra_xyz"], default="trace_bond",
        help="Choice operation(default trace_bond)")
    par.add_argument(
        'infile', nargs="?",
        help="Input lammpstrj file (if not provided, it will be selected automatically)")
    par.add_argument('-v', '--verbose', choices=[0, 1, 2], type=int, default=1,
        help="set verbosity level by number")
    par.add_argument(
        "-s", '--startstep', type=int, default=1,
        help="Start step of processing")
    par.add_argument(
        "-i", '--init', type=int, default=0,
        help="Run processing while init nstep")
    par.add_argument(
        '--cluster', action="store_true", default=False,
        help="Perform cluster analyzing ")
    par.add_argument(
        '-d', '--dump2mol', action="store_false", default=True,
        help="Dump inputted trj to another trj file which include column 'mol")
    args = par.parse_args()
    
    
    order = ["startstep", "cluster", "verbose", "dump2mol"]
    for r in order:
        setattr(OPERATION, r, getattr(args, r))
        setattr(CALCULATION, r, getattr(args, r))
    
    if args.init != 0:
        LAMMPSTRJ.show_init = args.init + args.startstep

    if args.mode in ["symbol_label", "adp", "map", "angle", "tetra_xyz"]:
        args.mode = f"output_{args.mode}"

    infile = glob.glob("*lammpstrj")[0] if not args.infile else  args.infile
    if hasattr(OPERATION, args.mode):
        OPERATION.setInfile(infile)
        operation = OPERATION()
        OPERATION.output = True
        getattr(operation, args.mode)()
    else:
        print(f"Error: {args.mode} is not a valid mode.")

if __name__ == "__main__":
    main()
