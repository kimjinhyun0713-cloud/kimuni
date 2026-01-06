import numpy as np
import pickle
from typing import Dict, List, Optional, Union
from .analysis import BOND, UnitCell
from .load import LAMMPSTRJ, Dump
from .functions import setMatrix
from .common import overwritePrint
from dataclasses import dataclass


@dataclass
class Absorption:
    
    C: int
    elem: str
    edge: Optional[Dict[int, int]] = None
    _label: Optional[Dict[str, int]] = None
    _Hs: Optional[List[int]] = None
    _Os: Optional[List[int]] = None
    
    def __getitem__(self, C):
        return self.edge[C]
    
    def __setitem__(self, C, val):
        if self.edge is None:
            self.edge = {}
        self.edge[C] = val

    def initalize_label(self):
        self._label = {}
        
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, val: tuple):
        if self._label is None:
            self._label = {}
        _label = val[0]
        _num = val[1]
        self._label[_label] = self._label.get(_label, 0) + _num
            
        
class ANALYZER(LAMMPSTRJ):
    verbose = True
    crystal_centered = False
    
    def __init__(self, infile: Optional[str], Index: Union[UnitCell, dict, None]):
        super().__init__(infile)
        self.infile = str(infile)
        if Index is None:
            pass
        elif isinstance(Index, dict):
            self.labels = Index.keys()
        else:
            self.labels = Index.label_keys
        for key in self.labels:
            setattr(self, key, np.array(Index[key], dtype=int))
        self.O_label = [key for key in self.labels if "O" in key]
        self.Si_label = [key for key in self.labels if "Si" in key]

    def _set_node_(self):
        self.co3_node = []
        self.ca_node = []
        C = np.nonzero(self._data[:, 0] == "C")[0]
        for index in C:
            self.co3_node.append(Absorption(index, "C", _label={}))
        bulk_mask = np.isin(self.bond.Ca_O[:, 0], self.bulk_Ca)
        ab_O = np.hstack([getattr(self, label) for label in self.O_label])
        ab_mask = np.isin(self.bond.Ca_O[:, 1], ab_O)
        mask = ab_mask & bulk_mask
        CaO_bond_on_layer = self.bond.Ca_O[mask]
        for Ca in np.unique(CaO_bond_on_layer[:, 0]):
            _node = Absorption(Ca, "Ca", _label={})
            for O in CaO_bond_on_layer[CaO_bond_on_layer[:, 0] == Ca][:, 1]:
                label = self._get_label_of_csh_O(O)
                if label is None:
                    msg = "[Error]: Label of 'O' doesnt exists"
                    raise ValueError(msg)
                _node.label = (label, 1)
            self.ca_node.append(_node)
        self._index_of_co3_node_ = {node.C: node for node in self.co3_node}
        self._index_of_ca_node_ = {node.C: node for node in self.ca_node}
        
    def _add_node_(self, _node):
        self.ca_node.append(_node)
        self._index_of_ca_node_ = {node.C: node for node in self.ca_node}
    
    def trace_step(self):
        self._manage_statistics_()
        for step in range(self.nstep):
            self.cur_step = step
            self._format_setting()
            self._update_step(step)
            self.__find_mole__()
            self._set_node_()
            self.analyze_absorption()# in case of calculate layer absorption
            self.stdout_stats()
        self.stdout_statistics()
        
    def center_step_(self):
        self._center_init_setting_()
        for step in range(self.nstep):
            self.cur_step = step
            self._format_setting()
            self._update_step(step)
            self._centering_()
        self.stdout_centering()

    def _center_init_setting_(self):
        self._update_step(0)
        self.bond = BOND(self._data, self._matrix, cartesian=True)
        self.bond.find("SiO4")
        SiO4 = np.unique(self.bond.SiO4).ravel()
        self.required_center_index = np.hstack([self.layer_Ca, SiO4])
        self.len_center_index = self.required_center_index.shape[0]
        self.dumper = Dump(natoms=self.natoms, elem=self.elem)
        if ANALYZER.crystal_centered:
            stdout = "[INFO] Adjust coordination to locate C-S-H in middle of the system"
        else:
            stdout = "[INFO] Adjust coordination to locate C-S-H in boundary of the system"
        print(stdout)
        Dump.set_dump_file(self.infile.replace(".lammpstrj", ".center.lammpstrj"))
        Dump.check_file_exists(remove=True)
        

    def _centering_(self):
        overwritePrint(self.stdout)
        all_cur_fract = self._data[:, 1:] @ np.linalg.inv(self._matrix)
        if ANALYZER.crystal_centered:
            c_shift = 0.5 - np.average(all_cur_fract[self.layer_Ca, 2])
            all_cur_fract[:, 2] += c_shift
        require_cur_fract = all_cur_fract[self.required_center_index]
        self.require_cur_fract = require_cur_fract.astype(float)
        cumsum_r = 0
        if hasattr(self, "require_pre_fract"):
            r = (self.require_cur_fract - self.require_pre_fract)
            r = (r - np.round(r))
            avg_r = np.average(r, axis=0)
            cumsum_r += avg_r
            r_ = (all_cur_fract - cumsum_r).astype(float)
            r_ -= np.floor(r_)
            view = r_ @ self._matrix
            self._data[:, 1:4] = view
        else:
            self._data[:, 1:] = all_cur_fract  @ self._matrix
        self.dumper.dump(self._data, self._matrix)
        self.require_pre_fract = self.require_cur_fract.copy()

    def stdout_centering(self):
        print(f"\n[INFO] '{Dump.lmp_name}' created")
        print("Run 'csh_make_map.py'")
        
    def _manage_statistics_(self):
        self.managed_moles = ["OH", "H3O", "CO3"]
        self.managed_moles += BOND.mole_hco3_family
        self.statistics = {mole: 0 for mole in self.managed_moles}
        self.statistics["free_Ca"] = 0
        self.statistics["absorbed_Ca"] = 0
        self.statistics["SiOH"] = 0
        self.statistics["node_of_co3_n_connected_ca"] = []
        self.statistics["node_of_whole_ca"] = []

    def stdout_statistics(self):
        exclude = {"node_of_co3_n_connected_ca", "node_of_whole_ca"}
        mole = {k: v / self.nstep
                for k, v in self.statistics.items()
                if k not in exclude}
        ab = [self.statistics[e] for e in exclude]
        with open("dat.pkl", "wb") as f:
            pickle.dump(mole, f)
            print("'dat.pkl' created")
        with open("ab_dat.pkl", "wb") as ff:
            pickle.dump(ab, ff)
            print("'ab_dat.pkl' created")
        print("Run 'csh_statistic.py'")

    def _format_setting(self):
        self.stdout = f"[STEP {self.cur_step:4d}] "
        self.fmt_stat_name = "{:11s}|"
        
            
    def _update_step(self, step: int = 0):
        self._data = self.ldata[step]
        self._lattice = self.lattice[step]
        self._angle = self.angle[step]
        self._matrix, _ = setMatrix(self._lattice, self._angle)

        
    def __find_mole__(self):
        self.bond = BOND(self._data, self._matrix, cartesian=True)
        self.bond.find_O_H(reverse=True, reverse_rcut=3.5)# calculate Hydrogen bonds
        self.bond.find_Ca_O()# in case of calculate layer absorption
        self.bond.find(*BOND.mole)
        self.bond.check_where_OH()

    def stdout_stats(self):
        self._compute_mole_stat()
        self._compute_ab_stat()
        self._compute_node_stat()

    def _compute_mole_stat(self):
        stdout = self.stdout + self.fmt_stat_name.format("Mole stat")
        fmt = "{}: {}  "
        for mole in self.managed_moles:
            if hasattr(self.bond, mole):
                val = getattr(self.bond, mole).shape[0]
                if val is not None:
                    if val != 0:
                        self.statistics[mole] += val
                        stdout += fmt.format(mole, val)
        print(stdout)

        
    def _compute_ab_stat(self):
        stdout = self.stdout + self.fmt_stat_name.format("Layer stat")
        fmt_ca = "freeCa: {}/{}  "
        fmt_sioh = "SiOH: {}"
        num_absorbed_ca = len(self.ca_node)# computed in '_set_node_'
        num_whole_ca = len(self.bulk_Ca)# computed in '_set_node_'
        num_sioh = len(self.bond.SiOH)# computed in '__find_mole__'
        self.statistics["free_Ca"] += (num_whole_ca - num_absorbed_ca)
        self.statistics["absorbed_Ca"] += num_absorbed_ca
        self.statistics["SiOH"] += num_sioh
        stdout += fmt_ca.format(num_absorbed_ca, num_whole_ca)
        stdout += fmt_sioh.format(num_sioh)
        print(stdout)

    def _compute_node_stat(self):
        stdout = self.stdout + self.fmt_stat_name.format("Node stat")
        labels = []
        for node in self.co3_node:
            labels.append(node.label)
            if node.edge is not None:
                for edge in node.edge:
                    ca_node = self._get_node_by_index_(edge, "ca")
                    labels.append(ca_node.label)
        for _node in self.ca_node:
            self.statistics["node_of_whole_ca"].append(_node.label)
            # print(_node.label, _node.C)
        self.statistics["node_of_co3_n_connected_ca"].append(labels)
        for labelDic in labels:
            if labelDic == {}:
                continue
            fmt_ = " |{}| "
            out_ = "  ".join(f"{k}: {v}" for k, v in labelDic.items())
            stdout += fmt_.format(out_)
        
        print(stdout)
        
    def analyze_absorption(self):
        for node in self.co3_node:
            self._setattr_co3_node(node)
            self._link_co3_to_ca(node)
            self._link_co3_to_sioh(node)
            self._link_hco3_to_sio(node)

    def _setattr_co3_node(self, node):
        node._Os = np.unique(self.bond.C_O[self.bond.C_O[:, 0] == node.C][:, 1])
        if node._Os.size != 3:
            raise ValueError("[Error]: No 'C-O' bonds exists")
        _Hs = np.unique(self.bond.O_H[np.isin(self.bond.O_H[:, 0], node._Os)][:, 1])
        node._Hs = _Hs if _Hs.size != 0 else None

    def _link_co3_to_ca(self, node):
        bond_Ca = self.bond.Ca_O[np.isin(self.bond.Ca_O[:, 1], node._Os)][:, 0]
        _Cas = np.unique(bond_Ca, return_counts=True)
        if len(_Cas[0]) != 0:
            _Cas = np.vstack(_Cas).T
            for (Cas, count) in _Cas:
                _node = self._get_node_by_index_(Cas, "ca")
                if _node is None:
                    _node = Absorption(Cas, "Ca", _label={"free": 1})
                    self._add_node_(_node)
                node[_node.C] = count
                _node[node.C] = count

    def _link_co3_to_sioh(self, node):
        if not hasattr(self.bond, "CO3"):
            return
        HO_mask_of_co3 = np.isin(self.bond.H_O[:, 1], node._Os)
        HO_mask_of_sioh = np.isin(self.bond.H_O[:, 0], self.bond.SiOH[:, 1])
        HO_mask = HO_mask_of_sioh & HO_mask_of_co3
        hydrogen_bond = self.bond.H_O[HO_mask]
        if hydrogen_bond.size != 0:
            for H_of_sioh in np.unique(hydrogen_bond[:, 0]):
                O_of_sioh = self.bond.SiOH[np.isin(self.bond.SiOH[:, 1], H_of_sioh)][:, 0]
                for O in O_of_sioh:
                    label = self._get_label_of_csh_O(O)
                    if label is not None:
                        label = label.replace("O", "H")
                        node.label = (label, 1)
                        print(node.label)
                    else:
                        msg = "[Error] The Label of silicate layer is not allocated"
                        raise ValueError(msg)
                
                
    def _link_hco3_to_sio(self, node):
        HO_mask = np.isin(self.bond.H_O[:, 0], node._Hs)
        hydrogen_bond = self.bond.H_O[HO_mask]
        if hydrogen_bond.size != 0:
            for O_of_sio in np.unique(hydrogen_bond[:, 1]):
                label = self._get_label_of_csh_O(O_of_sio)
                if label is not None:
                    label = label.replace("O", "H_")
                    node.label = (label, 1)
                    
            
    def _get_label_of_csh_O(self, index: int):
        for key in [key for key in self.labels if "O" in key]:
            val = getattr(self, key)
            if index in val:
                return key
        return None
    
    def _get_node_by_index_(self, index: int, mole: str): # mole: co3 | ca
        indexDic = getattr(self, f"_index_of_{mole}_node_")
        return indexDic.get(index, None)
        

    def vprint(self, string):
        if self.verbose:
            print(string)
            

class Run_trj_analysis():
    verbose = False
    crystal_centered = True
    
    def __init__(self, infile=None, mode="trace"):
        if infile is None:
            print("[INFO] No trj exists")
        ANALYZER.verbose = Run_trj_analysis.verbose
        ANALYZER.crystal_centered = Run_trj_analysis.crystal_centered
        self.analyzer = ANALYZER(infile, UnitCell(infile))
        getattr(self, mode)()

    def trace(self):
        self.analyzer.trace_step()

    def center(self):
        self.analyzer.center_step_()

        
def main():
    import argparse
    from pathlib import Path
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*",
                    help="#Infile")
    par.add_argument('-v', '--verbose', default=False, action="store_true",
                    help="set verbosity level by True 'or' 'False'")
    par.add_argument('-m', '--mode', choices=["trace", "center"], default="trace",
                     help=" Choose 'trace' or 'center'")
    par.add_argument('-c', '--crystal_centered', default=True, action="store_false",
                     help="set C-S-H in middle of the system")
    
    args = par.parse_args()
    ANALYZER.verbose = args.verbose
    ANALYZER.crystal_centered = args.crystal_centered
    infiles = args.infiles if len(args.infiles) != 0 else list(Path(".").rglob("e.lammpstrj"))
    for infile in infiles:
        Run_trj_analysis(infile, args.mode)
    return 0
    
if __name__ == "__main__":
    main()
