from .structure import STRUCTURE
import numpy as np
from dataclasses import dataclass
import importlib.resources as res
from contextlib import contextmanager

@dataclass
class BUILDER():
    
    model: dict
    base: str
    out: str | None = None
    
    @contextmanager
    def update_manager(self):
        bf = len(self.csh.df)
        try:
            yield
        finally:
            self.csh.check_rm_list()
            af = len(self.csh.df)
            self.csh.update_data()
            print(f"[Structure] Number of atoms: {bf} -> {af}")

    def value_check(self, **kwargs):
        """
        追加予定
        keywordの変数のvalueが正しいか確認
        """
        pass
            
    def set_model(self):
        self.base = self.base.replace(".cif", "") + ".cif"
        d = res.files("cshmd.data")
        self.infile = d.joinpath(f"./resources/cif/{self.base}")
        self.csh = STRUCTURE(self.infile, self.model)
        self.csh.outfile = self.out.replace(".cif", "") + ".cif"
        self.csh.verbose = False
            
    def supercell(self, supercell: list):        
        """
        config option
        supercell: list = [2, 2, 1]
        """
        with self.update_manager():
            if not isinstance(supercell, list) and len(supercell) != 3:
                warning = "[ERROR] command 'supercell' must be list"
                warning += f"got' {supercell!r}"
                raise ValueError(warning)
            self.csh.makeSupercell(supercell)
        

    def defect_BT(self,
                  scope: str | int = "all",
                  seed: int = 10,
                  termination: bool | str = True,
                  add_Ca: bool = False,
                  keep_OH: bool = False
                  ):
        """
        config option
        scope: str | int = "all"
        seed: int = 10
        termination: bool = True
        add_Ca: bool = False
        keep_OH: bool = False
        """
        with self.update_manager():
            BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
            if not isinstance(termination, bool) and termination != "half":
                warning = "[ERROR] command 'termination' must be boolean or 'half, "
                warning += f"got' {termination!r}"
                raise ValueError(warning)
            else:
                print(f"[Structure] Termination scope: {termination!r}")
                n_tetra = BT.shape[1]
            for i in range(BT.shape[0]):
                location = "top" if i == 0 else "bottom"
                msg = f"[Structure] Exists {n_tetra} Bridging Tetra in "
                msg += f"{location!r} side"
                print(msg)
                if scope == "all":
                    target = BT[i]
                elif scope == "half" or isinstance(scope, int):
                    n_del = n_tetra // 2 if scope == "half" else scope
                    if n_del > n_tetra:
                        warning = f"[ERROR] scope must be smaller than {n_tetra}"
                        raise ValueError(warning)
                    print(f"[Structure] Remove {n_del} Bridging Tetra of {location!r} side")
                    np.random.seed(seed)
                    mask = np.random.choice(n_tetra, size=n_del, replace=False)
                    target = BT[i][mask]
                else:
                    raise ValueError("[ERROR] Wrong Datatype")
                self.csh.remove_BT(
                    target,
                    termination,
                    add_Ca=add_Ca,
                    location=location,
                    keep_OH=keep_OH
                )

    def add_ca_BT(self, auto=False):
        """
        config option
        auto: bool = False
        """
        with self.update_manager():
            BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
            if BT.size == 0:
                print("[WARNING] No Bridging Tetra exists")
                return
            if not isinstance(auto, bool):
                warning = "[ERROR] command 'add_ca_BT' must be boolean"
                warning += f"got' {auto!r}"
                raise ValueError(warning)
            else:
                if auto:
                    charge = self.csh.return_charge()
                    if charge >= 0:
                        print("I found that we do not need to Ca ...")
                        return False
                    else:
                        required_Ca = int(charge / -4)
                        msg = f"Add {required_Ca} of Ca automatically "
                        msg += "in each side for 'Charge balancing'"
                else:
                    msg = "Add Ca above all Bridging Tetra"
                    required_Ca = "all"
                print(f"[Structure] {msg!r}")
            n_tetra = BT.shape[1]
            for i in range(BT.shape[0]):
                location = "top" if i == 0 else "bottom"
                msg = f"[Structure] Exists {BT.shape[1]}  in"
                msg += f"{location!r} side of Bridging Tetra"
                if required_Ca != "all":
                    mask = np.random.choice(n_tetra, size=required_Ca, replace=False)
                    target = BT[i][mask]
                else:
                    target = BT[i]
                add_param = [-1.5, 2, 1.7]
                self.csh.add_Ca(target=target, location=location, add_param=add_param)

                
    def dprot_BT(self, scope: str | int):
        with self.update_manager():
            BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
            if BT.size == 0:
                print("[WARNING] No Bridging Tetra exists")
                return
            if scope in ("upper", "lower", "all"):
                for i in range(BT.shape[0]):
                    if scope == "all":
                        location = "all"
                    else:
                        location = "top" if i == 0 else "bottom"
                    self.csh.deprotonate_BT(target=BT[i], scope=scope, location=location)
            else:
                print("NO")
        

    def out_model(self):
        self.csh.writeCif()
        charge = self.csh.return_charge()
        print(f"[INFO] Total Charge: {charge}")
