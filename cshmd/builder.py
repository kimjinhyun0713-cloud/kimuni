from .structure import STRUCTURE
import numpy as np
from dataclasses import dataclass
import importlib.resources as res

@dataclass
class BUILDER():
    
    model: dict
    base: str
    out: str

    def set_model(self):
        self.base = self.base.replace(".cif", "") + ".cif"
        d = res.files("cshmd.data")
        self.infile = d.joinpath(f"./template_cif/{self.base}")
        self.csh = STRUCTURE(self.infile, self.model)
        self.csh.outfile = self.out.replace(".cif", "") + ".cif"
        self.csh.verbose = False

            
    def supercell(self, supercell=[2, 2, 1]):
        """
        config option
        supercell: list = [2, 2, 1]
        """
        if supercell is None or len(supercell) == 0:
            return False
        self.csh.makeSupercell(supercell)

    def defect_BT(self,
                  scope: str | int = "all",
                  seed: int = 10,
                  termination: bool = True,
                  keep_OH: bool = False):
        """
        config option
        scope: str | int = "all"
        seed: int = 10
        termination: bool = True
        keep_OH: bool = False
        """
        BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
        for i in range(2):
            if scope == "all":
                target = BT[i]
            elif scope == "half" or isinstance(scope, int):
                n_tetra = BT[i].shape[0]
                n_del = n_tetra // 2 if scope == "half" else scope
                if n_del > n_tetra:
                    warning = f"ERROR: scope must be smaller than {n_tetra}"
                    raise ValueError(warning)
                print(f"[Structure] Remove {n_del} Bridging Teta")
                np.random.seed(seed)
                mask = np.random.choice(n_tetra, size=n_del, replace=False)
                target = BT[i][mask]
            else:
                raise ValueError("ERROR: Wrong Datatype")
            location = "top" if i == 0 else "bottom"
            self.csh.remove_BT(target, termination,
                               location=location, keep_OH=keep_OH)
        self.csh.remove()
                

    def out_model(self):
        self.csh.writeCif()
        charge = self.csh.return_charge()
        print(f"Total Charge: {charge}")

def main():
    import sys
    import yaml
    config = yaml.safe_load(sys.stdin.read())
    build = BUILDER(**config["info"])
    build.set_model()
    for c in config["command"]:
        getattr(build, c["func"])(**c["args"])
        build.csh.update_data()
    build.out_model()
        
if __name__ == "__main__":
    main()
