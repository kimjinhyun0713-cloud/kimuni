import hydra
import importlib.resources as res

@hydra.main(config_path="data", config_name="T14_base", version_base=None)
def config_base(cfg):
    if hasattr(cfg, "BT"):
        print("[Structure] ", cfg.BT)

def data_lib(fname):
    d = res.files("jinh_csh.data")
    infile = d.joinpath(fname)
    return infile
