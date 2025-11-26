import hydra

@hydra.main(config_path="data", config_name="T14_base", version_base=None)
def config_base(cfg):
    if hasattr(cfg, "BT"):
        print("[Structure] ", cfg.BT)

class HYDRA():
    def __init__(self, config):
        self.config = config
        
    def run(self):
        with hydra.initialize(config_path="data"):
            cfg = hydra.compose(config_name=self.config)
        print(cfg)
