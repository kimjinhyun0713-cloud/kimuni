#!/usr/bin/env python
import sys
import yaml
from cshmd.builder import Builder

class Typo_checker():
    def __init__(self, config):
        self._config = config
        self._check_outfile()
        
    def _check_outfile(self):
        if "out" not in self.config["info"].keys():
            out = self._config["name"].replace("yaml", "").replace("yml", "")
            self._config["info"]["out"] = out
            
    @property
    def config(self):
        return self._config

def main():
    import argparse
    par = argparse.ArgumentParser(description="", prog="")
    par.add_argument('infiles', nargs="*", help="#@ yaml")
    args = par.parse_args()
    for infile in args.infiles:
        print("-"*70)
        with open(infile) as yml:
            config = yaml.safe_load(yml)
            config["name"] = infile
            tc = Typo_checker(config)
            config = tc.config
            build = Builder(**config["info"])
            build.set_model()
            for c in config["command"]:
                print(f"\n [StructureManager] Reading command '{c['func']}'...")
                getattr(build, c["func"])(**c["args"])
            build.out_model()
        print("-"*70)
        
if __name__ == "__main__":
    main()
