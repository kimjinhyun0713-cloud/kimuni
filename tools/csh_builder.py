#!/usr/bin/env python
import sys
import yaml
from cshmd.builder import BUILDER

def main():
    config = yaml.safe_load(sys.stdin.read())
    build = BUILDER(**config["info"])
    build.set_model()
    for c in config["command"]:
        getattr(build, c["func"])(**c["args"])
        build.csh.update_data()
    build.out_model()
        
if __name__ == "__main__":
    main()
