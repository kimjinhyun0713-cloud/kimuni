#!/usr/bin/env python
from pathlib import Path
from cshmd.statstic import Ab_stats, Stats
import sys

def main():
    PATH_ab = list(Path(".").glob("*/ab_dat.pkl"))
    PATH_data = list(Path(".").glob("*/dat.pkl"))
    if len(PATH_ab) == 0:
        PATH_ab = list(Path(".").glob("ab_dat.pkl"))
        PATH_data = list(Path(".").glob("dat.pkl"))
    if len(PATH_ab) == 0:
        print("[INFO] No data")
        sys.exit(1)
    for path in PATH_data:
        Stats(path)
        
    for path in PATH_ab:
        s = Ab_stats(path)
    s.set_type()
    print()
    fmt_ = "######### {:^20s} ##########"
    labels = [Ab_stats.total_co3_label,
              Ab_stats.total_ca_label,
              Ab_stats.total_all_label,
              Stats.total_stats]
    phrases = ["Absorption of 'H' within  CO3",
               "Absorption of 'Ca' within CO3",
               "Absorption of 'All Ca'",
               "Mole Stat"]
    for labelDic, phrase in zip(labels, phrases):
        print(fmt_.format(phrase))
        for k, v in labelDic.items():
            fmt = "{}: {:.1f}"
            print(fmt.format(k, v / Stats.counts))
    print()
if __name__ == "__main__":
    main()
