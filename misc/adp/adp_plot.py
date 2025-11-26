#!/usr/bin/env python
import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse
import os
import re

par = argparse.ArgumentParser(description="test", prog="PROG")
par.add_argument('infile', nargs="*", help="one-dim values in format *.npz"
                 "(if not provided, it will be selected automatically)")
par.add_argument('-m', '--mode', choices=["a", "c", "w"], default="c", help="Choice mode, all, carbonate, water")
args = par.parse_args()
#pltdic.update()
# plt.tight_layout()
# plt.rcParams["text.usetex"] = True

mode_c = False
mode_w = False
nplot = None

class PLOT():
    iter_color = (c for c in ["black", "green", "red", "orange", "gray", "pink", "b", "yellow"])
    axs = None
    color = None
    lift_dic = {}
    
    @classmethod
    def setting(cls, nplot=5):
        nplot = int(nplot)
        cls.fig = plt.figure(figsize=(7 * nplot, 7))
        cls.axs = np.zeros((nplot, ), dtype=object)
        space = 1 / nplot
        for i in range(nplot):
            cls.axs[i] = cls.fig.add_axes([space * i + space * 0.1, 0.1, space * 0.8, 0.7]) 
    
    def setColor(self):
        if self.base == "DB9":
            self.color = "red"
        elif self.base == "T11":
            self.color = "orange"
        elif self.base == "DB14":
            self.color = "blue"
        elif self.base == "T13":
            self.color = "skyblue"
        elif self.base == "T12":
            self.color = "green"
        elif self.base == "DP11":
            self.color = "pink"
        elif self.base == "DP14":
            self.color = "purple"
        else:
            self.color = next(PLOT.iter_color)
    
    def __init__(self, npz_file):
        abspath = os.path.abspath(npz_file)
        self.base = re.search("([TBDP]{1,2}[0-9]{1,2})", abspath).group(1)
        self.npz_data = np.load(npz, allow_pickle=True)
        self.paramDic = {}
        if self.base == "T11":
            offset = 46.22867 - 9.84870
        if self.base == "DB9":
            offset = 43.19429 - 10.45980
        if self.base == "DB14":
            offset = 42.43882 - 10.65900
        if self.base == "T13":
            offset = 45.62926 - 10.54790
        if self.base == "DP11":
            offset = 43.56573 - 10.14340
        if self.base == "DP14":
            offset = 43.10224 - 10.27670
        self.offset = offset
        if mode_w:
            self.paramDic["oh"], self.paramDic["h2o"] = {}, {} 
            self.paramDic["oh"]["xlim"],  self.paramDic["oh"]["ylim"], self.paramDic["oh"]["title"] = (-1, 7), (0, 1), "$\mathrm{OH}^{-}$"
            self.paramDic["h2o"]["xlim"],  self.paramDic["h2o"]["ylim"],  self.paramDic["h2o"]["title"] = (-1, 7) , (0, 0.14), "$\mathrm{H}_2\mathrm{O}$"
            self.paramDic["oh"]["xlim"],  self.paramDic["oh"]["ylim"], self.paramDic["oh"]["title"] = (-1, offset + 1), (0, 1), "$\mathrm{OH}^{-}$"
            self.paramDic["h2o"]["xlim"],  self.paramDic["h2o"]["ylim"],  self.paramDic["h2o"]["title"] = (-1, offset + 1) , (0, 0.12), "$\mathrm{H}_2\mathrm{O}$"
            
        if mode_c:
            self.paramDic["ca"], self.paramDic["co3"], self.paramDic["hco3"] = {}, {}, {}
            self.paramDic["ca"]["xlim"], self.paramDic["ca"]["ylim"], self.paramDic["ca"]["title"]  = (-1, offset + 1), (0, 0.5), "$\mathrm{Ca}^{2+}$" 
            self.paramDic["co3"]["xlim"], self.paramDic["co3"]["ylim"], self.paramDic["co3"]["title"] = (-1, offset + 1) , (0, 0.5), "$\mathrm{CO}_3^{2-}$"
            self.paramDic["hco3"]["xlim"] , self.paramDic["hco3"]["ylim"],  self.paramDic["hco3"]["title"] = (-1, offset + 1), (0, 0.5), "$\mathrm{HCO}_3^{-}$"  
        def sort_order(label):
            return order.get(label, 10)
        order = {"ca": 2, "co3": 0, "hco3": 1, "h2o": 3, "oh": 4}
        keys = sorted(self.npz_data.files, key=lambda x: sort_order(x))
        keys = sorted(self.paramDic.keys(), key=lambda x: sort_order(x))
        for i, key in enumerate(keys):
            if i >= nplot:
                continue
            self.key = key
            self.p_num = i
            if key not in PLOT.lift_dic.keys():
                PLOT.lift_dic[key] = 0
            else:
                PLOT.lift_dic[key] += 0.8
            # print(key, PLOT.lift_dic[key])
            self.adp_plot()

            
            
        
    def adp_plot(self):
        value = self.npz_data[self.key]
        value = value[value != 0]
        self.setColor()
        if value.shape[0] < 10:
            return
        self.value = value
        bin_edges = np.arange(-4, 14 + 0.2, 0.2)
        bin_edges = np.arange(-4, 54 + 0.2, 0.2)
        hist, bin_edges = np.histogram(self.value, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist += PLOT.lift_dic[self.key]
        PLOT.axs[self.p_num].plot(bin_centers, hist, color=self.color, label=self.base, linewidth=1.5)

        PT = 2.15476
        BT = -2.2863
        offset = self.offset
        PLOT.axs[self.p_num].fill_between([0, PT], -2, 10, color='lightgray', hatch="/", alpha=0.3, zorder=1, label="BT")
        PLOT.axs[self.p_num].fill_between([0 + offset, - PT + offset], -2, 10, color='lightgray', hatch="/", alpha=0.3, zorder=1, label="BT")
        PLOT.axs[self.p_num].fill_between([ -2 + BT, 0], -2, 10, color='slategray',hatch="\\", alpha=0.3, zorder=1, label="PT")
        PLOT.axs[self.p_num].fill_between([offset, -BT + offset], -2, 10, color='slategray',hatch="\\", alpha=0.3, zorder=1, label="PT")
        # PLOT.axs[self.p_num].set_title(f"ADP of {self.paramDic[self.key]['title']}")
        PLOT.axs[self.p_num].set_xlim(self.paramDic[self.key]["xlim"][0], self.paramDic[self.key]["xlim"][1])
        PLOT.axs[self.p_num].set_ylabel(f"ADPs({self.paramDic[self.key]['title']})")
        PLOT.axs[self.p_num].set_xlabel(r"Distance / $\rm{\AA}$ ")
        PLOT.axs[self.p_num].set_ylim(self.paramDic[self.key]["ylim"][0], self.paramDic[self.key]["ylim"][1] * 5)
        PLOT.axs[self.p_num].tick_params(labelleft=False)
        # PLOT.axs[self.p_num].set_aspect('equal') 
        

    def plot_show(self):
        def sortLabel(labels):
            return priority.get(labels, 3)
        handles, labels = PLOT.axs[-1].get_legend_handles_labels()
        try:
            uniqueDict = dict(zip(labels, handles))
            labels, handles = zip(*uniqueDict.items())
            zipped = zip(handles, labels)
            priority = {"T11": 5, "T12": 6, "T13": 7, "DB9": 1, "DB14": 2, "DP11": 3, "DP14": 4}
            priority = {k: -v for (k, v) in priority.items()}
            priority["pocket"] = 8
            priority["PT"] = 9
            priority["BT"] = 10
            zippedLst = list(zipped)
            newPairLst = sorted(zippedLst, key=lambda x: sortLabel(x[1]))
            handles, labels = zip(*newPairLst)
        except ValueError:
            pass
        # PLOT.axs[-1].legend(handles, labels, bbox_to_anchor=(0.8, 1.1))
        PLOT.axs[-1].legend(handles, labels)
        filename = "adp.png"
        plt.show()
        PLOT.fig.savefig(filename)
        print(f"{filename} is created")
        PLOT.fig.patch.set_alpha(0)
        # plt.gca().set_aspect('equal', adjustable='box')

    
            

if len(args.infile) == 0:
    npzs = glob.glob("*npz")
else:
    npzs = args.infile

inputted = args.mode
if inputted == "a":
    nplot = 5
    mode_c = True
    mode_w = True
elif inputted == "c":
    nplot = 3
    mode_c = True
else:
    nplot = 1
    mode_w = True
    
PLOT.setting(nplot=nplot)
for npz in npzs:
    a = PLOT(npz)
a.plot_show()

