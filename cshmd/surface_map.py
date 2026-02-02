import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from .load import LAMMPSTRJ
from .analysis import UnitCell

class Map_Data(LAMMPSTRJ):
    
    def __init__(self, path: str, Index: UnitCell):
        super().__init__(path)
        self.path = path
        self.path_base = self.path.name
        self.path_resolve = path.resolve()
        self.par_path = path.parent
        self.map_path = self.par_path / f"{self.path_base}_map.npz"
        self.BT_Si_upper = np.array(Index["BT_Si"]["upper"], dtype=int)
        self.BT_Si_lower = np.array(Index["BT_Si"]["lower"], dtype=int)
        self.PT_Si_upper = np.array(Index["PT_Si"]["upper"], dtype=int)
        self.PT_Si_lower = np.array(Index["PT_Si"]["lower"], dtype=int)
        self.bulk_Ca = Index["bulk_Ca"]
        self.mapDic = {}
        self.stdout_map()
        
    def _get_elem_index(self, elem: str):
        matched = self.ldata[0, :, 0] == elem
        return matched if np.any(matched) else None
        
    def stdout_map(self):
        self.mapDic["Ca_xyz"] = self.ldata[:, self.bulk_Ca, 1:4]
        index_I = self._get_elem_index("I")
        if index_I is not None:
            self.mapDic["I_xyz"] = self.ldata[:, index_I, 1:4]
        for key in ("PT_Si_upper", "PT_Si_lower", "BT_Si_upper", "BT_Si_lower"):
            indexs = getattr(self, key)
            if indexs.size != 0:
                value = self.ldata[:, indexs, 1:3]
                base = np.average(self.ldata[:, indexs, 3])
                self.mapDic[f"{key}_xy"] = value
                self.mapDic[f"base_{key}"] = np.array(base, dtype=float)
                print(f"[INFO]: Added '{key}' to map")
            else:
                self.mapDic[f"{key}_xy"] = np.empty(0)
                self.mapDic[f"base_{key}"] = np.empty(0)
                print(f"[INFO]: Added '{key}' to map(size == 0)")
        np.savez(self.map_path, **self.mapDic)
        print(f"'{self.map_path}' is created")
        print("Run 'csh_plot_map.py'")

        
class Mapper():
    merged_Dic = {}
    base_setted = False
    
    @classmethod
    def get_map(cls, key):
        value = cls.merged_Dic[key]
        if "base" in key:
            return np.average(value) if not all(v.size == 0 for v in value) else None
        else:
            if not all(v.size == 0 for v in value):
                axis2 = value[0].shape[2]
                return np.vstack(value).reshape(-1, axis2)
            else:
                return np.empty(0)
    
    @classmethod
    def get_base_line(cls):
        val_list = []
        for key in ("base_PT_Si_upper",
                    "base_PT_Si_lower",
                    "base_BT_Si_upper",
                    "base_BT_Si_lower"):
            value = cls.get_map(key)
            if value is not None:
                val_list.append(value)
        return np.array(sorted(val_list), dtype=float)
            
    def __init__(self, path=None):
        if path is not None:
            self.path = path
            self.path_resolve = path.resolve()
            self._get_npz_()
        
    def _get_npz_(self):
        for key, value in np.load(self.path, allow_pickle=True).items():
            getted = Mapper.merged_Dic.get(key, None)
            if getted is None:
                Mapper.merged_Dic[key] = [value]
            else:
                Mapper.merged_Dic[key].append(value)


                
class Plotter():
    cwd = None

    @property
    def cutoff_size(self):
        return self._cutoff_size
    
    @cutoff_size.setter
    def cutoff_size(self, val):
        print(f"[INFO] CUTOFF setted to {val}")
        self._cutoff_size = val
    
    def __init__(self, path, mapper):
        self.fname = str(path).replace(".npz", "").replace(".lammpstrj_map", "")
        self.mapper = mapper
        self.where = "upper"
        self._cutoff_size = 6
        self.base = self.mapper.__class__.get_base_line()
        for e in ("Ca", "I"):
            value = self.mapper.get_map(f"{e}_xyz")
            setattr(self, f"{e}_z", value[:, 2])
            setattr(self, f"{e}_xy", value[:, 0:2])
        self.path_stdout = Plotter.cwd / "map.png"
        self.set_contour_colorbar = False

    def _set_cutoff_(self):
        self.cutoff = []
        self.cutoff.append(np.max(self.base) + self.cutoff_size)
        self.cutoff.append(np.min(self.base) - self.cutoff_size)
        print(f"[INFO] CUTOFF of adp: {self.cutoff_size}")
        
        
    def set_adp(self, **setup):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_alpha(0)
        plt.tight_layout()
        if setup.get("plot_Ca", False):
            self._adp_elem("Ca")
        if setup.get("plot_I", False):
            self._adp_elem("I")
            
    def _adp_elem(self, elem):
        value = getattr(self, f"{elem}_z")
        min_, max_ = np.min(value), np.max(value)
        bin_edges = np.arange(int(min_), np.ceil(max_) + 0.05, 0.05)
        hist, bin_edges = np.histogram(value, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if self.base.size != 0:
            ymax, ymin = np.max(hist), np.min(hist)
            self._set_cutoff_()
            self.ax.vlines(x=self.base, ymin=ymin,  ymax=ymax, color="red", linestyle="--", label="Silicate Chain")
            self.ax.vlines(x=self.cutoff, ymin=ymin, ymax=ymax, color="green", linestyle="--", label="Cutoff")
        self.ax.plot(bin_centers, hist, label="Ca distribution")
        self.ax.legend()
        plt.show()
        self.path_stdout = Plotter.cwd / f"{self.fname}_{elem}_adp.png"
        self.fig.savefig(self.path_stdout)
        print(f"Adp plot: '{str(self.path_stdout).split('/')[-1]}' is created")


        
    def set_contour(self, **setup):
        self.fig, self.ax = plt.subplots(2, figsize=(5, 9))
        for i in range(2):
            self.ax[i].set_aspect('equal', adjustable='box')
            self.ax[i].set_xlim(-5, 40)
            self.ax[i].set_ylim(-5, 30)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        self.fig.patch.set_alpha(0)
        self._set_cutoff_()
        colorbar = setup.get("colorbar", False)
        self.set_contour_colorbar = colorbar
        if setup.get("plot_Ca", True):
            self._contour_elem("Ca")
        if setup.get("plot_I", True):
            self._contour_elem("I")
        if setup.get("plot_Si", False):
            self._contour_Si()
        plt.show()
        self.fig.savefig(self.path_stdout)
        print(f"Contour plot: '{str(self.path_stdout).split('/')[-1]}' is created")


    def _contour_Si(self):
        for i in range(2):
            PT_key = f"PT_Si_{self.where}_xy"
            BT_key = f"BT_Si_{self.where}_xy"
            self.where = "lower" if self.where == "upper" else "lower"
            cmap = "plasma"
            for key in (BT_key, PT_key):                
                value = self.mapper.__class__.get_map(key)
                if value.size == 0:
                    continue
                X = value[:, 0]
                Y = value[:, 1]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=150)
                xcenters = (xedges[:-1] + xedges[1:]) / 2
                ycenters = (yedges[:-1] + yedges[1:]) / 2
                Xg, Yg = np.meshgrid(xcenters, ycenters, indexing="ij")
                Z = np.where(counts == 0, np.nan, counts)
                self.ax[i].contour(Xg, Yg, Z, levels=10, cmap=cmap)
                cmap = "winter" if cmap == "plasma" else "winter"
            self.ax[i].set_xlabel("X")
            self.ax[i].set_ylabel("Y")

        self.path_stdout = Plotter.cwd / f"{self.fname}_map_Si.png"


    def _contour_elem(self, elem):
        cdic = {"Ca": "turbo", "I": "brg"}
        cmap = cdic[elem]
        upper_base = self.mapper.get_map("base_PT_Si_upper")
        lower_base = self.mapper.get_map("base_PT_Si_lower")
        z = getattr(self, f"{elem}_z")
        xy = getattr(self, f"{elem}_xy")
        mask_upper = np.where(
            (z < self.cutoff[0]) & (z > upper_base),
            True, False)
        mask_lower = np.where(
            (z > self.cutoff[1]) & (z < lower_base), 
            True, False)
        xy_upper = xy[mask_upper]
        xy_lower = xy[mask_lower]
        for i, value in enumerate([xy_upper, xy_lower]):
            X = value[:, 0]
            Y = value[:, 1]
            counts, xedges, yedges = np.histogram2d(X, Y, bins=150)
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            Xg, Yg = np.meshgrid(xcenters, ycenters, indexing="ij")
            Z = np.where(counts == 0, np.nan, counts)
            base_number = 10
            vmin, vmax = 0.01, 500
            steps = 3
            log_min = np.floor(np.log(vmin) / np.log(base_number))
            log_max = np.ceil(np.log(vmax) / np.log(base_number))
            levels_custom = np.logspace(
                log_min, 
                log_max, 
                num=int(log_max - log_min) * steps + 1, 
                base=base_number
            )

            cp = self.ax[i].contourf(
                Xg, Yg, Z,
                levels=levels_custom,
                cmap=cmap,
                norm=colors.LogNorm()
            )
            locator = ticker.LogLocator(base=base_number)
            if self.set_contour_colorbar:
                cbar = self.fig.colorbar(cp, ax=self.ax[i])
                cbar.locator = locator
                cbar.formatter = ticker.LogFormatterMathtext(base=base_number)
                cbar.update_ticks()

        self.path_stdout = Plotter.cwd / f"{self.fname}_map_{elem}.png"
