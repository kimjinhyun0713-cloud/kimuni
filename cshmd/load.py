import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from .functions import setMatrix, calLattice
from .common import overwritePrint, get_logger
from .common import list2arr
from .util import lmp_head

logger = get_logger(__name__)


class Pkl:
    def __init__(self, path):
        self.parent = path.resolve().parent
        self.name = str(path.with_suffix(""))
        self.title_name = f"{str(self.parent).split('/')[-1]} - {self.name}"
        self.df = pd.read_pickle(path)

    @property
    def value(self):
        return self.df

    @value.setter
    def value(self, val):
        self.df = val


class Npy:
    def __init__(self, path):
        self.parent = path.resolve().parent
        self.name = str(path.with_suffix(""))
        self.title_name = f"{str(self.parent).split('/')[-1]} - {self.name}"
        self.value = np.load(path, allow_pickle=True)
        print(self.value.shape)


class Dump:
    lmp_name = "dump.lammpstrj"

    @classmethod
    def set_dump_file(cls, name):
        cls.lmp_name = name
        logger.info(f"[INFO]: Set Dump file to '{name}'")

    @classmethod
    def check_file_exists(cls, remove=False):
        p = Path(cls.lmp_name)
        exists = p.exists()
        if exists:
            logger.info(f"[INFO] Dump file {cls.lmp_name} exists")
            if remove:
                logger.info(f"[INFO] Delete {cls.lmp_name}")
                p.unlink()
        return exists

    def __init__(self, mode="lmp", natoms=None, elem=None):
        self.natoms = natoms
        self.mode = mode
        self.elem = elem
        self.cur_step = 1
        if self.mode == "lmp":
            self._lmp_setting_()

    def _lmp_setting_(self):
        self.box_fmt = "{:.6e} {:.6e} {:.6e}\n"
        self.tail_fmt = "{:<5} {:<2} {:<3} {:.6f} {:.6f} {:.6f}\n"

    def dump(self, data=None, matrix=None):
        if self.mode == "lmp":
            self.dump_lmp(data, matrix)

    def dump_lmp(self, data=None, matrix=None):
        a_vec, b_vec, c_vec = matrix
        xy, xz, yz = b_vec[0], c_vec[0], c_vec[1]
        box_string = ""
        box_string += self.box_fmt.format(
            min(0, xy, xz, xy + xz), matrix[0, 0] + max(0, xy, xz, xy + xz), xy
        )
        box_string += self.box_fmt.format(min(0, yz), matrix[1, 1] + max(0, yz), xz)
        box_string += self.box_fmt.format(0, matrix[2, 2], yz)
        stdout = (
            lmp_head.format((self.cur_step), self.natoms, box_string.strip()) + "\n"
        )
        index = 0
        for d in data:
            id_ = int(np.nonzero(self.elem == d[0])[0]) + 1
            stdout += self.tail_fmt.format(index, id_, *d)
            index += 1
        with open(Dump.lmp_name, "a") as o:
            o.write(stdout)


def load_yaml(infile):
    import yaml

    with open(infile) as o:
        data = yaml.load(o, Loader=yaml.FullLoader)
    return data


def poscar2Dic(infile):
    with open(infile) as o:
        base = o.readline().strip()
        dic = {"base": base}
        o.readline()
        lst = []
        for _ in range(3):
            lst.append(o.readline())
        matrix = [list(map(float, l.split())) for l in lst]
        matrix = list2arr(matrix)
        dic["matrix"] = matrix
        abc = np.linalg.norm(matrix, axis=1)
        dic["lattice"] = abc
        a, b, c = abc
        angle = []
        angle.append(np.arccos(matrix[1, :] @ matrix[2, :] / (b * c)) * 180 / np.pi)
        angle.append(np.arccos(matrix[0, :] @ matrix[2, :] / (a * c)) * 180 / np.pi)
        angle.append(np.arccos(matrix[0, :] @ matrix[1, :] / (a * b)) * 180 / np.pi)
        dic["angle"] = angle
        elem = o.readline().split()
        dic["elem"] = elem
        n_elem = [int(v) for v in o.readline().split()]
        o.readline()
        data = []
        for i, n in enumerate(n_elem):
            lst = []
            arr = np.empty((n, 4), dtype=object)
            arr[:, 0] = elem[i]
            for _ in range(n):
                lst.append([float(v) for v in o.readline().split()])
            arr[:, 1:4] = lst
            data.append(arr)
        dic["data"] = np.vstack(data)
        return dic


class LammpsStep:
    def __init__(
        self,
        lammpstrj_path: str,
        target_frames: list | None = None,
        to_fract: bool = False,
        every=None,
        start=None,
    ):
        self.path = Path(lammpstrj_path).resolve()
        self.base = self.path.parent
        self.name = str(self.path).split("/")[-1].replace(".lammpstrj", "")
        self.yielder = self.iter_lammpstrj(
            lammpstrj_path, target_frames, to_fract, every, start
        )

    def __iter__(self):
        return self.yielder

    def iter_lammpstrj(
        self, path, target_frames=None, to_fract=False, every=None, start=None
    ):
        def is_target(i):
            if i == 0:
                return True
            if target_frames is not None:
                if i in target_frames:
                    return True
            if i < start:
                return False
            return (i - start) % every == 0

        frame_index = 0
        target_frames = set(target_frames) if target_frames else None
        with open(path) as f:
            timestep = None
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("ITEM: TIMESTEP"):
                    timestep = int(f.readline())
                    if not is_target(frame_index):
                        while not f.readline().startswith("ITEM: ATOMS"):
                            pass
                        # natoms = int(f.readline())  # dummy read
                        for _ in range(natoms):
                            f.readline()
                        frame_index += 1
                        continue
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    natoms = int(f.readline())
                elif line.startswith("ITEM: BOX BOUNDS"):
                    lattice = np.fromfile(f, count=9, sep=" ").reshape(3, 3)
                    lattice, angle, zeropoint = calLattice(lattice)
                    matrix, _ = setMatrix(lattice, angle)
                    inv_matrix = np.linalg.inv(matrix)
                elif line.startswith("ITEM: ATOMS"):
                    cols = line.split()[2:]
                    idx = {c: i for i, c in enumerate(cols)}
                    raw = np.loadtxt(
                        (f.readline() for _ in range(natoms)), dtype=object
                    )
                    xyz = raw[:, [idx["x"], idx["y"], idx["z"]]].astype(float)
                    xyz -= zeropoint
                    if to_fract:
                        xyz = xyz @ inv_matrix
                    data = np.hstack([raw[:, idx["element"]].reshape(-1, 1), xyz])
                    logger.info(f"Frame number: {frame_index}")
                    yield {
                        "timestep": timestep,
                        "frame_index": frame_index,
                        "natoms": natoms,
                        "data": data,
                        "lattice": lattice,
                        "angle": angle,
                        # "columns": cols,
                        "data": data,
                        "matrix": matrix,
                    }
                    frame_index += 1


def lmp2Dic(lammpstrj, to_fract=False):
    """
    convert lammpstrj to dictionary of which havs or local values

    args
    ;; lammpstrj

    return
    ;; locals()
    """
    with open(lammpstrj) as o:
        readlines = o.readlines()
        natoms = int(readlines[3])
        ncol = len(readlines[8].split()) - 2
        nlines = natoms + 9
        nstep = len(readlines) / nlines
        assert nstep == int(nstep), (
            f"Please check the file is completely written nstep={nstep}"
        )
        nstep = int(nstep)
        ldata = np.zeros((nstep, natoms, 4), dtype=object)
        lattice = np.zeros((nstep, 3), dtype=float)
        angle = np.zeros((nstep, 3), dtype=float)
        logger.info(f"Loading {lammpstrj}")
        for s in range(nstep):
            skip = s * nlines
            lattice_ = np.array(
                list(map(lambda l: l.split(), readlines[skip + 5 : skip + 8])),
                dtype=float,
            )
            lattice[s, :], angle[s, :], zeropoint = calLattice(lattice_)
            data_ = readlines[skip + 9 : skip + nlines]
            data_ = " ".join(data_).strip().split()
            data_ = np.array(data_, dtype=object).reshape(-1, ncol)
            data_[:, 3:6] = data_[:, 3:6].astype(float)
            data_[:, 3] -= zeropoint[0]
            data_[:, 4] -= zeropoint[1]
            data_[:, 5] -= zeropoint[2]
            ldata[s, :, :] = data_[:, 2:6]
            overwritePrint(f"[INFO] Store step: {s}")
        print()
        elem = np.unique(data_[:, 2])
        type_ = np.unique(data_[:, 1])
        if to_fract:
            matrix_list = []
            for i in range(ldata.shape[0]):
                matrix_, _ = setMatrix(lattice[i, :], angle[i, :])
                matrix_list.append(matrix_)
                ldata[i, :, 1:4] = ldata[i, :, 1:4] @ np.linalg.inv(matrix_)
            lmatrix = np.vstack(matrix_list).reshape(-1, 3, 3)
            del matrix_list
        keyword = [
            "nlines",
            "ldata",
            "natoms",
            "nstep",
            "lattice",
            "angle",
            "elem",
            "ldata",
            "lmatrix",
            "ncol",
        ]
        dic = {k: v for k, v in locals().items() if k in keyword}
        return dic


def excel2data(infile):
    """
    convert excel(.xlsx) data to Pd.dataFrame
    args:
    ;; infile -> .xlsx

    return
    pd.DataFrame
    """
    with pd.ExcelFile(infile) as o:
        sheet_names = o.sheet_names
        for sheet in sheet_names:
            df = o.parse(sheet)
    return df


def excel2Dic(infile):
    dic = {}
    with pd.ExcelFile(infile) as o:
        sheet_names = o.sheet_names
        for sheet in sheet_names:
            dic[sheet] = o.parse(sheet)
    return dic


def cif2data(infile, cartesian=False):
    """
    convert cif data to pd.DateFrame includes whole columns in cif.
    For example symbol, occupancy

    args
    ;; infile -> str, infile
    ;;; cartesian  -> bool, convert fractional coordinations to cartesian(default: False)

    returns
    lattice -> list
    angle -> list
    data -> pd.DateFrame
    """
    with open(infile) as o:
        assert os.path.splitext(infile)[-1] == ".cif", "Make sure to put 'CIF'"
        read = o.read()
        lattice_ptn = r"_cell_length_[abc]+ +([0-9\.]+)"
        angle_ptn = r"_cell_angle_[a-zA-Z]+ +([0-9\.]+)"
        lattice = [float(v) for v in re.findall(lattice_ptn, read)]
        angle = [float(v) for v in re.findall(angle_ptn, read)]
        ptn_col = r"_atom_site_(.*)"
        col = re.findall(ptn_col, read)
        col = [c.strip() for c in col]
        data = read.split(col[-1])[-1]
        data = np.array(
            [s.split() for s in data.strip().split("\n")], dtype=object
        ).reshape(-1, len(col))
        data = pd.DataFrame(data, columns=col)
        data["fract_x"] = data["fract_x"].astype(float)
        data["fract_y"] = data["fract_y"].astype(float)
        data["fract_z"] = data["fract_z"].astype(float)
        if cartesian:
            matrix, _ = setMatrix(lattice, angle)
            data[["fract_x", "fract_y", "fract_z"]] = (
                data[["fract_x", "fract_y", "fract_z"]] @ matrix
            )
        return lattice, angle, data


class LAMMPSTRJ:
    """
    attribute:
    "elem", "nstep", "natoms", "ldata", "lattice", "angle" "matrix"
    """

    lmp_dict_col = ("elem", "nstep", "natoms", "ldata", "lattice", "angle", "lmatrix")

    def __init__(self, infile=None, col=None, to_fract=False):
        if infile is None:
            logger.warning("No trj")
        assert os.path.splitext(infile)[-1] == ".lammpstrj", (
            "Make sure to put lammpstrj"
        )
        dataDic = lmp2Dic(infile, to_fract=to_fract)
        self.lmp_dict_col = self.__class__.lmp_dict_col if col is None else col
        for key in self.lmp_dict_col:
            setattr(self, key, dataDic[key])
        if dataDic["ncol"] == 6:
            self.mol_lmp = False
        elif dataDic["ncol"] == 7:
            self.mol_lmp = True
        else:
            raise ValueError(
                "Sorry for did not considering of the system which dump values bigger than '7'"
            )

    def __str__(self):
        string = "\n"
        if hasattr(self, "natoms"):
            string += f"Number of atoms: {self.natoms}"
        if hasattr(self, "nstep"):
            string += f"\nNumber of steps: {self.nstep}"
        return string
