from .structure import StructureManager
from .common import get_logger
from .equation import Fit
import numpy as np
import importlib.resources as res
from contextlib import contextmanager

logger = get_logger(__name__)


class Builder:
    restriction = {}
    restriction["scope"] = ("half", "quarter", "one-third", "all")
    restriction["scope_value"] = (2, 4, 3, 1)
    restriction["add"] = ("Ca", "CaOH", "CaOH2")

    @contextmanager
    def update_manager(self, calculate_matrix=False):
        bf = len(self.csh.df)
        try:
            yield
        finally:
            self.csh.check_rm_list()
            af = len(self.csh.df)
            self.csh.update_data(calculate_matrix=calculate_matrix)
            if af != bf:
                logger.info(f"[Structure] Number of atoms: {bf} -> {af}")

    def __init__(self, **kwargs):
        for attr in ("model", "base", "out"):
            if attr in kwargs.keys():
                setattr(self, attr, kwargs[attr])
        self.set_model()

    def set_model(self):
        if hasattr(self, "model"):
            delattr(self, "model")
        self.base = self.base.replace(".cif", "") + ".cif"
        d = res.files("cshmd.data")
        self.infile = d.joinpath(f"./resources/cif/{self.base}")
        self.csh = StructureManager(self.infile)
        self.csh.outfile = self.out.replace(".cif", "") + ".cif"
        self.csh.verbose = False

    def supercell(self, supercell: list):
        """
        config option
        supercell: list = [2, 2, 1]
        """
        with self.update_manager(calculate_matrix=True):
            if not isinstance(supercell, list) and len(supercell) != 3:
                warning = "[ERROR] command 'supercell' must be list"
                warning += f"got' {supercell!r}"
                raise ValueError(warning)
            self.csh.makeSupercell(supercell)

    def len_add_extra_Ca(self, len_defect, len_site, MCL):
        param = Fit("Qfactor_csh_cong.csv", type="linear").params
        Q_factor = 2 / MCL
        target_CS = (Q_factor - param[1]) / param[0]
        total_defect = len_defect * len_site
        len_Ca = self.csh.elem_index["Ca"].shape[0]
        len_Si = self.csh.elem_index["Si"].shape[0]
        length = np.trunc(
            (target_CS * (len_Si - total_defect) - len_Ca - total_defect) / len_site
        )
        return length

    def defect_BT(
        self,
        where: str | list = "surface",
        MCL: float | None = None,
        termination: bool | str = True,
        add: str | None = None,
        add_interlayer_water: bool = False,
        correction_Q: bool = False,
        **typo_error,
    ):
        where = [where] if isinstance(where, str) else where
        direction_table = {
            ("surface", 0): "up",
            ("surface", 1): "down",
            ("bulk", 0): "down",
            ("bulk", 1): "up",
        }
        with self.update_manager():
            for key in where:
                BT = np.array(self.csh.tetra["BT"][key], dtype=int)
                if not isinstance(termination, bool) and termination != "half":
                    warning = "[ERROR] command 'termination' must be boolean or 'half, "
                    warning += f"got' {termination!r}"
                    raise ValueError(warning)
                else:
                    logger.info(f"[Structure] Termination scope: {termination!r}")
                    n_tetra = BT.shape[1]
                for i in range(BT.shape[0]):
                    # silicateの向き
                    index = i % 2
                    direction = direction_table[(key, index)]
                    msg = f"[Structure] Exists {n_tetra} Bridging Tetra in "
                    msg += f"{direction!r} side"
                    logger.info(msg)
                    if MCL is not None:
                        prune = self.csh.chain[key][index].prune_until_lenght(MCL)
                        if len(prune) == 0:
                            logger.warning("Already is in goal length")
                            return
                        else:
                            logger.info(f"Target MCL {MCL}, Remove {len(prune)} BT")
                        mask = np.isin(BT[i], prune)
                        target = BT[i][mask]
                    match key:
                        case "surface":
                            param = [0.5, 0, 1.8]
                            param = [[-0.68, -0.4, 1.22], [0.68, 0.4, 1.22]]
                            param_OH = [[1.50, -1, 2], [-1.50, 1, 2]]
                        case "bulk":
                            param = [[0, 0, 1], [0, 0, 1]]
                            param_OH = (
                                [[0, 2.1, -0.6], [0, -2.1, -0.6]]
                                if index == 1
                                else [[0, -2.1, -0.6], [0, 2.1, -0.6]]
                            )
                    param_H2O = [[2.2, 0.5, 0], [-2.2, -0.5, 0]]
                    add_option = {
                        "direction": direction,
                        "param": param,
                        "param_OH": param_OH,
                        "param_H2O": param_H2O,
                    }
                    if add_interlayer_water and key == "bulk":
                        # add_option["n_H2O"] = 2
                        add_option["n_H2O"] = 1
                    mole = add if add is not None else typo_error.get("mole", "Ca")
                    if key == "bulk":
                        mole = "CaOH"
                    add_option["mole"] = mole
                    logger.info(f"In {key}, add {mole} when Bridge Tetra removed.")
                    if correction_Q:
                        len_site = len(self.csh.tetra["BT"]["bulk"]) + len(
                            self.csh.tetra["BT"]["surface"]
                        )
                        len_add_extra_Ca = self.len_add_extra_Ca(
                            len_defect=len(target),
                            len_site=len_site,
                            MCL=self.csh.chain[(key, index)].mean_chain_length,
                        )
                        if len_add_extra_Ca < 0 and key == "bulk":
                            target = np.array(target)
                            n = int(-len_add_extra_Ca)
                            perm = np.random.permutation(len(target))
                            self.csh.remove_BT(target[perm[:n]], "half")
                            self.csh.remove_BT(
                                target[perm[n:]], termination, add_option
                            )
                        elif len_add_extra_Ca > 0 and key == "bulk":
                            target = np.array(target)
                            n = int(len_add_extra_Ca)
                            perm = np.random.permutation(len(target))
                            self.csh.remove_BT(target[n:], termination, add_option)
                            add_extra_option = add_option.copy()
                            add_extra_option["mole"] = "CaOH2"
                            add_extra_option["param"] = [-2.2, 0, 1]
                            add_extra_option["param_OH"] = (
                                [[0, 1.7, 0.5], [0, -1.7, 0.5]]
                                if index == 1
                                else [[0, -1.7, 0.5], [0, 1.7, 0.5]]
                            )
                            self.csh.remove_BT(
                                target[:n], termination, add_option, add_extra_option
                            )
                        else:
                            self.csh.remove_BT(target, termination, add_option)
                    else:
                        self.csh.remove_BT(target, termination, add_option)

    def add_BT(self, auto: bool = False, add: str | None = None):
        """
        config option
        auto: bool = False
        """
        with self.update_manager():
            BT = np.array(self.csh.tetra["BT"]["surface"], dtype=int)
            if BT.size == 0:
                logger.info("[WARNING] No Bridging Tetra exists")
                return
            if not isinstance(auto, bool):
                warning = "[ERROR] command 'add_ca_BT' must be boolean"
                warning += f"got' {auto!r}"
                raise ValueError(warning)
            else:
                if auto:
                    charge = self.csh.return_charge()
                    if charge >= 0:
                        logger.info("I found that we do not need to Ca ...")
                        return False
                    else:
                        required_Ca = int(charge / -4)
                        msg = f"Add {required_Ca} of Ca automatically "
                        msg += "in each side for 'Charge balancing'"
                else:
                    msg = "Add Ca above all Bridging Tetra"
                    required_Ca = "all"
                logger.info(f"[Structure] {msg!r}")
            n_tetra = BT.shape[1]
            for i in range(BT.shape[0]):
                direction = "up" if i == 0 else "down"
                msg = f"[Structure] Exists {BT.shape[1]}  in"
                msg += f"{direction!r} side of Bridging Tetra"
                if required_Ca != "all":
                    mask = np.random.choice(n_tetra, size=required_Ca, replace=False)
                    target = BT[i][mask]
                else:
                    target = BT[i]
                option = {}
                # option["param"] = [-1.5, 2.5, 4] if i == 0 else [-1.5, -2.5, 4]
                option["param"] = [[-1.5, 2.2, 1.22], [1.5, -2.2, 1.22]]
                option["param_OH"] = [[-1.7, 0.08, -2], [1.7, -0.08, -2]]
                if isinstance(add, str) and add in self.restriction["add"]:
                    match add:
                        case "Ca":
                            option["n_OH"] = 0
                        case "CaOH":
                            option["n_OH"] = 1
                        case "CaOH2":
                            option["n_OH"] = 2
                else:
                    raise ValueError(f"[ERROR] Wrong value! {add}")
                self.csh.add_Ca(target=target, direction=direction, **option)

    def dprot_BT(
        self, where: str | list = "surface", site: str = "all", scope: str = "all"
    ):
        site_choice = ("all", "upper", "lower")
        where = [where] if isinstance(where, str) else where
        with self.update_manager():
            for key in where:
                direction = "up" if key == "surface" else "down"
                BT = np.array(self.csh.tetra["BT"][key], dtype=int)
                if BT.size == 0:
                    logger.warning("No Bridging Tetra exists")
                    return
                if scope == "all":
                    for arr in BT:
                        kwargs = {"target": arr}
                        for attr in ("scope", "site", "direction"):
                            kwargs[attr] = locals()[attr]
                        self.csh.deprotonate_BT(**kwargs)
                        direction = "up" if direction == "down" else "down"
                else:
                    logger.warning("!!!!!Not yet developed!!!!!")
                    return

    def terminate_NBO(self, ratio=float, auto=False):
        if not auto and ratio == 0:
            logger.info("Do not run terminate, ratio == 0")
            return
        NBO = self.csh.find_NBO(return_only=True)
        mask_BT = np.isin(NBO, self.csh.surface_BT_O)
        intra_O = self.csh.Ca_O[np.isin(self.csh.Ca_O[:, 0], self.csh.intra_layer_ca)][
            :, 1
        ]
        surface_PT_O_not_in_intra = self.csh.surface_PT_O[
            ~np.isin(self.csh.surface_PT_O, intra_O)
        ]
        mask_PT = np.isin(NBO, surface_PT_O_not_in_intra)
        # Ca入ってる可能性
        # mask = mask_BT | mask_PT
        mask = mask_BT
        surface_NBO = NBO[mask]
        if np.sum(mask) == 0:
            logger.info("No termination site")
            return
        center_z = np.mean(self.csh.cao_range)
        upper_surface_NBO = surface_NBO[self.csh.data[surface_NBO, 3] > center_z]
        lower_surface_NBO = surface_NBO[self.csh.data[surface_NBO, 3] < center_z]
        n_NBO = upper_surface_NBO.shape[0]
        n_term = (
            -int(0.5 * self.csh.return_charge())
            if auto
            else np.round(n_NBO * ratio).astype(int)
        )
        logger.info(
            f"Terminate {ratio} ratio of silanol group -> {n_term} / {n_NBO} in 2sides"
        )
        target = []
        for indices in (upper_surface_NBO, lower_surface_NBO):
            selected = []
            points = self.csh.data[indices, 1:].astype(float)
            N = points.shape[0]
            idx = np.random.randint(N)
            selected.append(idx)
            target.append(indices[idx])
            distances = np.full(N, np.inf)
            for _ in range(n_term - 1):
                last = points[selected[-1]]
                delta = points - last
                delta -= self.csh.lattice * np.round(delta / self.csh.lattice)
                dist = np.linalg.norm(delta, axis=1)
                distances = np.minimum(distances, dist)
                idx = np.argmax(distances)
                selected.append(idx)
                target.append(indices[idx])
        self.csh.terminate(target)

    def out_model(self):
        self.csh.writeCif()
        charge = self.csh.return_charge()
        logger.info(f"Total Charge: {charge}")
