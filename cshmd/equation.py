from .common import list2arr, get_logger
import pandas as pd
import re
import io
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import importlib.resources as res

logger = get_logger(__name__)


class Csv:
    def __init__(self, path):
        self.path = path
        self.getData(path)

    def getData(self, outfile):
        with open(outfile) as o:
            body = o.read()
        body = re.sub("^#.*: ", "", body)
        body = re.sub(",", "", body)
        self.df = pd.read_csv(io.StringIO(body), sep=r"\s+")


class Fit(Csv):
    def __init__(self, path_name, type=None):
        d = res.files("cshmd.data")
        path = d.joinpath(f"./resources/csv/{path_name}")
        super().__init__(path)
        self.df = self.df.sort_values(by=self.df.columns[0])
        self.x = self.df.iloc[:, 0].to_numpy()
        self.y = self.df.iloc[:, 1].to_numpy()
        self.fit_type = None
        if type is None:
            logger.debug("No fit type specified. Data only mode.")
            pass
        elif type == "hyperbolic":
            self.perform_fit(initial_guess=[1.0, 0.5, 2.0])
        elif type == "linear":
            self.perform_fit_linear()
        elif type == "spline":
            self.perform_spline_fit()
        elif type == "rdf":
            self.perform_damped_oscillation()
        else:
            raise ValueError("Invalid fit type. Use 'hyperbolic' or 'linear'.")

    @property
    def params(self):
        return self.popt

    def __call__(self, xlim=None, data_only=False):
        if data_only:
            if xlim is not None:
                mask = (self.x > xlim[0]) & (self.x < xlim[1])
                return self.x[mask], self.y[mask]
            else:
                return self.x, self.y

        if xlim is not None:
            x_fit = np.linspace(xlim[0], xlim[1], 100)
        else:
            x_fit = np.linspace(min(self.x), max(self.x), 100)
        if self.fit_type == "linear":
            y_fit = self.linear_func(x_fit, *self.popt)
        elif self.fit_type == "hyperbolic":
            y_fit = self.hyperbolic_func(x_fit, *self.popt)
        elif self.fit_type == "spline":
            y_fit = self.spline(x_fit)
        elif self.fit_type == "damped_oscillation":
            y_fit = self.damped_oscillation(x_fit, *self.popt)
        else:
            raise ValueError("フィッティングが実行されていません。")

        return x_fit, y_fit

    @staticmethod
    def hyperbolic_func(x, a, b, c):
        return a / (x - b) + c

    @staticmethod
    def linear_func(x, a, b):
        return a * x + b

    @staticmethod
    def damped_oscillation(x, A, xi, k, phi):
        return 1 + A * np.exp(-x / xi) * np.sin(k * x + phi)

    def perform_fit(self, initial_guess=None):
        try:
            popt, pcov = curve_fit(
                self.hyperbolic_func, self.x, self.y, p0=initial_guess
            )
            print("✓ 双曲線フィッティング成功")
            print(
                f"得られたパラメータ: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}"
            )
            self.popt = popt
            self.pcov = pcov
            self.fit_type = "hyperbolic"  # フィット種類を記録
            return popt

        except RuntimeError as e:
            print(f"エラー: 双曲線の最適化が収束しませんでした。\n詳細: {e}")
            return None

    def perform_fit_linear(self, initial_guess=None):
        try:
            popt, pcov = curve_fit(self.linear_func, self.x, self.y, p0=initial_guess)
            print("✓ 線形フィッティング成功")
            print(f"得られたパラメータ: a(傾き)={popt[0]:.4f}, b(切片)={popt[1]:.4f}")
            self.popt = popt
            self.pcov = pcov
            self.fit_type = "linear"  # フィット種類を記録
            return popt

        except RuntimeError as e:
            print(f"エラー: 線形の最適化が収束しませんでした。\n詳細: {e}")
            return None

    def perform_spline_fit(self, smooth=0.1):
        """
        smooth:
            小さい → 元データに強く追従
            大きい → なめらか
        """
        self.spline = UnivariateSpline(self.x, self.y, s=smooth)
        self.fit_type = "spline"
        # self.y_fit = self.spline(self.x)
        # logger.info(f"smoothing factor = {smooth}")
        # return self.spline

    def perform_damped_oscillation(self, initial_guess=(2.0, 3.0, 6.0, 0.0)):
        self.fit_type = "damped_oscillation"
        popt, pcov = curve_fit(
            self.damped_oscillation, self.x, self.y, p0=initial_guess, maxfev=10000
        )

        self.popt = popt
        self.pcov = pcov

        # self.y_fit = self.damped_oscillation(
        #     self.x,
        #     *popt
        # )

        print("✓ damped oscillation fitting success")
        print(
            f"A   = {popt[0]:.4f}\n"
            f"xi  = {popt[1]:.4f}\n"
            f"k   = {popt[2]:.4f}\n"
            f"phi = {popt[3]:.4f}"
        )

        return popt

    def plot_result(self):
        if not hasattr(self, "popt"):
            print("先に perform_fit() を実行してください。")
            return

        plt.scatter(self.x, self.y, label="Original Data", color="black")
        x_fit = np.linspace(min(self.x), max(self.x), 100)
        y_fit = self.hyperbolic_func(
            x_fit, *self.popt
        )  # *popt で [a,b,c] をアンパックして渡す

        plt.plot(x_fit, y_fit, label="Hyperbolic Fit", color="purple")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()


def eq_water_density(temp: np.ndarray | list, version: int = 0) -> np.ndarray:
    """
    ref: Kell, George S. "Density, thermal expansivity, and
    compressibility of liquid water from 0. deg. to 150. deg..
    Correlations and tables for atmospheric pressure and saturation
    reviewed and expressed on 1968 temperature scale." Journal of Chemical
    and Engineering data 20.1 (1975): 97-105.

    URL: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://pubs.acs.org/doi/pdf/10.1021/je60064a005
    """
    temp = list2arr(temp)
    t = temp - 273.15
    if version == 0:
        eq = (
            (
                999.83952
                + 16.945176 * t
                - 7.9870401e-3 * t**2
                - 46.170461e-6 * t**3
                + 105.56302e-9 * t**4
                - 280.54253e-12 * t**5
            )
            / (1 + 16.87985e-3 * t)
            / 1000
        )
    else:
        raise ValueError("Not any version of equation yet")
    return eq


def relation_SiOH_ratio_according_to_CS(CS: np.ndarray | list):
    """
    APA:

    Casar, Z., Mohamed, A. K., Bowen, P., & Scrivener, K. (2023).
    Atomic-level and surface structure of calcium silicate hydrate nanofoils.
    The Journal of Physical Chemistry C, 127(37), 18652-18661.
    """
    return -0.4306 * CS + 0.8217


def pH(num_OH: int, num_H2O: int):
    C = num_OH / num_H2O
    return 14 + np.log10(C / 18 * 1000)
