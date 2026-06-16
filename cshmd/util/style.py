# myplot/style.py
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

COLORS_10 = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
    "#66CCEE",
]

COLORS_20 = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

linestyles = ["-", "--", ":", "-."]
markers = ["o", "s", "^", "D", "x"]


def get_color_cycle_10():
    return cycle(COLORS_10)


def get_color_cycle_20():
    return cycle(COLORS_20)


def plot_style_paper_0(
    update_color=False,
    low_quality=False,
    xcol=1,
    ycol=1,
    large_font=True,
    large_legend=False,
):
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.dpi": 300,
            "axes.linewidth": 1.2,
            # "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "legend.frameon": False,
            # "figure.autolayout": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    if update_color:
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS_10)
    if low_quality:
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
    # plt.rcParams.update({
    #     "axes.labelsize": "large",
    #     "axes.titlesize": "large",
    #     "xtick.labelsize": "large",
    #     "ytick.labelsize": "large",
    #     # "legend.fontsize": "large"
    # })
    if large_font:
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            }
        )
    if large_legend:
        plt.rcParams.update(
            {
                "legend.fontsize": 16,
            }
        )
    if xcol == 1 and ycol == 1:
        plt.rcParams.update(
            {
                "figure.figsize": (6, 4),
            }
        )
    elif xcol == 2 and ycol == 1:
        plt.rcParams.update(
            {
                "figure.figsize": (8, 4),
            }
        )
    elif xcol == 1 and ycol == 2:
        plt.rcParams.update(
            {
                "figure.figsize": (6, 6),
            }
        )


def pd_option_0():
    pd.options.display.float_format = "{:.3f}".format
