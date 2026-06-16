import numpy as np
import pandas as pd
from pathlib import Path
import sys
import subprocess as sp
import time
import logging


class AttrLoggerMixin:
    def __getattribute__(self, name):
        cls = object.__getattribute__(self, "__class__")
        logger.debug(f"Searching attribution: {name}")
        logger.debug(f"  checking instance of {cls.__name__}")
        if name in object.__getattribute__(self, "__dict__"):
            logger.debug(f"  -> FOUND in {cls.__name__} (instance)")
        for base in cls.__mro__:
            logger.debug(f"  checking {base.__name__}")
            if name in base.__dict__:
                logger.debug(f"  -> FOUND in {base.__name__}")
                break
        return super().__getattribute__(name)


class LeftFormatter(logging.Formatter):
    def format(self, record):
        header_fmt = "%(asctime)s [%(levelname)s] %(name)s (%(funcName)s)"
        header = logging.Formatter(header_fmt, datefmt="%m/%d %H:%M").format(record)
        header = f"{header:<80}"
        message = record.getMessage()
        return f"{header} : {message}"


def get_logger(name="cshmd"):
    name = name.replace(".py", "").split("/")[-1]

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = LeftFormatter()

    #
    # stdout (INFO以上)
    #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    #
    # 全ログ
    #
    log_handler = logging.FileHandler("cshmd.log", encoding="utf-8")
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(formatter)

    #
    # WARNING以上のみ
    # warningが出るまでファイルを作らない
    #
    err_handler = logging.FileHandler("cshmd.err", encoding="utf-8", delay=True)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(log_handler)
    logger.addHandler(err_handler)

    logger.propagate = False

    logger.debug(f"$$$$$$$$$$$$$$  Logger '{name:.<20}' initialized  $$$$$$$$$$$$$$")

    return logger


logger = get_logger(__name__)


def df2xyz(df: pd.DataFrame) -> np.ndarray:
    return df.loc[:, ["type_symbol", "fract_x", "fract_y", "fract_z"]].to_numpy()


def type2list(data):
    """
    convert int, float or str to list
    if only in case the data can converted to numerical value

    args:
    ;; data

    return
    ;; data
    """
    if isinstance(data, list):
        return data.copy()
    elif isinstance(data, np.ndarray):
        return list(data)
    elif isinstance(data, (int, float)):
        return [data]
    elif isinstance(data, str):
        try:
            return [float(data)]
        except ValueError:
            raise TypeError("Error: cannot convert to a numerical value")
    else:
        raise ValueError("Wrong Type")


def list2arr(data, dtype=float):
    """
    convert int, float, str or list to array
    if only in case the data can converted to  numerical value
    convert to arr if only
    if data's type is neither of that, then raise TypeError
    """
    if isinstance(data, np.ndarray):
        pass
    else:
        data = type2list(data)
        data = np.array(data, dtype=dtype)
    return data


def sum_traindata(path, return_df=False):
    """
    logger.debug name of traindata, nstep, natom recursively
    and if return_df, return DataFrame of that.

    stdout: name of traindata, nstep, natom

    args
    ;; path -> path

    return
    None if return_df df.DataFrame
    """
    total_nstep = 0
    generator = Path(path).rglob("coord.npy")
    if return_df:
        df_list = []
    for dat in generator:
        tname = str(dat).split("set.000")[0]
        logger.debug(f"{str(tname):<40s}", end="")
        loaded = np.load(dat)
        if loaded.ndim == 1:
            loaded.reshape(1, -1)
        try:
            nstep, natom = np.load(dat).shape
        except ValueError:
            logger.error("no data")
            continue
        if return_df:
            df = pd.Series([tname, nstep, natom])
            df_list.append(df)
        natom /= 3
        total_nstep += nstep

        logger.debug(f"Steps: {int(nstep):<10}", f"Atoms: {int(natom):<10}")
    logger.debug("Total steps: ", total_nstep)
    return pd.concat(df_list, ignore_index=True) if return_df else None


def find2pipe():
    """
    convert pipelines stdout to list

    input
    ;;stdout
    ;;e.g.) find . -name ".ext" |

    return
    list,  result of stdout
    """
    files = [Path(line.strip()) for line in sys.stdin if line.strip()]
    return files


def string_filter(ptn, lst):
    """
    return list of string which have speific pattern

    args
    ;;pattern which you want to filter, lst

    return
    ;; list
    """
    return list(filter(lambda s: ptn in s, lst))


def sp_run(command):
    return sp.run(command, capture_output=True, text=True)


def overwritePrint(string):
    print(f"\r{string}", end="", flush=True)


def checkTime(func):
    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(
            f"[RUNTIME] {func.__name__} in {end_time - start_time:.3f} seconds"
        )
        return result

    return time_wrapper
