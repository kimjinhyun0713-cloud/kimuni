from .common import list2arr
import numpy as np

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
        eq = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2 - 46.170461e-6 * t**3
              + 105.56302e-9 * t**4 - 280.54253e-12 * t**5) / (1 + 16.87985e-3 * t) / 1000
    else:
        raise ValueError("Not any version of equation yet")
    return eq
       
