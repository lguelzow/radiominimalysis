from radiominimalysis.modules.reconstruction import ldf_fitting
from radiominimalysis.utilities import geomagnetic_emission
import numpy as np

from scipy import integrate


def has_ldf_wrapper(xdata, params):

    # unpack parameters:
    params_dict = params.valuesdict()

    # unpack independent vars
    # antenna_positions_core_cs = in ground plane but core cs (z = 0)
    antenna_positions_core_cs, alpha, zenith, cs = xdata

    # Performs core fit, performs early late correction (not for z_vB)
    x_vB, y_vB, z_vB, c_early_late, distance_xmax_geometric, params_dict = \
        ldf_fitting.vary_core_and_perform_early_late_correction(
            params_dict, antenna_positions_core_cs, cs, zenith)

    # # calculate distance, phi with respect to core
    # r = np.sqrt(x_vB ** 2 + y_vB ** 2) / c_early_late

    return ldf_fitting.has_ldf_param(x_vB, y_vB, z_vB, c_early_late, distance_xmax_geometric, alpha, zenith, params_dict)


def ldf_gaus_sigmoid_shape_old(r, arel, r0, sig, slope, p, r02, a1=1):
    # a1 is just there to plot the sigmoid term independently (therefore set a1 = 0)
    p_eff = 2 * np.ones_like(r, dtype=np.float64)
    # p_eff[r < r0] = 2 * (r0 / r[r < r0]) ** (p / 1000)
    p_eff[r > r0] = 2 * (r0 / r[r > r0]) ** (p / 1000)
    # print(p_eff)
    return a1 * np.exp(-((r - r0) / sig) ** p_eff) + arel / (1 + np.exp(slope * (r / r0 - r02)))
    # return a1 + arel / (1 + np.exp(slope * (r / r0 - r02)))

def ldf_gaus_sigmoid_shape(r, arel, r0, sig, slope, p, p_slope, r02, a1=1):
    # a1 is just there to plot the sigmoid term independently (therefore set a1 = 0)
    p_eff = 2 * np.ones_like(r, dtype=np.float64)
    # mirror of Felix' description of the slope
    # p_eff[r < r0] = 2 * (r[r < r0] / r0) ** (p_slope / 1000)
    # fixed parameter
    p_eff[r < r0] = p_slope
    p_eff[r >= r0] = 2 * (r0 / r[r >= r0]) ** (p / 1000)

    gauss = np.exp(-((r - r0) / sig) ** p_eff)
    gauss[r < r0] = np.exp(-(np.abs(r[r < r0] - r0) / sig) ** p_eff[r < r0])
    # print(p_eff)
    return a1 * gauss + arel / (1 + np.exp(slope * (r / r0 - r02)))
    # return a1 + arel / (1 + np.exp(slope * (r / r0 - r02)))  


def f_E_geo_gaus_sigmoid(r, E_geo, arel, r0, sig, slope, p, r02):
    r_int = np.arange(r0 * 5)

    shape = ldf_gaus_sigmoid_shape_old(r, arel, r0, sig, slope, p, r02)
    norm = geomagnetic_emission.calculate_geomagnetic_energy(
        r_int, ldf_gaus_sigmoid_shape, {'arel': arel, 'r0': r0, 'sig': sig, 'slope': slope, 'p': p, "r02": r02})
    return E_geo * shape / norm

def f_E_geo_gaus_sigmoid_p_slope(r, E_geo, arel, r0, sig, slope, p, p_slope, r02):
    r_int = np.arange(r0 * 5)

    shape = ldf_gaus_sigmoid_shape(r, arel, r0, sig, slope, p, p_slope, r02)
    norm = geomagnetic_emission.calculate_geomagnetic_energy(
        r_int, ldf_gaus_sigmoid_shape, {'arel': arel, 'r0': r0, 'sig': sig, 'slope': slope, 'p': p, 'p_slope': p_slope, "r02": r02})
    
    if np.any(np.isnan(shape)) or np.any(np.isnan(norm)):
        print("LDF shape", np.any(np.isnan(shape)))
        # print(r, E_geo, arel, r0, sig, slope, p, p_slope, r02)
    return E_geo * shape / norm
