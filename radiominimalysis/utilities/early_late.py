from radiominimalysis.framework.parameters import showerParameters as shp, stationParameters as stp
import numpy as np


def early_late_correction_factor(z_vxB, r_core):
    return 1 + z_vxB / r_core  # = R/R_0 (see paper)


def early_late_correction_factor2(shower_axis, station_position, dxmax, core=np.array([0, 0, 0])):
    shower_direction = -1 * shower_axis
    return 1 + np.dot(station_position - core, shower_direction) / dxmax


def calculate_early_late_correction_factor(station_position_vBvvB, zenith, xmax, at, obs_level, return_distance=False):
    # distance core at ground and "antenna plane" along shower axis
    z_vxB = station_position_vBvvB[:, 2]

    # Distance core - shower maximum
    r_core_xmax = at.get_distance_xmax_geometric(zenith, xmax, obs_level)

    # Correction factor for early-late effect
    c_early_late = early_late_correction_factor(z_vxB, r_core_xmax)

    if return_distance:
        return c_early_late, r_core_xmax
    else:
        return c_early_late


def get_early_late_correction_factor(revent, at):

    if revent.has_station_parameter(stp.early_late_factor):
        return revent.get_station_parameter(stp.early_late_factor)

    xmax = revent.get_parameter(shp.xmax)
    zenith = revent.get_parameter(shp.zenith)

    station_positions_transformed = revent.get_station_position_vB_vvB()
    obs_level = revent.get_parameter(shp.observation_level)

    # distance core at ground and "antenna plane" along shower axis
    zz = station_positions_transformed[:, 2]

    if not revent.has_parameter(shp.distance_to_shower_maximum_geometric):
        r_core_xmax = at.get_distance_xmax_geometric(zenith=zenith, xmax=xmax, observation_level=obs_level)
        revent.set_parameter(
            shp.distance_to_shower_maximum_geometric, r_core_xmax)
    else:
        r_core_xmax = revent.get_parameter(
            shp.distance_to_shower_maximum_geometric)

    # Correction factor for early-late effect
    c_early_late = early_late_correction_factor(zz, r_core_xmax)
    revent.set_station_parameter(stp.early_late_factor, c_early_late)

    return c_early_late
