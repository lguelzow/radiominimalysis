import copy

import numpy as np
from radiotools.atmosphere import models as atm
from scipy import constants

from radiominimalysis.framework.parameters import (
    eventParameters as evp,
    showerParameters as shp,
)
from radiotools import helper as rdhelp


def get_time_for_const_refractivity(zenith, xmax, obs_level, at):

    # height above sea level
    vertical_height_xmax = at.get_vertical_height(zenith, xmax, obs_level)

    # distance between core (at a certain obs level) and the shower maximum
    # function returns distance for hight above ground (!, not sea level), however takes obs_level into account
    r_core_xmax = atm.get_distance_for_height_above_ground(
        vertical_height_xmax - obs_level, zenith, observation_level=obs_level
    )

    refractivity_at_xmax = atm.get_n(
        vertical_height_xmax, n0=(1 + 3.12e-4), model=at.model
    )

    # # only a rough estimation
    # cherenkov_radius_est = 900
    time_const = r_core_xmax / (constants.c / refractivity_at_xmax) * 1e9  # in ns
    time_uniform = r_core_xmax / constants.c * 1e9  # in ns

    return time_const, time_uniform


def get_propagation_time_along_straight_path(
    p1=None,
    p2=None,
    zenith=np.deg2rad(75),
    depth=750,
    obs_level=1400,
    layer_height=1,
    return_trajectory=False,
    atm_model=27,
    n_asl=(1 + 3.12e-4),
    at=None,
    debug=False,
    n_param=None,
):
    """Calculate propagation time between two points along a straigth line.
        Assumes that p1 is higher in atmosphere than p2.
        p1 and p2 have to be given in a coordinatesystem with the origin at sea level
        (altitude defined by obs_level).
        If p1 is not given it is calculated using zenith and depth.
        If p2 is not given it is set to [0, 0, obs_level] ("MC core").

    Parameters
    ----------
    p1 : array (3,)
        starting position (default: None)

    p2 : array (3,)
        ending position (default: None)

    zenith : float
        zenith angle of the starting point in the atmosphere [in radians] (if p1 is None) (default: 75 deg)

    depth : float
        slant depth of the starting point in the atmosphere [in g/cm^2] (if p1 is None) (default: 750)

    observation_level : float
        defines ground plane [in meter] (defaut: 1400)

    layer_height : float
        height of each layer for which deflection is calculated [in meter] (default: 1)
        This height is interpreted in a flat coordinate system, however the height of the layer
        is calculated correctly

    return_trajectory : bool
        if True, function returns an array of all position where deflection is caluclated,
        e.g., trajectory of the ray (default: False)

    atm_model : int
        Atmospheric model, see https://github.com/nu-radio/radiotools/blob/master/radiotools/atmosphere/models.py
        (default: 27)

    atm_model : float
        Refractive index at sea level (default: 1 + 3.12e-4)

    Returns
    -------
    float, float, float, array
        propagation_time, path_distance, weighted_sum_n, (positions)
    """

    if at is None:
        at = atm.Atmosphere(atm_model)

    if p1 is None:
        # height above sea level. Yes obs_level matters for zenith angle
        vertical_height = at.get_vertical_height(zenith, depth, obs_level)

        # function input: height above ground (!), not sea level. Take obs_level into account
        r_p1 = atm.get_distance_for_height_above_ground(
            vertical_height - obs_level, zenith, observation_level=obs_level
        )

        # p1 in "sea level cs"
        p1 = rdhelp.spherical_to_cartesian(zenith=zenith, azimuth=0) * r_p1 + np.array(
            [0, 0, obs_level]
        )
    else:
        p1 = copy.copy(p1)

    if p2 is None:
        # in ground cs
        p2 = np.array([0, 0, obs_level])
    else:
        p2 = copy.copy(p2)

    line = p1 - p2
    distance_between_points = np.linalg.norm(line)

    zenith_between_points = rdhelp.get_local_zenith_angle(p1, p2)
    obs_level_local = rdhelp.get_local_altitude(p2)

    vertical_height = (
        atm.get_height_above_ground(
            np.linalg.norm(p1 - p2),
            zenith_between_points,
            observation_level=obs_level_local,
        )
        + obs_level_local
    )

    # variables
    propagation_time, n_steps, weighted_sum_n, path_distance = 0, 0, 0, 0
    positions = [copy.copy(p1)]

    direction = line / distance_between_points
    step = direction * (layer_height / direction[-1])
    keep_going = True
    while keep_going:

        # is height of p1 w.r.t. sea level
        height = (
            atm.get_height_above_ground(
                np.linalg.norm(p1 - p2),
                zenith_between_points,
                observation_level=obs_level_local,
            )
            + obs_level_local
        )

        if n_param is not None:
            n = n_param(height)
        else:
            n = atm.get_n(height, n0=n_asl, model=atm_model)

        # break condition, since we have a fixed zenith angle per layer no need to end just above the ground
        if (p1 - step)[-1] < p2[-1]:
            keep_going = False
            step = direction * np.linalg.norm(p1 - p2)

        # add layer to variables
        propagation_time += np.linalg.norm(step) / (constants.c / n)
        weighted_sum_n += (n - 1) * np.linalg.norm(step)
        n_steps += 1
        path_distance += np.linalg.norm(step)

        # update for next iteration
        # calculate height at point of defelction for curved atmosphere
        p1 -= step

        if return_trajectory:
            positions.append(copy.copy(p1))

    if debug:
        print(p2 - p1)
        print(
            "calculate num (ana) for %.3f deg, %.3f m distance, %.1f m min height, %.1f m max height"
            % (
                np.rad2deg(zenith_between_points),
                distance_between_points,
                p1[-1],
                vertical_height,
            ),
            obs_level,
        )

    if return_trajectory:
        return propagation_time, path_distance, weighted_sum_n, np.array(positions)
    else:
        return propagation_time, path_distance, weighted_sum_n


def get_core_shift_from_refractivity_deflection(
    zenith,
    depth=750,
    obs_level=1400,
    layer_height=1,
    return_trajectory=False,
    atm_model=27,
    at=None,
    n_asl=(1 + 3.12e-4),
    p1=None,
    return_last_position=False,
    additional_curved_correction=True,
):
    """predict core shift with simple model using Snell deflection in a layered atmosphere.
        Therefore a single 'ray' is propagated from a starting point in the atmosphere to a ground plane.
        The starting point is defined with zenith, depth and obs_level or optional given as vector("p1").
        The vector has to provide the coordinates of the starting point in a cartesian coordinate
        system with
        the origin at [0, 0, obs_lvl].

    Parameters
    ----------
    zenith : float
        zenith angle of the starting point in the atmosphere [in radians]

    depth : float
        slant depth of the starting point in the atmosphere [in g/cm^2] (default: 750)

    observation_level : float
        defines ground plane [in meter] (defaut: 1400)

    layer_height : float
        height of each layer for which deflection is calculated [in meter] (default: 1)
        This height is interpreted in a flat coordinate system, however the height of the
        layer is calculated correctly

    return_trajectory : bool
        if True, function returns an array of all position where deflection is caluclated,
        e.g., trajectory of the ray (default: False)

    atm_model : int
        Atmospheric model, see https://github.com/nu-radio/radiotools/blob/master/radiotools/atmosphere/models.py
        (default: 27)

    n_asl : float
        Refractive index at sea level (default: 1 + 3.12e-4)

    p1 : array (3,)
        optional starting position (default: None)

    Returns
    -------
    float, float, float, float, float, float, array
        core_shift, core_shift_plane, total_deflection_angle, r_core_xmax,
        propagation_time, path_distance, weighted_sum_n, (positions)
    """
    if at is None:
        # initialize atmosphere
        at = atm.Atmosphere(atm_model)

    # if "p1" is not None, start simulating from "p1" with the direction "zenith"
    if p1 is not None:
        r_core = np.linalg.norm(p1)

        # height above sea level (not ground!)
        vertical_height = (
            atm.get_height_above_ground(
                d=r_core, zenith=np.arcsin(p1[0] / r_core), observation_level=obs_level
            )
            + obs_level
        )
        position_cs = copy.copy(p1)
    else:
        # distance between core (at a certain obs level) and depth
        # height above sea level (not ground!)
        vertical_height = at.get_vertical_height(
            zenith, depth, observation_level=obs_level
        )

        # function input:height above ground (!), not sea level.
        # Takes obs_level into account
        r_core = atm.get_distance_for_height_above_ground(
            vertical_height - obs_level, zenith, observation_level=obs_level
        )

        # initialize variables for starting point in atmsophere (e.g., shower maximum)
        position_cs = (
            rdhelp.spherical_to_cartesian(zenith=zenith, azimuth=0) * r_core
        )  # vector to point in core CS

    sinalpha1 = np.sin(zenith)
    n1 = atm.get_n(vertical_height, n0=n_asl, model=atm_model)
    height = vertical_height

    # variables
    propagation_time, n_steps, weighted_sum_n, path_distance = 0, 0, 0, 0
    positions = [copy.copy(position_cs)]

    keep_going = True
    while keep_going:
        # get n for next layer
        n2 = atm.get_n(height, n0=n_asl, model=atm_model)

        if additional_curved_correction:
            # refraction at layer in curved atmosphere. However position_cs is position in a flat
            # coordinate system. corrections account that for zenith (in flat geometry) the layers
            # are not horizontal
            angle_correction = np.arcsin(position_cs[0] / (height + atm.r_e))
            # print(np.rad2deg(zenith_tmp), position_cs[0], height, np.rad2deg(angle_correction))
            zenith_at_curved_layers = np.arcsin(sinalpha1) - angle_correction
            sinalpha2 = np.sin(zenith_at_curved_layers) * n1 / n2  # change in direction
            # here the assumption is made that the correction does not change in layer
            zenith_in_layer = np.arcsin(sinalpha2) + angle_correction

        else:
            # calculate deflection and path through layer
            sinalpha2 = sinalpha1 * n1 / n2
            zenith_in_layer = np.arcsin(sinalpha2)

        step = np.array([layer_height * np.tan(zenith_in_layer), 0, layer_height])

        # break condition, last step is down sized so that it finishes just above the ground
        if (position_cs - step)[-1] < 0:
            keep_going = False
            layer_height = position_cs[-1]  # - 1e-10
            step = np.array([layer_height * np.tan(zenith_in_layer), 0, layer_height])

        # add layer to variables
        propagation_time += np.linalg.norm(step) / (constants.c / n2)
        weighted_sum_n += (n2 - 1) * np.linalg.norm(step)
        n_steps += 1
        path_distance += np.linalg.norm(step)

        # update for next iteration
        # calculate height at point of defelction for curved atmosphere
        position_cs -= step
        distance_cs = np.sqrt(position_cs[0] ** 2 + position_cs[2] ** 2)
        zenith_cs = np.arctan2(position_cs[0], position_cs[2])
        height = (
            atm.get_height_above_ground(
                distance_cs, zenith_cs, observation_level=obs_level
            )
            + obs_level
        )
        sinalpha1 = np.sin(zenith_in_layer)
        n1 = n2

        if zenith_in_layer < 0:
            break

        if return_trajectory:
            positions.append(copy.copy(position_cs))

    # get core shift
    core_shift = position_cs[0]
    # core_shift = np.linalg.norm(position_cs)

    total_deflection_angle = np.rad2deg(zenith - zenith_in_layer)
    core_shift_plane = np.sin(np.pi / 2 - zenith) * core_shift

    if return_last_position:
        # only return last position, used for ....
        return position_cs[0]

    if return_trajectory:
        return (
            core_shift,
            core_shift_plane,
            total_deflection_angle,
            r_core,
            propagation_time,
            path_distance,
            weighted_sum_n,
            np.array(positions),
        )
    else:
        return (
            core_shift,
            core_shift_plane,
            total_deflection_angle,
            r_core,
            propagation_time,
            path_distance,
            weighted_sum_n,
        )


def get_predicted_core_displacement(revent, args=None):
    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()

        shower = revent.get_shower(key=shower_type)

    if args.realistic_input:
        zenith = shower.get_parameter(shp.zenith_recon)
        azimuth = shower.get_parameter(shp.azimuth_recon)
    else:
        zenith = shower.get_parameter(shp.zenith)
        azimuth = shower.get_parameter(shp.azimuth)

    core_shift_ground = get_core_shift_from_refractivity_deflection(
        zenith=zenith,
        depth= 750, # shower.get_parameter(shp.xmax),
        obs_level=shower.get_parameter(shp.observation_level),
        atm_model=shower.get_parameter(shp.atmosphere_model),
        n_asl=revent.get_parameter(evp.refractive_index_at_sea_level),
    )[0]

    
    core_ground = np.array(
        [np.cos(azimuth) * core_shift_ground, np.sin(azimuth) * core_shift_ground, 0]
    )

    # save predicted core displacement in shower revent instance
    shower.set_parameter(shp.prediceted_core_shift, core_ground)

    # return displacement in shower plane
    # return revent.get_coordinate_transformation().transform_to_vxB_vxvxB(core_ground)

    # version in ground plane
    return core_ground


def get_predicted_core(revent):
    shower = revent.get_shower()
    core_shift_ground = get_core_shift_from_refractivity_deflection(
        zenith=shower.get_parameter(shp.zenith),
        depth=shower.get_parameter(shp.xmax),
        obs_level=shower.get_parameter(shp.observation_level),
        atm_model=shower.get_parameter(shp.atmosphere_model),
        n_asl=revent.get_parameter(evp.refractive_index_at_sea_level),
    )[0]

    core_ground = (
        rdhelp.spherical_to_cartesian(
            zenith=np.pi / 2, azimuth=shower.get_parameter(shp.azimuth)
        )
        * core_shift_ground
    )
    return core_ground + np.array([0, 0, shower.get_parameter(shp.observation_level)])
