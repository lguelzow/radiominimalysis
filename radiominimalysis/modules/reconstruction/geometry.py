import warnings
from radiominimalysis.framework.parameters import stationParameters as stp, showerParameters as shp, \
    eventParameters as evp
from radiominimalysis.utilities.early_late import early_late_correction_factor, get_early_late_correction_factor
from radiominimalysis.utilities import refractive_displacement, cherenkov_radius

import radiotools.atmosphere.models as atm
from radiotools import helper as rdhelp
from radiotools import coordinatesystems as cs

import numpy as np
import ray
import sys


def reconstruct_geometry(events, para, at=None):
    for revent in events:
        reconstruct_geometry_revent(revent, para, at=at)
        

@ray.remote
def reconstruct_geometry_ray(revent, para, at=None):
    reconstruct_geometry_revent(revent, para, at=at)
    return revent


def reconstruct_geometry_revent(revent, para, at=None):

    # check whether it's a measured shower and change call of get_shower accordingly
    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()

        shower = revent.get_shower(key=shower_type)

    if at is None:
        if para.atmModel is None and para.gdasFile is None:
            at = revent.get_atmosphere()
        else:
            if para.gdasFile is not None:
                at = atm.Atmosphere(gdas_file=para.gdasFile)
            
            elif para.atmModel is not None:
                at = atm.Atmosphere(para.atmModel)
                print("Atmosphere successfully generated from input model!")
            
            if shower.has_parameter(shp.atmosphere_model):
                atm_model = shower.get_parameter(shp.atmosphere_model)
                if atm_model != at.model:
                    warnings.warn(f"Used atmospheric model ({at.model}) is not that of the shower ({atm_model})")
            else:
                warnings.warn("Use hardcoded atmospheric model")
    else:
        at = atm.Atmosphere(at)
        print("Atmosphere successfully generated from hard coded input!")

    obs_level = shower.get_parameter(shp.observation_level)
    
    print("Refractive Index at obs level: ", at.get_n(obs_level))

    if para.realistic_input:
        print("Use reconstructed zenith and avg Xmax=750 g/cm²")
        zenith = shower.get_parameter(shp.zenith_recon)
        azimuth = shower.get_parameter(shp.azimuth_recon)
        mag_field_vec = revent.get_parameter(evp.magnetic_field_vector)
        xmax = 750
        core = shower.get_parameter(shp.core_estimate)

        if shower.has_parameter(shp.zenith):
            # save MC dmax and density
            vertical_height_xmax = at.get_vertical_height(shower.get_parameter(shp.zenith), shower.get_parameter(shp.xmax), observation_level=obs_level)
            
            # distance between core (at a certain obs level) and the shower maximum
            # function returns distance for hight above ground (!, not sea level), however takes obs_level into account
            distance_to_xmax_mc = atm.get_distance_for_height_above_ground(vertical_height_xmax - obs_level, shower.get_parameter(shp.zenith), observation_level=obs_level)
            # also get density
            rho_xmax = atm.get_density(vertical_height_xmax, model=at.model) * 1e-3  # conversion from g/m3 to kg/m3
            # save parameters
            shower.set_parameter(shp.MC_distance_to_shower_maximum, distance_to_xmax_mc)
            shower.set_parameter(shp.MC_density_at_shower_maximum, rho_xmax)
            
            print("MC dmax and rho calculated!")

    else:
        zenith = shower.get_parameter(shp.zenith)
        azimuth = shower.get_parameter(shp.azimuth)
        mag_field_vec = revent.get_parameter(evp.magnetic_field_vector)
        xmax = shower.get_parameter(shp.xmax)
        core = shower.get_parameter(shp.core)
    

    
    # height above sea level
    try:
        vertical_height_xmax = at.get_vertical_height(zenith, xmax, observation_level=obs_level)
    except RuntimeError as e:
        print(e)
        return 0

    rho_xmax = atm.get_density(vertical_height_xmax, model=at.model) * 1e-3  # conversion from g/m3 to kg/m3
    shower.set_parameter(shp.density_at_shower_maximum, rho_xmax)
    
    # distance between core (at a certain obs level) and the shower maximum
    # function returns distance for hight above ground (!, not sea level), however takes obs_level into account
    r_core_xmax = atm.get_distance_for_height_above_ground(vertical_height_xmax - obs_level,
                                                        zenith, observation_level=obs_level)

    # print(r_core_xmax, xmax, np.rad2deg(zenith))

    n0 = revent.get_parameter(evp.refractive_index_at_sea_level)
    r_che = cherenkov_radius.get_cherenkov_radius_model_from_height(
        zenith, vertical_height_xmax, obs_level, n0, at.model)
    shower.set_parameter(shp.cherenkov_radius_model, r_che)

    shower.set_parameter(shp.distance_to_shower_maximum_geometric, r_core_xmax)
        

    # distance core and shower maximum in grammage
    distance_xmax = at.get_distance_xmax(zenith, xmax, observation_level=obs_level)  # in g/cm^2
    shower.set_parameter(shp.distance_to_shower_maximum_grammage, distance_xmax)

    if revent.has_station_parameter(stp.position):

        if para.realistic_input:
            # distance core at ground and "antenna plane" along shower axis
            # build in exception for if there's only 1 station
            trafo = cs.cstrafo(zenith, azimuth, magnetic_field_vector=mag_field_vec)
            z_vxB = trafo.transform_to_vxB_vxvxB(revent.get_station_parameter(stp.position), core=core).reshape(-1,3)[:, 2]


        else:
            # distance core at ground and "antenna plane" along shower axis
            # build in exception for if there's only 1 station
            z_vxB = revent.get_station_position_vB_vvB().reshape(-1,3)[:, 2]

        # Correction factor for early-late effect, c = R/R_0 (see paper)
        c_early_late = early_late_correction_factor(z_vxB, r_core_xmax)

        # early_late_correction_factor
        revent.set_station_parameter(stp.early_late_factor, c_early_late)

        print("Geometric parameters successfully reconstructed!")

    else:
        print("Geometry reconstructed, but no antennas found!")


def reconstruct_station_geometry(events, para):

    for revent in events:
        shower = revent.get_shower()

        at = revent.get_atmosphere()

        zenith = shower.get_parameter(shp.zenith)
        azimuth = shower.get_parameter(shp.azimuth)
        xmax = shower.get_parameter(shp.xmax)
        obs_level = shower.get_parameter(shp.observation_level)

        distance_to_xmax_core = shower.get_parameter(shp.distance_to_shower_maximum_grammage)
        distance_to_xmax_core_geo = shower.get_parameter(shp.distance_to_shower_maximum_geometric)

        shower_axis = revent.get_shower_axis()
        core = shower.get_parameter(shp.core)

        # set core to observation level
        core_xmax_vec = shower_axis * distance_to_xmax_core_geo + core

        station_positions = revent.get_station_parameter(stp.position)

        station_zeniths = -1 * np.ones(len(station_positions))
        distance_to_xmax_stations = -1 * np.ones(len(station_positions))
        distance_to_xmax_stations_geometric = -1 * np.ones(len(station_positions))

        # height of xmax over ground (not sea level)
        h_xmax_above_obs = atm.get_height_above_ground(distance_to_xmax_core_geo, zenith,
                                                       observation_level=obs_level)

        # loop over stations (clean for thinning)
        mask = [True] * len(station_positions)
        for idx, station_position in enumerate(station_positions):
            if not mask[idx]:
                continue

            # if not para.verbose:
            #     print(revent.get_run_number(), idx)

            # calculate vector of line of view from atenna to xmax
            station_xmax_vec = core_xmax_vec - station_position
            distance_to_xmax_station_geo = np.linalg.norm(station_xmax_vec)

            # Define new 'observation heigt' == hight of station
            r_e = 6.371 * 1e6 + station_position[-1]
            # h xmax above station (for stations not on a ground plane)
            h_xmax_above_obs_station = h_xmax_above_obs + (obs_level - station_position[-1])

            # calculate zenith angle at station with curved atmosphere
            zenith_station = np.arccos((h_xmax_above_obs_station ** 2 + 2 * r_e * h_xmax_above_obs_station - distance_to_xmax_station_geo ** 2) / \
                                       (2 * r_e * distance_to_xmax_station_geo))

            distance_to_xmax_stations_geometric[idx] = distance_to_xmax_station_geo
            station_zeniths[idx] = zenith_station

            if 0:  # time consuming
                # atmosphere in g/cm2 between station and xmax
                distance_to_xmax_station = at.get_atmosphere(zenith_station, h_low=station_position[-1],
                                                             h_up=h_xmax_above_obs+obs_level)
                distance_to_xmax_stations[idx] = distance_to_xmax_station

        revent.set_station_parameter(stp.zenith, np.array(station_zeniths))
        revent.set_station_parameter(stp.distance_to_shower_maximum_grammage, np.array(distance_to_xmax_core))
        revent.set_station_parameter(stp.distance_to_shower_maximum_geometric, np.array(distance_to_xmax_stations_geometric))


def find_overestimated_signals_frequency_slope(events, para):
    """
    This function determines which pulses are affected by thinning. It uses the linear slope of the spectra in the vxB polarisation.
    Due to thinning high frequency noise can artifically enhance the simulated signals.
    Once this noise reaches the bandwidth 30 - 80 MHz we have to reject those signals. This depends on the lateral distance.
    It turns out that the distance is quite constant with energy.
    """

    for revent in events:

        lateral_distance = revent.get_station_axis_distance()

        if not revent.has_station_parameter(stp.early_late_factor):
            at = atm.Atmosphere(model=revent.get_shower().get_parameter(shp.atmosphere_model))
            c_early_late = get_early_late_correction_factor(revent, at)
        else:
            c_early_late = revent.get_station_parameter(stp.early_late_factor)

        lateral_distance = lateral_distance / c_early_late

        # get fit parameter of 1 deg polyfit
        freq_slope = revent.get_station_parameter(stp.frequency_slope)

        # select vxB polarisation and 1 deg param
        # vxvxB is affected earlier than vxB
        freq_slope = freq_slope[:, 0, 0]

        r_min_arg = np.argmin(freq_slope)
        r_min = lateral_distance[r_min_arg]
        revent.set_station_parameter(stp.thinning_distance, r_min)

        normalized_distance = lateral_distance / r_min

        thin_mask = normalized_distance < para.thinning_cut
        revent.set_station_parameter(stp.cleaned_from_thinning, thin_mask)


# same function but without loop
def find_overestimated_signals_frequency_slope_revent(revent, para):
    """
    This function determines which pulses are affected by thinning. It uses the linear slope of the spectra in the vxB polarisation.
    Due to thinning high frequency noise can artifically enhance the simulated signals.
    Once this noise reaches the bandwidth 30 - 80 MHz we have to reject those signals. This depends on the lateral distance.
    It turns out that the distance is quite constant with energy.
    """


    lateral_distance = revent.get_station_axis_distance()

    if not revent.has_station_parameter(stp.early_late_factor):
        at = atm.Atmosphere(model=revent.get_shower().get_parameter(shp.atmosphere_model))
        c_early_late = get_early_late_correction_factor(revent, at)
    else:
        c_early_late = revent.get_station_parameter(stp.early_late_factor)

    lateral_distance = lateral_distance / c_early_late

    # get fit parameter of 1 deg polyfit
    freq_slope = revent.get_station_parameter(stp.frequency_slope)

    # select vxB polarisation and 1 deg param
    # vxvxB is effacted earier than vxB
    freq_slope = freq_slope[:, 0, 0]

    r_min_arg = np.argmin(freq_slope)
    r_min = lateral_distance[r_min_arg]
    revent.set_station_parameter(stp.thinning_distance, r_min)

    normalized_distance = lateral_distance / r_min

    thin_mask = normalized_distance < 0.85
    revent.set_station_parameter(stp.cleaned_from_thinning, thin_mask)
    print(f"{np.sum(thin_mask)}/{len(thin_mask)} antennas cut due to thinning effects!")


def correct_core_with_model_revent(revent, para):

    at = revent.get_atmosphere()
    shower = revent.get_shower()
    xmax = shower.get_parameter(shp.xmax)
    obs_level = shower.get_parameter(shp.observation_level)
    zenith = shower.get_parameter(shp.zenith)

    core_shift_radius, _, _, _, _, _, _ = \
        refractive_displacement.get_core_shift_from_refractivity_deflection(
            zenith, depth=xmax, obs_level=obs_level, at=at, layer_height=10)

    core_shift = rdhelp.spherical_to_cartesian(
        zenith=np.deg2rad(90), azimuth=shower.get_parameter(shp.azimuth)) * core_shift_radius
    shower.set_parameter(shp.core, shower.get_parameter(shp.core) + core_shift)
    shower.set_parameter(shp.prediceted_core_shift, core_shift)


@ray.remote
def correct_core_with_model_ray(revent, para):
    correct_core_with_model_revent(revent, para)
    return revent


def correct_core_with_model(events, para):
    for revent in events:
       correct_core_with_model_revent(revent, para)
