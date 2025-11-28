import sys
import warnings

import lmfit
import numpy as np
from radiotools.atmosphere import models as atm

from radiominimalysis.framework.parameters import (
    eventParameters as evp,
    showerParameters as shp,
    stationParameters as stp,
)

from radiominimalysis.modules.reconstruction import signal_emissions as se
from radiominimalysis.utilities import charge_excess, cherenkov_radius, early_late, geomagnetic_emission


def vary_core_and_get_early_late_factor(
        params_dict, antenna_positions_core_cs, cs,
        observation_level, atmodel, n0, zenith=None, core=None, plot_flag=False):

    # pop removes core from dict and put it in variable (if not present = 0)
    core_x = params_dict.pop('core_x', 0)
    core_y = params_dict.pop('core_y', 0)

    if core_x == 0 and plot_flag:
        core_x = core[0]
        core_y = core[1]

    # antenna_positions_core_cs = in ground plane but core cs (z = 0)
    if np.all(np.around(antenna_positions_core_cs.reshape(-1,3)[:, -1])):
        warnings.warn("It seems that antenna positions are not in core cs, e.g. z!=0."
                      " This is okay if simulation is in shower plane or on a spherical obs plane")

    # core fit (fit in shower plan) & unpack position
    # core_fit_in_shower_plane = np.array([core_x, core_y, 0])
    # core_fitted_in_ground_plane = cs.transform_from_vxB_vxvxB_2D(
    #     core_fit_in_shower_plane)
    
    # core fit (fit in ground plane) & unpack position
    core_fitted_in_ground_plane = np.array([core_x, core_y, 0])
    # print("core fit", core_fitted_in_ground_plane)

    core_fit_in_shower_plane = cs.transform_to_vxB_vxvxB(
        core_fitted_in_ground_plane)
    
    # print("ground plane core: ", core_fitted_in_ground_plane)
    x_vB, y_vB, z_vB = np.squeeze(np.split(
        cs.transform_to_vxB_vxvxB(antenna_positions_core_cs, core=core_fitted_in_ground_plane).T, 3))

    # pop removes distance_xmax_geometric from dict and put it in variable (raises error if not present)
    distance_xmax_geometric = params_dict['distance_xmax_geometric']

    # If gauss_sigmoid is used (arel in params_dict) and r0 is not fitted
    if zenith is not None and "arel" in params_dict and "r0" not in params_dict:
        r0 = cherenkov_radius.get_cherenkov_radius_model_from_distance(
            zenith, distance_xmax_geometric, observation_level, n0, atmodel)
        # params_dict["r0"] = r0 # introduce arbitrary factor to match predicted r0 to fit

    # calculate early-late correction factor
    c_early_late = early_late.early_late_correction_factor(
        z_vB, distance_xmax_geometric)
    
    if np.any(c_early_late < 0):
        print("Negative correction factors detected:", np.array(c_early_late), np.array(z_vB), distance_xmax_geometric, np.rad2deg(zenith))

    if np.any(distance_xmax_geometric < 0):
        print(np.any(distance_xmax_geometric < 0))

    # remove if in (only needed for parameter "expr")
    params_dict.pop("average_dxmax", None)
    params_dict.pop("average_density", None)
    params_dict.pop("density_at_xmax", None)
    params_dict.pop("azimuth_angle", None)
    params_dict.pop("r0_start_value", None)

    return x_vB, y_vB, z_vB, c_early_late, distance_xmax_geometric, params_dict


def update_param_from_density(params, zenith, observation_level, model):

    if "density_at_xmax" in params and params["distance_xmax_geometric"].vary:
        if not isinstance(params, lmfit.Parameters) or zenith is None:
            sys.exit("Can not compute parameter from density. Abort ...")
        distance_xmax_geometric = params["distance_xmax_geometric"]

        h_max = atm.get_height_above_ground(
            distance_xmax_geometric, zenith, observation_level) + observation_level
        rho_max = atm.get_density(h_max, model=model) * 1e-3

        params["density_at_xmax"].set(value=rho_max)
        found_param = False
        for name, param in params.items():
            if param.expr is not None and "density_at_xmax" in param.expr:
                params.eval(param.expr)
                found_param = True

        if not found_param:
            sys.exit(
                "Found not Parameter which parameterized to density even though it is given...")

    return params


def get_charge_excess_fraction(r_el, dxmax, rho_max, zenith=None, param=None, new_ce_param=True, Auger_param=False):
    
    
    if new_ce_param:
        if param is None:
            # Felix' 30-80 parametrisation
            # param = [-1.17523609e-06, 3.48154734e-01, 1.6068519502678418,
            #             1.66965694e+01, 3.31954904e+00, -5.73577715e-03]
            
            if Auger_param:
                # parameters from Lukas new fit for 50-200 MHz
                # use these for 50-200 MHz Argentina
                param = [-1.37266723e-06, 3.02018018e-01, 1.46508803e+00, 1.31382072e+01, 2.98380964e+00, 1.78471809e-01]
            
            else:
                # parameters from Lukas new fit for 50-200 MHz for China
                # use these for 50-200 MHz for GP300 site
                param = [-9.03992069e-07, 2.28710354e-01, 1.62957071e+00, 1.77729341e+00, 1.42776016e+00, 1.66010236e-01]

        # print("r", r_el[0:10])
        # print("dmax", dxmax)
        # print("Density", rho_max)
        # print("parametrisation", param)

        a = charge_excess.charge_excess_fraction_icrc21(
            (r_el, dxmax, rho_max), *param)

    else:
        if param is None:
            param = [0.37313183, 1.31124484, -0.1889835, 6.71403002]
        
        # rough estimation, zeniths > 80 deg -> a = 0
        if zenith < np.deg2rad(80.001):
            a = charge_excess.charge_excess_fraction_icrc19(
                (r_el, dxmax, rho_max), *param)
        else:
            a = 0
       
    a = np.where(a < 0, 0, a)
    return a


def objective_ldf_geo_pos(
        params, xdata, f_geo_ldf,
        observation_level, atmodel, n0,
        zenith=None,
        rel_weight=0.02, add_abs_weight=1e-4, do_sum=False, alpha=None):

    params = update_param_from_density(
        params, zenith, observation_level, atmodel)

    # unpack parameters:
    if isinstance(params, lmfit.Parameters):
        params_dict = params.valuesdict()
    else:
        params_dict = params

    is_core_fit = np.abs(params_dict['core_x']) > 0.1 or np.abs(
        params_dict['core_y']) > 0.1

    # unpack independent vars
    # antenna_positions_core_cs = in ground plane but core cs (z = 0)
    antenna_positions_core_cs, energy_fluence_vector, cs = xdata

    # Performs core fit, get early late correction factor, unpacks params_dict
    x_vB, y_vB, z_vB, c_early_late, _, params_dict = \
        vary_core_and_get_early_late_factor(
            params_dict, antenna_positions_core_cs, cs,
            observation_level=observation_level, atmodel=atmodel, n0=n0, zenith=zenith)

    # calculate distance
    r_corrected = np.squeeze(np.sqrt(x_vB ** 2 + y_vB ** 2) / c_early_late)

    # calculate model prediction
    f_geomagnetic_model = f_geo_ldf(r_corrected, **params_dict)

    # calculates f_geo from position, only includes station with np.cos(phi_const) > 0.9 or np.cos(phi_const) < -0.9
    # if use_vxB_axis is True (default is False). default value is -1
    station_position_core = np.squeeze(np.array([x_vB, y_vB, z_vB]).T)
    f_geo_pos = se.seperate_radio_emission_from_position(station_position_core, energy_fluence_vector, c_early_late,
                                                         recover_vxB=False, set_vxB_to_value=-1, get_only_f_geo=True,
                                                         fitted_core=is_core_fit)
    
    mask = np.all([f_geo_pos > 0, ~np.isnan(f_geo_pos)], axis=0)

    f_geo_pos = f_geo_pos * c_early_late ** 2

    # get weights
    f_geo_weight = f_geo_pos * rel_weight + add_abs_weight * np.amax(f_geo_pos)
    f_geo_weight[~mask] = 1e9

    # calculate objective function (is used in minimization, default: leastsq)
    chi = (f_geomagnetic_model - f_geo_pos) / f_geo_weight

    if do_sum:
        return np.sum(chi ** 2)
    else:
        return chi


def ldf_has_param(
        x_vB, y_vB, z_vB, c_early_late,
        distance_xmax_geometric, alpha, zenith, geo_params,
        observation_level, atmodel, f_geo_ldf, 
        ce_fraction_param=None, new_ce_param=True, Auger_param=False):


    # calculate distance, phi with respect to core
    r = np.sqrt(x_vB ** 2 + y_vB ** 2)
    phi = np.arctan2(y_vB, x_vB)

    # early late correction for distance:
    # used in prediction of the geomagnetic emission and charge excess correction
    # this distance describes the position the station has in the shower plane, e.g., the off axis angle
    r_corrected = r / c_early_late

    if np.any(r < 0):
        print("radius", np.any(r < 0))

    if np.any(r_corrected < 0):
        print("radius_corr", np.any(r_corrected < 0))

    # if geo_params["distance_xmax_geometric"] < 5500:
    #     print(geo_params["distance_xmax_geometric"])
    #     print(np.rad2deg(zenith))

    # removes dmax parameter from LDF parameter dictionary if still included
    if geo_params["distance_xmax_geometric"]:
        geo_params.pop("distance_xmax_geometric", None)

    # rotationally symmetric ldf, e.g., signal expectation for the electromagnetic emission in the shower plane
    f_geomagnetic_model = f_geo_ldf(r_corrected, **geo_params)
    if np.any(np.isnan(f_geomagnetic_model)):
        print("f_geomagnetic_model", np.any(np.isnan(f_geomagnetic_model)))
    # convert distance in density
    rho_max = atm.get_density_for_distance(distance_xmax_geometric,
                                           zenith,
                                           observation_level=observation_level,
                                           model=atmodel) * 1e-3  # conversion g/m3 -> kg/m3

    # charge-excess fraction
    a = get_charge_excess_fraction(
        r_corrected, distance_xmax_geometric, rho_max, zenith, 
        ce_fraction_param, new_ce_param, Auger_param=Auger_param)
    
    if np.any(np.isnan(a)):
        print("a", np.any(np.isnan(a)))

    # prediction for asymmetric radio footprint in shower plane
    f_vxB_fit = f_geomagnetic_model * \
        (1 + np.cos(phi) / np.sin(alpha) * np.sqrt(a)) ** 2.

    # inverse early late correction -> incorporate uncertainty from early late effect
    f_vxB_fit = f_vxB_fit / c_early_late ** 2

    if np.any(np.isnan(f_vxB_fit)):
        print("f_vxB_fit has NaNs", np.any(np.isnan(f_vxB_fit)))
        print(distance_xmax_geometric, np.rad2deg(zenith))
    
    # retruns model prediction for the measured energy fluence in the vxB polarisation
    return f_vxB_fit


def ldf_geo_param(
        x_vB, y_vB, c_early_late, f_vxB,
        distance_xmax_geometric, alpha, zenith,
        observation_level, atmodel, 
        ce_fraction_param=None, new_ce_param=True, Auger_param=False):

    # calculate distance, phi with respect to core
    r = np.sqrt(x_vB ** 2 + y_vB ** 2)
    phi = np.arctan2(y_vB, x_vB)

    # early late correction for distance:
    # used in prediction of the geomagnetic emission and charge excess correction
    # this distance describes the position the station has in the shower plane, e.g., the off axis angle
    r_corrected = r / c_early_late

    # convert distance in density
    rho_max = atm.get_density_for_distance(distance_xmax_geometric,
                                           zenith,
                                           observation_level=observation_level,
                                           model=atmodel) * 1e-3  # conversion g/m3 -> kg/m3

    # charge-excess fraction
    a = get_charge_excess_fraction(
        r_corrected, distance_xmax_geometric, rho_max, zenith, 
        ce_fraction_param, new_ce_param, Auger_param=Auger_param)

    # print("Fluence", f_vxB[0:10])
    # print("EL", c_early_late[0:10])
    # print("station angle", phi[0:10])
    # print("Geomagnetic angle", alpha)
    # print("CE", a[0:10])

    f_geomagnetic = geomagnetic_emission.calculate_geo_magnetic_energy_fluence(
        f_vxB * c_early_late ** 2, phi, alpha, a)

    # sort = np.argsort(r)
    # for x1, x2, x3, x4, x5, x6 in zip(r[sort], f_geomagnetic[sort], a[sort], c_early_late[sort], np.cos(phi)[sort], f_vxB[sort]):
    #     # print(x1, x2, x6, x3, x4, x5, (1 + x5 /
    #     #                            np.abs(np.sin(alpha)) * np.sqrt(x3)) ** 2)
    #     print(np.sqrt(x3), np.abs(np.sin(alpha)),
    #           (1 + x5 / np.abs(np.sin(alpha)) * np.sqrt(x3)) ** 2)

    return f_geomagnetic


def objectiv_ldf_has_param(
        params, xdata, f_vxB, f_geo_ldf, event, fit_mask,
        observation_level, atmodel, n0,
        fluence_error,
        rel_weight=0.03, add_abs_weight=1e-4, 
        new_ce_param=True, do_sum=False, 
        noisy_data=None, bad_timing_data=None, snr_data=None, 
        IDs=None, saturated_data=None, core_fit=None, Auger_param=False):

    # unpack parameters:
    if isinstance(params, lmfit.Parameters):
        params_dict = params.valuesdict()
    else:
        params_dict = params

    # print(params_dict)

    # unpack independent vars
    # antenna_positions_core_cs = in ground plane but core cs (z = 0)
    antenna_positions_core_cs, alpha, zenith, cs = xdata

    # apply thinning cut to fluence and stations if applicable
    if np.sum(fit_mask) != len(f_vxB):
        f_vxB = f_vxB[fit_mask]
        antenna_positions_core_cs = antenna_positions_core_cs[fit_mask]

    # Performs core fit, performs early late correction (not for z_vB)
    x_vB, y_vB, z_vB, c_early_late, distance_xmax_geometric, params_dict = \
        vary_core_and_get_early_late_factor(
            params_dict, antenna_positions_core_cs, cs,
            observation_level, atmodel, n0, zenith=zenith)

    if "ce0" in params_dict:
        # remove ce* from params_dict.
        ce_param = {key: params_dict.pop(key, 0)
                    for key in ["ce0", "ce1", "ce2"]}
    else:
        ce_param = None

    # calculate model prediction
    f_vxB_model = ldf_has_param(
        x_vB, y_vB, z_vB, c_early_late, distance_xmax_geometric,
        alpha, zenith, params_dict, observation_level, atmodel,
        ce_fraction_param=ce_param, f_geo_ldf=f_geo_ldf, new_ce_param=new_ce_param, Auger_param=Auger_param)

    if rel_weight and add_abs_weight:
        f_vxB_uncer = rel_weight * f_vxB + add_abs_weight * np.amax(f_vxB)

    else:
        # print("Use errors calculated from traces!")
        f_vxB_uncer = fluence_error[fit_mask]

    # check for NaN in output
    if np.isnan(f_vxB_model).any() or np.isnan(f_vxB).any() or np.isnan(f_vxB_uncer).any():
        print("NaNs detected in fluence model!")
        print("f_geo_model", np.isnan(f_vxB_model).any())
        print("f_vxB", np.isnan(f_vxB).any())
        print("f_vxB_error", np.isnan(f_vxB_uncer).any())


    # save model fluence and for fit evaluation
    event.set_station_parameter(stp.vxB_fluence_model, f_vxB_model)

    # calculate objective function (is used in minimization, default: leastsq)
    chi = (f_vxB_model - f_vxB) / f_vxB_uncer

    if np.isnan(chi).any():
        print(chi)

    if do_sum:
        return np.sum(chi ** 2)
    else:
        return chi
