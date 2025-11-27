import copy
import functools
import re
import sys

import lmfit
import numpy as np
import ray
from matplotlib import pyplot as plt
from radiotools.atmosphere import models as atm
from radiotools import helper
from radiotools import helper as rdhelp, coordinatesystems

from RadioAnalysis.framework.parameters import (
    eventParameters as evp,
    showerParameters as shp,
    stationParameters as stp,
)
from RadioAnalysis.modules.reconstruction import signal_emissions as se
from RadioAnalysis.modules.reconstruction.iminuit_wrapper import MyMinuitMinimizer
from RadioAnalysis.modules.reconstruction.ldf_fit_functions import (
    objectiv_ldf_has_param,
    objective_ldf_geo_pos,
)
from RadioAnalysis.modules.reconstruction.cherenkov_geometry import fit_cherenkov_ring_to_footprint_revent
from RadioAnalysis.modules.reconstruction.ldf_plotting import plot_ldf
from RadioAnalysis.utilities import (
    cherenkov_radius,
    early_late,
    ldfs,
    refractive_displacement,
)
from RadioAnalysis.utilities import helpers, energyreconstruction

at_model_for_avg = None


@functools.lru_cache(256)
def get_average_distance_to_xmax(zenith, observation_level, at_model, depth=750):
    global at_model_for_avg
    if at_model_for_avg is None or at_model_for_avg.model != at_model:
        at_model_for_avg = atm.Atmosphere(at_model)
    return at_model_for_avg.get_distance_xmax_geometric(
        zenith, depth, observation_level=observation_level
    )


# stupid fit
def E_rad_geo_sinalpha(E_cr, A=26.86e6, B=1.989):
    return A * (E_cr / 1e18) ** B


def get_parameter_gaus_sigmoid(
    revent, para,
    take_avg=False,
    take_core_pred=False,
    use_rho=False,
    param_r0=False,
    offline_param=False,
    param_p=False,
    param_p_slope=False,
    site_param_Auger=True,
    param_sig=False,
    param_arel=False,
    param_r02=False,
    fit_dmax=True,
    core_fit=True,
):

    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()

        shower = revent.get_shower(key=shower_type)

    # shower-wide pre-determinded parameters
    obs_level = shower.get_parameter(shp.observation_level)
    model = shower.get_parameter(shp.atmosphere_model)

    # already pre calculated depending on realistic or MC input
    dmax = shower.get_parameter(shp.distance_to_shower_maximum_geometric)
    # only used when MC parameters are needed

    # use reconstructed values if realistic input is provided
    if para.realistic_input:
        zenith = shower.get_parameter(shp.zenith_recon)
        azimuth = shower.get_parameter(shp.azimuth_recon)
        alpha = shower.get_parameter(shp.geomagnetic_angle_recon)
        # average primary energy for fit start value
        energy = 10 ** (18.5)

    else:
        zenith = shower.get_parameter(shp.zenith)
        azimuth = shower.get_parameter(shp.azimuth)
        alpha = revent.get_geomagnetic_angle()
        energy = shower.get_parameter(shp.energy)

    azenith = np.deg2rad(np.around(np.rad2deg(zenith), 4))
    avg_dmax = get_average_distance_to_xmax(azenith, obs_level, model)
    high_avg_dmax = get_average_distance_to_xmax(azenith, obs_level, model, depth=400)
    low_avg_dmax = get_average_distance_to_xmax(azenith, obs_level, model, depth=1200)

    print(f"dmax for zenith {np.around(np.rad2deg(zenith), 4)}: Low | Avg. | High: ", np.round(low_avg_dmax), np.round(avg_dmax), np.round(high_avg_dmax))

    if take_avg:
        dmax = avg_dmax

    # construct model parameters
    params = lmfit.Parameters()
    # constrict dmax fit around reasonable values for the reconstructed zenith angle
    params.add("distance_xmax_geometric", value=dmax, min=low_avg_dmax, max=high_avg_dmax, vary=fit_dmax)
    params.add("average_dxmax", value=avg_dmax, vary=False)  # dummy parameter

    if use_rho:
        rhomax = shower.get_parameter(shp.density_at_shower_maximum)

        h_avg = atm.get_height_above_ground(avg_dmax, azenith, obs_level) + obs_level
        rho_avg = atm.get_density(h_avg, model=model) * 1e-3

        # needs to be updated in fit function
        params.add("density_at_xmax", value=rhomax, vary=False)
        params.add("average_density", value=rho_avg, vary=False)  # dummy parameter

    # add azimuth angle parameter for core coordinate relation
    params.add("azimuth_angle", value=azimuth, vary=False)

    # MC core is not true core
    if take_core_pred:

        core_predict = refractive_displacement.get_predicted_core_displacement(revent, para)
        # core_predict2 = fit_cherenkov_ring_to_footprint_revent(revent)
        # print("Core prediction", core_predict)
        max_deviation = 1500

    else:
        # only calculate to set the predicted core variable
        refr_displacement = refractive_displacement.get_predicted_core_displacement(revent, para)
        # actual core prediction here
        core_predict = [0, 0]
        max_deviation = 6500
    

    # freely fit the shower core if realistic input is provided
    # use this if you use an estimated core as a start value
    if para.realistic_input:
        params.add("core_x", value=core_predict[0], vary=core_fit, min=-max_deviation, max=max_deviation)
        params.add("core_y", value=core_predict[1], vary=core_fit, min=-max_deviation, max=max_deviation)


    # fit the shower core while fixing it to the projection of the shower axis
    # use this if you use the MC core
    else:
        '''
        if statements to determine which sector of the coordinate system the shower is coming from
        then set boundaries for core fit so that the shower core cannot be fit "above" coordinate origin
        inside each sector are another instance of if to prevent issues with boundary cases
        '''

        # 0-90 degree sector: x and y both positive
        if 0 <= azimuth < np.pi / 2:

            if np.round(np.abs(core_predict[0]), 2) == 0.:
                params.add("core_y", value=core_predict[1], vary=core_fit, min=0, max=core_predict[1] + max_deviation)
                params.add("core_x", expr='core_y / tan(azimuth_angle)', vary=False)

            else:
                params.add("core_x", value=core_predict[0], vary=core_fit, min=0, max=core_predict[0] + max_deviation)
                params.add("core_y", expr='core_x * tan(azimuth_angle)', vary=False)
        
        # 90-180 degree sector: x negative and y positive
        elif np.pi / 2 <= azimuth < np.pi:

            if np.round(np.abs(core_predict[0]), 2) == 0.:
                params.add("core_y", value=core_predict[1], vary=core_fit, min=0, max=core_predict[1] + max_deviation)
                params.add("core_x", expr='core_y / tan(azimuth_angle)', vary=False)

            else:
                params.add("core_x", value=core_predict[0], vary=core_fit, min=core_predict[0] - max_deviation, max=0)
                params.add("core_y", expr='core_x * tan(azimuth_angle)', vary=False)
        
        # 180-270 degree sector: x and y both negative
        elif np.pi <= azimuth < 3 * np.pi / 2:

            if np.round(np.abs(core_predict[0]), 2) == 0.:
                params.add("core_y", value=core_predict[1], vary=core_fit, min=core_predict[1] - max_deviation, max=0)
                params.add("core_x", expr='core_y / tan(azimuth_angle)', vary=False)

            else:
                params.add("core_x", value=core_predict[0], vary=core_fit, min=core_predict[0] - max_deviation, max=0)
                params.add("core_y", expr='core_x * tan(azimuth_angle)', vary=False)
        
        # 270-360 degree sector: x positive and y negative
        elif 3 * np.pi / 2 <= azimuth < 2 * np.pi:

            if np.round(np.abs(core_predict[0]), 2) == 0.:
                params.add("core_y", value=core_predict[1], vary=core_fit, min=core_predict[1] - max_deviation, max=0)
                params.add("core_x", expr='core_y / tan(azimuth_angle)', vary=False)

            else:
                params.add("core_x", value=core_predict[0], vary=core_fit, min=0, max=core_predict[0] + max_deviation)
                params.add("core_y", expr='core_x * tan(azimuth_angle)', vary=False)


    E_geo_est = (
        # E_rad_geo_sinalpha(shower.get_parameter(shp.energy)) * np.sin(alpha) ** 2
        E_rad_geo_sinalpha(energy) * np.sin(alpha) ** 2
    )
    params.add("E_geo", value=3 * E_geo_est, min=1e5, max=1e13)

    # if not it is calculated in objective function
    if not param_r0:
        # only use it as a start value here
        r0 = 0.91 * cherenkov_radius.get_cherenkov_radius_model_from_distance(
            zenith,
            dmax,
            obs_level,
            revent.get_parameter(evp.refractive_index_at_sea_level),
            model,
        )
        params.add("r0", value=r0, min=r0 * 0.5, max=r0 * 2, vary=True)
        
        # save r0 start vlaue for plots
        shower.set_parameter(shp.r0_start, r0)

    else:
        # effective r0 parametrisation
        r0 = cherenkov_radius.get_cherenkov_radius_model_from_distance(
            zenith,
            dmax,
            obs_level,
            revent.get_parameter(evp.refractive_index_at_sea_level),
            model,
        )
        params.add("r0_start_value", value=r0, vary=False)
        
        # save r0 start vlaue for plots
        shower.set_parameter(shp.r0_start, r0)

        if site_param_Auger:

            # AUGER
            # fit with all sims
            params.add("r0", vary=False, expr="r0_start_value * (0.94061131 + (-2.2048e-07) \
                * distance_xmax_geometric + (-15960366.4) / distance_xmax_geometric / distance_xmax_geometric)")

            # fit with only low E protons
            # params.add("r0", expr="r0_start_value * (0.94334282 + (-1.9953e-07) * distance_xmax_geometric + (-15093641.6) / distance_xmax_geometric / distance_xmax_geometric)")

        else:

            # GRAND
            # fit with all sims
            params.add("r0", vary=False, expr="r0_start_value * (0.81214082 + (6.1305e-07) \
                * distance_xmax_geometric + (-31818142.4) / distance_xmax_geometric / distance_xmax_geometric)")
           
            # fit with only low E protons
            # params.add("r0", expr="r0_start_value * (0.94334282 + (-1.9953e-07) * distance_xmax_geometric + (-15093641.6) / distance_xmax_geometric / distance_xmax_geometric)")


    if offline_param:
        params.add(
            "sig",
            expr="0.16848311 * pow(distance_xmax_geometric - 5000, 0.69447957) + 39.81137662",
        )
        params.add(
            "p",
            expr="1.85054143e+00 / (1 + exp(-4.20849856e-05 * (distance_xmax_geometric + 2.86110554e+04)))",
        )
        params.add(
            "arel",
            expr="7.47991658e-01 + 8.11304687e-07 * distance_xmax_geometric + \
                 1.72095209e+07 / distance_xmax_geometric / distance_xmax_geometric",
        )
        params.add(
            "r02",
            expr="5.33720788e-01 + 8.37285946e-07 * distance_xmax_geometric + \
                 5.27475125e+07 / distance_xmax_geometric / distance_xmax_geometric",
        )

    # adding my own fit parameters from here
    # they're always at the bottom of each indent

    else:

        if not param_sig:
            if site_param_Auger:
                # params.add(
                #     'sig', value=1.39506207e-01 * (dmax - 5000) ** 7.09861340e-01 + 5.39164661e+01
                #     - 1.84052852e-03 * (dmax - avg_dmax))

                # Lukas first try for start values
                # AUGER
                params.add("sig", value=230, vary=True, min=0)  # , max=1000)

                # start values using the parametrisations
                # params.add(
                #     "sig", min=0,
                #     value=2.82209732e-02 * (dmax - 5000) ** 8.00859075e-01 + 6.12971298e+01)

            else:
                # GRAND
                params.add("sig", value=250, vary=True, min=0, max=600)

        else:
            if site_param_Auger:
                # Lukas Auger 50-200MHz parametrisation
                params.add(
                    "sig",
                    expr="2.71147327e-02 * (distance_xmax_geometric - 5000) ** 8.05075351e-01 + 6.19714390e+01",
                )

                # Felix' final parametrisation
                # params.add(
                #     "sig",
                #     expr="0.13176183 * (distance_xmax_geometric - 5000) ** 0.71437054 +  56.30941015",
                # )
            else:
                # Lukas China 50-200MHz parametrisation
                params.add(
                    "sig",
                    expr="3.94026614e-02 * (distance_xmax_geometric - 5000) ** 7.59980861e-01 + 6.09984088e+01",
                )


        if not param_p:
            if site_param_Auger:
                # params.add('p', min=10, max=1000,
                #             value=1.63989873e+02 * np.exp(-2.75777834e-05 * dmax) +
                #             6.81576551e+01 + 0 * (dmax - avg_dmax))

                # Lukas first try for start values
                # AUGER
                params.add("p", value=257, vary=True, min=0, max=400)

                # start values using the parametrisations
                # params.add(
                #     "p", min=0, max=350, 
                #     value=1.06591645e+02 * np.exp(4.49404087e-05 * dmax) + 2.31442963e+02,
                # )

            else:
                # GRAND
                params.add("p", value=220, vary=True, min=0, max=350) # 275

                # start values using the parametrisations
                # params.add(
                #     "p",  vary=True, min=0,
                #     value=2.47077170e+02 + (-2.06317205e-04) * dmax + (-8.93411554e+09) \
                #              / dmax / dmax,
                # )

        else:
            if site_param_Auger:
                # Lukas Auger 50-200MHz parametrisation
                params.add(
                    "p",
                    expr="2.82212329e+02 + (-3.65695435e-04) * distance_xmax_geometric + (-6.45711498e+09) \
                             / distance_xmax_geometric / distance_xmax_geometric",
                )
                

                # Felix' final parametrisation
                # params.add(
                #     "p",
                #     expr="1.54942405e+02 * exp((-1) * 2.50177591e-05 * distance_xmax_geometric) + 6.49133050e+01",
                # )  #  wo d750 first param

            else:
                # Lukas China 50-200MHz parametrisation
                params.add(
                    "p",
                    expr="2.46529404e+02 + (-2.23065116e-04) * distance_xmax_geometric + (-9.29903285e+09) \
                             / distance_xmax_geometric / distance_xmax_geometric",
                )


        if not param_p_slope:
            if site_param_Auger:
                
                # Lukas first try for start values
                # AUGER
                params.add("p_slope", value=1.6, vary=True, min=1.0, max=2.2)
                # params.add("p_slope", value=2, vary=False)

                # start values using the parametrisations
                # params.add(
                #     'p_slope', min=1, max=2.2,
                #     value=1.46438531e+00  + (-5.26817171e-08) * dmax + 3.29440576e+07 / dmax / dmax
                # )

            else:
                # Lukas first try for start values
                # GRAND
                params.add("p_slope", value=1.6, vary=True, min=1.2, max=1.8)

        else:
            if site_param_Auger:
                # Lukas Auger 50-200MHz param
                params.add(
                        "p_slope",
                        expr="1.46438531e+00  + (-5.26817171e-08) * distance_xmax_geometric + 3.29440576e+07 \
                        / distance_xmax_geometric / distance_xmax_geometric")
            
            else:
                # Lukas GRAND 50-200MHz param
                params.add(
                        "p_slope",
                        expr="1.52037333e+00 + (3.82623066e-07) * distance_xmax_geometric + (-4.26437099e+07) \
                        / distance_xmax_geometric / distance_xmax_geometric"
                    ) #1.52037333e+00,3.82623066e-07,-4.26437099e+07


        if not param_arel:
            if site_param_Auger:
                # params.add(
                #     'arel', min=0.5, max=1, value=7.47991658e-01 + 8.11304687e-07 * dmax +
                #     1.72095209e+07 / dmax / dmax)

                # Lukas first try for start values
                # AUGER
                params.add("arel", value=0.25, min=0.01, max=1)

                # start values using the parametrisations
                # params.add(
                #     'arel', min=0.1, max=1, 
                #     value=1.93240394e-01 + 4.82191106e-07 * dmax +
                #     1.55642760e+07 / dmax / dmax
                # )

            else:
                # GRAND
                params.add("arel", value=0.3, min=0.15, max=1)

        else:
            if site_param_Auger:
                # Lukas Auger 50-200MHz param
                params.add(
                   "arel",
                    expr="2.32868923e-01 + 2.06291784e-07 * distance_xmax_geometric + -3.78691182e+06 \
                         / distance_xmax_geometric / distance_xmax_geometric",
                )

                # Felix' final param
                # params.add(
                #     "arel",
                #     expr="7.57449757e-01  + 7.68399447e-07 * distance_xmax_geometric + 1.98026585e+07 \
                #          / distance_xmax_geometric / distance_xmax_geometric",
                # )

            else:
                params.add(
                    'arel', expr="2.65672136e-01 + 5.74786264e-07 * distance_xmax_geometric + (-1.92805026e+07) \
                            / distance_xmax_geometric / distance_xmax_geometric"
                    )
                        

        if not param_r02:
            if site_param_Auger:
                # params.add(
                #     'r02', min=0, max=1, value=7.29362930e-01 / \
                #         (1 + np.exp(-8.24785821e-05 * (dmax - 4.17849683e+03))))

                # Lukas first try for start values
                # AUGER
                params.add("r02", value=0.5, min=0.01, max=0.81, vary=True) # 0.85

                # start values using the parametrisations
                # params.add(
                #     'r02', min=0.1, max=0.90, 
                #     value=7.44485345e-01 + 5.29056889e-07 * dmax + (-1.50450570e+08) / dmax / dmax
                # )

            else:
                # GRAND
                params.add("r02", value=0.6, min=0.0, max=0.78, vary=True) # free: 0.77

                # start values using the parametrisations
                # params.add(
                #     'r02', min=0.0, max=0.77, vary=True,
                #     value=5.82931288e-01 + 1.50518509e-06 * dmax + (-1.99075364e+08) / dmax / dmax
                # )

        else:
            if site_param_Auger:
                # Lukas Auger 50-200MHz param
                params.add(
                    "r02",
                    expr="5.98281908e-01 + 1.29545941e-06 * distance_xmax_geometric + \
                          (-9.86454075e+07) / distance_xmax_geometric / distance_xmax_geometric",
                )

                # Felix' final param
                # params.add(
                #     "r02",
                #     expr="5.51709206e-01 + 6.87661913e-07 * distance_xmax_geometric + \
                #             6.61581738e+07 / distance_xmax_geometric / distance_xmax_geometric",
                # )

            else:
                params.add(
                    "r02",
                    expr="5.60308106e-01 +  1.04413624e-06 * distance_xmax_geometric + \
                        (-1.43854800e+08) / distance_xmax_geometric / distance_xmax_geometric",
                )

    # params.add("slope", value=6.25, vary=False, min=0, max=50)

    # for GRAND version
    params.add("slope", value=6, vary=False, min=0, max=50)

    # slope parameter for Felix 30-80 Mhz fit
    #params.add('slope', value=5, min=1, max=10, vary=False)

    # print(params)

    return params


def get_selection_mask(
    revent,
    reject_vxB=False,
    reject_thinning=False,
    dist_lim=False,
    select_stations=False,
):
    # select stations for fit

    station_position_vBvvB = revent.get_station_position_vB_vvB()

    if select_stations:
        select_or_cleaned = np.array([False] * len(station_position_vBvvB))
        indecies = helpers.get_index_for_random_stations(revent)
        select_or_cleaned[indecies] = True
    else:
        select_or_cleaned = np.array([True] * len(station_position_vBvvB))

    if reject_thinning:
        thinning_clean_masks = revent.get_station_parameter(stp.cleaned_from_thinning)
        select_or_cleaned = np.all([select_or_cleaned, thinning_clean_masks], axis=0)

    if dist_lim:
        r_che = refractive_displacement.get_cherenkov_radius_param_revent(revent)
        zenith = revent.get_shower().get_parameter(shp.zenith)
        num_rche = 4 if zenith < np.deg2rad(70) else 3
        mask = np.all(
            [
                np.sqrt(np.sum(station_position_vBvvB[:, :-1] ** 2, axis=1))
                < num_rche * r_che
            ],
            axis=0,
        )
        select_or_cleaned = np.all([select_or_cleaned, mask], axis=0)

    if reject_vxB:
        phi = revent.get_station_angle_to_vB()
        vxB = helpers.mask_polar_angle(phi, angles_in_deg=[0, 180, 360], atol_in_deg=1)
        if np.sum(vxB) != 60:
            print("Waring, reject not 60 station because of vxB")
        select_or_cleaned = np.all([select_or_cleaned, ~vxB], axis=0)

    return select_or_cleaned


def fit_pos_has_ldf_compare(events, para):

    for revent in events:
        shower = revent.get_shower()
        if re.search("0000", "%06d" % revent.get_run_number()):
            print(revent.get_run_number())

        station_positions = revent.get_station_parameter(stp.position)
        energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)
        select_or_cleaned = get_selection_mask(revent)
        core_gp = shower.get_parameter(shp.core)
        zenith = shower.get_parameter(shp.zenith)

        if core_gp[0] != 0:
            sys.exit("core is already altered")

        antenna_positions_core_cs = station_positions - core_gp
        cs = revent.get_coordinate_transformation()
        xdata = [
            antenna_positions_core_cs[select_or_cleaned],
            energy_fluence_vector[select_or_cleaned],
            cs,
        ]

        atmodel = shower.get_parameter(shp.atmosphere_model)
        obs_lvl = shower.get_parameter(shp.observation_level)
        n0 = revent.get_parameter(evp.refractive_index_at_sea_level)

        # set up data for minimization
        fcn_kwargs1 = {
            "xdata": xdata,
            "f_geo_ldf": ldfs.f_E_geo_gaus_sigmoid,
            "zenith": zenith,
            "rel_weight": 0.02,
            "add_abs_weight": 1e-4,
            "observation_level": obs_lvl,
            "atmodel": atmodel,
            "n0": n0,
            "do_sum": True,
        }

        # fcn_kwargs2 = {'xdata': xdata,
        #             'f_geo_ldf': ldfs.f_E_geo_gaus_sigmoid_simple_p,
        #             "zenith": zenith,
        #             "rel_weight": 0.02,
        #             "add_abs_weight": 1e-4,
        #             "observation_level": obs_lvl,
        #             "atmodel": atmodel, "n0": n0,
        #             'do_sum': True}

        params1 = get_parameter_gaus_sigmoid(
            revent,
            fit_dmax=False,
            core_fit=True,
            param_p=False,
            param_r0=False,
            take_avg=False,
        )

        params2 = params1
        # params2["p"].set(value=1.8, min=1, max=2)

        mym1 = MyMinuitMinimizer(
            objective_ldf_geo_pos, params1, fcn_kwargs1, verbose=para.plot
        )

        # construct model
        fcn_kwargs2 = copy.deepcopy(fcn_kwargs1)

        fcn_kwargs2["do_sum"] = False
        ldf_model = lmfit.Minimizer(
            objective_ldf_geo_pos, params2, fcn_kws=fcn_kwargs2, nan_policy="omit"
        )
        result = ldf_model.minimize("least_squares")
        pars2 = result.params.valuesdict()

        # mym2 = MyMinuitMinimizer(objective_ldf_geo_pos,
        #                          params2, fcn_kwargs2, verbose=para.plot)

        mym1.migrad()
        mym1.hesse()
        # mym2.migrad()
        # mym2.hesse()

        pars1 = mym1.get_all_param_value_dict()
        fig, axs = plot_ldf(
            pars1,
            fcn_kwargs1,
            pos=True,
            opt="return",
            plot_gaus_kwargs={"color": "C2", "label": '"soft" p'},
        ) # type: ignore

        # pars2 = mym2.get_all_param_value_dict()
        axs = plot_ldf(
            pars2,
            fcn_kwargs2,
            pos=True,
            opt="return",
            axs=axs,
            plot_gaus_kwargs={"color": "C1", "label": '"hard" p'},
        )

        axs[0].legend(ncol=2, loc="lower right")
        plt.tight_layout()
        plt.savefig("ldf_comp_%06d%s.png" % (revent.get_run_number(), para.label))

    post_process_store(revent, mym1, select_or_cleaned)
    post_process_store(revent, result, select_or_cleaned)


def data_fitting_sequence(objective, params, fcn_kwargs, para):
    # construct model
    ldf_model = lmfit.Minimizer(
        objective, params, fcn_kws=fcn_kwargs, nan_policy="omit"
    )

    try:
        # minimization
        # with default method error of changing array size is causing abort
        result = ldf_model.minimize("least_squares")
    except ValueError as e:
        return None

    if para.plot:
        print(lmfit.fit_report(result))

    # update parameter
    params = result.params
    params["distance_xmax_geometric"].set(vary=True)

    # construct model
    ldf_model = lmfit.Minimizer(
        objective, params, fcn_kws=fcn_kwargs, nan_policy="omit"
    )

    try:
        # minimization
        # with default method error of changing array size is causing abort
        result = ldf_model.minimize("least_squares")
    except ValueError as e:
        return None

    if para.plot:
        print(lmfit.fit_report(result))

    # update parameter
    params = result.params
    params["distance_xmax_geometric"].set(vary=False)
    params["core_x"].set(vary=True)
    params["core_y"].set(vary=True)

    # construct model
    ldf_model = lmfit.Minimizer(
        objective, params, fcn_kws=fcn_kwargs, nan_policy="omit"
    )

    try:
        # minimization
        # with default method error of changing array size is causing abort
        result = ldf_model.minimize("least_squares")
    except ValueError as e:
        return None

    if para.plot:
        print(lmfit.fit_report(result))

    # update parameter
    params = result.params
    params["distance_xmax_geometric"].set(vary=True)

    # construct model
    ldf_model = lmfit.Minimizer(
        objective, params, fcn_kws=fcn_kwargs, nan_policy="omit"
    )

    try:
        # minimization
        # with default method error of changing array size is causing abort
        result = ldf_model.minimize("least_squares")
    except ValueError as e:
        return None

    return result


def _fit_pos_has_ldf_E_geo(revent, para):

    shower = revent.get_shower()
    if re.search("00000", "%06d" % revent.get_run_number()):
        print(revent.get_run_number())

    station_positions = revent.get_station_parameter(stp.position)

    if len(station_positions) < 5:
        return 0

    energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)

    select_or_cleaned = get_selection_mask(revent, dist_lim=False, reject_vxB=False)

    core_gp = shower.get_parameter(shp.core)
    zenith = shower.get_parameter(shp.zenith)

    atmodel = shower.get_parameter(shp.atmosphere_model)
    obs_lvl = shower.get_parameter(shp.observation_level)
    n0 = revent.get_parameter(evp.refractive_index_at_sea_level)

    # if core_gp[0] != 0:
    #     sys.exit("core is already altered")

    antenna_positions_core_cs = station_positions - core_gp
    cs = revent.get_coordinate_transformation()
    xdata = [
        antenna_positions_core_cs[select_or_cleaned],
        energy_fluence_vector[select_or_cleaned],
        cs,
    ]

    real = False
    if 1:
        p_soft = True

        # parameters for free fit for 50-200 MHz
        params = get_parameter_gaus_sigmoid(
            revent,
            fit_dmax=True,
            core_fit=True,
            take_core_pred=True,
            use_rho=False,
            offline_param=False,
            site_param_Auger=True,
            take_avg=False,
            param_r0=True,
            param_sig=False,
            param_p=False,
            param_arel=False,
            param_r02=False,
        )

        if p_soft:
            func = ldfs.f_E_geo_gaus_sigmoid
        else:
            func = ldfs.f_E_geo_gaus_sigmoid_simple_p
    else:
        params = get_parameter_pol3(revent)
        func = ldfs.f_E_geo

    # set up data for minimization
    fcn_kwargs = {
        "xdata": xdata,
        "f_geo_ldf": func,
        "zenith": zenith,
        "rel_weight": 0.03,
        "add_abs_weight": 1e-4 if not real else 0.03,
        "observation_level": obs_lvl,
        "atmodel": atmodel,
        "n0": n0,
        "alpha": np.rad2deg(shower.get_parameter(shp.geomagnetic_angle)),
    }

    if 0:
        fcn_kwargs["do_sum"] = True
        mym = MyMinuitMinimizer(
            objective_ldf_geo_pos, params, fcn_kwargs, verbose=para.plot
        )

        try:
            mym.migrad()
            mym.hesse()

        except ValueError as e:
            print("failed minuit", e, revent.get_run_number())
            shower.set_parameter(shp.fit_result, False)
            return 0

        result = mym
        pars = result.get_all_param_value_dict()

        if para.plot or result.get_all_param_value_dict()["arel"] > 1:
            # if para.plot:
            # print(pars)
            result.print_result()

    else:
        # construct model
        fcn_kwargs["do_sum"] = False
        ldf_model = lmfit.Minimizer(
            objective_ldf_geo_pos, params, fcn_kws=fcn_kwargs, nan_policy="omit"
        )

        if not real:
            try:
                # minimization
                # with default method error of changing array size is causing abort
                result = ldf_model.minimize("least_squares")
            except ValueError as e:
                print("failed", e, revent.get_run_number())
                shower.set_parameter(shp.fit_result, False)
                return 0
        else:
            # not called
            result = data_fitting_sequence(
                objective_ldf_geo_pos, params, fcn_kwargs, para
            )

            if result is None:
                print("failed", revent.get_run_number())
                shower.set_parameter(shp.fit_result, False)
                return 0

        if para.plot:
            print(lmfit.fit_report(result))

        pars = result.params.valuesdict()
        # print(pars)

    if para.plot:
        print(revent.get_run_number())
        title = r"E = %.2f EeV, $\theta$ = %.2f$^\circ$, $\alpha$ = %.2f$^\circ$" % (
            shower.get_parameter(shp.energy) / 1e18,
            np.rad2deg(shower.get_parameter(shp.zenith)),
            np.rad2deg(shower.get_parameter(shp.geomagnetic_angle)),
        )
        plot_ldf(
            pars,
            fcn_kwargs,
            pos=True,
            opt="save",
            label="_%06d%s" % (revent.get_run_number(), para.label),
            title=title,
        )

    post_process_store(revent, result, select_or_cleaned)


@ray.remote
def fit_pos_has_ldf_E_geo_ray(revent, para):
    _fit_pos_has_ldf_E_geo(revent, para)
    return revent


def fit_pos_has_ldf_E_geo(events, para):

    for revent in events:
        _fit_pos_has_ldf_E_geo(revent, para)


def get_fixed_params_from_prev_fit(revent, all_free):
    shower = revent.get_shower()
    params = shower.get_parameter(shp.geomagnetic_ldf_parameter)

    # construct model parameter
    lmfit_params = lmfit.Parameters()
    for key in params:
        lmfit_params.add(key, value=params[key], vary=all_free)

    lmfit_params.add(
        "distance_xmax_geometric",
        value=shower.get_parameter(shp.distance_to_shower_maximum_geometric_fit),
        min=5001,
        max=300000,
        vary=all_free,
    )

    # with prev fit, stations are already are already around sym center
    lmfit_params.add("core_x", value=0, min=-500, max=500, vary=all_free)
    lmfit_params.add("core_y", value=0, min=-500, max=500, vary=all_free)

    lmfit_params["E_geo"].set(vary=all_free)
    lmfit_params["r0"].set(min=100, max=2000)
    return lmfit_params


def fit_ce_fraction_sequence(objective, params, fcn_kwargs, para):

    ldf_model = lmfit.Minimizer(objectiv_ldf_has_param, params, fcn_kws=fcn_kwargs)

    # minimization
    try:
        result = ldf_model.minimize()
    except ValueError as e:
        print("failed (first fit):", e)
        return None

    if 0:
        # update parameter
        params = result.params
        params["core_x"].set(vary=True)
        params["core_y"].set(vary=True)

        # model
        ldf_model = lmfit.Minimizer(objectiv_ldf_has_param, params, fcn_kws=fcn_kwargs)

        # minimization
        try:
            result = ldf_model.minimize()
        except ValueError as e:
            print("failed (core fit):", e)
            return None

    if 0:
        # update parameter
        params = result.params
        # params['ce1'].set(vary=True)
        params["ce2"].set(vary=True)

        ldf_model = lmfit.Minimizer(objectiv_ldf_has_param, params, fcn_kws=fcn_kwargs)

        # minimization
        try:
            result = ldf_model.minimize()
        except ValueError as e:
            print("failed (ce2 fit):", e)
            return None

    if 0:
        # update parameter
        params = result.params
        params["ce0"].set(vary=True)
        # params['ce1'].set(vary=False)

        ldf_model = lmfit.Minimizer(objectiv_ldf_has_param, params, fcn_kws=fcn_kwargs)

        # minimization
        try:
            result = ldf_model.minimize()
        except ValueError as e:
            print("failed (ce0 fit):", e)
            return None

    return result


def _fit_param_has_ldf(revent, para):

    if re.search("00000", "%06d" % revent.get_run_number()):
        print("Run number:", revent.get_run_number())

    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()

        shower = revent.get_shower(key=shower_type)

    GRAND_shower = shower_type == evp.GRAND_shower


    # check whether simulation has any antennas
    if not revent.has_station_parameter(stp.position):
        print("No triggered antennas detected! Skipping...")

        if GRAND_shower:
                run_no = revent.get_run_number()
                evt_no = revent.get_id()
                # write empty entry to data file for failed reconstructions
                with open("event_candidate_params.csv", "a") as txt_file:
                        txt_file.write(str(int(run_no)) + " " + str(int(evt_no)) + "\n")
        return 0

    # randomly eliminate 10% of stations to test reconstruction quality (rounded down)
    station_count = len(revent.get_station_parameter(stp.energy_fluence))

    if 0:
        eliminate_inds = np.random.choice(np.arange(station_count), size=int(np.round(station_count * 0.1)), replace=False)
        random_mask = np.full(station_count, True)
        random_mask[eliminate_inds] = False
        print(f"Randomly eliminating {np.sum(~random_mask)}/{station_count} of stations!")

    else:
        random_mask = np.full(station_count, True)

    pos_vB = np.array(revent.get_station_position_vB_vvB(key=shower_type, realistic_input=para.realistic_input))

    atmodel = shower.get_parameter(shp.atmosphere_model)
    obs_lvl = shower.get_parameter(shp.observation_level)
    n0 = revent.get_parameter(evp.refractive_index_at_sea_level)
    mag_field = revent.get_parameter(evp.magnetic_field_vector)

    # realistic shower data with estimated core and reconstructed arrival direction
    if para.realistic_input:
        print("Use estimated and reconstructed parameters for LDF fit!")
        zenith = shower.get_parameter(shp.zenith_recon)
        azimuth = shower.get_parameter(shp.azimuth_recon)
        core_gp = shower.get_parameter(shp.core_estimate)
        alpha = shower.get_parameter(shp.geomagnetic_angle_recon)
        cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=mag_field)
        station_ids = np.array(revent.get_station_parameter(stp.id))[random_mask]

    # parameters to use if you want to use the MC shower data
    else:
        zenith = shower.get_parameter(shp.zenith)
        core_gp = shower.get_parameter(shp.core)
        alpha = revent.get_geomagnetic_angle()
        cs = revent.get_coordinate_transformation()
        station_ids = 0 # for starshapes!

    # fluence and positions
    f_vxB = np.array(revent.get_station_parameter(stp.energy_fluence)[:, 0])[random_mask]
    station_positions = np.array(revent.get_station_parameter(stp.position))[random_mask]

    print(f"{len(f_vxB)} signal antennas!")

    #
    # conditional parameters for extra plot details
    #
    
    # fluences of points cut by ADC noise criterion
    if revent.has_station_parameter(stp.energy_fluence_noisy):
        # get boisy stations and fluence
        f_vxB_noisy = revent.get_station_parameter(stp.energy_fluence_noisy)[:, 0]
        print(f"{len(f_vxB_noisy)} noisy antennas in ADC!")
        station_positions_noisy = revent.get_station_parameter(stp.positions_noisy)
        errors_noisy = revent.get_station_parameter(stp.error_noisy)
        # put in one variable and corrct to right core system
        noisy_data = [f_vxB_noisy, station_positions_noisy - core_gp, errors_noisy]

    else:
        noisy_data = [np.array(None), np.array(None), np.array(None)]
        print("0 noisy antennas in ADC!")
        
    # fluences of points cut by peak timing noise criterion
    if revent.has_station_parameter(stp.energy_fluence_bad_timing):
        # get boisy stations and fluence
        f_vxB_bad_timing = revent.get_station_parameter(stp.energy_fluence_bad_timing)[:, 0]
        print(f"{len(f_vxB_bad_timing)} bad timing antennas!")
        station_positions_bad_timing = revent.get_station_parameter(stp.positions_bad_timing)
        errors_bad_timing = revent.get_station_parameter(stp.error_bad_timing)
        # put in one variable and corrct to right core system
        bad_timing_data = [f_vxB_bad_timing, station_positions_bad_timing - core_gp, errors_bad_timing]

    else:
        bad_timing_data = [np.array(None), np.array(None), np.array(None)]
        print("0 bad timing antennas!")

    # fluences of points cut by SNR criterion
    if revent.has_station_parameter(stp.energy_fluence_snr):
        # get snr cut stations and fluence
        f_vxB_snr = revent.get_station_parameter(stp.energy_fluence_snr)[:, 0]
        print(f"{len(f_vxB_snr)} SNR cut antennas!")
        station_positions_snr = revent.get_station_parameter(stp.positions_snr)
        errors_snr = revent.get_station_parameter(stp.error_snr)
        # put in one variable and corrct to right core system
        snr_data = [f_vxB_snr, station_positions_snr - core_gp, errors_snr]

    else:
        snr_data = [np.array(None), np.array(None), np.array(None)]
        print("0 SNR cut antennas!")


    if revent.has_station_parameter(stp.saturated_fluence):
        # get list of saturated antennas
        saturated_mask = revent.get_station_parameter(stp.saturated)[random_mask]
        f_vxB_saturated = revent.get_station_parameter(stp.saturated_fluence)[:, 0]
        print(f"{len(f_vxB_saturated)} saturated antennas!")
        station_positions_saturated = revent.get_station_parameter(stp.saturated_positions)
        errors_saturated = revent.get_station_parameter(stp.saturated_errors)
        # put in one variable and correct to right core system
        saturated_data = [f_vxB_saturated, station_positions_saturated - core_gp, errors_saturated]
    else:
        # set mask to all False otherwise
        saturated_mask = np.zeros(f_vxB.shape, dtype=bool)
        saturated_data = [np.array(None), np.array(None), np.array(None)]
        print("0 saturated antennas!")

        
    re_fit = False


    # if core_gp[0] != 0 and not re_fit:
    # masks for reconstruction and signal model case to improve fit quality in both
    if 1: # revent.has_station_parameter(stp.energy_fluence_snr) or revent.has_station_parameter(stp.energy_fluence_noisy): # version for reconstruction
        
        print("Use calculated errors!")
        
        if 1:
            # get error calculated from noise fluence, detector sensitivity uncertainty and RMS of noise window
            f_vxB_error = revent.get_station_parameter(stp.vxB_error)[random_mask]

            # set previous error parameters to wrong
            rel_weight = abs_weight = False

        else:
            # define error values for fluence
            rel_weight = 0.125 # 0.05 # multiplied with fluence value 
            abs_weight = 5e-3 # 5e-4 # multiplied with max fluence value
            
            f_vxB_error = f_vxB * rel_weight + abs_weight * np.amax(f_vxB)
        
        # mask for SNR cut on fluence values
        fit_mask = np.array((f_vxB / f_vxB_error) ** 2 > 0)
        # use inverted saturation mask to eliminate the antennas affected by saturation
        fit_mask = ~saturated_mask

        # remove infill if requested
        if revent.has_station_parameter(stp.infill_id) and para.remove_infill:
            # get all antenna IDs that don't belong to the infill
            infill_ids = revent.get_station_parameter(stp.infill_id)
            # select event antenna IDs for them
            infill_mask = np.isin(station_ids, infill_ids)

            # replace fit_mask with this
            fit_mask = np.all([~saturated_mask, infill_mask], axis=0)

            print(f"{np.sum(~infill_mask)} in-fill and {np.sum(saturated_mask)} saturated stations | {np.sum(fit_mask)}/{len(f_vxB)} remain!")

        # only infill if requested
        elif revent.has_station_parameter(stp.infill_id) and para.only_infill:
            # get all antenna IDs that don't belong to the infill
            only_infill_ids = revent.get_station_parameter(stp.only_infill_id)
            # select event antenna IDs for them
            only_infill_mask = np.isin(station_ids, only_infill_ids)

            # replace fit_mask with this
            fit_mask = np.all([~saturated_mask, only_infill_mask], axis=0)

            print(f"{np.sum(~only_infill_mask)} only in-fill and {np.sum(saturated_mask)} saturated stations | {np.sum(fit_mask)}/{len(f_vxB)} remain!")


        else:
            if np.sum(saturated_mask) > 0:
                print(f"{np.sum(saturated_mask)}/{len(f_vxB)} saturated stations for this event!")

        # save f-vxB fluence with error for later pull evaluation
        revent.set_station_parameter(stp.vxB_fluence_simulated, f_vxB[fit_mask])
        revent.set_station_parameter_error(stp.vxB_fluence_simulated, f_vxB_error[fit_mask])

        if np.sum(fit_mask) <= 4:
            print(f"{np.sum(fit_mask)}/{len(f_vxB)} stations are not enough within fit range for this event! Skipping...")

            if GRAND_shower:
                run_no = revent.get_run_number()
                evt_no = revent.get_id()
                # write empty entry to data file for failed reconstructions
                with open("event_candidate_params.csv", "a") as txt_file:
                        txt_file.write(str(int(run_no)) + " " + str(int(evt_no)) + "\n")

            return 0
        
    else: # version for signal model so far
        
        print("Use modelled errors!")

        # define error values for fluence
        rel_weight = 0.03 # multiplied with fluence value
        abs_weight = 1e-4  # multiplied with max fluence value

        f_vxB_error = f_vxB * rel_weight + abs_weight * np.amax(f_vxB)

        # mask for pulses affected by thinning
        # fit_mask = revent.get_station_parameter(stp.cleaned_from_thinning)
        # include all stations in fit
        fit_mask = np.ones(f_vxB.shape, dtype=bool)
        
        # save f-vxB fluence with error for later pull evaluation
        revent.set_station_parameter(stp.vxB_fluence_simulated, f_vxB[fit_mask])
        revent.set_station_parameter_error(stp.vxB_fluence_simulated, f_vxB_error[fit_mask])

        if np.sum(fit_mask) <= 4:
            print(f"{np.sum(fit_mask)} stations are not enough within fit range for this event! Skipping...")
            return 0
        
    
    # for reveral of realistic input value
    if para.realistic_input:
        use_MC = False
        
    else:
        use_MC = True
        
    # True for Auger fit; False for GRAND fit (both 50-200 MHz)
    if atmodel == 27:
        print("Using Auger atmosphere model and parametrisations!")
        Auger = True
    else:
        print("Using GRAND atmosphere model and parametrisations!")
        Auger = False

    if re_fit:
        params = get_fixed_params_from_prev_fit(revent, all_free=False)
    else:
        params = get_parameter_gaus_sigmoid(
            revent, para, 
            take_avg=para.realistic_input,           # for reconstruction, use an averaged dmax as start value for LDF fit while keeping true value in data bank
            fit_dmax=True,
            core_fit=True,
            take_core_pred=use_MC,    # only use this for simulation with MC core given and used in fit
            site_param_Auger=Auger,  
            param_r0=True,           
            param_sig=True,
            param_p=True,
            param_arel=True,
            param_p_slope=True,
            param_r02=True,
            # current fit order: r02 -> p_slope -> arel - p -> sig -> r0
        )

        func = ldfs.f_E_geo_gaus_sigmoid_p_slope


    # data for fit
    antenna_positions_core_cs = station_positions - core_gp
    xdata = [antenna_positions_core_cs, alpha, zenith, cs]

    # set up data for minimization
    fcn_kwargs = {
        "event": revent,
        "xdata": xdata,
        "core_fit": core_gp,
        "IDs": station_ids,
        "noisy_data": noisy_data,
        "bad_timing_data": bad_timing_data,
        "snr_data": snr_data,
        "saturated_data": saturated_data,
        "fit_mask": fit_mask,
        "f_vxB": f_vxB,
        "fluence_error": f_vxB_error, 
        "f_geo_ldf": func,
        "rel_weight": rel_weight,
        "add_abs_weight": abs_weight,
        "observation_level": obs_lvl,
        "atmodel": atmodel,
        "n0": n0,
        "new_ce_param": True,
        "Auger_param": Auger
    }

    # model
    ldf_model = lmfit.Minimizer(objectiv_ldf_has_param, params, fcn_kws=fcn_kwargs)

    if 1:
        # minimization
        result = ldf_model.minimize()
        try:
            result = ldf_model.minimize()
        except (TypeError, ValueError) as e:
            print(revent.get_run_number(), "has param fit failed:", e)
            shower.set_parameter(shp.fit_result, False)
            return 1
        
        print("Fit successful! Zenith angle", np.round(np.rad2deg(zenith), 2))
        # print(f"Est./true dmax: {shower.get_parameter(shp.distance_to_shower_maximum_geometric)}")
        # print(f"Fitted dmax:    {result.params['distance_xmax_geometric'].value}")

    else:

        result = data_fitting_sequence(objectiv_ldf_has_param, params, fcn_kwargs, para)

        if result is None:
            print("fit_ce_fraction_sequence failed", revent.get_run_number())
            shower.set_parameter(shp.fit_result, False)
            return 1

    if para.realistic_input:
        # get density at xmax
        rho_max = shower.get_parameter(shp.density_at_shower_maximum)
        # avg_rho = 0.3113839703192573 # for Auger 
        avg_rho = 0.2171571758825658 # for GRAND
        # get radiation energy with error
        egeo_fit = result.params['E_geo'].value
        egeo_fit_err = result.params['E_geo'].stderr
        dmax_fit = result.params['distance_xmax_geometric'].value
        dmax_fit_err = result.params['distance_xmax_geometric'].stderr
        if egeo_fit_err == None: egeo_fit_err = 0
        if dmax_fit_err == None: dmax_fit_err = 0
        rec_data = np.array([egeo_fit, np.sin(alpha), rho_max])

        # fit values for density correction and conversion from Sgeo to Eem
        rec_fit_values = np.array([13.4858593e9, 1.9961499, 196.5323011, -0.0030236, 0.0338899, 0.1928640, 1.1631093])
        # reconstruct electromagnetic energy for the event
        eem_pred, eem_err = energyreconstruction.get_Eem_from_Egeo(*rec_data, *rec_fit_values, egeo_err=egeo_fit_err, rho_avg=avg_rho)

        redchi = helpers.convert_fit_result(result)['redchi']
        # get reconstructed zenith angle for title
        zenith_mc = shower.get_parameter(shp.zenith_recon)
        azimuth_mc = shower.get_parameter(shp.azimuth_recon)
        # energy_em = shower.get_parameter(shp.electromagnetic_energy)
        # plot title
        # title = r"$E_\mathrm{em}^\mathrm{MC}$ = %.2f EeV, $E_\mathrm{em}^\mathrm{rec}$ = %.2f $\pm$ %.2f EeV, $d_\mathrm{max}^\mathrm{fit}$ = %.0f $\pm$ %.0f km" \
        #         "\n" \
        #         r"$\theta$ = %.1f$^\circ$, $\phi$ = %.1f$^\circ$, $\alpha$ = %.1f$^\circ$, $\chi^2/$ndf = %.2f" % (
        #     energy_em / (10 ** 18),
        #     eem_pred / (10 ** 18), eem_err / (10 ** 18),
        #     dmax_fit / 1e3, dmax_fit_err / 1e3, # in km
        #     np.rad2deg(zenith_mc),
        #     np.rad2deg(azimuth_mc),
        #     np.rad2deg(alpha),
        #     redchi
        # )
                
        # title = r"$E_\mathrm{em}^\mathrm{MC}$ = %.2f EeV, $E_\mathrm{em}^\mathrm{rec}$ = %.2f $\pm$ %.2f EeV," \
        #         "\n" \
        #         r"$\theta$ = %.1f $\pm$ %.1f$^\circ$, $\phi$ = %.1f $\pm$ %.1f$^\circ$, $\chi^2/$ndf = %.2f" % (
        #     energy_em / (10 ** 18),
        #     eem_pred / (10 ** 18), eem_err / (10 ** 18),
        #     np.rad2deg(zenith_mc), shower.get_parameter_error(shp.zenith_recon),
        #     np.rad2deg(azimuth_mc), shower.get_parameter_error(shp.azimuth_recon),
        #     redchi
        # )

        # print(dmax_fit, dmax_fit_err)

        run_no = revent.get_run_number()
        evt_no = revent.get_id()

        # title = r"(Run %.0f , Event %.0f): $E_\mathrm{em}^\mathrm{rec}$ = %.2f $\pm$ %.2f EeV, $d_\mathrm{max}^\mathrm{fit}$ = %.0f $\pm$ %.0f km" \
        #         "\n" \
        #         r"$\theta$ = %.1f$^\circ$, $\phi$ = %.1f$^\circ$, $\alpha$ = %.1f$^\circ$, $\chi^2/$ndf = %.2f" % (
        #     run_no, evt_no,
        #     eem_pred / (10 ** 18), eem_err / (10 ** 18),
        #     dmax_fit / 1e3, dmax_fit_err / 1e3, # in km
        #     np.rad2deg(zenith_mc),
        #     np.rad2deg(azimuth_mc),
        #     np.rad2deg(alpha),
        #     redchi
        # )

        print(run_no, evt_no)
        # get CR id from user input...
        id = "0" # str(input("Candidate name? \n "))

        # title for ICRC version
        title = id + r": $E_\mathrm{em}^\mathrm{rec}$ = %.2f $\pm$ %.2f EeV, $d_\mathrm{max}^\mathrm{fit}$ = %.0f $\pm$ %.0f km, $\chi^2/$ndf = %.2f," \
                 "\n" \
                 r"$\theta^\mathrm{rec}$ = %.1f $\pm$ %.1f$^\circ$, $\phi^\mathrm{rec}$ = %.1f $\pm$ %.1f$^\circ$, $\alpha^\mathrm{rec}=$ %.1f$^\circ$" % (
            #run_no, evt_no, 
            eem_pred / (10 ** 18), eem_err / (10 ** 18),
            dmax_fit / 1e3, dmax_fit_err / 1e3, # in km
            redchi,
            np.rad2deg(zenith_mc), np.rad2deg(shower.get_parameter_error(shp.zenith_recon)),
            np.rad2deg(azimuth_mc), np.rad2deg(shower.get_parameter_error(shp.azimuth_recon)),
            np.rad2deg(alpha), 
        )



    else:
        # get density at xmax
        rho_max = shower.get_parameter(shp.density_at_shower_maximum)
        avg_rho = 0.2171571758825658 # mean value for GP300 simulations
        # get radiation energy with error
        egeo_fit = result.params['E_geo'].value
        egeo_fit_err = result.params['E_geo'].stderr
        dmax_fit = result.params['distance_xmax_geometric'].value
        dmax_fit_err = result.params['distance_xmax_geometric'].stderr
        if egeo_fit_err == None: egeo_fit_err = 0
        if dmax_fit_err == None: dmax_fit_err = 0
        rec_data = np.array([egeo_fit, np.sin(alpha), rho_max])

        # fit values for density correction and conversion from Sgeo to Eem
        rec_fit_values = np.array([13.4858593e9, 1.9961499, 196.5323011, -0.0030236, 0.0338899, 0.1928640, 1.1631093])
        # reconstruct electromagnetic energy for the event
        eem_pred, eem_err = energyreconstruction.get_Eem_from_Egeo(*rec_data, *rec_fit_values, egeo_err=egeo_fit_err, rho_avg=avg_rho)
        
        run_no = revent.get_run_number()
        evt_no = revent.get_id()
        
        redchi = helpers.convert_fit_result(result)['redchi']

        energy_em = shower.get_parameter(shp.electromagnetic_energy)
        zenith_mc = shower.get_parameter(shp.zenith)
        azimuth_mc = shower.get_parameter(shp.azimuth)
        title = r"$E_\mathrm{em}^\mathrm{MC}$ = %.2f, $E_\mathrm{em}^\mathrm{rec}$ = %.2f $\pm$ %.2f EeV," \
            "\n" \
            r"$\theta^\mathrm{MC}$ = %.2f$^\circ$, $\phi^\mathrm{MC}$ = %.2f$^\circ$, $\chi^2/$ndf = %.2f" % (
            energy_em / (10 ** 18),
            eem_pred / (10 ** 18), eem_err / (10 ** 18),
            np.rad2deg(zenith_mc),
            np.rad2deg(azimuth_mc),
            redchi
        )
        title = r"$E_\mathrm{em}^\mathrm{MC}$ = %.2f, $E_\mathrm{em}^\mathrm{rec}$ = %.2f $\pm$ %.2f EeV, $\theta^\mathrm{MC}$ = %.2f$^\circ$, $\phi^\mathrm{MC}$ = %.2f$^\circ$, $\chi^2/$ndf = %.2f" % (
            energy_em / (10 ** 18),
            eem_pred / (10 ** 18), eem_err / (10 ** 18),
            np.rad2deg(zenith_mc),
            np.rad2deg(azimuth_mc),
            redchi
        )
        
    # print(energy_em / (10 ** 18))
    # print(np.rad2deg(alpha))
    
    if para.plot:
        print(lmfit.fit_report(result))
        pars = result.params.valuesdict()
        plot_label = f"run{run_no}_event{evt_no}"

        plot_ldf(
            pars,
            fcn_kwargs,
            pos=False,
            opt="save",
            label="_%06d%s" % (revent.get_run_number(), para.label),
            title=title,
            plot_label=plot_label,
            revent=revent,
        )

    post_process_store(revent, result, np.full_like(f_vxB[fit_mask], True), para)


@ray.remote
def fit_param_has_ldf_ray(revent, para):
    _fit_param_has_ldf(revent, para)
    return revent


def fit_param_has_ldf(events, para):

    for revent in events:
        _fit_param_has_ldf(revent, para)


def post_process_store(revent, result, select_or_cleaned, para):

    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()

        shower = revent.get_shower(key=shower_type)

    if para.realistic_input:
        zenith, zenith_error = shower.get_parameter_and_error(shp.zenith_recon)
        azimuth, azimuth_error = shower.get_parameter_and_error(shp.azimuth_recon)
        alpha = shower.get_parameter(shp.geomagnetic_angle_recon)
        dir_err = shower.get_parameter(shp.pointing_error)

        energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)
        core_gp = shower.get_parameter(shp.core_estimate)

        station_positions = revent.get_station_parameter(stp.position)
        antenna_positions_core_cs = station_positions - core_gp
        cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=revent.get_parameter(evp.magnetic_field_vector))

    else:
        zenith = shower.get_parameter(shp.zenith)
        azimuth = shower.get_parameter(shp.azimuth)

        energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)
        core_gp = shower.get_parameter(shp.core)

        station_positions = revent.get_station_parameter(stp.position)
        antenna_positions_core_cs = station_positions - core_gp
        cs = revent.get_coordinate_transformation()

    ### store results ###
    # fit statistics: chi2, ndf, ndata, .... Best fitting parameter + errors
    if isinstance(result, MyMinuitMinimizer):
        if not result.valid:
            print("Result not valid %d" % revent.get_run_number())
        redchi = result.fval / (np.sum(select_or_cleaned) - result.nfit)
        if 0:
            print("n_free", (np.sum(select_or_cleaned) - result.nfit))
            print("chisqr", result.fval)
            print("redchi", redchi)
            # print(result.fmin)

        shower.set_parameter(
            shp.fit_result,
            {
                "chi2": result.fval,
                "nfit": result.nfit,
                "valid": result.valid,
                "ndata": np.sum(select_or_cleaned),
                "redchi": redchi,
            },
        )
        bf_params = result.get_all_param_value_dict()
        bf_params_err = result.get_all_param_error_dict()
    else:
        if 0:
            for key in ["nfree", "chisqr", "redchi"]:
                print(key, result.__dict__[key])
        bf_params = result.params.valuesdict()
        bf_params_err = {key: result.params[key].stderr for key in bf_params}

        try:
            non_valid_error = np.any(
                np.isnan([bf_params_err[key] for key in bf_params_err])
            )
        except TypeError:  # if for exampe something is None
            non_valid_error = True

        if non_valid_error:
            result.__dict__["success"] = False

        shower.set_parameter(
            shp.fit_result, helpers.convert_fit_result(result)
        )  # convert fit_result to dict
    # get fitted core, update core
    core_fit_ground_plane = np.array(
        [bf_params.pop("core_x", 0), bf_params.pop("core_y", 0), 0]
    )
    core_fit_ground_plane_err = np.array(
        [bf_params_err.pop("core_x", 0), bf_params_err.pop("core_y", 0), 0]
    )

    # temporarily transform to ground plane system
    shower.set_parameter(shp.core_fit, core_fit_ground_plane)

    core_pred = shower.get_parameter(shp.prediceted_core_shift)

    shower.set_parameter(shp.core_pred_shower_plane, cs.transform_to_vxB_vxvxB(np.array([core_pred[0], core_pred[1], 0])))
        
    shower.set_parameter(shp.core_fit_shower_plane, 
                            cs.transform_to_vxB_vxvxB(np.array([core_fit_ground_plane[0], core_fit_ground_plane[1], 0])))

    if 1:

        run_no = revent.get_run_number()
        evt_no = revent.get_id()

        core_bary = shower.get_parameter(shp.core_estimate)
        core_fit = shower.get_parameter(shp.core_estimate) + core_fit_ground_plane

        # reconstructed energy

        # get density at xmax
        rho_max = shower.get_parameter(shp.density_at_shower_maximum)
        avg_rho = 0.2171571758825658 # mean value for GP300 simulations
        # get radiation energy with error
        egeo_fit = result.params['E_geo'].value
        egeo_fit_err = result.params['E_geo'].stderr
        dmax_fit = result.params['distance_xmax_geometric'].value
        dmax_fit_err = result.params['distance_xmax_geometric'].stderr
        if egeo_fit_err == None: egeo_fit_err = 0
        if dmax_fit_err == None: dmax_fit_err = 0
        rec_data = np.array([egeo_fit, np.sin(alpha), rho_max])

        # fit values for density correction and conversion from Sgeo to Eem
        rec_fit_values = np.array([13.4858593e9, 1.9961499, 196.5323011, -0.0030236, 0.0338899, 0.1928640, 1.1631093])
        # reconstruct electromagnetic energy for the event
        eem_pred, eem_err = energyreconstruction.get_Eem_from_Egeo(*rec_data, *rec_fit_values, egeo_err=egeo_fit_err, rho_avg=avg_rho)

        print(f"Event and Run Number: {run_no}, {evt_no}")
        print(f"Reconstructed Electromagnetic Energy: {eem_pred}({eem_err})")
        print(f"Arrival Direction: {np.rad2deg(zenith)}, {np.rad2deg(azimuth)}")
        print("Est. Core: ", shower.get_parameter(shp.core_estimate))
        print("Fitted core: ", shower.get_parameter(shp.core_estimate) + core_fit_ground_plane)
        print("Fitted core error: ", core_fit_ground_plane_err)
        shower.set_parameter_error(shp.core_fit, core_fit_ground_plane_err)

        # redchi
        redchi = helpers.convert_fit_result(result)['redchi']

        if 1: #(np.rad2deg(dir_err) < 1) and (np.rad2deg(zenith) < 85) and (eem_err / eem_pred < 1) and (dmax_fit_err / dmax_fit < 1):
            # write data into file
            with open("event_candidate_params.csv", "a") as txt_file:
                    txt_file.write(str(int(run_no)) + " " + str(int(evt_no)) + " " + str(np.round(redchi, 2)) + " " \
                                + str(np.round(eem_pred, 2)) + " " + str(np.round(eem_err, 2)) + " " \
                                + str(np.round(np.rad2deg(zenith), 3)) + " " + str(np.round(np.rad2deg(zenith_error), 3)) + " " \
                                + str(np.round(np.rad2deg(azimuth), 3))+ " " + str(np.round(np.rad2deg(azimuth_error), 3)) + " " \
                                + str(np.round(np.rad2deg(dir_err), 3))+ " " \
                                + str(np.round(core_bary[0], 2)) + " " + str(np.round(core_bary[1], 2)) + " " \
                                + str(np.round(core_fit[0], 2)) + " " + str(np.round(core_fit[1], 2)) + "\n")
                    
        else: 
            with open("event_candidate_params.csv", "a") as txt_file:
                        txt_file.write(str(int(run_no)) + " " + str(int(evt_no)) + "\n")

    # bf_params.pop('average_dxmax', None)  # if in remove it
    # returns distance_xmax_geometric and removes it from dict
    d_xmax_geo_fit = bf_params.pop("distance_xmax_geometric")
    shower.set_parameter(shp.distance_to_shower_maximum_geometric_fit, d_xmax_geo_fit)
    shower.set_parameter(shp.estimated_distance_to_shower_maximum_geometric, bf_params['average_dxmax'])
    d_xmax_geo_fit_err = bf_params_err.pop("distance_xmax_geometric")
    shower.set_parameter_error(
        shp.distance_to_shower_maximum_geometric_fit, d_xmax_geo_fit_err
    )

    if "slope" in bf_params and "r0" not in bf_params:
        r0 = cherenkov_radius.get_cherenkov_radius_model_from_distance(
            zenith,
            d_xmax_geo_fit,
            shower.get_parameter(shp.observation_level),
            revent.get_parameter(evp.refractive_index_at_sea_level),
            shower.get_parameter(shp.atmosphere_model),
        )
        bf_params["r0"] = r0 * 0.91
        bf_params_err["r0"] = 0

    # for convinience store parameter for geomagnetic emission
    shower.set_parameter(shp.geomagnetic_ldf_parameter, bf_params)
    shower.set_parameter_error(shp.geomagnetic_ldf_parameter, bf_params_err)

    # antenna_positions_core_cs = in ground plane but core cs (z = 0)
    # core fit (explizit in transform_to_vxB_vxvxB) & unpack position
    core_in_ground_plane = core_fit_ground_plane

    x_vB, y_vB, z_vB = np.squeeze(
        np.split(
            cs.transform_to_vxB_vxvxB(
                antenna_positions_core_cs, core=core_in_ground_plane
            ).T,
            3,
        )
    )

    # update values
    c_early_late = early_late.early_late_correction_factor(z_vB, d_xmax_geo_fit)
    revent.set_station_parameter(stp.early_late_factor, c_early_late)

    if 0:  # core_x != 0 or core_y != 0:
        shower.set_parameter(
            shp.core, shower.get_parameter(shp.core) + core_in_ground_plane
        )

        # update values
        pos_cor = revent.get_station_position_vB_vvB()
        f_geo_pos, f_ce_pos = se.seperate_radio_emission_from_position(
            pos_cor,
            energy_fluence_vector,
            c_early_late,
            recover_vxB=False,
            set_vxB_to_value=-1,
            get_only_f_geo=False,
            fitted_core=True,
        )  # , revent=revent)

        revent.set_station_parameter(stp.geomagnetic_fluence_positional, f_geo_pos)
        revent.set_station_parameter(stp.charge_excess_fluence_positional, f_ce_pos)
