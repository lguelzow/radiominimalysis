import copy
import sys

import numpy as np
from numpy.linalg import norm as betrag
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
import matplotlib.ticker as tick    

from scipy.interpolate import griddata

from radiotools import coordinatesystems as cs
from radiotools import helper as hp

from radiominimalysis.modules.reconstruction import signal_emissions as se
from radiominimalysis.modules.reconstruction.ldf_fit_functions import (
    ldf_geo_param,
    vary_core_and_get_early_late_factor,
)
from radiominimalysis.utilities import geomagnetic_emission, ldfs, early_late

from PWF_reconstruction.utils import create_times, chi2_PWF, mean

from radiominimalysis.framework.parameters import (
    eventParameters as evp,
    showerParameters as shp,
    stationParameters as stp,
)


def plot_f_gaus_sigmoid(
    axs, pars, rmodel, func, func_shape, label=("LDF fit"), color="k"
):
    pars.pop("distance_xmax_geometric", None)
    pars.pop("average_dxmax", None)
    pars.pop("core_x", None)
    pars.pop("core_y", None)

    dtot = func(rmodel, **pars)

    dummy0 = copy.deepcopy(pars)
    egeo = dummy0.pop("E_geo")
    norm = geomagnetic_emission.calculate_geomagnetic_energy(
        np.arange(pars["r0"] * 5), ldfs.ldf_gaus_sigmoid_shape, dummy0
    )

    dummy1 = copy.deepcopy(dummy0)
    dummy1["a1"] = 0
    d1 = egeo * func_shape(rmodel, **dummy1) / norm

    dummy2 = copy.deepcopy(dummy0)
    dummy2["arel"] = 0
    d2 = egeo * func_shape(rmodel, **dummy2) / norm

    cherenkov_radius = rmodel[np.argmax(dtot)]
    
    if 0:
        
        # np.savez_compressed(f"/cr/users/guelzow/simulations/radiominimalysis/ldf_eval/50_200_ldf", rmodel, dtot)
        low_ldf_file = np.load('/cr/users/guelzow/simulations/radiominimalysis/ldf_eval/30_80_ldf.npz')
        r_data = low_ldf_file['arr_0']
        f_data = low_ldf_file['arr_1']
        

    for a in axs:
        a.plot(rmodel, dtot, "-", lw=3, zorder=10, color=color, label=label)
        # a.plot(rmodel, d2, "-.", lw=3, zorder=10, color=color, label="Gaussian")
        # a.plot(rmodel, d1, "--", lw=3, zorder=10, color=color, label="Sigmoid")
        # plot area under the curve
        a.fill_between(rmodel, dtot, np.zeros(len(dtot)), color="dodgerblue", alpha=0.4, label=r"Energy in radio")
        
        # a.plot(r_data, f_data / 1000, "-", lw=5, zorder=10, color="red", label=label)
        # a.plot(rmodel, d2, "-.", lw=3, zorder=10, color="red", label="Gaussian")
        # a.plot(rmodel, d1, "--", lw=3, zorder=10, color="red", label="Sigmoid")
        # a.fill_between(r_data, f_data / 1000, np.zeros(len(dtot)), color="red", alpha=0.3, label=r"$E_\mathrm{geo}^\mathrm{fit}$")

    return cherenkov_radius


def plot_f_ldf(
    axs, pars, rmodel, f_geo_ldf, label=r"poly 3$^\mathrm{rd}$ (old)", color="dodgerblue"
):
    pars.pop("density_at_xmax", None)  # here okay
    pars.pop("average_density", None)  # here okay
    pars.pop("distance_xmax_geometric", None)  # here okay
    pars.pop("average_dxmax", None)  # here okay
    pars.pop("core_x", None)  # here okay
    pars.pop("core_y", None)  # here okay

    for a in axs:
        a.plot(
            rmodel,
            f_geo_ldf(rmodel, **pars),
            "-",
            color="C1",
            label=r"poly 3$^\mathrm{rd}$ (old)",
        )


# def plot_ldf(pars, energy_fluence_vector, antenna_positions_core_cs, alpha, zenith, cs, select_or_cleaned, pos=False):
def plot_ldf(
    pars,
    kwargs,
    pos=False,
    opt="show",
    label="",
    axs=None,
    plot_gaus_kwargs={},
    title=None,
    plot_label=None,
    revent=None,
):
    # save core position
    core_backup = np.array([pars['core_x'], pars['core_y']])

    fit_parameters = pars

    plt.rcParams['font.size'] = 30

    pos = False

    add_res = False

    xlim = 1
    
    # fig = plt.figure(figsize=(14, 10))
    #     # fig.subplots_adjust(hspace=0, wspace=0.3, left=0.1, right=0.95)
    #     if add_res:
    #         ax = fig.add_axes((0.16, 0.3, 0.83, 0.64))
    #         ax_inner = fig.add_axes((0.6, 0.64, 0.38, 0.29))
    #         ax_res = fig.add_axes((0.16, 0.1, 0.83, 0.19))

    if axs is None:
        # fig = plt.figure(figsize=(12, 9))
        # if add_res:
        #     ax = fig.add_axes((0.13, 0.3, 0.86, 0.64))
        #     ax_inner = fig.add_axes((0.6, 0.64, 0.38, 0.29))
        #     ax_res = fig.add_axes((0.13, 0.1, 0.86, 0.19))
        fig = plt.figure(figsize=(9, 10))
        if add_res:
            ax = fig.add_axes((0.16, 0.3, 0.83, 0.6))
            ax_inner = fig.add_axes((0.6, 0.64, 0.38, 0.29))
            ax_res = fig.add_axes((0.16, 0.1, 0.83, 0.19))
        else:
            ax = fig.add_axes((0.20, 0.13, 0.78, 0.86))
            ax_inner = fig.add_axes((0.5, 0.45, 0.43, 0.5))
    else:
        ax, ax_inner = axs

    xdata = kwargs["xdata"]
    if pos:
        antenna_positions_core_cs, energy_fluence_vector, cs = xdata
        zenith = kwargs.pop("zenith", None)
        alpha = kwargs.pop("alpha", None)
    else:
        # read data for plotting
        Auger_param = kwargs["Auger_param"]
        
        # core start value
        core = kwargs.pop("core_fit", None)

        antenna_positions_core_cs_all, alpha, zenith, cs = xdata
        f_vxB_all = kwargs.pop("f_vxB", None)
        f_vxB_error_all = kwargs.pop("fluence_error", None)

        antenna_IDs = kwargs.pop("IDs", None)
        # print(antenna_IDs)
        # mask for cut stations
        fit_mask = kwargs.pop("fit_mask", None)

        # all stations considered in the fit
        f_vxB = f_vxB_all[fit_mask]
        f_vxB_errors = f_vxB_error_all[fit_mask]
        antenna_positions_core_cs = antenna_positions_core_cs_all[fit_mask]
        
        # antennas that were cut in ADC noise
        f_vxB_noisy, antennas_noisy, errors_noisy = kwargs["noisy_data"]
        
        # antennas that were cut in SNR cut
        f_vxB_bad_timing, antennas_bad_timing, errors_bad_timing = kwargs["bad_timing_data"]

        # antennas that were cut in SNR cut
        f_vxB_snr, antennas_snr, errors_snr = kwargs["snr_data"]

        # saturated antennas
        f_vxB_saturated, stations_saturated, errors_saturated = kwargs["saturated_data"]

        # unused at the moment
        # antennas eliminated by thinning
        rev_fit_mask = ~fit_mask
        f_vxB_thin = f_vxB_all[rev_fit_mask]
        stations_thin = antenna_positions_core_cs_all[rev_fit_mask]
        
        if revent:
            if revent.has_station_parameter(stp.times):
                antenna_times = revent.get_station_parameter(stp.times)[fit_mask]
                # print(len(antenna_times), len(antenna_positions_core_cs))

    observation_level = kwargs["observation_level"]
    n0 = kwargs["n0"]
    model = kwargs["atmodel"]

    # get position of the whole array 
    if 1:
        plot_array = True
        # read antenna positions from file
        # GP65_june2025 antennas
        file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/GP65_june25.txt", dtype = "str")
        # GP80 antennas
        file_gp80 = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/list_antennas_GP80_202412_202504_xyz_origin_DAQ.txt", dtype = "str")
        # GP300
        # file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/GP300_layout.txt", dtype = "str")
        # GRAND10k
        # file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/GRAND10k_layout.txt", dtype = "str")

        # put antenna positions in x,y,z in array
        # GP65_june2025
        array_positions = np.array([[file[i, 2].astype(float), (-1) * file[i, 1].astype(float), file[i, 3].astype(float)] for i in range(len(file))])
        # subtract position of old origin
        core_offset = np.array([21.73, -55.25, -0.38]) # -0.38
        array_positions = array_positions - core_offset # - np.array([3813.05629, 1457.20845, 0]
        # GP80
        array_positions_gp80 = np.array([[file_gp80[i, 1].astype(float), file_gp80[i, 2].astype(float), file_gp80[i, 3].astype(float)] for i in range(len(file_gp80))])
        array_positions_gp80 = array_positions_gp80 - core_offset
        # GP300 & 10k
        # array_positions = np.array([[file[i, 2].astype(float), file[i, 3].astype(float), file[i, 4].astype(float)] for i in range(len(file))])
        # array_positions2 = np.array([[file2[i, 2].astype(float), file2[i, 3].astype(float), file2[i, 4].astype(float)] for i in range(len(file2))])
        
        antenna_ids = np.array([file_gp80[i, 0].astype(float) for i in range(len(file_gp80))])
        plot_ids = antenna_ids
        # mask to only plot the untriggered antennas
        untriggered_mask = np.isin(antenna_ids, antenna_IDs)
        antenna_ids = antenna_ids[~untriggered_mask]

        array_positions_core_cs = array_positions - np.array([core[0], core[1], 0])
        array_positions_core_cs_gp80 = array_positions_gp80 - np.array([core[0], core[1], 0])

        # print(array_positions_core_cs)
        # print(antenna_positions_core_cs)

    # the actual LDF fit stations!
    (
        x_vB,
        y_vB,
        z_vB,
        c_early_late,
        distance_xmax_geometric_fit,
        params_dict,
    ) = vary_core_and_get_early_late_factor(
        fit_parameters,
        antenna_positions_core_cs,
        cs,
        n0=n0,
        observation_level=observation_level,
        atmodel=model,
        zenith=zenith,
        core=core_backup,
        plot_flag=True
    )
    
    # repeat for noisy stations if present
    if (f_vxB_noisy.any() != None):
        (
            x_vB_noisy,
            y_vB_noisy,
            z_vB_noisy,
            c_early_late_noisy,
            distance_xmax_geometric_fit,
            params_dict,
        ) = vary_core_and_get_early_late_factor(
            fit_parameters,
            antennas_noisy,
            cs,
            n0=n0,
            observation_level=observation_level,
            atmodel=model,
            zenith=zenith,
            core=core_backup,
            plot_flag=True
        )
        
    # repeat for bad timing stations if present
    if (f_vxB_bad_timing.any() != None):
        (
            x_vB_bad_timing,
            y_vB_bad_timing,
            z_vB_bad_timing,
            c_early_late_bad_timing,
            distance_xmax_geometric_fit,
            params_dict,
        ) = vary_core_and_get_early_late_factor(
            fit_parameters,
            antennas_bad_timing,
            cs,
            n0=n0,
            observation_level=observation_level,
            atmodel=model,
            zenith=zenith,
            core=core_backup,
            plot_flag=True
        )


    # repeat for snr stations if present
    if (f_vxB_snr.any() != None):
        (
            x_vB_snr,
            y_vB_snr,
            z_vB_snr,
            c_early_late_snr,
            distance_xmax_geometric_fit,
            params_dict,
        ) = vary_core_and_get_early_late_factor(
            fit_parameters,
            antennas_snr,
            cs,
            n0=n0,
            observation_level=observation_level,
            atmodel=model,
            zenith=zenith,
            core=core_backup,
            plot_flag=True
        )

    # repeat for saturated stations
    if (f_vxB_saturated.any() != None):
        (
            x_vB_saturated,
            y_vB_saturated,
            z_vB_saturated,
            c_early_late_saturated,
            distance_xmax_geometric_fit,
            params_dict,
        ) = vary_core_and_get_early_late_factor(
            fit_parameters,
            stations_saturated,
            cs,
            n0=n0,
            observation_level=observation_level,
            atmodel=model,
            zenith=zenith,
            core=core_backup,
            plot_flag=True
        )

    if plot_array:
        (
            x_vB_array,
            y_vB_array,
            z_vB_array,
            c_early_late_array,
            distance_xmax_geometric_fit,
            params_dict,
        ) = vary_core_and_get_early_late_factor(
            fit_parameters,
            array_positions_core_cs,
            cs,
            n0=n0,
            observation_level=observation_level,
            atmodel=model,
            zenith=zenith,
            core=core_backup,
            plot_flag=True
        )
        
        # if also plotting the deployed layout for the measurements
        if 1:
            (
                x_vB_gp80,
                y_vB_gp80,
                z_vB_gp80,
                c_early_late_gp80,
                distance_xmax_geometric_fit,
                params_dict,
            ) = vary_core_and_get_early_late_factor(
                fit_parameters,
                array_positions_core_cs_gp80,
                cs,
                n0=n0,
                observation_level=observation_level,
                atmodel=model,
                zenith=zenith,
                core=core_backup,
                plot_flag=True
            )
         
        # print(x_vB, y_vB, z_vB, c_early_late)
        # print(x_vB_array, y_vB_array, z_vB_array, c_early_late_array)
    

    if pos:
        station_position_vBvvB = np.array([x_vB, y_vB, z_vB]).T
        f_geo_pos, f_ce_pos = se.seperate_radio_emission_from_position(
            station_position_vBvvB,
            energy_fluence_vector,
            c_early_late,
            recover_vxB=False,
            set_vxB_to_value=-1,
            get_only_f_geo=False,
            fitted_core=True,
        )

        fgeo = f_geo_pos * c_early_late ** 2
        fgeo_weight = fgeo * kwargs["rel_weight"] + kwargs["add_abs_weight"] * np.amax(
            fgeo
        )
    else:
        if "ce0" in params_dict:
            ce_param = {key: params_dict.pop(key, 0) for key in ["ce0", "ce1", "ce2"]}
        else:
            ce_param = None


        # LDF stations
        # already el corrected
        fgeo = ldf_geo_param(
            x_vB,
            y_vB,
            c_early_late,
            f_vxB,
            distance_xmax_geometric_fit,
            alpha,
            zenith,
            observation_level=observation_level,
            atmodel=model,
            ce_fraction_param=ce_param, Auger_param=Auger_param
        )

        if 0: # kwargs["rel_weight"] and kwargs["add_abs_weight"]:
            # use old errors
            fgeo_weight = fgeo * kwargs["rel_weight"] + kwargs["add_abs_weight"] * np.amax(fgeo)

        # also convert fluence errors
        fgeo_weight = ldf_geo_param(
            x_vB,
            y_vB,
            c_early_late,
            f_vxB_errors,
            distance_xmax_geometric_fit,
            alpha,
            zenith,
            observation_level=observation_level,
            atmodel=model,
            ce_fraction_param=ce_param, Auger_param=Auger_param
        )
        
        r = np.sqrt(x_vB ** 2 + y_vB ** 2) / c_early_late



        # repeat for noisy stations
        if (f_vxB_noisy.any() != None):
        # already el corrected
            fgeo_noisy = ldf_geo_param(
                x_vB_noisy,
                y_vB_noisy,
                c_early_late_noisy,
                f_vxB_noisy,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )

            # also convert fluence errors
            fgeo_weight_noisy = ldf_geo_param(
                x_vB_noisy,
                y_vB_noisy,
                c_early_late_noisy,
                errors_noisy,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )
        
            r_noisy = np.sqrt(x_vB_noisy ** 2 + y_vB_noisy ** 2) / c_early_late_noisy


        # repeat for bad timing stations
        if (f_vxB_bad_timing.any() != None):
        # already el corrected
            fgeo_bad_timing = ldf_geo_param(
                x_vB_bad_timing,
                y_vB_bad_timing,
                c_early_late_bad_timing,
                f_vxB_bad_timing,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )

            # also convert fluence errors
            fgeo_weight_bad_timing = ldf_geo_param(
                x_vB_bad_timing,
                y_vB_bad_timing,
                c_early_late_bad_timing,
                errors_bad_timing,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )
        
            r_bad_timing = np.sqrt(x_vB_bad_timing ** 2 + y_vB_bad_timing ** 2) / c_early_late_bad_timing



        # repeat for snr stations
        if (f_vxB_snr.any() != None):
        # already el corrected
            fgeo_snr = ldf_geo_param(
                x_vB_snr,
                y_vB_snr,
                c_early_late_snr,
                f_vxB_snr,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )

            # also convert fluence errors
            fgeo_weight_snr = ldf_geo_param(
                x_vB_snr,
                y_vB_snr,
                c_early_late_snr,
                errors_snr,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )
        
            r_snr = np.sqrt(x_vB_snr ** 2 + y_vB_snr ** 2) / c_early_late_snr

        # repeat for saturated stations
        if (f_vxB_saturated.any() != None):
            fgeo_saturated = ldf_geo_param(
                x_vB_saturated,
                y_vB_saturated,
                c_early_late_saturated,
                f_vxB_saturated,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )

            # also convert fluence errors
            fgeo_weight_saturated = ldf_geo_param(
                x_vB_saturated,
                y_vB_saturated,
                c_early_late_saturated,
                errors_saturated,
                distance_xmax_geometric_fit,
                alpha,
                zenith,
                observation_level=observation_level,
                atmodel=model,
                ce_fraction_param=ce_param, Auger_param=Auger_param
            )
            
            r_saturated = np.sqrt(x_vB_saturated ** 2 + y_vB_saturated ** 2) / c_early_late_saturated
        


    '''
    ##
    PLOTTING
    ##
    '''



    if axs is None:  # avoid drawing data twice
        for a in [ax, ax_inner]:
            neg_mask = fgeo > 0
            a.errorbar(
                r[neg_mask],
                fgeo[neg_mask],
                fgeo_weight[neg_mask],
                marker="o",
                ls="",
                color="dodgerblue", # dodgerblue
                markersize=12,
                label=r"$f_{\,\mathrm{geo}}^{\,\mathrm{par}}$"
            )
            
            # np.savez_compressed(f"/cr/users/guelzow/simulations/radiominimalysis/ldf_eval/50_200_ldf", r[neg_mask], fgeo[neg_mask])

            # repeat for cut stations
            if 0: #(f_vxB_noisy.any() != None):
                a.errorbar(
                    r_noisy,
                    fgeo_noisy,
                    np.abs(fgeo_weight_noisy),
                    marker="x",
                    ls="",
                    color="black",
                    markersize=12,
                    alpha=0.5,
                    label=r"ADC noise cut"
                )

            # repeat for cut stations
            if 0: # (f_vxB_bad_timing.any() != None):
                a.errorbar(
                    r_bad_timing,
                    fgeo_bad_timing,
                    np.abs(fgeo_weight_bad_timing),
                    marker="s",
                    ls="",
                    color="green",
                    markersize=12,
                    alpha=0.5,
                    label=r"Peak timing cut"
                )
                
            # repeat for cut stations
            if (f_vxB_snr.any() != None):
                a.errorbar(
                    r_snr,
                    fgeo_snr,
                    np.abs(fgeo_weight_snr),
                    marker="^",
                    ls="",
                    color="navy",
                    markersize=12,
                    alpha=0.5,
                    label=r"SNR cut"
                )

            # repeat for saturated stations
            if (f_vxB_saturated.any() != None):
                a.errorbar(
                    r_saturated,
                    fgeo_saturated,
                    np.abs(fgeo_weight_saturated),
                    marker="v",
                    ls="",
                    color="red",
                    markersize=12,
                    alpha=0.5,
                    label=r"Saturated DUs"
                )


    rmodel = np.linspace(0, r.max() * 2.1, 600)
    func_geo = kwargs["f_geo_ldf"]

    """if func_geo == ldfs.f_E_geo:
        for a in [ax, ax_inner]:
            a.plot(rmodel, func_geo(rmodel, **params_dict), 'C1-', lw=3, zorder=10,
                   label=r"poly 3$^\mathrm{rd}$ (old)")"""

    if func_geo == ldfs.f_E_geo_gaus_sigmoid_p_slope:
        plot_f_gaus_sigmoid(
            [ax, ax_inner],
            params_dict,
            rmodel,
            func_geo,
            ldfs.ldf_gaus_sigmoid_shape,
            **plot_gaus_kwargs
        )

    elif func_geo == ldfs.f_E_geo_gaus_sigmoid_simple_p:
        plot_f_gaus_sigmoid(
            [ax, ax_inner],
            params_dict,
            rmodel,
            func_geo,
            ldfs.ldf_gaus_sigmoid_shape_simple_p,
            **plot_gaus_kwargs
        )

    else:
        sys.exit("Invalid f_geo_ldf")
        
    # title = "GP300"
    
    font=30    
    
    # ax.set_title(title, fontsize=font-5)
    # ax.set_ylabel(r"Signal intensity [a.u.]")
    ax.set_ylabel(r"Energy fluence [$\mathrm{eV}\,\mathrm{m}^{-2}$]", fontsize=font)
    
    f_ge0_model = func_geo(r[neg_mask], **params_dict)

    if add_res:
        neg_mask = fgeo > 0
        
        ax_res.errorbar( # plot( 
            r[neg_mask],
            fgeo[neg_mask] / f_ge0_model, # (fgeo[neg_mask] - f_ge0_model) / fgeo_weight[neg_mask],
            fgeo_weight[neg_mask] / f_ge0_model,
            marker="o",
            ls="",
            color="dodgerblue",
            markersize=12,
        )

        bins=np.linspace(-3.5, 3.5, 10)
        # ax_hist = ax_res.twiny()
        # ax_hist.hist((fgeo[neg_mask] - f_ge0_model) / fgeo_weight[neg_mask], bins=bins, histtype='stepfilled', orientation='horizontal', linewidth=2, color='blue', alpha=0.2)
        # ax_hist.set_xticklabels([])

        ax_res.set_xlabel("Axis distance [m]", fontsize=font)
        # ax_res.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])
        # ax_res.set_yticks([-3, -2, -1, 0, 1, 2, 3], fontsize=15)
        ax_res.set_xlim(ax.get_xlim())
        ax_res.set_ylim(0.65, 1.35)
        ax_res.set_yticks([0.7, 0.85, 1, 1.15, 1.3])
        ax_res.tick_params(axis="y", labelsize=font-5)
        ax.set_xticklabels([])
        from matplotlib.ticker import FixedLocator

        ax_res.yaxis.set_minor_locator(FixedLocator([0.95, 1.05]))
        ax_res.grid(axis="y")
        ax_res.grid(axis="x")
        # ax_res.grid(axis="y", which="minor", ls="--")
        ax_res.set_ylabel(r"$f_{\,\mathrm{geo}}^{\,\mathrm{par}} / f_\mathrm{LDF}$", fontsize=font)
        # ax_res.yaxis.set_label_coords(-0.09, 0.5)
        ax.yaxis.set_label_coords(-0.09, 0.5)
        ax_res.set_xlim(-5, None)#1800)
        ax_res.set_xlim(-5, 1700)
    else:
        ax.set_xlabel(r"Axis distance $r$ [m]", fontsize=font)

    ax_inner.set_yscale("log")
    f_max =  max(np.concatenate([fgeo, np.array(f_ge0_model)])) * 1.2 # max(fgeo) * 1.2 #
    
    ax_inner.remove()
    ax.grid(axis="x")
    ax.set_xlim(-5, 750) # (-5, max(r) * 0.4)#1800) # rmodel.max() * 0.25)
    
    # ax.set_ylim(-0.01 * f_max, 1020)#f_max)
    ax.set_ylim(0, 2700)#f_max)
    
    # 10k 
    # ax.set_xlim(-5, 1700)
    
    # ax.set_ylim(-0.01 * f_max, 275)#f_max)

    
    # # 10k 
    # ax.set_xlim(-5, 5600)
    # ax_res.set_xlim(-5, 5600)
    # ax.set_ylim(-0.01 * f_max, 65000)#f_max)
    
    # labels_y = ["", 0, 10, 20, 30, 40, 50, 60]
    # ax.set_yticklabels(labels_y)
    
    # starshape LDFs 
    # ax.set_xlim(-50, 1300)
    # ax_res.set_xlim(-50, 1300)
    # ax.set_ylim(-50, 2500)#f_max)
    

    # ax.legend(loc="upper right", fontsize=font-5)

    # if (f_vxB_noisy.any() != None):
    #     # print(fgeo_cut)
    #     ax_inner.set_ylim(0.05 * np.abs(fgeo_noisy).min(), 2 * fgeo.max())
    ax_inner.set_ylim(0.05 * fgeo[fgeo > 0].min(), 2 * fgeo.max())
    ax_inner.set_xlim(-50, rmodel.max() * 0.5)

    if title is None:
        title = r"$\theta$ = %.2f$^\circ$, %s" % (
            np.rad2deg(zenith),
            label.replace("_", " "),
        )

    if opt == "show":
        # ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    elif opt == "save":
        # ax.legend()
        plt.tight_layout()
        # plt.savefig(f"ldf_{plot_label}.png")
        print("Save LDF plot in pdf file")
        plt.savefig("ldf%s.png" % label)
        plt.close()
    elif axs is None:
        return fig, [ax, ax_inner]
    else:
        return [ax, ax_inner]
    


    # set up interpolation for fluence map
    # resolution of interpolation grid
    lattice_const = 600j

    # meshgrid within the interpolation limit with fineness of lattice constant/resolution
    XI, YI = np.mgrid[-np.max(x_vB):np.max(x_vB):lattice_const, \
                      -np.max(x_vB):np.max(x_vB):lattice_const]
    
    # colourmap name
    cmap = ['seismic', cmr.freeze, 'gnuplot2_r', 'hot','bone','plasma','PuRd','magma','brg']
    # cmap = cmap[2]

    # interpolation algorithms
    interp = ['nearest', 'bicubic', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    
    # add additional residual 2D plot
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(21, 6))

    # change antenna positions in the first plot in the used version
    axs[0].plot(x_vB / c_early_late, y_vB / c_early_late, "o", markersize=3, c='black', alpha=1)
    axs[1].plot(x_vB / c_early_late, y_vB / c_early_late, "o", markersize=3, c='black', alpha=1)

    # set axis parameters
    [ax.set_xlabel(r"vxB [m]") for ax in axs]
    [ax.set_ylabel(r"vx(vxB) [m]") for ax in axs]
    [ax.set_xlim(-np.max(x_vB) * 3, np.max(x_vB) * 3) for ax in axs]
    [ax.set_ylim(-np.max(x_vB) * 3, np.max(x_vB) * 3) for ax in axs]
    [ax.set_aspect('equal') for ax in axs]

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = [make_axes_locatable(ax) for ax in axs]
    cax = [divider[i].append_axes("right", size="5%", pad=0.05) for i in range(len(axs))]

    # to display core position in shower plane, define shower plane coordinate system
    # define coordinate system transformations
    core_vxB = cs.transform_to_vxB_vxvxB(np.array([0, 0, 0]))
    # in both subplots
    [ax.scatter(core_vxB[0], core_vxB[1], marker='*', s=100, c='black') for ax in axs]


    # data grid points and values for total fluence
    (x_data, y_data, values) = ((x_vB / c_early_late)[neg_mask], (y_vB / c_early_late)[neg_mask], fgeo[neg_mask])

    # interpolated grid
    resampled = griddata((x_data,y_data), values, (XI,YI), method='cubic')

    # generate and assign interpolation data to the interpolated grid
    fluence_map = axs[0].imshow(resampled.T, origin='lower', cmap=cmap[2], interpolation=interp[0], alpha=1, \
                             extent=[-np.max(x_vB), np.max(x_vB), -np.max(x_vB), np.max(x_vB)], vmin=0)
    
    # assign colorbar to this subplot
    cbar = fig.colorbar(fluence_map, cax=cax[0], format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label(r"Total fluence (symmetrised)")

    
    # second subplot
    # data grid points and values for fluence residuals 
    (x_data, y_data, values) = ((x_vB / c_early_late)[neg_mask], (y_vB / c_early_late)[neg_mask], fgeo[neg_mask] / f_ge0_model)

    # interpolated grid
    resampled = griddata((x_data,y_data), values, (XI,YI), method='cubic')

    # generate and assign interpolation data to the interpolated grid
    residual_map = axs[1].imshow(resampled.T, origin='lower', cmap=cmap[0], interpolation=interp[0], alpha=1, \
                             extent=[-np.max(x_vB), np.max(x_vB), -np.max(x_vB), np.max(x_vB)], vmin=0.88, vmax=1.12)

    # assign colorbar to this subplot
    cbar = fig.colorbar(residual_map, cax=cax[1], format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label(r"Residuals of LDF fit")

    plt.tight_layout()
    # plt.show()
    # plt.savefig("core_2D.png")
    plt.close()


    # 2 or 3 panel LDF plots
    if 1: 
        only_ground_plane = True

        # plot layouts for 3 or 2 plot panels
        if only_ground_plane:
            fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 10))
        else:
            fig, axs = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(24, 10))
    

        fig.suptitle(title, fontsize=30-5)

        # meshgrid of area of footprint
        XI, YI = np.mgrid[-max(rmodel) * 1.5:max(rmodel) * 1.5:lattice_const, \
                          -max(rmodel) * 1.5:max(rmodel) * 1.5:lattice_const]

        testx, testy = np.mgrid[-1:1:3j, -1:1:3j]

        # print(testx)
        # print(testy)

        grid_coord_test = np.array([[testx.flatten()[i], testy.flatten()[i], 0] for i in range(len(testx.flatten()))])
        # print(grid_coord_test)
        
        
        # 3D coordinates of the meshgrid
        # grid_coord = np.array([[XI.flatten()[i], YI.flatten()[i], 0] for i in range(len(YI.flatten()))])

        # transform the grid to ground plane
        # grid_ground = cs.transform_from_vxB_vxvxB_2D(grid_coord)

        # transform back to matrix format
        # XI_gp = np.array(np.array_split(grid_ground[:, 0], np.sqrt(len(grid_ground))))
        # YI_gp = np.array(np.array_split(grid_ground[:, 1], np.sqrt(len(grid_ground))))
        # print(XI.shape)
        
        # calculate predicted fluence of footprint for each cell
        dtot = func_geo(np.sqrt(XI ** 2 + YI ** 2), **params_dict)
        ldf = np.array(func_geo(rmodel, **params_dict))

        r_che = plot_f_gaus_sigmoid(
            [axs[0]],
            params_dict,
            rmodel,
            func_geo,
            ldfs.ldf_gaus_sigmoid_shape,
            **plot_gaus_kwargs
        )

        circle_angles = np.deg2rad(np.linspace(0, 360, 1000))
        cherenkov_ring = np.array([[r_che * np.cos(circle_angles[i]), r_che * np.sin(circle_angles[i]), 0] for i in range(len(circle_angles))])
        cherenkov_ring_double = np.array([[2 * r_che * np.cos(circle_angles[i]), 2 * r_che * np.sin(circle_angles[i]), 0] for i in range(len(circle_angles))])

        che_ground = cs.transform_from_vxB_vxvxB_2D(cherenkov_ring)
        che_ground_double = cs.transform_from_vxB_vxvxB_2D(cherenkov_ring_double)

        axs[0].errorbar(
                r[neg_mask],
                fgeo[neg_mask],
                fgeo_weight[neg_mask],
                marker="o",
                ls="",
                color="dodgerblue",
                markersize=12,
                label=r"$f_{\,\mathrm{geo}}^{\,\mathrm{par}}$ at each DU"
            )
        
        # repeat for cut stations
        if (f_vxB_noisy.any() != None):
            axs[0].errorbar(
                r_noisy,
                fgeo_noisy,
                np.abs(fgeo_weight_noisy),
                marker="x",
                ls="",
                color="black",
                markersize=12,
                alpha=0.5,
                label=r"ADC noise cut"
            )

        # repeat for cut stations
        if (f_vxB_bad_timing.any() != None):
            axs[0].errorbar(
                r_bad_timing,
                fgeo_bad_timing,
                np.abs(fgeo_weight_bad_timing),
                marker="s",
                ls="",
                color="green",
                markersize=12,
                alpha=0.5,
                label=r"Peak timing cut"
            )
            
        # repeat for cut stations
        if (f_vxB_snr.any() != None):
            axs[0].errorbar(
                r_snr,
                fgeo_snr,
                np.abs(fgeo_weight_snr),
                marker="^",
                ls="",
                color="navy",
                markersize=12,
                alpha=0.5,
                label=r"SNR cut"
            )

        # repeat for saturated stations
        if (f_vxB_saturated.any() != None):
            axs[0].errorbar(
                r_saturated,
                fgeo_saturated,
                np.abs(fgeo_weight_saturated),
                marker="v",
                ls="",
                color="red",
                markersize=12,
                alpha=0.5,
                label=r"Saturated DUs"
            )
        
        axs[0].set_title("LDF fit to DU measurements", fontsize=font-5)
        # axs[0].set_title(r"LDF fit to $f_\mathrm{geo}^{\,\mathrm{par}}$", fontsize=font-5)
        # axs[0].set_xlim(0, 5600) # 10k
        # axs[0].set_xlim(0, 1700) # GP300
        # axs[0].set_xlim(0, max(rmodel))
        axs[0].set_ylim(0, max(np.concatenate([fgeo, ldf])) * 1.2)
        axs[0].set_xlabel(r"Axis distance $r$ [m]", fontsize=font-5)
        axs[0].set_ylabel(r"Geomagnetic fluence [eV$\,$m$^{-2}$]", fontsize=font-5)
        axs[0].legend(loc="upper right", fontsize=15)
        axs[0].tick_params(axis="both", labelsize=font-5)
        # labels_y = ["", 0, 10, 20, 30, 40, 50, 60]
        # axs[0].set_yticklabels(labels_y)

        axs[0].grid(axis="x")


        # ground plane plot
        # print 2D footprint and antennas with fluence values to see if they match
        # plot all antennas of the array
        # x as y and y as  x because x points to north

        # transform antennas back to ground plane in proper coordinate system
        # gp_array = cs.transform_from_vxB_vxvxB(np.array([[x_vB_array[i], y_vB_array[i], z_vB_array[i]] for i in range(len(x_vB_array))]))
        # axs[1].scatter(-gp_array[:, 1], gp_array[:, 0], marker="+", s=50, color="red", alpha=1)

        # axs[1].plot(-che_ground[:, 1], che_ground[:, 0], c="black", linewidth=1.5, ls="--", label="Cherenkov radius")
        axs[1].plot(-che_ground_double[:, 1], che_ground_double[:, 0], c="black", linewidth=1.5, ls="--", label="High intensity region")

        # get axis limits for meshgrid
        xmin, xmax = axs[1].get_xlim() # -max(rmodel) * 1.5, max(rmodel) * 1.5 
        ymin, ymax = axs[1].get_ylim() # - max(rmodel) * 1.5, max(rmodel) * 1.5 

        # mesh grid lattice constant
        grid_size = 200j


        XI_gp, YI_gp = np.mgrid[min(xmin, ymin):max(xmax, ymax):grid_size, \
                                min(xmin, ymin):max(xmax, ymax):grid_size]

        # XI_gp, YI_gp = np.mgrid[-1:1:3j, \
        #                         -1:1:3j,]
        # print(XI_gp)
        # print(YI_gp)

        # get 2D coordinates from mesh
        grid_gp = np.array([[XI_gp.flatten()[i], YI_gp.flatten()[i], 0] for i in range(len(YI_gp.flatten()))])
        grid_sp = cs.transform_to_vxB_vxvxB(grid_gp)

        # calculate early-late correction factor for all mesh grid points
        c_early_late_gp = early_late.early_late_correction_factor(grid_sp[:, 2], distance_xmax_geometric_fit)

        # # transform back into matrix format
        # XI_sp= np.array(np.array_split(grid_gp[:, 0], np.sqrt(len(grid_gp))))
        # YI_sp = np.array(np.array_split(grid_gp[:, 1], np.sqrt(len(grid_gp))))

        # print(XI_sp)
        # print(YI_sp)

        # transform back into matrix format
        XI_sp= np.array(np.array_split(grid_sp[:, 0], np.sqrt(len(grid_sp))))
        YI_sp = np.array(np.array_split(grid_sp[:, 1], np.sqrt(len(grid_sp))))
        # also transform early-late factors accordingly
        c_early_late_matrix = np.array(np.array_split(c_early_late_gp, np.sqrt(len(grid_sp))))

        # calculate LDF value for meshgrid radii and apply early-late factors
        dtot_gp = func_geo(np.sqrt(XI_sp ** 2 + YI_sp ** 2) / c_early_late_matrix, **params_dict)

        # print 2D footprint and antennas with fluence values to see if they match
        pcm = axs[1].contourf(-YI_gp, XI_gp, dtot_gp, levels=1000, vmin=0, vmax=max(np.concatenate([fgeo, dtot_gp.flatten()])), cmap='cmr.freeze_r' # 'gnuplot2_r' \
                            , label="LDF fit footprint", alpha=1)

        # plot all antennas of the array
        # x as y and y as  x because x points to north

        # transform antennas back to ground plane in proper coordinate system
        gp_array = cs.transform_from_vxB_vxvxB(np.array([[x_vB_array[i], y_vB_array[i], z_vB_array[i]] for i in range(len(x_vB_array))]))
        # axs[1].scatter(-gp_array[:, 1], gp_array[:, 0], marker="+", s=15, color="red", alpha=1, label="GP300 DUs", edgecolors="black")
        axs[1].scatter(-gp_array[:, 1], gp_array[:, 0], marker="o", s=50, color="white", alpha=1, label="Antennas", linewidth=1.5, edgecolors="black")
        # axs[1].scatter(array_positions2[:, 0] - 2500, array_positions2[:, 1] - 35000, marker="o", s=3, color="red", alpha=1, label="GP300")
        # for i in range(len(antenna_ids)):
        #     if antenna_ids[i] == 1078 or antenna_ids[i] == 1032:
        #         axs[1].annotate(str(int(antenna_ids[i]))+" (noisy)", (-gp_array[i, 1] + 75, gp_array[i, 0]), fontsize=10, color="gray", annotation_clip=True)
        #     else:
        #         axs[1].annotate(int(antenna_ids[i]), (-gp_array[i, 1] + 75, gp_array[i, 0]), fontsize=10, color="gray", annotation_clip=False)

        if 0:
            gp80_array = cs.transform_from_vxB_vxvxB(np.array([[x_vB_gp80[i], y_vB_gp80[i], z_vB_gp80[i]] for i in range(len(x_vB_gp80))]))
            axs[1].scatter(-gp80_array[:, 1], gp80_array[:, 0], marker="o", edgecolors="black", s=40, color="white", alpha=1, label="Deployed at TOM")

        # station label
        label = ("DUs with Signal" + "\n" + "")

        # also transform the signal stations to ground plane properly
        gp_signal = cs.transform_from_vxB_vxvxB(np.array([[x_vB[i], y_vB[i], z_vB[i]] for i in range(len(x_vB))]))
        pcm = axs[1].scatter(-gp_signal[:, 1], gp_signal[:, 0],  #  label="DUs with signal", \
                       s=100, c=fgeo, alpha=1, vmin=0, vmax=max(np.concatenate([fgeo, dtot_gp.flatten()])), cmap='cmr.freeze_r', edgecolors='red', linewidth=1.5)
        
        # plot bad timing antennas in green
        if 0: # (f_vxB_bad_timing.any() != None):
            # also transform the signal stations to ground plane properly
            try:
                gp_peak_timing = cs.transform_from_vxB_vxvxB(np.array([[x_vB_bad_timing[i], y_vB_bad_timing[i], z_vB_bad_timing[i]] for i in range(len(x_vB_bad_timing))]))
                axs[1].scatter(-gp_peak_timing[:, 1], gp_peak_timing[:, 0], label="Bad timing DUs", \
                        marker="s", s=30, c="green", alpha=1, edgecolors='red', linewidth=1.5)
            except:
                gp_peak_timing = cs.transform_from_vxB_vxvxB(np.array([[x_vB_bad_timing, y_vB_bad_timing, z_vB_bad_timing]]))
                axs[1].scatter(-gp_peak_timing[1], gp_peak_timing[0], label="Bad timing DUs", \
                        marker="s", s=30, c="green", alpha=1, edgecolors='red', linewidth=1.5)

        # print(np.round(gp_array, 2))
        # print(np.round(gp_signal, 2))
        # [-1.90330e+02  3.36910e+02  2.16600e+01]
        # [ -258.68   354.21     0.  ]
        # [68.35, -17.3, 0]

        # [ 3.07250e+02  6.21570e+02  1.19400e+01]
        # [  245.14   639.74     0.  ]
        # [62.25, -18.17, 0]

        # define core 
        core_g = np.array([0, 0, 0])
        # in both subplots
        axs[1].scatter(core_g[0], core_g[1], marker='*', s=175, c='white', edgecolors='black', label="Radio Symmetry Centre")

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.02)
        colorbar = plt.colorbar(pcm, cax=cax, label=r'Energy fluence [eV/m$^2$]')
        # colorbar.ax.set_yticklabels([0, 10, 20, 30, 40, 50])
        colorbar.ax.tick_params(axis='both', labelsize=font-5)

        # core rectangle
        # axs[1].add_patch(plt.Rectangle((475000, -2500), 5000, 5000, ls="--", ec="blue", fc="none", label="MC cores"))
        
        # axs[1].fill_between(np.linspace(0, 1, 1), np.linspace(0, 1, 1), np.linspace(0, 1, 1), alpha=0.5, color="blue", label="LDF fit footprint")
        
        axs[1].legend(loc="best", fontsize=15)
        
        # GP300
        # labels_x = ["", -40, -20, 0]
        # labels_y = ["", -5, -0, 5, 10, 15]
        
        # 10k
        # labels_x = ["", -40, -20, 0]
        # labels_y = ["", -40, -20, 0, 20, 40]
        
        # axs[1].set_xticklabels(labels_x)
        # axs[1].set_yticklabels(labels_y)
        axs[1].tick_params(axis="both", labelsize=font-5)
        

        # set axis parameters
        axs[1].set_title("Model footprint in ground plane", fontsize=font-5)
        axs[1].set_title("Radio footprint on sparse array", fontsize=font-5)
        # axs[1].set_xlim(-40000, 25000) # xmin, xmax)
        # axs[1].set_ylim(None, 5000)
        # axs[1].set_ylim(-55000, 51000) # ymin, ymax)
        axs[1].set_xlabel(r"Easting [km]", fontsize=font-5)
        axs[1].set_ylabel(r"Northing [km]", fontsize=font-5)
        axs[1].set_aspect('equal')
        axs[1].set_xlim(-2500, 2500)
        axs[1].set_ylim(-2500, 2500)


        # add shower plane plot if requested
        if not only_ground_plane:
            # print 2D footprint and antennas with fluence values to see if they match
            axs[2].contourf(XI, YI, dtot, levels=1000, vmin=0, vmax=max(np.concatenate([fgeo, ldf])), cmap='cmr.freeze_r' # 'gnuplot2_r' \
                            , label="LDF model footprint", alpha=1)
            # plot all antennas of the array
            # axs[2].scatter(x_vB_array / c_early_late_array, y_vB_array / c_early_late_array, marker="+", s=75, color="red", alpha=0.9)
            # for i in range(len(antenna_ids)):
            #     if antenna_ids[i] == 1078 or antenna_ids[i] == 1032:
            #         axs[2].annotate(str(int(antenna_ids[i]))+" (noisy)", (x_vB_array[i] / c_early_late_array[i], y_vB_array[i] / c_early_late_array[i]), fontsize=10, annotation_clip=False)
            #     else:
            #         axs[2].annotate(int(antenna_ids[i]), (x_vB_array[i] / c_early_late_array[i], y_vB_array[i] / c_early_late_array[i]), fontsize=10, annotation_clip=True)
                
            axs[2].plot(cherenkov_ring[:, 0], cherenkov_ring[:, 1], c="black", linewidth=1.5, ls="--", label="Cherenkov radius")

            pcm2 = axs[2].scatter(x_vB / c_early_late, y_vB / c_early_late, \
                        s=125, c=fgeo, alpha=1, vmin=0, vmax=max(np.concatenate([fgeo, ldf])), cmap='cmr.freeze_r', edgecolors='red', linewidth=1.5, label="DUs with signal")
            
            # to display core position in shower plane, define shower plane coordinate system
            # define coordinate system transformations
            core_vxB = cs.transform_to_vxB_vxvxB(np.array([0, 0, 0]))
            # in both subplots
            axs[2].scatter(core_vxB[0], core_vxB[1], marker='*', s=175, c='white', edgecolors='black', label="Radio symmetry centre")
            
            axs[2].legend(loc="lower left", fontsize=15)

            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes("right", size="5%", pad=0.02)
            colorbar = plt.colorbar(pcm2, cax=cax, label=r'Energy fluence [eV/m$^2$]')
            colorbar.ax.tick_params(axis='both', labelsize=16)


            # set axis parameters
            axs[2].set_title("Shower Plane View")
            axs[2].set_xlabel(r"vxB [m]")
            axs[2].set_ylabel(r"vx(vxB) [m]")
            axs[2].set_xlim(-max(rmodel) * 0.2, max(rmodel) * 0.2)
            axs[2].set_ylim(-max(rmodel) * 0.2, max(rmodel) * 0.2)
            axs[2].set_aspect('equal')

        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"array_plot_{plot_label}.png", dpi=400)
        plt.close()










    # 4 panel
    # test whether you have reconstructed parameters:
    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()
        shower = revent.get_shower(key=shower_type)
        
    if shower.has_parameter(shp.zenith_recon): 
        font=32

        # plot layouts for 3 or 2 plot panels
        fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(20, 18))

        fig.suptitle(title, fontsize=32)

        # meshgrid of area of footprint
        XI, YI = np.mgrid[-max(rmodel) * 1.5:max(rmodel) * 1.5:lattice_const, \
                          -max(rmodel) * 1.5:max(rmodel) * 1.5:lattice_const]

        testx, testy = np.mgrid[-1:1:3j, -1:1:3j]

        # print(testx)
        # print(testy)

        grid_coord_test = np.array([[testx.flatten()[i], testy.flatten()[i], 0] for i in range(len(testx.flatten()))])
        # print(grid_coord_test)
        
        
        # 3D coordinates of the meshgrid
        # grid_coord = np.array([[XI.flatten()[i], YI.flatten()[i], 0] for i in range(len(YI.flatten()))])

        # transform the grid to ground plane
        # grid_ground = cs.transform_from_vxB_vxvxB_2D(grid_coord)

        # transform back to matrix format
        # XI_gp = np.array(np.array_split(grid_ground[:, 0], np.sqrt(len(grid_ground))))
        # YI_gp = np.array(np.array_split(grid_ground[:, 1], np.sqrt(len(grid_ground))))
        # print(XI.shape)
        
        # calculate predicted fluence of footprint for each cell
        dtot = func_geo(np.sqrt(XI ** 2 + YI ** 2), **params_dict)
        ldf = np.array(func_geo(rmodel, **params_dict))

        r_che = plot_f_gaus_sigmoid(
            [axs[0, 0]],
            params_dict,
            rmodel,
            func_geo,
            ldfs.ldf_gaus_sigmoid_shape,
            **plot_gaus_kwargs
        )

        circle_angles = np.deg2rad(np.linspace(0, 360, 1000))
        cherenkov_ring = np.array([[r_che * np.cos(circle_angles[i]), r_che * np.sin(circle_angles[i]), 0] for i in range(len(circle_angles))])
        cherenkov_ring_double = np.array([[2 * r_che * np.cos(circle_angles[i]), 2 * r_che * np.sin(circle_angles[i]), 0] for i in range(len(circle_angles))])

        che_ground = cs.transform_from_vxB_vxvxB_2D(cherenkov_ring)
        che_ground_double = cs.transform_from_vxB_vxvxB_2D(cherenkov_ring_double)

        axs[0, 0].errorbar(
                r[neg_mask],
                fgeo[neg_mask],
                fgeo_weight[neg_mask],
                marker="o",
                ls="",
                color="dodgerblue",
                markersize=12,
                label=r"Fluence @ antenna"
            )
        
        # for i in range(len(antenna_IDs)):
        #     axs[0, 0].annotate(int(antenna_IDs[i]), (r[neg_mask][i] - 50, fgeo[neg_mask][i] + 4), fontsize=12, color="black", annotation_clip=False)
        
        # repeat for cut stations
        if (f_vxB_noisy.any() != None):
            axs[0, 0].errorbar(
                r_noisy,
                fgeo_noisy,
                np.abs(fgeo_weight_noisy),
                marker="x",
                ls="",
                color="black",
                markersize=12,
                alpha=0.5,
                label=r"ADC noise cut"
            )

        # repeat for cut stations
        if (f_vxB_bad_timing.any() != None):
            axs[0, 0].errorbar(
                r_bad_timing,
                fgeo_bad_timing,
                np.abs(fgeo_weight_bad_timing),
                marker="s",
                ls="",
                color="green",
                markersize=12,
                alpha=0.5,
                label=r"Peak timing cut"
            )
            
        # repeat for cut stations
        if (f_vxB_snr.any() != None):
            axs[0, 0].errorbar(
                r_snr,
                fgeo_snr,
                np.abs(fgeo_weight_snr),
                marker="^",
                ls="",
                color="navy",
                markersize=12,
                alpha=0.5,
                label=r"SNR cut"
            )

        # repeat for saturated stations
        if (f_vxB_saturated.any() != None):
            axs[0, 0].errorbar(
                r_saturated,
                fgeo_saturated,
                np.abs(fgeo_weight_saturated),
                marker="v",
                ls="",
                color="red",
                markersize=12,
                alpha=0.5,
                label=r"Saturated DUs"
            )
        
        axs[0, 0].set_title("LDF fit to fluence data points", fontsize=font-5)
        # axs[0].set_title(r"LDF fit to $f_\mathrm{geo}^{\,\mathrm{par}}$", fontsize=font-5)
        # axs[0].set_xlim(0, 5600) # 10k
        # axs[0].set_xlim(0, 1700) # GP300
        # axs[0].set_xlim(0, max(rmodel))
        axs[0, 0].set_ylim(0, max(np.concatenate([fgeo, ldf])) * 1.2)
        axs[0, 0].set_xlabel(r"Axis distance $r$ [m]", fontsize=font-5)
        axs[0, 0].set_ylabel(r"Geomagnetic fluence [eV$\,$m$^{-2}$]", fontsize=font-5)
        axs[0, 0].legend(loc="upper right", fontsize=font-12)
        axs[0, 0].tick_params(axis="both", labelsize=font-5)
        # labels_y = ["", 0, 10, 20, 30, 40, 50, 60]
        # axs[0].set_yticklabels(labels_y)

        axs[0, 0].grid(axis="x")
        
        
        
        # plot for antenna timings
        
        for shower in revent.get_showers():
            shower_type = shower.get_shower_type()

            shower = revent.get_shower(key=shower_type)
            
        zenith = shower.get_parameter(shp.zenith_recon)
        azimuth = shower.get_parameter(shp.azimuth_recon)
        
        # refractive index at ground level
        n_atm = 1.0002598670971181
        # speed of light
        c_light = 2.997924580e8
        
        # normal vector of plane wave front defined by reconstructed arrival direction
        pwf_norm_vec = np.array([-np.sin(zenith) * np.cos(azimuth), -np.sin(zenith) * np.sin(azimuth), -np.cos(zenith)])
        
        # antenna positions
        positions = cs.transform_from_vxB_vxvxB(np.array([[x_vB[i], y_vB[i], z_vB[i]] for i in range(len(x_vB))]))
        
        # calculate distance to PWF
        distance_pwf = np.array([np.dot((positions[i] - mean(positions)), pwf_norm_vec) / betrag(pwf_norm_vec) for i in range(len(positions))])
        distance_pwf_core = np.array([np.dot((positions[i] - np.array([0, 0, 0])), pwf_norm_vec) / betrag(pwf_norm_vec) for i in range(len(positions))])
        
        # all in nanoseconds
        # time from mean position
        pred_timing = distance_pwf / (c_light / n_atm) * 1e9
        # also time from mean position, but with time jitter
        t_PWF = create_times(positions, pwf_norm_vec, sigma=0, c=c_light, n=n_atm) * 1e9
        t_PWF = t_PWF - t_PWF.mean()
        # time from shower core fit position
        core_timing = distance_pwf_core / (c_light / n_atm) * 1e9
        # data arrival time substracted by their own mean
        arrival_timing = (antenna_times -  antenna_times.mean()) * 1e9
        
        # shift both times to get relative time from
        t_PWF = t_PWF - core_timing[0]
        arrival_timing = arrival_timing - core_timing[0]

        # calulate PWF goodness of fit
        chi_PWF = chi2_PWF(t_PWF / 1e9, arrival_timing / 1e9, sigma=15 / 1e9)
        
        # plot timing residuals of measuring antennas
        axs[0, 1].errorbar(distance_pwf_core, (arrival_timing - t_PWF), abs(15), label=r"rel. arrival time", \
            ls="", marker=".", markersize=15, color="navy", alpha=1, capsize=8, markerfacecolor="navy", markeredgewidth=2)
        # plot 0 line
        xmin, xmax = axs[0, 1].get_xlim()
        # axs[0, 1].plot(r, (pred_timing - pred_timing), ls="--", c="black", lw=2, label="PWF propagation" + "\n" + r"$\chi_\mathrm{PWF}^2=$" + f"{np.round(chi_PWF, 2)}")
        axs[0, 1].hlines(0, distance_pwf_core.min(), distance_pwf_core.max(), ls="--", color="black", lw=2, label="PWF propagation" + "\n" + r"$\chi_\mathrm{PWF}^2=$" + f"{np.round(chi_PWF, 2)}")
        
        for i in range(len(antenna_IDs)):
            axs[0, 1].annotate(int(antenna_IDs[i]), (distance_pwf_core[i] + 30, (arrival_timing[i] - t_PWF[i])), fontsize=15, color="black", annotation_clip=False)
        
        axs[0, 1].grid(alpha=0.5)
        axs[0, 1].set_title("Accuracy of arrival direction", fontsize=font-5)
        axs[0, 1].set_xlabel(r"Distance to PWF at core position [m]", fontsize=font-5)
        axs[0, 1].set_ylabel(r"Timing residuals $t_\mathrm{PWF}$ - $t_\mathrm{ant}$ [ns]", fontsize=font-5)
        axs[0, 1].legend(fontsize=font-15)
        axs[0, 1].tick_params(axis="both", labelsize=font-5)
        


        # ground plane plot
        # print 2D footprint and antennas with fluence values to see if they match
        # plot all antennas of the array
        # x as y and y as  x because x points to north

        # transform antennas back to ground plane in proper coordinate system
        # gp_array = cs.transform_from_vxB_vxvxB(np.array([[x_vB_array[i], y_vB_array[i], z_vB_array[i]] for i in range(len(x_vB_array))]))
        # axs[1].scatter(-gp_array[:, 1], gp_array[:, 0], marker="+", s=50, color="red", alpha=1)

        # axs[1].plot(-che_ground[:, 1], che_ground[:, 0], c="black", linewidth=1.5, ls="--", label="Cherenkov radius")
        axs[1, 0].plot(-che_ground_double[:, 1], che_ground_double[:, 0], c="black", linewidth=1.5, ls="--")

        # get axis limits for meshgrid
        xmin, xmax = axs[1, 0].get_xlim() # -max(rmodel) * 1.5, max(rmodel) * 1.5 
        ymin, ymax = axs[1, 0].get_ylim() # - max(rmodel) * 1.5, max(rmodel) * 1.5 

        # mesh grid lattice constant
        grid_size = 200j


        XI_gp, YI_gp = np.mgrid[min(xmin, ymin):max(xmax, ymax):grid_size, \
                                min(xmin, ymin):max(xmax, ymax):grid_size]

        # XI_gp, YI_gp = np.mgrid[-1:1:3j, \
        #                         -1:1:3j,]
        # print(XI_gp)
        # print(YI_gp)

        # get 2D coordinates from mesh
        grid_gp = np.array([[XI_gp.flatten()[i], YI_gp.flatten()[i], 0] for i in range(len(YI_gp.flatten()))])
        grid_sp = cs.transform_to_vxB_vxvxB(grid_gp)

        # calculate early-late correction factor for all mesh grid points
        c_early_late_gp = early_late.early_late_correction_factor(grid_sp[:, 2], distance_xmax_geometric_fit)

        # # transform back into matrix format
        # XI_sp= np.array(np.array_split(grid_gp[:, 0], np.sqrt(len(grid_gp))))
        # YI_sp = np.array(np.array_split(grid_gp[:, 1], np.sqrt(len(grid_gp))))

        # print(XI_sp)
        # print(YI_sp)

        # transform back into matrix format
        XI_sp= np.array(np.array_split(grid_sp[:, 0], np.sqrt(len(grid_sp))))
        YI_sp = np.array(np.array_split(grid_sp[:, 1], np.sqrt(len(grid_sp))))
        # also transform early-late factors accordingly
        c_early_late_matrix = np.array(np.array_split(c_early_late_gp, np.sqrt(len(grid_sp))))

        # calculate LDF value for meshgrid radii and apply early-late factors
        dtot_gp = func_geo(np.sqrt(XI_sp ** 2 + YI_sp ** 2) / c_early_late_matrix, **params_dict)

        # print 2D footprint and antennas with fluence values to see if they match
        axs[1, 0].contourf(-YI_gp, XI_gp, dtot_gp, levels=1000, vmin=0, vmax=max(np.concatenate([fgeo, dtot_gp.flatten()])), cmap='cmr.freeze_r' # 'gnuplot2_r' \
                            , label="LDF fit footprint", alpha=1)

        # plot all antennas of the array
        # x as y and y as  x because x points to north

        # transform antennas back to ground plane in proper coordinate system
        gp_array = cs.transform_from_vxB_vxvxB(np.array([[x_vB_array[i], y_vB_array[i], z_vB_array[i]] for i in range(len(x_vB_array))]))
        # axs[1, 0].scatter(-gp_array[:, 1], gp_array[:, 0], marker="+", s=15, color="red", alpha=1, label="GP300 DUs", edgecolors="black")
        # axs[1].scatter(array_positions2[:, 0] - 2500, array_positions2[:, 1] - 35000, marker="o", s=3, color="red", alpha=1, label="GP300")

        if 1:
            gp80_array = cs.transform_from_vxB_vxvxB(np.array([[x_vB_gp80[i], y_vB_gp80[i], z_vB_gp80[i]] for i in range(len(x_vB_gp80))]))
            gp80_plot = gp80_array
            axs[1, 0].scatter(-gp80_array[~untriggered_mask][:, 1], gp80_array[~untriggered_mask][:, 0], marker="o", s=80, c="white", edgecolors='black', linewidth=1.5) # , label="Untriggered DUs") 
            
            # for i in range(len(plot_ids)):
            #     if 0: # antenna_ids[i] == 1078 or antenna_ids[i] == 1032:
            #         axs[1, 0].annotate(str(int(antenna_ids[i]))+" (noisy)", (-gp80_array[i, 1] + 75, gp80_array[i, 0]), fontsize=10, color="gray", annotation_clip=True)
            #     else:
            #         axs[1, 0].annotate(int(plot_ids[i]), (-gp80_plot[i, 1] - 175, gp80_plot[i, 0] + 110), fontsize=9, color="gray", annotation_clip=False)

        # also transform the signal stations to ground plane properly
        gp_signal = cs.transform_from_vxB_vxvxB(np.array([[x_vB[i], y_vB[i], z_vB[i]] for i in range(len(x_vB))]))
        pcm = axs[1, 0].scatter(-gp_signal[:, 1], gp_signal[:, 0], label=r"Fluence @ antenna", \
                       s=200, c=fgeo, alpha=1, vmin=0, vmax=max(np.concatenate([fgeo, dtot_gp.flatten()])), cmap='cmr.freeze_r', edgecolors='white', linewidth=1.2)
        
        # plot bad timing antennas in green
        if (f_vxB_bad_timing.any() != None):
            # also transform the signal stations to ground plane properly
            try:
                gp_peak_timing = cs.transform_from_vxB_vxvxB(np.array([[x_vB_bad_timing[i], y_vB_bad_timing[i], z_vB_bad_timing[i]] for i in range(len(x_vB_bad_timing))]))
                axs[1, 0].scatter(-gp_peak_timing[:, 1], gp_peak_timing[:, 0], label="Bad timing DUs", \
                        marker="s", s=60, c="green", alpha=1, edgecolors='red', linewidth=1.5)
            except:
                gp_peak_timing = cs.transform_from_vxB_vxvxB(np.array([[x_vB_bad_timing, y_vB_bad_timing, z_vB_bad_timing]]))
                axs[1, 0].scatter(-gp_peak_timing[1], gp_peak_timing[0], label="Bad timing DUs", \
                        marker="s", s=60, c="green", alpha=1, edgecolors='red', linewidth=1.5)

        # print(np.round(gp80_array, 2))
        # print(np.round(gp_signal, 2))
        # [-1.90330e+02  3.36910e+02  2.16600e+01]
        # [ -258.68   354.21     0.  ]
        # [68.35, -17.3, 0]

        # [424.3  422.2 -9.53]
        # [ 402.57  477.45   -9.15]
        # [21.73, -55.25, -0.38]

        # define core 
        core_g = np.array([0, 0, 0])
        # in both subplots
        # axs[1, 0].plot(-che_ground_double[:, 1], che_ground_double[:, 0], c="black", linewidth=1.5, ls="--", label="2x Cherenkov radius")

        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.02)
        colorbar = plt.colorbar(pcm, cax=cax, label=r'Energy fluence [keV/m$^2$]')
        # colorbar.ax.set_yticklabels([0, 10, 20, 30, 40, 50])
        colorbar.ax.tick_params(axis='both', labelsize=font-5)

        # core rectangle
        # axs[1].add_patch(plt.Rectangle((475000, -2500), 5000, 5000, ls="--", ec="blue", fc="none", label="MC cores"))
        
        axs[1, 0].scatter(core_g[0], core_g[1], marker='*', s=175, c='white', edgecolors='black', label="Radio Symmetry Centre")
        axs[1, 0].fill_between(np.linspace(0, 1, 1), np.linspace(0, 1, 1), np.linspace(0, 1, 1), alpha=0.5, color="blue", label="LDF footprint")
        # axs[1, 0].plot(-che_ground_double[:, 1], che_ground_double[:, 0], c="black", linewidth=1.5, ls="--", label="2x Cherenkov radius")
        
        axs[1, 0].legend(fontsize=font-18)
        
        # GP300
        # labels_x = ["", -40, -20, 0]
        # labels_y = ["", -5, -0, 5, 10, 15]
        
        # 10k
        # labels_x = ["", -40, -20, 0]
        # labels_y = ["", -40, -20, 0, 20, 40]
        
        # axs[1].set_xticklabels(labels_x)
        # axs[1].set_yticklabels(labels_y)
        axs[1, 0].tick_params(axis="both", labelsize=font-5)
        

        # set axis parameters
        axs[1, 0].set_title("Footprint in ground plane", fontsize=font-5)
        # axs[1].set_xlim(-40000, 25000) # xmin, xmax)
        # axs[1].set_ylim(None, 5000)
        # axs[1].set_ylim(-55000, 51000) # ymin, ymax)
        axs[1, 0].set_xlabel(r"Easting [km]", fontsize=font-5)
        axs[1, 0].set_ylabel(r"Northing [km]", fontsize=font-5)
        axs[1, 0].set_aspect('equal')





        # shower plane plot for vxB fluence asymmetry

        # print 2D footprint and antennas with fluence values to see if they match
        axs[1, 1].contourf(XI, YI, dtot, levels=1000, vmin=0, vmax=max(np.concatenate([fgeo, ldf])), cmap='cmr.freeze_r' # 'gnuplot2_r' \
                        , label="LDF model footprint", alpha=1)
        # plot all antennas of the array
        # axs[2].scatter(x_vB_array / c_early_late_array, y_vB_array / c_early_late_array, marker="+", s=75, color="red", alpha=0.9)
        # for i in range(len(antenna_ids)):
        #     if antenna_ids[i] == 1078 or antenna_ids[i] == 1032:
        #         axs[2].annotate(str(int(antenna_ids[i]))+" (noisy)", (x_vB_array[i] / c_early_late_array[i], y_vB_array[i] / c_early_late_array[i]), fontsize=10, annotation_clip=False)
        #     else:
        #         axs[2].annotate(int(antenna_ids[i]), (x_vB_array[i] / c_early_late_array[i], y_vB_array[i] / c_early_late_array[i]), fontsize=10, annotation_clip=True)
            
        # axs[1, 1].scatter(x_vB_array / c_early_late_array, y_vB_array / c_early_late_array, \
        #             marker="+", s=50, c="red", edgecolors='black', linewidth=1.5, label="GP300 DUs")
        
        axs[1, 1].scatter(x_vB_gp80[~untriggered_mask] / c_early_late_gp80[~untriggered_mask], y_vB_gp80[~untriggered_mask] / c_early_late_gp80[~untriggered_mask], \
                    marker="o", s=50, c="white", edgecolors='black', linewidth=1.5, label="Untriggered antennas")    
        
        for i in range(len(antenna_IDs)):
            axs[1, 1].annotate(int(antenna_IDs[i]), (x_vB[i] / c_early_late[i] - 50, y_vB[i] / c_early_late[i] + 40), fontsize=12, color="gray", annotation_clip=False)

        pcm2 = axs[1, 1].scatter(x_vB / c_early_late, y_vB / c_early_late, \
                    s=250, c=fgeo, alpha=1, vmin=0, vmax=max(np.concatenate([fgeo, dtot_gp.flatten()])), # vmax=max(np.concatenate([f_vxB * (c_early_late ** 2), ldf])), \
                    cmap='cmr.freeze_r', edgecolors='white', linewidth=1.2, label=r"Fluence @ antenna")
        
        # plot bad timing antennas in green
        if (f_vxB_bad_timing.any() != None):
            # also transform the signal stations to ground plane properly
            axs[1, 1].scatter(x_vB_bad_timing / c_early_late_bad_timing, y_vB_bad_timing / c_early_late_bad_timing, label="Bad timing DUs", \
                            marker="s", s=100, c="green", alpha=1, edgecolors='red', linewidth=1.5)

        
        # print(f_vxB * (c_early_late ** 2))
        # print(fgeo)
        # print(f_vxB * (c_early_late ** 2) / fgeo)
        
        # to display core position in shower plane, define shower plane coordinate system
        # define coordinate system transformations
        core_vxB = cs.transform_to_vxB_vxvxB(np.array([0, 0, 0]))
        # in both subplots
        axs[1, 1].scatter(core_vxB[0], core_vxB[1], marker='*', s=175, c='white', edgecolors='black')
        
        # axs[1, 1].plot(cherenkov_ring[:, 0], cherenkov_ring[:, 1], c="grey", linewidth=1.5, ls="--", label="Cherenkov radius")
        axs[1, 1].fill_between(np.linspace(0, 1, 1), np.linspace(0, 1, 1), np.linspace(0, 1, 1), alpha=0.5, color="blue", label="LDF footprint")
        
        axs[1, 1].legend(loc="lower left", fontsize=font-15)

        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.02)
        colorbar = plt.colorbar(pcm2, cax=cax, label=r'Energy fluence [eV/m$^2$]')
        colorbar.ax.tick_params(axis='both', labelsize=font-5)


        # set axis parameters
        axs[1, 1].set_title("Shower plane", fontsize=font-5)
        axs[1, 1].set_xlabel(r"$\vec{v}\times\vec{B}$ [m]", fontsize=font-5)
        axs[1, 1].set_ylabel(r"$\vec{v}\times(\vec{v}\times\vec{B})$ [m]", fontsize=font-5)
        axs[1, 1].set_xlim(-max(rmodel) * 0.6, max(rmodel) * 0.6)
        axs[1, 1].set_ylim(-max(rmodel) * 0.6, max(rmodel) * 0.6)
        axs[1, 1].set_aspect('equal')

        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"quadrupanel_{plot_label}.png", dpi=400)
        plt.close()
