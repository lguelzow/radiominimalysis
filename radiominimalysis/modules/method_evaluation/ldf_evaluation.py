from tkinter import E
from radiominimalysis.framework import factory
from radiominimalysis.framework.parameters import \
    showerParameters as shp, eventParameters as evp, stationParameters as stp
from radiominimalysis.utilities import stats as helperstats, energyreconstruction

from radiominimalysis.modules.method_evaluation.gauss_sigmoid_param import get_bins_for_x_from_binned_data

from radiotools.atmosphere import models as atm
from radiotools.analyses import radiationenergy
from radiotools import helper

from radiominimalysis.utilities import pyplots

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr

import sys
import numpy as np
import iminuit


def get_data_bins(data, bins, return_bins=False, eps=1e-6):
    n, edges = np.histogram(data, bins)

    edges[-1] += eps  # to include the highest value in the bin
    masks = np.array(
        [np.all([data >= edges[idx], data < edges[idx + 1]], axis=0) for idx in range(len(n))])
    center_values = edges[:-1] + (edges[1:] - edges[:-1]) / 2
    
    if np.sum(masks) != len(data):
        print("get_data_bins(): sum(masks) != len(data)")
        print(np.sum(masks), len(data), data[~np.any(masks, axis=0)])

    if return_bins:
        return center_values, masks, edges
    else:
        return center_values, masks


def get_pp_index_acronym(pnum, corsika=False):
    acr = ["p", "He", "N", "Fe"]
    colors = ["r", "C1", "g", "b"]
    if corsika:
        idx = [14, 402, 1407, 5626].index(pnum)
    else:
        idx = [2212, 1000002004, 1000007014, 1000026056].index(pnum)
    return idx, acr[idx], colors[idx]


def evaluate_gauss_sigmoid_chi(events, para):
    # has_shower = np.array([ev.has_shower(evp.rd_shower) for ev in events])
    # print("Events with Rd shower: %d / %d" %
    #       (np.sum(has_shower), len(has_shower)))
    # events = events[has_shower]

    mask = factory.has_parameter(events, shp.fit_result)
    print(mask)
    print("Events with fit result: %d / %d" % (np.sum(mask), len(mask)))
    events = events[mask]

    if True:
        # ['nfev', 'nvarys', 'ndata', 'nfree', 'chisqr', 'redchi', 'init_values', 'success']
        fit_result = factory.get_parameter(events, shp.fit_result)

        dmax_mc = factory.get_parameter(
            events, shp.distance_to_shower_maximum_geometric)
        geomag_angle_mc = np.array(
        [ev.get_geomagnetic_angle() for ev in events])
        
        rho_mc = factory.get_parameter(events, shp.density_at_shower_maximum)
    
        eem_mc = factory.get_parameter(events, shp.electromagnetic_energy)
        egeo_mc = energyreconstruction.get_Egeo(eem_mc, np.sin(geomag_angle_mc), rho_mc)

        label = "E_\mathrm{geo}^\mathrm{MC}"
    else:
        fit_result = factory.get_parameter(events, shp.fit_result)
        mask = fit_result != False
        print("Events with successful fit: %d / %d" %
              (np.sum(mask), len(mask)))
        events = events[mask]
        fit_result = fit_result[mask]

        # e_mc = factory.get_parameter(events, shp.electromagnetic_energy, evp.sim_shower)
        dmax_mc = factory.get_parameter(
            events, shp.distance_to_shower_maximum_geometric, evp.sim_shower)
        label = "E_\mathrm{eem}^\mathrm{MC}"

    def get_ele(key):
        return np.array([ele[key] for ele in fit_result])

    if "nfree" in fit_result[0]:
        ndf = get_ele("nfree")
    else:
        ndf = get_ele('ndata') - get_ele("nfit")

    redchi = get_ele("redchi")

    print(np.mean(redchi), np.std(redchi), np.quantile(redchi, 0.95))
    mask = ndf > -1
    max_val = 15 

    fig = plt.figure()
    ax1 = plt.subplot(221)
    ax1.plot(dmax_mc[mask] / 1e3, redchi[mask], "o", alpha=0.1)
    ax1.set_xlabel(r"$d_\mathrm{max}^\mathrm{MC}$ [km]")
    ax1.set_ylabel(r"red. chi-square")

    ax2 = plt.subplot(223)
    ax2.plot(np.log10(egeo_mc)[mask], redchi[mask], "o", alpha=0.1)
    # ax2.plot(redchi[mask], redchi[mask], "o", alpha=0.1)
    ax2.set_xlabel(r"lg($%s$ / eV)" % label)
    ax2.set_ylabel(r"red. chi-square")

    ax3 = plt.subplot(122)
    if max_val is not None:
        bins = np.linspace(0, max_val)
        overflow = np.sum(redchi[mask] > max_val)
    else:
        bins = 100
        overflow = 0

    median = np.quantile(redchi[mask], 0.5)
    q68 = np.quantile(redchi[mask], 0.68)
    q95 = np.quantile(redchi[mask], 0.95)

    ax3.hist(redchi[mask], bins)
    ax3.axvline(median, color="k", ls="--")
    s = (r"N = {}" + "\n" + 
         "overflow = {:d}" + "\n" + 
         "median = {:.2f}" + "\n" + 
         "$\sigma_{{95}}$ = {:.2f}").format(
        len(redchi[mask]), overflow, median, q95)
    ax3.text(0.95, 0.95, s, transform=ax3.transAxes, fontsize=10,
             horizontalalignment='right', verticalalignment='top')
    ax3.set_xlabel(r"red. chi-square")

    [ax.set_ylim(0, max_val) for ax in [ax1, ax2]]
    [ax.grid() for ax in [ax1, ax2, ax3]]

    plt.tight_layout()
    if para.save:
        fig.savefig("chi%s.png" % para.label)
    else:
        plt.show()


def event_selection_ldf_validation(events, 
                                   has_any_stations=False,
                                   high_zenith_cut=False,
                                   low_geomagnetic_angle=False,
                                   energy_range_cut=False,
                                   only_proton=False,
                                   only_iron=False,
                                   saturation_cut=False,
                                   mc_xmax_cut=False,
                                   direction_error_cut=False,
                                   station_number_cut_SNR_cut=False,
                                   station_number_cut_LDF_fit=False,
                                   has_fit_parameters=True,
                                   has_successful_fit=True,
                                   cherenkov_radius_cut=False,
                                   cherenkov_radius_cut_with_saturated=False,
                                   dmax_error_cut=False,
                                   chi_percentile_cut=False,
                                   egeo_error_cut=False,
                                   large_dmax_cut=False,
                                   has_compare_fluence=False
                                   ):

    # total number of events to be used for analysis
    total = len(events)

    # exclude events with zenith angles over 85°
    if high_zenith_cut:
        zenith = factory.get_parameter(events, shp.zenith)
        mask = np.rad2deg(zenith) <= 85
        # mask = np.rad2deg(zenith) >= 75
        # mask = np.all([np.rad2deg(zenith) > 68, np.rad2deg(zenith) <= 85], axis=0)
        print(r"Events with zenith angle <= 85e < 85: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]
        
    # cut events with small geomagnetic angles
    if low_geomagnetic_angle:
        alpha = factory.get_parameter(events, shp.geomagnetic_angle)
        mask = alpha > 0.35
        print(r"Events with geomagnetic angle > 20°: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]

    # limit events to a specific energy range
    if energy_range_cut:
        Eem = factory.get_parameter(events, shp.electromagnetic_energy)
        mask = np.all([Eem > 10 ** 18.5, Eem < 10 ** 19.5], axis=0)
        # mask = Eem > 10 ** 18.5
        # mask = Eem < 10 ** 19.1
        print(r"Events with Eem above 1e17.5 eV: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]

    # selct only one type of primary cosmic ray particle
    if only_proton or only_iron:
        # contigency for both
        if only_proton and only_iron: 
            print("Both primaries selected! Please choose only one!")
            exit()
        # readout the primary types and make the masks
        primary = factory.get_parameter(events, shp.primary_particle)
        print(primary)
        only_proton_mask = (primary == "14.0")#  or (primary == 14)
        only_iron_mask = (primary == "5626.0") #  or (primary == 5626)
        # choose the right mask
        if only_proton:
            print(r"Events with proton primary: %d / %d --- %.2f" %
              (np.sum(only_proton_mask), len(only_proton_mask), 100 * np.sum(only_proton_mask) / total))
            events = events[only_proton_mask]

        if only_iron:
            print(r"Events with iron primary: %d / %d --- %.2f" %
              (np.sum(only_iron_mask), len(only_iron_mask), 100 * np.sum(only_iron_mask) / total))
            events = events[only_iron_mask]
            
    # exclude events with bad MC Xmax
    if mc_xmax_cut:
        xmax_mc = factory.get_parameter(events, shp.xmax)
        mask = xmax_mc > 200
        print(r"Events with resonable mc xmax > 200: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]
        
    # cut events with too large error on arrival direction
    if direction_error_cut:
        if np.all(factory.has_parameter(events, shp.pointing_error)):
            dir_err = factory.get_parameter(events, shp.pointing_error)
            mask = dir_err < 0.5 # np.rad2deg(dir_err) < 0.5 # in degrees for MC
            print(r"Events with dir. error < 0.5°: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
            events = events[mask]
             
    # check whether simulation has any antennas
    # also maybe not necessary
    if has_any_stations:
        # check whether simulation has any antennas
        mask = np.array([ev.has_station_parameter(stp.position) for ev in events])
        print(r"Events with any triggered stations: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]
   
    # exclude stations with too few stations after SNR cut
    if station_number_cut_SNR_cut:
        n_stat = np.array([len(ev.get_total_energy_fluence()) for ev in events]) # maybe change this to the fluence after the cut
        mask = n_stat >= station_number_cut_SNR_cut
        print(rf"Events with # of stations passing SNR cut >= {station_number_cut_SNR_cut}: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]
        
     # amount of stations after saturated stations removed
    if station_number_cut_LDF_fit:
        n_stat = np.array([len(ev.get_station_parameter(stp.vxB_fluence_simulated)) for ev in events])
        mask = n_stat >= station_number_cut_LDF_fit
        print(rf"Events with # of stations viable for LDF fit >= {station_number_cut_LDF_fit}: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]

    # check whether events have fit parameters
    if has_fit_parameters:
        mask = factory.has_parameter(events, shp.geomagnetic_ldf_parameter)
        print(r"Events with fit parameter: %d / %d --- %.2f" %
                (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]

    # check whether events were fitted sucessfully
    if has_successful_fit:
        fit_result = factory.get_parameter(events, shp.fit_result)
        success = np.array([x["success"] for x in fit_result])
        print(r"Events with successful fit: %d / %d --- %.2f" %
            (np.sum(success), len(success), 100 * np.sum(success) / total))
        events = events[success]

    # exclude events that have no station inside the fitted cherenkov radius
    if cherenkov_radius_cut:
        if np.all(factory.has_parameter(events, shp.geomagnetic_ldf_parameter)):
            geo_ldf_params_fit = factory.get_parameter(events, shp.geomagnetic_ldf_parameter)
            # cherenkov radius
            r0 = np.array([x['r0'] for x in geo_ldf_params_fit])
            
            # only use non-saturated antennas for this criterium
            if (np.sum(np.array([ev.has_station_parameter(stp.saturated) for ev in events])) > 0) and \
               (np.sum(np.array([np.sum(ev.get_station_parameter(stp.saturated)) for ev in events])) > 0):
                    
                pos_vB = [event.get_station_position_vB_vvB()[~event.get_station_parameter(stp.saturated)] for event in events]               
            else:
                pos_vB = [event.get_station_position_vB_vvB() for event in events]
            # calculate radius in shower plane for every station of every event
            r = [np.array([np.sqrt(pos_vB[i][j, 0] ** 2 + pos_vB[i][j, 1] ** 2) for j in range(len(pos_vB[i]))]) for i in range(len(pos_vB))]
            mask = np.array([min(r[i]) < cherenkov_radius_cut * r0[i] for i in range(len(r0))])
            print(rf"Events with at least 1 station inside {cherenkov_radius_cut} x r0: %d / %d --- %.2f" %
                (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
            events = events[mask]
            
    # exclude events that have no station inside the fitted cherenkov radius
    if cherenkov_radius_cut_with_saturated:
        if np.all(factory.has_parameter(events, shp.geomagnetic_ldf_parameter)):
            geo_ldf_params_fit = factory.get_parameter(events, shp.geomagnetic_ldf_parameter)
            # cherenkov radius
            r0 = np.array([x['r0'] for x in geo_ldf_params_fit])
            pos_vB = [event.get_station_position_vB_vvB() for event in events]
            # calculate radius in shower plane for every station of every event
            r = [np.array([np.sqrt(pos_vB[i][j, 0] ** 2 + pos_vB[i][j, 1] ** 2) for j in range(len(pos_vB[i]))]) for i in range(len(pos_vB))]
            mask = np.array([min(r[i]) < cherenkov_radius_cut * r0[i] for i in range(len(r0))])
            print(rf"Events with at least 1 station inside {cherenkov_radius_cut} x r0: %d / %d --- %.2f" %
                (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
            events = events[mask]
            
    # eliminate events with more than 10% saturated stations
    # given they have the parameter
    # probably not needed anymore
    if saturation_cut and np.sum(np.array([ev.has_station_parameter(stp.saturated) for ev in events])) > 0:
        saturation_mask = np.ones(events.shape, dtype=bool)
        for i in range(len(events)):
            saturated = events[i].get_station_parameter(stp.saturated)
            if np.sum(saturated) / len(saturated) > 0: # 0.1
                saturation_mask[i] = False

        print(r"Events with no saturated antennas: %d / %d --- %.2f" %
              (np.sum(saturation_mask), len(saturation_mask), 100 * np.sum(saturation_mask) / total))
        events = events[saturation_mask]
 
    # exclude events large relative errors on dmax
    if dmax_error_cut:
        if np.all(factory.has_parameter_error(events, shp.distance_to_shower_maximum_geometric_fit)):
            dmax_fit, dmax_fit_err = factory.get_parameter_and_error(
                events, shp.distance_to_shower_maximum_geometric_fit)
            mask = np.all([0.001 < dmax_fit_err / dmax_fit,
                           dmax_fit_err / dmax_fit < 0.3], axis=0)
            print(r"Events with 0.001 < rel dmax err < 0.3: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
            events = events[mask]

    # exclude events with large relative egeo errors
    if egeo_error_cut:
        if np.all(factory.has_parameter_error(events, shp.geomagnetic_ldf_parameter)):
            egeo_fit = np.array([x['E_geo'] for x in factory.get_parameter(
                events, shp.geomagnetic_ldf_parameter)])
            egeo_fit_err = np.array([x['E_geo'] for x in factory.get_parameter_error(
                events, shp.geomagnetic_ldf_parameter)])
            mask = np.all([0.001 < egeo_fit_err / egeo_fit,
                           egeo_fit_err / egeo_fit < 0.3], axis=0)
            print(r"Events with 0.001 < rel Egeo err < 0.3: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
            # mask = np.all([egeo_fit_err / egeo_fit < 0.3], axis=0)
            # print("Events with rel Egeo err < 0.3: %d / %d" %
            #       (np.sum(mask), len(mask)))
            
            events = events[mask]
            
    # exclude events with large reduced chi squared
    if chi_percentile_cut:
        fit_result = factory.get_parameter(events, shp.fit_result)
        redchi = np.array([x["redchi"] for x in fit_result])
        q95 = np.quantile(redchi, .95)
        mask = redchi < q95
        print("Red. Chi^2 95% quantile: {:.3f}, cut".format(q95))
        events = events[mask]

    # exclude events with very large dmax
    if large_dmax_cut:
        dmax_fit, dmax_fit_err = factory.get_parameter_and_error(
            events, shp.distance_to_shower_maximum_geometric_fit)
        mask = dmax_fit < 500e3
        print(r"Events with dmax < 500e3: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]
        
    if has_compare_fluence:
        # only pass events with compare fluence
        mask = np.array([ev.has_station_parameter(stp.fluence_compare_MC) for ev in events])
        print(r"Events with any fluence to compare: %d / %d --- %.2f" %
              (np.sum(mask), len(mask), 100 * np.sum(mask) / total))
        events = events[mask]

    return events



def evaluate_dmax_fit(events, para):
    from matplotlib import rc
    rc('font', size = 25)
    print("Start of dmax evaluation!")
    # selection for events for fit dependent characters!
    events = event_selection_ldf_validation(events,
                                   high_zenith_cut=True,
                                   mc_xmax_cut=True,
                                   has_any_stations=True, 
                                   direction_error_cut=True, 
                                   station_number_cut_SNR_cut=5,
                                   station_number_cut_LDF_fit=5,
                                   has_fit_parameters=True,
                                   has_successful_fit=True,
                                   cherenkov_radius_cut=1,
                                   chi_percentile_cut=False,
                                   dmax_error_cut=True,
                                   egeo_error_cut=True,
                                   large_dmax_cut=True, 
                                   saturation_cut=False
                                   )

    
    dmax_fit, dmax_fit_err = factory.get_parameter_and_error(events, shp.distance_to_shower_maximum_geometric_fit)
    
    # calculate MC xmax from the saved MC data in the events
    # and compare to xmax mc the xmax calculated from dmax_fit
    # takes a long time to run
    if para.realistic_input:
        
        distance_to_xmax_mc = factory.get_parameter(events, shp.MC_distance_to_shower_maximum)

        at = atm.Atmosphere(model=41)
        xmax_mc = factory.get_parameter(events, shp.xmax)
        zenith_mc = factory.get_parameter(events, shp.zenith)
        zenith_recon = factory.get_parameter(events, shp.zenith_recon)
        obs_level = factory.get_parameter(events, shp.observation_level)

        # vertical_height_xmax = np.array([at.get_vertical_height(zenith_mc[i], xmax_mc[i], observation_level=obs_level[0]) for i in range(len(zenith_mc))])
        
        # # distance between core (at a certain obs level) and the shower maximum
        # # function returns distance for hight above ground (!, not sea level), however takes obs_level into account
        # distance_to_xmax_mc = np.array([atm.get_distance_for_height_above_ground(
        #                         vertical_height_xmax[i] - obs_level[0], zenith_mc[i], observation_level=obs_level[0]) 
        #                                 for i in range(len(zenith_mc))])
        
        xmax_rec = np.array([at.get_xmax_from_distance(dmax_fit[i], zenith_recon[i], observation_level=obs_level[0]) for i in range(len(zenith_mc))])

        xmax_dev = xmax_mc - xmax_rec

    # use MC dmax from event data
    # for realistic input, this is a reconstructed or estimated value
    else:
        distance_to_xmax_mc = factory.get_parameter(events, shp.distance_to_shower_maximum_geometric)
        
        at = atm.Atmosphere(model=41)
        xmax_mc = factory.get_parameter(events, shp.xmax)
        obs_level = factory.get_parameter(events, shp.observation_level)
        zenith_mc = factory.get_parameter(events, shp.zenith)
        
        xmax_rec = np.array([at.get_xmax_from_distance(dmax_fit[i], zenith_mc[i], observation_level=obs_level[0]) for i in range(len(zenith_mc))])
        
        xmax_dev = xmax_mc - xmax_rec

    fig_dmax, axs = plt.subplots(2, sharex=True, figsize=(11, 9))
    # fig_dmax.subplots_adjust(hspace=0.0)
    
    colour = "dodgerblue"


    axs[0].errorbar(distance_to_xmax_mc / 1e3, dmax_fit / 1e3,
                     dmax_fit_err / 1e3, marker="o", ls="", alpha=0.1, c=colour)
    axs[0].plot([distance_to_xmax_mc.min() / 1e3, distance_to_xmax_mc.max() / 1e3], [
        distance_to_xmax_mc.min() / 1e3, distance_to_xmax_mc.max() / 1e3], "k--", lw=3, zorder=10, label=r"$d_\mathrm{max}^\text{MC}$")
    axs[0].set_ylabel(r"$d_\mathrm{max}$ [km]")

    if 1:
        yvar = dmax_fit / distance_to_xmax_mc
        axs[1].set_ylabel(
            r"$d_\mathrm{max} / d_\mathrm{max}^\mathrm{MC}$")

        axs[1].errorbar(
            distance_to_xmax_mc / 1e3, yvar, dmax_fit_err / distance_to_xmax_mc, c=colour,
            marker="o", ls="", alpha=0.1,
            label=r"$\mu$ = %.3f, $\sigma$ = %.3f" % (np.nanmean(yvar), np.nanstd(yvar)))
        axs[1].set_ylim(0.25, 1.75)
        axs[1].axhline(1, c="k", ls="--", lw=3, zorder=10)

    else:
        yvar = (dmax_fit - distance_to_xmax_mc) / dmax_fit_err
        axs[1].set_ylabel(
            r"($d_\mathrm{max}^\mathrm{fit} - d_\mathrm{max}^\mathrm{MC}$) / $\sigma_{d_\mathrm{max}^\mathrm{fit}}$")

        axs[1].plot(distance_to_xmax_mc / 1e3, yvar, "o", alpha=0.1,
                    label=r"$\mu$ = %.3f, $\sigma$ = %.3f" %
                    (np.nanmean(yvar), np.nanstd(yvar)))
        axs[1].axhline(0, c="k", ls="--", zorder=10)

    if 1:
        n, xcen, y_mean_binned, y_std_binned, edges = \
            helperstats.get_binned_data(
                distance_to_xmax_mc / 1e3, yvar,
                10, skip_empty_bins=False, return_bins=True)

        xerr = np.array([xcen - edges[:-1], edges[1:] - xcen])
        mask = n > 5
        axs[1].errorbar(xcen[mask], y_mean_binned[mask], y_std_binned[mask],
                        xerr[:, mask], color="red", marker="s", ls="", markersize=8, lw=3, zorder=3)

    axs[1].set_xlabel(r"$d_\mathrm{max}^\mathrm{MC}$ [km]")
    axs[0].legend()
    axs[1].legend()
    [ax.grid() for ax in axs]
    fig_dmax.tight_layout()
    fig_dmax.savefig("fit_result_dmax%s.png" % para.label)
    plt.close()
    
    
    # plot of xmax recon resolution
    if 1: # para.realistic_input:
        bins = np.linspace(-500, 500, 60)
        res_mean = np.mean(xmax_dev)
        res_width = np.std(xmax_dev)
        
        print("Outliers: ", np.sum((abs(xmax_dev) > 500)))

        xmax_dev = xmax_dev[abs(xmax_dev) < 500]

        fig_xmax_recon = plt.figure(figsize=(11, 9))
        plt.title(r'Quality of $X_\mathrm{max}$ reconstruction', fontsize=30)
        # histogram 
        plt.hist(xmax_dev, bins=bins, histtype='step', linewidth=4, color='dodgerblue', \
                label=(r"$\mu =$ %.2f; " + r"$\sigma =$ %.2f") % (res_mean, res_width))
        
        plt.xlabel(r"$X_\mathrm{max}^\mathrm{MC}$ - $X_\mathrm{max}^\mathrm{rec}$ [g/cm²]")
        plt.ylabel("# of Events")
        plt.xlim(-500, 500)
        plt.ylim(None, 500)
        plt.legend(fontsize=30-5, loc="upper left")
        plt.savefig("xmax_recon_perf.png")
        plt.close()



def evaluate_xmax_fit(events, para):
    events = event_selection_ldf_validation(events, 
                                   high_zenith_cut=True,
                                   mc_xmax_cut=True,
                                   station_number_cut=5,
                                   dmax_error_cut=True,
                                   egeo_error_cut=True,
                                   large_dmax_cut=True)

    xmax_mc = factory.get_parameter(
        events, shp.xmax)
    zenith_mc = factory.get_parameter(events, shp.zenith)
    obs_level = factory.get_parameter(events, shp.observation_level)

    at = atm.Atmosphere(27)

    distance_to_xmax_mc = factory.get_parameter(
        events, shp.distance_to_shower_maximum_geometric)

    dmax_fit, dmax_fit_err = factory.get_parameter_and_error(
        events, shp.distance_to_shower_maximum_geometric_fit)

    xmax_fit = np.array([at.get_xmax_from_distance(d, z, o) for d, z, o in zip(dmax_fit, zenith_mc, obs_level)])

    fig_dmax, ax = plt.subplots(1, sharex=True)


    yvar = xmax_fit - xmax_mc
    ax.set_ylabel(
        r"$X_\mathrm{max}^\mathrm{fit} - X_\mathrm{max}^\mathrm{MC}$")

    ax.errorbar(xmax_mc, yvar, 0, marker="o", ls="", alpha=0.3,
                    label=r"$\mu$ = %.3f, $\sigma$ = %.3f" %
                    (np.nanmean(yvar), np.nanstd(yvar)))
    ax.axhline(1, c="k", ls="--", zorder=10)

    if 1:
        n, xcen, y_mean_binned, y_std_binned, edges = \
            helperstats.get_binned_data(
                xmax_mc, yvar,
                10, skip_empty_bins=False, return_bins=True)

        xerr = np.array([xcen - edges[:-1], edges[1:] - xcen])
        mask = n > 5
        ax.errorbar(xcen[mask], y_mean_binned[mask], y_std_binned[mask],
                        xerr[:, mask], color="C3", marker="o", ls="", zorder=3)

    ax.set_xlabel(r"$X_\mathrm{max}^\mathrm{MC}$ [km]")
    ax.legend()
    ax.grid()
    fig_dmax.tight_layout()

    if not para.save:
        plt.show()
    else:
        try:
            fig_dmax.savefig("fit_result_xmax%s.png" % para.label)
        except:
            pass


def evaluate_egeo(events, para):
    print("\n" + "Start of Egeo evaluation: ")
    events = event_selection_ldf_validation(events,
                                   high_zenith_cut=True,
                                   mc_xmax_cut=True,
                                   has_any_stations=True, 
                                   direction_error_cut=True, 
                                   station_number_cut_SNR_cut=5,
                                   station_number_cut_LDF_fit=5,
                                   has_fit_parameters=True,
                                   has_successful_fit=True,
                                   cherenkov_radius_cut=1,
                                   chi_percentile_cut=False,
                                   dmax_error_cut=True,
                                   egeo_error_cut=True,
                                   large_dmax_cut=True, 
                                   saturation_cut=False
                                   )

    geo_ldf_params_fit, geo_ldf_params_fit_err = factory.get_parameter_and_error(
        events, shp.geomagnetic_ldf_parameter)
    egeo_fit = np.array([x['E_geo'] for x in geo_ldf_params_fit])
    egeo_fit_err = np.array([x['E_geo'] for x in geo_ldf_params_fit_err])

    geomag_angle_mc = np.array(
        [ev.get_geomagnetic_angle() for ev in events])
        
    rho_mc = factory.get_parameter(
        events, shp.density_at_shower_maximum)
    
    eem_mc = factory.get_parameter(events, shp.electromagnetic_energy)
    egeo_mc = energyreconstruction.get_Egeo(eem_mc, np.sin(geomag_angle_mc), rho_mc)

    fig, axs = plt.subplots(1, 2)

    q68 = np.quantile(egeo_fit_err / egeo_fit, 0.68)
    label = r"$n_\mathrm{over}$ = %d" % np.sum(egeo_fit_err / egeo_fit > 0.1) + "\n" \
        + r"$\sigma_{68} = %.3f$" % q68

    axs[0].hist(egeo_fit_err / egeo_fit, np.linspace(0, 0.1), lw=3,
                facecolor=(137 / 255, 196 / 255, 244 / 255, 1), label=label)
    axs[0].set_xlabel(r"$\sigma_{E_\mathrm{geo}} / E_\mathrm{geo}$")
    axs[0].legend(fontsize="small")
    
    q68 = np.quantile(np.abs(egeo_fit / egeo_mc) - 1, 0.68)
    label = r"$n_\mathrm{over}$ = %d" % np.sum(np.abs(egeo_fit / egeo_mc) - 1 > 0.5) + "\n" \
        + r"$\sigma_{68} = %.3f$" % q68

    axs[1].hist(np.abs(egeo_fit / egeo_mc) - 1, np.linspace(0, 3), lw=3,
                facecolor=(137 / 255, 196 / 255, 244 / 255, 1), label=label)
    axs[1].set_xlabel(r"$|E_\mathrm{geo} / E_\mathrm{geo}^\mathrm{MC} - 1 |$")
    axs[1].legend(fontsize="small")  
    
    fig.tight_layout()
    plt.savefig("energy-eval.png")
    # plt.show()
    plt.close()


def print_magnetic_field_variation(events, para):

    magnetic_field_vectors = np.array(
        [ev.get_parameter(evp.magnetic_field_vector) for ev in events])

    strength = np.linalg.norm(magnetic_field_vectors, axis=1)
    inclinations = np.array([helper.get_inclination(x) for x in magnetic_field_vectors])
    print(np.mean(strength), np.std(strength))
    print(np.mean(np.rad2deg(inclinations)), np.std(np.rad2deg(inclinations)))



def evaluate_fit_result(events, para):

    # set global font size
    plt.rcParams.update({'font.size': 30})
    # pre selection of events for energy reconstruction plots
    # for parameters like zenith angle and energy range 
    # and not fit dependent parameters
    compare_events = events

    print("\n" + "Preselection of events!")
    events = event_selection_ldf_validation(events, has_any_stations=False,
                                                    high_zenith_cut=True,
                                                    low_geomagnetic_angle=False, #True,
                                                    energy_range_cut=False,
                                                    has_fit_parameters=False,
                                                    has_successful_fit=False,
                                                    only_proton=False, 
                                                    only_iron=False
                                                    )

    full_events = events

    print("\n" + "Start of general fit evaluation!")

    # events = remove_dublicated_events(events)
    
    not_stshp_sims = True

    # selection for events for fit dependent characters!
    events = event_selection_ldf_validation(events,
                                   mc_xmax_cut=True,
                                   has_any_stations=True, 
                                   direction_error_cut=True, 
                                   station_number_cut_SNR_cut=5,
                                   station_number_cut_LDF_fit=5,
                                   has_fit_parameters=True,
                                   has_successful_fit=True,
                                   cherenkov_radius_cut=1,
                                   chi_percentile_cut=False,
                                   dmax_error_cut=not_stshp_sims,
                                   egeo_error_cut=not_stshp_sims,
                                   large_dmax_cut=not_stshp_sims, 
                                   saturation_cut=False
                                   )

    # shower parameters
    obs_level = factory.get_parameter(events, shp.observation_level)
    atmodels = factory.get_parameter(events, shp.atmosphere_model)

    # mc parameters
    xmax_mc = factory.get_parameter(events, shp.xmax)
    zenith_mc = factory.get_parameter(events, shp.zenith)
    eem_mc = factory.get_parameter(events, shp.electromagnetic_energy)
    geomag_angle_mc = np.array([ev.get_geomagnetic_angle() for ev in events])
        

    # parameters that are either MC or calculated from reconstructed quantities
    # depending on whether realistic_input was used for previous calculations
    rho_mc = factory.get_parameter(events, shp.density_at_shower_maximum)
    
    print("Mean of average density: ", np.mean(rho_mc))
    print("Average density for shower with 75° and 750g/cm2: ", 
          radiationenergy.get_average_density(41, np.deg2rad(75), 750))
    
    # reconstructed parameters
    # truth value of realistic_input determines whether MC or reconstructed values are used for calculations
    if para.realistic_input:
        zenith_recon = factory.get_parameter(events, shp.zenith_recon)
        geomag_angle_recon = factory.get_parameter(events, shp.geomagnetic_angle_recon)

        zenith_rec_or_mc = zenith_recon
        geomag_angle_rec_or_mc = geomag_angle_recon

    else:
        zenith_rec_or_mc = zenith_mc
        geomag_angle_rec_or_mc = geomag_angle_mc

    #
    # fit parameters
    #

    geo_ldf_params_fit, geo_ldf_params_fit_err = factory.get_parameter_and_error(events, shp.geomagnetic_ldf_parameter)
    egeo_fit = np.array([x['E_geo'] for x in geo_ldf_params_fit])
    egeo_fit_err = np.array([x['E_geo'] for x in geo_ldf_params_fit_err])

    for key in geo_ldf_params_fit_err[0]:
        err = np.array([x[key] for x in geo_ldf_params_fit_err])
        val = np.array([x[key] for x in geo_ldf_params_fit])

    dmax_fit, dmax_fit_err = factory.get_parameter_and_error(events, shp.distance_to_shower_maximum_geometric_fit)

    # fluence values and fitted values for pull plot
    f_vxB= np.array(factory.get_parameter(events, stp.vxB_fluence_simulated))
    f_vxB_error = np.array(factory.get_parameter_error(events, stp.vxB_fluence_simulated))
    f_vxB_model = np.array(factory.get_parameter(events, stp.vxB_fluence_model))

    # take fitted density if it was fitted
    if "density_at_xmax" in geo_ldf_params_fit[0]:
        rho_fit = np.array([x['density_at_xmax'] for x in geo_ldf_params_fit])
    else:
        rho_fit = np.zeros_like(rho_mc)
        for idx, (d, z, obs, model) in enumerate(zip(dmax_fit, zenith_rec_or_mc, obs_level, atmodels)):
            h = atm.get_height_above_ground(d, z, obs) + obs
            rho_fit[idx] = atm.get_density(h, model=model) * 1e-3

    # avg_rho = 0.3113839703192573 # for Auger
    avg_rho = 0.2171571758825658 # for GRAND

    
    run_number = np.array([ev.get_run_number() for ev in events])
    nr, nu = np.unique(run_number, return_counts=True)
    # print(np.unique(nu, return_counts=True), nr[nu > 1])

    # whether you want to use the rho calculated with the dmax from in geometry (which can be an estimate depending on realistic input)
    # or the rho calculated with the dmax from the fit
    if 0: # para.realistic_input:
        lr = r"\rho_\mathrm{max}"
        rho_fit_or_mc = rho_fit
    else:
        lr = r"\rho^\mathrm{MC}_\mathrm{max}"
        rho_fit_or_mc = rho_mc
    
    eem_rec_label = r"E_\mathrm{em}(S_\mathrm{geo})"

    # data to feed into energy reconstruction functions
    xdata = [egeo_fit, np.sin(geomag_angle_rec_or_mc), rho_fit_or_mc]
    
    ref_energy = 10 ** 19
    
    # Auger 50-200 
    # x0 = {"s19": 5.6059195e+09, "gamma": 1.9930619,"p0": 0.8577229, "p1": -1.4530023, "p2": 0, "p3": 0, "a_exp": 1.7511544}
    
    # Dunhuang 50-200 # 13.0725662e9 # for GP300  # 14.6359355e9  # for starshapes   13.4858593  # for GRAND10k
    x0 = {"s19": 13.4858593e9, "gamma": 1.9961499,"p0": 196.5323011, "p1": -0.0030236, "p2": 0.0338899, "p3": 0.1928640, "a_exp": 0}# 1.1631093}
    
    print("x0", x0)

    def cost(s19, gamma, p0, p1, p2, p3, a_exp):
        egeo_pred = energyreconstruction.get_Egeo(
            eem_mc, np.sin(geomag_angle_rec_or_mc), rho_fit_or_mc, s19, gamma,
            p0, p1, p2, p3, a_exp, rho_avg=avg_rho, ref_energy=ref_energy)

        chi = (egeo_fit - egeo_pred) / egeo_fit_err
        return np.sum(chi ** 2)

    m = iminuit.Minuit(cost, **x0)
    m.errordef = 1
    # 0 = fitted values; 1 = fixed values
    # m.fixed = [0, 0, 0, 0, 1, 1, 0]
    m.fixed = [1, 1, 1, 1, 1, 1, 1]
    # Auger
    # m.limits = [[1e9, 1e10], [1,3], [-10,20], [-10,10], [-0.0001, 0.0001], [-0.0001,0.0001 ], [1, 3]]
    # GRAND
    m.limits = [[0.8e9, 20e9], [x0["gamma"] * 0.90, x0["gamma"] * 1.1], [-1000,1000], [-10,10], [0, 3], [0,0.5], [1, 3]]
    m.migrad()
    m.hesse()
    print(m) # print the fitted values for the density correction

    popt = np.array(m.values)
    uncert = np.array(m.errors)
    # print("popt", popt, "uncert", uncert)
    pout = "S19 = %.7f GeV, gamma = %.7f, p0 = %.7f, p1 = %.7f, p2 = %.7f, p3 =%.7f, c_alpha = %.7f" % (popt[0] / 1e9, popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
    print(pout)
    eem_pred, eem_err = energyreconstruction.get_Eem_from_Egeo(
        *xdata, *popt, egeo_err=egeo_fit_err, rho_avg=avg_rho, ref_energy=ref_energy)

    srad, srad_err = energyreconstruction.get_Sgeo(
        *xdata, *popt[2:], rho_avg=avg_rho, egeo_err=egeo_fit_err)
    



    '''
    ---------------------------------------------START OF PLOT SECTION ------------------------------------------------------
    '''
    
    if 1:
        # eem - Ecr violin plot
        add_zenith = True
        fontsize = 40
        
        # green for RdStar
        # colourscheme = ["viridis", "mediumseagreen", "darkblue"]
        
        # orange for GRAND Star-shapes
        # colourscheme = ["plasma", "red", "darkmagenta"]
        
        # blue for GP300
        # colourscheme = [cmr.cosmic, "dodgerblue", "navy"]
        
        # purple for GRAND10k
        colourscheme = [cmr.bubblegum, "fuchsia" , "purple"]
        
        # plot title
        # title = "GP300 | Simulated Performance"
        title = "GRAND10k | Simulated Performance"
        
        if 1:
            fig_eng = plt.figure(figsize=(18, 15)) # 18, 15
            fig_eng.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.95)
            ax1 = fig_eng.add_axes((.1, .1, .35, .825))
            ax_res = fig_eng.add_axes((.55, .565, .4, .36))
            ax_res2 = fig_eng.add_axes((.55, .1, .4, .36))

        # if add_zenith:
        #     fig_eng = plt.figure(figsize=(20, 10)) # 18, 15
        #     fig_eng.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.95)
        #     ax1 = fig_eng.add_axes((.1, .15, .275, .825)) # ax1 = fig_eng.add_axes((.1, .1, .275, .825))
        #     ax2 = fig_eng.add_axes((.49, .5, .235, .425))
        #     ax_res = fig_eng.add_axes((.49, .1, .235, .37))
        #     ax3 = fig_eng.add_axes((.75, .5, .235, .425))
        #     ax_res2 = fig_eng.add_axes((.75, .1, .235, .37))
        # else:
        #     fig_eng = plt.figure(figsize=(24, 15))
        #     fig_eng.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.95)
        #     ax1 = fig_eng.add_axes((.125, .1, .35, .84))
        #     ax2 = fig_eng.add_axes((.625, .525, .35, .415))
        #     ax_res = fig_eng.add_axes((.625, .1, .35, .415), sharex=ax2)

        if 0: # whether to plot the diagonal as a 2D histogram or not
            # plot power law
            label = r"E_\mathrm{radio}-E_\mathrm{em} correlation" # (r"$S_{19}$ = %.2f GeV" + "\n" + r"$\gamma$ = %.0f") % (popt[0] / 1e9, popt[1])
            
            # prediction for srad
            srad_pred = popt[0] * (np.array([eem_mc.min(), eem_mc.max()]) / ref_energy) ** popt[1]
            
            ax1.plot([eem_mc.min(), eem_mc.max()], srad_pred, "k--", lw=3, zorder=10, label=label)
            ax1.set_ylabel(r"$S_\mathrm{geo}$ [eV]", fontsize=fontsize)
            
            bins_eem = np.logspace(17, 21, 50)
            bins_sgeo = np.logspace(6, 14, 75)
            
            # 2D histogram of the data
            # x and y edges are the bins, z is the amount of entries in each bin in arrays representing the vertical columns
            bin_freq, xedges, yedges = np.histogram2d(
                eem_mc, srad, bins=[bins_eem, bins_sgeo])
            
            # print(eem_mc)
            # print(srad)
            
            # print(xedges)
            # print(yedges)
            # print(bin_freq)

            # to include the highest value in the bin
            yedges[-1] += 1e-6
            # get center values
            center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

            bins_eem[-1] += 1e-6
            bin_centers = bins_eem[:-1] + (bins_eem[1:] - bins_eem[:-1]) / 2

            # calculate mean of each colum
            mean_pos = np.array(
                [np.sum(
                    np.array(
                        [center_values[j] * bin_freq[i][j]
                        for j in range(len(bin_freq[i]))])) / np.sum(bin_freq[i])
                for i in range(len(bin_freq))])
            # print(mean_pos)

            # calculate standard deviation
            std_dev = np.sqrt(
                np.array(
                    [np.sum(
                        np.array(
                            [bin_freq[i][j] * (center_values[j] - mean_pos[i]) ** 2
                            for j in range(len(bin_freq[i]))])) /
                    (np.sum(bin_freq[i]) - 1) for i in range(len(bin_freq))]))

            # add means and standard deviations to the 2D histograms as points and errorbars
            # ax1.errorbar(bin_centers, mean_pos, std_dev, marker='o', color='r', linestyle='', ms=4, label='')

            # transpone z
            bin_freq = bin_freq.T

            # assign colour values to the bins
            pcm = ax1.pcolormesh(
                xedges, yedges, bin_freq, shading='flat', # norm=mpl.colors.LogNorm(),
                cmap=colourscheme[0]) # "viridis_r"

            # cbi = plt.colorbar(pcm, orientation='vertical', pad=0.01, ax=ax1, fraction=0.30)
            # cbi.ax.tick_params(axis='both', labelsize=fontsize-15)
            # cbi.set_label('# of events', fontsize=fontsize-10)
            
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="4%", pad=0.04)
            colorbar = plt.colorbar(pcm, cax=cax)
            colorbar.ax.tick_params(axis='both', labelsize=fontsize-12)
            colorbar.set_label(size=fontsize, label='# of events')
            
        else:
            label = r"True $E_\mathrm{radio}-E_\mathrm{em}$" + "\n" + "correlation" # (r"$S_{19}$ = %.2f GeV" + "\n" + r"$\gamma$ = %.4f"\
            #          ) % \
            #           (popt[0] / 1e9, popt[1])
            #  + "\n" + r"$\langle \rho_\mathrm{max} \rangle$ = %.3f kg$\,$m$^{-3}$") 
            ax1.errorbar(eem_mc, srad, srad_err, marker="o", ls="", alpha=0.3, c=colourscheme[1])
            srad_pred = popt[0] * (np.array([eem_mc.min() * 0.9, eem_mc.max() ]) / ref_energy) ** popt[1]

            ax1.plot([eem_mc.min() * 0.9, eem_mc.max()], srad_pred, "k--", lw=3, zorder=10, label=label)
            ax1.set_ylabel(r"$S_\mathrm{geo}$ [eV]", fontsize=fontsize)
            ax1.set_ylabel(r"Corrected Radio Energy $E_\text{radio}$ [eV]", fontsize=fontsize)

        ax1.set_xticks([1e17, 1e18, 1e19, 1e20])
        ax1.set_xlim(eem_mc.min() * 0.9, 1e20+1) # eem_mc.min() * 0.9, eem_mc.max() * 1.1)
        ax1.set_ylim(srad.min(), srad.max()) # 1e6
        # ax1.set_xlabel(r"$E_\mathrm{em}^\mathrm{MC}$ [eV]", fontsize=fontsize)
        ax1.set_xlabel(r"True $E_\mathrm{em}$ [eV]", fontsize=fontsize)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid()
        ax1.legend(loc="upper left", fontsize=fontsize-10)
        ax1.tick_params(labelsize=fontsize-8)

        energies_bins_center, energies_mask, energy_edges = get_data_bins(
            np.log10(eem_mc), bins=10, return_bins=True)
        eem_rec_ratio_energy = np.array(
            [eem_pred[mask] / eem_mc[mask] for mask in energies_mask], dtype=object)
        # print(eem_rec_ratio_energy)
        mask = np.sum(energies_mask, axis=1) > 5

        # ax2 = pyplots.violinplot(eem_rec_ratio_energy[mask], energies_bins_center[mask], spread=0.12,  # 0.225 for sparse
        #                         ax1=ax2, add_projection=False, save=False, ylim=(0.87, 1.13), grid=True, colours=colourscheme[1:3])
        # ax2.tick_params(axis="x", labelsize=0)
        # ax2.set_ylabel(r'$%s$ / $E_\mathrm{em}^\mathrm{MC}$' % eem_rec_label, fontsize=fontsize)
        # ax2.xaxis.set_major_locator(MaxNLocator(5)) 
        # ax2.set_yticks([0.9, 0.95, 1, 1.05, 1.1])

        
        n = np.array([len(x) for x in eem_rec_ratio_energy[mask]])
        std = np.array([np.std(x) for x in eem_rec_ratio_energy[mask]])
        mean = np.array([np.mean(x) for x in eem_rec_ratio_energy[mask]])
        res = std * 100
        res_err = std / np.sqrt(2 * (n - 1)) * 100
        bias = (mean - 1) * 100
        bias_err = std / np.sqrt(n) * 100

        ax_res.errorbar(energies_bins_center[mask], res, res_err,
            marker='o', markersize=10, lw=3, ls="", label='Resolution', c=colourscheme[1], capsize=5)
        ax_res.errorbar(energies_bins_center[mask], bias, bias_err,
            marker='v', markersize=10, lw=3, ls="", label='Bias', c=colourscheme[2], capsize=5)
        ax_res.set_ylabel(r'$\sigma$, $\mu - 1$  [$\%$]', fontsize=fontsize)
        ax_res2.set_ylabel(r'$\sigma$, $\mu - 1$  [$\%$]', fontsize=fontsize)
        ax_res.set_xlabel(
            r"log$_{10}(E_\mathrm{em}^\mathrm{MC}$ / eV)", fontsize=fontsize)
        ax_res.set_xlabel(
            r"log$_{10}(E_\mathrm{em}$ / eV)", fontsize=fontsize)
        
        # set ylim of res plots according to max value
        if max(res) > 30 or max(bias) > 30:
            ax_res.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
            ax_res.set_ylim(-4, 75)
            
        elif max(res) > 12.5 or max(bias) > 12.5:
            ax_res.set_yticks([0, 5, 10, 15, 20])
            ax_res.set_ylim(-4, 18)
            
        elif max(res) > 10 or max(bias) > 10:
            ax_res.set_yticks([- 2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 20])
            ax_res.set_ylim(-3, 15)
        
        elif max(res) > 6 or max(bias) > 6:
            ax_res.set_yticks([-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
            ax_res.set_ylim(-3, 12)
        
        else:
            ax_res.set_yticks([-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
            ax_res.set_ylim(-1.5, 7.5)
            
        print("Resolution: ", res, np.mean(res))
        print("Bias: ", bias, np.mean(bias))

        ax_res.set_xticks([17, 18, 19, 20]) # ax2.get_xticks())
        ax_res.grid()
        ax_res.legend(loc='upper right', fontsize=fontsize-12)

        # x1, x2 = ax1.get_xlim()
        ax_res.tick_params(axis="y", labelsize=fontsize-10)
        ax_res.set_xticks([17, 18, 19, 20])
        # ax_res.set_xlim(np.log10(x1), np.log10(x2))
        # ax2.set_xlim(np.log10(x1), np.log10(x2))

        
        x1, x2 = ax1.get_xlim()
        ax_res.set_xlim(17, 20)
        ax_res.tick_params(labelsize=fontsize-10)
        run_number = np.array([ev.get_run_number() for ev in events])
        fig_eng.suptitle(title, fontsize=fontsize+5)
        fig_eng.tight_layout()

        if add_zenith:

            xval = np.rad2deg(zenith_mc)
            xlabel = r"Zenith angle [°]"

            bins = 10 # np.arange(65, 85.01, 4) # 10
            # if len(np.unique(xval)) < 100:
            #     bins = np.arange(65, 85.01, 2)

            bc, masks, edges = get_data_bins(
                xval, bins=bins, return_bins=True)
            
            eem_rec_ratio_energy = np.array(
                [eem_pred[mask] / eem_mc[mask] for mask in masks], dtype=object)
            mask = np.sum(masks, axis=1) > 5

            # ax3 = pyplots.violinplot(
            #     eem_rec_ratio_energy[mask], bc[mask], spread=0.7, ylabel=None,  # eem_spread, 
            #     ax1=ax3, add_projection=False, save=False, ylim=(0.87, 1.13), grid=True, colours=colourscheme[1:3])
            # ax3.xaxis.set_major_locator(MaxNLocator(5)) 
            # ax3.set_yticks([0.9, 0.95, 1, 1.05, 1.1])

            # ax3.tick_params(axis="both", labelsize=0)
            # ax3.set_ylim(ax2.get_ylim())

            n = np.array([len(x) for x in eem_rec_ratio_energy[mask]])
            std = np.array([np.std(x) for x in eem_rec_ratio_energy[mask]])
            mean = np.array([np.mean(x) for x in eem_rec_ratio_energy[mask]])
            res = std * 100
            res_err = std / np.sqrt(2 * (n - 1)) * 100
            bias = (mean - 1) * 100
            bias_err = std / np.sqrt(n) * 100

            ax_res2.errorbar(bc[mask], res, res_err,
                marker='o', markersize=10, lw=3, ls="", label='Resolution', c=colourscheme[1], capsize=5)
            ax_res2.errorbar(bc[mask], bias, bias_err,
                marker='v', markersize=10, lw=3, ls="", label='Bias', c=colourscheme[2], capsize=5)
        
            ax_res2.set_xlabel(xlabel, fontsize=fontsize-1)

            # ax3.set_xlim(64, 85.2)
            # plthelpers.draw_zenith_angles_on_top_xaxis(ax3, xval="sin2", theta=np.deg2rad(
            #    [68, 70, 72, 75, 80, 85]), grid={"b": False}, fontsize=fontsize-10)

            ax_res2.set_yticks(ax_res.get_yticks())
            ax_res2.set_ylim(ax_res.get_ylim())
            ax_res2.set_xlim(65, 85)
            ax_res2.tick_params(axis="y", labelsize=fontsize-10)
            ax_res2.tick_params(axis="x", labelsize=fontsize-10)
            # ax_res2.set_xticks(ax3.get_xticks())
            ax_res2.grid()
            ax_res2.legend(loc='upper right', fontsize=fontsize-12)
            # ax_res2.set_xlim(ax3.get_xlim())
            
            
    # evaluation of arrival direction reconstruction
    if para.realistic_input:
        bins_zenith = np.linspace(-1.5, 1.5, 25)
        bins_azimuth = np.linspace(-1.5, 1.5, 25)
        
        print("Selection for arrival direction evaluation:")
        # selection for events for fit dependent characters!
        arr_events = event_selection_ldf_validation(compare_events,
                                   has_any_stations=True,
                                   direction_error_cut=True, 
                                   station_number_cut_SNR_cut=5,
                                   has_fit_parameters=False,
                                   has_successful_fit=False,
                                   )

        zenith_true = factory.get_parameter(arr_events, shp.zenith)
        zenith_reconstructed = factory.get_parameter(arr_events, shp.zenith_recon)

        # calculate deviation of reconstructed from MC zenith
        zenith_diff = np.rad2deg(zenith_reconstructed - zenith_true)


        azimuth_true = factory.get_parameter(arr_events, shp.azimuth)
        azimuth_reconstructed = factory.get_parameter(arr_events, shp.azimuth_recon)

         # calculate deviation of reconstructed from MC azimuth
        azimuth_diff = np.rad2deg(azimuth_reconstructed - azimuth_true)
        
        out_zenith = np.abs(zenith_diff) > 1.5
        out_azimuth = np.abs(azimuth_diff) > 1.5

        mask_zen = np.abs(zenith_diff) < 1.5
        mask_azi = np.abs(azimuth_diff) < 1.5
        print(f"Cut zenith outliers: {np.sum(out_zenith)} / {len(out_zenith)}")
        print(f"Cut azimuth outliers: {np.sum(out_azimuth)} / {len(out_azimuth)}")
        
        zenith_diff = zenith_diff[mask_zen]
        azimuth_diff = azimuth_diff[mask_azi]
        
        zenith_mean = np.mean(zenith_diff)
        zenith_width = np.std(zenith_diff)

        azimuth_mean = np.mean(azimuth_diff)
        azimuth_width = np.std(azimuth_diff)
        
        np.savez_compressed(f"arrival_dir_10k_adc", zenith_diff, azimuth_diff)
        
        font=40

        # make the plot
        fig_arrival, (axis1, axis2) = plt.subplots(1, 2, figsize=(26, 15), sharex=False, sharey=True)

        # histogram for zenith reconstruction
        # fig_arrival.suptitle(r'Histogram of ratio between reconstructed and MC arrival direction')
        # histogram of fit pull values
        axis1.hist(zenith_diff, bins=bins_zenith, histtype='step', linewidth=4, color=colourscheme[1], \
                label=(r"$\mu =$ %.4f; " + r"$\sigma =$ %.4f") % (zenith_mean, zenith_width))
        axis1.set_xlabel(r"Deviation $\theta_\mathrm{rec} - \theta_\mathrm{MC}$ [°]", fontsize=font)
        axis1.set_ylabel("# of Events", fontsize=font)

        # histogram for azimuth reconstruction
        axis2.hist(azimuth_diff, bins=bins_azimuth, histtype='step', linewidth=4, color=colourscheme[2], \
                label=(r"$\mu =$ %.4f; " + r"$\sigma =$ %.4f") % (azimuth_mean, azimuth_width))
        axis2.set_xlabel(r"Deviation $\phi_\mathrm{rec} - \phi_\mathrm{MC}$ [°]", fontsize=font)
        axis2.set_ylabel("# of Events", fontsize=font)

        axis1.legend(loc="upper right", fontsize=font-10)
        axis2.legend(loc="upper right", fontsize=font-10)
        # axis1.set_yscale("log")
        axis1.set_ylim(None, 4600)
        axis1.tick_params(axis="both", labelsize=font-5)
        axis2.tick_params(axis="both", labelsize=font-5)
        
        fig_arrival.savefig("arrival_direction_eval_hist.png")
        plt.close()


    if 1:
        # histogram of LDF fit pull values

        font = 40

        # define pull value
        fluence_pull = np.concatenate(np.array((f_vxB - f_vxB_model) / f_vxB_error))
        pull_mean = np.mean(fluence_pull)
        pull_width = np.std(fluence_pull)
        
        # np.savez_compressed(f"LDF_fit_pull_gp300_adc", fluence_pull)
        
        pull_data_file = np.load('/cr/users/guelzow/simulations/radiominimalysis/ldf_eval/LDF_fit_pull_gp300_adc.npz')
        pull_data = pull_data_file['arr_0']
        pull_mean_data = np.mean(pull_data)
        pull_width_data = np.std(pull_data)
        

        # generate Normalverteilung to compare and calculate mean StD
        normal = np.random.normal(size=len(fluence_pull))
        gauss_mean = np.mean(normal)
        gauss_width = np.std(normal)
        # define bins
        bins=np.linspace(-5, 5, 75)

        fig_pull_LDF_fit, axx = plt.subplots(1, figsize=(24, 12))
       
        axx.fill_between(np.linspace(0, 1, 1), np.linspace(0, 1, 1), np.linspace(0, 1, 1), alpha=0.5, color="white", label=r"$\frac{f_\mathrm{geo}^{\,\mathrm{par}} - f_\mathrm{geo}^{\,\mathrm{LDF}}}{\sigma (f_\mathrm{geo}^{\,\mathrm{LDF}})}$")
        
        # histogram of fit pull values
        axx.hist(fluence_pull, bins=bins, histtype='stepfilled', linewidth=4, color='dodgerblue', alpha=0.7, \
                label=(r"$\mu =$ %.2f; " + r"$\sigma =$ %.2f") % (pull_mean, pull_width))
        
        
        if 1:
            # generate Normalverteilung to compare and calculate mean StD
            normal_data = np.random.normal(size=len(pull_data))

            # histogram of fit pull values
            axx.hist(pull_data, bins=bins, histtype='stepfilled', linewidth=4, color='purple', alpha=0.7, \
                    label=(r"$\mu =$ %.2f; " + r"$\sigma =$ %.2f") % (pull_mean_data, pull_width_data))
            
            # histogram of Gaussian data
            axx.hist(normal_data, bins=bins, histtype='step', linewidth=3, linestyle=':', color='black',)
            
            
        # histogram of Gaussian data
        axx.hist(normal, bins=bins, histtype='step', linewidth=3, linestyle=':', color='black', \
                label=("Normal distribution" + "\n" + r"$\mu =$ %.2f; " + r"$\sigma =$ %.2f") % (gauss_mean, gauss_width))
        
        axx.set_xlabel(r"Pull of $f_\mathrm{geo}^{\,\mathrm{par}}$ to LDF fit" , fontsize=font)
        axx.set_ylabel("# of antennas", fontsize=font)
        axx.tick_params(axis="both", labelsize=font-10)
        axx.set_xlim(-5, 5)
        # axx.set_yscale("log")
        axx.legend(fontsize=font-15)
        fig_pull_LDF_fit.savefig("LDF_fit_pull.png")
        plt.close()
        

    # angle dependent efficiency plot
    if 1:

        bins = (np.linspace(17, 20, 16), np.linspace(65, 85, 16))

        # full events are all except events above 85°
        zenith_full = np.rad2deg(factory.get_parameter(full_events, shp.zenith))
        zenith_test = np.rad2deg(factory.get_parameter(events, shp.zenith))
        energy_full = factory.get_parameter(full_events, shp.electromagnetic_energy)
        energy_test = factory.get_parameter(events, shp.electromagnetic_energy)

        values, xbins1, ybins1 = np.histogram2d(np.log10(energy_test), zenith_test, bins=bins)
        full_values, xbins2, ybins2 = np.histogram2d(np.log10(energy_full), zenith_full, bins=bins)

        if (values / full_values).any() > 1:
            print("Ratios over 1, something is wrong!")

        fig_angles, ax_angles = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

        plt.pcolormesh(xbins2, ybins2, np.transpose(values / full_values), cmap=colourscheme[0], vmin=0, vmax=1)
        ax_col = plt.colorbar(label="Fraction of successful reconstructions")# , fontsize=20)

        ax_angles.set_xticks([17, 17.5, 18, 18.5, 19, 19.5, 20])
        ax_angles.set_yticks([65, 70, 75, 80, 85])
        ax_angles.tick_params(labelsize=18)
        ax_angles.set_ylabel("Zenith Angle [°]", fontsize=20)
        ax_angles.set_xlabel(r"$\mathrm{log}_{10}(E_\mathrm{em}^\mathrm{MC}$ / eV)", fontsize=20)
        # fig_angles.suptitle(title, fontsize=25)
        ax_angles.set_xlim(17, 20)
        fig_angles.tight_layout()
        fig_angles.savefig("angles_efficiency.png")
        plt.close()


    # evaluation of energy fluence from efield reconstruction
    if compare_events[0].has_station_parameter(stp.fluence_compare_MC):
        
        font=45
        
        cbar_label = "# of antennas"

        print("Selection for fluence comparison!")
        events_fluence = event_selection_ldf_validation(compare_events, high_zenith_cut=True,
                                                    energy_range_cut=False,
                                                    has_fit_parameters=False,
                                                    station_number_cut_SNR_cut=False,
                                                    has_successful_fit=False,
                                                    has_compare_fluence=True)

        # bins_compare = 100 # np.linspace(-10, 10, 100)
        # # read out MC fluence
        fluence_MC = np.array(factory.get_parameter(events_fluence, stp.fluence_compare_MC))
        # # read reconstructed energy fluence
        fluence_rec = np.array(factory.get_parameter(events_fluence, stp.fluence_compare_rec))

        fluence_MC_conc = np.concatenate(fluence_MC)
        fluence_rec_conc = np.concatenate(fluence_rec)

        ratio_hist_data = (fluence_rec_conc - fluence_MC_conc) / fluence_MC_conc

        # 2D histograms along energy, zenith and axis distance

        # get x axis quantities
        energy_compare = factory.get_parameter(events_fluence, shp.energy)
        zenith_compare = np.rad2deg(factory.get_parameter(events_fluence, shp.zenith))
        azimuth_compare = np.rad2deg(factory.get_parameter(events_fluence, shp.azimuth))

        # y axis data
        fluence_deviation = ratio_hist_data

        out = np.abs(fluence_deviation) > 5

        mask_compare = np.abs(fluence_deviation) < 5
        print(f"Cut outliers > 1: {np.sum(out)} / {len(out)}")

        # number of stations per event not eliminated by mask
        n_stations_per_event = [len(x) for x in fluence_MC]

        # assign the event cherenkov radius to every detector in that event
        # r0_rep = np.repeat(r0_comp, n_stations_per_event)

        energy_rep = np.repeat(energy_compare, n_stations_per_event)
        zen_rep = np.repeat(zenith_compare, n_stations_per_event)
        azimuth_rep = np.repeat(azimuth_compare, n_stations_per_event)

        # bin edges on vertical axis
        deviation_binning = np.linspace(-5, 5, 250)

        # mid point of bins on horizontal axis
        bins_energy = np.linspace(17, 20, 21)
        bins_zenith = np.linspace(65, 85, 20)
        bins_azimuth = np.linspace(0, 360, 20)
        bins_f_MC = np.logspace(-1, 6, 20)

        # MAKE THE PLOTS
        # subplots
        fig_fluence_2D, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(35, 20))

        # adjust space around figure edges
        fig_fluence_2D.subplots_adjust(wspace=0.2, left=0.1, right=0.95, bottom=0.15)

        # set title for figure
        # fig_fluence_2D.suptitle(r'Evaluation of rec. fluence: MC vs. reconstructed ADC E-Field (with MC core and arrival direction)', fontsize=30)

        axs[0, 0].set_ylabel(
            r'$\left(f\,_\mathrm{vxB}^\mathrm{rec} - f\,_\mathrm{vxB}^\mathrm{MC}\right) / f\,_\mathrm{vxB}^\mathrm{MC}$',
            fontsize=font)
        # axs[0, 0].set_xscale('log')
        axs[0, 0].set_xlabel(r'$\mathrm{log}_{10}(\mathrm{Energy} / \mathrm{EeV})$',fontsize=font)
        axs[0, 0].set_xticks([17., 18., 19., 20.])
        axs[0, 0].set_ylim(-1.1, 1.1)
        axs[0, 0].set_xlim(17, 20)

        # 2D histogram of the data
        # x and y edges are the bins, z is the amount of entries in each bin in arrays representing the vertical columns
        bin_freq, xedges, yedges = np.histogram2d(
            np.log10(energy_rep[mask_compare]), fluence_deviation[mask_compare], bins=[bins_energy, deviation_binning])

        # to include the highest value in the bin
        yedges[-1] += 1e-6
        # get center values
        center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

        bins_energy[-1] += 1e-6
        bin_centers = bins_energy[:-1] + (bins_energy[1:] - bins_energy[:-1]) / 2

        # calculate mean of each colum
        mean_energy = np.array(
            [np.sum(
                np.array(
                    [center_values[j] * bin_freq[i][j]
                    for j in range(len(bin_freq[i]))])) / np.sum(bin_freq[i])
            for i in range(len(bin_freq))])
        # print(mean_pos)

        # calculate standard deviation
        std_energy = np.sqrt(
            np.array(
                [np.sum(
                    np.array(
                        [bin_freq[i][j] * (center_values[j] - mean_energy[i]) ** 2
                        for j in range(len(bin_freq[i]))])) /
                (np.sum(bin_freq[i]) - 1) for i in range(len(bin_freq))]))
        
        print("Energy mean: ", mean_energy)
        print("Energy std: ", std_energy)

        # add means and standard deviations to the 2D histograms as points and errorbars
        axs[0, 0].errorbar(bin_centers, mean_energy, std_energy, marker='s', color='black', linestyle='', ms=12, markerfacecolor="red", markeredgewidth=2)

        # transpone z
        bin_freq = bin_freq.T

        # assign colour values to the bins
        pcm = axs[0, 0].pcolormesh(
            xedges, yedges, bin_freq, shading='flat', norm=mpl.colors.LogNorm(),
            cmap=None)

        cbi = plt.colorbar(pcm, orientation='vertical', pad=0.02, ax=axs[0, 0])
        cbi.ax.tick_params(axis='both', labelsize=font-5)
        cbi.set_label(cbar_label, fontsize=font)

        # add grid to figure as well as title
        axs[0, 0].grid(True)


        # 2nd plot
        # cherenkov radius plot
        bin_freq, xedges, yedges = np.histogram2d(
            fluence_MC_conc[mask_compare], fluence_deviation[mask_compare], bins=[bins_f_MC, deviation_binning])

        # to include the highest value in the bin
        yedges[-1] += 1e-6
        # get center values
        center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

        bins_f_MC[-1] += 1e-6
        bin_centers = bins_f_MC[:-1] + (bins_f_MC[1:] - bins_f_MC[:-1]) / 2

        # calculate mean of each colum
        mean_f_MC = np.array(
            [np.sum(
                np.array(
                    [center_values[j] * bin_freq[i][j]
                    for j in range(len(bin_freq[i]))])) / np.sum(bin_freq[i])
            for i in range(len(bin_freq))])

        # calculate standard deviation
        std_f_MC = np.sqrt(
            np.array(
                [np.sum(
                    np.array(
                        [bin_freq[i][j] * (center_values[j] - mean_f_MC[i]) ** 2
                        for j in range(len(bin_freq[i]))])) /
                (np.sum(bin_freq[i]) - 1) for i in range(len(bin_freq))]))
        
        print("Fluence mean: ", mean_f_MC)
        print("Fluence std: ", std_f_MC)

        # add means and standard deviations to the 2D histograms as points and errorbars
        axs[0, 1].errorbar(bin_centers, mean_f_MC, std_f_MC, marker='s', color='black', linestyle='', ms=12, markerfacecolor="red", markeredgewidth=2)

        bin_freq = bin_freq.T

        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel(r'MC Fluence $f\,_\mathrm{vxB}^\mathrm{MC}$ [$\mathrm{eV}\,\mathrm{m}^{-2}$]', fontsize=font)

        pcm = axs[0, 1].pcolormesh(xedges, yedges, bin_freq,
                                shading='flat', norm=mpl.colors.LogNorm(), cmap=None)

        axs[0, 1].grid(True)

        cbi = plt.colorbar(pcm, orientation='vertical', pad=0.02, ax=axs[0, 1])
        cbi.ax.tick_params(axis='both', labelsize=font-5)
        cbi.set_label(cbar_label, fontsize=font)


        # 3rd plot
        # zenith plot
        bin_freq, xedges, yedges = np.histogram2d(
            zen_rep[mask_compare], fluence_deviation[mask_compare], bins=[bins_zenith, deviation_binning])

        # to include the highest value in the bin
        yedges[-1] += 1e-6
        # get center values
        center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

        bins_zenith[-1] += 1e-6
        bin_centers = bins_zenith[:-1] + (bins_zenith[1:] - bins_zenith[:-1]) / 2

        # calculate mean of each colum
        mean_zen = np.array(
            [np.sum(
                np.array(
                    [center_values[j] * bin_freq[i][j]
                    for j in range(len(bin_freq[i]))])) / np.sum(bin_freq[i])
            for i in range(len(bin_freq))])

        # calculate standard deviation
        std_zen = np.sqrt(
            np.array(
                [np.sum(
                    np.array(
                        [bin_freq[i][j] * (center_values[j] - mean_zen[i]) ** 2
                        for j in range(len(bin_freq[i]))])) /
                (np.sum(bin_freq[i]) - 1) for i in range(len(bin_freq))]))

        print("Zenith mean: ", mean_zen)
        print("Zenith std: ", std_zen)

        # add means and standard deviations to the 2D histograms as points and errorbars
        axs[1, 0].errorbar(bin_centers, mean_zen, std_zen, marker='s', color='black', linestyle='', ms=12, markerfacecolor="red", markeredgewidth=2)

        bin_freq = bin_freq.T

        # axs[1].set_ylabel(
        #     r'$\left(f\,_\mathrm{geo}^\mathrm{pos} - f\,_\mathrm{geo}^\mathrm{par}\right) / f\,_\mathrm{geo}^\mathrm{pos}$',
        #     fontsize=12)
        axs[1, 0].set_xscale('linear')
        axs[1, 0].set_xlim(65, 85)
        axs[1, 0].set_xlabel('Zenith Angle [°]', fontsize=font)
        axs[1, 0].set_ylabel(
            r'$\left(f\,_\mathrm{vxB}^\mathrm{rec} - f\,_\mathrm{vxB}^\mathrm{MC}\right) / f\,_\mathrm{vxB}^\mathrm{MC}$',
            fontsize=font)

        pcm = axs[1, 0].pcolormesh(xedges, yedges, bin_freq,
                                shading='flat', norm=mpl.colors.LogNorm(), cmap=None)

        axs[1, 0].grid(True)

        cbi = plt.colorbar(pcm, orientation='vertical', pad=0.02, ax=axs[1, 0])
        cbi.ax.tick_params(axis='both', labelsize=font-5)
        cbi.set_label(cbar_label, fontsize=font)
        

        # 4th plot
        # azimuth plot
        bin_freq, xedges, yedges = np.histogram2d(
            azimuth_rep[mask_compare], fluence_deviation[mask_compare], bins=[bins_azimuth, deviation_binning])

        # to include the highest value in the bin
        yedges[-1] += 1e-6
        # get center values
        center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

        bins_azimuth[-1] += 1e-6
        bin_centers = bins_azimuth[:-1] + (bins_azimuth[1:] - bins_azimuth[:-1]) / 2

        # calculate mean of each colum
        mean_azi = np.array(
            [np.sum(
                np.array(
                    [center_values[j] * bin_freq[i][j]
                    for j in range(len(bin_freq[i]))])) / np.sum(bin_freq[i])
            for i in range(len(bin_freq))])

        # calculate standard deviation
        std_azi = np.sqrt(
            np.array(
                [np.sum(
                    np.array(
                        [bin_freq[i][j] * (center_values[j] - mean_azi[i]) ** 2
                        for j in range(len(bin_freq[i]))])) /
                (np.sum(bin_freq[i]) - 1) for i in range(len(bin_freq))]))
        
        print("Azimuth mean: ", mean_azi)
        print("Azimuth std: ", std_azi)

        # add means and standard deviations to the 2D histograms as points and errorbars
        axs[1, 1].errorbar(bin_centers, mean_azi, std_azi, marker='s', color='black', linestyle='', ms=12, markerfacecolor="red", markeredgewidth=2)

        bin_freq = bin_freq.T

        axs[1, 1].set_xscale('linear')
        axs[1, 1].set_xlim(0, 360)
        axs[1, 1].set_xticks([0, 90, 180, 270, 360])
        axs[1, 1].set_xlabel('Azimuth Angle [°]', fontsize=font)

        pcm = axs[1, 1].pcolormesh(xedges, yedges, bin_freq,
                                shading='flat', norm=mpl.colors.LogNorm(), cmap=None)

        axs[1, 1].grid(True)

        cbi = plt.colorbar(pcm, orientation='vertical', pad=0.02, ax=axs[1, 1])
        cbi.ax.tick_params(axis='both', labelsize=font-5)
        cbi.set_label(cbar_label, fontsize=font)
        
        axs[0, 0].tick_params(axis='both', labelsize=font-5)
        axs[0, 1].tick_params(axis='both', labelsize=font-5)
        axs[1, 0].tick_params(axis='both', labelsize=font-5)
        axs[1, 1].tick_params(axis='both', labelsize=font-5)
        
        axs[0, 0].hlines(y=0, xmin=0, xmax=1e8, linewidth=2, color='black', ls="--")
        axs[0, 1].hlines(y=0, xmin=0, xmax=1e8, linewidth=2, color='black', ls="--")
        axs[1, 0].hlines(y=0, xmin=0, xmax=1e8, linewidth=2, color='black', ls="--")
        axs[1, 1].hlines(y=0, xmin=0, xmax=1e8, linewidth=2, color='black', ls="--")

        plt.tight_layout()
        plt.savefig("E_field_recon_eval_binned.png")
        plt.close()


    # 2D histograms for resolution and bias
    if 0: 
        # number of bins for the histogram in both dimensions
        bins = 12 # results in bins - 1 center values
        zenith_bins = np.linspace(65, 85, bins)
        energy_bins = np.linspace(16.85, 20, bins)

        # get centres of zenith bins
        zenith_centres = np.array([zenith_bins[i] + ((zenith_bins[i+1] - zenith_bins[i]) / 2) for i in range(len(zenith_bins) - 1)])

        # make masks for each zenith bin to slect the energies for each of them
        zenith_masks = np.array([np.all([np.rad2deg(zenith_mc) >= zenith_bins[i], \
                                        np.rad2deg(zenith_mc) < zenith_bins[i+1]], axis=0)  \
                                            for i in range(len(zenith_bins) - 1)])

        # array of eem for each zenith bin
        eem_mc_hist = np.array([factory.get_parameter(events[mask], shp.electromagnetic_energy) 
                                for mask in zenith_masks], dtype=object)
        
        # array of fit parameters for each zenith bin to reconstruct eem 
        geo_ldf_params_fit_hist = np.array([factory.get_parameter(events[mask], shp.geomagnetic_ldf_parameter) 
                                            for mask in zenith_masks], dtype=object)
        # array of the fitted/reconstructed geomagnetic energies
        egeo_fit_hist = np.array([[x['E_geo'] 
                                  for x in geo_ldf_params_fit_hist[i]] 
                                  for i in range(len(geo_ldf_params_fit_hist))], dtype=object)
        
        # calculate relevant character for realistic input
        # TODO: check validity
        if para.realistic_input:
            # geomagnetic angle for each bin
            geomag_angle_hist = np.array([factory.get_parameter(events[mask], shp.geomagnetic_angle_recon) 
                                        for mask in zenith_masks], dtype=object)

            # generate array in the same shape as the others
            rho_hist = np.zeros_like(eem_mc_hist)

            # iterate over the zenith bins and the events in it
            for idx, mask in enumerate(zenith_masks):
                # empty list for entries to fill into
                rho_list = []
                for event in events[mask]:
                    # read the relevant parameters from the shower
                    shower = event.get_shower()
                    dmax = shower.get_parameter(shp.distance_to_shower_maximum_geometric_fit)
                    # importantly reconstructed zenith angle here
                    zenith = shower.get_parameter(shp.zenith_recon)
                    obs_lvl = shower.get_parameter(shp.observation_level)
                    atmodel = shower.get_parameter(shp.atmosphere_model)
                    # calculate height above ground
                    height = atm.get_height_above_ground(dmax, zenith, obs_lvl) + obs_lvl
                    # calculate density of shower maximum
                    rho = atm.get_density(height, model=atmodel) * 1e-3
                    # append value to list
                    rho_list.append(rho)
                # add list of rho values as an array to rho_hist
                rho_hist[idx] = np.array(rho_list)

            # print(rho_hist, rho_hist.shape)

        # or use the MC values
        else: 
            # geomagnetic angle for each bin
            geomag_angle_hist = np.array([factory.get_parameter(events[mask], shp.geomagnetic_angle) 
                                        for mask in zenith_masks], dtype=object)
            # array of density at shower max for each bin
            rho_hist = np.array([factory.get_parameter(events[mask], shp.density_at_shower_maximum) 
                                for mask in zenith_masks], dtype=object)
            # print(rho_hist, rho_hist.shape)

        # input array for eem reconstruction
        xdata_hist = np.array([[egeo_fit_hist[i], np.sin(geomag_angle_hist[i]), rho_hist[i]] 
                               for i in range(len(eem_mc_hist))], dtype=object)

        # array of reconstructed eem
        eem_pred_hist = np.array([energyreconstruction.get_Eem_from_Egeo(*xdata_hist[i], *popt, rho_avg=avg_rho) 
                                  for i in range(len(eem_mc_hist))], dtype=object)

        # binning of each zenith bin into energy bins
        # shape: (bins, 3) as bins_centers, masks, edges
        double_bins = np.array([get_data_bins(np.log10(eem_mc_hist[i]), bins=energy_bins, return_bins=True) 
                                                                                for i in range(len(eem_mc_hist))], dtype=object)

        # ratios of reconstructed vs. MC energies for each bin
        eem_rec_ratio_hist = np.array([[eem_pred_hist[i][mask] / eem_mc_hist[i][mask] 
                                               for mask in double_bins[i, 1]] 
                                               for i in range(len(eem_mc_hist))], dtype=object)

        bin_amount = np.array([[len(eem_pred_hist[i][mask])
                                               for mask in double_bins[i, 1]] 
                                               for i in range(len(eem_mc_hist))], dtype=object)

        # maks to check whether each bin has a sufficient number of simulations (5)
        masks_hist = np.array([np.sum(double_bins[i, 1], axis=1) > 5 for i in range(len(eem_mc_hist))], dtype=object)

        # calculate std and mean for each bin (here without the mask, so you get nans if there's no data)
        std_hist = np.array([[np.std(x) 
                              for x in eem_rec_ratio_hist[i]] 
                              for i in range(len(eem_mc_hist))], dtype=object)
        mean_hist = np.array([[np.mean(x) 
                         for x in eem_rec_ratio_hist[i]] 
                         for i in range(len(eem_mc_hist))], dtype=object)

        res_hist = np.array([np.array([std_hist[i, j] * 100 for j in range(len(eem_mc_hist))]) for i in range(len(eem_mc_hist))])
        bias_hist = np.array([np.array([(mean_hist[i, j] - 1) * 100 for j in range(len(eem_mc_hist))]) for i in range(len(eem_mc_hist))])

        # print(eem_rec_ratio_hist)

        # plot for resolution
        fig_hist_res, ax_hist_res = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

        plt.pcolormesh(double_bins[0, 0], zenith_centres, res_hist, cmap='Greys', vmin=0, vmax=50)
        plt.colorbar(label="Resolution of energy reconstruction [%]")# , fontsize=20)

        for i in range(len(bin_amount)):
            for j in range(len(bin_amount[i])):
                ax_hist_res.annotate(bin_amount[i, j], (double_bins[0, 0][j] - 0.05, zenith_centres[i] - 0.4), c='blue')


        ax_hist_res.set_xticks([17, 17.5, 18, 18.5, 19, 19.5, 20])
        ax_hist_res.set_yticks([65, 70, 75, 80, 85])
        ax_hist_res.tick_params(labelsize=18)
        ax_hist_res.set_ylabel("Zenith Angle [°]", fontsize=20)
        ax_hist_res.set_xlabel(r"log($E_\mathrm{em}$/EeV)", fontsize=20)
        # fig_angles.suptitle(title, fontsize=25)
        ax_hist_res.set_xlim(ax_angles.get_xlim())
        fig_hist_res.tight_layout()
        fig_hist_res.savefig("b-resolution_2Dhist.png")
        plt.close()

        # plot for bias
        fig_hist_bias, ax_hist_bias = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

        # ax_hist_res.hist2d(res_energy, res_zenith, bins=(50, 50), cmap='Greys_r')
        plt.pcolormesh(double_bins[0, 0], zenith_centres, bias_hist, cmap='bwr', vmin=-50, vmax=50)
        plt.colorbar(label="Bias of energy reconstruction [%]")

        ax_hist_bias.set_xticks([17, 17.5, 18, 18.5, 19, 19.5, 20])
        ax_hist_bias.set_yticks([65, 70, 75, 80, 85])
        ax_hist_bias.tick_params(labelsize=18)
        ax_hist_bias.set_ylabel("Zenith Angle [°]", fontsize=20)
        ax_hist_bias.set_xlabel(r"log($E_\mathrm{em}$/EeV)", fontsize=20)
        ax_hist_bias.set_xlim(ax_angles.get_xlim())
        # fig_angles.suptitle(title, fontsize=25)
        fig_hist_bias.tight_layout()
        fig_hist_bias.savefig("bias_2Dhist.png")
        plt.close()


        if para.realistic_input:

            # determine events list
            sat_events = events # full_events

            # repeat procedure for saturated antennas
            zenith_hist = np.rad2deg(factory.get_parameter(sat_events, shp.zenith))

            # make masks for each zenith bin to select the energies for each of them
            # use zenith angles from full events list here
            zenith_masks = np.array([np.all([zenith_hist >= zenith_bins[i], \
                                            zenith_hist < zenith_bins[i+1]], axis=0)  \
                                                for i in range(len(zenith_bins) - 1)])

            # array of eem for each zenith bin
            eem_mc_hist_full = np.array([factory.get_parameter(sat_events[mask], shp.electromagnetic_energy) 
                                    for mask in zenith_masks], dtype=object)
            
            # binning of each zenith bin into energy bins
            # shape: (bins, 3) as bins_centers, masks, edges
            double_bins = np.array([get_data_bins(np.log10(eem_mc_hist_full[i]), bins=energy_bins, return_bins=True) 
                                                                                    for i in range(len(eem_mc_hist_full))], dtype=object)

            # array for saturated antennas for each zenith bin
            saturated_hist = np.array([factory.get_parameter(sat_events[mask], stp.saturated) 
                                    for mask in zenith_masks], dtype=object)
            
            # number of saturated stations for each bin
            saturated_hist_energy = np.array([[saturated_hist[i][mask] for mask in double_bins[i, 1]] for i in range(len(eem_mc_hist_full))], dtype=object)

            # get the number/fraction of saturated antennas for each bin
            number_of_saturated_hist = np.array([[np.array([np.sum(mask) #/ len(mask)
                                for mask in saturated_hist_energy[i, x]])
                                for x in range(len(saturated_hist_energy[i]))] 
                                for i in range(len(eem_mc_hist_full))], dtype=object)
            
            # calculate the mean in each bin
            mean_saturated_hist = np.array([[np.median(x) 
                                for x in number_of_saturated_hist[i]] 
                                for i in range(len(eem_mc_hist_full))], dtype=object)
            
            final_sat_hist = np.array([np.array([mean_saturated_hist[i, j] for j in range(len(saturated_hist))]) for i in range(len(saturated_hist))])


            number_of_unsaturated_hist = np.array([[np.array([len(~mask) #/ len(mask)
                                for mask in saturated_hist_energy[i, x]])
                                for x in range(len(saturated_hist_energy[i]))] 
                                for i in range(len(eem_mc_hist_full))], dtype=object)
            
            # calculate the mean in each bin
            mean_unsaturated_hist = np.array([[np.median(x) 
                                for x in number_of_unsaturated_hist[i]] 
                                for i in range(len(eem_mc_hist_full))], dtype=object)
            
            final_unsat_hist = np.array([np.array([mean_unsaturated_hist[i, j] for j in range(len(saturated_hist))]) for i in range(len(saturated_hist))])


            # plot for saturated antennas
            fig_hist_sat, ax_hist_sat = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

            # print(double_bins[0, 0], zenith_centres, final_sat_hist)

            # ax_hist_res.hist2d(res_energy, res_zenith, bins=(50, 50), cmap='Greys_r')
            plt.pcolormesh(double_bins[0, 0], zenith_centres, final_sat_hist, cmap='Greys')
            plt.colorbar(label="# of saturated antennas per event per bin")

            ax_hist_sat.set_xticks([17, 17.5, 18, 18.5, 19, 19.5, 20])
            ax_hist_sat.set_yticks([65, 70, 75, 80, 85])
            ax_hist_sat.tick_params(labelsize=18)
            ax_hist_sat.set_ylabel("Zenith Angle [°]", fontsize=20)
            ax_hist_sat.set_xlabel(r"log($E_\mathrm{em}$/eV)", fontsize=20)
            # fig_angles.suptitle(title, fontsize=25)
            fig_hist_sat.tight_layout()
            fig_hist_sat.savefig("b_saturated_2Dhist.png")
            plt.close()


            # plot for saturated antennas
            fig_hist_unsat, ax_hist_unsat = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

            # print(double_bins[0, 0], zenith_centres, final_sat_hist)

            # ax_hist_res.hist2d(res_energy, res_zenith, bins=(50, 50), cmap='Greys_r')
            plt.pcolormesh(double_bins[0, 0], zenith_centres, final_unsat_hist, cmap='Greys')
            plt.colorbar(label="# of antennas per event per bin")

            ax_hist_unsat.set_xticks([17, 17.5, 18, 18.5, 19, 19.5, 20])
            ax_hist_unsat.set_yticks([65, 70, 75, 80, 85])
            ax_hist_unsat.tick_params(labelsize=18)
            ax_hist_unsat.set_ylabel("Zenith Angle [°]", fontsize=20)
            ax_hist_unsat.set_xlabel(r"log($E_\mathrm{em}$/eV)", fontsize=20)
            # fig_angles.suptitle(title, fontsize=25)
            fig_hist_unsat.tight_layout()
            fig_hist_unsat.savefig("b_unsaturated_2Dhist.png")
            print("Histograms saved!")
            plt.close()

    # density correction plot
    if 1:
        fig_rho, ax = plt.subplots(1, figsize=(16, 9))
    
        def corr(density,  p0=3.94278610e-01, p1=-2.37010471e+00, p2=0.05, p3=0.1, dens_avg=None, p=1):
            if dens_avg is None:
                dens_avg = radiationenergy.get_average_density()
            return (1 - p0 + p0 * np.exp(p1 * (density - dens_avg)) - p2 / density + p3) ** p

        y = (egeo_fit / (np.sin(geomag_angle_rec_or_mc) ** popt[6] * popt[0])) ** 0.5 / (eem_mc / ref_energy) ** (popt[1] / 2)
        
        print(y, np.mean(y))

        y = y / np.mean(y)
        ylabel = r"$\frac{y}{\langle y\rangle}$ with $y = \sqrt{\frac{E_\mathrm{geo}^\mathrm{fit}}{\sin^{c_\alpha}\left(\alpha\right)\cdot S_{19}}}\cdot \left(\frac{10^{19}\,\mathrm{eV}}{E_\text{em}^\text{MC}}\right)^{\frac{\gamma}{2}}$"
        # ylabel = r"Density correction"

        if 0:
            sct = ax.scatter(rho_fit_or_mc, y, alpha=0.6)
        elif 1:
            sct = ax.scatter(rho_fit_or_mc, y, c=np.sin(geomag_angle_rec_or_mc), alpha=0.4, cmap=colourscheme[0], label="Air-shower simulations")
            cbi = plt.colorbar(sct, pad=0.02)
            cbi.set_label(r"$\sin(\alpha)$")
            cbi.ax.tick_params(labelsize=30)

        else:
            azi_deg_nord = helper.get_normalized_angle(factory.get_parameter(events, shp.azimuth), interval=[-np.pi, np.pi])
            sct = ax.scatter(rho_fit_or_mc, y, c=azi_deg_nord,
                             cmap="seismic", alpha=0.4)
            cbi = plt.colorbar(sct, pad=0.02)
            cbi.set_label("azimuth (0 = east)")

        n, bins_center, y_mean_binned, y_std_binned = \
            helperstats.get_binned_data(rho_fit_or_mc, y, 10,
                                    return_bins=False, skip_empty_bins=True)
        mask = n > 10
        ax.errorbar(bins_center[mask], y_mean_binned[mask], y_std_binned[mask],
                    color="black", marker="s", markerfacecolor="red", markeredgewidth=1.5, ls="", lw=3, markersize=10, label="Mean values")

        rho_fit_or_mc.sort()

        label = (r"$p_0$ = %.2f" + "\n" + r"$p_1$ = %.2f / (kg$\,$m$^{-3}$)" + "\n" \
                     + r"$p_2$ = %.2f $\,$kg$\,$m$^{-3}$" + "\n" + r"$p_3$ = %.2f" + "\n" \
                     + r"$\langle \rho_\mathrm{max} \rangle$ = %.2f kg$\,$m$^{-3}$" + "\n" \
                     + r"$c_\alpha$ = %.2f") % \
                      (popt[2], popt[3], popt[4], popt[5], avg_rho, popt[6])
                      
        label = "Energy correction"

        ax.plot(rho_fit_or_mc, corr(
            rho_fit_or_mc, p0=popt[2], p1=popt[3], p2=popt[4], p3=popt[5], dens_avg=avg_rho), c="black", linestyle="--", lw=3, label=label)

        ax.grid()
        ax.legend(ncol=1, fontsize=25)

        ax.set_xlabel(r"$\rho_\mathrm{air}$ [kg$\,$m$^{-3}$]")# r"$%s$ [kg$\,$m$^{-3}$]" % lr)
        ax.set_ylabel("Normalised energy ratio") # ylabel)
        ax.tick_params(labelsize=30)
        ax.set_title("GP300")# "GRAND@Auger")
        fig_rho.tight_layout()

    # Xmax Plot
    if 1:
        fig_xmax = plt.figure()

        x1, y1, w = 0.12, 0.14, 0.1
        ax0 = fig_xmax.add_axes([x1, y1, 1-w-x1, 1-w-y1])
        ax1 = fig_xmax.add_axes([x1, 1-w, 1-w-x1, w])
        ax2 = fig_xmax.add_axes([1-w, y1, w, 1-w-y1])

        var = eem_pred / eem_mc
        # mask = np.all([var > 0, var < 2], axis=0)
        # print(np.sum(mask), len(mask))
        mask = np.array([True] * len(var))
        ax0.plot(xmax_mc[mask], var[mask], "o", alpha=0.3,
            label="overflow = %d\nunderflow = %d" % (np.sum(var > 1.25), np.sum(var < 0.75)))

        if 1:
            n, xcen, y_mean_binned, y_std_binned, edges = \
                helperstats.get_binned_data(
                    xmax_mc[mask], var[mask],
                    10, skip_empty_bins=False, return_bins=True)

            xerr = np.array([xcen - edges[:-1], edges[1:] - xcen])
            mask2 = n > 3
            ax0.errorbar(xcen[mask2], y_mean_binned[mask2], y_std_binned[mask2],
                            xerr[:, mask2], color="C3", marker="s", ls="", markeredgewidth=3, markersize=10, lw=3, zorder=3)
            ax0.errorbar(xcen[mask2], y_mean_binned[mask2], y_std_binned[mask2] / np.sqrt(n[mask2]), 
                         capsize=10, color="C3", marker="s", ls="", markeredgewidth=3, markersize=10, lw=3, zorder=3)

        ax0.set_ylim(0.75, 1.25)

        ax1.hist(xmax_mc[mask], np.linspace(*ax0.get_xlim()), alpha=0.3)
        ax1.set_xlim(*ax0.get_xlim())
        ax2.hist(var[mask], np.linspace(*ax0.get_ylim()),
                orientation="horizontal", alpha=0.3)
        ax2.set_ylim(*ax0.get_ylim())
        ax1.set_axis_off()
        ax2.set_axis_off()

        ax0.set_ylabel(
            r"$E_\mathrm{em}(E_\mathrm{geo}$) / $E_\mathrm{em}^\mathrm{MC}$")
        ax0.set_xlabel(r"$X_\mathrm{max}^\mathrm{MC}$ [g$\,$cm$^{-2}$]")
        ax0.legend(fontsize="small")
        ax0.grid()

    # Primary Plot
    if 1:
        # hist energy resolution
        fig_primary, ax = pyplots.rphp.get_histogram(
            eem_pred / eem_mc, bins=np.linspace(0.5, 1.5, 25), figsize=(13, 10.5),
            stat_kwargs={
                'fontsize': "small", "ha": "right", "posx": 0.985, "posy": 0.98},
            xlabel=r'$E_\mathrm{em}(E_\mathrm{geo}^\mathrm{fit})$ / $E_\mathrm{em}^\mathrm{MC}$', ylabel="# of events")
        
        if 1:
            pp = factory.get_parameter(
                events, shp.primary_particle)

            ax2 = fig_primary.add_axes([0.15, 0.25, 0.25, 0.2])
            for pdx, p in enumerate(np.unique(pp)):
                pmask = pp == p
                _, l, c = get_pp_index_acronym(int(float(p)), corsika=True)
                ax.hist((eem_pred / eem_mc)[pmask], np.linspace(0.5, 1.5, 25), histtype="step", color=c, edgecolor=c,
                        alpha=1, lw=3, label=r"%s: $\mu$ = %.4f,""\n\t"r"$\sigma$ = %.3f "
                        % (l, np.mean((eem_pred / eem_mc)[pmask]), np.std((eem_pred / eem_mc)[pmask])))
                
                ax2.errorbar(np.mean((eem_pred / eem_mc)[pmask]), len(np.unique(pp)) - pdx,
                            xerr=np.std((eem_pred / eem_mc)[pmask]), marker="d", color=c, lw=3, markersize=20)
        
            ax.legend(loc="upper left", fontsize="x-small")
        
            ax2.axvline(1, color="grey", zorder=0)
            ax2.set_ylim(0.5, 4.5)
            # Y AXIS -BORDER
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax2.set_xlabel(r"$\mu \pm \sigma$", fontsize="small")
            # ax2.set_xticks([0.95, 1, 1.05])
            # ax2.set_xticklabels([0.95, 1, 1.05], fontsize="x-small")
            ax2.tick_params(axis='x', labelsize="x-small")
        ax.set_xlim(0.55, 1.4)


    if not para.save:
        plt.show()
    else:
        fig_eng.savefig("fit_result_eng%s.pdf" % (para.label))
        try:
            fig_rho.savefig("fit_result_density_correction%s.pdf" % (para.label))
        except:
            pass
        try:
            fig_eng2.savefig("fit_result_eng_pull%s.pdf" % (para.label))
        except:
            pass
        try:
            fig_primary.savefig(fname="eem_rec_res%s.pdf" % (para.label))
        except:
            pass
        fig_xmax.savefig("fit_result_xmax%s.pdf" % (para.label))

        # try:
        #     fig_xmax.savefig("fit_result_xmax%s.pdf" % (para.label))
        # except:
        #     print("n")
        #     pass


def evaluate_fit_result_star(events, para):
    has_shower = np.array([ev.has_shower(evp.rd_shower) for ev in events])
    print("Events with Rd shower: %d / %d" %
          (np.sum(has_shower), len(has_shower)))
    events = events[has_shower]
    
    mask = factory.has_parameter(events, shp.geomagnetic_ldf_parameter)
    print("Events with fit parameter: %d / %d" % (np.sum(mask), len(mask)))
    events = events[mask]
    
    fit_result = factory.get_parameter(events, shp.fit_result)
    success = np.array([x["success"] for x in fit_result])
    print("Events with successful fit: %d / %d" % (np.sum(success), len(success)))
    events = events[success]

    geo_ldf_params_fit, geo_ldf_params_fit_err = factory.get_parameter_and_error(
        events, shp.geomagnetic_ldf_parameter)

    distance_to_xmax_mc = factory.get_parameter(
        events, shp.distance_to_shower_maximum_geometric)
    zenith_mc = factory.get_parameter(events, shp.zenith)

    atmodels = factory.get_parameter(events, shp.atmosphere_model)
    n0s = factory.get_parameter(events, evp.refractive_index_at_sea_level)

    atms = np.array([atmodels, n0s]).T
    unique_atms = np.unique(atms, axis=0)[::-1]
    atm_masks = np.array([np.all(atms == ua, axis=1) for ua in unique_atms])

    egeo_fit = np.array([x['E_geo'] for x in geo_ldf_params_fit])
    egeo_fit_err = np.array([x['E_geo'] for x in geo_ldf_params_fit_err])

    dmax_fit, dmax_fit_err = factory.get_parameter_and_error(
        events, shp.distance_to_shower_maximum_geometric_fit)

    # obs_level = factory.get_parameter(events, shp.observation_level)
    # if "density_at_xmax" in geo_ldf_params_fit[0]:
    #     rho_fit = np.array([x['density_at_xmax'] for x in geo_ldf_params_fit])
    # else:
    #     rho_fit = np.zeros_like(rho_mc)
    #     for idx, (d, z, obs, model) in enumerate(zip(dmax_fit, zenith_mc, obs_level, atmodels)):
    #         h = atm.get_height_above_ground(d, z, obs) + obs
    #         rho_fit[idx] = atm.get_density(h, model=model) * 1e-3
            
    bins = get_bins_for_x_from_binned_data(
        distance_to_xmax_mc / 1e3, np.around(np.rad2deg(zenith_mc), 1))

    fig_dmax, axs = plt.subplots(2, sharex=True)
    for idx, (mask, ua) in enumerate(zip(atm_masks, unique_atms)):

        alpha = 0.3
        if len(unique_atms) > 1:
            alpha = 5 / (np.sqrt(np.sum(mask)))

        axs[0].errorbar(distance_to_xmax_mc[mask] / 1e3, dmax_fit[mask] / 1e3,
                        dmax_fit_err[mask] / 1e3, marker="o", ls="", alpha=alpha,
                        label=r"ATM: %d, N$_0$ = %.2e" % (ua[0], ua[1] - 1))

        yvar = (dmax_fit / distance_to_xmax_mc)[mask]

        axs[1].errorbar(distance_to_xmax_mc[mask] / 1e3, yvar,
                        dmax_fit_err[mask] / distance_to_xmax_mc[mask],
                        marker="o", ls="", alpha=alpha,
                        label=r"$\mu$ = %.3f, $\sigma$ = %.3f" %
                        (np.nanmean(yvar), np.nanstd(yvar)))

        if 1:
            n, xcen, y_mean_binned, y_std_binned, edges = \
                helperstats.get_binned_data(
                    distance_to_xmax_mc[mask] / 1e3, yvar,
                    bins, skip_empty_bins=False, return_bins=True)
            
            xerr = np.array([xcen - edges[:-1], edges[1:] - xcen])
            n_mask = n > 5
            cidx = 3 if len(unique_atms) == 1 else idx
            axs[1].errorbar(xcen[n_mask], y_mean_binned[n_mask], y_std_binned[n_mask],
                            xerr[:, n_mask], color="C%d" % cidx, marker="o", ls="", zorder=3, markersize=10)
    
    axs[0].plot([distance_to_xmax_mc.min() / 1e3, distance_to_xmax_mc.max() / 1e3], [
                distance_to_xmax_mc.min() / 1e3, distance_to_xmax_mc.max() / 1e3], "k--", zorder=10)
    axs[1].set_ylim(0.75, 1.25)
    
    axs[0].set_ylabel(r"$d_\mathrm{max}^\mathrm{fit}$ [km]")
    axs[1].set_xlabel(r"$d_\mathrm{max}^\mathrm{MC}$ [km]")
    axs[1].set_ylabel(
        r"$d_\mathrm{max}^\mathrm{fit} / d_\mathrm{max}^\mathrm{MC}$")
    
    axs[1].axhline(1, c="k", ls="--", zorder=10)
    [ax.grid() for ax in axs]
    [ax.legend(ncol=2, fontsize=20) for ax in axs]
    fig_dmax.tight_layout()


    egeo_mc = factory.get_parameter(events, shp.geomagnetic_energy)

    fig_eng, axs = plt.subplots(3)
    for idx, (mask, ua) in enumerate(zip(atm_masks, unique_atms)):

        alpha = 0.3
        if len(unique_atms) > 1:
            alpha = 5 / (np.sqrt(np.sum(mask)))

        axs[0].plot(egeo_mc[mask], egeo_fit[mask], "o", alpha=alpha,
                    label=r"ATM: %d, N$_0$ = %.2e" % (ua[0], ua[1] - 1))
    
        axs[1].plot(egeo_mc[mask], egeo_fit[mask] / egeo_mc[mask], "o", alpha=alpha,
                    label=r"$\mu$ = %.3f, $\sigma$ = %.3f" %
                    (np.mean((egeo_fit / egeo_mc)[mask]), np.std((egeo_fit / egeo_mc)[mask])))
    
        # cor = stats.pearsonr(distance_to_xmax_mc[mask] / 1e3,
        #                      egeo_fit[mask] / egeo_mc[mask])[0]

        axs[2].plot(distance_to_xmax_mc[mask] / 1e3,
                    egeo_fit[mask] / egeo_mc[mask], "o", alpha=alpha)
                    #,label="%.3f" % cor)


    axs[0].plot([egeo_mc.min(), egeo_mc.max()], [
                        egeo_mc.min(), egeo_mc.max()], "k--")

    axs[1].set_xlabel(r"$E_\mathrm{geo}^\mathrm{MC}$ [eV]")
    axs[1].set_ylabel(
        r"$E_\mathrm{geo}^\mathrm{fit} / E_\mathrm{geo}^\mathrm{MC}$")
    axs[1].set_xscale("log")
    axs[1].axhline(1, c="k", ls="--", zorder=10)


    axs[2].set_xlabel(r"$d_\mathrm{max}^\mathrm{MC}$ [km]")
    axs[2].set_ylabel(
        r"$E_\mathrm{geo}^\mathrm{fit} / E_\mathrm{geo}^\mathrm{MC}$")

    axs[0].set_ylabel(r"$E_\mathrm{geo}^\mathrm{fit}$ [eV]")
    axs[0].set_xlabel(r"$E_\mathrm{geo}^\mathrm{MC}$ [eV]")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    [ax.grid() for ax in axs]
    [ax.legend(ncol=len(unique_atms), fontsize=15) for ax in axs]
    fig_eng.tight_layout()

    if not para.save:
        plt.show()
    else:
        fig_dmax.savefig("fit_result_dmax_star%s.png" % para.label)
        fig_eng.savefig("fit_result_eng_star%s.png" % para.label)
    sys.exit()
