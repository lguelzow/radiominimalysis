import sys
import os
import argparse
import numpy as np

from radiominimalysis.input_output import coreas_reader
from radiominimalysis.framework.parameters import showerParameters as shp, eventParameters as evp, \
    stationParameters as stp

from radiominimalysis.modules.reconstruction import geometry, signal_emissions
from radiominimalysis.utilities import cherenkov_radius


from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.ticker as tick    
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmasher as cmr

from scipy.interpolate import griddata

# parser for command line arguments
parser = argparse.ArgumentParser(description='')

# parser asks for hdf5 input files to plot results from
parser.add_argument('paths', metavar='PATH', type=str, nargs='*', default=[],
                    help='Choose hdf5 input file(s).')

# parser asks for atmosphere model to use in calculation
parser.add_argument('-gd', '--gdasFile', metavar='PATH', type=str, nargs='*', default=None,
                    help='Choose gdas atmosphere file.')

# parser asks for atmosphere model to use in calculation
parser.add_argument('-m', '--atmModel', metavar='int', type=int, nargs='?', default=1,
                    help='Set the model id for atmospheric model')

parser.add_argument(
    "-real",
    "--realistic_input",
    action="store_true",
    default=None,
    help="Whether realistic simulation is the input (default: False)"
)

# read arguments from the command line after the function itself
args = parser.parse_args()

# use CoREAS reader tool to read out the showers from the input files
reader = coreas_reader.readCoREASShower(args.paths)
# print(reader[0])

# frame numbers for animation
frame_number = -1

# loop over all events put into the program
for revent in reader.run():

    frame_number += 1

    # set if shower is going upwards
    #
    # remember to manually set the shower core if True
    # upwards = True
    upwards = False

    # check if shower is going upwards or downwards
    if upwards == True:
        # shower coordinate for upwards
        core = [0., 0., 922.]
    else:
        # axis limits for plots
        core = None

    # for each event, get parameters: 
    # station position; station axis distance; various fluence components at each station

    # read out station positions
    #
    # specify a core for upward going showers
    pos_vB = revent.get_station_position_vB_vvB(core=core)
    pos_xy = revent.get_station_position_cs()
    r = revent.get_station_axis_distance()

    # calculate fluences
    f_vxB_vxvxB_v = revent.get_station_parameter(stp.energy_fluence)
    f_vxB = f_vxB_vxvxB_v[:, 0]
    f_vxvxB = f_vxB_vxvxB_v[:, 1]
    f_v = f_vxB_vxvxB_v[:, 2]
    f_vxb_vxvxB = np.array([(f_vxB_vxvxB_v[i, 0] + f_vxB_vxvxB_v[i, 1]) for i in range(len(f_vxB_vxvxB_v))])
    # list with total energy fluence
    f_total = np.array([(f_vxB_vxvxB_v[i, 0] + f_vxB_vxvxB_v[i, 1] + f_vxB_vxvxB_v[i, 2]) for i in range(len(f_vxB_vxvxB_v))])

    # read file/plot name from input
    fname = args.paths
    
    # reconstruct_geometry calculates among other things:
    # distance to Xmax: shp.distance_to_shower_maximum_geometric; early-late correction factor: stp.early_late_factor
    # and adds them to the revent
    geometry.reconstruct_geometry([revent], args)
    
    # now import shower from the revent
    shower = revent.get_shower()

    # read out atmosphere model
    at = revent.get_atmosphere()
    atm_model = shower.get_parameter(shp.atmosphere_model)

    # and read out the parameters that were just calculated in reconstruct_geometry
    dmax = shower.get_parameter(shp.distance_to_shower_maximum_geometric)
    c_early_late = revent.get_station_parameter(stp.early_late_factor)
    density = shower.get_parameter(shp.density_at_shower_maximum)
    # distance_grammage = shower.get_parameter(shp.distance_to_shower_maximum_grammage)
    shower_energy = shower.get_parameter(shp.energy) / 10 ** 18
    # radiation_energy = shower.get_parameter(shp.radiation_energy) / 10 ** 18
    geomagnetic_angle = shower.get_parameter(shp.geomagnetic_angle)
    zenith = np.rad2deg(shower.get_parameter(shp.zenith))
    azimuth = np.rad2deg(shower.get_parameter(shp.azimuth))
    radius = np.sqrt(pos_vB[:, 0] ** 2 + pos_vB[:, 1] ** 2)
    radius_real = np.sqrt(pos_xy[:, 0] ** 2 + pos_xy[:, 1] ** 2)

    # print(dmax)

    # calculate cherenkov radius
    r_cheren = cherenkov_radius.get_cherenkov_radius_model_revent(revent)

    # print(radius.shape)
    # print(distance_grammage)

    #
    # optional lines for calculation of fluence components (geomagnetic/charge excess)
    #

    # signal_emissions calculates energy fluence from geomagnetic and charge-excess emission:
    signal_emissions.reconstruct_geomagnetic_and_charge_excess_emission_from_position([revent], None)
    signal_emissions.reconstruct_emission_from_param([revent], None)
    # signal_emissions.reconstruct_emission_from_param([revent], \
    #     [-4.75242667e-06, 1.05486603e+00, 1.46304172e+00, 3.74551575e+00, 2.97604913e+00, 5.04797419e-02])

    # rho = shower.get_parameter(shp.density_at_shower_maximum)

    # print(rho)

    # and now we read out the calculated fluence for this file
    f_geo_posi = revent.get_station_parameter(stp.geomagnetic_fluence_positional)
    f_geo_param = revent.get_station_parameter(stp.geomagnetic_fluence_parametric)
    f_ce_param = revent.get_station_parameter(stp.charge_excess_fluence_parametric)

    # print(f_vxB.shape)
    # print(f_vxvxB.shape)
    # print(f_geo_param.shape)
    # print(f_ce_param.shape)

    # read out charge excess fraction from parametrisation
    a_ce = revent.get_station_parameter(stp.charge_excess_fraction_parametric)



    # print most important shower parameters
    print("Zenith Angle: ", zenith)
    print("Azimuth Angle: ", azimuth)
    print("Geomagnetic Angle: ", np.rad2deg(geomagnetic_angle))
    print("Distance to shower maximum: ", dmax)
    print("Total Shower Energy: ", shower_energy, " EeV")
    # print("Radiation Energy: ", radiation_energy)

    # normalise all distance data by converting units of Cherenkov radii, and apply early-late correction
    vxB_axis = pos_vB[:, 0] / r_cheren / c_early_late * r_cheren
    vxvxB_axis = pos_vB[:, 1] / r_cheren / c_early_late * r_cheren

    # parameters for the correction for the air density
    rho0 = 0.504
    rho1 = -2.708 # (kg/m^-3)^-1
    # mean air density at shower maximum for a theta=75° air shower
    rho_mean = 0.3 # kg/m^-3

    # equation for density correction
    density_correction = 1 / (1 - rho0 + rho0 * np.exp(rho1 * (density - rho_mean))) ** 2

    # correction for geomagnetic field
    geo_correction = 1 / np.sin(geomagnetic_angle) ** 2


    # normalise fluence with dmax² and apply early-late correction where appropriate
    # normalise to GeV as well
    f_ground = f_vxB # / 1e3
    f_vxB = f_vxB * c_early_late ** 2 #  / 1e3
    # f_vxB = f_vxB * c_early_late ** 2
    f_vxvxB = f_vxvxB * c_early_late ** 2 # / 1e3
    # f_vxvxB = f_vxvxB * c_early_late
    f_geo_param = f_geo_param # / 1e3
    f_ce_param = f_ce_param # / 1e3
    f_geo_posi = f_geo_posi * c_early_late ** 2 #  / 1e3

    # print(c_early_late)


    #
    #
    # start plot!!
    #
    #

    # set global fontsize
    plt.rcParams['font.size'] = 18

    # set how many subplots there should be (1 row, 3 columns), they share axes 
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(18, 6))
    # fig.suptitle(r"CR energy$=%.3f$ $\,\mathrm{EeV}$" %shower_energy + "\n" + r"Zenith angle$=%.1f °$" %zenith \
    #               + "\n" + r"Geomagnetic angle$=%.2f °$" %np.rad2deg(geomagnetic_angle) \
    #               + "\n" + r"Azimuth angle$=%.0f °$" %azimuth)


    # for all subplots
    # Make interpolated plot of energy fluence over the antenna grid

    # 
    # current version:
    # plot vxB and vxvxB polarisations, geomagnetic and charge excess all next to each other

    # resolution of interpolation grid
    lattice_const = 800j
    print("Grid resolution: ", lattice_const)

    #
    # check if shower is going upwards or downwards
    #
    # upwards and core position are set at the top
    if upwards == True:
        # axis limits for plots
        plot_scale = 0.3333 * np.max(radius)
    else:
        # axis limits for plots
        plot_scale = r_cheren / r_cheren * 1.5 * r_cheren

    
    # limits for the interpolation
    dist_scale = r_cheren / r_cheren * 1.8 * r_cheren # np.max(radius)
    # print(dist_scale)
    # meshgrid within the interpolation limit with fineness of lattice constant/resolution
    XI, YI = np.mgrid[-dist_scale:dist_scale:lattice_const, \
                      -dist_scale:dist_scale:lattice_const]

    # colourmap name
    cmap = [cmr.flamingo, cmr.freeze_r, 'gnuplot2_r', 'hot','bone','plasma','PuRd','magma','brg']
    cmap = cmap[1]
    # interpolation algorithms
    interp = ['nearest', 'bicubic', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    
    #
    # formatting for all subplots
    #
    # sets aspect ratio for all plots to 1 (quadratic frame)
    [ax.set_aspect(1) for ax in axs]
    # enables grid background for all plots
    # [ax.grid() for ax in axs]
    # makes antenna arms appear further than the signal does for all plots
    [ax.set_facecolor("whitesmoke") for ax in axs]
    # set xy lims for all plots
    axs[0].set_xlim(-plot_scale, plot_scale)
    axs[1].set_xlim(-plot_scale, plot_scale)
    axs[2].set_xlim(-plot_scale, plot_scale)
    axs[0].set_ylim(-plot_scale, plot_scale)
    axs[1].set_ylim(-plot_scale, plot_scale)
    axs[2].set_ylim(-plot_scale, plot_scale)
    # plot antenna positions for all plots
    # change antenna positions in the first plot in the used version
    # axs[1].plot(vxB_axis, vxvxB_axis, "o", markersize=1, c='white', alpha=1)
    axs[2].plot(vxB_axis, vxvxB_axis, "o", markersize=1, c='white', alpha=1)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = [make_axes_locatable(ax) for ax in axs]
    cax = [divider[i].append_axes("right", size="5%", pad=0.05) for i in range(len(axs))]


    #
    # all for axs[0], the first plot
    #
    
    # first version is the plot of antennas projected into the shower plane

    # data grid points and values for total fluence
    (x_data, y_data, values) = (vxB_axis, vxvxB_axis, f_ground)
    # (x_data, y_data, values) = (pos_vB[:, 0] / r_cheren, pos_vB[:, 1] / r_cheren, f_ground)

    # plot antenna positions
    axs[0].plot(pos_vB[:, 0], pos_vB[:, 1], "o", markersize=1, c='white', alpha=1)

    # interpolated grid
    resampled = griddata((x_data,y_data), values, (XI,YI), method='cubic')
    
    # generate and assign interpolation data to the interpolated grid
    fluence_map = axs[0].imshow(resampled.T, origin='lower', cmap=cmap, interpolation=interp[0], alpha=1, \
                             extent=[-dist_scale, dist_scale, -dist_scale, dist_scale]) # , vmin=0, vmax=cmax)

    # contour plot to show the high  intensity region
    grav_pot = axs[0].contour(XI, YI, resampled, colors='white', alpha = 0.4, \
                            levels = np.linspace(max(values) * 0.9499999, max(values) * 0.95, 2))
    
    # assign colorbar to this subplot
    cbar = fig.colorbar(fluence_map, ax=axs[0], cax=cax[0], format=tick.FormatStrFormatter('%.0f'))
    cbar.set_label(r"Energy fluence [ev/m$^2$]")

    # labels and titles and limits
    # axs[0].set_xlim(-plot_scale * 1.15, plot_scale * 1.15)
    # axs[0].set_ylim(-plot_scale * 1.15, plot_scale * 1.15)
    axs[0].set_xlabel(r"$\vec{v} \times \vec{B}$ [m]")# [$r/r_0$]")
    axs[0].set_ylabel(r"$\vec{v} \times (\vec{v} \times \vec{B})$ [m]")# [$r/r_0$]")
    axs[0].set_title(r"$\vec{v} \times \vec{B}$ fluence in" +  "\n shower plane", fontsize=16)
    axs[0].set_title(r"No correction", fontsize=16)

    
    # second version of first plot, comment in if appropriate
    
    # # in this version, we plot the antenna grid on the ground plane

    # # option for realistic footprint on ground plane if applicable
    # (x_data, y_data, values) = (pos_xy[:, 0], pos_xy[:, 1], f_ground)

    # # plot antennas on the ground plane
    # real_antenna_grid = axs[0].scatter(pos_xy[:, 0], pos_xy[:, 1], s=40, c=f_ground, alpha=1, cmap=cmap)
 
    # # assign colorbar to this subplot
    # cbar = fig.colorbar(real_antenna_grid, ax=axs[0], cax=cax[0], format=tick.FormatStrFormatter('%.0f'))
    # cbar.set_label(r"$f_{\vec{v} \times \vec{B}}$[$\mathrm{keVm}^{-2}$]")

    # # set limits
    # axs[0].set_xlim(-3000, 3000)
    # axs[0].set_ylim(-3000, 3000)

    # # assign axis labels and subplot title
    # axs[0].grid()
    # axs[0].set_xlabel("West <--> East [m]")
    # axs[0].set_ylabel("South <--> North [m]")
    # axs[0].set_title(r"Raw $\vec{v} \times \vec{B}$ fluence on ground plane")
    

    #
    # all for axs[1], the second plot
    #
    # data grid points and values for vxB fluence
    (x_data, y_data, values) = (vxB_axis, vxvxB_axis, f_vxB / max(f_vxB))
    # (x_data, y_data, values) = (vxB_axis * r_cheren, vxvxB_axis * r_cheren, f_vxB / max(f_vxB))

    # interpolated grid
    resampled = griddata((x_data,y_data), values, (XI,YI), method='cubic')
    
    # generate and assign interpolation data to the interpolated grid
    fluence_map = axs[1].imshow(resampled.T, origin='lower', cmap=cmap, interpolation=interp[0], alpha=1, \
                             extent=[-dist_scale, dist_scale, -dist_scale, dist_scale]) # , vmin=0, vmax=cmax)
 
    # contour plot to show the high  intensity region
    grav_pot = axs[1].contour(XI, YI, resampled, colors='white', alpha = 0.4, linewidths=2, \
                levels = np.linspace(max(resampled.flatten()) * 0.89999, max(resampled.flatten()) * 0.9, 2))
    # axs[1].clabel(grav_pot, fontsize=12)

    # assign colorbar to this subplot
    cbar = fig.colorbar(fluence_map, ax=axs[1], cax=cax[1], format=tick.FormatStrFormatter('%.1f'))
    # cbar.set_label(r"Energy fluence [ev/m$^2$]")
    cbar.set_label("Signal intensity [a. u.]") # (r"$f_{\vec{v} \times \vec{B}} \cdot c_{\mathrm{EL}}^2$ [ev/m$^2$]")
    # cbar.ax.tick_params(labelsize=10)

    # assign axis labels and subplot title
    axs[1].set_xlabel(r"$\vec{v} \times \vec{B}$ [m]") # [$r/r_0$]")
    axs[1].set_ylabel(r"$\vec{v} \times (\vec{v} \times \vec{B})$ [m]") # [$r/r_0$]")
    # axs[1].set_title(r"Early-late corrected" + "\n" + r"$\vec{v} \times \vec{B}$ fluence", fontsize=16)
    axs[1].set_title("Corrected\npositions & fluence", fontsize=16)
    


    #
    # all for axs[2], the third plot
    #
    # data grid points and values for vxB fluence
    (x_data, y_data, values) = (vxB_axis, vxvxB_axis, f_geo_param)
    # (x_data, y_data, values) = (vxB_axis, vxvxB_axis, f_geo_posi)

    # interpolated grid
    resampled = griddata((x_data,y_data), values, (XI,YI), method='cubic')

    # print(resampled)
    
    
    # generate and assign interpolation data to the interpolated grid
    fluence_map = axs[2].imshow(resampled.T, origin='lower', cmap=cmap, interpolation=interp[0], alpha=1, \
                             extent=[-dist_scale, dist_scale, -dist_scale, dist_scale]) # , vmin=0, vmax=cmax)
 
    # contour plot to show the high  intensity region
    grav_pot = axs[2].contour(XI, YI, resampled, colors='white', alpha = 0.4, \
                              levels = np.linspace(max(values) * 0.869999, max(values) * 0.87, 2))

    # assign colorbar to this subplot
    cbar = fig.colorbar(fluence_map, ax=axs[2], cax=cax[2], format=tick.FormatStrFormatter('%.0f'))
    cbar.set_label(r"Energy fluence [ev/m$^2$]")
    # cbar.set_label(r"$f_\mathrm{geo}^{\,\mathrm{par}}\,$[ev/m$^2$]")
    # cbar.ax.tick_params(labelsize=8)

    # assign axis labels and subplot title
    axs[2].set_xlabel(r"$\vec{v} \times \vec{B}$ [m]")# [$r/r_0$]")
    # axs[2].set_ylabel(r"$\vec{v} \times (\vec{v} \times \vec{B})$ [$r/r_0$]")
    axs[2].set_title("Geomagnetic\ncomponent", fontsize=16)

    '''
    #
    # all for axs[3], the fourth plot
    #
    # data grid points and values for vxB fluence
    (x_data, y_data, values) = (vxB_axis, vxvxB_axis, f_ce_param)

    # interpolated grid
    resampled = griddata((x_data,y_data), values, (XI,YI), method='cubic')
    
    # generate and assign interpolation data to the interpolated grid
    fluence_map = axs[3].imshow(resampled.T, origin='lower', cmap=cmap, interpolation=interp[0], alpha=1, \
                             extent=[-dist_scale, dist_scale, -dist_scale, dist_scale]) #, vmin=0, vmax=cmax)
 
    # assign colorbar to this subplot
    cbar = fig.colorbar(fluence_map, ax=axs[3], cax=cax[3], format=tick.FormatStrFormatter('%.0f'))
    cbar.set_label(r"$f_\mathrm{ce}^\mathrm{par} \cdot d_{max}^2$ [$\mathrm{GeV}$]")
    # cbar.ax.tick_params(labelsize=8)

    # assign axis labels and subplot title
    axs[3].set_xlabel(r"$\vec{v} \times \vec{B}$ [$r/r_0$]")
    axs[3].set_ylabel(r"$\vec{v} \times (\vec{v} \times \vec{B})$ [$r/r_0$]")
    axs[3].set_title("Charge excess emission \n (parametric)")
    '''

    # puts plots closer together
    plt.tight_layout()
    plt.savefig(r"fluence_map_frame_00%i.png" %frame_number)
    plt.show()
    plt.close()
