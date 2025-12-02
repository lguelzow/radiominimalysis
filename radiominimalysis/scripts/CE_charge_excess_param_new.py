from __future__ import print_function
from radiominimalysis.framework.parameters import showerParameters as shp, stationParameters as stp
from radiominimalysis.framework import factory

from radiominimalysis.utilities import helpers, plthelpers, stats
from radiominimalysis.utilities import  charge_excess as ce
from radiominimalysis.utilities import cherenkov_radius

from radiominimalysis.input_output import coreas_reader

from radiominimalysis.modules.method_evaluation import pyplots, pyplots_utils
from radiominimalysis.modules.reconstruction import geometry, signal_emissions

from radiotools import helper as rdhelp

import iminuit 

from matplotlib import pyplot as plt
from scipy import optimize
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('paths', metavar='PATH', type=str, nargs='*', default=[],
                    help='Choose hdf5 input file(s).')

parser.add_argument(
    '-m', '--atmModel', metavar='int', type=int, nargs='?', default=None,
    help='Set the model id for atmospheric model')

parser.add_argument('--thinning_cut', nargs='?', type=float, default=0.85,
                    help="(default: 0.85)")

# parser asks for atmosphere model to use in calculation
parser.add_argument(
    "-gd",
    "--gdasFile",
    metavar="PATH",
    type=str,
    nargs="*",
    default=None,
    help="Choose gdas atmosphere file.",
)            

parser.add_argument(
    "-real",
    "--realistic_input",
    action="store_true",
    default=None,
    help="Whether realistic simulation is the input (default: False)"
)

eventfactory = factory.EventFactory()

args = parser.parse_args()

reader = coreas_reader.readCoREASShower(args.paths)
reader.get_factory(eventfactory)
events = eventfactory.get_events()
reader.end()

print("Read %d event(s)." % len(events))

# read xmax data
xmax = factory.get_parameter(events, shp.xmax)

# make mask to skip sims where no xmax was found
mask_xmax = np.invert(np.isnan(xmax))
    
# delete those wrong sims from the event list
events = events[mask_xmax]

print("Read %d event(s) with Xmax." % len(events))

# calculates among other things:
# distance to Xmax: shp.distance_to_shower_maximum_geometric
geometry.reconstruct_geometry(events, args)

geometry.find_overestimated_signals_frequency_slope(events, args)

# calculation of fluences from station position
signal_emissions.reconstruct_geomagnetic_and_charge_excess_emission_from_position(events, args)


def parameterize_charge_excess_fraction(events, para):
    
    # cut events with small geomagnetic angles
    if 1:
        alpha = factory.get_parameter(events, shp.geomagnetic_angle)
        mask = alpha > 0.35
        print(f"Cut {np.sum(~mask)} events with geomagnetic angle < 20°")
        events = events[mask]

    f_geo = factory.get_parameter(events, stp.geomagnetic_fluence_positional)
    f_ce = factory.get_parameter(events, stp.charge_excess_fluence_positional)
    alpha = factory.get_parameter(events, shp.geomagnetic_angle)
    charge_excess_fractions = np.sin(alpha[:, None]) ** 2 * f_ce / f_geo

    c_early_late = factory.get_parameter(events, stp.early_late_factor, dtype=float)
    thinning_clean_masks = factory.get_parameter(events, stp.cleaned_from_thinning)

    # considers changed core!
    r_stations = np.array([ev.get_station_axis_distance() for ev in events], dtype=float)
    r_stations = (r_stations / c_early_late).astype(float)

    # calculate cherenkov radius
    r_cheren = np.array([cherenkov_radius.get_cherenkov_radius_model_revent(ev) for ev in events], dtype=float)

    dxmaxs_geo = factory.get_parameter(events, shp.distance_to_shower_maximum_geometric)
    density = factory.get_parameter(events, shp.density_at_shower_maximum)

    off_axis_angle = np.array(r_stations / dxmaxs_geo[:, None], dtype=float)

    # Set Weights here! One can use the std of the data on one ring (calculate_mean_ce_fraction_and_weights)
    charge_excess_fraction_weights = np.ones(off_axis_angle.shape)

    # charge_excess_fractions (_weights) is < 0 on axis where it is not vaild
    mask = np.array(np.all([charge_excess_fractions > 0, 
                   f_geo > 0, charge_excess_fraction_weights > 0,
                   thinning_clean_masks], axis=0), dtype=bool)
    n_stations_per_event = [np.sum(x) for x in mask]

    # masking also flattens
    off = np.rad2deg(off_axis_angle)[mask]
    charge_excess_fractions = charge_excess_fractions[mask]
    r_stations = r_stations[mask]
    charge_excess_fraction_weights = charge_excess_fraction_weights[mask]

    # print(off.shape)
    # print(np.max(off))

    # expand arrays to station-based size
    rho_xmaxs_repeat = np.repeat(density, n_stations_per_event)
    dxmaxs_geo_rep = np.repeat(dxmaxs_geo, n_stations_per_event)
    r_cheren_rep = np.repeat(r_cheren, n_stations_per_event)
    alpha_rep = np.repeat(np.rad2deg(alpha), n_stations_per_event)
    # normalise axis distance with cherenkov radius
    r_norm = r_stations / r_cheren_rep


    # define cost function for chi^2 minimisation fit
    def cost(pars, xdata, ydata, func, weights=None):
        # depending on parameters given, add the appropriate missing ones
        if len(pars) == 3:
            # pars = [0.37313183, 1.31124484, *pars]
            # add parameters for dmax part of equation
            pars = [0.4141706 , 1.23891786, *pars]
        elif len(pars) == 2:
            # add parameters for density part of equation
            pars = [*pars, 1.66965694e+01, 3.31954904e+00, -5.73577715e-03]

        # predicted values of function
        ypred = func(xdata, *pars)

        # set weights
        if np.all(weights == None):
            weights = np.ones_like(ydata)
        
        # calculate chi
        chi = (ydata - ypred) / weights

        # return chi^2
        return np.sum(chi ** 2)

    # input data for fit
    xdata = [r_stations, dxmaxs_geo_rep, rho_xmaxs_repeat] # , r_cheren_rep]

    # print(xdata)

    # parameters from Lukas new fit for 50-200 MHz for Auger
    # use these for 50-200 MHz
    # param = [-1.37266723e-06, 3.02018018e-01, 1.46508803e+00, 1.31382072e+01, 2.98380964e+00, 1.78471809e-01]

    # parameters from Lukas new fit for 50-200 MHz for China
    # use these for 50-200 MHz for GP300 site
    param = [-9.03992069e-07, 2.28710354e-01, 1.62957071e+00, 1.77729341e+00, 1.42776016e+00, 1.66010236e-01]

    # these are the parameters given in Felix' thesis (see calender book (11.01.2023))
    # [-1.17523609e-06, 3.48154734e-01, 1.6068519502678418, 1.66965694e+01, 3.31954904e+00, -5.73577715e-03]

    # parametrisation function, definined in CE file, as well as start values for parameters
    #
    func, p0 = ce.charge_excess_fraction_icrc21, param

    
    # minimise loss function with iminuit
    # p0 are start values for parameters

    # minimize(fun, x0, args=())

    res = iminuit.minimize(cost, p0, args=(
        xdata, charge_excess_fractions, func, 
        charge_excess_fraction_weights))
    print(res)
    print('=================================')
    param = res.x
    print(param)
    if len(param) == 3:
        param = [0.37313183, 1.31124484, *param]
    elif len(param) == 2:
        param = [*param, 1.66965694e+01, 3.31954904e+00, -5.73577715e-03]        
 

    # colourmap = alpha_rep
    colourmap = dxmaxs_geo_rep / 1000

    plt.rcParams.update({'font.size': 15})

    pyplots.scatter_color_2d(
                r_norm, charge_excess_fractions, colourmap,
                title="GRAND@Auger Analytical", # cscale="log",
                cmap="viridis", clabel=r"$d_\mathrm{max}$ [km]",
                # ylabel=r"$a / \frac{p_0 \cdot e(p_1 \cdot (\rho - \rho_{avg}))}{\langle p_0 \cdot e(p_1 \cdot (\rho - \rho_{avg})) \rangle}$",
                xlabel=r"Axis distance $r/r_0$", ylabel=r"$a_\mathrm{ce}=\mathrm{sin}^2(\alpha)\cdot \frac{f_\mathrm{ce}}{f_\mathrm{geo}}$", # r"$a = \sin^2\alpha \cdot f_\mathrm{ce} / f_\mathrm{geo}$",
                ylim=(-0.001, 0.06), alpha=1,
                # scatter_kwargs={"edgecolor": "k"},
                # func={"x": distances_flatten, "y": fit_func(xdata, *fit_params), "fmt": "k^", "label": "Fit rho"},
                fname="ce_fraction_rho%s.png" % ('x'))

    pyplots.scatter_color_2d(
                r_norm, func(xdata, *param), colourmap,
                title="GRAND@Auger Parametrisation", # cscale="log",
                cmap="viridis", clabel=r"$d_\mathrm{max}$ [km]",
                xlabel=r"Axis distance $r/r_0$", ylabel=r"Parameterisation of $a_\mathrm{ce}$", # (\alpha, \rho_\mathrm{max}, d_\mathrm{max})$",
                ylim=(-0.001, 0.06), alpha=1,
                # scatter_kwargs={"edgecolor": "k"},
                # func={"x": distances_flatten, "y": fit_func(xdata, *fit_params), "fmt": "k^", "label": "Fit rho"},
                fname="ce_fraction_rho_%s.png" % ('param'))
    
    # save plotted data
    if 0:
        np.savez_compressed(f"charge_excess_fraction_China", r_norm, func(xdata, *param))


parameterize_charge_excess_fraction(events, None)
# plt.show()