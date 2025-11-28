import argparse
import numpy as np

from radiominimalysis.framework import factory
from radiominimalysis.input_output import coreas_reader
from radiominimalysis.framework.parameters import showerParameters as shp, eventParameters as evp, \
    stationParameters as stp
from radiominimalysis.modules.CoREASanalysis import charge_excess_parametrization, charge_excess_validation, gauss_sigmoid_param, ldf_evaluation

from radiominimalysis.utilities import cherenkov_radius, charge_excess as ce
from radiominimalysis.utilities import helpers as rdutils

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from radiominimalysis.modules.reconstruction import geometry, signal_emissions

parser = argparse.ArgumentParser(description='')

parser.add_argument('paths', metavar='PATH', type=str, nargs='*', default=[],
                    help='Choose hdf5 input file(s).')

parser.add_argument(
    '-m', '--atmModel', metavar='int', type=int, nargs='?', default=None,
    help='Set the model id for atmospheric model')

parser.add_argument('--thinning_cut', nargs='?', type=float, default=0.85,
                    help="(default: 0.85)")

# parser asks for atmosphere model to use in calculation
parser.add_argument('-gd', '--gdasFile', metavar='PATH', type=str, nargs='*', default=None,
                    help='Choose gdas atmosphere file.')

parser.add_argument(
    "-real",
    "--realistic_input",
    action="store_true",
    default=None,
    help="Whether realistic simulation is the input (default: False)"
)

args = parser.parse_args()

eventfactory = factory.EventFactory()

reader = coreas_reader.readCoREASShower(args.paths)
reader.get_factory(eventfactory)
events = eventfactory.get_events()
reader.end()

print("Read %d event(s)." % len(events))

# read xmax data
xmax = factory.get_parameter(events, shp.xmax)

# make mask to skip sims where no xmax was found
mask_xmax = np.invert(np.isnan(xmax))
print(mask_xmax)
    
# delete those wrong sims from the event list
events = events[mask_xmax]

print("Read %d event(s) with Xmax." % len(events))

# calculates among other things:
# distance to Xmax: shp.distance_to_shower_maximum_geometric
geometry.reconstruct_geometry(events, args)

geometry.find_overestimated_signals_frequency_slope(events, args)

# calculation of fluences from station position
signal_emissions.reconstruct_geomagnetic_and_charge_excess_emission_from_position(
    events, args)

# calculation by of CE parametrisation
signal_emissions.reconstruct_parametric_charge_excess_fraction(events, None)

# get geomagnetic fluence from parametrisation
signal_emissions.reconstruct_emission_from_param(events, None)


# positional fluences
f_geo_pos = factory.get_parameter(events, stp.geomagnetic_fluence_positional)
f_ce_pos = factory.get_parameter(events, stp.charge_excess_fluence_positional)

# parametric fluences
f_geo_param = factory.get_parameter(events, stp.geomagnetic_fluence_parametric)
f_ce_param = factory.get_parameter(events, stp.charge_excess_fluence_parametric)

# calculate cherenkov radii
r0 = np.array([cherenkov_radius.get_cherenkov_radius_model_revent(e)
               for e in events])

# station axis distance
r_stations = np.array([ev.get_station_axis_distance() for ev in events])
# print(r_stations.shape)

# avoid pulses affected by thinning
thinning_clean_masks = factory.get_parameter(events, stp.cleaned_from_thinning)
# print(thinning_clean_masks[0:10])

# get early-late correction factors
c_early_late = factory.get_parameter(events, stp.early_late_factor)

# apply early-late correction
r_stations = r_stations / c_early_late
print(r_stations.flatten().shape)

# f_geo_param is per definition already early-late corrected
f_geo_pos *= c_early_late ** 2
f_ce_pos *= c_early_late ** 2

# define mask for leaving thinning affected pulses out
mask = np.array(
    np.all([f_geo_pos > 0, thinning_clean_masks],
           axis=0),
    dtype=bool)

# deviation of positional and parametric f_geo
deviation_f_geo = (f_geo_pos - f_geo_param) / f_geo_pos

# print(mask.shape)
# print(deviation_f_geo.shape)

# eliminate outliers and print number of them
div_tmp = deviation_f_geo[mask]
mask2 = np.abs(div_tmp) > 0.3
out = np.sum(mask2)

if 1:
    print("Antennas affected by thinning: ", np.sum(~mask), mask.flatten().shape)
    print("Cut outliers >0.3", out, mask2.flatten().shape)
    mask = np.all([mask, np.abs(deviation_f_geo) < 0.3], axis=0)

# number of stations per event not eliminated by mask
n_stations_per_event = [np.sum(x) for x in mask]

# assign the event cherenkov radius to every detector in that event
r0_rep = np.repeat(r0, n_stations_per_event)
# print(r0_rep.shape)

# apply mask to deviation data
deviation_f_geo = deviation_f_geo[mask]

# bin edges on vertical axis
div_binning = np.linspace(-0.3, 0.3, 61)

# mid point of bins on horizontal axis
bins_f_pos = np.logspace(-1, 6, 20)
bins_r0 = np.linspace(0, 2.5, 25)

####################################################################

# PLOTS

# differences in the plots so far could be because of a cut with geomagnetic angle > 20°

plt.rcParams.update({'font.size': 16})

# subplots
fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(11, 5))

# adjust space around figure edges
fig.subplots_adjust(wspace=0.2, left=0.1, right=0.95, bottom=0.15)

# set title for figure
fig.suptitle(r'GP300', fontsize=20)

axs[0].set_ylabel(
    r'$\left(f\,_\mathrm{geo}^\mathrm{pos} - f\,_\mathrm{geo}^\mathrm{par}\right) / f\,_\mathrm{geo}^\mathrm{pos}$',
    fontsize=16)
axs[0].set_xscale('log')
axs[0].set_xlabel(
    r'Analytical Fluence $f\,_\mathrm{geo}^\mathrm{pos}$ [$\mathrm{eV}\,\mathrm{m}^{-2}$]',
    fontsize=16)

# 2D histogram of the data
# x and y edges are the bins, z is the amount of entries in each bin in arrays representing the vertical columns
bin_freq, xedges, yedges = np.histogram2d(
    f_geo_pos[mask], deviation_f_geo, bins=[bins_f_pos, div_binning])

# to include the highest value in the bin
yedges[-1] += 1e-6
# get center values
center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

bins_f_pos[-1] += 1e-6
bin_centers = bins_f_pos[:-1] + (bins_f_pos[1:] - bins_f_pos[:-1]) / 2

# calculate mean of each colum
mean_pos = np.array(
    [np.sum(
        np.array(
            [center_values[j] * bin_freq[i][j]
             for j in range(len(bin_freq[i]))])) / np.sum(bin_freq[i])
     for i in range(len(bin_freq))])


# calculate standard deviation
std_dev = np.sqrt(
    np.array(
        [np.sum(
            np.array(
                [bin_freq[i][j] * (center_values[j] - mean_pos[i]) ** 2
                 for j in range(len(bin_freq[i]))])) /
         (np.sum(bin_freq[i]) - 1) for i in range(len(bin_freq))]))

print("Mean: ", mean_pos)
print("Standard deviation: ", std_dev)

# add means and standard deviations to the 2D histograms as points and errorbars
axs[0].errorbar(bin_centers, mean_pos, std_dev, marker='s',
                color='black', markerfacecolor="red", markeredgewidth=1,  linestyle='', ms=4, label='1')

# transpone z
bin_freq = bin_freq.T

# assign colour values to the bins
pcm = axs[0].pcolormesh(
    xedges, yedges, bin_freq, shading='flat', norm=mpl.colors.LogNorm(),
    cmap="plasma")

cbi = plt.colorbar(pcm, orientation='vertical', pad=0.02, ax=axs[0])
cbi.ax.tick_params(axis='both', labelsize=16)
cbi.set_label('# Antennas', fontsize=16)

# add grid to figure as well as title
axs[0].grid(True)
# axs[0].set_title("Comparison vs pos. Fluence", fontsize=10)

# 2nd plot
# cherenkov radius plot
bin_freq, xedges, yedges = np.histogram2d(
    r_stations[mask] / r0_rep, deviation_f_geo, bins=[bins_r0, div_binning])

print(r_stations[mask].shape)

# to include the highest value in the bin
yedges[-1] += 1e-6
# get center values
center_values = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2

bins_r0[-1] += 1e-6
bin_centers = bins_r0[:-1] + (bins_r0[1:] - bins_r0[:-1]) / 2

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

print("Mean2: ", mean_pos)
print("Standard deviation2: ", std_dev)

# add means and standard deviations to the 2D histograms as points and errorbars
axs[1].errorbar(bin_centers, mean_pos, std_dev, marker="s",
                color='black', markerfacecolor="red", markeredgewidth=1,  linestyle='', ms=4, label='1')

bin_freq = bin_freq.T

axs[1].set_ylabel(
    r'$\left(f\,_\mathrm{geo}^\mathrm{pos} - f\,_\mathrm{geo}^\mathrm{par}\right) / f\,_\mathrm{geo}^\mathrm{pos}$',
    fontsize=16)
axs[1].set_xscale('linear')
axs[1].set_xlim(0, 2.0)
axs[1].set_xticks([0, 0.5, 1, 1.5, 2])
axs[1].set_ylim(-0.3, 0.3)
axs[1].set_xlabel('Axis Distance $r$ / $r_0$', fontsize=16)

pcm = axs[1].pcolormesh(xedges, yedges, bin_freq,
                        shading='flat', norm=mpl.colors.LogNorm(), cmap="plasma")

axs[1].grid(True)
# axs[1].set_title("GP300", fontsize=12)

cbi = plt.colorbar(pcm, orientation='vertical', pad=0.02, ax=axs[1])
cbi.ax.tick_params(axis='both', labelsize=16)
cbi.set_label('# Antennas', fontsize=16)
# axs[0].legend()

plt.tight_layout()
plt.savefig("CE_evaluation_GP300.pdf")
