# from numpy.lib.shape_base import tile
from radiominimalysis.utilities import (
    ldfs,
    early_late,
    cherenkov_radius,
    refractive_displacement,
)
from radiominimalysis.framework.parameters import (
    showerParameters as shp,
    stationParameters as stp,
    eventParameters as evp,
)
from radiominimalysis.modules.reconstruction import geometry

from radiominimalysis.modules.reconstruction.ldf_fit_functions import (
    objective_ldf_geo_pos,
    objectiv_ldf_has_param,
)
from radiominimalysis.modules.reconstruction.ldf_fitting import fit_param_has_ldf as LDF_fit_param
from radiominimalysis.modules.reconstruction.iminuit_wrapper import MyMinuitMinimizer
from radiominimalysis.modules.reconstruction.ldf_plotting import plot_ldf

from radiotools.atmosphere import models as atm

from matplotlib import pyplot as plt

import lmfit
import numpy as np
import argparse
import os
import sys
import copy
import re
import ray
import functools


parser = argparse.ArgumentParser(description="")

# parser asks for hdf5 input files to plot results from
parser.add_argument(
    "paths",
    metavar="PATH",
    type=str,
    nargs="*",
    default=[],
    help="Choose hdf5 input file(s).",
)

# parser asks for atmosphere model to use in calculation
parser.add_argument(
    "-m",
    "--atmModel",
    metavar="int",
    type=int,
    nargs="?",
    default=1,
    help="Set the model id for atmospheric model",
)

parser.add_argument(
    "-p", "--plot", action="store_true", help="Plot results (default: false)"
)

parser.add_argument(
    "--label",
    metavar="label",
    type=str,
    nargs="?",
    default="",
    help="Set the plot label",
)

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

parser.add_argument(
    "-rmif",
    "--remove_infill",
    action="store_true",
    default=None,
    help="Whether to remove infill antennas(default: False)"
)

parser.add_argument(
    "-oif",
    "--only_infill",
    action="store_true",
    default=None,
    help="Whether to only use infill antennas(default: False)"
)

parser.add_argument('--thinning_cut', nargs='?', type=float, default=0.85,
                    help="(default: 0.85)")

# read arguments from the command line after the function itself
args = parser.parse_args()

print(f"Input: {args.paths}")

# check if input is .hdf5 file
if args.paths[0][-5:] == ".hdf5":
    from radiominimalysis.input_output import coreas_reader

    # read data from simulation file
    reader = coreas_reader.readCoREASShower(input_files=args.paths)

# for DC2 simulations
# check if input is directory
elif os.path.isdir(args.paths[0]):
    from radiominimalysis.input_output import root_reader

    # use CoREAS reader tool to read out the showers from the input files
    # print(args.paths[0])
    reader = root_reader.ReadRootInput(args.paths)

# check if input is GP80 data file
elif args.paths[0][-5:] == ".root":
    from radiominimalysis.input_output import root_reader

    # read data from data file
    reader = root_reader.ReadRootInput(args.paths)


for revent in reader.run():

    for shower in revent.get_showers():
        shower_type = shower.get_shower_type()

        shower = revent.get_shower(key=shower_type)

    if 1: # shower.has_parameter(shp.xmax):
        if 1: # not np.isnan(shower.get_parameter(shp.xmax)):

            geometry.reconstruct_geometry([revent], args)

            # geometry.find_overestimated_signals_frequency_slope_revent(revent, args)

            LDF_fit_param([revent], args)
