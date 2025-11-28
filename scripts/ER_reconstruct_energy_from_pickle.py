import argparse

import numpy as np

# from radiominimalysis.input_output import coreas_reader
from radiominimalysis.modules.method_evaluation.gauss_sigmoid_param import (
    evaluate_gauss_sigmoid_pars as GS_param
)
from radiominimalysis.modules.method_evaluation import ldf_evaluation as LDF_eval
from radiominimalysis.modules.reconstruction import geometry
from radiominimalysis.modules.reconstruction import signal_emissions
# from radiominimalysis.modules.reconstruction import RAY_parallel_functions

# from radiominimalysis.modules.ADSTanalysis import efficiency as eff

from matplotlib import pyplot as plt

from radiominimalysis.framework import factory
from radiominimalysis.framework.parameters import showerParameters as shp

from radiominimalysis.utilities import parsers

# get command line arguments
# read out in separate function
args = parsers.fit_arguments()

#print(args)

# initialise factory object for saving results
readout_factory = factory.EventFactory()

readout_factory.read_events_from_file(args.paths[0])

events = readout_factory.get_events()

# get preprocessed events from .pickle file
# and parallelise processing them
# events = RAY_parallel_functions.ray_parallelisation(args)

# evaluate energy reconstruction
# LDF_eval.evaluate_egeo(events, args)

LDF_eval.evaluate_fit_result(events, args)

# eff.plot_aperture_and_event_numbers(events, args)

# LDF_eval.evaluate_dmax_fit(events, args)

# GS_param(events, args)

# LDF_eval.evaluate_gauss_sigmoid_core(events, args)

if args.save:
    print("Saved plot performance and Gauss-Sigmoid parameters in files")
