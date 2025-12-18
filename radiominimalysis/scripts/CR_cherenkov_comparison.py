import argparse

import numpy as np

from radiominimalysis.input_output import coreas_reader
from radiominimalysis.modules.method_evaluation import gauss_sigmoid_param as GS_param

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

alpha = factory.get_parameter(events, shp.geomagnetic_angle)
mask = alpha > 0.5

print("Applying geomagnetic angle filter...")
print(f"Amount of events with sufficiently large geomagnetic angle: {len(events[mask])}/{len(events)}")
events = events[mask]

GS_param.evaluate_fitted_cherenkov_radius(events, args)