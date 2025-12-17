import numpy as np

from radiominimalysis.framework import factory
from radiominimalysis.framework.parameters import (
    showerParameters as shp,
)
from radiominimalysis.modules.reconstruction import (
    RAY_parallel_functions,
)
from radiominimalysis.utilities import parsers, energyreconstruction

from radiominimalysis.modules.method_evaluation import ldf_evaluation as LDF_eval
from radiominimalysis.modules.method_evaluation.gauss_sigmoid_param import (
    evaluate_gauss_sigmoid_pars as GS_param
)

from matplotlib import rc
rc('font', size = 20.0)

# get command line arguments
# read out in separate function
args = parsers.fit_arguments()

# get preprocessed events from .pickle file
events = RAY_parallel_functions.ray_parallelisation(args)
