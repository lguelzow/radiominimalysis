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

# core = alpha = factory.get_parameter(events, shp.core)
# print(core[0:10])

if args.fit_results:

    # get some parameters of events of the reconstructed events for filtering
    alpha = factory.get_parameter(events, shp.geomagnetic_angle_recon)
    rho = factory.get_parameter(events, shp.density_at_shower_maximum)
    zenith = factory.get_parameter(events, shp.zenith)
    azimuth = factory.get_parameter(events, shp.azimuth)

    # filter out the smallest zenith angles (in radians)
    mask = alpha > 0.35

    print("Applying geomagnetic angle filter...")
    print(f"Amount of events with sufficiently large geomagnetic angle: {len(events[mask])}/{len(events)}")

    # execute function to evaluate the fits and plot the reduced chi-square
    LDF_eval.evaluate_gauss_sigmoid_chi(events[mask], args)

    # evaluate energy reconstruction
    # LDF_eval.evaluate_egeo(events[mask], args)

    # evaluate Gauss-Sigmoid parameters and plot their behaviour against dmax
    # GS_param(events[mask], args)

    LDF_eval.evaluate_fit_result(events[mask], args)

    # LDF_eval.evaluate_gauss_sigmoid_core(events[mask], args)

    # LDF_eval.evaluate_dmax_fit(events[mask], args)

    if args.save:
        print("Saved plot performance and Gauss-Sigmoid parameters in files")
