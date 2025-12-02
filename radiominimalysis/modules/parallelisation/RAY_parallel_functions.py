import argparse
import os
import re
import sys

import numpy as np
import ray

from radiominimalysis.framework import factory as eventfactory
from radiominimalysis.framework.parameters import (
    eventParameters as evp,
    showerParameters as shp,
    stationParameters as stp,
)
from radiominimalysis.modules.reconstruction import geometry, ldf_fitting, signal_emissions
from radiominimalysis.utilities import helpers, refractive_displacement

# function to parallely use the code that takes a long time
# for example fits and reconstructions and anything that processes simulations
# returns processed event factory and optionally saves them as a file


def ray_parallelisation(args):

    # define functions usable in parallelisation
    functions = {
        "geometry": geometry.reconstruct_geometry_ray,
        "prefit_calculations": calculate_geometry_and_signal_emissions_ray,
        "fit_from_pickle_pos": only_fit_pos_has_ldf_E_geo_ray,
        "fit_from_pickle_param": only_fit_param_has_ldf_E_geo_ray,
        "core_prediction": get_predicted_core_displacement_ray,
        "root_reader": read_root_file_ray
    }

    # usable shower types in enum
    shower_types = {
        "rd_shower": evp.rd_shower,
        "sd_shower": evp.sd_shower,
        "sim_shower": evp.sim_shower,
        "GRAND_shower": evp.GRAND_shower
    }

    print("Use {} as shower type!".format(args.shower))

    # converting showertype string to enum name of showertype
    args.shower = shower_types[args.shower]
    # formatting label, if not empty add a trailing "_"
    helpers.format_label(args)
    # events = []

    # check if there are paths given
    if not len(args.paths) > 0:
        sys.exit("No valid input data.")

    # initialise factory object for saving results
    factory = eventfactory.EventFactory()

    # check if input file is pickle
    if args.paths[0][-7:] == ".pickle":

        for idx, path in enumerate(args.paths):
            # read events from .pickle file into factory
            factory.read_events_from_file(path)

    # check if input is .hdf5 file
    elif args.paths[0][-5:] == ".hdf5":
        from radiominimalysis.input_output import coreas_reader

        # read data from simulation file
        reader = coreas_reader.readCoREASShower(
            input_files=args.paths, verbose=args.verbose
        )
        # and save it in the factory
        reader.get_factory(factory)

    # for DC2 simulations
    # check if input is directory
    elif os.path.isdir(args.paths[0]):

        from radiominimalysis.input_output.root_reader import ReadRootInput
        # from radiominimalysis.input_output.root_reader import read_root_file
        

        # use ROOT reader tool to read out the showers from the input files
        # print(args.paths[0])
        reader = ReadRootInput(args.paths)

        # and save it in the factory
        # reader.get_factory(factory)


    # check if input data is .root file
    elif args.paths[0][-5:] == ".root":
        from radiominimalysis.input_output import adst_reader

        # read data from simulation file
        reader = adst_reader.readADSTEvent(
            input_files=args.paths, reference="SD", verbose=args.verbose
        )
        # and save it in the factory
        reader.get_factory(factory)

    # abort if no suitable input files are found
    else:
        sys.exit("File type unsupported: %s" % args.paths[0])


    # only do xmax selection if file read in is not a root file
    if not os.path.isdir(args.paths[0]):
        # save read out events in array
        events = factory.get_events()
        mask_has_xmax = eventfactory.has_parameter(events, shp.xmax)
        events = events[mask_has_xmax]
        print("Read %d event(s)." % len(events))

        # read xmax data
        xmax = eventfactory.get_parameter(events, shp.xmax)
        # make mask to skip sims where no xmax was found
        mask_xmax =np.invert(np.isnan(xmax))
        print("Mean of Xmax: ", np.mean(np.array(xmax[mask_xmax])))
        # print(mask_xmax)
        
        # delete those wrong sims from the event list
        events = events[mask_xmax]

        print("Read %d event(s) with Xmax." % len(events))

        [revent.set_default_shower_type(args.shower) for revent in events]
        
    # select for specific energies and zenith angles
    if 0:
        # get zenith and energy data
              
        Eem = eventfactory.get_parameter(events, shp.electromagnetic_energy)
        
        # make mask for both criteria
        # mask_both = np.all([np.rad2deg(zenith) <= 70.5, np.rad2deg(zenith) >= 69.5, Eem > 8 * 10 ** 17, Eem < 1.3 * 10 ** 18], axis=0) # GP300
        # mask_both = np.all([np.rad2deg(zenith) >= 82, Eem > 5 * 10 ** 19], axis=0) # GRAND10k
        
        # print(np.sum(np.array([len(ev.get_station_parameter(stp.energy_fluence)[:, 0]) > 11  for ev in events])))
        
        mask_ef = np.array([ev.has_station_parameter(stp.energy_fluence)  for ev in events])
        events = events[mask_ef]
        zenith = eventfactory.get_parameter(events, shp.zenith)
        
        mask_both = np.all([np.rad2deg(zenith) >= 65, np.array([len(ev.get_station_parameter(stp.energy_fluence)[:, 0]) > 11  for ev in events])], axis=0)
        
        events = events[mask_both]
        print(f"Read {len(events)} events with zenith angle and Eem in desired range!")
        
        
     # select for specific energies and zenith angles
    if 0:
        # desired event number or numbers
        # GP300
        run_want = [1]
        event_want = [103006]
        
        # 10k
        # run_want = [1]
        # event_want = [112999]
        
        run_no = np.array([ev.get_run_number() for ev in events])
        evt_no = np.array([ev.get_id() for ev in events])
        print(run_no, evt_no)
        # make mask for both criteria
        mask_both = np.all([run_no == run_want, evt_no == event_want], axis=0) # GP300
        # mask_both = np.all([np.rad2deg(zenith) <= 80, np.rad2deg(zenith) >= 85, Eem > 10 ** 19], axis=0) # GRAND10k
        
        events = events[mask_both]
        print(f"Reading event: {np.array([ev.get_run_number() for ev in events])}, {np.array([ev.get_id() for ev in events])}")

    # select specific event
    # only necessary if using .pickle files
    if args.event_id is not None:
        mask = np.any(
            [
                [
                    re.match("%06d" % id, "%06d" %
                             (ev.get_run_number())) is not None
                    for ev in events
                ]
                for id in args.event_id
            ],
            axis=0,
        )
        print("Selected %d event(s)" % np.sum(mask))

        # change event array to only contain selected event(s)
        events = events[mask]

    # allot resources for parallelisation
    memory_gb = max(int(args.parallel_jobs * 2 / 3), 1) * 1024 ** 3

    # start parallelisation with ray
    ray.init(
        num_cpus=args.parallel_jobs,
        _enable_object_reconstruction=True,
        _memory=memory_gb,
        object_store_memory=memory_gb,
    )

    # give the input parameters to ray
    para_ref = ray.put(args)

    # print("Input Parameters for parallelised function: ", para_ref)
    # print(args)

    # select function to be executed in parallel
    func = functions[args.function]

    print("Execute", func.__dict__["_function_name"])

    if args.function == "root_reader":
        outs = [func.remote(path, para_ref) for path in np.array(args.paths)]
        
        # and add them to the result array
        events = ray.get(outs)
        events = np.concatenate(events)

    else:
        # return results of the calculations
        outs = [func.remote(e, para_ref) for e in events]

        # and add them to the result array
        events = ray.get(outs)


    print("Executed", func.__dict__["_function_name"])

    if args.save:
        # if no argument is passed to --save
        if not isinstance(args.save, str):
            # if more than one input file is passed or it's not in .pickle format
            if not len(args.paths) == 1 or args.paths[0][-7:] != ".pickle":
                sys.exit("No valid file name to store factory: %s" %
                         args.paths[0])

            # name result file same as input .pickle
            fname = args.paths[0]

        # if a name is given with --save
        else:
            if args.save[-7:] != ".pickle":
                args.save += ".pickle"

            # save pickle file in factories directory
            # fname = os.path.join("factories/", args.save)
            fname = os.path.join("/cr/aera02/huege/guelzow/factories_thesis", args.save)

        # make a new factory with the events update with results
        new_factory = eventfactory.EventFactory(events=events)
        # and save the factory to the pickle
        new_factory.save_events_to_file(fname)

    print(f"Parallel-processed {len(events)} events!")
    # return updated event list
    return np.array(events)

# parallelise function execution with ray
# function to fit over LDFs of all simulations and
# return plots of the fitted LDF function and the Gauss-Sigmoid parameters


@ray.remote
def LDF_fit_fgeo_pos_ray(revent, para, at=None):

    # reconstruct geometric parameters (angles, positions, etc.)
    geometry.reconstruct_geometry_revent(revent, para, at=at)

    # finds out which pulses are affected by thinning
    geometry.find_overestimated_signals_frequency_slope_revent(revent, para)

    # calculation of fluences from station position (not parametrisations)
    signal_emissions.reconstruct_geomagnetic_and_charge_excess_emission_from_position_revent(
        revent, para
    )

    # fitting the LDF
    ldf_fitting._fit_pos_has_ldf_E_geo(revent, para)
    return revent


# parallelise function execution with ray
# function to do all the steps that lead up to the fit
# return event with all parameters needed for the fit
@ray.remote
def calculate_geometry_and_signal_emissions_ray(revent, para, at=None):

    # reconstruct geometric parameters (angles, positions, etc.)
    geometry.reconstruct_geometry_revent(revent, para, at=at)

    # if not para.realistic_input:
    #     # finds out which pulses are affected by thinning
    #     geometry.find_overestimated_signals_frequency_slope_revent(revent, para)

    #     # calculation of fluences from station position (not parametrisations)
    #     signal_emissions.reconstruct_geomagnetic_and_charge_excess_emission_from_position_revent(revent, para)

    return revent


# parallelise function execution with ray
# perform only the fit
# this needs all the necessary parameters to be calculated before
@ray.remote
def only_fit_pos_has_ldf_E_geo_ray(revent, para):

    # fitting the LDF
    ldf_fitting._fit_pos_has_ldf_E_geo(revent, para)
    return revent

# parallelise function execution with ray
# perform only the fit (with parametrisation f_geo)
# this needs all the necessary parameters to be calculated before
@ray.remote
def only_fit_param_has_ldf_E_geo_ray(revent, para):

    # fitting the LDF
    ldf_fitting._fit_param_has_ldf(revent, para)
    return revent

# parallelise function execution with ray
# get core predictions
# this needs all the necessary parameters to be calculated before
@ray.remote
def get_predicted_core_displacement_ray(revent, args):

    # fitting the LDF
    refractive_displacement.get_predicted_core_displacement(revent, args)
    return revent

# parallelise function execution with ray
# read in root files
@ray.remote
def read_root_file_ray(path, args):
    
    from radiominimalysis.input_output.root_reader import read_root_file

    # and save it in the factory
    event_list = read_root_file([path], args)
    
    return event_list