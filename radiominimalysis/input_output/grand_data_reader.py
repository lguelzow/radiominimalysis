import radiominimalysis.framework.revent
import radiominimalysis.framework.shower
from radiominimalysis.framework.parameters import showerParameters as shp
from radiominimalysis.framework.parameters import stationParameters as stp
from radiominimalysis.framework.parameters import eventParameters as evp

from RadioAnalysis.input_output.VocToEfield import efield_reconstruction_from_ADC as efield_recon

from radiominimalysis.modules.reconstruction import geometry

from radiotools import helper as rdhelp, coordinatesystems

import matplotlib.pyplot as plt

from PWF_reconstruction.recons_PWF import PWF_semianalytical, cov_matrix

# these libraries will only work within the grandlib docker or conda environment
# from grand.grandlib_classes.grandlib_classes import *
# import grand.dataio.root_trees as groot

from grand.grandlib_classes import *
import grand.dataio as groot

from grand.geo.coordinates import (
    Geodetic,
    LTP,
    GRANDCS,
    CartesianRepresentation,
)  # RK

import numpy as np
from scipy.signal import hilbert
import math
import re
import os
import sys
import copy
import warnings
from datetime import timedelta
import argparse


def get_antenna_pos_from_GPS(antenna_file=False, origin=False, cart_file=False):
    """
    use data file with geodetic/GPS antenna positions to generate cartesian antenna positions for specified origin
    
    :param antenna_file: path to antenna file
    .param origin: coordinate system origin in GPS coordinates
    :param cart_file: generate text file with positions and IDs?
    
    returns: antenna positions as an array 
             antenna IDs
    """

    # read data from antenna file
    if file:
        file = antenna_file
    else:
        file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/antennas_gp65_full_GPS.txt", dtype = "str")
        # file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/list_antennas_GP41_GPS.txt", dtype = "str")

    # put antenna positions in [latitude, longitude, height] and antenna IDs in arrays
    antenna_positions_geodetic = np.array([[file[i, 2].astype(float), file[i, 1].astype(float), file[i, 3].astype(float)] for i in range(len(file))])
    antenna_ids = np.array([file[i, 0].astype(float) for i in range(len(file))])

    # define origin in same system
    if origin:
        origin_geoid = origin
    else:
        # DAQ room origin
        origin_geoid = Geodetic(latitude=40.99434, longitude=93.94177, height=1262)  

    # transforms antennas to cartesian coordinates in array reference frame
    antenna_positions_xyz = np.array([GRANDCS(arg=Geodetic(latitude=pos[0], longitude=pos[1], height=pos[2]), obstime="2024-01-01", location=origin_geoid) \
                                    for pos in antenna_positions_geodetic])

    # convert to standard array format
    antennas_xyz = np.array([[antenna_positions_xyz[i, 0, 0], antenna_positions_xyz[i, 1, 0], antenna_positions_xyz[i, 2, 0]] for i in range(len(antenna_positions_xyz))])

    # generate .txt of the antenna positions with IDs
    if cart_file:
        with open("cartesia_antenna_pos.txt", "w") as txt_file:
            for i in range(len(antennas_xyz)):
                txt_file.write(str(int(antenna_ids[i])) + " " + str(antennas_xyz[i, 0]) + " " + str(antennas_xyz[i, 1]) + " " + str(antennas_xyz[i, 2]) + "\n")
                
    return antennas_xyz, antenna_ids
    
    

def read_measurement_data_from_root(filename, event_and_run, origin=False, plot_adc_traces=False):
    """
    Read out event data from a single event in a file
    
    :param filename: path to measurement data file
    :param event_and_run: run & event nr of desired event
    :param origin: coordinate origin
    :param plot_adc_traces: Want to plot ADC traces?
    
    returns: adc_traces --- analogue-to-digital converter traces from measurement
             antennas_from_data --- cartesian positions of antennas
             ids_from_data --- antenna IDS
             arrival_times --- time of measurement at each antenna [ns]
    """
    
    if origin:
        origin_geoid = origin
    else:
        origin_geoid = Geodetic(latitude=40.99434, longitude=93.94177, height=1262)

    # initialise datafile class from grandlib
    d_input = groot.DataFile(filename)

    # print contents of file
    # d_input.print()

    # load adc and run info trees
    tadc = d_input.tadc
    trun = d_input.trun
    trawvoltage = d_input.trawvoltage

    #get the list of events
    events_list = tadc.get_list_of_events()
    nb_events = len(events_list)
    print("Number of events in file: ", nb_events)

    # load data of specific event
    tadc.get_event(event_and_run[0], event_and_run[1])
    trun.get_run(event_and_run[1])
    trawvoltage.get_event(event_and_run[0], event_and_run[1])
    print(event_and_run)

    # get ADC traces
    trace_ADC = np.asarray(tadc.trace_ch, dtype=np.float32)

    # get gps positions of the antennas from trawvoltage tree
    du_lat = np.asarray(trawvoltage.gps_lat, dtype=np.float32)
    du_long = np.asarray(trawvoltage.gps_long, dtype=np.float32)
    du_alt = np.asarray(trawvoltage.gps_alt, dtype=np.float32)

    # transforms antennas to cartesian coordinates in array reference frame
    antennas = np.array([GRANDCS(arg=Geodetic(latitude=du_lat[i], longitude=du_long[i], height=du_alt[i]), obstime="2024-01-01", location=origin_geoid) 
                                      for i in range(len(du_lat))])
    
    # antenna positions extracted from file with obs level added
    antennas_from_data = np.array([[antennas[i, 0, 0], antennas[i, 1, 0], antennas[i, 2, 0]] for i in range(len(antennas))])
    

    '''
    # in principle the right way to get the antennas from an event, but file are not set up right
    # comment out but use later!
    full_antennas = np.array(trun.du_xyz)
    full_antennas_geodetic = np.array(trun.du_geoid)
    full_ids = np.array(trun.du_id)

    ids_from_data = np.array(tadc.du_id)

    mask = np.isin(full_ids, ids_from_data)

    # get the antenna position of the event
    antennas_in_event = full_antennas[mask]
    '''

    # get antenna positions from external data file
    # IDs of the triggered antennas
    ids_from_data = np.array(tadc.du_id)
    
    # get antenna arrival times
    event_antenna_s = np.array(tadc.du_seconds) 
    event_antennas_ns = np.array(tadc.du_nanoseconds)

    # get total antenna times
    antenna_times = event_antenna_s - min(event_antenna_s)
    antenna_times =  event_antenna_s + (event_antennas_ns / 1e9)
    
    # different way to get antenna times
    du_s = event_antenna_s.flatten()
    du_s = (du_s - du_s.min()).astype(np.float64)
    du_ns = event_antennas_ns.flatten() / 1e9 
    du_trig = du_s + du_ns.astype(np.float64)
    arrival_times = du_trig


    # only take the last 3 entries of ADC trace
    # which are the NWU channels
    adc_traces = trace_ADC[:, 1:4]

    print("Extracted measurement data. Processing...")
    
    # reconstruct arrival direction
    zenith_rec, azimuth_rec = np.rad2deg(PWF_semianalytical(antennas_from_data, arrival_times))
    print("1. Rec. Arrival Direction", np.round(zenith_rec, 2), np.round(azimuth_rec, 2))


    if plot_adc_traces:

        # plot ADC traces
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(30, 10))
        ax[0].plot(range(len(trace_ADC[2][0])), trace_ADC[2][1], c="blue", label="Ch1")

        ax[1].plot(range(len(trace_ADC[2][0])), trace_ADC[2][2], c="red", label="Ch2")

        ax[2].plot(range(len(trace_ADC[2][0])), trace_ADC[2][3], c="black", label="Ch3")

        ax[0].set_xlabel("Time Samples [2ns]")
        ax[1].set_xlabel("Time Samples [2ns]")
        ax[2].set_xlabel("Time Samples [2ns]")
        ax[0].set_ylabel(r"ADC Counts")
        fig.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f'ADC_test.png')
        plt.close()


    # return all data relevant to reconstruction
    return adc_traces, antennas_from_data, ids_from_data, arrival_times



def read_measurement_data_from_root_directory(filename, origin=False, plot_adc_traces=False):
    """
    Read out event data from larger ROOT file, containing many events
    
    :param filename: path to measurement data file
    :param event_and_run: run & event nr of desired event
    :param origin: coordinate origin
    :param plot_adc_traces: Want to plot ADC traces?
    
    returns: adc_traces --- analogue-to-digital converter traces from measurement
             antennas_from_data --- cartesian positions of antennas
             ids_from_data --- antenna IDS
             arrival_times --- time of measurement at each antenna [ns]
             
    """
    
    if origin:
        origin_geoid = origin
    else:
        origin_geoid = Geodetic(latitude=40.99434, longitude=93.94177, height=1262)

    # initialise datafile class from grandlib
    d_input = groot.DataDirectory(filename)

    # print contents of file
    # d_input.print()

    # load adc and run info trees
    tadc = d_input.tadc
    trun = d_input.trun
    trawvoltage = d_input.trawvoltage

    #get the list of events
    events_list = tadc.get_list_of_events()
    nb_events = len(events_list)
    print("Number of events in file: ", nb_events)

    # define empty lists to append single event data to
    adc_traces = []
    antennas_from_data = []
    ids_from_data = []
    arrival_times = []

    # get adc traces
    for ev in d_input.tadc:
        trace_ADC = np.array(ev.trace_ch)

        # only take the last 3 entries of ADC trace
        # which are the NWU channels
        event_adc_traces = trace_ADC[:, 1:4]
        # print(f"Shape of ADC traces: {event_adc_traces.shape}")

        # append to overall list
        adc_traces.append(event_adc_traces)

    # get antenna positions and times
    for ev in d_input.trawvoltage:
        du_lat = np.array(ev.gps_lat)
        du_long = np.array(ev.gps_long)
        du_alt = np.array(ev.gps_alt)

        # transforms antennas to cartesian coordinates in array reference frame
        antennas = np.array([GRANDCS(arg=Geodetic(latitude=du_lat[i], longitude=du_long[i], height=du_alt[i]), obstime="2024-01-01", location=origin_geoid) 
                            for i in range(len(du_lat))])

        # antenna positions extracted from file with obs level added
        event_antennas = np.array([[antennas[i, 0, 0], antennas[i, 1, 0], antennas[i, 2, 0]] for i in range(len(antennas))])

        # IDs of the triggered antennas
        event_ids = np.array(ev.du_id)

        # get antenna arrival times
        event_antenna_s = np.array(ev.du_seconds) 
        event_antennas_ns = np.array(ev.du_nanoseconds)

        # get total antenna times
        antenna_times = event_antenna_s - min(event_antenna_s)
        antenna_times =  event_antenna_s + (event_antennas_ns / 1e9)
        
        # different way to get antenna times
        du_s = event_antenna_s.flatten()
        du_s = (du_s - du_s.min()).astype(np.float64)
        du_ns = event_antennas_ns.flatten() / 1e9 
        du_trig = du_s + du_ns.astype(np.float64)
        # print(antenna_times)
        # print(du_trig)
        antenna_times = du_trig

        # reconstruct arrival direction as a test
        # print(event_antennas.shape, antenna_times.shape)
        zenith_rec, azimuth_rec = np.rad2deg(PWF_semianalytical(event_antennas, antenna_times))
        print("1. Rec. Arrival Direction", np.round(zenith_rec, 2), np.round(azimuth_rec, 2))

        antennas_from_data.append(event_antennas)
        ids_from_data.append(event_ids)
        arrival_times.append(antenna_times)

        if plot_adc_traces: 
            # plot ADC traces
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(30, 10))
            ax[0].plot(range(len(trace_ADC[2][0])), trace_ADC[2][1], c="blue", label="Ch1")

            ax[1].plot(range(len(trace_ADC[2][0])), trace_ADC[2][2], c="red", label="Ch2")

            ax[2].plot(range(len(trace_ADC[2][0])), trace_ADC[2][3], c="black", label="Ch3")

            ax[0].set_xlabel("Time Samples [2ns]")
            ax[1].set_xlabel("Time Samples [2ns]")
            ax[2].set_xlabel("Time Samples [2ns]")
            ax[0].set_ylabel(r"ADC Counts")
            fig.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f'ADC_test.png')
            plt.close()


    # return all data relevant to reconstruction
    return adc_traces, antennas_from_data, ids_from_data, arrival_times, events_list