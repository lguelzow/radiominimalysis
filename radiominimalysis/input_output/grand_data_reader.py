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


# parser = argparse.ArgumentParser(description="")

# # parser asks for hdf5 input files to plot results from
# parser.add_argument(
#     "paths",
#     metavar="PATH",
#     type=str,
#     nargs="*",
#     default=[],
#     help="Choose hdf5 input file(s).",
# )

# read arguments from the command line after the function itself
# args = parser.parse_args()


# plt.rcParams.update({'font.size': 15})

# read data from antenna file
file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/antennas_gp65_full_GPS.txt", dtype = "str")
# file = np.genfromtxt("/cr/users/guelzow/simulations/antenna_layouts/list_antennas_GP41_GPS.txt", dtype = "str")

# put antenna positions in [latitude, longitude, height] and antenna IDs in arrays
antenna_positions_geodetic = np.array([[file[i, 2].astype(float), file[i, 1].astype(float), file[i, 3].astype(float)] for i in range(len(file))])
antenna_ids = np.array([file[i, 0].astype(float) for i in range(len(file))])

# define origin in same system
# origin_geoid = Geodetic(latitude=trun.origin_geoid[0], longitude=trun.origin_geoid[1], height=trun.origin_geoid[2])
# origin_geoid = Geodetic(latitude=40.98455810546875, longitude=93.9522476196289, height=1242)
# DAQ room origin
origin_geoid = Geodetic(latitude=40.99434, longitude=93.94177, height=1262)

# set global GRAND reference frame
array_reference_frame = GRANDCS(arg=origin_geoid, obstime="2024-01-01", location=origin_geoid)       

# transforms antennas to cartesian coordinates in array reference frame
antenna_positions_xyz = np.array([GRANDCS(arg=Geodetic(latitude=pos[0], longitude=pos[1], height=pos[2]), obstime="2024-01-01", location=origin_geoid) \
                                  for pos in antenna_positions_geodetic])

antennas_xyz = np.array([[antenna_positions_xyz[i, 0, 0], antenna_positions_xyz[i, 1, 0], antenna_positions_xyz[i, 2, 0]] for i in range(len(antenna_positions_xyz))])

# with open("50_antennas.txt", "w") as txt_file:
#     for i in range(len(antennas_xyz)):
#         txt_file.write(str(int(antenna_ids[i])) + " " + str(antennas_xyz[i, 0]) + " " + str(antennas_xyz[i, 1]) + " " + str(antennas_xyz[i, 2]) + "\n")


def plot_ADC_traces(filename, event_and_run):

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


def read_measurement_data_from_root(filename, event_and_run):
    '''
    filename: path to GRAND .root measurement data file

    event_and_run: event and run number of event to be read, tuple like, e.g. (2, 4506)
    '''

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
    0451
    
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

    # mask to select their positions from full list
    mask_event = np.isin(antenna_ids, ids_from_data)

    # get the antenna position of the event
    antennas_in_event = antenna_positions_xyz[mask_event]
    # get it into the right shape
    #antennas_in_event = np.array([[antennas_in_event[i, 0, 0], antennas_in_event[i, 1, 0], antennas_in_event[i, 2, 0]] for i in range(len(antennas_in_event))])

    # get antenna arrival times
    antenna_s = np.array(tadc.du_seconds) * 1e9
    antennas_ns = np.array(tadc.du_nanoseconds)

    # only take the last 3 entries of ADC trace
    # which are the NWU channels
    adc_traces = trace_ADC[:, 1:4]

    print("Extracted measurement data. Processing...")
    
    antenna_times = antenna_s - min(antenna_s)

    antenna_times =  antenna_s + (antennas_ns)
    # reconstruct arrival direction
    print(antennas_from_data, antenna_times)
    zenith_rec, azimuth_rec = np.rad2deg(PWF_semianalytical(antennas_from_data, antenna_times))
    print("1. Rec. Arrival Direction", np.round(zenith_rec, 2), np.round(azimuth_rec, 2))


    # plot the antenna positions from file and from data
    fig, ax = plt.subplots(1, 1)
    ax.scatter(antennas_from_data[:, 1], antennas_from_data[:, 0], marker='o', color='black', label='du_event')
    for i, txt in enumerate(ids_from_data):
        ax.annotate(int(txt), (antennas_from_data[i, 1], antennas_from_data[i, 0]))

    ax.scatter(np.mean(antennas_from_data[:, 1]), np.mean(antennas_from_data[:, 0]), marker='x', color='red', label='barycenter/estimated core')

    ax.set_xlabel("Easting [m]") # Longitude [°]")
    ax.set_ylabel("Northing [m]") # Latitude [°]")
    ax.axis('equal')
    ax.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig("event_layout.png")
    plt.close()


    if 0: 
        # plot the antenna positions from file and from data
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].scatter(-antenna_positions_xyz[:, 1], antenna_positions_xyz[:, 0], marker='x', color='black', label='du_xyz')
        for i, txt in enumerate(antenna_ids):
            ax[0].annotate(int(txt), (-antenna_positions_xyz[i, 1], antenna_positions_xyz[i, 0]))

        ax[1].scatter(antenna_positions_geodetic[:, 1], antenna_positions_geodetic[:, 0], marker='v', color='red', label='du_geoid')
        for i, txt in enumerate(antenna_ids):
            ax[1].annotate(int(txt), (antenna_positions_geodetic[i, 1], antenna_positions_geodetic[i, 0]))

        ax[0].set_xlabel("Easting [m]") # Longitude [°]")
        ax[0].set_ylabel("Northing [m]") # Latitude [°]")
        ax[1].set_xlabel("Longitude [°]")
        ax[1].set_ylabel("Latitude [°]")
        ax[0].axis('equal')
        ax[1].axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig("gp65_compare.png")
        plt.close()

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
    return adc_traces, antennas_from_data, ids_from_data, antenna_s, antennas_ns



def read_measurement_data_from_root_directory(filename):
    '''
    filename: path to GRAND .root measurement data file

    event_and_run: event and run number of event to be read, tuple like, e.g. (2, 4506)
    '''

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
        # TODO: find out why it sometimes works with or without the seconds
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

        if 0: 
            # plot the antenna positions from file and from data
            fig, ax = plt.subplots(1, 1)
            ax.scatter(event_antennas[:, 1], event_antennas[:, 0], marker='o', color='black', label='du_event')
            for i, txt in enumerate(event_ids):
                ax.annotate(int(txt), (event_antennas[i, 1], event_antennas[i, 0]))

            ax.scatter(np.mean(event_antennas[:, 1]), np.mean(event_antennas[:, 0]), marker='x', color='red', label='barycenter/estimated core')

            ax.set_xlabel("Easting [m]") # Longitude [°]")
            ax.set_ylabel("Northing [m]") # Latitude [°]")
            ax.axis('equal')
            ax.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig("event_layout.png")
            plt.close()

            # plot the antenna positions from file and from data
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].scatter(-antenna_positions_xyz[:, 1], antenna_positions_xyz[:, 0], marker='x', color='black', label='du_xyz')
            for i, txt in enumerate(antenna_ids):
                ax[0].annotate(int(txt), (-antenna_positions_xyz[i, 1], antenna_positions_xyz[i, 0]))

            ax[1].scatter(antenna_positions_geodetic[:, 1], antenna_positions_geodetic[:, 0], marker='v', color='red', label='du_geoid')
            for i, txt in enumerate(antenna_ids):
                ax[1].annotate(int(txt), (antenna_positions_geodetic[i, 1], antenna_positions_geodetic[i, 0]))

            ax[0].set_xlabel("Easting [m]") # Longitude [°]")
            ax[0].set_ylabel("Northing [m]") # Latitude [°]")
            ax[1].set_xlabel("Longitude [°]")
            ax[1].set_ylabel("Latitude [°]")
            ax[0].axis('equal')
            ax[1].axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig("gp65_compare.png")
            plt.close()

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


    

filename = "/cr/aera02/huege/guelzow/gp80_data/sps/grand/data/gp80/GrandRoot/2025/02/GP80_20250219_010150_RUN9966_CD_ChanXYZ-20dB-GP80-43DUs-X2X-Y2Y-dunhuangTest-UDRUN147-t3offlineVersion1p5-20250218-174914-1000-0001.root"
# filename = "/cr/aera02/huege/guelzow/gp80_data/sps/grand/data/gp80/GrandRoot/2025/02/GP80_20250204_045218_RUN9322_CD_ChanXYZ-20dB-GP80-43DUs-X2X-Y2Y-dunhuangTest-UDRUN144-t3offlineVersion1p5-20250204-043102-1000-0001.root"
event_and_run = (26, 9966)

# read_measurement_data_from_root(filename, event_and_run)