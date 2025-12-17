import radiominimalysis.framework.revent
import radiominimalysis.framework.shower
from radiominimalysis.framework.parameters import showerParameters as shp
from radiominimalysis.framework.parameters import stationParameters as stp
from radiominimalysis.framework.parameters import eventParameters as evp

from radiotools import helper as rdhelp, coordinatesystems

import matplotlib.pyplot as plt

from PWF_reconstruction.recons_PWF import PWF_semianalytical, cov_matrix, angular_error

# these libraries will only work within the grandlib docker or conda environment
# for oldRFchain
# from grand.grandlib_classes.grandlib_classes import *
# import grand.dataio.root_trees as groot

# for newRFchain
from grand.grandlib_classes import *
import grand.dataio as groot

# import the rest of the guardians of the galaxy:
# import grand.manage_log as mlg
import sim2root.Common.raw_root_trees as RawTrees # this is here in Common

import numpy as np
from scipy.signal import hilbert
import math
import re
import os
import sys
import copy
import warnings
from datetime import timedelta
import time

conversion_fieldstrength_cgs_to_SI = 2.99792458e4

eps_0 = 8.8541878188e-12
c = 2.99792458e8

class ReadRootInput:

    def __init__(self, input, verbose=False):
        """
        init method

        initialize ReadRootInput

        Parameters
        ----------
        input: input directory or file
            directory with root input files for a simulation
            .root file in case of measurement data
        verbose: bool
        """
        if len(input) == 0:
            sys.exit("No input file(s)! Abort...")

        # make input files into a list
        if not isinstance(input, list):
            input = [input]

        # self.__t = 0
        # self.__t_event_structure = 0
        # self.__t_per_event = 0
        self.__input = input
        # self.__current_input_file = 0
        # self.__verbose = verbose

        # self.__read_highlevel = read_in_highlevel_file
        # self.__add_traces_from_highlevel = add_traces_from_highlevel
        # self.__add_traces_from_observer = add_traces_from_observer


    def run(self):
        """
        read in a full input directory. Usually multiple simulations!

        """
        # fill and return event list
        evt = read_root_file(self.__input)
        # try:
        #     evt = read_root_file(self.__input)
        #     # print("test", evt)
        # except Exception as e:
        #     print("Failure to read {}\nError: {}".format(self.__input, e))

        # give event list to variable calling the class    
        if isinstance(evt, list):
            for e in evt:
                yield e
        else:
            yield evt
        

    def end(self):
        return timedelta(seconds=self.__t)

    def get_factory(self, factory=None):
        from radiominimalysis.framework.factory import EventFactory
        if factory is None:
            factory = EventFactory()

        [factory.add_event(evt) for evt in self.run()]
        return factory

    def get_events(self):
        return [evt for evt in self.run()]


def read_root_file(input_directories, args=None):

    # list for fully initialised events at the end
    events = []

    fluence_ratio = []

    GP300_antennas_IDs = np.array(range(0, 300))
    
    GP300_antennas_IDs_only_infill = np.sort(np.array([86, 76, 72, 71, 75, 84, 60, 50, 38, 49, 59, 83, 90, 64, 54, 34, 26, 25, 33, 53, 63, 89, 68, \
                                                       42, 22, 18, 12, 17, 21, 41, 67, 80, 46, 30, 10, 4, 3, 9, 29, 45, 79, 56, 36, 14, 6, 0, 5, 13, \
                                                       35, 55, 77, 43 ,27, 7, 1, 2, 8, 28, 44, 78, 65, 39, 19, 15, 11, 16, 20, 40, 66, 87, 61, 51, \
                                                       31, 23, 24, 32, 52, 62, 88, 81, 57, 47, 37, 48, 58, 82, 73, 69, 70, 74, 85]))

    GP300_antennas_infill_regular_grid = np.sort(np.array([86, 76, 75, 84, 38, 83, 89, 33, 34, 90, 42, 12, 41, 80, 10, 9, 79, 36, 0, 35, 77, \
                                                           7, 8, 78, 39, 11, 40, 87, 31, 32, 88, 81, 37, 82, 73, 74, 85]))

    # take out the whole infill and reinsert the antennas that also lie on the regular grid
    GP300_antennas_without_infill = np.sort(np.concatenate((GP300_antennas_IDs[~np.isin(GP300_antennas_IDs, GP300_antennas_IDs_only_infill)], \
                                                   GP300_antennas_infill_regular_grid)))


    print("all input directories: ", input_directories)

    for input_file in input_directories:

        print("current directory: ", input_file)

        # check if input is a directory
        if os.path.isdir(input_file):
            # define read data bool
            read_data = True
            single_file = False

            # check if there's a root file containing the efield
            # this makes it a simulation input file so we put read_data to False
            directory = os.listdir(input_file)
            for file in directory:
                try:
                    if file.startswith('efield_'):
                        read_data = False
                except:
                    print("If you get this message, you either have to put a >b< in front of 'efield' in the if-condition or remove the b!")
                    return 0

            # print which input is used
            if read_data:
                print("Reading measurement data!")
            else:
                print("Reading simulation data!")


        # check if input_file is GP80 data file
        elif input_file[-5:] == ".root":
            print("Reading single measurement file!")
            # set variable about which part of the function to call
            read_data = True
            single_file = True

        else:
            print("Please provide a valid sim2root output directory")
            exit()

        #
        # process measurement data in this part 
        #

        if read_data:

            from RadioAnalysis.input_output.VocToEfield import efield_reconstruction_from_measurement
            from radiominimalysis.input_output import grand_data_reader

            # decide whether to save efield traces
            save_efield_traces = False

            # open text file for event candidates
            with open("event_candidate_params.csv", "a") as txt_file:
                # write labels into text file
                txt_file.write("#Run_no" + " " + "Evt_no" + " " + "chi2/ndf" + " " \
                            + "Eem" + " " + "Eem_err" + " " \
                            + "Zenith" + " " + "Zenith_err" + " " \
                            + "Azimuth" + " " + "Azimuth_err" + " " \
                            + "Pointing_err" + " "  \
                            + "Bary_centre_x" + " " + "Bary_centre_y" + " " \
                            + "Core_fit_x" + " " + "Core_fit_y" + "\n")

            # get site data from an arbitrary simulation file
            site_data = groot.DataDirectory("/cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_ADC_noise/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-proton-AN_0000")
            trun = site_data.trun_l0
            # tshower = site_data.tshower_l0

            # get magnetic field
            # mag_field_vector = tshower.magnetic_field
            obs_level = 1262
            # in geodetic coordinates
            # origin_geoid = np.array([40.98455810546875, 93.9522476196289, obs_level])
            origin_geoid = np.array([40.99434, 93.94177, obs_level])

            # read in magnetic field vector
            # mag_field_inclination = np.deg2rad(tshower.magnetic_field[0])
            # mag_field_vector = np.array([np.cos(mag_field_inclination), 0, -np.sin(mag_field_inclination)])
            mag_field_vector = np.array([0.47554672, 0., -0.87969047])

            # read just a single event in a large data file
            if single_file:

                # manually give event and run numbers
                event_number = 676
                run_number = 10070
                
                # call function which reads measurement data and returns it as np.array
                adc_traces, antennas_triggered, antenna_IDs, arrival_times = grand_data_reader.read_measurement_data_from_root(filename=input_file, event_and_run=(event_number, run_number))

                # put in list so it works with the for loop
                adc_traces = [adc_traces]
                antennas_triggered = [antennas_triggered]
                antenna_IDs = [antenna_IDs]
                arrival_times = [arrival_times]

            # read every event from compiled root file
            else: 
                # get all data from event directory
                adc_traces, antennas_triggered, antenna_IDs, arrival_times, event_list = grand_data_reader.read_measurement_data_from_root_directory(filename=input_file)

            # loop over all events
            for i in range(len(adc_traces)):
                
                # put them back
                traces = adc_traces[i]
                antennas = antennas_triggered[i]
                IDs = antenna_IDs[i]              
                timing = arrival_times[i]

                if not single_file:
                    # define event and run number
                    event_number, run_number = event_list[i]

                # if run_number == 576 or run_number ==  2574 or run_number == 2626 or run_number == 4442 or run_number == 6222 or run_number == 4666:
                #     print("one of the best events!")
                # else:
                #     continue

                # add correct observation level to antenna positions
                antennas_with_obs_level = antennas + np.array([0, 0 , origin_geoid[2]])

                # set trace parameters for efield reconstruction and fluence calculations
                tstep = 2 # adc_traces[0][0][1] - adc_traces[0][0][1] # in nanoseconds
                trace_length = 1024 # len(adc_traces[0][0])

                # use efield reconstruction to recover electric field from ADC traces
                # angles are given in radians
                rec_traces, recon_antenna_mask, saturation_flags, noisy_flags, bad_timing_flags, core_est, \
                zenith_rec, zenith_err, azimuth_rec, azimuth_err, dir_err, timings = efield_reconstruction_from_measurement(event_number, run_number, 
                                                                                                                    traces,
                                                                                                                    antennas,
                                                                                                                    IDs,
                                                                                                                    timing,
                                                                                                                    origin_geoid,
                                                                                                                    trace_length,
                                                                                                                    tstep)
            
                if save_efield_traces:
                        np.savez_compressed(f"rec_efield_traces_run{run_number}_event{event_number}", \
                                            zenith_rec, zenith_err, azimuth_rec, azimuth_err, dir_err, rec_traces, recon_antenna_mask, saturation_flags)
                        
                        
                # all traces
                traces = rec_traces
                # mask for antennas passing all efield recon quality cuts
                passing_antennas = recon_antenna_mask
                # mask for saturated antennas
                saturated_antennas = saturation_flags
                # mask for antennas eliminated by ADC noise cut
                noisy_antennas = noisy_flags
                # mask for antennas eliminated by timing cut
                bad_timing_antennas = bad_timing_flags
                        
                # all antennas in the event with proper obs_level added
                antennas_in_event = antennas_with_obs_level

                # initialise coordinate system of the shower
                ctrans = coordinatesystems.cstrafo(zenith_rec, azimuth_rec, magnetic_field_vector=mag_field_vector)

                # calculate energy fluence with function from the efield traces
                energy_fluence, signal_to_noise, fluence_error = calculate_fluence_from_efield(traces, ctrans, tstep=2, energy_fluence_list=[], SNR_list=[], energy_fluence_error=[], real_data=True)
                # print(np.array(energy_fluence).shape, np.array(signal_to_noise).shape, np.array(fluence_error).shape)

                # SNR threshold
                threshold = 25
                # SNR mask for all traces
                SNR_mask_all = (np.array(signal_to_noise)) >= threshold
                # SNR mask only for the antennas passing the efield recon quality cuts
                SNR_mask = SNR_mask_all & passing_antennas                
                # mask for all passing antennas that don't pass SNR cut
                cut_mask = np.logical_not(SNR_mask_all) & passing_antennas

                print("(Run ", run_number, "; Event ", event_number, f"): {np.sum(SNR_mask)}/{np.sum(passing_antennas)}/{len(antennas_in_event)} -- Viable Ant. / Recon. Ant. / Total Ant." + "\n" + \
                                  f"Quality cuts: {np.sum(noisy_antennas)}/{np.sum(bad_timing_antennas)}/{np.sum(cut_mask)} -- ADC noise cut / Peak timing cut / SNR cut"+ "\n" + \
                                  f"{np.sum(saturated_antennas)} saturated antennas in the event!")

                # fake unit test
                if np.sum(SNR_mask) + np.sum(noisy_antennas) + np.sum(bad_timing_antennas) + np.sum(cut_mask) != len(antennas_in_event):
                    print(f"Number of antennas ({len(antennas_in_event)}) doesn't match passed {np.sum(SNR_mask)} and cut ({np.sum(noisy_antennas)}/{np.sum(bad_timing_antennas)}/{np.sum(cut_mask)}) amounts!!")
                    exit()
                    
                # time delay for single looks into sims
                if 0:
                    sec = input("Sleep for how many seconds? \n ")
                    print('Going to sleep for', sec, 'seconds...')
                    time.sleep(int(sec))

                # save parameters into radiominimalysis revent class

                # 
                # Event-wide parameters
                #

                evt = radiominimalysis.framework.revent.REvent(run_number, event_number)
                evt.set_parameter(evp.file, input_file.split("/")[-1])
                evt.set_parameter(evp.time, 0) # TODO: do I need this?

                # hard code refractive index at sea level fpr China
                evt.set_parameter(evp.refractive_index_at_sea_level, 1.000273455)

                # set magnetic field vector of the site
                evt.set_parameter(evp.magnetic_field_vector, mag_field_vector)

                # create shower object and set shower parameters
                shower = radiominimalysis.framework.shower.Shower(evp.GRAND_shower)
                
                # set atmosphere model and observation level
                shower.set_parameter(shp.atmosphere_model, 41)
                shower.set_parameter(shp.observation_level, obs_level)


                #
                # Section for reconstructed and estimated information
                #

                # reconstructed arrival direction
                shower.set_parameter(shp.zenith_recon, zenith_rec)
                shower.set_parameter_error(shp.zenith_recon, zenith_err)
                shower.set_parameter(shp.azimuth_recon, azimuth_rec)
                shower.set_parameter_error(shp.azimuth_recon, azimuth_err)
                shower.set_parameter(shp.pointing_error, dir_err)

                # calculated geomagnetic angle
                shower_axis = rdhelp.spherical_to_cartesian(zenith_rec, azimuth_rec)
                alpha = np.arccos(np.dot(shower_axis, mag_field_vector) / (np.linalg.norm(shower_axis) * np.linalg.norm(mag_field_vector)))
                shower.set_parameter(shp.geomagnetic_angle_recon, alpha)

                # shower core estimate
                shower.set_parameter(shp.core_estimate, np.array([core_est[0], core_est[1], obs_level]))

                evt.add_shower(shower)

                # 
                # Section for Station parameters
                # 

                # set stations parameters from only the stations in the event, filtered by the SNR cut
                evt.set_station_parameter(stp.energy_fluence, np.array(energy_fluence)[SNR_mask])
                evt.set_station_parameter(stp.vxB_error, np.array(fluence_error)[SNR_mask])

                # evt.set_station_parameter(stp.frequency_slope, np.array(frequency_slope)[SNR_mask])
                evt.set_station_parameter(stp.position, np.array(antennas_in_event)[SNR_mask])
                evt.set_station_parameter(stp.times, np.array(timings)[SNR_mask])
                # print((np.array(timings)[SNR_mask] - min(np.array(timings)[SNR_mask])) * 1e6)
                # print(np.array(antennas_in_event)[SNR_mask])
                evt.set_station_parameter(stp.id, np.array(IDs[SNR_mask]))
                evt.set_station_parameter(stp.saturated, saturated_antennas[SNR_mask])

                # ADC noise cut
                if np.sum(noisy_antennas) > 0:
                    evt.set_station_parameter(stp.rejected_noisy, IDs[noisy_antennas])
                    evt.set_station_parameter(stp.energy_fluence_noisy, np.array(energy_fluence)[noisy_antennas])
                    evt.set_station_parameter(stp.positions_noisy, np.array(antennas_in_event)[noisy_antennas])
                    evt.set_station_parameter(stp.error_noisy, np.array(fluence_error)[noisy_antennas])
                    
                # bad timing cut
                if np.sum(bad_timing_antennas) > 0:
                    evt.set_station_parameter(stp.rejected_bad_timing, IDs[bad_timing_antennas])
                    evt.set_station_parameter(stp.energy_fluence_bad_timing, np.array(energy_fluence)[bad_timing_antennas])
                    evt.set_station_parameter(stp.positions_bad_timing, np.array(antennas_in_event)[bad_timing_antennas])
                    evt.set_station_parameter(stp.error_bad_timing, np.array(fluence_error)[bad_timing_antennas])
                    
                # SNR cut
                if np.sum(cut_mask) > 0:
                    evt.set_station_parameter(stp.rejected_snr, IDs[cut_mask])
                    evt.set_station_parameter(stp.energy_fluence_snr, np.array(energy_fluence)[cut_mask])
                    evt.set_station_parameter(stp.positions_snr, np.array(antennas_in_event)[cut_mask])
                    evt.set_station_parameter(stp.error_snr, np.array(fluence_error)[cut_mask])
                    
                if np.sum(saturated_antennas) > 0:
                    # save flagged antennas separately
                    evt.set_station_parameter(stp.saturated_fluence, np.array(energy_fluence)[saturated_antennas])
                    evt.set_station_parameter(stp.saturated_positions, np.array(antennas_in_event)[saturated_antennas])
                    evt.set_station_parameter(stp.saturated_errors, np.array(fluence_error)[saturated_antennas])
                    
                print("event done")
                    
                    
                # add event to complete list of events
                events.append(evt)




        #
        # read simulation data!
        #

        else:
            # initialise file into groot class
            d_input = groot.DataDirectory(input_file)
            try:
                d_input = groot.DataDirectory(input_file)
            except:
                print(f"No ADC traces found in {input_file}, skipping to next directory...")
                continue

            #Get the trees L0
            tshower_l0 = d_input.tshower_l0
            # print(tshower_l0)

            # get antenna position and obslevel trees
            trun_l0 = d_input.trun_l0
            trun_l1 = d_input.trun_l1

            trunefieldsim_l0 = d_input.trunefieldsim_l0
            trunefieldsim_l1 = d_input.trunefieldsim_l1

            #Get the trees L1 for traces
            tefield_l0 = d_input.tefield_l0
            tefield_l1 = d_input.tefield_l1
            trunefieldsim_l1=d_input.trunefieldsim_l1

            # test adc file
            tadc_l1 = d_input.tadc_l1

            #get the list of events
            events_list = tshower_l0.get_list_of_events()
            nb_events = len(events_list)

            print(len(events_list), "events found in directory. Reading...")

            # extract the information out of all events in arrays
            event_and_run = np.array([event for event in events_list])

            # read in general run formation: antenna positions and obs level
            # this assumes all events in one directory belong to the same run!!
            trun_l0.get_run(event_and_run[0, 1])

            # read antenna position from root file for the whole run
            antenna_positions = np.array(trun_l0.du_xyz) + np.array([0, 0, 1265.5424])

            # check if right observation level is given
            if any(antenna_positions[:, 2] > 2000) or any(antenna_positions[:, 2] < 500):
                print("Likely wrong observation level given! Needs to be in reference to sea level! Max and Min: ", max(antenna_positions[:, 2]), min(antenna_positions[:, 2]))
                exit()

            # TODO: make sure this is imported from grandlib properly. Currently hardcoded to an old value there
            obs_level = trun_l0.origin_geoid[2] # 1265.5424 
            mag_field_vector = tshower_l0.magnetic_field

            # If there are no events in the file, exit
            if nb_events == 0:
                sys.exit("There are no events in the file! Exiting.")

            # extract the information out of all events in arrays
            event_and_run = np.array([event for event in events_list])

            SNR_global = []


            i=0 # loop index
            for event_number,run_number in events_list:
                i+=1
                assert isinstance(event_number, int)
                assert isinstance(run_number, int)
                # logger.debug(f"Running event_number: {event_number}, run_number: {run_number}")

                #this gives the indices of the antennas of the array participating in this event
                event_dus_indices = tefield_l1.get_dus_indices_in_run(trun_l0)

                # sampling time in ns, sampling freq = 1e9/dt_ns
                dt_ns_l1 = np.asarray(trun_l0.t_bin_size)[event_dus_indices]

                #time window parameters. time windows go from t0-t_pre to t0+t_post
                t_pre_L1=trunefieldsim_l1.t_pre
                t_post_L1=trunefieldsim_l1.t_post

                tadc_l1.get_event(event_number, run_number)
                
                # load data of current event to be read later
                tshower_l0.get_event(event_number, run_number)
                tefield_l0.get_event(event_number, run_number)
                tefield_l1.get_event(event_number, run_number)

                # extract time of event in s and ns
                event_second = tshower_l0.core_time_s
                event_nano = tshower_l0.core_time_ns

                # read in magnetic field vector
                mag_field_inclination = np.deg2rad(tshower_l0.magnetic_field[0])
                mag_field_vector = np.array([np.cos(mag_field_inclination), 0, -np.sin(mag_field_inclination)])

                # choose the right antenna positions for the event

                # first take the antenna IDs
                antenna_IDs = np.array(tefield_l1.du_id)
                # and the IDs of all antennas in the array
                full_array_IDs = np.array(trun_l0.du_id)

                # make mask for the antennas in the event to cast on the complete array
                antenna_mask = np.isin(full_array_IDs, antenna_IDs)

                # now apply mask to antenna positions
                antennas_in_event = antenna_positions[antenna_mask]

                #
                # traces
                #
                
                # parameter to govern whether efield traces are artificially made or reconstructed
                # and decide whether to only use measurable quantities
                recon_efield = False
                use_MC = False
                save_MC = True
                save_efield_traces = False
                compare_fluence = False

                # decide which traces will be used
                if recon_efield:

                    from RadioAnalysis.input_output.VocToEfield import efield_reconstruction_from_ADC

                    recon_string = "recon"
                    
                    # use efield reconstruction to recover electric field from ADC traces
                    # angles are given in radians
                    rec_traces, recon_antenna_mask, saturation_flags, noisy_flags, bad_timing_flags, core_est, zenith_rec, zenith_err, azimuth_rec, azimuth_err, dir_err = efield_reconstruction_from_ADC(event_number, run_number,
                                                                                                                tadc_l1,
                                                                                                                tefield_l0,
                                                                                                                tshower_l0,
                                                                                                                trun_l0, trun_l1,
                                                                                                                trunefieldsim_l0, trunefieldsim_l1,
                                                                                                                use_MC, plot_trace_comp=False)
                    
                    if save_efield_traces:
                        np.savez_compressed(f"rec_efield_traces_run{run_number}_event{event_number}", \
                                            zenith_rec, zenith_err, azimuth_rec, azimuth_err, dir_err, rec_traces, recon_antenna_mask, saturation_flags)

                    # check whether event was not able to be reconstructed
                    # and jump to next event in that case
                    if not zenith_rec and not azimuth_rec:
                        print(f"Zenith angle: {zenith_rec} -> No direction reconstruction possible. Skipping to next event!")
                        continue

                    # all traces
                    traces = rec_traces
                    # mask for antennas passing all efield recon quality cuts
                    passing_antennas = recon_antenna_mask
                    # mask for saturated antennas
                    saturated_antennas = saturation_flags
                    # mask for antennas eliminated by ADC noise cut
                    noisy_antennas = noisy_flags
                    # mask for antennas eliminated by timing cut
                    bad_timing_antennas = bad_timing_flags
                    # mask for fluence comparison for passing antennas that aren't saturated
                    compare_mask = np.all([passing_antennas, np.logical_not(saturated_antennas)], axis=0)

                
                else:
                    recon_string = "L1"
                    # read efield traces to be calculated into fluence
                    sim_traces = np.asarray(tefield_l1.trace, dtype=np.float32)
                    # read traces without filter for frequency slope and thinning cut
                    # traces_L0 = np.asarray(tefield_l0.trace, dtype=np.float32)
                
                    # empty masks for the parameters quality cuts are done for in efield recon
                    passing_antennas = np.ones(antenna_IDs.shape, dtype=bool) 
                    saturated_antennas = np.zeros(antenna_IDs.shape, dtype=bool)               
                    noisy_antennas = np.zeros(antenna_IDs.shape, dtype=bool)   
                    bad_timing_antennas = np.zeros(antenna_IDs.shape, dtype=bool)   
                    
                    traces = sim_traces

                    # reconstruct arrival direction for the case of L1 with no MC parameters
                    # angles in radians. Print is converted
                    if not use_MC:
                        arrival_time_error = 5 * 1e-9 # 5 nanoseconds

                        antenna_times =  np.array(tefield_l0.du_seconds) + np.array(tefield_l0.du_nanoseconds) / 1e9 - min(np.array(tefield_l0.du_seconds))
                        # reconstruct arrival direction
                        zenith_rec, azimuth_rec = PWF_semianalytical(np.array(antennas_in_event), antenna_times)

                        # if direction reconstruction fails, skip to next event
                        if np.isnan(zenith_rec) or np.isnan(azimuth_rec):
                            print(f"No viable direction reconstruction for {len(antennas_in_event)} stations!")
                            continue

                        # calculate covariance matrix and arrival direction errors
                        cov_mat = cov_matrix(zenith_rec, azimuth_rec, np.array(antennas_in_event), arrival_time_error)
                        zenith_err = np.sqrt(cov_mat[0, 0])
                        azimuth_err = np.sqrt(cov_mat[1, 1])
                        dir_err = angular_error(zenith_rec, cov_mat)

                        print(f"True Arrival Direction: ({tshower_l0.zenith}, {tshower_l0.azimuth})")
                        print(f"Rec. Arrival Direction: ({np.round(np.rad2deg(zenith_rec), 2)}({np.round(np.rad2deg(zenith_err), 2)}), {np.round(np.rad2deg(azimuth_rec), 2)}({np.round(np.rad2deg(azimuth_err), 2)})")
                        print("Absolute Pointing Error: ", np.rad2deg(dir_err), " Degrees")

                        core_est = np.array([np.mean(antennas_in_event[:, 0]), np.mean(antennas_in_event[:, 1]), obs_level])
                        print("True Core: ", np.array([tshower_l0.shower_core_pos[0], tshower_l0.shower_core_pos[1], obs_level]))
                        print("Core Est.: ", core_est)

                    else:
                        zenith_rec = False


                # for comparison, take L0 traces
                if compare_fluence:
                    recon_string = "L0"
                    # use L1 traces
                    # compare_traces = np.asarray(tefield_l1.trace, dtype=np.float32)
                    # use L0 traces
                    compare_traces = np.asarray(tefield_l0.trace, dtype=np.float32)
                    compare_mask = np.ones(len(compare_traces), dtype=bool)
                    # print(len(compare_traces), np.sum(compare_mask), len(traces))


                # save MC information for propagating later
                if save_MC:
                    zenith_MC = np.deg2rad(tshower_l0.zenith)
                    azimuth_MC = np.deg2rad(tshower_l0.azimuth)
                    core_MC = np.array([tshower_l0.shower_core_pos[0], tshower_l0.shower_core_pos[1], obs_level])

                else: 
                    # otherwise make them empty so they don't give errors
                    zenith_MC = azimuth_MC = core_MC = None
                

                # use shape of the trace
                trace_shape = traces.shape
                nb_du = trace_shape[0]
                sig_size = trace_shape[-1]
                # logger.info(f"Event has {nb_du} DUs, with a signal size of: {sig_size}")


                # initialise radiotools coordinate instance to convert to vxB system
                # for MC parameters
                if use_MC:
                    ctrans = coordinatesystems.cstrafo(zenith_MC, azimuth_MC, magnetic_field_vector=mag_field_vector)

                # for reconstructed parameters
                else:
                    ctrans = coordinatesystems.cstrafo(zenith_rec, azimuth_rec, magnetic_field_vector=mag_field_vector)

                # define list for energy fluence and SNR of all stations of the event to be saved
                noise_fluence = []
                frequency_slope = np.zeros([nb_du, 3, 2])


                # calculate energy fluence with function from the efield traces
                if 1: # event_number == 2:
                    energy_fluence, signal_to_noise, fluence_error = calculate_fluence_from_efield(traces, ctrans, tstep=2, energy_fluence_list=[], SNR_list=[], energy_fluence_error=[])
                else:
                    continue

                # calculate "MC" fluence to compare to reconstructed fluence
                if compare_fluence and np.sum(compare_mask) > 0 and (len(compare_mask) == len(energy_fluence)):
                    
                    # select the antennas that made it through the efield reconstruction and are unsaturated for the comparison
                    energy_fluence_comp, _, _ = calculate_fluence_from_efield(np.array(compare_traces)[compare_mask], ctrans, tstep=0.5, energy_fluence_list=[], SNR_list=[], noise_or_L0=True)
                    fluence_MC = np.array(energy_fluence_comp)

                    # calculate the ratio of the reconstructed and MC fluence while disregarding the saturated antennas
                    print("Calculating MC fluence for comparison!")
                    fluence_rec = np.array(energy_fluence)[compare_mask]
                    # also save positions for comparison
                    antennas_compare = antennas_in_event[compare_mask]

                    # check if any antennas were compared
                    if len(fluence_MC[:, 0]) > 1: 
                        # only save vxB component
                        fluence_compare_MC = fluence_MC[:, 0]
                        fluence_compare_rec = fluence_rec[:, 0]

                        # print(fluence_compare_rec, fluence_compare_MC)
                        fluence_ratio.append((fluence_compare_rec - fluence_compare_MC) / fluence_compare_MC)
                            

                    else:
                        # otherwise, save an empty array
                        fluence_compare_MC = fluence_compare_rec = np.array([])
                            

                # now define mask for stations depending on whether the efield was reconstructed and already cuts applied there
                if recon_efield: 
                    threshold = 25
                else:            
                    threshold = 25

                # SNR for all traces
                SNR_mask_all = (np.array(signal_to_noise)) >= threshold
                # SNR mask only for the antennas passing the efield recon quality cuts
                SNR_mask = SNR_mask_all & passing_antennas
                # add to global SNR list
                # if event_number < 10:
                #     SNR_global.append(signal_to_noise)
                
                # mask for all passing antennas that don't pass SNR cut
                cut_mask = np.logical_not(SNR_mask_all) & passing_antennas

                print("Event", i, f": {np.sum(SNR_mask)}/{np.sum(passing_antennas)}/{len(antennas_in_event)} -- Viable Ant. / Recon. Ant. / Total Ant." + "\n" + \
                                  f"Quality cuts: {np.sum(noisy_antennas)}/{np.sum(bad_timing_antennas)}/{np.sum(cut_mask)} -- ADC noise cut / Peak timing cut / SNR cut"+ "\n" + \
                                  f"{np.sum(saturated_antennas)} saturated antennas in the event!")


                # fake unit test
                if np.sum(SNR_mask) + np.sum(noisy_antennas) + np.sum(bad_timing_antennas) + np.sum(cut_mask) != len(antennas_in_event):
                    print(f"Number of antennas ({len(antennas_in_event)}) doesn't match passed {np.sum(SNR_mask)} and cut ({np.sum(noisy_antennas)}/{np.sum(bad_timing_antennas)}/{np.sum(cut_mask)}) amounts!!")
                    exit()

                #########
                # now feed data into radiominimalysis framework
                #########


                #
                # Secton for event-wide parameters
                #

                evt = radiominimalysis.framework.revent.REvent(run_number, event_number)
                evt.set_parameter(evp.file, input_file.split("/")[-1])
                evt.set_parameter(evp.time, 0) # TODO: do I need this?

                # hard code refractive index at sea level fpr China
                evt.set_parameter(evp.refractive_index_at_sea_level, 1.000273455)

                # create shower object and set shower parameters
                shower = radiominimalysis.framework.shower.Shower(evp.sim_shower)
                
                # set atmosphere model and observation level
                shower.set_parameter(shp.atmosphere_model, 41)
                shower.set_parameter(shp.observation_level, obs_level)

                # set magnetic field vector of the site
                evt.set_parameter(evp.magnetic_field_vector, mag_field_vector)


                #
                # Section for MC information
                #

                # arrival direction
                shower.set_parameter(shp.zenith, zenith_MC)
                shower.set_parameter(shp.azimuth, azimuth_MC) 

                # calculated geomagnetic angle
                shower_axis = rdhelp.spherical_to_cartesian(zenith_MC, azimuth_MC)
                alpha = np.arccos(np.dot(shower_axis, mag_field_vector) / (np.linalg.norm(shower_axis) * np.linalg.norm(mag_field_vector)))
                shower.set_parameter(shp.geomagnetic_angle, alpha)

                # shower core
                shower.set_parameter(shp.core, core_MC)

                # energy and primary information
                shower.set_parameter(shp.primary_particle, tshower_l0.primary_type)
                shower.set_parameter(shp.energy, tshower_l0.energy_primary * 1e9) # convert to eV
                shower.set_parameter(shp.electromagnetic_energy, tshower_l0.energy_em * 1e9) # convert to eV

                # shower maximum
                shower.set_parameter(shp.xmax, tshower_l0.xmax_grams)


                #
                # Section for reconstructed and estimated information
                #

                # only save if actually reconstructed
                if zenith_rec:

                        # reconstructed arrival direction
                    shower.set_parameter(shp.zenith_recon, zenith_rec)
                    shower.set_parameter_error(shp.zenith_recon, zenith_err)
                    shower.set_parameter(shp.azimuth_recon, azimuth_rec)
                    shower.set_parameter_error(shp.azimuth_recon, azimuth_err)
                    shower.set_parameter(shp.pointing_error, dir_err)

                    # calculated geomagnetic angle
                    shower_axis = rdhelp.spherical_to_cartesian(zenith_rec, azimuth_rec)
                    alpha = np.arccos(np.dot(shower_axis, mag_field_vector) / (np.linalg.norm(shower_axis) * np.linalg.norm(mag_field_vector)))
                    shower.set_parameter(shp.geomagnetic_angle_recon, alpha)

                    # shower core estimate
                    shower.set_parameter(shp.core_estimate, np.array([core_est[0], core_est[1], obs_level]))

                evt.add_shower(shower)

                # 
                # Section for Station parameters
                # 

                # require any antennas to pass SNR cut to be saved
                if np.sum(SNR_mask) > 1:

                    # set stations parameters from only the stations in the event that will be passed to LDF recon
                    evt.set_station_parameter(stp.energy_fluence, np.array(energy_fluence)[SNR_mask])
                    evt.set_station_parameter(stp.vxB_error, np.array(fluence_error)[SNR_mask])
                    evt.set_station_parameter(stp.signal_to_noise_ratio, np.array(signal_to_noise)[SNR_mask])

                    # evt.set_station_parameter(stp.frequency_slope, np.array(frequency_slope)[SNR_mask])
                    evt.set_station_parameter(stp.position, np.array(antennas_in_event)[SNR_mask])
                    evt.set_station_parameter(stp.id, np.array(antenna_IDs[SNR_mask]))
                    evt.set_station_parameter(stp.array_ids, GP300_antennas_IDs)
                    evt.set_station_parameter(stp.infill_id, GP300_antennas_without_infill)
                    evt.set_station_parameter(stp.only_infill_id, GP300_antennas_IDs_only_infill)
                    
                    evt.set_station_parameter(stp.saturated, saturated_antennas[SNR_mask])
                    
                    # save stations that were cut to display later
                    # print(antenna_IDs[noisy_antennas].shape)
                    # print(np.array(energy_fluence)[noisy_antennas].shape)
                    # print(np.array(antennas_in_event)[noisy_antennas].shape)
                    # print(np.array(fluence_error)[noisy_antennas].shape)
                        
                    # print(antenna_IDs[bad_timing_antennas].shape)
                    # print(np.array(energy_fluence)[bad_timing_antennas].shape)
                    # print(np.array(antennas_in_event)[bad_timing_antennas].shape)
                    # print(np.array(fluence_error)[bad_timing_antennas].shape)
                        
                    # # SNR cut
                    # print(antenna_IDs[cut_mask].shape)
                    # print(np.array(energy_fluence)[cut_mask].shape)
                    # print(np.array(antennas_in_event)[cut_mask].shape)
                    # print(np.array(fluence_error)[cut_mask].shape)      

                    # ADC noise cut
                    if np.sum(noisy_antennas) > 0:
                        evt.set_station_parameter(stp.rejected_noisy, antenna_IDs[noisy_antennas])
                        evt.set_station_parameter(stp.energy_fluence_noisy, np.array(energy_fluence)[noisy_antennas])
                        evt.set_station_parameter(stp.positions_noisy, np.array(antennas_in_event)[noisy_antennas])
                        evt.set_station_parameter(stp.error_noisy, np.array(fluence_error)[noisy_antennas])
                        
                    # bad timing cut
                    if np.sum(bad_timing_antennas) > 0:
                        evt.set_station_parameter(stp.rejected_bad_timing, antenna_IDs[bad_timing_antennas])
                        evt.set_station_parameter(stp.energy_fluence_bad_timing, np.array(energy_fluence)[bad_timing_antennas])
                        evt.set_station_parameter(stp.positions_bad_timing, np.array(antennas_in_event)[bad_timing_antennas])
                        evt.set_station_parameter(stp.error_bad_timing, np.array(fluence_error)[bad_timing_antennas])
                        
                    # SNR cut
                    if np.sum(cut_mask) > 0:
                        evt.set_station_parameter(stp.rejected_snr, antenna_IDs[cut_mask])
                        evt.set_station_parameter(stp.energy_fluence_snr, np.array(energy_fluence)[cut_mask])
                        evt.set_station_parameter(stp.positions_snr, np.array(antennas_in_event)[cut_mask])
                        evt.set_station_parameter(stp.error_snr, np.array(fluence_error)[cut_mask])
                        
                    if np.sum(saturated_antennas) > 0:
                        # save flagged antennas separately
                        evt.set_station_parameter(stp.saturated_fluence, np.array(energy_fluence)[saturated_antennas])
                        evt.set_station_parameter(stp.saturated_positions, np.array(antennas_in_event)[saturated_antennas])
                        evt.set_station_parameter(stp.saturated_errors, np.array(fluence_error)[saturated_antennas])

                    if compare_fluence and np.sum(compare_mask) > 0 and (len(compare_mask) == len(energy_fluence)):
                        print("Saving compare fluence!")
                        evt.set_station_parameter(stp.fluence_compare_MC, fluence_compare_MC[SNR_mask])
                        evt.set_station_parameter(stp.fluence_compare_rec, fluence_compare_rec[SNR_mask])
                        evt.set_station_parameter(stp.antennas_compare, antennas_compare[SNR_mask])
                    

                # add event to complete list of events
                events.append(evt)
                    
            print("Events processed from directory: ", len(events))

            # # SNR plot
            # fig_SNR = plt.figure()
            # SNR_conc = np.concatenate(SNR_global)


            # # histogram for zenith reconstruction
            # plt.title(r'Signal-to-noise ratio for 10 events (new RFchain)')
            # # histogram of fit pull values
            # plt.hist(SNR_conc, bins=100, histtype='step', linewidth=1, color='black', \
            #         label=(r"Antenna SNR" \
            #             + "\n" + r"$\mu =$ %.2f; " + r"$\sigma =$ %.2f") % (np.mean(signal_to_noise), np.std(signal_to_noise)))
            # plt.xlabel(r"Signal-to-Noise Ratio" )
            # plt.ylabel("# of Antennas")
            # # plt.legend()
            # fig_SNR.savefig("SNR_hist_new_RF.png")
            # plt.close()


            # fluence comparison plot
            if compare_fluence and recon_efield:
                conc = np.concatenate(fluence_ratio)

                plt.figure()
                plt.hist(conc, bins=50, histtype="step", color="black", \
                         label=f"$\mu$={np.round(np.mean(conc), 2)}; $\sigma$={np.round(np.std(conc), 2)}")
                plt.xlabel("Relative Deviation (Rec/MC)")
                plt.ylabel("# Stations")
                plt.yscale("log")
                plt.xlim(-5, 5)
                plt.legend()
                plt.savefig("fluence_compare_event.png")
                plt.close()

    print(f"In total, processed so far {len(events)} events!")
    return events



def calculate_fluence_from_efield(traces, ctrans, tstep=2, energy_fluence_list=[], energy_fluence_error=[], SNR_list=[], noise_or_L0=False, real_data=False):
    """
    traces: electric field traces to calculate energy fluence from

    tstep in ns
    """


    for stations in range(len(traces)):

        # print traces in ground plane
        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
        # ax[0].set_title("North")
        # ax[0].plot(range(len(traces[stations, 0])), traces[stations, 0], c="blue", label="North")

        # ax[1].set_title("West")
        # ax[1].plot(range(len(traces[stations, 0])), traces[stations, 1], c="blue", label="West")

        # ax[2].set_title("Vertical")
        # ax[2].plot(range(len(traces[stations, 0])), traces[stations, 2], c="blue", label="Vertical")
        # [ax[i].set_xlabel("Time Samples [2ns]") for i in range(3)]
        # [ax[i].set_ylabel(r"Electric Field Strength [$\mu \mathrm{V/m}$]") for i in range(3)]
        # ax[2].legend(loc="upper right")
        # # ax[0].set_xlim(0, 800)
        # plt.tight_layout()
        # plt.savefig(f'test_trace_ground_{stations}.png')
        # plt.close()


        # perform rotation to vxB system
        # transpose the elements of traces to get the right format for the transformation
        traces_showerplane = ctrans.transform_to_vxB_vxvxB(np.transpose(traces[stations]))
        # traces_showerplane_L0 = ctrans.transform_to_vxB_vxvxB(np.transpose(traces_L0[stations]))

        # efield trace L1 in vxB system
        trace_x = np.array(traces_showerplane[:, 0] * 1e-6) # transform to V/m
        trace_y = np.array(traces_showerplane[:, 1] * 1e-6) # transform to V/m
        trace_z = np.array(traces_showerplane[:, 2] * 1e-6) # transform to V/m

        # amount of entries in L0 trace
        # L0_entries = len(np.array(traces_showerplane_L0[:, 0]))

        # define time of the trace
        tstep = tstep # in ns
        # tstep_L0 = 0.5 # in ns
        tracetime = tstep * np.array(range(len(trace_x)))

        # print(tracetime)
        
        
        ########
        # calculate frequency slope
        ########


        if noise_or_L0:
            frequencyResolution = 100 # kHz

            # frequency limits for bandwidth cut for frequency slope calculation
            flow = 50
            fhigh = 200

            L0_entries = len(np.array(traces_showerplane[:, 0]))

            # add zeros to beginning and end of the trace to increase the frequency resolution (this is not a resampling)
            n_samples = int(np.round(1 / (tstep * 1e-9) / (frequencyResolution * 1e3)))
            # increase number of samples to a power of two for FFT performance reasons
            n_samples = int(2 ** math.ceil(math.log(n_samples, 2)))

            tracetime = tstep * np.arange(n_samples)

            n_start = (n_samples - L0_entries) // 2
            padded_trace = np.zeros((n_samples, 3))
            padded_trace[n_start:(n_start + L0_entries)] = np.array(traces_showerplane[:, 0:3] * 1e-6) # transform to V/m

            # get frequency spectrum
            spec = np.fft.rfft(padded_trace, axis=-2)

            # get new time and frequency binning
            ff_one = np.fft.rfftfreq(n_samples, tstep * 1e-9)  # frequencies in Hz


            # apply bandwidth cut
            window = np.zeros(len(ff_one))
            window[(ff_one >= flow * 1e6) & (ff_one <= fhigh * 1e6)] = 1
            filtered_spec = np.array([spec[..., 0] * window,
                                    spec[..., 1] * window,
                                    spec[..., 2] * window])

            # get filtered time series
            filt = np.fft.irfft(filtered_spec, n_samples, axis=-1)

            # convert trace x, y and z
            trace_x, trace_y, trace_z = filt[0], filt[1], filt[2]


        # # compute frequency slope in all three polarizations
        # # calculate frequencies
        # ff = np.fft.rfftfreq(len(tracetime_L0), tstep_L0 * 1e-9) * 1e-6 # in MHz
        # mask = (ff > flow) & (ff < fhigh)

        # # Loop over three polarizations
        # for iPol in range(3):
        #     # Fit slope
        #     mask2 = filtered_spec[iPol][mask] > 0
        #     if np.sum(mask2):
        #         xx = ff[mask][mask2]
        #         yy = np.log10(np.abs(filtered_spec[iPol][mask][mask2]))
        #         z = np.polyfit(xx, yy, 1)

        #         # and save frequency slope
        #         frequency_slope[stations][iPol] = z

        

        #########
        # calculate energy fluence from e field traces
        #########


        # length of signal window
        signal_window = 100 # in ns

        # reject stations with low SNR
        # calculate Hilbert envelope
        hilbenv = np.abs(hilbert([trace_x, trace_y, trace_z], axis=1))
        # combine to form a total trace for all channels
        mag_hilbert = np.sum(hilbenv ** 2, axis=0) ** 0.5
        # define peak time
        peak_time_sum = tracetime[np.argmax(mag_hilbert)]

        # contingency for trace time found too early or too late
        # Will be filtered out by SNR cut anyways
        if not noise_or_L0 and not real_data and ((peak_time_sum < 700) or (peak_time_sum > 900)): 
            # set to true value manually.
            # print("Peak found outside of true signal window:", peak_time_sum)
            peak_time_sum = 800
            # print("Setting to true signal time: ", peak_time_sum)

        if real_data:
            peak_time_sum = 520

        # define masks for signal and noise window (hardcoded for GRAND DC2)
        # signal_mask = (300 <= np.arange(len(trace_x))) & (500 > np.arange(len(trace_x)))
        signal_mask = (tracetime > (peak_time_sum - signal_window / 2.)) & (tracetime < (peak_time_sum + signal_window / 2.))
        # print(np.sum(signal_mask))
        # very specific, only for DC2 output where the signal starts at 400ns. 
        # here the noise window goes from 0 to 350ns
        noise_mask = (tracetime < (peak_time_sum - signal_window)) 
        # print(np.sum(noise_mask))

        # automatically change z mask for real data
        if real_data:
            # special signal and noise masks for unfiltered z channel of GP80 output
            offset = 80 # ns
            signal_mask_z = (tracetime > (peak_time_sum - offset - signal_window / 2.)) & (tracetime < (peak_time_sum - offset + signal_window / 2.))
            noise_mask_z = (tracetime < (peak_time_sum - offset - signal_window))

            print("DATA Z-SIGNAL MASK IN EFFECT... USE NORMAL SIGNAL MASK FOR SIMULATIONS!!!!!!!!!!!!")

        else:
            signal_mask_z = signal_mask
            noise_mask_z = noise_mask

         # contingency for if trace time found too early or too late
        if not noise_or_L0:
            if peak_time_sum <= 200:
                noise_mask = tracetime > (max(tracetime) - 256)
                print(f"Noise window adjusted to {min(tracetime[noise_mask])}-{max(tracetime[noise_mask])} ns with {np.sum(noise_mask)} entries! Signal peak at {peak_time_sum} ns")

            elif peak_time_sum > (max(tracetime) - 256):
                noise_mask = tracetime < 450
                print(f"Noise window adjusted to {min(tracetime[noise_mask])}-{max(tracetime[noise_mask])} ns with {np.sum(noise_mask)} entries! Signal peak at {peak_time_sum} ns")

            if np.sum(noise_mask) == 0:
                print("noise window length=0")
                print(np.sum(noise_mask))
                exit()


        # calculate root mean square of noise part of trace
        rms_old = np.sqrt((len(mag_hilbert[noise_mask_z]) - 1) ** (-1) * np.sum([(trace_x[noise_mask_z] - np.mean(trace_x[noise_mask_z])) ** 2, \
                    (trace_y[noise_mask_z] - np.mean(trace_y[noise_mask_z])) ** 2, (trace_z[noise_mask_z] - np.mean(trace_z[noise_mask_z])) ** 2]))
        
        # calculate root mean square of hilbert envelope within noise mask
        rms_hilb = np.sqrt((len(mag_hilbert[noise_mask_z]) - 1) ** (-1) * np.sum([(hilbenv[0][noise_mask_z] - np.mean(hilbenv[0][noise_mask_z])) ** 2, \
                    (hilbenv[1][noise_mask_z] - np.mean(hilbenv[1][noise_mask_z])) ** 2, (hilbenv[2][noise_mask_z] - np.mean(hilbenv[2][noise_mask_z])) ** 2]))
        
        rms_hilb = rms_hilb
        
        # for data join normal mask and shifted mask to get the true maximum of the hilbert envelope
        if real_data:
            joint_mask = np.ma.mask_or(signal_mask, signal_mask_z)
        else:
            joint_mask = signal_mask

        # find peak amplitude of total trace
        peak_amplitude = np.max(mag_hilbert[joint_mask])

        # to filter out low signal stations, define SNR value
        SNR = (peak_amplitude / rms_hilb) ** 2


        # print(f"{stations} Peak amplitude: ", peak_amplitude)
        # print("RMS_old: ", rms_old)
        # print("RMS_hilb: ", rms_hilb)
        
        # print("SNR_old: ", (peak_amplitude / rms_old) ** 2)
        # print("SNR_hilb: ", SNR)

        # plot the reconstructed traces in vxB plane
        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
        # ax[0].set_title("vxB")
        # ax[0].plot(range(len(trace_x)), trace_x * 1e6, c="blue", label="vxB")

        # ax[1].set_title("vxvxB")
        # ax[1].plot(range(len(trace_x)), trace_y * 1e6, c="blue", label="vxvxB")

        # ax[2].set_title("v")
        # ax[2].plot(range(len(trace_x)), trace_z * 1e6, c="blue", label="v")
        # [ax[i].set_xlabel("Time Samples [2ns]") for i in range(3)]
        # [ax[i].set_ylabel(r"Electric Field Strength [$\mu\mathrm{V/m}$]") for i in range(3)]
        # ax[2].legend(loc="upper right")
        # # ax[0].set_xlim(0, 800)
        # plt.tight_layout()
        # plt.savefig(f'test_trace_vxB_{stations}.png')
        # # plt.savefig(f'test_trace_vxB_{recon_string}_{antenna_IDs[stations]}.png')
        # plt.close()

        # calculate fluence vectors for all stations of the event
        # pulse window: 100 ns around signal peak
        # noise window: rest of the trace
        if not noise_or_L0:
            fluence_vectors = np.array([fluence_formula(trace_x[signal_mask], trace_x[noise_mask], t_step=tstep * 1e-9), \
                                        fluence_formula(trace_y[signal_mask], trace_y[noise_mask], t_step=tstep * 1e-9), \
                                        fluence_formula(trace_z[signal_mask_z], trace_z[noise_mask_z], t_step=tstep * 1e-9)])

            # simply use the noise window as the signal window
            noise_fluence = np.array([fluence_formula(trace_x[noise_mask], t_step=tstep * 1e-9, noise_or_L0=True, len_signal=len(signal_mask)), \
                                      fluence_formula(trace_y[noise_mask], t_step=tstep * 1e-9, noise_or_L0=True, len_signal=len(signal_mask)), \
                                      fluence_formula(trace_z[noise_mask_z], t_step=tstep * 1e-9, noise_or_L0=True, len_signal=len(signal_mask))])
            

            # set detector error
            detector_error = 0.075
            if real_data: detector_error = detector_error
            # print(tstep * np.sum(noise_mask))
            # calculate errors for the energy fluence
            fluence_error = total_fluence_error(fluence_vectors, 
                                                noise_fluence, 
                                                efield_RMS=rms_hilb, 
                                                t_step=tstep * 1e-9, 
                                                time_window=tstep * 1e-9 * np.sum(noise_mask_z), 
                                                detector_error=detector_error
                                                )
            
            energy_fluence_error.append(fluence_error)
        
        else: 
            # calculate L0 fluence for validation
            fluence_vectors = np.array([fluence_formula(trace_x[signal_mask], t_step=tstep * 1e-9, noise_or_L0=True), \
                                        fluence_formula(trace_y[signal_mask], t_step=tstep * 1e-9, noise_or_L0=True), \
                                        fluence_formula(trace_z[signal_mask], t_step=tstep * 1e-9, noise_or_L0=True)])
        
        # print("Fluence value with error:", fluence_vectors[0], fluence_error)

        # only add energy fluence to array that will be saved if SNR is sufficently high
        SNR_list.append(SNR)
        energy_fluence_list.append(fluence_vectors)

    return energy_fluence_list, SNR_list, energy_fluence_error



def fluence_formula(pulse_window, noise_window=None, t_step=2e-9, noise_or_L0=False, len_signal=None):
    '''
    Parameters:

    pulse_window:       trace data of the length of the pulse window (usually 100 ns, centred around peak of pulse)

    noise_window:       trace data of the length of noise window (can vary)

    t_step:             length of time step (in seconds)
    '''

    conversion_factor_integrated_signal = 2.65441729e-3 * 6.24150934e18  # to convert V**2/m**2 * s -> J/m**2 -> eV/m**2

    # normal calculation with subtraction of noise fluence from signal fluence    
    if not noise_or_L0:
        if len(noise_window) == 0:
            print("PROBLEM with noise window!")
            exit()

        fluence_vector_component = conversion_factor_integrated_signal * t_step * (np.sum(pulse_window ** 2) - (len(pulse_window) / len(noise_window)) * np.sum(noise_window ** 2))

    # when you only want to calculate noise fluence or the fluence of L0 traces
    else:
        # condition to normalise length of window if used to calculate noise fluence
        if len_signal:
            normalisation = len_signal / len(pulse_window)
        else:
            # for L0 fluence do not modify the fluence calculated here
            normalisation = 1
    
        fluence_vector_component = conversion_factor_integrated_signal * t_step * (np.sum(pulse_window ** 2)) * normalisation

    return fluence_vector_component



def signal_fluence_error(energy_fluence, efield_RMS, t_step, time_window):
    """
    Parameters:

        energy_fluence:     3-dimensional energy fluence vector from the signal window (eV/m²)

        efield_RMS:         root mean square of noise part of trace (micro Volt / m)
        
        t_step:             length of time between time steps in the traces (seconds)

        time_window:        length of time of the noise window (seconds)

    Returns:

        sigma/signal fluence error:     Gaussian + RMS term of the fluence error (eV / m²)
    """

    conversion_factor = 6.242e18 # conversion factor from J/m² to eV/m²

    # gaussian error term
    # term in front of the brackets is in J / m²
    # 1st term in bracket is already in eV / m²
    # 2nd term in brackets is in J / m²
    sigma = np.sqrt(2 * eps_0 * c  * conversion_factor * t_step * efield_RMS ** 2 * \
            (2 * np.abs(energy_fluence[0]) / conversion_factor \
             + eps_0 * c * conversion_factor * time_window * efield_RMS ** 2 ))

    return sigma


def total_fluence_error(energy_fluence, noise_fluence, efield_RMS, t_step, time_window, detector_error):

    """
    Parameters:

        energy_fluence:     3-dimensional energy fluence vector from the signal window (eV/m²)

        noise_fluence:      3-dimensional energy fluence vector from the NOISE window (eV/m²)

        efield_RMS:         root mean square of noise part of trace (micro Volt / m)
        
        t_step:             length of time between time steps in the traces (seconds)

        time_window:        length of time of the noise window (seconds)

        detector_error:     uncertainty on the detector sensitivity 

    Returns:

        total error:        total fluence error (eV / m²)
    """

    # get the gaussian error
    signal_error = signal_fluence_error(energy_fluence, efield_RMS, t_step, time_window)

    # add all component up to find the total error
    # all terms here should be in eV / m²
    # only take the first component of the energy fluence, the vxB fluence
    error  = np.sqrt(signal_error ** 2 + (2 * detector_error * energy_fluence[0]) ** 2 + noise_fluence[0] ** 2)

    # print("Signal error: ", signal_error)
    # print("Detector error: ", 2 * detector_error * energy_fluence[0])
    # print("Noise Fluence: ", noise_fluence[0])
    # print("vxB Fluence and error: ", energy_fluence[0], error)

    return error