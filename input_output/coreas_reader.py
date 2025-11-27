import RadioAnalysis.framework.revent
import RadioAnalysis.framework.shower
from RadioAnalysis.framework.parameters import showerParameters as shp
from RadioAnalysis.framework.parameters import stationParameters as stp
from RadioAnalysis.framework.parameters import eventParameters as evp

from radiotools import helper as rdhelp, coordinatesystems

import h5py
import numpy as np
import time
import re
import sys
import copy
import warnings
from datetime import timedelta

conversion_fieldstrength_cgs_to_SI = 2.99792458e4


class readCoREASShower:

    def __init__(self, input_files, verbose=False, read_in_highlevel_file=True, add_traces_from_highlevel=False, add_traces_from_observer=False):
        """
        init method

        initialize readCoREASShower

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        verbose: bool
        read_in_highlevel_file: bool
            if true, highlevel files are read
        """
        if len(input_files) == 0:
            sys.exit("No input file(s)! Abort...")

        if not isinstance(input_files, list):
            input_files = [input_files]

        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_files = input_files
        self.__current_input_file = 0
        self.__verbose = verbose

        self.__read_highlevel = read_in_highlevel_file
        self.__add_traces_from_highlevel = add_traces_from_highlevel
        self.__add_traces_from_observer = add_traces_from_observer


    def run(self):
        """
        read in a full CoREAS simulation

        """
        while (self.__current_input_file < len(self.__input_files)):
            t = time.time()
            t_per_event = time.time()

            if self.__verbose:
                print('Reading %s ...' % self.__input_files[self.__current_input_file])

            if self.__read_highlevel:
                try:
                    evt = read_coreas_highlevel_files(
                        self.__input_files[self.__current_input_file], self.__add_traces_from_highlevel)
                except Exception as e:
                    print("Failure to read {}\nError: {}".format(self.__input_files[self.__current_input_file], e))
            else:
                ValueError("not implemented....")

            if self.__add_traces_from_observer:
                if not isinstance(evt, list):
                    evt = [evt]
                for e in evt:
                    try:
                        set_traces_from_observers(
                            e, self.__input_files[self.__current_input_file])
                    except KeyError:
                        sys.exit("Failed to read in traces from observer. You need full hdf5 files for this (not highlevel files).")

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            self.__current_input_file += 1

            if isinstance(evt, list):
                for e in evt:
                    yield e
            else:
                yield evt

        dt = timedelta(seconds=self.__t)
        print("total time used by this module is {}".format(dt))
        print("\tcreate event structure {}".format(
            timedelta(seconds=self.__t_event_structure)))
        print("\per event {}".format(timedelta(seconds=self.__t_per_event)))

    def end(self):
        return timedelta(seconds=self.__t)

    def get_factory(self, factory=None):
        from RadioAnalysis.framework.factory import EventFactory
        if factory is None:
            factory = EventFactory()

        [factory.add_event(evt) for evt in self.run()]
        return factory

    def get_events(self):
        return [evt for evt in self.run()]


def read_coreas_highlevel_files(input_file, add_traces=False):
    corsika = h5py.File(input_file, "r")
    f_coreas = corsika["CoREAS"]

    evt = RadioAnalysis.framework.revent.REvent(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])
    evt.set_parameter(evp.file, input_file.split("/")[-1])
    evt.set_parameter(evp.time, f_coreas.attrs["GPSSecs"])
    evt.set_parameter(evp.refractive_index_at_sea_level, f_coreas.attrs["GroundLevelRefractiveIndex"])

    # create shower object
    shower = RadioAnalysis.framework.shower.Shower(evp.sim_shower)
    
    shower.set_parameter(shp.atmosphere_model, corsika["inputs"].attrs["ATMOD"])
    # in auger CS, function just works if inputs are in highlevel file
    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    evt.set_parameter(evp.magnetic_field_vector, magnetic_field_vector)
    shower.set_parameter(shp.azimuth, azimuth)  # normalized from 0 - 360
    shower.set_parameter(shp.zenith, zenith)

    # calculated geomagnetic angle
    shower_axis = rdhelp.spherical_to_cartesian(zenith, azimuth)
    alpha = np.arccos(np.dot(shower_axis, magnetic_field_vector) \
        / (np.linalg.norm(shower_axis) * np.linalg.norm(magnetic_field_vector)))
    shower.set_parameter(shp.geomagnetic_angle, alpha)

    # radio obs lvl might differ from particle obs lvl "corsika["inputs"].attrs["OBSLEV"] / 100"
    radio_obs_lvl = f_coreas.attrs["CoreCoordinateVertical"] / 100  # conversion in meter
    radio_core_north = f_coreas.attrs["CoreCoordinateNorth"] / 100  # conversion in meter
    radio_core_west = f_coreas.attrs["CoreCoordinateWest"] / 100  # conversion in meter
    shower.set_parameter(shp.core, np.array([-radio_core_west, radio_core_north, radio_obs_lvl]))  # in principle the core could be set to a different value
    shower.set_parameter(shp.observation_level, radio_obs_lvl)

    # test for coordinate system 
    # ctrans = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)
    # print("Transformed core: ", ctrans.transform_to_vxB_vxvxB(np.array([-radio_core_west, radio_core_north, radio_obs_lvl])))

    if "inputs" in corsika:
        f_inputs = corsika["inputs"]
        shower.set_parameter(shp.primary_particle, f_inputs.attrs["PRMPAR"])
    else:
        print("No inputs group present hdf5 file {}".format(input_file))

    if "highlevel" not in corsika:
        
        print("No highlevel group present hdf5 file {}".format(input_file))
        evt.add_shower(shower)
        return evt

    # get highlevel information calculated with the coreas_to_hdf5.py converter of the coreas trunk
    f_highlevel = corsika["highlevel"]

    shower.set_parameter(shp.energy, f_highlevel.attrs["energy"])
    shower.set_parameter(shp.electromagnetic_energy, f_highlevel.attrs["Eem"])
    shower.set_parameter(shp.invisible_energy, f_highlevel.attrs["Einv"])

    # this is calculated from CoREAS using sparse tables. dont trust it!
    if "DistanceOfShowerMaximum" in f_coreas.attrs:
        shower.set_parameter(
            shp.distance_to_shower_maximum_geometric2, f_coreas.attrs["DistanceOfShowerMaximum"] / 100)
        print("Added distance to shower maximum from Corsika!")
   
    if "Gaisser-Hillas-Fit" in f_highlevel.attrs and len(f_highlevel.attrs["Gaisser-Hillas-Fit"]):
        shower.set_parameter(shp.xmax2, f_highlevel.attrs["Gaisser-Hillas-Fit"][2])
        warnings.warn('Take GH fit result from CORSIKA. That can be wrong for very inclined shower ~85 deg')
    
    if "gaisser_hillas_dEdX" in f_highlevel.attrs and len(f_highlevel.attrs["gaisser_hillas_dEdX"]):
        shower.set_parameter(shp.xmax, f_highlevel.attrs["gaisser_hillas_dEdX"][2])
        print(f_highlevel.attrs["gaisser_hillas_dEdX"][2])
        warnings.warn('Take GH fit result from hdf5-coverter. This can be different to CORSIKA result')

    evt.add_shower(shower)

    planes = list(f_highlevel.keys())
    if len(planes) > 1:
        warnings.warn('Multiple observation plane, creating an event for each plane')
        events = []
        for plane in planes:
            # get observer data
            obs_plane = f_highlevel[plane]
            evt_obs = copy.deepcopy(evt)
            shower = evt_obs.get_shower()

            try:
                shower.set_parameter(shp.slicing_method, np.array(
                    obs_plane.attrs["slicing_method"]))
                try:
                    shower.set_parameter(shp.slicing_edges, np.array(
                        obs_plane["slicing_boundaries"][0]))
                except:
                    bounds = np.squeeze(
                        np.unique(obs_plane["slicing_boundaries"], axis=0))
                    if bounds.ndim == 1:
                        bounds = np.vstack([bounds, np.array([0, 0])])
                    shower.set_parameter(shp.slicing_edges, bounds)
            except:
                pass

            # in case simulation is not a star shaped
            try:
                shower.set_parameter(shp.radiation_energy, obs_plane.attrs["radiation_energy"])
            except KeyError:
                pass

            pos = np.array(obs_plane["antenna_position"])
            x, y, z = np.mean(pos, axis=0)
            if np.abs(x) > 1 or np.abs(y) > 1:
                warnings.warn("Stations are not centric ...")
            
            if np.std(pos[:, -1]) < 1e-6:
                warnings.warn("Stations seem to be on a plane. Set obs lev and core to: %f" % z)
                shower.set_parameter(shp.core, np.array([-radio_core_west, radio_core_north, z]))  # in principle the core could be set to a different value
                shower.set_parameter(shp.observation_level, z)

            set_data_from_obs_plane(evt_obs, obs_plane)
       
            if add_traces:
                set_traces_from_highlevel(evt_obs, plane=obs_plane)
            
            if len(pos) > 5:
                events.append(evt_obs)
        
        return events

    else:
        # get observer data
        obs_plane = f_highlevel[planes[0]]

        # in case simulation is not a star shaped
        for pair in [[shp.radiation_energy, "radiation_energy"], 
                     [shp.geomagnetic_energy, "radiation_energy_geo"],
                     [shp.charge_excess_energy, "radiation_energy_ce"]]:
            try:
                shower.set_parameter(pair[0], obs_plane.attrs[pair[1]])
            except KeyError:
                pass

        set_data_from_obs_plane(evt, obs_plane)

        if add_traces:
            set_traces_from_highlevel(evt, input_file)
    
        return evt

    # should not reach this point
    raise Exception


def set_data_from_obs_plane(evt, obs_plane):
    # Station level
    evt.set_number_of_stations(len(np.array(obs_plane["antenna_position"])))

    pos = np.array(obs_plane["antenna_position"])
    evt.set_station_parameter(stp.position, pos)
    #evt.set_station_parameter("station_position_vBvvB", np.array(obs_plane["antenna_position_vBvvB"]))
    evt.set_station_parameter(stp.name, np.array(obs_plane["antenna_names"]))

    #evt.set_station_parameter("energy_fluence", np.array(obs_plane["energy_fluence"]))
    evt.set_station_parameter(stp.energy_fluence, np.array(
        obs_plane["energy_fluence_vector"]))  # it is not a real vector
    evt.set_station_parameter(stp.peak_amplitude, np.array(
        obs_plane["amplitude"]))  # it is not a real vector
    evt.set_station_parameter(
        stp.frequency_slope, np.array(obs_plane["frequency_slope"]))

    try:
        evt.set_station_parameter(
            stp.stokes_parameter, np.array(obs_plane["stokes_parameter"]))
    except KeyError:
        pass

    try:
        evt.set_station_parameter(
            stp.time, np.array(obs_plane["peak_time"]))
    except KeyError:
        pass


def set_traces_from_highlevel(evt, input_file=None, plane=None):
    if plane is None:
        if input_file is None:
            sys.exit("Invalid arguments set_traces_from_highlevel: Both are None. Abort ...")
        fhigh = h5py.File(input_file, "r")

        highlevel = fhigh["highlevel"]

        plane = highlevel[list(highlevel.keys())[0]]
    else:
        if input_file is None:
            sys.exit("Invalid arguments set_traces_from_highlevel: Both are not None. Abort ...")

    times_filtered = plane['times_filtered']
    traces_filtered = plane['traces_filtered']

    evt.set_station_parameter(stp.traces_filtered_downsampled, np.array(traces_filtered))
    evt.set_station_parameter(stp.times_filtered_downsampled, np.array(times_filtered))


def set_traces_from_observers(evt, input_file):
    corsika = h5py.File(input_file, "r")
    coreas = corsika["CoREAS"]

    # from radiotools import coordinatesystems
    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    ctrans = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)

    traces = []
    times = []
    for _, obs in coreas["observers"].items():
        data = np.copy(np.array(obs))

        # convert CORSIKA to AUGER coordinates (AUGER y = CORSIKA x, AUGER x = - CORSIKA y
        data[:, 1], data[:, 2] = -obs[:, 2], obs[:, 1]

        data[:, 1:4] = ctrans.transform_to_vxB_vxvxB(data[:, 1:4])

        # convert to SI units
        data[:, [1, 2, 3]] *= conversion_fieldstrength_cgs_to_SI
        traces.append(data[:, [1, 2, 3]])
        times.append(data[:, 0])

    evt.set_station_parameter(stp.traces, np.array(traces))
    evt.set_station_parameter(stp.times, np.array(times))


def antenna_id(antenna_name, default_id):
    """
    This function parses the antenna name given in a CoREAS simulation and tries to find an ID
    It can be extended to other name patterns
    """

    if re.match(b"AERA_", antenna_name):
        new_id = int(antenna_name.strip("AERA_"))
        return new_id
    else:
        return default_id


def get_angles(corsika):
    """
    Converting angles in corsika coordinates to local coordinates
    """

    if 'inputs' not in corsika:
        raise KeyError("\"inputs\" is not stored in the hdf5 file. You might have a file converted with an deprecated CoREAS trunk,")

    zenith = np.deg2rad(corsika['inputs'].attrs["THETAP"][0])
    azimuth = rdhelp.get_normalized_angle(3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]))

    # this is an option for the modified coreas_to_hdf5 converter that removes the auger coordinate transformation
    # azimuth = rdhelp.get_normalized_angle(np.pi + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]))

    Bx, Bz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(Bz, Bx)

    B_strength = (Bx ** 2 + Bz ** 2) ** 0.5

    # in local coordinates north is + 90 deg
    magnetic_field_vector = B_strength * rdhelp.spherical_to_cartesian(np.pi * 0.5 + B_inclination, 0 + np.pi * 0.5)

    # this is an option for the modified coreas_to_hdf5 converter that removes the auger coordinate transformation
    # magnetic_field_vector = B_strength * np.array([np.cos(B_inclination), 0, -np.sin(B_inclination)])

    return zenith, azimuth, magnetic_field_vector


def read_in_hdf_files():
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('paths', metavar='Path to ADST root file(s)', type=str, nargs='*', default=[],
                    help='Choose input file(s).')

    parser.add_argument('-of', '--output_filename', metavar='file name', type=str, help='', required=True)
    
    parser.add_argument('--verbose', action='store_true', help='(default: false)')

    args = parser.parse_args()

    if np.any([x[-5:] != '.hdf5' for x in args.paths]):
        sys.exit("Only hdf5 files are supported. Found at least one non hdf5 file. Abort ...")

    # input_files, verbose=False, read_in_highlevel_file=True, add_traces_from_highlevel=False, add_traces_from_observer=False
    reader = readCoREASShower(
        input_files=args.paths, verbose=args.verbose)

    factory = reader.get_factory()
    reader.end()
    factory.save_events_to_file(args.output_filename)


if __name__ == "__main__":
    read_in_hdf_files()
