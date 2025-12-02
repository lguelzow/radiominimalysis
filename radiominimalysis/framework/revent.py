from __future__ import absolute_import, division, print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle

from radiominimalysis.framework.event import BaseEvent
from radiominimalysis.framework.parameters import stationParameters as stp
from radiominimalysis.framework.parameters import showerParameters as shp
from radiominimalysis.framework.parameters import eventParameters as evp
from radiotools import coordinatesystems as cs
from radiotools import helper as rdhelp
from radiotools.atmosphere import models as atm

import numpy as np

import logging


atmodels = {}


class REvent(BaseEvent):
    '''
    Inherits from BaseEvent.

    Methods:
        get_atmosphere(key=None): Get the atmosphere model associated with the event.
        get_shower_axis(key=None): Get the shower axis associated with the event.
        get_all_parameters(): Get a list of all parameters in the event.
        get_total_energy_fluence(): Get the total energy fluence of the event.
        get_geomagnetic_angle(shower_axis=None, key=None): Get the geomagnetic angle of the event.
        get_station_position_cs(key=None, core=None): Get the station position in the standard coordinate system.
        get_station_position_vB_vvB(key=None, core=None): Get the station position in the vxB vxvxB coordinate system.
        get_station_position_early_late(key=None): Get the station position in the early-late coordinate system.
        get_station_axis_distance(key=None, core=None): Get the distance from the station to the shower axis.
        get_station_angle_to_vB(key=None, core=None): Get the angle of the station relative to vB.
        get_station_angle_to_x(key=None, core=None): Get the angle of the station relative to x.
        get_coordinate_transformation(key=None): Get the coordinate transformation for the event.
    '''

    def __init__(self, run_number=-1, event_id=-1):
        '''
        Initializes an REvent object using the constructor of BaseEvent

        Parameters:
            run_number (int): The run number of the event.
            event_id (int): The event ID.
        '''
        
        # call constructor of BaseEvent
        super(REvent, self).__init__(run_number, event_id)


    def get_atmosphere(self, key=None):
        '''
        Get the atmosphere model associated with the event.

        Parameters:
            key: The key to identify the shower type. If None, the default shower type is used.

        Returns:
            atmodels[at_model] (atm.Atmosphere): The atmosphere model.
        '''

        global atmodels
        at_model = self.get_shower(key).get_parameter(shp.atmosphere_model)

        # add atmosphere model to global dictionary if not already in it
        if at_model not in atmodels:
            atmodels[at_model] = atm.Atmosphere(at_model)

        # return the atmosphere model number
        return atmodels[at_model]


    def get_shower_axis(self, key=None):
        '''
        Get the shower axis associated with the event.

        Parameters:
            key: The key to identify the shower type. If None, the default shower type is used.

        Returns:
            The shower axis coordinates in cartesian form.
        '''

        # return all showers in the oject of the specified type type
        shower = self.get_shower(key)

        # calculate shower axis from zenith and azimuth angle 
        return rdhelp.spherical_to_cartesian(shower.get_parameter(shp.zenith), shower.get_parameter(shp.azimuth))

    def get_all_parameters(self):
        '''
        Get a list of all parameters in the event.

        Returns:
            A list of all parameters in the event
        '''

        # return a list of the keys existing in the object
        return list(self._parameters.keys())

    # def get_magnetic_field_vector(self):
    #     # get_magneticfield_azimuth -> declination + 90 deg (rotation in auger cs?)
    #     return rdhelp.get_magnetic_field_vector_from_inc(
    #             self.get_parameter("magnetic_field_inclination"), self.get_parameter("magnetic_field_declination"))

    def get_total_energy_fluence(self):
        """
        Calculates the total energy fluence of the event.

        Returns:
            total_energy_fluence: The total energy fluence.
        """

        # sums the vectors with the energy fluence for each array components and return the total fluence as an array
        return np.sum(self.get_station_parameter(stp.energy_fluence), axis=-1)


    def get_geomagnetic_angle(self, shower_axis=None, key=None):
        """
        Calculates the geomagnetic angle between the shower axis and the magnetic field vector.

        Parameters:
            The shower axis vector. If not provided, it is obtained from the event parameters.
            key: The key to retrieve the shower.

        Returns:
            The geomagnetic angle in radians.
        """
        
        # get magnetic field vector from event parameters
        magnetic_field_vector = self.get_parameter(evp.magnetic_field_vector)
        # also get shower axis from event if it not provided
        if shower_axis is None:
            shower_axis = self.get_shower_axis(key)
        
        # return the angle between the vector and the axis
        return np.arccos(np.dot(shower_axis, magnetic_field_vector) \
            / (np.linalg.norm(shower_axis) * np.linalg.norm(magnetic_field_vector)))


    def get_station_position_cs(self, key=None, core=None):
        """
        Calculates the station position in the shower coordinate system.
        Core is (0, 0, 0) in shower coordinate system.

        Parameters:
            key: The key to retrieve the shower.
            core: The shower core position. If not provided, it is obtained from the shower object.

        Returns:
            station_position_cs: The station position in the shower coordinate system.
        """

        # get shower core from shower object, if not provided
        if core is None:
            core = self.get_shower(key).get_parameter(shp.core)
        # return station position relative to core position
        return self.get_station_parameter(stp.position) - core


    def get_station_position_vB_vvB(self, key=None, core=None, realistic_input=False):
        """
        Returns the antenna position in the vxB/vxvxB coordinate system.

        Parameters:
            key: The key to retrieve the shower.
            core: The shower core position. If not provided, it is obtained from the shower object.

        Returns:
            station_position_vB_vvB: The antenna position in the vxB vxvxB coordinate system.
        """

        # get coordinate trafo from radiotools module
        cs = self.get_coordinate_transformation(key, realistic_input=realistic_input)

        if realistic_input:
            # get shower object with provided key
            shower = self.get_shower(key)
            core = shower.get_parameter(shp.core_estimate)

        # get shower core from shower object, if not provided
        if core is None:
            core = self.get_shower(key).get_parameter(shp.core)


        # use trafo to transform station position in shower coordinate system to vxb/vxvxB system
        return cs.transform_to_vxB_vxvxB(self.get_station_parameter(stp.position), core=core)
    

    def get_station_position_early_late(self, key=None):
        """
        Returns the station position in the early-late/shower plane coordinate system.

        Parameters:
            key: The key to retrieve the shower.

        Returns:
            station_position_early_late: The station position in the early-late/shower plane coordinate system.
        """
        
        # get coordinate trafo from radiotools module
        cs = self.get_coordinate_transformation(key)
        # use function from radiotools to to transform shower coordinate system position to shower plane
        return cs.transform_to_early_late(self.get_station_parameter(stp.position),
                                          core=self.get_shower().get_parameter(shp.core))
    

    def get_station_axis_distance(self, key=None, core=None):
        """
        Calculates the distance between the station and the shower axis.

        Parameters:
            key (str, optional): The key to retrieve the shower.
            core (ndarray, optional): The shower core position. If not provided, it is obtained from the key.

        Returns:
            station_axis_distance (float): The distance between the station and the shower axis.
        """

        # Felix comment: calculation via rotation in vxB-vxvxB system is far less efficient
        
        # import shower object with provided key
        shower = self.get_shower(key)

        # get shower core from shower object, if not provided
        if core is None:
            core = shower.get_parameter(shp.core)

        # use function from radiotools to calculate station distance to shower axis
        # calculate needed zenith, azimuth and station position first
        return rdhelp.get_distance_to_showeraxis(core=core,
                                                 zenith=shower.get_parameter(shp.zenith),
                                                 azimuth=shower.get_parameter(shp.azimuth),
                                                 antennaPosition=self.get_station_parameter(stp.position))


    def get_station_angle_to_vB(self, key=None, core=None):
        """
        Calculates the angle between the station position and the vxB-vxvxB plane in the vB coordinate system.

        Parameters:
            key (str, optional): The key to retrieve the shower.
            core: The shower core position. If not provided, it is obtained from the key.

        Returns:
            station_angle_to_vB: The angle between the station position and the vxB vxvxB coordinate system.
        """
        
        # get station position in vxB-vxvxB system
        station_pos = self.get_station_position_vB_vvB(key=key, core=core)

        # return station angle to the vxb-vxvxB plane
        return rdhelp.get_normalized_angle(np.arctan2(station_pos[:, 1], station_pos[:, 0]))
    

    def get_station_angle_to_x(self, key=None, core=None):
        """
        Calculates the angle between the station position and the x-axis in the shower coordinate system.

        Parameters:
            key: The key to retrieve the shower.
            core: The shower core position. If not provided, it is obtained from the key.

        Returns:
            station_angle_to_x (ndarray): The angle between the station position and the x-axis in the shower coordinate system.
        """

        # get station position in shower coordinate system
        station_pos_cs = self.get_station_position_cs(key=key, core=core)

        # return station angle to the x-axis
        return rdhelp.get_normalized_angle(np.arctan2(station_pos_cs[:, 1], station_pos_cs[:, 0]))


    def get_coordinate_transformation(self, key=None, realistic_input=False):
        """
         Retrieves the coordinate transformation object for the shower.

        Parameters:
            key: The key to retrieve the shower.
            realistic_input: Whether to use MC values or reconstructed values

        Returns:
            coordinate_transformation (cs.CoordinateTransformation): The coordinate transformation object.
        """
        
        # if event doesn't have a magnetic field vector, calculate it
        if not self.has_parameter(evp.magnetic_field_vector):
            # self.set_parameter("magnetic_field_vector",
            #     rdhelp.get_magnetic_field_vector_from_inc(
            #         self.get_parameter("magnetic_field_inclination"),
            #         self.get_parameter("magnetic_field_declination")))
            # raise KeyError("Magnetic field should have been caluclated")
            print("Magnetic field should have been calculated")
            magnetic_field_vector = rdhelp.get_magnetic_field_vector()
        else:
            magnetic_field_vector = self.get_parameter(evp.magnetic_field_vector)

        # get shower object with provided key
        shower = self.get_shower(key)

        # use reconstructed arrival direction
        if realistic_input:
            zenith = shower.get_parameter(shp.zenith_recon)
            azimuth = shower.get_parameter(shp.azimuth_recon)

        else:
            zenith = shower.get_parameter(shp.zenith)
            azimuth = shower.get_parameter(shp.azimuth)

        # return coordinate transformation object associated with shower coordinate system of the event
        return cs.cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)


# does this function even work? it's also inside the class
def get_station_axis_distance(revent):
    """
    Retrieves the station axis distance of the given event.
    It's the distance between the station and the shower core on the ground

    Parameters:
        revent (RadioEvent): The RadioEvent object.

    Returns:
        The station axis distance.
    """
    
    # return axis distance of the 
    return revent.get_station_axis_distance()
