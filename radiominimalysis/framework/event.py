from __future__ import absolute_import, division, print_function

try:
    import cPickle as Pickle
except ImportError:
    import pickle

from radiominimalysis.framework.parameter_storage import (
    ParameterStorage,
    compare_parameter_storage,
)
from radiominimalysis.framework import parameter_serialization, parameters
import radiominimalysis.framework.shower

import warnings
import logging

# create logger
logger = logging.getLogger("BaseEvent")


class BaseEvent(ParameterStorage):
    """
    Inherits from ParameterStorage which stores the parameters related to a radio event
    in dictionaries.
    """

    def __init__(self, run_number, event_id):
        """
        Initialises the class with a run number and event ID.

        Then ParameterStorage's __init__ is called to generate a dictionary for event level parameters

        Station Parameters are stored in separate dictionaries
        """

        self._run_number = run_number
        self._id = event_id

        # store general event information
        super().__init__(key_type=parameters.eventParameters)

        # store station related parameter in an array: (n_stations, parameter)
        self._n_stations = 0
        self._station_parameters = {}
        self._station_parameter_errors = {}

        # set default shower type
        self.__default_shower_type = parameters.eventParameters.sim_shower
        self.__showers = []

    def set_default_shower_type(self, key):
        """
        Function takes key and sets default shower type to it.

        Raises error if key doesn't match one of the 3 types
        """

        # check that key matches a shower type
        if not (
            key == parameters.eventParameters.sim_shower
            or key == parameters.eventParameters.sd_shower
            or key == parameters.eventParameters.rd_shower
            or key == parameters.eventParameters.GRAND_shower
        ):
            logger.error("You try to set an invald shower type: {}".format(key))
            # raise error
            raise ValueError("You try to set an invald shower type: {}".format(key))

        # set default shower type
        self.__default_shower_type = key

    def get_shower(self, key=None):
        """
        Retrieves the shower object associated with the specified key.

        Returns all shower objects associated with the specified key.

        Raises error if no shower is found with the specified key.
        """

        # use default shower type as key if none is specified
        if key is None:
            key = self.__default_shower_type

        # return all air showers in the event object if key matches
        for shower in self.get_showers():
            # print(shower.get_shower_type(), key)
            # print(key)
            # print(shower.get_shower_type() == key)
            if shower.get_shower_type() == key:
                return shower

        # only shower with no type
        if len(self.__showers) == 1 and self.__showers[0].get_shower_type() == None:
            warnings.warn(
                "Event containts only one shower with no type set. Returning this one..."
            )
            return self.__showers[0]

        # print out the type of the single shower and raise error
        print("Shower has type:", self.__showers[0].get_shower_type())
        logger.error('Could not return any shower with key "{}".'.format(key))
        raise ValueError('Could not return any shower with key "{}".'.format(key))

    def get_showers(self):
        """
        Retrieves all the shower objects associated with an event object.

        Yield the shower object associated with the event (as a generator)
        """

        for shower in self.__showers:
            yield shower

    def has_shower(self, key=None):
        """
        Checks if an event object contains a shower of the specified type

        Returns number of shower if no key is specified, otherwise bool value.
        """

        # if no key is specified, return the amount of shower in event object
        if key is None:
            return len(self.__showers) > 0

        # loops over all showers and stops the loop if a shower of specified type is found
        has = False
        for shower in self.get_showers():
            has = shower.get_shower_type() == key
            if has:
                break

        # returns True if shower type is found, False if it is no
        return has

    def add_shower(self, shower):
        """
        Adds a shower object to the event.

        Parameter: shower object to add
        """

        # append shower object to event object
        self.__showers.append(shower)

    def get_id(self):
        """
        Retrieves the ID of the event.

        Returns the id (int) of the event.
        """

        return self._id

    def get_run_number(self):
        """
        Retrieves the run number of the event.

        Returns the run_number (int) of the event.
        """

        return self._run_number

    def get_run_number_str(self):
        """
        Retrieves the run number of the event as a string with leading zeros.
        """

        # convert run number to string and bring it into XXXXXX format
        return "{:06d}".format(self._run_number)

    def set_number_of_stations(self, n):
        """
        Sets the number of stations associated with the event.

        Parameters: The number of stations (n, int)
        """

        self._n_stations = n

    def get_number_of_stations(self):
        """
        Retrieves the number of stations associated with the event.

        Returns:
            n_stations (int): The number of stations.
        """
        return self._n_stations

    def get_station_parameter(self, key):
        """
        Retrieves the value of a station parameter associated with the specified key.

        Parameters:
            key (stationParameters): The key specifying the station parameter.

        Returns:
            value of stationParameter
        """

        # key must be enum format, otherwise error is raised
        if not isinstance(key, parameters.stationParameters):
            logger.error(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )
            raise ValueError(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )

        # return value of parameter
        return self._station_parameters[key]

    def set_station_parameter(self, key, value):
        """
        Sets the value of a station parameter associated with the specified key.

        Parameters:
            key (stationParameters): The key specifying the station parameter.
            value: The value to be set for the station parameter.
        """

        # key must be enum format, otherwise error is raised
        if not isinstance(key, parameters.stationParameters):
            logger.error(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )
            raise ValueError(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )

        # sets value of station parameter to specified value
        self._station_parameters[key] = value

    def has_station_parameter(self, key):
        """
        Checks if the event has a station parameter associated with the specified key.

        Parameters:
            key (stationParameters): The key specifying the station parameter.

        Returns:
            has (bool): True if the event has a station parameter with the specified key, False otherwise.
        """

        # key must be enum format, otherwise error is raised
        if not isinstance(key, parameters.stationParameters):
            logger.error(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )
            raise ValueError(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )

        # True if the event has a station parameter with the specified key, False otherwise.
        return key in self._station_parameters

    def get_station_parameter_error(self, key):
        """
        Retrieves the error value of a station parameter associated with the specified key.

        Parameters:
            key (stationParameters): The key specifying the station parameter.

        Returns:
            value: The error value of the station parameter.
        """

        # key must be enum format, otherwise error is raised
        if not isinstance(key, parameters.stationParameters):
            logger.error(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )
            raise ValueError(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )

        # returns error of specified parameter
        return self._station_parameter_errors[key]

    def set_station_parameter_error(self, key, value):
        """
        Sets the error value of a station parameter associated with the specified key.

        Parameters:
            key (stationParameters): The key specifying the station parameter.
            value: The error value to be set for the station parameter.
        """

        # key must be enum format, otherwise error is raised
        if not isinstance(key, parameters.stationParameters):
            logger.error(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )
            raise ValueError(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )

        # sets error of station parameter to the specified value
        self._station_parameter_errors[key] = value

    def has_station_parameter_error(self, key):
        """
        Checks if the event has an error value for a station parameter associated with the specified key.

        Parameters:
            key (stationParameters): The key specifying the station parameter.

        Returns:
            has (bool): True if the event has an error value for the station parameter with the specified key, False otherwise.
        """

        # key must be enum format, otherwise error is raised
        if not isinstance(key, parameters.stationParameters):
            logger.error(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )
            raise ValueError(
                'parameter key "{}" needs to be of type parameters.stationParameters'.format(
                    key
                )
            )

        # True if the event has an error value for the station parameter with the specified key, False otherwise.
        return key in self._station_parameter_errors

    def __eq__(self, other):
        """
        Compares the event object with another event object to check for equality.

        Parameters:
            other: The other event object to compare with.

        Returns:
            are_equal (bool): True if the event objects are equal, False otherwise.
        """

        # execute ParameterStorage's equality check first
        if not super().__eq__(other):
            return False

        # check if other event has same shower type as this one
        for shower in self.get_showers():
            if not other.has_shower(shower.get_shower_type()):
                print(shower.get_shower_type(), "in left but not right event.")
                return False

            shower2 = other.get_shower(shower.get_shower_type())

            if not shower == shower2:
                return False

        # check if this event object has the same shower type as the other event object
        for shower in other.get_showers():
            if not self.has_shower(shower.get_shower_type()):
                print(shower.get_shower_type(), "in right but not left event.")
                return False

        # initialise ParameterStorage object with this event object's station parameters
        station_parameters_left = ParameterStorage(parameters.stationParameters)
        station_parameters_left.add_parameter_dict(
            self._station_parameters, self._station_parameter_errors
        )

        # initialise ParameterStorage object with the other event object's station parameters
        station_parameters_right = ParameterStorage(parameters.stationParameters)
        station_parameters_right.add_parameter_dict(
            other._station_parameters, other._station_parameter_errors
        )

        # use ParameterStorage compare function to compare them
        if not compare_parameter_storage(
            station_parameters_left,
            station_parameters_right,
            parameters.stationParameters,
            False,
        ):
            return False

        return True

    def serialize(self):
        """
        Serializes the event object into a pickle format.

        Returns:
            serialized_data: The serialized representation of the event object.
        """

        shower_pkls = []
        for shower in self.get_showers():
            shower_pkls.append(shower.serialize())

        data = {
            "_station_parameters": parameter_serialization.serialize(
                self._station_parameters
            ),
            "_station_parameter_errors": parameter_serialization.serialize(
                self._station_parameter_errors
            ),
            "_parameters": parameter_serialization.serialize(self._parameters),
            "_parameter_errors": parameter_serialization.serialize(
                self._parameter_errors
            ),
            "showers": shower_pkls,
            "_run_number": self._run_number,
            "_id": self._id,
            "_n_stations": self._n_stations,
        }

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        """
        Deserializes the pickle-formatted data and initializes the event object.

        Parameters:
            data_pkl: The pickle-formatted data to be deserialized.
        """

        data = pickle.loads(data_pkl)

        if "showers" in data.keys():
            for shower_pkl in data["showers"]:
                shower = radiominimalysis.framework.shower.Shower(None)
                shower.deserialize(shower_pkl)
                self.add_shower(shower)

        self._station_parameters = parameter_serialization.deserialize(
            data["_station_parameters"], parameters.stationParameters
        )
        self._parameters = parameter_serialization.deserialize(
            data["_parameters"], parameters.eventParameters
        )
        if "_parameter_errors" in data:
            self._parameter_errors = parameter_serialization.deserialize(
                data["_parameter_errors"], parameters.eventParameters
            )
        else:
            self._parameter_errors = {}

        if "_station_parameter_errors" in data:
            self._station_parameter_errors = parameter_serialization.deserialize(
                data["_station_parameter_errors"], parameters.stationParameters
            )
        else:
            self._station_parameter_errors = {}

        self._run_number = data["_run_number"]
        self._id = data["_id"]
        self._n_stations = data["_n_stations"]
