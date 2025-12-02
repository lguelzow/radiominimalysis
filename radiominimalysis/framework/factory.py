from __future__ import absolute_import, division, print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import sys
import time
from datetime import timedelta

import radiominimalysis.framework.revent
from radiominimalysis.framework.parameters import (
    stationParameters,
    showerParameters,
    eventParameters,
)


class EventFactory(object):
    """
    A factory class for creating and managing events.

    Parameters:
        events (list): A list of events to initialize the factory with (default: None).
        input_file (str): The path to a file containing events to read from (default: None).

    Attributes:
        __events (list): The internal list of events.
    """

    def __init__(self, events=None, input_file=None):

        # initialise class with internal, empty list of events
        self.__events = []

        # read in events from a given list or array
        if events is not None:
            for event in events:
                # check that event is a radio event
                if not isinstance(event, radiominimalysis.framework.revent.REvent):
                    # print error message
                    print(event, type(event))
                    sys.exit(
                        "Event must be a radiominimalysis.framework.revent.REvent object"
                    )

                # append event to internal list of events
                self.__events.append(event)

        # read in events from a given input file
        if input_file is not None:
            # use function defined in this module to read in events
            self.read_events_from_file(input_file)

    def add_event(self, event):
        """
        Adds an event to the factory.

        Parameters:
            event: The event object to add.
        """

        # append event to internal list of events
        self.__events.append(event)

    def get_events(self):
        """
        Retrieves the events stored in the factory object.

        Returns:
            events (ndarray): An array of event objects.
        """

        # return events stored in the factory object as a numpy array
        return np.asarray(self.__events)

    def get_n_events(self):
        """
        Retrieves the number of events stored in the factory object

        Returns:
            n_events (int): The number of events.
        """

        # return length of list of events stored in the factory object
        return len(self.get_events())

    def serialize(self):
        """
        Serializes the event factory and its events.

        Returns:
            serialized_data: The serialized representation of the event factory.
        """

        # create an empty list of pickles for all the events stored in the factory object
        event_pkls = []

        # for each event in the factory, serialise the event and append to the pickle list
        for event in self.__events:
            event_pkls.append(event.serialize())

        # create a dictionary containing all the events stored in the factory object
        data = {"__events": event_pkls}

        # return serialized data
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        """
        Deserializes the pickle-formatted data and initializes the event factory.

        Parameters:
            data_pkl: The pickle-formatted data to be deserialized.
        """

        # load data from the pickle input
        data = pickle.loads(data_pkl)

        # check that data a dictionary of serialised data from the serialise function
        if "__events" in data.keys():
            for event_pkl in data["__events"]:
                event = radiominimalysis.framework.revent.REvent(0, 0)
                event.deserialize(event_pkl)
                self.add_event(event)

    def save_events_to_file(self, fname):
        t = time.time()
        fname += ".pickle" if (fname[-7:] != ".pickle") else ""
        print("Store {} event(s) in {}".format(self.get_n_events(), fname))
        with open(fname, "wb") as fout:
            factory_plk = self.serialize()
            fout.write(factory_plk)
        print(
            "total time used to write events to file: {}".format(
                timedelta(seconds=time.time() - t)
            )
        )

    def read_events_from_file(self, fname):
        try:
            t = time.time()
            with open(fname, "rb") as fopen:
                factory_pkl = fopen.read()
                self.deserialize(factory_pkl)
            dt = timedelta(seconds=time.time() - t)
            print(
                "total time used to read {} events from file: {}".format(
                    len(self.__events), dt
                )
            )
        except AttributeError:
            print("Uses old style factory")
            event_factory = pickle.load(open(fname, "rb"))
            self.__events = event_factory.__dict__["_internal_event_list"]

    def get_parameter(self, parameter, shower_type=None, dtype=None):
        return get_parameter(self.get_events(), parameter, shower_type, dtype)

    def get_parameter_error(self, parameter, shower_type=None, dtype=None):
        return get_parameter_error(self.get_events(), parameter, shower_type, dtype)

    def has_parameter(self, parameter, shower_type=None, dtype=None):
        return has_parameter(self.get_events(), parameter, shower_type, dtype)

    def has_parameter_error(self, parameter, shower_type=None, dtype=None):
        return has_parameter_error(self.get_events(), parameter, shower_type, dtype)

    def __eq__(self, other):
        evs1 = self.get_events()
        evs2 = other.get_events()

        if not len(evs1) == len(evs2):
            return False

        for ev1, ev2 in zip(evs1, evs2):
            if not ev1 == ev2:
                return False

        return True


def get_parameter(events, parameter, shower_type=None, dtype=None):
    if isinstance(parameter, eventParameters):
        return np.asarray([ev.get_parameter(parameter) for ev in events], dtype=dtype)
    elif isinstance(parameter, showerParameters):
        # return np.asarray([ev.get_shower(shower_type).get_parameter(parameter) if ev.has_shower(shower_type) else None for ev in events], dtype=dtype)
        return np.asarray(
            [ev.get_shower(shower_type).get_parameter(parameter) for ev in events],
            dtype=dtype,
        )
    elif isinstance(parameter, stationParameters):
        dtype = object or dtype  # for nested arrays the dtype should be object
        return np.asarray(
            [ev.get_station_parameter(parameter) for ev in events], dtype=dtype
        )
    elif callable(parameter):
        print(parameter)
        np.asarray([parameter(ev) for ev in events], dtype=dtype)

    else:
        raise ValueError("{} is not a supported parameter type.".format(parameter))


def get_parameter_error(events, parameter, shower_type=None, dtype=None):
    if isinstance(parameter, eventParameters):
        return np.asarray(
            [ev.get_parameter_error(parameter) for ev in events], dtype=dtype
        )
    elif isinstance(parameter, showerParameters):
        return np.asarray(
            [
                ev.get_shower(shower_type).get_parameter_error(parameter)
                if ev.has_shower(shower_type)
                else None
                for ev in events
            ],
            dtype=dtype,
        )
    elif isinstance(parameter, stationParameters):
        dtype = object or dtype  # for nested arrays the dtype should be object
        return np.asarray(
            [ev.get_station_parameter_error(parameter) for ev in events], dtype=dtype
        )
    else:
        raise ValueError("{} is not a supported parameter type.".format(parameter))


def get_ids(events):
        return np.array([[ev.get_id(), ev.get_run_number()] for ev in events])

def get_parameter_and_error(*args):
    val = get_parameter(*args)
    err = get_parameter_error(*args)
    return val, err


def has_parameter(events, parameter, shower_type=None, dtype=None):
    if isinstance(parameter, eventParameters):
        return np.asarray([ev.has_parameter(parameter) for ev in events], dtype=dtype)
    elif isinstance(parameter, showerParameters):
        # return np.asarray([ev.get_shower(shower_type).has_parameter(parameter) if ev.has_shower(shower_type) else None for ev in events], dtype=dtype)
        return np.asarray(
            [
                ev.get_shower(shower_type).has_parameter(parameter)
                if ev.has_shower(shower_type)
                else False
                for ev in events
            ],
            dtype=dtype,
        )
    # elif isinstance(parameter, stationParameters):
    #     return np.asarray([ev.get_station_parameter(parameter) for ev in events], dtype=dtype)
    elif callable(parameter):
        print(parameter)
        np.asarray([parameter(ev) for ev in events], dtype=dtype)

    else:
        raise ValueError("{} is not a supported parameter type.".format(parameter))


def has_parameter_error(events, parameter, shower_type=None, dtype=None):
    if isinstance(parameter, eventParameters):
        return np.asarray(
            [ev.has_parameter_error(parameter) for ev in events], dtype=dtype
        )
    elif isinstance(parameter, showerParameters):
        # return np.asarray([ev.get_shower(shower_type).has_parameter_error(parameter) if ev.has_shower(shower_type) else None for ev in events], dtype=dtype)
        return np.asarray(
            [
                ev.get_shower(shower_type).has_parameter_error(parameter)
                for ev in events
            ],
            dtype=dtype,
        )
    # elif isinstance(parameter, stationParameters):
    #     return np.asarray([ev.get_station_parameter(parameter) for ev in events], dtype=dtype)
    elif callable(parameter):
        print(parameter)
        np.asarray([parameter(ev) for ev in events], dtype=dtype)

    else:
        raise ValueError("{} is not a supported parameter type.".format(parameter))


def set_parameter(events, parameter, values, shower_type, dtype=None):
    if isinstance(parameter, eventParameters):
        [ev.set_parameter(parameter, value) for ev, value in zip(events, values)]
    elif isinstance(parameter, showerParameters):
        [
            ev.get_shower(shower_type).set_parameter(parameter, value)
            for ev, value in zip(events, values)
        ]
    else:
        raise ValueError("{} is not a supported parameter type.".format(parameter))


def set_parameter_error(events, parameter, values, shower_type, dtype=None):
    if isinstance(parameter, eventParameters):
        [ev.set_parameter_error(parameter, value) for ev, value in zip(events, values)]
    elif isinstance(parameter, showerParameters):
        [
            ev.get_shower(shower_type).set_parameter_error(parameter, value)
            for ev, value in zip(events, values)
        ]
    else:
        raise ValueError("{} is not a supported parameter type.".format(parameter))
