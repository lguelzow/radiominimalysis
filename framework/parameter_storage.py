from __future__ import absolute_import, division, print_function
import math

import numpy as np
import sys
import logging
import collections
from RadioAnalysis.framework.parameters import stationParameters
logger = logging.getLogger('ParameterStorage')

class ParameterStorage(object):
    '''
    - Dictionary-like storage class
    
    - Lets you store and access parameters using keys



    '''
    
    # 
    def __init__(self, key_type):
        '''
        - initialisation function

        - specifies the key type to be used to access parameters and their errors, e.g. int, string or float, etc.

        - _parameters is a dictionary which stores the parameter values

        - _parameter_errors stores the corresponding errors
        '''

        self._key_type = key_type
        self._parameters = {}
        self._parameter_errors = {}


    def get_parameter(self, key):
        '''
        Function takes a key and returns the parameter value
        '''

        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
        
        return self._parameters[key]


    def set_parameter(self, key, value):
        '''
        Function takes a key and a value and sets the parameter value to it
        '''
        
        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
        
        # set the parameter value
        self._parameters[key] = value


    def has_parameter(self, key):
        '''
        Function takes a key and checks if it exists in the dictionary. Returns boolean value
        '''
        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
        
        # return True if key exists, False if it does not
        return key in self._parameters


    def get_parameter_error(self, key):
        '''
        Function takes a key and returns the error value of the corresponding parameter
        '''

        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
        
        return self._parameter_errors[key]


    def set_parameter_error(self, key, value):
        '''
        Function takes a key and an error value and sets error value of the corresponding parameter to it
        '''

        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
        
        # set the parameter error value
        self._parameter_errors[key] = value


    def has_parameter_error(self, key):
        '''
        Function takes a key and checks if the parameter error exists in the dictionary. Returns boolean value
        '''

        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
        
        # return True if error exists, False if it does not
        return key in self._parameter_errors


    def get_parameter_and_error(self, key):
        '''
        Function takes a key and returns parameter with corresponding error as a tuple
        '''

        # raise error if the key type is not matching
        if not isinstance(key, self._key_type):
            logger.error("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))
            raise ValueError("parameter error key \"{}\" needs to be of type \"{}\"".format(key, self._key_type))

        return self.get_parameter(key), self.get_parameter_error(key)


    def __eq__(self, other):
        '''
        - Function takes another instance of ParameterStorage

        - Compares the this and other instance (see compare_parameter_storage)

        - Returns boolean value
        '''

        # compares key types of the instances 
        # print this instance's key type and return False if they do not match
        if not self._key_type == other._key_type:
            print("key_type")
            return False

        return compare_parameter_storage(self, other, self._key_type, False)

    
    def add_parameter_dict(self, dick, dick_err=None):
        '''
        - Function takes dictionary of parameter values (and errors)

        - Sets current dictionary to those values
        '''

        self._parameters = dick
        if dick_err is not None:
            self._parameter_errors = dick_err



def compare_parameter_storage(storage1, storage2, parameters, report_failues=False):
    """
    Compare two instances of a ParameterStorage object by comparing the values of specific parameters.

    Parameters:
    storage1 (ParameterStorage): The first parameter storage object to compare.
    storage2 (ParameterStorage): The second parameter storage object to compare.
    parameters (list): A list of parameter keys to compare between the two objects.
    report_failues (bool, optional): If True, print out the name of any parameters that do not match. Defaults to False.

    Returns:
    bool: True if all specified parameters match between the two objects, False otherwise.
    """
    
    # Initialize the flag variable indicating whether all parameters match.
    same = True
    
    # Iterate over each parameter in the given parameter key list
    for parameter in parameters:
        
        # Try to compare the parameter values between the two objects.
        try:
            # Compare the existence of the parameter between the two objects.
            if not storage1.has_parameter(parameter) == storage2.has_parameter(parameter):
                if report_failues:
                    # print parameter name that doesn't match
                    print("Does not exist in both objects: ", parameter)
                    same = False
                else:
                    return False

            # Compare the errors of the parameter between the two objects.
            if not storage1.has_parameter_error(parameter) == storage2.has_parameter_error(parameter):
                if report_failues:
                    # print parameter name of which the error doesn't match
                    print("Does not exist in both objects: ", parameter)
                    same = False
                else:
                    return False

            # If the parameter exists in both objects, compare its value.
            if storage1.has_parameter(parameter) and storage2.has_parameter(parameter):

                # Get the parameter values from each object.
                parameter1 = storage1.get_parameter(parameter)
                parameter2 = storage2.get_parameter(parameter)

                # If the parameter is a string or bytes object, compare directly.
                if isinstance(parameter1, (str, bytes)):
                    if not parameter1 == parameter2:
                        if report_failues:
                            # print parameter name of which the values don't match
                            print("Values are different: ", parameter)
                            same = False
                        else:
                            return False
                
                # If parameter is a stationParameter, compare each element of the list.
                elif parameter == stationParameters.name:
                    if len(parameter1) == len(parameter2):
                        for value1, value2 in zip(parameter1, parameter2):
                            if value1 != value2:
                                if report_failues:
                                    # print parameter name of which the values don't match
                                    print("Values are different: ", parameter)
                                    same = False
                                else:
                                    return False
                    else:
                        if report_failues:
                            # print parameter name of which the array lengths don't match
                            print("Parameter array lengths are different: ", parameter)
                            same = False
                        else:
                            return False

                # If the parameter is a dictionary or ordered dictionary, compare recursively.
                elif isinstance(parameter1, (collections.OrderedDict, dict)):
                    match = len(parameter1) == len(parameter2)
                    if match:
                        for key in parameter1:
                            match = key in parameter2
                            if match:
                                match = parameter1[key] == parameter2[key]

                    if not match:
                        if report_failues:
                            print("get_parameter", parameter)
                            same = False
                        else:
                            return False
                
                # Otherwise, compare using numpy.allclose
                else:
                    match = True
                    if isinstance(parameter1, np.ndarray):
                        match = parameter1.shape == parameter2.shape

                    if match:
                        match = np.allclose(parameter1, parameter2)

                    if not match:
                        if report_failues:
                            print("get_parameter", parameter)
                            same = False
                        else:
                            return False
            
            # If the parameter has a error values in both objects, compare the errors
            if storage1.has_parameter_error(parameter) and storage2.has_parameter_error(parameter):
                pstr = "Doesn't match: "
                parameter1 = storage1.get_parameter_error(parameter)
                parameter2 = storage2.get_parameter_error(parameter)
                
                if isinstance(parameter1, (str, bytes)):
                    if not parameter1 == parameter2:
                        if report_failues:
                            print(pstr, parameter)
                            same = False
                        else:
                            return False

                elif parameter == stationParameters.name:
                    if len(parameter1) == len(parameter2):
                        for value1, value2 in zip(parameter1, parameter2):
                            if value1 != value2:
                                if report_failues:
                                    print(pstr, parameter)
                                    same = False
                                else:
                                    return False
                    else:
                        if report_failues:
                            print(pstr, parameter)
                            same = False
                        else:
                            return False

                elif isinstance(parameter1, (collections.OrderedDict, dict)):
                    match = len(parameter1) == len(parameter2)
                    if match:
                        for key in parameter1:
                            match = key in parameter2
                            if match:
                                match = parameter1[key] == parameter2[key]
                    
                    if not match:
                        if report_failues:
                            print(pstr, parameter)
                            same = False
                        else:
                            return False

                else:
                    match = True
                    if isinstance(parameter1, np.ndarray):
                        match = parameter1.shape == parameter2.shape

                    if match:
                        match = np.allclose(parameter1, parameter2)

                    if not match:
                        if report_failues:
                            print(pstr, parameter)
                            same = False
                        else:
                            return False
        
        except TypeError as e:
            print(parameter, parameter1, type(parameter1))
            print(parameter, parameter2, type(parameter2))
            sys.exit(e)
    
    # return whether the objects are the same or not
    return same
