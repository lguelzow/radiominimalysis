from __future__ import absolute_import, division, print_function

# Import the `cPickle` module if it is available (faster version of the `pickle` module)
# If `cPickle` is not available, fall back to using `pickle`
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys

# Import modules from the `radiominimalysis` package
from radiominimalysis.framework import parameter_serialization, parameters
from radiominimalysis.framework.parameter_storage import ParameterStorage

import logging
logger = logging.getLogger('Shower')

class Shower(ParameterStorage):
    """
    Inherits from `ParameterStorage` and provides additional functionality
    for storing and retrieving parameters of an air shower.

    Parameters
    ----------
    key : any, optional
        The key to use for storing and retrieving the parameters. 
        Initialises ParameterStorage with the parameters.showerParameters key type (Enum)
    """

    def __init__(self, key=None):
        """
        Initialize the `Shower` object.
        """

        self.__shower_type = key

        # Call the 'ParameterStorage' constructor with the showerParameters Enum key type
        super().__init__(key_type=parameters.showerParameters)

    def get_shower_type(self):
        '''
        Returns shower type
        '''
        return self.__shower_type

    def serialize(self):
        data = {'_parameters': parameter_serialization.serialize(self._parameters),
                '_parameter_errors': parameter_serialization.serialize(self._parameter_errors),
                '__shower_type': self.__shower_type}

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self._parameters = parameter_serialization.deserialize(data['_parameters'], parameters.showerParameters)
        if '_parameter_errors' in data:
            self._parameter_errors = parameter_serialization.deserialize(data['_parameter_errors'], parameters.showerParameters)

        if '__shower_type' in data:
            self.__shower_type = data['__shower_type']
