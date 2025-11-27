from __future__ import absolute_import, division, print_function, unicode_literals

def serialize(object):
    """
    Serializes a dictionary object by converting the key to a string 
    and adding the key-value pair to a new dictionary. 
    
    Returns:
         The resulting dictionary.

    """

    reply = {}
    for entry in object:
        reply[str(entry)] = object[entry]
    return reply

def deserialize(object, parameter_enum):
    """
    Deserializes a dictionary object by converting its keys from strings to the corresponding
    enum values in parameter_enum.

    Args:
        object (dict): A dictionary object to be deserialized.
        parameter_enum (enum.Enum): An enum representing the parameter names.

    Returns:
        dict: A deserialized dictionary object with enum keys.
    """
    
    reply = {}
    for entry in parameter_enum:
        if str(entry) in object:
            reply[entry] = object[str(entry)]
    return reply
