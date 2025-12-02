import numpy as np
import warnings

from radiotools.atmosphere import models as atm

from radiominimalysis.framework.parameters import showerParameters as shp, \
    eventParameters as evp, stationParameters as stp


def get_cherenkov_radius_model_from_depth(zenith, depth, obs_level, n0, model=None, at=None):
    """ Calculates the radius of the (Cherenkov) cone with an apex at a given depth along a 
        shower axis with a given zenith angle. The open angle of the cone equals the 
        Cherenkov angle for a value of the refractive index at this position.

    Paramter:

    zenith : double
        Zenith angle (in radian) under which a shower is observed 
    
    depth : double
        Slant depth (in g/cm^2), i.e., shower maximum of the observed shower
    
    obs_level : double
        Altitude (in meter) of the plane at which the shower is observed 

    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model. Needed when no "at" is given

    at : radiotools.atmosphere.models.Atmosphere
        Atmospheric (density) profile model. Provides the density profile of the atmosphere in the typical 5-layer param.

    Return : cherenkov Radius

    """
    if at is None:
        at = atm.Atmosphere(model=model)

    d = at.get_distance_xmax_geometric(zenith, depth, obs_level)
    return get_cherenkov_radius_model_from_distance(zenith, d, obs_level, n0, at.model)


def get_cherenkov_radius_model_from_height(zenith, height, obs_level, n0, model):
    """ Calculates the radius of the (Cherenkov) cone with an apex at a given height above sea level on a
        shower axis with a given zenith angle. The open angle of the cone equals the 
        Cherenkov angle for a value of the refractive index at this position.

    Paramter:

    zenith : double
        Zenith angle (in radian) under which a shower is observed 
    
    height : double
        Height above sea level (in m) of the apex, i.e., shower maximum of the observed shower
    
    obs_level : double
        Altitude (in meter) of the plane at which the shower is observed 

    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model.

    Return : cherenkov Radius

    """

    angle = get_cherenkov_angle_model(height, n0, model)
    dmax = atm.get_distance_for_height_above_ground(
        height - obs_level, zenith, observation_level=obs_level)
    return cherenkov_radius(angle, dmax)


def get_cherenkov_radius_model_from_distance(zenith, d, obs_level, n0, model):
    """ Calculates the radius of the (Cherenkov) cone with an apex at a given distance from ground 
        along the shower axis with a given zenith angle. The open angle of the cone equals the 
        Cherenkov angle for a value of the refractive index at this position.

    Paramter:

    zenith : double
        Zenith angle (in radian) under which a shower is observed 
    
    d : double
        Distance from ground to the apex, i.e., shower maximum of the observed shower along the shower axis (in m)
    
    obs_level : double
        Altitude (in meter) of the plane at which the shower is observed 

    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model.

    Return : cherenkov Radius

    """
    height = atm.get_height_above_ground(
        d, zenith, observation_level=obs_level) + obs_level
    angle = get_cherenkov_angle_model(height, n0, model)
    return cherenkov_radius(angle, d)


def get_cherenkov_angle_model(height, n0, model):
    """ Return cherenkov angle for given height above sea level, 
        refractive index at sea level and atmospheric model. 

    Paramter:

    height : double
        Height above sea level (in m)
    
    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model.

    Return : cherenkov angle

    """
    n = atm.get_n(height, n0=n0, model=model)
    return cherenkov_angle_model(n)


def cherenkov_angle_model(n):
    """ Return cherenkov angle for given refractive index.

    Paramter:

    n : double
        Refractive index

    Return : cherenkov angle

    """
    return np.arccos(1 / n)


def cherenkov_radius(angle, d):
    """ Return (cherenkov) radius

    Paramters
    ---------

    angle : double
        (Opening) angle of the cone (in rad)

    d : double
        Heigth of the cone (typically called distance, in meter)

    Returns
    -------
    
    radius : double
        (Cherenkov) radius

    """
    return np.tan(angle) * d


# old: cherenkov_angle_from_density_refractivity(rho, dxmax, n_asl, rho_0, ...)
# param of the cherenkov angle from star-shape simulations
# here cherenkov radius refers to radius of strongest emission
# is used to determine rmax in the RdIdealGrid simulations!
def cherenkov_angle_param(
        height, dist, n0, model,
        a=9.48990456e-01, b=4.48698860e+03,
        c=1.43097665e+00, d=2.46630811e+06):

    n = atm.get_n(height, n0=n0, model=model)
    A = a - (b / dist) ** (c) - dist / d

    return A * cherenkov_angle_model(n)
# # old param
# def cherenkov_angle_from_density(x, A=0.24905864, B=0.92165234):
#     return np.deg2rad(A * np.log(x) + B)


# old: def get_param_cherenkov_radius_from_density(revent, n_asl=n_asl_ref, key=None):
def get_cherenkov_radius_param_revent(revent, key=None):
    shower = revent.get_shower(key)

    dmax = shower.get_parameter(shp.distance_to_shower_maximum_geometric)
    obs_lvl = shower.get_parameter(shp.observation_level)
    zenith = shower.get_parameter(shp.zenith)
    height = atm.get_height_above_ground(dmax, zenith, obs_lvl) + obs_lvl

    n0 = revent.get_parameter(evp.refractive_index_at_sea_level)
    model = shower.get_parameter(shp.atmosphere_model)
    angle = cherenkov_angle_param(height, dmax, n0, model)

    return cherenkov_radius(angle, dmax)


def get_cherenkov_radius_model_revent(revent, key=None):
    shower = revent.get_shower(key)

    dmax = shower.get_parameter(shp.distance_to_shower_maximum_geometric)
    obs_lvl = shower.get_parameter(shp.observation_level)
    zenith = shower.get_parameter(shp.zenith)
    height = atm.get_height_above_ground(dmax, zenith, obs_lvl) + obs_lvl

    n0 = revent.get_parameter(evp.refractive_index_at_sea_level)
    model = shower.get_parameter(shp.atmosphere_model)
    angle = get_cherenkov_angle_model(height, n0, model)

    return cherenkov_radius(angle, dmax)


# old: get_param_cherenkov_radius
def get_cherenkov_radius_param_from_depth(
        zenith, depth, obs_level, n0, model=None, at=None):

    if at is None:
        at = atm.Atmosphere(model)

    height = at.get_vertical_height(zenith, depth, obs_level)
    d = atm.get_distance_for_height_above_ground(
        height - obs_level, zenith, obs_level)
    angle = cherenkov_angle_param(height, d, n0, at.model)
    return cherenkov_radius(angle, d)


# old: def get_cherenkov_radius_from_distance(zenith, dmax, observation_level=1400, model=27):
def get_cherenkov_radius_param_from_distance(
        zenith, dist, obs_level, n0, model):
    h_asl = atm.get_height_above_ground(
        d=dist, zenith=zenith, observation_level=obs_level) + obs_level

    angle = cherenkov_angle_param(h_asl, dist, n0, model)
    return cherenkov_radius(angle, dist)

