from radiotools.analyses import radiationenergy
from radiotools.atmosphere import models as atm

import numpy as np

"""
POS(ICRC2019)294:
    - p0 = 3.94278610e-01
    - p1 = -2.37010471e+00
    - rho_avg = 0.6481835289520177 (= radiationenergy.get_average_density())
    - s19 = 1.40839405e+09 eV
    - gamma = 1.99469492e+00

POS(ICRC2021)209:
    - p0 = 0.497
    - p1 = -2.737
    - rho_avg = 0.3 kg/m3
    - s19 = 3.15e+09 eV
    - gamma = 2.000

"""

kp0 = 0.497
kp1 = -2.737
krho_avg = 0.3
ks19 = 3.15e+09
kgamma = 2.000

def get_expected_S(Eem):
    # get the expected (density and geometry corrected) radio energy estimator
    # parameter from jcap paper
    return 1.683 * 1e7 * (Eem / 1e18) ** (2.006)


def get_expected_Erad(Eem, sinalpha, density):
    return radiationenergy.get_radiation_energy(get_expected_S(Eem), sinalpha=sinalpha, density=density)


def get_Eem_from_Erad(Erad, sinalpha, density,  A=1.683e7, B=2.006, p0=0.250524463912, p1=-2.95290494):
    return (radiationenergy.get_S(Erad, sinalpha, density, p0=p0, p1=p1) / A ) ** (1 / B) * 1e18  # jcap


def get_Egeo(eem, sinalpha, density, s19=13.4858593e9, gamma=1.9961499, 
             p0=196.5323011, p1=-0.0030236, p2=0.033889, p3=0.1928640, a_exp=11.1631093, rho_avg=0.2171571758825658, ref_energy=1e19):
    """ get geomagnetic radiation energy from electromagnetic shower energy

    Parameters
    ----------
    eem : float
        electromagnetic shower energy (in eV)
    sinalpha: float
        sine of angle between shower axis and geomagnetic field
    density : float
        density at shower maximum in kg/m^3

    Returns
    --------
    float:
        radiation energy (in eV)
    """
    sgeo = s19 * (eem / ref_energy) ** gamma
    return sgeo * sinalpha ** a_exp * (1 - p0 + p0 * np.exp(p1 * (density - rho_avg)) - p2 / density + p3) ** 2


# def get_Egeo_fit(xdata, para):
#     Eem, sinalpha, density = xdata
#     return get_Egeo(Eem, sinalpha, density, *para)


def get_Sgeo(egeo, sinalpha, density, p0=kp0, p1=kp1, p2=0.05, p3=0.1, a_exp=1.38, rho_avg=krho_avg, egeo_err=None):
    """ get corrected radiation energy (S_RD) for geomagnetic radiation energy

    Parameters
    ----------
    egeo : float
        geomagnetic radiation energy (in eV)
    sinalpha: float
        sine of angle between shower axis and geomagnetic field
    density : float
        density at shower maximum in kg/m^3

    egeo_err : float (optinal)
        uncertainty on geomagnetic radiation energy (in eV)

    Returns
    --------
    float:
        corrected geomagnetic radiation energy (in eV)
    float (optinal, if egeo_err is given):
        uncertainty on the corrected geomagnetic radiation energy (in eV)
    """
    sgeo = egeo / sinalpha ** a_exp / \
        (1 - p0 + p0 * np.exp(p1 * (density - rho_avg)) - p2 / density + p3) ** 2

    if egeo_err is None:
        return sgeo
    else:
        sgeo_err = egeo_err * sgeo / egeo
        return sgeo, sgeo_err


def get_Eem(sgeo, s19=ks19, gamma=kgamma, ref_energy=1e19):
    """ get electron magnetic shower energy from corrected geomagnetic radiation energy

    Parameters
    ----------
    sgeo : float
        corrected geomagnetic radiation energy (in eV)
    s19: float
        normalisation, i.e., radiation energy per 10 EeV electromagnet shower energy (in eV)
    gamma : float
        exponent of power law

    Returns
    -------

    float:
        corrected geomagnetic radiation energy (in eV)
    """
    return (sgeo / s19) ** (1 / gamma) * ref_energy


def get_Eem_from_Egeo(egeo, sinalpha, density, s19=ks19, gamma=kgamma,
                      p0=kp0, p1=kp1, p2=0.05, p3=0.1, a_exp=1.38, rho_avg=krho_avg, egeo_err=None, ref_energy=1e19):
    """ get electron magnetic shower energy from (uncorrected) geomagnetic radiation energy

    Parameters
    ----------
    egeo : float
        geomagnetic radiation energy (in eV)
    sinalpha: float
        sine of angle between shower axis and geomagnetic field
    density : float
        density at shower maximum in kg/m^3
    s19: float
        normalisation, i.e., radiation energy per 10 EeV electromagnet shower energy (in eV)
    gamma : float
        exponent of power law

    egeo_err : float (optinal)
        uncertainty on geomagnetic radiation energy (in eV)

    Returns
    -------

    float:
        electromagnetic shower radiation energy (in eV)
    float (optinal, if egeo_err is given):
        uncertainty electromagnetic shower radiation energy (in eV)
    """

    sgeo = get_Sgeo(egeo, sinalpha, density, p0, p1, p2, p3, a_exp, rho_avg)

    eem = get_Eem(sgeo, s19, gamma, ref_energy=ref_energy)

    if egeo_err is not None:
        sgeo_err = egeo_err / \
            (sinalpha ** a_exp * (1 - p0 + p0 * np.exp(p1 * (density - rho_avg)) - p2 / density + p3) ** 2)  # ignores uncert on sin(alpha) and density

        eem_err = sgeo_err * ref_energy / (sgeo * gamma) * (sgeo / s19) ** (1 / gamma)

        return eem, eem_err
    else:
        return eem


# def fit_Eem_from_Egeo(xdata, para):
#     Egeo, sinalpha, density = xdata
#     p0, p1, S19, gamma = para
#     Srad = get_Sgeo(Egeo, sinalpha, density, p0, p1)
#     return (Srad / S19) ** (1 / gamma) * 10 ** 19


def geomagnetic_density_correction_and_error_ICRC19(rho, erho):
    rho_avg = 0.6481835289520177
    p0 = 3.94278610e-01
    p1 = -2.37010471e+00
    return geomagnetic_density_correction_and_error(rho, erho, p0, p1, rho_avg)


def geomagnetic_density_correction_and_error_ICRC21(rho, erho):
    rho_avg = krho_avg
    p0 = kp0
    p1 = kp1
    return geomagnetic_density_correction_and_error(rho, erho, p0, p1, rho_avg)


def geomagnetic_density_correction_and_error(rho, erho, p0, p1, rho_avg):

    efunc = np.exp(p1 * (rho - rho_avg))
    densityCorrection = 1 / (1 - p0 + p0 * efunc) ** 2
    edensityCorrection = np.abs(-2 * p0 * p1 * efunc * erho) \
        / (1 - p0 + p0 * efunc) ** 3

    if rho_avg == 0:
        norm_fact = np.mean(densityCorrection)
        densityCorrection /= norm_fact
        edensityCorrection /= norm_fact

    return densityCorrection, edensityCorrection


def uncertainty_in_density_from_zenith_angle_uncertainty(
        zenith, zenith_err, dmax, rho,
        observation_level=1400, model=27):

    rho_down = atm.get_density_for_distance(
        dmax, zenith - zenith_err, observation_level=observation_level, model=model) * 1e-3
    rho_up = atm.get_density_for_distance(
        max, zenith + zenith_err, observation_level=observation_level, model=model) * 1e-3

    rho_ucert = [rho_up - rho, rho - rho_down]

    return rho_ucert


def uncertainty_in_density_from_zenith_angle_uncertainty_event(events):
    import radiominimalysis.framework.revent
    import warnings
    from radiominimalysis.framework.parameters import showerParameters as shp, \
        eventParameters as evp, stationParameters as stp

    if isinstance(events, radiominimalysis.framework.revent):
        events = [events]

    rho_uncerts = np.empty((len(events), 2))

    for edx, revent in enumerate(events):
        shower = revent.get_shower(evp.rd_shower)

        zenith, zenith_err = shower.get_parameter_and_error(shp.zenith)

        dmax = shower.get_parameter(
            shp.distance_to_shower_maximum_geometric_fit)

        rho = shower.get_parameter(
            shp.density_at_shower_maximum)

        if shower.has_parameter(shp.observation_level):
            obs_lvl = shower.get_parameter(shp.observation_level)
        else:
            warnings.warn("Use hardcoded observation level of 1400m!!!")
            obs_lvl = 1400
        
        if shower.has_parameter(shp.atmosphere_model):
            atm_model = shower.get_parameter(shp.atmosphere_model)
        else:
            warnings.warn("Use hardcoded atm model 27!!!")
            atm_model = 27

        rho_uncerts[edx] = uncertainty_in_density_from_zenith_angle_uncertainty(
            zenith, zenith_err, dmax, rho,
            observation_level=obs_lvl, model=atm_model)

    return np.squeeze(rho_uncerts)
  
