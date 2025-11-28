from radiominimalysis.framework.parameters import stationParameters as stp, showerParameters as shp
from radiominimalysis.utilities import charge_excess as ce, geomagnetic_emission, cherenkov_radius as che

from radiominimalysis.utilities import helpers
from radiotools import helper as rdhelp

import sys
import numpy as np
from scipy import interpolate
import warnings
import functools

try:
    from interpolation_Fourier import interp2d_fourier
    imported_fourier = True
except ImportError as e:
    print(e)
    imported_fourier = False


def find_maximum_in_footprints(events, para):
    

    for idx, revent in enumerate(events):
        if idx % 100 == 0:
            print(idx)

        r0 = che.get_cherenkov_radius_model_revent(revent)

        pvxB = revent.get_station_position_vB_vvB()
        pel = revent.get_station_position_early_late()
        fvxB = revent.get_station_parameter(stp.energy_fluence)[:, 0]
        c_el = revent.get_station_parameter(stp.early_late_factor)
        if 1:
            x, y, f = find_maximum(
                pvxB / c_el[:, None], fvxB * c_el ** 2, 2 * r0)
        else:
            x, y, f = find_maximum(pvxB, fvxB, 2 * r0)

        
        revent.get_shower().set_parameter(shp.maximum_energy_fluence_interpolated_2d, f)
        revent.get_shower().set_parameter(shp.position_maximum_energy_fluence_interpolated_2d, np.array([x, y]))


def find_maximum(positions, signal, rmax, n_interp_grid=200):

    if not imported_fourier:
        sys.exit("Fourier interpolation not imported")

    positions = np.around(positions, 1)
    interp_func = interp2d_fourier(
        positions[:, 0], positions[:, 1], signal)

    # define positions where to interpolate

    xs = np.linspace(-rmax, rmax, n_interp_grid)
    ys = np.linspace(-rmax, rmax, n_interp_grid)


    xx, yy = np.meshgrid(xs, ys)

    # points within a circle
    in_star = xx ** 2 + \
        yy ** 2 <= np.amax(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    signal_interp = np.where(in_star, interp_func(xx, yy), 1)

    xmax, ymax = np.where(signal_interp == signal_interp.max())

    return xx[0, xmax], yy[ymax, 0], signal_interp.max()



def get_parametric_charge_excess_fraction(revent, param=None):
    
    # print("using right function for fit")

    c_early_late = revent.get_station_parameter(stp.early_late_factor)
    shower = revent.get_shower()

    r_stations = revent.get_station_axis_distance()
    c_early_late = revent.get_station_parameter(stp.early_late_factor)

    dxmaxs_geo = shower.get_parameter(
        shp.distance_to_shower_maximum_geometric)
    density = shower.get_parameter(shp.density_at_shower_maximum)

    r_stations = r_stations / c_early_late
    '''
    if 0:
        # POS(ICRC2019)294
        # this parametrization is done without ce pos on vxB
        param = [0.37313183, 1.31124484, -0.1889835, 6.71403002]

        if shower.get_parameter(shp.zenith) < np.deg2rad(80.1):

            charge_excess_fraction_param = ce.charge_excess_fraction_icrc19(
                [r_stations, dxmaxs_geo, density], *param)
            revent.set_station_parameter(
                stp.charge_excess_fraction_parametric, charge_excess_fraction_param)

        else:
            warnings.warn(
                'Charge-excess parametrization set to 0 beyond 80 degree.')
            charge_excess_fraction_param = np.zeros(c_early_late.shape)
    '''    
    if param == None:
        # new, with core fit, after tuning with has param ldf fit
        # param = [0.3128554141417296, 1.6068519502678418,
        #         1.66965694e+01, 3.31954904e+00, -5.73577715e-03]
        

        # these are the parameters given in Felix' thesis (see calender book (11.01.2023))
        # use these for 30-80 MHz
        # param = [-1.17523609e-06, 3.48154734e-01, 1.6068519502678418, 1.66965694e+01, 3.31954904e+00, -5.73577715e-03]


        # parameters from Lukas new fit for 50-200 MHz for Auger
        # use these for 50-200 MHz
        # param = [-1.37266723e-06, 3.02018018e-01, 1.46508803e+00, 1.31382072e+01, 2.98380964e+00, 1.78471809e-01]


        # parameters from Lukas new fit for 50-200 MHz for china
        # use these for 50-200 MHz for GP300 site
        param = [-9.03992069e-07, 2.28710354e-01, 1.62957071e+00, 1.77729341e+00, 1.42776016e+00, 1.66010236e-01]

        # print("r", r_stations[0:10])
        # print("dmax", dxmaxs_geo)
        # print("Density", density)
        # print("parametrisation", param)

        charge_excess_fraction_param = ce.charge_excess_fraction_icrc21(
            [r_stations, dxmaxs_geo, density], *param)
        
        # set all negative entries to zero
        charge_excess_fraction_param[charge_excess_fraction_param < 0] = 0
        
        revent.set_station_parameter(
            stp.charge_excess_fraction_parametric, charge_excess_fraction_param)
        
    else:
        charge_excess_fraction_param = ce.charge_excess_fraction_icrc21(
            [r_stations, dxmaxs_geo, density], *param)
        
        # set all negative entries to zero
        charge_excess_fraction_param[charge_excess_fraction_param < 0] = 0
        
        revent.set_station_parameter(
            stp.charge_excess_fraction_parametric, charge_excess_fraction_param)


    return charge_excess_fraction_param


def reconstruct_parametric_charge_excess_fraction(events, para):

    for revent in events:
        charge_excess_fraction_param = get_parametric_charge_excess_fraction(revent)
        revent.set_station_parameter(stp.charge_excess_fraction_parametric, charge_excess_fraction_param)


def seperate_radio_emission_with_rotational_symmetry(position_vBvvB, energy_fluence_vector, xx=None, yy=None):
    if not imported_fourier:
        sys.exit("The module interpolation_Fourier was not imported. "
            "You can not use \"seperate_radio_emission_with_rotational_symmetry\"."
            "Check why this is. ")

    interp_func_vB = interp2d_fourier(
        position_vBvvB[:, 0], position_vBvvB[:, 1], energy_fluence_vector[:, 0])

    interp_func_vvB = interp2d_fourier(
        position_vBvvB[:, 0], position_vBvvB[:, 1], energy_fluence_vector[:, 1])

    if xx is None and yy is None:
        xx = position_vBvvB[:, 0]
        yy = position_vBvvB[:, 1]
        
    signal_interp_vB = interp_func_vB(xx, yy)
    signal_interp_vvB = interp_func_vvB(xx, yy)
    signal_interp_vB_flipped = interp_func_vB(-xx, -yy)

    f_geo = 1 / 4. * \
        np.square(np.sqrt(signal_interp_vB) +
                    np.sqrt(signal_interp_vB_flipped))

    f_ce_vxB = 1 / 4. * \
        np.square(np.sqrt(signal_interp_vB) -
                    np.sqrt(signal_interp_vB_flipped))

    f_ce_vxvxB = signal_interp_vvB
    # TODO: Check if that is actually true:
    f_ce = f_ce_vxB + f_ce_vxvxB

    return f_geo, f_ce


# when used in fitting with fixed core the declorater might speed up the process
# @functools.lru_cache(maxsize=16)
def seperate_radio_emission_from_position(
                position_vBvvB, energy_fluence_vector, c_early_late=None,
                recover_vxB=True, set_vxB_to_value=np.nan, get_only_f_geo=False, fitted_core=False, revent=None):
    """
    This function returns the emission from the geomagnetic (f_geo) and charge excess (f_ce) emission for given
    observer. Keep in might that the calculation failes for station on or close th the vxB axis -> ~ 1/0.
    You can set those values to a fixed value with "set_vxB_to_value". You recalculate then with another function with
    recover_vxB = True (is default). Therefore you need the early late correction factor. if they are not given 1 is used.
    to save time you can just compute the geomagnetic component with get_only_f_geo = True.
    """

    # angle between observer and positive vxB axis
    phi = rdhelp.get_normalized_angle(np.arctan2(position_vBvvB[:, 1], position_vBvvB[:, 0]))

    if fitted_core:
        # when fitting core, the stations are close but not on the vxB
        # station close to the vxB axis have to be treating differently
        #arccos(0.99) ~ 8 deg -> 1/sin(phi) ~ 10
        #arccos(0.95) ~ 18 deg

        ## This was used for icrc21
        # vxB_pos = np.cos(phi) > 0.9
        # vxB_neg = np.cos(phi) < -0.9

        vxB_pos = np.cos(phi) > 0.9961946980917455  # cos(5deg)
        vxB_neg = np.cos(phi) < -0.9961946980917455  # cos(5deg)

        vxB_axis = np.any([vxB_pos, vxB_neg], axis=0)
        # print(vxB_axis)
    else:
        vxB_axis = helpers.mask_polar_angle(
            phi, angles_in_deg=[0, 180], atol_in_deg=1)
        vxB_pos = helpers.mask_polar_angle(
            phi, angles_in_deg=0, atol_in_deg=1)
        vxB_neg = helpers.mask_polar_angle(
            phi, angles_in_deg=180, atol_in_deg=1)
            
    if revent is not None:
        print(revent)
        shower = revent.get_shower()
        cs = revent.get_coordinate_transformation()
        pos_vB_orig = cs.transform_to_vxB_vxvxB(revent.get_station_parameter(stp.position),
                core=np.array([0, 0, shower.get_parameter(shp.observation_level)]))
        angles = np.around(np.rad2deg(rdhelp.get_normalized_angle(np.arctan2(pos_vB_orig[:, 1], pos_vB_orig[:, 0]))), 1)
        angles[angles >= 360] -= 360

        print(np.cos(phi)[angles == 45])
        print(np.cos(phi)[angles == 225])

    f_vxB = energy_fluence_vector[:, 0]
    f_vxvxB = energy_fluence_vector[:, 1]

    f_geo_pos = np.full_like(f_vxB, set_vxB_to_value)
    f_geo_pos[~vxB_axis] = np.square(np.sqrt(f_vxB[~vxB_axis]) - (np.cos(phi[~vxB_axis]) / np.abs(np.sin(phi[~vxB_axis]))) * np.sqrt(f_vxvxB[~vxB_axis]))

    if not get_only_f_geo:
        f_ce_pos = np.full_like(f_vxB, set_vxB_to_value)
        f_ce_pos[~vxB_axis] = 1 / np.sin(phi[~vxB_axis]) ** 2 * f_vxvxB[~vxB_axis]

    if recover_vxB:
        if c_early_late is None:
            warnings.warn("Seperating emissions on vxB axis with out early late correction.")
            c_early_late = np.ones(f_geo_pos.shape)

        # early late correction
        r_axis = np.sqrt(position_vBvvB[:, 0] ** 2 + position_vBvvB[:, 1] ** 2) / c_early_late
        f_vxB = f_vxB * c_early_late ** 2

        # seperate positive and negative axis
        r_pos, r_neg = r_axis[vxB_pos], r_axis[vxB_neg]

        # # to avoid that stations getting to close to each other when fitting the core
        # r_pos_u, idx_pos = np.unique(np.around(r_pos, -1), return_index=True)
        # r_neg_u, idx_neg = np.unique(np.around(r_neg, -1), return_index=True)
        # # f_vxB[vxB_pos][idx_pos], f_vxB[vxB_neg][idx_neg]

        # interpolation is needed because of early late correction the true axis distance are not equal
        # if x_new is out of bound (will happen) than f_vxB of the clostest station for the lower bound and 0 for the upper bounds is used
        f_vxB_pos = interpolate.interp1d(r_pos, f_vxB[vxB_pos], bounds_error=False, fill_value=(f_vxB[np.argmin(r_axis)], 0), kind='quadratic')
        f_vxB_neg = interpolate.interp1d(r_neg, f_vxB[vxB_neg], bounds_error=False, fill_value=(f_vxB[np.argmin(r_axis)], 0), kind='quadratic')

        # reverse early late correction (since is not for the other axis yet)
        f_geo_pos[vxB_pos] = ce.f_geo_on_vxB(f_vxB_pos(r_pos), f_vxB_neg(r_pos)) / c_early_late[vxB_pos] ** 2
        f_geo_pos[vxB_neg] = ce.f_geo_on_vxB(f_vxB_pos(r_neg), f_vxB_neg(r_neg)) / c_early_late[vxB_neg] ** 2

        if not get_only_f_geo:
            f_ce_pos[vxB_pos] = ce.f_ce_on_vxB(f_vxB_pos(r_pos), f_vxB_neg(r_pos)) / c_early_late[vxB_pos] ** 2
            f_ce_pos[vxB_neg] = ce.f_ce_on_vxB(f_vxB_pos(r_neg), f_vxB_neg(r_neg)) / c_early_late[vxB_neg] ** 2

        if revent is not None:
            from matplotlib import pyplot as plt
            plt.plot(r_pos, f_vxB[vxB_pos], 'C0o')
            plt.plot(r_pos, f_vxB_neg(r_pos), 'C1s')
            plt.plot(r_neg, f_vxB[vxB_neg], 'C1o')
            plt.plot(r_neg, f_vxB_pos(r_neg), 'C0s')
            # plt.show()


    if not get_only_f_geo:
        return f_geo_pos, f_ce_pos
    else:
        return f_geo_pos


def reconstruct_geomagnetic_and_charge_excess_emission_from_position(events, para):

    for revent in events:
        shower = revent.get_shower()

        position_vBvvB = revent.get_station_position_vB_vvB()
        c_early_late = revent.get_station_parameter(stp.early_late_factor)
        energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)

        f_geo_pos, f_ce_pos = seperate_radio_emission_from_position(position_vBvvB, energy_fluence_vector,
                                                                    c_early_late=c_early_late, recover_vxB=False,
                                                                    set_vxB_to_value=-1, fitted_core=True)
        #temp
        # f_geo_pos[np.isnan(f_geo_pos)] = 0
        # f_ce_pos[np.isnan(f_ce_pos)] = 0
        if np.any(np.isnan(f_geo_pos)) or np.any(np.isnan(f_ce_pos)):
            print(f_geo_pos)
            print(f_ce_pos)
            raise ValueError("Invalid results for the positional emission fluences")

        revent.set_station_parameter(stp.geomagnetic_fluence_positional, f_geo_pos)
        revent.set_station_parameter(stp.charge_excess_fluence_positional, f_ce_pos)


# same function, but without loop
def reconstruct_geomagnetic_and_charge_excess_emission_from_position_revent(revent, para):

    shower = revent.get_shower()

    position_vBvvB = revent.get_station_position_vB_vvB()
    c_early_late = revent.get_station_parameter(stp.early_late_factor)
    energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)

    f_geo_pos, f_ce_pos = seperate_radio_emission_from_position(position_vBvvB, energy_fluence_vector,
                                                                    c_early_late=c_early_late, recover_vxB=False,
                                                                    set_vxB_to_value=-1, fitted_core=True)
    #temp
    # f_geo_pos[np.isnan(f_geo_pos)] = 0
    # f_ce_pos[np.isnan(f_ce_pos)] = 0
    if np.any(np.isnan(f_geo_pos)) or np.any(np.isnan(f_ce_pos)):
        print(f_geo_pos)
        print(f_ce_pos)
        raise ValueError("Invalid results for the positional emission fluences")

    revent.set_station_parameter(stp.geomagnetic_fluence_positional, f_geo_pos)
    revent.set_station_parameter(stp.charge_excess_fluence_positional, f_ce_pos)


def get_emission_from_param(revent):
    """
    reconstructs the energy fluence from the geomagnetic emission using a parametrization of the charge excess
    fraction. It is just well define with early late corrected axis distances
    """
    shower = revent.get_shower()

    alpha = shower.get_parameter(shp.geomagnetic_angle)

    energy_fluence_vector = revent.get_station_parameter(stp.energy_fluence)
    f_vxB = energy_fluence_vector[:, 0]

    c_early_late = revent.get_station_parameter(stp.early_late_factor)
    station_axis_distance = revent.get_station_axis_distance()

    # angle between station and positive vxB direction
    phi = revent.get_station_angle_to_vB()

    # charge excess fraction parametrized (with early late corrected axis distance)
    # a = revent.get_station_parameter(stp.charge_excess_fraction_parametric)
    a = get_parametric_charge_excess_fraction(revent)

    # print("Fluence", f_vxB[0:10])
    # print("EL", c_early_late[0:10])
    # print("station angle", phi[0:10])
    # print("Geomagnetic angle", alpha)
    # print("CE", a[0:10])


    # calculate pure geomagnetic contribution
    f_geo = geomagnetic_emission.calculate_geo_magnetic_energy_fluence(f_vxB * c_early_late ** 2, phi, alpha, a)

    # This has never been validated!
    # f_geo_pos = revent.get_station_parameter(stp.geomagnetic_fluence_positional)
    f_ce = a / (np.sin(alpha) ** 2) * f_geo

    # ASSUMPTION!
    # f_ce = (energy_fluence_vector[:, 0] + energy_fluence_vector[:, 1]) * c_early_late ** 2 - f_geo
    # f_ce = (energy_fluence_vector[:, 0] + energy_fluence_vector[:, 1]) * c_early_late ** 2 * a
    
    return f_geo, f_ce, a


def reconstruct_emission_from_param(events, para):

    for revent in events:

        # calculate pure geomagnetic contribution
        f_geo, f_ce, a = get_emission_from_param(revent)

        revent.set_station_parameter(stp.geomagnetic_fluence_parametric, f_geo)
        revent.set_station_parameter(stp.charge_excess_fluence_parametric, f_ce)
        revent.set_station_parameter(stp.charge_excess_fraction_parametric, a)


if __name__ == "__main__":
    def plot_maximum_in_footprints(events, para):
        from matplotlib import pyplot as plt
        from radiominimalysis.framework import factory

        pos = factory.get_parameter(events, shp.position_maximum_energy_fluence_interpolated_2d)
        f = factory.get_parameter(events, shp.maximum_energy_fluence_interpolated_2d)
        azimuth = factory.get_parameter(events, shp.azimuth)

        fig, ax = plt.subplots()
        sct = ax.scatter(pos[:, 0], pos[:, 1], c=np.rad2deg(azimuth), alpha=0.1)
        ax.set_aspect(1)
        cbi = plt.colorbar(sct, pad=0.01)
        plt.show()
