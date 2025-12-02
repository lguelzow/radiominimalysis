from radiominimalysis.framework.parameters import stationParameters as stp, showerParameters as shp
from radiominimalysis.utilities import charge_excess as ce, geomagnetic_emission, cherenkov_radius as che

import sys
import numpy as np


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
