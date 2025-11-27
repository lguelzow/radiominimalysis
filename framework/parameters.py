from aenum import Enum


class eventParameters(Enum):
    '''
    - Enumeration class defining callable parameters (abbr. as evp)

    - Parameters: most general attributes and classification types for air shower measurements

    - Can be accessed using dot notation, for example, "evp.time"

    '''

    # important general parameters for event
    time = 1  # gps sec
    magnetic_field_vector = 2  # in auger CS
    rotation_angle_for_magfield_declination = 3
    refractive_index_at_sea_level = 4

    gamma_cut = 10

    # classifications of the event type
    rd_shower = 100 # radio detector shower
    sim_shower = 101 # simulated shower
    sd_shower = 102 # surface detector shower
    reference_shower = 110

    GRAND_shower = 103 # measured GRAND shower

    file = 1000


class showerParameters(Enum):
    '''
    - Enumeration class defining callable parameters (abbr. as shp)

    - Parameters: shower-wide attributes, often different version of the same parameter

    - Can be accessed using dot notation, for example, "shp.azimuth"

    '''

    # angles in deg, shower axis direction towards shower maximum
    azimuth = 1  # auger CS, defined between 0 - 360 deg
    zenith = 2

    # Energies of specific shower components, all energies in eV
    energy = 3  # primary cosmic ray energy / total shower energy
    electromagnetic_energy = 4  # energy of the electromagnetic cascade (e+/e-)
    invisible_energy = 5
    radiation_energy = 6  # total radiated energy in radio waves
    geomagnetic_energy = 7  # radiation energy of the geomagnetic emission
    charge_excess_energy = 8 # radiation energy of the charge excess emission
    
    # geometric attributes
    xmax = 9  # position of shower maximum in grammage g/cm2
    distance_to_shower_maximum_geometric = 10  # distance from core at ground (not sea level) to shower maximum in meter via radiotools
    distance_to_shower_maximum_grammage = 11  # like distance_to_shower_maximum_geometric but in g/cm2
    gemomagnetic_angle = 12  # angle between magnetic field and shower axis
    core = 13 # coordinates of the shower core
    observation_level = 14  # altitude of observer plane above sea level
    primary_particle = 15  # primary particle identification
    atmosphere_model = 16 # numbered atmosphere models (1: US standard; 27: Auger average; 40/41: China sites)
    geomagnetic_angle = 17  # same like 12
    density_at_shower_maximum = 18  # in kg/m3

    # derived from real MC parameters
    MC_distance_to_shower_maximum = 19
    MC_density_at_shower_maximum = 20

    # what is rit????
    xmax_rit = 30  # reconstructed depth of the rit maximum
    shower_axis_rit = 31  # with rit reconstructed shower axis
    core_rit = 32 # corrdinates of rit shower core

    # reconstructed direction and estimates from efield recon
    core_estimate = 33
    zenith_recon = 34
    azimuth_recon = 35
    geomagnetic_angle_recon = 36

    core_fit_shower_plane = 37
    core_pred_shower_plane = 38

    pointing_error = 39

    # alternatives?
    xmax2 = 40
    distance_to_shower_maximum_geometric2 = 41  # same like 10 but from CoREAS/CORSIKA

    # parameters related to the LDF fit
    geomagnetic_ldf_parameter = 50  # parameter of the geomagnetic ldf (f_ABCD, including r_0)
    has_ldf_parameter = 51  # has ldf fit parameter (including core, distance to xmax, geomagnetic ldf parameters)
    fit_result = 52  # if False, fit was not successful
    core_fit = 53  # fitted in vB, vvB

    # various versions of determined Cherenkov radius
    cherenkov_radius_model = 58  # r_che = tan(arccos(1 / n(h_max))) * d_max
    cherenkov_radius_fit = 59  # one value per shower (from ring fit)
    cherenkov_radius = 60  # array of length 6-8 for each arm (can also be determined during ring fit)
    cherenkov_geomagnetic_fluence = 61  # array of length 6-8 for each arm
    cherenkov_distance_to_shower_maximum_geometric = 62
    cherenkov_distance_to_shower_maximum_grammage = 63
    
    r0_start = 64


    radiation_energy_thin = 98
    charge_excess_energy_position = 99  # geomagnetic emission from positional emission
    geomagnetic_energy_position = 100  # geomagnetic emission from positional emission
    radio_energy_estimator = 101
    reconstructed_electromagnetic_energy = 102
    distance_to_shower_maximum_geometric_fit = 103  # see distance_to_shower_maximum_geometric, fitted
    estimated_distance_to_shower_maximum_geometric = 104

    reconstruction_level = 110  # arbitary float, see offline for more information
    contained = 111  # shower being contained, i.e., core lies within hull of event stations

    radius = 120  # as fitted by spherical fit (~ distance to xmax) 

    prediceted_core_shift = 200  # core shift as distance on ground
    shower_size = 300

    # simulation specific parameters
    slicing_method = 400  # can be an array for each slice (with two entries when you have double slicing)
    slicing_edges = 401  # lower and upper edge (with two entries when you have double slicing)

    position_maximum_energy_fluence_interpolated_2d = 411
    maximum_energy_fluence_interpolated_2d = 412

    passed = 500
    trigger = 501

    # HACK! might be used for offline sd signal
    n_triggered_stations = 1000
    station_ids = 1001  # ids of all triggerd stations
    station_signals = 1002 
    candidate_stations = 1003  # ids of all candidate stations


class stationParameters(Enum):
    '''
    - Enumeration class defining callable parameters (abbr. as stp)

    - Parameters: station specific attributes, usually stored in arrays for the whole shower

    - Can be accessed using dot notation, for example, "stp.position"

    '''

    # on event level this parameters are stored as arrays: (n_antennas, paramerter)
    position = 1
    name = 3
    energy_fluence = 4  # signal energy fluence (vector!), typically:  vxB, vxvxB, v
    vxB_error = 6 # error of the vxB fluence, calculated from noise window RMS, noise fluence and detector sensitivity uncertainty
    frequency_slope = 7  # slope of the frequency spectrum, polinom of first order in log (y-axis)
    signal_to_noise_ratio = 8  # signal_to_noise_ratio_vector = 105
    noise_energy_fluence = 9
    sim_energy_fluence = 10
    noise_rms = 11
    peak_amplitude = 12

    early_late_factor = 13 # correction factor for eliminating signal differences from geometric effects

    # energy fluences 
    geomagnetic_fluence_positional = 14  # energy fluence of the geomagnetic emission from known position (not early late corrected!)
    charge_excess_fluence_positional = 15  # energy fluence of the charge excess emission from known position (not early late corrected!)
    geomagnetic_fluence_parametric = 16  # energy fluence of the geomagnetic emission from param (early late corrected)
    charge_excess_fluence_parametric = 17  # energy fluence of the geomagnetic emission from param (early late corrected)

    vxB_fluence_simulated = 17 # energy fluence of the geomagnetic emission determined from simulations
    vxB_fluence_model = 18 # energy fluence of the geomagnetic emission determined from signal model
    
    # fraction of the total emission made up by charge excess
    charge_excess_fraction_parametric = 19  # defined as a = sin(alpha) ** 2 * f_ce / f_geo (early late corrected!)
    
    stokes_parameter = 20  # Stokes parameter: I, Q, U, V in eV/m2

    time = 21
    sim_time = 22
    trace_start_time = 23
    
    # ADC noise cut
    rejected_noisy = 24
    positions_noisy = 25 # positions of stations with too high SNR in efields
    energy_fluence_noisy = 26 # signal energy fluence of stations with too high SNR in efields
    error_noisy = 27 # error of the vxB fluence of stations with too high SNR in efields
    
    # bad timing cut
    rejected_bad_timing = 28
    positions_bad_timing = 29 # positions of stations with too high SNR in efields
    energy_fluence_bad_timing = 30 # signal energy fluence of stations with too high SNR in efields
    error_bad_timing = 31 # error of the vxB fluence of stations with too high SNR in efields
    
    # quality cut parameters
    rejected_snr = 32
    positions_snr = 33 # positions of stations with too high SNR in efields
    energy_fluence_snr = 34 # signal energy fluence of stations with too high SNR in efields
    error_snr = 35 # error of the vxB fluence of stations with too high SNR in efields
    
    # saturated stations
    saturated = 36
    saturated_fluence = 37
    saturated_positions = 38
    saturated_errors = 39

    zenith = 50  # angle of shower maximum from station

    distance_to_shower_maximum_geometric = 51  # distance from station at ground (not sea level) to shower maximum in meter
    distance_to_shower_maximum_grammage = 52  # like distance_to_shower_maximum_geometric but in g/cm2

    # thinning parameters for data cuts
    cleaned_from_thinning = 100  # mask, if true: no impact from thinning
    thinning_distance = 101  # beyond this distance stations are neglected because of thinning
    id = 102 
    signal_to_noise_ratio_vector = 105

    # compare parameter
    fluence_compare_MC = 109
    fluence_compare_rec = 110
    antennas_compare = 111

    # infill
    infill_id = 112
    only_infill_id = 113
    array_ids = 114

    traces = 200
    times = 201

    traces_filtered_downsampled = 300
    times_filtered_downsampled = 301

    #smoothness = 400

    # offline
    energy_fluence_vxB_predicted = 1000
    ldf_fit_positon = 1001

    #
