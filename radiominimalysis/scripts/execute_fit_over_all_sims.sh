python3 /cr/users/guelzow/simulations/radiominimalysis/scripts/LDF_process_all_sims_parallel.py \
--paths /cr/aera02/huege/guelzow/factories_LDF/GP300_ex_adc_pre_fit.pickle \
--save GP300_ex_adc_fitted_LDFs.pickle \
--fit_results \
--parallel_jobs 24 \
--atmModel 41 \
--function fit_from_pickle_param \
--shower sim_shower \
--realistic_input \
--plot \


# --only_infill \
# --remove_infill \
 

#
# GP300 grid
#

# DC2 with realistic reconstruction
# DC2_Coreas_realistic_pre_fit.pickle + DC2_Coreas_realistic_fitted_LDFs.pickle

# DC2 L1
# DC2_Coreas_L1_pre_fit.pickle + DC2_Coreas_L1_fitted_LDFs.pickle
# DC2_Coreas_L1_no_MC_pre_fit.pickle + DC2_Coreas_L1_no_MC_fitted_LDFs.pickle

# DC2 ADC with MC params
# DC2_Coreas_ADC_MC_pre_fit.pickle + DC2_Coreas_ADC_MC_fitted_LDFs.pickle

# DC2 subset
# DC2_Coreas_subset_pre_fit.pickle + DC2_Coreas_subset_fitted_LDFs.pickle

# DC2 iron/proton
# DC2_Coreas_iron_pre_fit.pickle   + DC2_Coreas_iron_fitted_LDFs.pickle
# DC2_Coreas_proton_pre_fit.pickle + DC2_Coreas_proton_fitted_LDFs.pickle

# CoREAS DC2 HDF5
# DC2_Coreas_hdf5_pre_fit.pickle + DC2_Coreas_hdf5_fitted_LDFs.pickle

# CoREAS DC2 NoJitter (from ROOT files)
# DC2_Coreas_no_jitter_pre_fit.pickle + DC2_Coreas_no_jitter_fitted_LDFs.pickle


#
# Starshapes
#

# AUGER: Auger_50-200_pre-fit_factory.pickle + Auger_50-200_fitted_LDFs.pickle

# GRAND: Dunhuang_stshp_pre_fit.pickle + Dunhuang_stshp_fitted_LDFs.pickle
