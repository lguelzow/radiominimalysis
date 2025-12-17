python3 /cr/users/guelzow/simulations/radiominimalysis/radiominimalysis/scripts/LDF_fit_parallel.py \
--paths /cr/aera02/huege/guelzow/factories_thesis/GP300_L1_MC_pull_read-in.pickle \
--save GP300_L1_MC_pull_pre_fit.pickle \
--parallel_jobs 24 \
--atmModel 41 \
--function prefit_calculations \
--verbose \
--shower sim_shower \
# --realistic_input


# pre-fit includes geometry and signal emissions
# to only reconstruct geometry, use function "geometry"


# CoREAS DC2 with realistic reconstruction
# /cr/users/guelzow/simulations/radiominimalysis/ldf_eval/factories/DC2_Coreas_read-in_realistic.pickle
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_ADC_noise/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-*
# DC2_Coreas_realistic_pre_fit.pickle

# CoREAS DC2 with ADC reconstruction and MC parameters
# /cr/users/guelzow/simulations/radiominimalysis/ldf_eval/factories/DC2_Coreas_ADC_MC_read-in.pickle
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_ADC_noise/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-*
# DC2_Coreas_ADC_MC_pre_fit.pickle

# CoREAS DC2 L1
# /cr/users/guelzow/simulations/radiominimalysis/ldf_eval/factories/DC2_Coreas_L1_read-in.pickle
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/full_library/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-*
# DC2_Coreas_L1_pre_fit.pickle

# CoREAS DC2 subset (L1)
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/full_library/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-random-library-iron_0032
# DC2_Coreas_subset_pre_fit.pickle

# CoREAS DC2 HDF5
# /cr/users/guelzow/simulations/sim-libs/DC2/hdf5_no_noise/SIM*
# DC2_Coreas_hdf5_pre_fit.pickle

# CoREAS NoJitter
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_no_jitter/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-random-library-*
# DC2_Coreas_NJ_pre_fit.pickle

# ZHaireS DC2
# /cr/aera02/huege/guelzow/GRAND_DC2/ZHaireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/
# DC2_ZHaireS_pre_fit.pickle


#
# Starshapes
#

# Auger: 
# /cr/users/guelzow/simulations/sim-libs/Auger_data/50-200_RdLib/SIM*
# Auger_50-200_pre-fit_factory.pickle

# GRAND:
# /cr/users/guelzow/simulations/sim-libs/Dunhuang_stshp/china_stshps/**/**/**/**/*_highlevel.hdf5
# Dunhuang_stshp_pre_fit.pickle