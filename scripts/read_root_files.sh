python3 /cr/users/guelzow/simulations/radiominimalysis/scripts/LDF_process_all_sims_parallel.py \
--paths /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_ADC_L1_64/sim* \
--save GP300_L1_MC_pull_read-in.pickle \
--parallel_jobs 24 \
--atmModel 41 \
--function root_reader \
--verbose \
--shower sim_shower \

# reads in DC2 output files in a parallelised way

# Auger: 
# /cr/users/guelzow/simulations/sim-libs/Auger_data/50-200_RdLib/SIM*
# Auger_50-200_pre-fit_factory.pickle

# GRAND:
# /cr/users/guelzow/simulations/sim-libs/Dunhuang_stshp/china_stshps/**/**/**/**/*_highlevel.hdf5
# Dunhuang_stshp_pre_fit.pickle

# ZHaireS DC2
# /cr/aera02/huege/guelzow/GRAND_DC2/ZHaireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/
# DC2_ZHaireS_read-in.pickle

# CoREAS DC2 ADC
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_ADC_noise/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-*

# CoREAS DC2 L1
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/full_library/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-*

# CoREAS DC2 HDF5
# /cr/users/guelzow/simulations/sim-libs/DC2/hdf5_no_noise/SIM*
# DC2_Coreas_hdf5_read-in.pickle

# CoREAS NoJitter
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/lib_no_jitter/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-random-library-*
# DC2_Coreas_NJ_read-in.pickle

# CoREAS DC2 subset
# /cr/aera02/huege/guelzow/GRAND_DC2/CoREAS/full_library/sim_Dunhuang_20170331_220000_RUN1_CD_Coreas-DC2-random-library-iron_0032
# DC2_Coreas_subset_read-in.pickle