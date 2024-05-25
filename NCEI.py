import argparse
import glob
import os
import step01 as update_prof_and_tile_points_on_profiles
import step02 as update_spatial_bin_index_on_prepared_profiles
import step03 as update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles
import step04 as update_sigmaTS_on_prepared_profiles
import step05 as update_gamma_factor_on_prepared_profiles
import step06 as update_prof_insitu_T_to_potential_T
import step07 as update_zero_weight_points_on_prepared_profiles
import step08 as update_remove_zero_T_S_weighted_profiles_from_MITprof
import step09 as update_remove_extraneous_depth_levels
import step10 as update_decimate_profiles_subdaily_to_once_daily
from tools import MITprof_read, MITprof_write_to_nc

def NCEI_pipeline(dest_dir, input_dir):

    # Get a list of all netCDF files present in input directory 
    netCDF_files = glob.glob(os.path.join(input_dir, '*.nc'))
    # init dictionary for storing MITprofs data
    MITprofs = {}

    # ==========================================================================================
    # ========================== START OF NEED PATHS/ PAREMS ===================================
    # ==========================================================================================

    # Needed paths:
    # Set grid_dir
    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'

    # Path to llc090_sphere_point_n_10242_ids.bin and llc090_sphere_point_n_02562_ids.bin
    sphere_bin = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/grid_llc90/sphere_point_distribution'

    # Path to WOA13_v2_TS_clim_merged_with_potential_T.nc
    clim_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology'

    # Path to Salt_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin and Theta_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin
    CTD_TS_bin = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/CTD sigma TS/'

    # Step 1: update_prof_and_tile_points_on_profiles
    llcN = 90                       # Which grid to use, 90 or 270
    wet_or_all = 1                  # 0 = interpolated to nearest wet point, 1 = interpolated all points, regardless of wet or dry

    # Step 4: update_sigmaTS_on_prepared_profiles parems
    respect_existing_zero_weights = 0   # 0 = no, 1 = yes
    new_S_floor = 0.005                 # set this to zero if S_floor is unused 
    new_T_floor = 0                     # set this to zero if T_floor is unused 

    # Step 5: update_gamma_factor_on_prepared_profiles parems
    apply_gamma_factor = 1          #   0 = remove gamma factor from sigma, 1 = apply gamma to sigma
                                    #   gamma factor is factor 1/sqrt(alpha), where alpha = area/max(area) of the grid cell area in which this profile is found.

    # Step 6: update_prof_insitu_T_to_potential_T parems
    replace_missing_S_with_clim_S = 1   # 1 = replace, 0 = do not replace

    # Step 7: update_zero_weight_points_on_prepared_profiles
    # Various parems inside of if block pretaining to 'adjust' on lines 142 - 161 within script

    # Step 10: update_decimate_profiles_subdaily_to_once_daily
    distance_tolerance = 5e3        # radius within which profiles are considered to be at the same location [in meters]
    closest_time = 120000           # HHMMSS: if there is more than one profile per day in a location, choose the one
                                    # that is closest in time to 'closest time' default is noon 
    method = 1                      # method 0 or 1

    # ==========================================================================================
    # ========================== END OF NEED PATHS/ PAREMS =====================================
    # ==========================================================================================
    
    steps_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    steps_to_save = [10]

    # NOTE: add in loadin feature 

    if len(netCDF_files) != 0:
        for file in netCDF_files:
            MITprofs = MITprof_read(file, 0)
            basename = os.path.basename(file)
            if MITprofs != 0 :
                if 1 in steps_to_run:
                    # Updates prof_points and tile interpolation points so that the MITgcm knows which grid points to use for the cost 
                    update_prof_and_tile_points_on_profiles.main(MITprofs, grid_dir, llcN, wet_or_all)
                    if 1 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 1, basename)
                if 2 in steps_to_run:
                    # Updates each profile with a bin index that is specified from some file.
                    update_spatial_bin_index_on_prepared_profiles.main(sphere_bin, MITprofs, grid_dir)
                    if 2 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 2, basename)
                if 3 in steps_to_run:
                    # Assigns the WOA13 T and S climatology values to MITprof objects. 
                    update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles.main(clim_dir, MITprofs)
                    if 3 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 3, basename)
                # Update MITprof objects with new T and S uncertainity fields
                if 4 in steps_to_run:
                    update_sigmaTS_on_prepared_profiles.main(MITprofs, grid_dir, CTD_TS_bin, respect_existing_zero_weights, new_S_floor, new_T_floor)
                    if 4 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 4, basename)
                if 5 in steps_to_run:
                    # Updates the MITprof profiles with a new sigma based on whether we are applying or removing the 'gamma' factor
                    update_gamma_factor_on_prepared_profiles.main(MITprofs, grid_dir, apply_gamma_factor, llcN)
                    if 5 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 5, basename)
                if 6 in steps_to_run:
                    # Updates profile insitu temperatures so that they are in portential temperatures
                    update_prof_insitu_T_to_potential_T.main(MITprofs, replace_missing_S_with_clim_S)
                    if 6 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 6, basename)
                if 7 in steps_to_run:
                    # Zeros out profile profTweight and profSweight on points that match some criteria 
                    update_zero_weight_points_on_prepared_profiles.main('adjust', MITprofs)
                    if 7 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 7, basename)
                if 8 in steps_to_run:  
                    # Remove profiles that whose T and S weights are all zero from from MITprof structures
                    update_remove_zero_T_S_weighted_profiles_from_MITprof.main(MITprofs)
                    if 8 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 8, basename)
                if 9 in steps_to_run:    
                    update_remove_extraneous_depth_levels.main(MITprofs)
                    if 9 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 9, basename)
                if 10 in steps_to_run:    
                    # Decimates profiles with subdaily sampling at the same location to once-daily sampling
                    update_decimate_profiles_subdaily_to_once_daily.main(MITprofs, distance_tolerance, closest_time, method)
                    if 10 in steps_to_save:
                        MITprof_write_to_nc(dest_dir, MITprofs, 10, basename)
            else:
                raise Exception("No info in NetCDF files")
    else:
        raise Exception("No NetCDF files found")

def main(dest_dir, input_dir):
    NCEI_pipeline(dest_dir, input_dir)

if __name__ == '__main__':
  
   
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", action= "store",
                    help = "File path to NETCDF files containing MITprofs info." , dest= "input_dir",
                    type = str, required= True)
    
    parser.add_argument("-d", "--dest_dir", action= "store",
                help = "File path where you would like to store generated NETCDF file." , dest= "dest_dir",
                type = str, required= True)
    

    args = parser.parse_args()


    input_dir = args.input_dir
    dest_dir = args.dest_dir

    input_dir = "/home/sweet/Desktop/New Data/PYTHON/test"
    dest_dir = "/home/sweet/Desktop/Final NETCDF/PYTHON"
    main(dest_dir, input_dir)
