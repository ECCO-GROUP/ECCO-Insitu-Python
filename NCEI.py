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

def NCEI_pipeline(dest_dir, run_code, input_dir):

    # Get a list of all netCDF files present in input directory 
    netCDF_files = glob.glob(os.path.join(input_dir, '*.nc'))
    # init dictionary for storing MITprofs data
    MITprofs = {}

    if run_code == 90:
        grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    elif run_code == 270:
        grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/grid_llc270_common'
    else:
        raise Exception("Invalid run code passed")
    
    # Needed paths:
    # Path to llc090_sphere_point_n_10242_ids.bin and llc090_sphere_point_n_02562_ids.bin
    # NOTE: since this reads from grid_llc90, I assume there are different files for grid llc270?
    sphere_bin = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/grid_llc90/sphere_point_distribution'
    # Path to WOA13_v2_TS_clim_merged_with_potential_T.nc
    clim_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology'
    # Path to Salt_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin and Theta_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin
    CTD_TS_bin = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/CTD sigma TS/'

    if len(netCDF_files) != 0:
        for file in netCDF_files:
            MITprofs = MITprof_read(file, 0)
            if MITprofs != 0 :
                # Updates prof_points and tile interpolation points so that the MITgcm knows which grid points to tuse for the cost 
                update_prof_and_tile_points_on_profiles.main(run_code, MITprofs, grid_dir)
                # Updates each profile with a bin index that is specified from some file.
                update_spatial_bin_index_on_prepared_profiles.main(sphere_bin, MITprofs, grid_dir)
                # replaced interpolation w/ mat file
                update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles.main(clim_dir, MITprofs)
                # Update MITprof objects with new T and S uncertainity fields
                update_sigmaTS_on_prepared_profiles.main('20190126_do_not_respect_existing_weights', MITprofs, grid_dir,  CTD_TS_bin)
                # Updates the MITprof profiles with a new sigma based on whether we are applying or removing the 'gamma' factor
                update_gamma_factor_on_prepared_profiles.main('20181202_apply_gamma', MITprofs, grid_dir)
                # Updates profile insitu temperatures so that they are in portential temperatures
                update_prof_insitu_T_to_potential_T.main('20181202_use_clim_for_missing_S', MITprofs)
                # Zeros out profile profTweight and profSweight on points that match some criteria 
                update_zero_weight_points_on_prepared_profiles.main('20190126_high_lat', MITprofs)
                # Remove profiles that whose T and S weights are all zero from from MITprof structures
                update_remove_zero_T_S_weighted_profiles_from_MITprof.main(MITprofs)
                update_remove_extraneous_depth_levels.main(MITprofs)
                # Decimates profiles with subdaily sampling at the same location to once-daily sampling
                update_decimate_profiles_subdaily_to_once_daily.main('20181218_2', MITprofs)
            else:
                raise Exception("No info in NetCDF files")
    else:
        raise Exception("No NetCDF files found")

    MITprof_write_to_nc(dest_dir, MITprofs, 10)

def main(dest_dir, run_code, input_dir):
    NCEI_pipeline(dest_dir, run_code, input_dir)

if __name__ == '__main__':
  
   
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run_code", action= "store",
                        help = "Run code: 90 or 270" , dest= "run_code",
                        type = int, required= True)

    parser.add_argument("-i", "--input_dir", action= "store",
                    help = "File path to NETCDF files containing MITprofs info." , dest= "input_dir",
                    type = str, required= True)
    
    parser.add_argument("-d", "--dest_dir", action= "store",
                help = "File path where you would like to store generated NETCDF file." , dest= "dest_dir",
                type = str, required= True)
    

    args = parser.parse_args()

 
    run_code = args.run_code
    input_dir = args.input_dir
    dest_dir = args.dest_dir
 

    input_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/converted_to_MITprof"
    dest_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/Python-Dest"
    run_code = 90

    main(dest_dir, run_code, input_dir)
