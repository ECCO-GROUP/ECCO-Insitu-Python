# ECCO-Insitu Processing Pipeline
Sweet Zhang 5/21/2023

The **NCEI.py** is a python script that is designed to process a NETCDF file that contains WOD data. There are 10 steps included in the pipeline that run various data verification tests to insure data quality. The pipeline relies on a number of binary files which can be downloaded [here](https://drive.google.com/drive/folders/17h0qMS7vVimet8FXieGP1mWhnqnY0ljr?usp=sharing), please set the needed paths to these files accordingly.

## Preprocessing: csv_to_nc.py
This script creates a set of NETCDF files from a '.csv' file containing WOD data. This script does some preliminary error checks of the following fields: 
- prof_flag: origin flag verification to make sure data is accurate
  - Origin flag information link [here](https://www.nodc.noaa.gov/OC5/WOD/CODES/s_96_origflagset.html)
- datetime: makes sure date and time is valid
- longitude: value must be between - 180/180 and have valid units of 'decimal degrees'
- latitude: value must be between -90/90 and have valid units of 'decimal degrees'
- depth column: data must exist and have valid units of 'm'
- temperature column: data must exist and have valid units of 'degrees C'
- salinity column: if salinity data exists, must have valid units of 'PSS'

Any profiles failing the aforementioned tests will be excluded from the final NETCDF file. A log file will be generated containing a description of the failed test and the line number it occured at in the '.csv' file.

### Running the script
FLAG          | DESCRIPTION
------------- | -------------
-i            | Path of input directory where CSV files are stored
-d            | Path of directory where output NETCDF and log files will be stored

Output files: 
- Log file: log_[filename]_[YYYY-MM-DD-HH-MM].txt
- NETCDF files: [filename]_[year].csv

### Example Data
Please see the **preprocessing_examples** folder for examples of data input and output.
- Input file: ocldb1525460187.4974.CTD.csv 
- Output files: all '.nc' files in folder, a '.txt' log file

## NCEI.py
This script creates a set of NETCDF files containing processed data. The steps of the data verification process are outlined below.
1. **update_prof_and_tile_points_on_profiles(MITprofs, grid_dir, llcN, wet_or_all)**
   - Updates prof_points and tile interpolation points so that the MITgcm knows which grid points to use for the cost 
2. **update_spatial_bin_index_on_prepared_profiles(sphere_bin, MITprofs, grid_dir)**
   - Updates each profile with a bin index that is specified from some file.
3. **update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles(clim_dir, MITprofs)**
   - Assigns the WOA13 T and S climatology values to MITprof objects. 
4. **update_sigmaTS_on_prepared_profiles(MITprofs, grid_dir, CTD_TS_bin, respect_existing_zero_weights, new_S_floor, new_T_floor)**
   - Update MITprof objects with new T and S uncertainity fields
5. **update_gamma_factor_on_prepared_profiles(MITprofs, grid_dir, apply_gamma_factor, llcN)**
   - Updates the MITprof profiles with a new sigma based on whether we are applying or removing the 'gamma' factor
6. **update_prof_insitu_T_to_potential_T(MITprofs, replace_missing_S_with_clim_S)**
    - Updates profile insitu temperatures so that they are in portential temperatures
7. **update_zero_weight_points_on_prepared_profiles('adjust', MITprofs)**
    - Zeros out profile profTweight and profSweight on points that match some criteria 
8. **update_remove_zero_T_S_weighted_profiles_from_MITprof(MITprofs)**
    - Remove profiles that whose T and S weights are all zero from from MITprof structures
9. **update_remove_extraneous_depth_levels(MITprofs)**
    - Remove profiles that whose T and S weights are all zero from from MITprof structures
10. **update_decimate_profiles_subdaily_to_once_daily(MITprofs, distance_tolerance, closest_time, method)**
    - Decimates profiles with subdaily sampling at the same location to once-daily sampling

### Running the script
Before running the script, there are some input parems to adjust within the function **NCEI_pipeline** located in the **NCEI.py** file. Between lines 25-65, the following paths pertaining to the binary file set linked above need to be specified:
- grid_dir
  - Path to grid_llc90 or grid_llc270 folder
- sphere_bin
  - Path to sphere_point_distribution folder containing files llc090_sphere_point_n_10242_ids.bin and llc090_sphere_point_n_02562_ids.bin
- clim_dir
  - Path to TS_climatology folder containing file WOA13_v2_TS_clim_merged_with_potential_T.nc
- CTD_TS_bin
  - Path to CTD_sigma_TS folder containing files Salt_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin and Theta_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin

In addition to these paths, various input parems can be adjusted depending on user specifications.
Step 1:
  - **llcN**: this number should correspond to grid_dir (90 or 270)
  - **wet_or_all**: 0 = interpolated to nearest wet point, 1 = interpolated all points, regardless of wet or dry
    
Step 4: 
  - **respect_existing_zero_weights**: 0 = no, 1 = yes
  - **new_S_floor**: set this to zero if S_floor is unused 
  - **new_T_floor**: set this to zero if T_floor is unused

Step 5:
  - **apply_gamma_factor**: 0 = remove gamma factor from sigma, 1 = apply gamma to sigma. Gamma factor is factor 1/sqrt(alpha), where alpha = area/max(area) of the grid cell area in which this profile is found.

Step 6:
  - **replace_missing_S_with_clim_S**: 1 = replace, 0 = do not replace

Step 7:
  - Various parems inside of if block pretaining to 'adjust' on lines 142 - 161 within script

Step 10:
  - **distance_tolerance**: radius within which profiles are considered to be at the same location in meters
  - **closest_time**: HHMMSS: if there is more than one profile per day in a location, choose the one that is closest in time to 'closest time' default is noon 
  - **method**: located within step10, choose method 0 or 1

Right below these steps located on lines 69-70, adjust parems **steps_to_run** and **steps_to_save** if there is need to step any step or save any intermediate files. 

FLAG          | DESCRIPTION
------------- | -------------
-i            | Path of input directory where NETCDF files are stored
-d            | Path of directory where output NETCDF will be stored

Output files: 
- NETCDF files: [filename]_[step]_[number]_[DATA_FLAG]_[year].csv
Example: if input NETCDF name is **ocldb1525460187.4974.CTD_1995.nc**, output filename will be **ocldb15254601874974_step_10_CTD_1995.nc**

### Example Data
Please see the **processed_examples** folder for examples of data output

## Future Tasks
- [ ] Clean up unpopulated fields in NETCDF files
- [ ] Memory allocation issue (step 4), figure out an more efficent way of computing interpolations
      
Author: Sweet Zhang, Ian Fenty
Transferred to ECCO-GROUP 2024-05-16

