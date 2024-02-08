import argparse
import glob
import os
import numpy as np
import step01 as one
import step02 as two
import netCDF4 as nc

from tools import MITprof_read

def MITprof_write_to_nc(MITprof):

    dest_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/Python-Dest"

    nc_path = os.path.join(dest_dir, "step_two.nc")
    
    nc_file = nc.Dataset(nc_path, 'w')

    x, y = MITprof['prof_T'].shape
    one_dim_size = nc_file.createDimension('one_dim', len(MITprof['prof_HHMMSS'])) 
    multi_dim_size_x = nc_file.createDimension('dim_x', x)
    multi_dim_size_y = nc_file.createDimension('dim_y', y)

    # Create NETCDF variables
    prof_HHMMSS_var = nc_file.createVariable('prof_HHMMSS', np.int64, 'one_dim')
    prof_HHMMSS_var[:] = MITprof['prof_HHMMSS']

    prof_YYMMDD_var = nc_file.createVariable('prof_YYYYMMDD', np.int64, 'one_dim')
    prof_YYMMDD_var[:] = MITprof['prof_YYMMDD']
    
    prof_lat_var = nc_file.createVariable('prof_lat', np.float64, 'one_dim')
    prof_lat_var[:] = MITprof['prof_lat']

    prof_lon_var = nc_file.createVariable('prof_lon', np.float64, 'one_dim')
    prof_lon_var[:] = MITprof['prof_lon']

    prof_interp_XC11 = nc_file.createVariable('prof_interp_XC11', np.float64, 'one_dim')
    prof_interp_XC11[:] = MITprof['prof_interp_XC11']

    prof_interp_YC11 = nc_file.createVariable('prof_interp_YC11', np.float64, 'one_dim')
    prof_interp_YC11[:] = MITprof['prof_interp_YC11']

    prof_interp_XCNINJ = nc_file.createVariable('prof_interp_XCNINJ', np.float64, 'one_dim')
    prof_interp_XCNINJ[:] = MITprof['prof_interp_XCNINJ']

    prof_interp_YCNINJ = nc_file.createVariable('prof_interp_YCNINJ', np.float64, 'one_dim')
    prof_interp_YCNINJ[:] = MITprof['prof_interp_YCNINJ']

    prof_interp_i = nc_file.createVariable('prof_interp_i', np.float64, 'one_dim')
    prof_interp_i[:] = MITprof['prof_interp_i']

    prof_interp_j = nc_file.createVariable('prof_interp_j', np.float64, 'one_dim')
    prof_interp_j[:] = MITprof['prof_interp_j']

    prof_interp_weights = nc_file.createVariable('prof_interp_weights', np.float64, 'one_dim')
    prof_interp_weights[:] = MITprof['prof_interp_weights']

    prof_interp_lon = nc_file.createVariable('prof_interp_lon', np.float64, 'one_dim')
    prof_interp_lon[:] = MITprof['prof_interp_lon']

    prof_interp_lat = nc_file.createVariable('prof_interp_lat', np.float64, 'one_dim')
    prof_interp_lat[:] = MITprof['prof_interp_lat']

    prof_point = nc_file.createVariable('prof_point', np.float64, 'one_dim')
    prof_point[:] = MITprof['prof_point']

    prof_bin_id_a = nc_file.createVariable('prof_bin_id_a', np.float64, 'one_dim')
    prof_bin_id_a[:] = MITprof['prof_bin_id_a']

    prof_bin_id_b = nc_file.createVariable('prof_bin_id_b', np.float64, 'one_dim')
    prof_bin_id_b[:] = MITprof['prof_bin_id_b']

    # Create var - multi dim

    prof_S_var = nc_file.createVariable('prof_S', np.float64, ('dim_x', 'dim_y'))
    prof_S_var.units = 'N/A'
    prof_S_var[:] = MITprof['prof_S']

    prof_T_var = nc_file.createVariable('prof_T', np.float64, ('dim_x', 'dim_y'))
    prof_T_var.units = 'Degree C'
    prof_T_var[:] = MITprof['prof_T']

    prof_depth_var = nc_file.createVariable('prof_depth', np.float64, ('dim_x', 'dim_y'))
    prof_depth_var.units = 'm'
    prof_depth_var[:] = MITprof['prof_depth']

    prof_depth_f_flag_var = nc_file.createVariable('prof_Tflag', np.float64, ('dim_x', 'dim_y'))
    prof_depth_f_flag_var.units = 'N/A'
    prof_depth_f_flag_var[:] = MITprof['prof_Tflag']

    prof_depth_o_flag_var = nc_file.createVariable('prof_Sflag', np.float64, ('dim_x', 'dim_y'))
    prof_depth_o_flag_var.units = 'N/A'
    prof_depth_o_flag_var[:] = MITprof['prof_Sflag']

    nc_file.close()

# what does NODC stand for?
def NODC_pipeline(dest_dir, file_type, input_dir):

    # Get a list of all netCDF files present in input directory 
    netCDF_files = glob.glob(os.path.join(input_dir, '*.nc'))
    # init dictionary for storing MITprofs data
    MITprofs = {}

    # NOTE: var created: filepath_data_in for list of file
    # moved from step 01
    if len(netCDF_files) != 0:
        for file in netCDF_files:
            MITprofs = MITprof_read(file)
            if MITprofs != 0 :
                one.main(file_type, MITprofs, "grid_dir")
                two.main(file_type, MITprofs, "blah")
            else:
                raise Exception("No info in NetCDF files")
    else:
        raise Exception("No NetCDF files found")

    MITprof_write_to_nc(MITprofs)

    return
    """
    ## Runs loop to process files
    for step = steps_to_run:
        ['STEP ' num2str(step)]
        tmp_name_current =['MITprofs_new_' num2str(step)];
        [tmp_name_current ' <-- ' tmp_name_last]
        
        switch step
            case 0
                run_code = '20181202_llc90'
                eval([tmp_name_current '= update_prof_and_tile_points_on_profiles(run_code, eval(tmp_name_last))']);
            case 1
                run_code = '10242 and 2562'
                eval([tmp_name_current '= update_spatial_bin_index_on_prepared_profiles(run_code, eval(tmp_name_last))']);
            case 2
                run_code = '20181202_NODC'
                eval([tmp_name_current '= update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles(run_code, eval(tmp_name_last))']);
            case 3
                run_code = '20190126_do_not_respect_existing_weights'
                eval([tmp_name_current '= update_sigmaTS_on_prepared_profiles(run_code,  eval(tmp_name_last))']);
            case 4
                run_code = '20181202_apply_gamma'
                eval([tmp_name_current ' = update_gamma_factor_on_prepared_profiles(run_code, eval(tmp_name_last))']);
            case 5
                run_code = '20181202_use_clim_for_missing_S'
                eval([tmp_name_current '= update_prof_insitu_T_to_potential_T(run_code, eval(tmp_name_last))']);
            case 6
                run_code = '20190126_high_lat'
                eval([tmp_name_current '= update_zero_weight_points_on_prepared_profiles(run_code,  eval(tmp_name_last))']);
            case 7 
                run_code = '20181202'
                eval([tmp_name_current ' = update_remove_zero_T_S_weighted_profiles_from_MITprof(run_code, eval(tmp_name_last))']);
                %verify_only_profiles_with_nonzero_weights(eval(tmp_name_current));            
            case 8 
                eval([tmp_name_current ' = update_remove_extraneous_depth_levels(eval(tmp_name_last), make_figs, fig_dir)']);
                %% eval([tmp_name_current ' = update_remove_extraneous_depth_levels(eval(tmp_name_last), ' ...
                %%    'field_name, make_figs, fig_dir)']);
            case 9
                run_code = '20181218_2';
                %% eval([tmp_name_current '= update_decimate_profiles_subdaily_to_once_daily(run_code, ' ...
                %%    'eval(tmp_name_last), field_name, make_figs, fig_dir)']);
                eval([tmp_name_current '= update_decimate_profiles_subdaily_to_once_daily(run_code, ' ...
                    'eval(tmp_name_last), make_figs, fig_dir)']);
                %verify_only_profiles_with_nonzero_weights(eval(tmp_name_current));
 
        
        ## save memory, remove the last MITprofs_new fields from memory
        eval(['clear ' tmp_name_last])
        tmp_name_last = tmp_name_current;
        
        ## saves steps_to_save files
        if ismember(step, steps_to_save)
            cd(output_dir)
            save(['MITprofs_new_' field_name '_step_' num2str(step) '.mat'],[tmp_name_current]','-v7.3')
        end
        

    ## Save final to NetCDF format if specified
    # don't you always want to save to NetCDF?
    if save_final_to_netcdf
        for i = 1:length(eval(tmp_name_current))
            tmp = eval(tmp_name_current);
            MITprof = tmp{i};
            
            years = unique(floor(MITprof.prof_YYYYMMDD./1e4));
            
            if length(years) > 1
                stop
            else
                y(i) = years;
            end
            
            % convert prof desc to character array.
            MITprof.prof_descr=char(MITprof.prof_descr);
            
            fname_base = [field_name '_' file_suffix '_' num2str(years)]
            cd(output_dir)
            % write netcdf
            write_profile_structure_to_netcdf(MITprof, [fname_base '.nc'])
        end
    end
    """

def main(dest_dir, file_type, input_dir):
    # need to change name later
    NODC_pipeline(dest_dir, file_type, input_dir)

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dest_dir", action= "store",
                        help = "The destination where output files will be stored" , dest= "dest_dir",
                        type = str, required= True)

    parser.add_argument("-f", "--file_type", action= "store",
                        help = "The input filetype to be processed (CTD, MRB, APB, GLD, XBT)" , dest= "file_type",
                        type = str, required= True)
    

    args = parser.parse_args()

 
    dest_dir = args.dest_dir
    file_type = args.file_type
    """
    #steps_to_run = [0, 1, 2, 3 4 5 6 7 8 9]
    #steps_to_save = [6 9]
    #save_final_to_netcdf = 1

    #file_suffix = '20190131'
    #make_figs= 0

    # needed: where the .nc files are located
    input_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/converted_to_MITprof"

    # using 1997.nc (same file running in MATLAB pipeline)
    dest_dir = 90
    file_type = 10242

    main(dest_dir, file_type, input_dir)
