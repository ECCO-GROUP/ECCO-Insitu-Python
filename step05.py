import copy
import glob
import os

import numpy as np

from tools import MITprof_read, load_llc270_grid, load_llc90_grid


def update_gamma_factor_on_prepared_profiles(run_code, MITprofs, grid_dir):

    # SET INPUT PARAMETERS
    fillVal=-9999
    checkVal=-9000

    # --- save format  ---
    #   0 = netcdf
    #   1 = matlab
            
    # ----  apply_gamma_factor  ----- 
    #   0 = remove gamma factor from sigma
    #   1 = apply gamma to sigma
    #   gamma factor is factor 1/sqrt(alpha)
    #    where alpha = area/max(area) of the grid cell area in which
    #    this profile is found.

    save_output_to_disk = 1

    if run_code == '20181202_apply_gamma': 
        apply_gamma_factor = 1 # use the gamma or not.
        save_output_to_disk = 0
        llcN = 90

    elif run_code == '20190125_remove_gamma':
        apply_gamma_factor = 0 #use the gamma or not.
        save_output_to_disk = 0
        llcN = 90

    elif run_code ==  '20181015_llc90_GOSHIP_apply_gamma':
        raise Exception("uncoded (NETCDF part)")
        basedir = '/home/ifenty/data/observations/insitu/GO-SHIP/SIO_201810/TO_JPL_2018/'
        input_dir = os.path.join(basedir, 'step_04_zero_weights')
        output_dir = os.path.join(basedir, 'step_05_gamma')
        
        file_suffix_in  = 'step_04'
        file_suffix_out = 'step_05'
        """
        cd(input_dir);
        f = dir(['*nc']);
        for i = 1:length(f)
            fDataIn{i}=   f(i).name;
            
            tmp = f(i).name;
            tmp1 = strfind(tmp, file_suffix_in);
            tmp2 = length(file_suffix_in);
            fsi_start = tmp1;
            fsi_end   = tmp1 + tmp2;
            tmp3 = [tmp(1:fsi_start-1) file_suffix_out tmp(fsi_end:end)];
            
            fDataOut{i} = tmp3
        
        
        ilists = 1:length(f);
        """
        apply_gamma_factor = 1 # use the gamma or not.
        save_format = 0;  # 0 = netcdf, 1=matlab. 
        
        llcN = 90
        
    elif run_code ==  '20181015_llc90_ITP_apply_gamma':
        raise Exception("uncoded - everything")
        """
        basedir = ['/home/ifenty/data/observations/insitu/ITP/SIO_201810/TO_JPL_2018/']
        input_dir = [basedir 'step_04_zero_weights'];
        output_dir = [basedir 'step_05_gamma']
        
        file_suffix_in  = 'step_04'
        file_suffix_out = 'step_05'
        clear fData*;
        cd(input_dir);
        f = dir(['*nc']);
        for i = 1:length(f)
            fDataIn{i}=   f(i).name;
            
            tmp = f(i).name;
            tmp1 = strfind(tmp, file_suffix_in);
            tmp2 = length(file_suffix_in);
            fsi_start = tmp1;
            fsi_end   = tmp1 + tmp2;
            tmp3 = [tmp(1:fsi_start-1) file_suffix_out tmp(fsi_end:end)];
            
            fDataOut{i} = tmp3
        end
        
        ilists = 1:length(f);
        
        apply_gamma_factor = 1; % use the gamma or not.
        save_format = 0;  %0 = netcdf, 1=matlab. 
        
        llcN = 90
        """
    elif run_code ==  '20181015_ARGO_apply_gamma':
        raise Exception("uncoded - everything")
        """        
        basedir = ['/home/ifenty/data/observations/insitu/ARGO/from_gael_June2018/combined_by_latest/']
        input_dir = [basedir 'step_04_zero_weights'];
        output_dir = [basedir 'step_05_gamma']
        
        file_suffix_in  = 'step_04.nc'
        file_suffix_out = 'step_05.nc'
        
        cd(input_dir);
        f = dir(['*nc']);
        for i = 1:length(f)
            fDataIn{i}=   f(i).name;
            fDataOut{i} =[f(i).name(1:end-length(file_suffix_in)) file_suffix_out];
        end
        
        ilists = 1:length(f);
        
        apply_gamma_factor = 1; % use the gamma or not.
        save_format = 0;  %0 = netcdf, 1=matlab. 
        
        llcN = 90
        """
    elif run_code ==  'NODC_20181008_apply_gamma_clim_S':
        raise Exception("uncoded - everything")
        """
        basedir = ['/home/ifenty/data/observations/insitu/NODC/NODC_20180508/all/']
        input_dir = [basedir 'step_05_zero_weights_clim_S'];
        output_dir = [basedir 'step_06_gamma_clim_S']
        
        file_suffix_in  = 'step_05'
        file_suffix_out = 'step_06'
        clear fData*;
        cd(input_dir);
        f = dir(['*nc']);
        for i = 1:length(f)
            fDataIn{i}=   f(i).name;
            
            tmp = f(i).name;
            tmp1 = strfind(tmp, file_suffix_in);
            tmp2 = length(file_suffix_in);
            fsi_start = tmp1;
            fsi_end   = tmp1 + tmp2;
            tmp3 = [tmp(1:fsi_start-1) file_suffix_out tmp(fsi_end:end)];
            
            fDataOut{i} = tmp3
        end
        
        ilists = 1:length(f);
        
        apply_gamma_factor = 1; % use the gamma or not.
        save_format = 0;  %0 = netcdf, 1=matlab. 
        
        llcN = 90
        """

    #  load the RAC field only if we are applying a gamma factor
    #  when we remove a gamma factor the gamma value is already stored in the
    #  profile file.
    if apply_gamma_factor == 1:
        if llcN == 90:
            lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90, z_top_90, z_bot_90, hFacC_90, AI_90, z_cen_90 = load_llc90_grid(grid_dir)
            RAC = copy.deepcopy(RAC_90_pf)
                
        if llcN == 270:
            lon_270, lat_270, blank_270, wet_ins_270_k, X_270, Y_270, Z_270, bathy_270, good_ins_270, RAC_270_pf = load_llc270_grid(grid_dir)
            RAC = copy.deepcopy(RAC_270_pf)

    # for ilist = 1:length(MITprofs)
    #  load weights
    tmpT = MITprofs['prof_Tweight']
    #tmpT(find(tmpT < 0))=NaN;
    tmpT[np.where(tmpT < 0)[0]] = np.nan
    if 'prof_S' in MITprofs:
        tmpS = MITprofs['prof_Sweight']
        tmpS[np.where(tmpS <0)[0]] = np.nan
   
    # apply or remove the gamma factor
    if apply_gamma_factor == 1:
        # ['applying area factor -- ']
        # find ratio of each grid cell area to the maximum
        # alpha = RAC./max(RAC(:));
        alpha = np.squeeze(RAC/ np.max(RAC))
        pp = MITprofs['prof_point'].astype(int)
        alpha_pp = alpha.flatten(order = 'F')[pp]
        MITprofs['prof_area_gamma'] = alpha_pp
        fac = copy.deepcopy(alpha_pp)
        
    elif apply_gamma_factor == 0:
        #NOTE: go over w/ IAN
        # why would we remove it when we haven't added it?
        # ['removing area factor -- ']
        alpha_pp = MITprofs['prof_area_gamma']
        fac = 1/ alpha_pp
        MITprofs['prof_area_gamma'] = np.ones(MITprofs['prof_area_gamma'].size)
    
    # loop through k, apply fac to T and S weights
    for k in np.arange(MITprofs['prof_depth'].size):
        tmpTk = tmpT[:,k]
        tmpTk = tmpTk * fac
        tmpT[:,k] = tmpTk
        
        #['processing S'];
        if 'prof_S' in MITprofs:
            tmpSk = tmpS[:,k]
            tmpSk = tmpSk * fac
            tmpS[:,k] = tmpSk

    # overwrite T and S weight fields
    MITprofs['prof_Tweight'] = tmpT
    
    if 'prof_S' in MITprofs:
        MITprofs['prof_Sweight'] = tmpS
    
def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    
    update_gamma_factor_on_prepared_profiles(run_code, MITprofs, grid_dir)

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run_code", action= "store",
                        help = "Run code: 90 or 270" , dest= "run_code",
                        type = int, required= True)

    parser.add_argument("-g", "--grid_dir", action= "store",
                        help = "File path to 90/270 grids" , dest= "grid_dir",
                        type = str, required= True)
    
    parser.add_argument("-m", "--MIT_dir", action= "store",
                    help = "File path to NETCDF files containing MITprofs info." , dest= "MIT_dir",
                    type = str, required= True)
    

    args = parser.parse_args()

    run_code = args.run_code
    grid_dir = args.grid_dir
    MITprofs_fp = args.MIT_dir
    """

    #MITprofs_fp = '/home/sweet/Desktop/ECCO-Insitu-Ian/Python-Dest/step_4.nc'
    MITprofs_fp = '/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/20190131_END_CHAIN'

    """
    if run_code != 90 or run_code != 270:
        raise Exception("Runcode has to be 90 or 270!")
   
    """
    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 5)

    run_code = '20181202_apply_gamma'
    grid_dir = "hehe"

    main(run_code, MITprofs, grid_dir)