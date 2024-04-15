import argparse
import copy
import glob
import os
from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sio
import netCDF4 as nc
from scipy.interpolate import griddata
from step02 import TriScatteredInterp
from tools import MITprof_read, load_llc90_grid, patchface3D, sph2cart
from scipy import interpolate


def interp_2D_to_arbitrary_z_levels(orig_data_xz, orig_z_centers, new_z_centers, extend_last_good_value_to_depth):

    # force z values to be Nx1
    orig_z_centers = np.squeeze(orig_z_centers)
    new_z_centers = np.squeeze(new_z_centers)

    if orig_data_xz.shape[1] == orig_z_centers.size:
        orig_data_xz = orig_data_xz.T

    # the orig data must have nans where there is no data
    # loop through the i,j of the original xyz field.
    if extend_last_good_value_to_depth:
        raise Exception("UNTRANSLATED")
        # pull the number of elements (x = rows)
        n_recs = orig_data_xz.shape[0]
        #extended_data_xz = orig_data_xz.*NaN;
        extended_data_xz = orig_data_xz * np.nan
        # go through each element in x
        """
        for i = 1:size(orig_data_xz,1)

            if (mod(i, floor(n_recs/100))==0)
                %i./n_recs
                %toc
            end
           
            tmp = squeeze(orig_data_xz(i,:));
            nnans = find(~isnan(tmp));
            
            if length(nnans) > 0
                nans = find(isnan(tmp));
                % make all of the successive values the same as the last good
                % values in z
                tmp(nans) = tmp(nnans(end));
                
                % update extended data
                extended_data_xz(i,:) = tmp;
            end
        end
        orig_data_xz = extended_data_xz;intersect(zero_weight_T_ins
    end
    """

    # data must be in rows = x (different points)
    #                 columns = z (different depths);
        
    # convert masked arr to non-masked type
    new_z_centers = new_z_centers.filled(np.nan)
    # Create interpolation function for all rows of orig_data_xz simultaneously
    interp_func = interpolate.interp1d(orig_z_centers, orig_data_xz.T, kind='linear', bounds_error=False, fill_value=np.nan)

    # Interpolate at new_z_centers
    new_data_xz = interp_func(new_z_centers)

    return new_data_xz


def interp_3D_to_arbitrary_z_levels(orig_data_xyz, orig_z_centers, new_z_centers, extend_last_good_value_to_depth):
    
    nx, ny, nz = orig_data_xyz.shape

    # convert shape to 2D
    orig_data_xz = np.reshape(orig_data_xyz, (nx*ny, nz), order='F')

    # interp using 2D script
    new_data_xz = interp_2D_to_arbitrary_z_levels(orig_data_xz, orig_z_centers, new_z_centers, extend_last_good_value_to_depth)

    nz_new = new_z_centers.shape[0]
    #new_data_xyz = reshape(new_data_xz, [nx ny nz_new]);
    new_data_xyz = np.reshape(new_data_xz, (nx, ny, nz_new), order = 'F')

    return new_data_xyz

def make_F_llc90_WET_INS_SURF_XYZ_to_INDEX(AI_90, X_90, Y_90, Z_90, wet_ins_90_k):

    # AI_90 is loaded with load_llc90_grid.m
    # AI_90_wet = AI_90(wet_ins_90_k{1});
    AI_90_wet = AI_90.flatten(order = 'F')[wet_ins_90_k[0]] 

    # these are the x,y,z coordinates of the 'wet' cells in llc90
    xyz_wet= np.column_stack((X_90.flatten(order = 'F')[wet_ins_90_k[0]], Y_90.flatten(order = 'F')[wet_ins_90_k[0]], Z_90.flatten(order = 'F')[wet_ins_90_k[0]]))

    # these AI points are in no domain, they are just a list of points
    # a mapping between these x,y,z points and the AI_90 index
    # you provide an x,y,z and F_llc90_XYZ_to_INDEX provides you with an
    # index in the global llc90 file correpsonding with the nearest
    # neighbor.

    F_llc90_WET_INS_SURF_XYZ_to_INDEX = TriScatteredInterp(xyz_wet, AI_90_wet,'nearest')
    return F_llc90_WET_INS_SURF_XYZ_to_INDEX


def make_llc90_z_map(z_top_90, z_bot_90):

    #z_entire_column = (1:z_bot_90(end));
    z_entire_column = np.arange(0, z_bot_90[-1] - 1)
    nz = 50
    z_map = np.zeros_like(z_entire_column)

    for i in np.arange(nz):
        ztop = z_top_90[i]
        zbot = z_bot_90[i]
        #zinds = find(z_entire_column >= ztop & z_entire_column < zbot);
        zinds = np.where((z_entire_column >= ztop) & (z_entire_column < zbot))[0]
        z_map[zinds] = i
    
    return z_map

def update_sigmaTS_on_prepared_profiles(run_code, MITprofs, grid_dir):

    # SET INPUT PARAMETERS
    fillVal=-9999
    checkVal=-9000

    # new_T_floor, new_S_floor.  If greater than zero, ensure that the 
    # minimum uncertainty in T and S are at least as high as new_T_floor,
    # new_S_floor
    new_T_floor = 0
    new_S_floor = 0
            
    # respect_existing_zero_weights:  
    #   0 = no
    #   1 = yes
            
    make_figs = 0
    save_output_to_disk = 35
    
    if run_code == '20190126_do_not_respect_existing_weights':
        save_output_to_disk = 0
        # use if the MITprof fields
        # do not have some zero weights
        respect_existing_zero_weights = 0
        new_S_floor = 0.005
                  
    if run_code == '20190126_respect_existing_weights':
        save_output_to_disk = 0
        # use if the MITprof fields
        # already have some zero weights
        respect_existing_zero_weights = 1
        new_S_floor = 0.005
            
    if run_code == '20181202_do_not_respect_existing_weights':
        save_output_to_disk = 0
        # use if the MITprof fields
        # do not have some zero weights
        respect_existing_zero_weights = 0
            
    if run_code == '20181202_respect_existing_weights':
        save_output_to_disk = 0
        #  use if the MITprof fields
        # already have some zero weights
        respect_existing_zero_weights = 1
    
    # Init profile package
    #  Read in grid for sigma and climatology

    lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90, z_top_90, z_bot_90, hFacC_90, AI_90, z_cen_90 = load_llc90_grid(grid_dir)
    z_map = make_llc90_z_map(z_top_90, z_bot_90)

    hf = np.copy(hFacC_90)
    hf[hf != 1] = np.nan

    # default sigma dir and filenames
    sigma_dir  = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/CTD sigma TS/'

    # salt
    sigma_S_path = os.path.join(sigma_dir, 'Salt_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin')
    llcN = 90
    mform = '>f4'
    siz = [llcN, 13*llcN, 50]
    #sigma_S = readbin(fsigST{1},[llcN 13*llcN 50],1,'real*4',0,'ieee-be');
    with open(sigma_S_path, 'rb') as fid:
        sigma_S = np.fromfile(fid, dtype=mform)
        sigma_S = sigma_S.reshape((siz[0], siz[1], siz[2]), order='F')
    sigma_S_90_pf, faces = patchface3D(llcN, llcN*13, 50, sigma_S , 2)

    # theta
    sigma_T_path = os.path.join(sigma_dir, 'Theta_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin')
    #sigma_T = readbin(fsigST{2},[llcN 13*llcN 50],1,'real*4',0,'ieee-be');
    with open(sigma_T_path, 'rb') as fid:
        sigma_T = np.fromfile(fid, dtype=mform)
        sigma_T = sigma_T.reshape((siz[0], siz[1], siz[2]), order='F')

    sigma_T_90_pf = patchface3D(llcN, llcN*13, 50, copy.deepcopy(sigma_T) , 2)
   
    # Prepare the nearest neighbor mapping
    F_llc90_WET_INS_SURF_XYZ_to_INDEX = make_F_llc90_WET_INS_SURF_XYZ_to_INDEX(AI_90, X_90, Y_90, Z_90, wet_ins_90_k)

    # verify that our little trick works in 4 parts of the earth
    deg2rad = np.pi/180
    """
    for i in np.arange(1,5):
        if i == 1:
            test_lat = 56
            test_lon = -40
        if i == 2:
            test_lat = 60
            test_lon = 10
        if i == 3:
            test_lat = -63.9420
            test_lon = -2.0790
        if i == 4:
            test_lat = -69
            test_lon = 60
    
        test_x, test_y, test_z = sph2cart(test_lon*deg2rad, test_lat*deg2rad, 1)
        testarr = np.asarray([test_x, test_y, test_z])
        test_ind = int(F_llc90_WET_INS_SURF_XYZ_to_INDEX(testarr)[0])

        print("===============================")
        print('original (line 1) vs closest (line 2) x,y,z')
        print("{} {} {}".format(X_90.flatten(order = 'F')[test_ind], Y_90.flatten(order = 'F')[test_ind], Z_90.flatten(order = 'F')[test_ind]))
        print("{} {} {}".format(test_x, test_y, test_z))
        
        print('original (line 1) vs closest (line 2) lat lon')
        print("{} {}".format(test_lat, test_lon))
        print("{} {}".format(lat_90.flatten(order = 'F')[test_ind], lon_90.flatten(order = 'F')[test_ind]))
    """

    # initialize remapped sigma field
    sigma_T_MITprof_z = []
    sigma_S_MITprof_z = []

    num_profs = len(MITprofs['prof_lat'])
    num_prof_depths = len(MITprofs['prof_depth'])

    # pull original weights
    orig_profTweight = MITprofs['prof_Tweight']
    if 'prof_S' in MITprofs:
        orig_profSweight = MITprofs['prof_Sweight']
    
    # ['mapping profiles to llc grid']
    prof_lon = MITprofs['prof_lon'].astype(np.float64)
    prof_lat = MITprofs['prof_lat'].astype(np.float64)
    prof_x, prof_y, prof_z = sph2cart(prof_lon*deg2rad, prof_lat*deg2rad, 1)
    
    # map a llc90 grid index to each profile.
    prof_arr = np.column_stack((prof_x, prof_y, prof_z))
    prof_llc90_cell_index = F_llc90_WET_INS_SURF_XYZ_to_INDEX(prof_arr).astype(int)
   
    # interp sigmas to the new vertical levels if it hasn't already been interpolated
    sigma_T_MITprof_z = interp_3D_to_arbitrary_z_levels(sigma_T, z_cen_90, MITprofs['prof_depth'], 0)
    # ['interpolated sigma T to new levels']
    sigma_T_MITprof_z_flat = np.reshape(sigma_T_MITprof_z, (90*1170, num_prof_depths), order = 'F')
    # ['finished interpolating and reshaping ']
    # Apply floor to sigma S where  sigma S >= 0
    if new_T_floor > 0:
        #ins = find(sigma_T_MITprof_z_flat >= 0);
        ins = np.where(sigma_T_MITprof_z_flat >=0)[0]
        sigma_T_MITprof_z_flat[ins] = np.maximum(new_T_floor, sigma_T_MITprof_z_flat[ins])
    
    if 'prof_S' in MITprofs and not sigma_S_MITprof_z:
        
        sigma_S_MITprof_z = interp_3D_to_arbitrary_z_levels(sigma_S, z_cen_90, MITprofs['prof_depth'], 0)
        #['interpolated sigma S to new levels']
        sigma_S_MITprof_z_flat = np.reshape(sigma_S_MITprof_z, (90*1170, num_prof_depths), order = 'F')
        #['finished interpolating and reshaping ']
        if new_S_floor > 0:
            # Apply floor to sigma S where  sigma S >= 0
            ins = np.where(sigma_S_MITprof_z_flat >= 0)[0]
            sigma_S_MITprof_z_flat[ins] = np.maximum(new_S_floor, sigma_S_MITprof_z_flat[ins])
   
    # map sigma field to profile points & make weights & apply weights
    tmp_sigma_T = sigma_T_MITprof_z_flat[prof_llc90_cell_index,:]
    #tmp_weight_T = 1./tmp_sigma_T.^2;
    tmp_weight_T = 1. / (tmp_sigma_T ** 2)
    MITprofs['prof_Tweight'] = tmp_weight_T

    if 'prof_S' in MITprofs:
        tmp_sigma_S = sigma_S_MITprof_z_flat[prof_llc90_cell_index,:]
        # tmp_weight_S = 1./tmp_sigma_S.^2;
        tmp_weight_S = 1. / (tmp_sigma_S ** 2)
        MITprofs['prof_Sweight'] = tmp_weight_S
    # ['weights applied']
        
    if new_T_floor > 0:
        if 'prof_Terr' in MITprofs:
            # Set the field that notes whatever floor has been applied to the sigmas;
            MITprofs['prof_Terr'] = np.zeros_like(MITprofs['prof_Terr']) + new_T_floor
    if new_S_floor > 0:
        if 'prof_Serr' in MITprofs:
            MITprofs['prof_Serr'] = np.zeros_like(MITprofs['prof_Serr']) + new_S_floor
    
    # SET WEIGHTS TO ZERO IF THEY CAME WITH ZERO.
    if respect_existing_zero_weights:
        raise Exception("translated but untested")
        # JUST FOR ACCOUNTING 
        num_zero_Tweights = np.sum(MITprofs['prof_Tweight'] == 0)

        # FIND DATA WITH ZERO WEIGHT
        zero_orig_weight_ins= np.where(orig_profTweight == 0)[0]
  
        # ['prof Tweight == 0  ' num2str(length(zero_orig_weight_ins))]
        # APPLY ZEROS TO WEIGHTS
        MITprofs['prof_Tweight'][zero_orig_weight_ins] = 0

        num_zero_Tweights_2 = np.sum(MITprof.prof_Tweight == 0)
        
        # ['num zero T weights: before/after orig zero weight mask']
        # [num_zero_Tweights num_zero_Tweights_2]
        
        if 'prof_S' in MITprofs:
            num_zero_Sweights = np.sum(MITprof['prof_Sweight'] == 0)
            zero_orig_weight_ins= np.where(orig_profSweight == 0)[0]
            
            #['prof Sweight == 0  ' num2str(length(zero_orig_weight_ins))]
            # APPLY ZEROS TO WEIGHTS
            MITprof['prof_Sweight'][zero_orig_weight_ins] = 0
            num_zero_Sweights_2 = np.sum(MITprof.prof_Sweight == 0)
            
            #['num zero S weights: before/after orig zero weight mask']
            #[num_zero_Sweights num_zero_Sweights_2]

    else:
        print("STEP 4: not respecting the zero weights of the original profiles")
    
def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    print("step 04: update_sigmaTS_on_prepared_profiles")
    update_sigmaTS_on_prepared_profiles(run_code, MITprofs, grid_dir)

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

    MITprofs_fp = '/home/sweet/Desktop/ECCO-Insitu-Ian/Python-Dest'
    MITprofs_fp = '/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/20190131_END_CHAIN'

    """
    if run_code != 90 or run_code != 270:
        raise Exception("Runcode has to be 90 or 270!")
    """
    
    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:    # moved from step 01
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 4)

    run_code = '20190126_do_not_respect_existing_weights'
    grid_dir = "hehe"

    main(run_code, MITprofs, grid_dir)
