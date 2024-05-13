import argparse
import glob
import os
import numpy as np
from scipy.interpolate import griddata
from tools import MITprof_read, intrep_check, load_llc270_grid, load_llc90_grid, sph2cart
from scipy import interpolate


def interp_2D_to_arbitrary_z_levels(orig_data_xz, orig_z_centers, new_z_centers):
    """
    This script interpolates some 2D structure with x,y,z to a new 2D
    structure with x,y, z_new
    """

    # force z values to be Nx1
    orig_z_centers = np.squeeze(orig_z_centers)
    new_z_centers = np.squeeze(new_z_centers)

    if orig_data_xz.shape[1] == orig_z_centers.size:
        orig_data_xz = orig_data_xz.T

    # data must be in rows = x (different points)
    #                 columns = z (different depths);
    # convert masked arr to non-masked type
    new_z_centers = new_z_centers.filled(np.nan)
    # Create interpolation function for all rows of orig_data_xz simultaneously
    interp_func = interpolate.interp1d(orig_z_centers, orig_data_xz.T, kind='linear', bounds_error=False, fill_value=np.nan)

    # Interpolate at new_z_centers
    new_data_xz = interp_func(new_z_centers)

    return new_data_xz

def interp_3D_to_arbitrary_z_levels(orig_data_xyz, orig_z_centers, new_z_centers):
    """
    This script interpolates some 3D structure with x,y,z to a new 3D
    structure with x,y, z_new
    """
    
    nx, ny, nz = orig_data_xyz.shape

    # convert shape to 2D
    orig_data_xz = np.reshape(orig_data_xyz, (nx*ny, nz), order='F')

    # interp using 2D script
    new_data_xz = interp_2D_to_arbitrary_z_levels(orig_data_xz, orig_z_centers, new_z_centers)

    nz_new = new_z_centers.shape[0]
    new_data_xyz = np.reshape(new_data_xz, (nx, ny, nz_new), order = 'F')

    return new_data_xyz

def make_llc90_z_map(z_top_90, z_bot_90):
    # NOTE: not used

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

def update_sigmaTS_on_prepared_profiles(run_code, MITprofs, grid_dir, sigma_dir):
    """
    Update MITprof objects with new T and S uncertainty fields 
    Input Parameters:
        run_code:
        20190126_do_not_respect_existing_weights
        20190126_respect_existing_weights
        20181202_do_not_respect_existing_weights
        20181202_respect_existing_weights
        
        MITprof: a single MITprof object
        grid_dir: directory path of grid to be read in
        sigma_dir: Path to Salt_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin and Theta_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin
        
    Output:
        Operates on MITprofs directly 
    """

    # new_T_floor, new_S_floor.  If greater than zero, ensure that the 
    # minimum uncertainty in T and S are at least as high as new_T_floor,
    # new_S_floor
    new_T_floor = 0
    new_S_floor = 0
            
    # respect_existing_zero_weights:  
    #   0 = no
    #   1 = yes
    if run_code == '20190126_do_not_respect_existing_weights':
        # use if the MITprof fields
        # do not have some zero weights
        respect_existing_zero_weights = 0
        new_S_floor = 0.005
        grid_code = 90
        
                  
    if run_code == '20190126_respect_existing_weights':
        # use if the MITprof fields
        # already have some zero weights
        respect_existing_zero_weights = 1
        new_S_floor = 0.005
            
    if run_code == '20181202_do_not_respect_existing_weights':
        # use if the MITprof fields
        # do not have some zero weights
        respect_existing_zero_weights = 0
            
    if run_code == '20181202_respect_existing_weights':
        #  use if the MITprof fields
        # already have some zero weights
        respect_existing_zero_weights = 1
    
    if grid_code == 90:
        wet_ins_90_k, X_90, Y_90, Z_90, AI_90, z_cen_90, lat_90, lon_90 = load_llc90_grid(grid_dir, 4)
    if grid_code == 270:
        wet_ins_90_k, X_90, Y_90, Z_90, AI_90, z_cen_90, lat_90, lon_90 = load_llc270_grid(grid_dir, 4)

    # salt
    sigma_S_path = os.path.join(sigma_dir, 'Salt_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin')
    llcN = 90
    mform = '>f4'
    siz = [llcN, 13*llcN, 50]
    with open(sigma_S_path, 'rb') as fid:
        sigma_S = np.fromfile(fid, dtype=mform)
        sigma_S = sigma_S.reshape((siz[0], siz[1], siz[2]), order='F')

    # theta
    sigma_T_path = os.path.join(sigma_dir, 'Theta_sigma_smoothed_method_02_masked_merged_capped_extrapolated.bin')
    with open(sigma_T_path, 'rb') as fid:
        sigma_T = np.fromfile(fid, dtype=mform)
        sigma_T = sigma_T.reshape((siz[0], siz[1], siz[2]), order='F')

    # verify that our little trick works in 4 parts of the earth
    deg2rad = np.pi/180
    xyz_wet = np.column_stack((X_90.flatten(order = 'F')[wet_ins_90_k[0]], Y_90.flatten(order = 'F')[wet_ins_90_k[0]], Z_90.flatten(order = 'F')[wet_ins_90_k[0]]))
    AI = AI_90.flatten(order = 'F')[wet_ins_90_k[0]] 
    intrep_check(xyz_wet, AI, X_90, Y_90, Z_90, lat_90, lon_90, 4)

    # initialize remapped sigma field
    sigma_T_MITprof_z = []
    sigma_S_MITprof_z = []

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
    xyz_wet = np.column_stack((X_90.flatten(order = 'F')[wet_ins_90_k[0]], Y_90.flatten(order = 'F')[wet_ins_90_k[0]], Z_90.flatten(order = 'F')[wet_ins_90_k[0]]))
    AI = AI_90.flatten(order = 'F')[wet_ins_90_k[0]] 
    prof_llc90_cell_index = griddata(xyz_wet, AI, np.column_stack((prof_x, prof_y, prof_z)), 'nearest').astype(int)
   
    # interp sigmas to the new vertical levels if it hasn't already been interpolated
    sigma_T_MITprof_z = interp_3D_to_arbitrary_z_levels(sigma_T, z_cen_90, MITprofs['prof_depth'])
    # ['interpolated sigma T to new levels']
    sigma_T_MITprof_z_flat = np.reshape(sigma_T_MITprof_z, (90*1170, num_prof_depths), order = 'F')
    # ['finished interpolating and reshaping ']
    # Apply floor to sigma S where  sigma S >= 0
    if new_T_floor > 0:
        ins = np.where(sigma_T_MITprof_z_flat >=0)[0]
        sigma_T_MITprof_z_flat[ins] = np.maximum(new_T_floor, sigma_T_MITprof_z_flat[ins])
    if 'prof_S' in MITprofs and not sigma_S_MITprof_z:
        
        sigma_S_MITprof_z = interp_3D_to_arbitrary_z_levels(sigma_S, z_cen_90, MITprofs['prof_depth'])
        sigma_S_MITprof_z_flat = np.reshape(sigma_S_MITprof_z, (90*1170, num_prof_depths), order = 'F')
        if new_S_floor > 0:
            # Apply floor to sigma S where  sigma S >= 0
            ins = np.where(sigma_S_MITprof_z_flat >= 0)[0]
            sigma_S_MITprof_z_flat[ins] = np.maximum(new_S_floor, sigma_S_MITprof_z_flat[ins])

    # map sigma field to profile points & make weights & apply weights
    tmp_sigma_T = sigma_T_MITprof_z_flat[prof_llc90_cell_index,:]
    tmp_weight_T = 1. / (tmp_sigma_T ** 2)
    MITprofs['prof_Tweight'] = tmp_weight_T
    if 'prof_S' in MITprofs:
        tmp_sigma_S = sigma_S_MITprof_z_flat[prof_llc90_cell_index,:]
        tmp_weight_S = 1. / (tmp_sigma_S ** 2)
        MITprofs['prof_Sweight'] = tmp_weight_S
    
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
    
def main(run_code, MITprofs, grid_dir, sigma_dir):

    print("step 04: update_sigmaTS_on_prepared_profiles")
    update_sigmaTS_on_prepared_profiles(run_code, MITprofs, grid_dir, sigma_dir)

if __name__ == '__main__':

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
    
    parser.add_argument("-s", "--sigma_dir", action= "store",
                help = "File path sigma salt and theta files." , dest= "sigma_dir",
                type = str, required= True)
    

    args = parser.parse_args()

    run_code = args.run_code
    grid_dir = args.grid_dir
    MITprofs_fp = args.MIT_dir
    sigma_dir = args.sigma_dir
    
    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 4)

    main(run_code, MITprofs, grid_dir, sigma_dir)
