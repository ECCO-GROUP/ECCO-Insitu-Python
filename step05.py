import argparse
import copy
import glob
import os
import numpy as np
from tools import MITprof_read, load_llc270_grid, load_llc90_grid

def update_gamma_factor_on_prepared_profiles(MITprofs, grid_dir, apply_gamma_factor, llcN):
    """
    Updates the MITprof profiles with a new sigma based 
    on whether we are applying or removing the 'gamma' factor 
    
    Input Parameters:
        run_code:

        llcN: corresponds to grid used
        apply_gamma_factor: 0 to remove, 1 to apply gamma to sigma
        MITprof: a single MITprof object
        grid_dir: directory path of grid to be read in

    Output:
        Operates on MITprofs directly 
    """

    #  load the RAC field only if we are applying a gamma factor
    #  when we remove a gamma factor the gamma value is already stored in the
    #  profile file.
    if apply_gamma_factor == 1:
        if llcN == 90:
            RAC_90_pf = load_llc90_grid(grid_dir, 5)
            RAC = copy.deepcopy(RAC_90_pf)
                
        if llcN == 270:
            RAC_270_pf = load_llc270_grid(grid_dir, 5)
            RAC = copy.deepcopy(RAC_270_pf)

    #  load weights
    tmpT = MITprofs['prof_Tweight']
    tmpT[np.where(tmpT < 0)[0]] = np.nan
    if 'prof_S' in MITprofs:
        tmpS = MITprofs['prof_Sweight']
        tmpS[np.where(tmpS <0)[0]] = np.nan
   
    # apply or remove the gamma factor
    if apply_gamma_factor == 1:
        # find ratio of each grid cell area to the maximum
        alpha = np.squeeze(RAC/ np.max(RAC))
        pp = MITprofs['prof_point'].astype(int)
        alpha_pp = alpha.flatten(order = 'F')[pp]
        MITprofs['prof_area_gamma'] = alpha_pp
        fac = copy.deepcopy(alpha_pp)
        
    elif apply_gamma_factor == 0:
        alpha_pp = MITprofs['prof_area_gamma']
        fac = 1/ alpha_pp
        MITprofs['prof_area_gamma'] = np.ones(MITprofs['prof_area_gamma'].size)
    
    # loop through k, apply fac to T and S weights
    for k in np.arange(MITprofs['prof_depth'].size):
        tmpTk = tmpT[:,k]
        tmpTk = tmpTk * fac
        tmpT[:,k] = tmpTk
        
        if 'prof_S' in MITprofs:
            tmpSk = tmpS[:,k]
            tmpSk = tmpSk * fac
            tmpS[:,k] = tmpSk

    MITprofs['prof_Tweight'] = tmpT
    if 'prof_S' in MITprofs:
        MITprofs['prof_Sweight'] = tmpS
    
def main(MITprofs, grid_dir, apply_gamma_factor, llcN):
    
    print("step05: update_gamma_factor_on_prepared_profiles")
    update_gamma_factor_on_prepared_profiles(MITprofs, grid_dir, apply_gamma_factor, llcN)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--grid_dir", action= "store",
                        help = "File path to 90/270 grids" , dest= "grid_dir",
                        type = str, required= True)
    
    parser.add_argument("-m", "--MIT_dir", action= "store",
                    help = "File path to NETCDF files containing MITprofs info." , dest= "MIT_dir",
                    type = str, required= True)
    

    args = parser.parse_args()

    grid_dir = args.grid_dir
    MITprofs_fp = args.MIT_dir

    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 5)

    llcN = 90
    apply_gamma_factor = 1          #   0 = remove gamma factor from sigma, 1 = apply gamma to sigma
                                    #   gamma factor is factor 1/sqrt(alpha), where alpha = area/max(area) of the grid cell area in which this profile is found.


    main(MITprofs, grid_dir, apply_gamma_factor, llcN)