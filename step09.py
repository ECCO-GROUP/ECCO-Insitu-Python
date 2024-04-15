import copy
import glob
import os
import numpy as np
import numpy.ma as ma
import datetime as dt
from step07 import count_profs_with_nonzero_weights
from step08 import extract_profile_subset_from_MITprof
from tools import MITprof_read
import scipy.io as sio

def count_nonzero_weight_profiles_with_depth(MITprof):
    """
    This script returns the number of profiles with nonzero weights as a 
    function of depth.  Separate values for temperature and salinity.
    """
    depths = MITprof['prof_depth']

    y = np.zeros(MITprof['prof_Tweight'].shape)
    y[np.where(MITprof['prof_Tweight'] > 0)] = 1
    T_counts = np.sum(y, axis = 0)

    if 'prof_S' in MITprof:
        y = np.zeros(MITprof['prof_Sweight'].shape)
        y[np.where(MITprof['prof_Sweight'] > 0)] = 1
        S_counts = np.sum(y, axis = 0)
    else:
        S_counts = np.zeros(depths.shape)
    
    return depths, T_counts, S_counts


def update_remove_extraneous_depth_levels(run_code, MITprofs, grid_dir):

    dd_pre, tt, ss = count_nonzero_weight_profiles_with_depth(MITprofs)

    nonzero_TTs = np.where(tt > 0)[0]

    if len(nonzero_TTs) > 0:
        max_ziT = nonzero_TTs[-1]
    else:
        max_ziT = 0
    
    nonzero_SSs = np.where(ss > 0)[0]
    if len(nonzero_SSs) > 0:
        max_ziS = nonzero_SSs[-1]
    else:
        max_ziS = 0

    max_zi = max(max_ziT, max_ziS)

    print(f'\tmax zi {max_zi}, depth {dd_pre[max_zi]}\n')
    
    if max_zi < tt.shape[0]:
        
        print(f'max zi is less than size (TT,1) {max_zi} {tt.shape[0]}')
        print('extracting subsets in depth \n')

        MITprofs_new = extract_profile_subset_from_MITprof(MITprofs, [], np.arange(max_zi + 1))
    
        print(f"Dimension of prof_T {MITprofs_new['prof_T'].shape}")

        MITprofs.update(MITprofs_new)
    else:
        print(f'max zi is the same as prof depth, no need to cut out missing depth levels')

   
def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    print("update_remove_extraneous_depth_levels")
    update_remove_extraneous_depth_levels(run_code, MITprofs, grid_dir)

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
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 9)

    run_code = '20181202'
    grid_dir = "hehe"

    # Convert all masked arrs to non-masked types
    for keys in MITprofs.keys():
        if ma.isMaskedArray(MITprofs[keys]):
            MITprofs[keys] = MITprofs[keys].filled(np.NaN)
    
    main(run_code, MITprofs, grid_dir)