import argparse
import glob
import os
import numpy as np
import numpy.ma as ma
from step07 import count_profs_with_nonzero_weights
from tools import MITprof_read

def extract_profile_subset_from_MITprof(MITprofs, prof_ins, prof_depth_ins):
    """
    % this script extracts a subset of profiles from an MITprof object.
    % it returns this subset as a new MITprof object

    % --- arguments --- 
    % MITprof : a MITprof object

    % prof_ins : a list of profiles to extract from MITprof
    %            if empty then we take them all

    % prof_depth_ins : a list of depth indices to extract from the MITprof
    %                  if empty then we take them all.
    """
    # create an empty dictionary
    MITprofSub = {}

    # the number of profiles is the length of any one of the 1D fields of
    # MITprof.  I pick the date field arbitrarily.
    num_profs = len(MITprofs['prof_YYYYMMDD'])
    num_depths = len(MITprofs['prof_depth'])

    # check whether an empty set of profile indices was passed
    if len(prof_ins) == 0:
        prof_ins = np.arange(num_profs)
        print('including all profiles')
    else:
        print('subsetting some profiles')
    # check whether an empty set of profile depth indices was passed
    if len(prof_depth_ins) == 0:
        prof_depth_ins = np.arange(num_depths)
        print('including all depths')
    else:
        print('subsetting depths')
    
    # loop through every field in MITprof
    for key in MITprofs.keys():
        # pull the values of this filed
        x = MITprofs[key]
        # determine the dimensions of the field
        sz = x.shape
        # if the length of the first dimension of the object is the number
        # of profiles then this is a object that we need to subset
        if sz[0] == num_profs:
            if len(sz) == 1 or sz[1] == 1:
                # if it's a 1D object then just directly pull the subset
                tmp = x[prof_ins]
            elif len(sz) == 2 and sz[1] == num_depths:
                # otherwise it's 2D.  then we pull the subset from the first
                # dimension (second dimension is depth)
                tmp = x[prof_ins[:, np.newaxis], prof_depth_ins]
            elif len(sz) == 2 and sz[1] > 1:
                tmp = x[prof_ins, :]
            else:
                raise Exception('Do not know what to do with {} of size {}'.format(key, x.shape))
                
        # if the length of the first dimension is the number of depths
        # then subset to the prof_depth_ins
        elif sz[0] == num_depths:
            tmp = x[prof_depth_ins]
        else:
            # if the object length is not the same as the number of profiles
            # then we'll just copy it over 
            tmp = x

        MITprofSub.update({key: tmp})
   

    return MITprofSub
    

def update_remove_zero_T_S_weighted_profiles_from_MITprof(MITprofs):
    """
    Remove profiles that whose T and S weights are all zero from 
    from MITprof structures
    
    Input Parameters:
        MITprof: a single MITprof object

    Output:
        Operates on MITprofs directly 
    """     

    total_Tweight = np.sum(MITprofs['prof_Tweight'])
    total_Sweight = np.sum(MITprofs['prof_Sweight'])

    nnt_orig, nns_orig, nnts_orig, np_orig, zwti_orig, zwsi_orig, zwtsi_orig,nzwti_orig, nzwsi_orig, nzwtsi_orig = count_profs_with_nonzero_weights(MITprofs)
        
    print(f'\tnum profs: {np_orig} \n\tnum nonzero T: {nnt_orig} \n\tnum nonzero S: {nns_orig} \n\tnum nonzero TS: {nnts_orig}')
    
    num_nan_profs = np.where(np.isnan(MITprofs['prof_Tweight'].flatten(order = 'F')))[0]
    num_profs_to_remove = np_orig - len(nzwtsi_orig)

    print(f'\t# profs to nix: {num_profs_to_remove}')
    print(f'\tTotal T weight: {total_Tweight}')
    print(f'\tTotal S weight: {total_Sweight}')

    if len(num_nan_profs) > 0:
        raise Exception('you have nans in your weights, this should never happen')

    
    if len(nzwtsi_orig) > 0:
        if num_profs_to_remove > 0:
            
            MITprof_new = extract_profile_subset_from_MITprof(MITprofs, nzwtsi_orig, [])

            total_Tweight = np.sum(MITprof_new['prof_Tweight'])
            total_Sweight = np.sum(MITprof_new['prof_Sweight'])

            nnt_new, nns_new, nnts_new, np_new, zwti_new, zwsi_new, zwtsi_new, nzwti_new, nzwsi_new, nzwtsi_new = count_profs_with_nonzero_weights(MITprof_new)

            print(f'\tnum profs: {np_new} \n\tnum nonzero T: {nnt_new} \n\tnum nonzero S: {nns_new} \n\tnum nonzero TS: {nnts_new}')

            num_profs_to_remove = np_new - len(nzwtsi_new)

            print(f'\t# profs to nix: {num_profs_to_remove}')
            print(f'\tTotal T weight: {total_Tweight}')
            print(f'\tTotal S weight: {total_Sweight}')

            # make sure subsetting worked
            a1 = np.nansum(np.nansum((MITprofs['prof_S'] - MITprofs['prof_Sclim'])**2 * MITprofs['prof_Sweight']))
            a2 = np.nansum(np.nansum((MITprof_new['prof_S'] - MITprof_new['prof_Sclim'])**2 * MITprof_new['prof_Sweight']))

            b1 = np.nansum(np.nansum((MITprofs['prof_T'] - MITprofs['prof_Tclim'])**2 * MITprofs['prof_Tweight']))
            b2 = np.nansum(np.nansum((MITprof_new['prof_T'] - MITprof_new['prof_Tclim'])**2 * MITprof_new['prof_Tweight']))

            print('\n\ttotal T cost old/new {:10.30f} / {:10.30f} \n'.format(b1, b2))
            print('\ttotal S cost old/new {:10.30f} / {:10.30f} \n'.format(a1, a2))
            diff = np.abs(b1 - b2)
            print('difference b/w b1 + b2: {:10.10e}'.format(diff))
    
            if a1 != a2:
                print('profile s costs difference is small')
                if np.abs(a1 - a2) > 1:
                    raise Exception('profile s costs is big')

            if b1 != b2:
                print('profile t costs difference is small')
                if np.abs(b1 - b2) > 1:
                    raise Exception('profile t costs is big')

        else:
            print('no bad profs!')
            MITprof_new = MITprofs 
    
    else: # no good profs left
        print('no good profs left, making empty MITprof_new')
        MITprof_new = []
    
    MITprofs.update(MITprof_new)
        
def main(MITprofs):

    print("update_remove_zero_T_S_weighted_profiles_from_MITprof")
    update_remove_zero_T_S_weighted_profiles_from_MITprof(MITprofs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--MIT_dir", action= "store",
                    help = "File path to NETCDF files containing MITprofs info." , dest= "MIT_dir",
                    type = str, required= True)
    

    args = parser.parse_args()

    run_code = args.run_code
    grid_dir = args.grid_dir
    MITprofs_fp = args.MIT_dir

    MITprofs_fp = '/home/sweet/Desktop/ECCO-Insitu-Ian/Python-Dest'
    MITprofs_fp = '/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/20190131_END_CHAIN'

    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 8)

    # Convert all masked arrs to non-masked types
    for keys in MITprofs.keys():
        if ma.isMaskedArray(MITprofs[keys]):
            MITprofs[keys] = MITprofs[keys].filled(np.NaN)
    
    main(MITprofs)