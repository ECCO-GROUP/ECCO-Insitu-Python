import copy
import glob
import os
import numpy as np
import numpy.ma as ma
import datetime as dt
from step07 import count_profs_with_nonzero_weights
from step08 import extract_profile_subset_from_MITprof, update_remove_zero_T_S_weighted_profiles_from_MITprof
from tools import MITprof_read, sph2cart
import scipy.io as sio

def distmat(xy, varargin):

    # process inputs
    n, dims = xy.shape
    numels = n*n*dims
    opt = 2
    if numels > 5e4:
        opt = 3
    elif n < 20:
        opt = 1

    opt = max(1, min(4, round(abs(varargin))))

    # distance matrix calculation options
    if opt == 1: # half as many computations (symmetric upper triangular property)
        """
        [k,kk] = find(triu(ones(n),1));
        dmat = zeros(n);
        dmat(k+n*(kk-1)) = sqrt(sum((xy(k,:) - xy(kk,:)).^2,2));
        dmat(kk+n*(k-1)) = dmat(k+n*(kk-1));
        """
        print("uncoded")
    if opt == 2: # fully vectorized calculation (very fast for medium inputs)
        a = np.reshape(xy,(1 ,n ,dims), order = 'F') # 1 9 3
        b = np.reshape(xy,(n ,1 ,dims), order= 'F')
        dmat = np.sqrt(np.sum((a[np.zeros((n), dtype=int), :, :] - b[:, np.zeros((n), dtype= int),:])**2, axis = 2))
        
    if opt == 3: # partially vectorized (smaller memory requirement for large inputs)
        """
        dmat = zeros(n,n);
        for k = 1:n
            dmat(k,:) = sqrt(sum((xy(k*ones(n,1),:) - xy).^2,2));
        end
        """
        print("uncoded")
    if opt == 4: # another compact method, generally slower than the others
        """
        a = (1:n);
        b = a(ones(n,1),:);
        dmat = reshape(sqrt(sum((xy(b,:) - xy(b',:)).^2,2)),n,n);
        """
        print("uncoded")

    return dmat, opt

def update_decimate_profiles_subdaily_to_once_daily(run_code, MITprofs, grid_dir):
    """
    % This script decimates profiles with subdaily sampling at the same
    % location to once-daily sampling.
    """

    # SET INPUT PARAMETERS
    fillVal=-9999
    checkVal=-9000

    debug_code=0

    # distance_tolerance: radius within which profiles are considered to be at
    # the same location [in meters]
    distance_tolerance = 10e3

    # if there is more than one profile per day in a location, choose the one
    # that is closest in time to 'closest time' default is noon
    closest_time = 120000 # HHMMSS

    if run_code == '20181218_1':
        save_output_to_disk = 0
        distance_tolerance = 5e3 # meters
        closest_time = 120000  # noon
        method = 0
    if run_code == '20181218_2':
        save_output_to_disk = 0
        distance_tolerance = 5e3 # meters
        closest_time = 120000  # noon
        method = 1

    num_profs = len(MITprofs['prof_YYYYMMDD'])
        
    if method == 0:
        unique_prof_lat = []
        unique_prof_lon = [] 
        prof_num = 0

        profs_to_decimate = np.ones(MITprofs['prof_YYYYMMDD'].shape)

        deg2rad = np.pi/180
        X, Y, Z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 6357000)
                 
        while(np.sum(profs_to_decimate)) > 0:
            
            print(f'profs left {np.sum(profs_to_decimate)}')
            profs_left_ins = np.where(profs_to_decimate > 0)[0]
            num_profs_left = len(profs_left_ins)
           
            if num_profs_left > 0:

                prof_num = prof_num + 1
                ones_n = np.ones(num_profs_left)
           
                # consider the next on the list
                cur_i = np.where(profs_to_decimate == 1)[0][0]

                lat1 = MITprofs['prof_lat'][cur_i]
                lon1 = MITprofs['prof_lon'][cur_i]

                unique_prof_lon.append(lon1)
                unique_prof_lat.append(lat1)
 
                X_prof = X[cur_i]
                Y_prof = Y[cur_i]
                Z_prof = Z[cur_i]

                d = np.sqrt( (X_prof - X[profs_left_ins])**2 + (Y_prof - Y[profs_left_ins])**2 + (Z_prof - Z[profs_left_ins])**2)
    
                d[np.where(np.isnan(d))] = 0

                b = np.argsort(d)
                a = d[b]
      
                ins_close_sort = np.where(a < distance_tolerance)
                # the indexes of the points that are close on the
                # same day on the 'sorted distance' list
                ins_close = profs_left_ins[b[ins_close_sort]]
                num_close = len(ins_close)

                days = np.unique(MITprofs['prof_YYYYMMDD'][ins_close]).astype(int)
                num_days = len(days)

                num_in_day = np.zeros(num_days)
                ins_day = np.zeros(num_days)

                if num_days == 1:
                    ins_day_close = np.where(MITprofs['prof_YYYYMMDD'][ins_close] == days)
                    num_in_day = len(ins_day_close)
                    ins_day = ins_close[ins_day_close]
                    if len(ins_day) > 1:
                        bb = np.argsort(np.abs(MITprofs['prof_HHMMSS'][ins_day] - closest_time))
                        ins_closest_to_target_time = bb[0]
                        ins_to_decimate = ins_day[bb[1:]]
                        MITprofs['prof_Tweight'][ins_to_decimate,:] = 0
                        if 'prof_S' in MITprofs:
                            MITprofs['prof_Sweight'][ins_to_decimate,:] = 0
                        
                    profs_to_decimate[ins_day] = 0
                    
                else: 
                    for di in np.arange(days):
                        day = days[di]
                        # the indexes of the points that are 1) close and 2) on the
                        # same day on the 'ins_close' list
                        ins_day_close = np.where(MITprofs['prof_YYYYMMDD'][ins_close] == day)
                        num_in_day[di] = len(ins_day_close)

                        # the indexes of the points that are 1) close and 2) on the
                        # same day on the original list
                        ins_day[di] = ins_close[ins_day_close]
                        
                        if len(ins_day[di]) > 1:
                            bb = np.argsort(np.abs(MITprofs['prof_HHMMSS'][ins_day[di]] - closest_time))
                            ins_closest_to_target_time = bb[0]
                            ins_to_decimate = ins_day[di][bb[1:]]
                            MITprofs['prof_Tweight'][ins_to_decimate, :] = 0
                            if 'prof_S' in MITprofs:
                                MITprofs['prof_Sweight'][ins_to_decimate, :] = 0
  
                        profs_to_decimate[ins_day[di]] = 0 
    elif method == 1:
        
        deg2rad = np.pi/180
        
        X, Y, Z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 6357000)

        days = np.unique(MITprofs['prof_YYYYMMDD'])

        toss_set_all_day = {}
        toss_set_all = []
        total_toss = 0
        
        for di in np.arange(len(days)):

            ins_day = np.where(MITprofs['prof_YYYYMMDD'] == days[di])[0]
            n_di = len(ins_day)
            
            d, opt = distmat(np.stack((X[ins_day], Y[ins_day], Z[ins_day]), axis = 1), 2)
   
            keep_set = []
            toss_set  = []

            for p in np.arange(n_di):
                if ins_day[p] not in toss_set and ins_day[p] not in keep_set:
                    p_close = np.where(d[p,:] < distance_tolerance)[0]
                    p_close_ins_day = ins_day[p_close]

                    if len(p_close_ins_day) > 1:
                        b = np.argsort(np.abs(MITprofs['prof_HHMMSS'][p_close_ins_day]-120000))
                        toss_set = np.union1d(toss_set, p_close_ins_day[b[1:]])


            total_toss = total_toss + len(toss_set)
            toss_set_all = np.union1d(toss_set_all, toss_set)

        toss_set_all = toss_set_all.astype(int)
        MITprofs['prof_Tweight'][toss_set_all,:] = 0
        MITprofs['prof_Sweight'][toss_set_all,:] = 0
       
        update_remove_zero_T_S_weighted_profiles_from_MITprof('20181202', MITprofs , "nlasd")
    

    print('Num T and S weight > 0, post')
    print('{:>10} {:>10}'.format(np.sum(MITprofs['prof_Tweight'] > 0), np.sum(MITprofs['prof_Sweight'] > 0)))

    

def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    print("update_decimate_profiles_subdaily_to_once_daily")

    print('Num T and S weight > 0, pre')
    print('{:>10} {:>10}'.format(np.sum(MITprofs['prof_Tweight'] > 0), np.sum(MITprofs['prof_Sweight'] > 0)))

    update_decimate_profiles_subdaily_to_once_daily(run_code, MITprofs, grid_dir)

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
        MITprofs = MITprof_read(file, 10)

    run_code = '20181218_2'
    grid_dir = "hehe"

    # Convert all masked arrs to non-masked types
    for keys in MITprofs.keys():
        if ma.isMaskedArray(MITprofs[keys]):
            MITprofs[keys] = MITprofs[keys].filled(np.NaN)
    
    main(run_code, MITprofs, grid_dir)