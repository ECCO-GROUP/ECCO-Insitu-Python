import argparse
import glob
import os
import numpy as np
import scipy.io as sio
import netCDF4 as nc
from scipy.interpolate import griddata
from tools import MITprof_read, intrep_check, sph2cart

def update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles(TS_clim_dir, MITprofs):
    """
    Assigns the WOA13 T and S climatology values to MITprof objects. 

    Input Parameters:
        TS_clim_dir: Path to WOA13_v2_TS_clim_merged_with_potential_T.nc
        MITprof: a single MITprof object

    Output:
        Operates on MITprofs directly 
    
    """

    fillVal=-9999

    TS_clim_fname = 'WOA13_v2_TS_clim_merged_with_potential_T.nc'
    TS_clim_filename = os.path.join(TS_clim_dir, TS_clim_fname)
    TS_data = nc.Dataset(TS_clim_filename)

    T_clim = TS_data.variables['potential_T_monthly'][:].filled(np.nan)
    S_clim = TS_data.variables['S_monthly'][:].filled(np.nan)

    lon = TS_data.variables['lon'][:]
    lat = TS_data.variables['lat'][:]
    
    clim_depths =  TS_data.variables['depth'][:]
    num_clim_depths = len(clim_depths)

    # mesh the climatology lon and lats
    lon_woam, lat_woam = np.meshgrid(lon.data, lat.data)
    deg2rad = np.float64(np.pi/180.0)
 
    # POINTS TO USE ARE THOSE POINTS WITH VALID DATA at the surface
    subset = S_clim[0,0].flatten(order= 'F')
    good_clim_ins = np.where(~np.isnan(subset))[0]

    lon_woam = lon_woam.flatten(order='F').astype(np.float64)
    lat_woam = lat_woam.flatten(order='F').astype(np.float64)

    X_woa, Y_woa, Z_woa = sph2cart(lon_woam[good_clim_ins]*deg2rad, lat_woam[good_clim_ins]*deg2rad, 1)
    AI = np.arange(0,X_woa.size).astype(np.float64)
    
    # these are the x,y,z coordinates of all points in the climatology
    xyz = np.column_stack((X_woa, Y_woa, Z_woa)).astype(np.float64)

    # verify that our little trick works in 4 parts of the earth
    intrep_check(xyz, AI, X_woa, Y_woa, Z_woa, lat_woam, lon_woam, 3, good_clim = good_clim_ins)

    num_profs = len(MITprofs['prof_lat'])
    num_prof_depths = len(MITprofs['prof_depth'])

    # bad data = profX_flag > 0 --> 0 weight, valid climatology value.
    #  no data => profX == -9999, valid value in climatology, 0 in weight
    
    # determine the month for every profile
    prof_month = ((MITprofs['prof_YYYYMMDD'] % 10000) // 100).astype(int)

    # 'mapping profiles to x,y,z'
    MITprofs['prof_lon'] = MITprofs['prof_lon'].astype(np.float64)
    MITprofs['prof_lat'] = MITprofs['prof_lat'].astype(np.float64)
    prof_x, prof_y, prof_z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 1)
    
    # map a climatology grid index to each profile.
    prof_clim_cell_index = griddata(xyz, AI, (prof_x, prof_y, prof_z), method='nearest')
    prof_clim_cell_index = prof_clim_cell_index.astype(int)
    
    # go through each z level in the profile array
    # set the default climatology value to be fillVal (-9999)
    prof_clim_T = np.ones((num_profs, num_prof_depths)) * fillVal
    prof_clim_S = np.ones((num_profs, num_prof_depths)) * fillVal

    for k in np.arange(min(num_prof_depths, num_clim_depths)):
        T_clim_k = T_clim[:,k,:,:]
        S_clim_k = S_clim[:,k,:,:]
        
        # get the T and S at each profile point at this depth level
        for m in np.arange(12):
            T_clim_mk = T_clim_k[m, :, :]
            S_clim_mk = S_clim_k[m, :, :]
    
            T_clim_mk = T_clim_mk.flatten(order = 'F')[good_clim_ins]
            S_clim_mk = S_clim_mk.flatten(order = 'F')[good_clim_ins]

            profs_in_month = np.where(prof_month == m + 1)[0]
            prof_clim_cell_index_m = prof_clim_cell_index[profs_in_month]

            prof_clim_T_tmp = T_clim_mk[prof_clim_cell_index_m]
            prof_clim_S_tmp = S_clim_mk[prof_clim_cell_index_m]

            tmp_T = prof_clim_T[:,k]
            tmp_T[profs_in_month] = prof_clim_T_tmp

            tmp_S = prof_clim_S[:,k]
            tmp_S[profs_in_month] = prof_clim_S_tmp
            
            prof_clim_T[:,k] = tmp_T
            prof_clim_S[:,k] = tmp_S
    

    # NOTE: print(np.where(np.isnan(prof_clim_S))[0].size)
    # Fill -9999 for NaNs

    prof_clim_S_temp = prof_clim_S.flatten(order= 'F')
    prof_clim_S_temp[np.where(np.isnan(prof_clim_S_temp))[0]]= fillVal
    prof_clim_S = prof_clim_S_temp.reshape((prof_clim_S.shape), order='F')

    prof_clim_T_temp = prof_clim_T.flatten(order= 'F')
    prof_clim_T_temp[np.where(np.isnan(prof_clim_T_temp))[0]]= fillVal
    prof_clim_T = prof_clim_T_temp.reshape((prof_clim_T.shape), order='F')
    
    MITprofs['prof_Tclim'] = prof_clim_T
    MITprofs['prof_Sclim'] = prof_clim_S

def main(TS_clim_dir, MITprofs):

    print("step 03: update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles")
    update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles(TS_clim_dir, MITprofs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--TS_clim_dir", action= "store",
                        help = "Directory to TS Clim file" , dest= "ts_dir",
                        type = int, required= True)
    
    parser.add_argument("-m", "--MIT_dir", action= "store",
                    help = "File path to NETCDF files containing MITprofs info." , dest= "MIT_dir",
                    type = str, required= True)
    

    args = parser.parse_args()

    TS_clim_dir = args.ts_dir
    MITprofs_fp = args.MIT_dir

    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(file, 3)
    
    main(TS_clim_dir, MITprofs)
