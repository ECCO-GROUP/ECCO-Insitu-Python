import argparse
import glob
import os
from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sio
import netCDF4 as nc
from scipy.interpolate import griddata
from tools import MITprof_read, sph2cart

def update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles(run_code, MITprofs):


    fillVal=-9999
    # climatolgy dir and filenames
    TS_clim_dir = ''


    if run_code == '20181202_NODC':
            
        TS_clim_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology'
        TS_clim_fname = 'WOA13_v2_TS_clim_merged_with_potential_T.nc'
        
        TS_clim_filename = os.path.join(TS_clim_dir, TS_clim_fname)
        TS_data = nc.Dataset(TS_clim_filename)

        #T_clim = WOA_2013_v2_clim.potential_T_monthly
        T_clim = TS_data.variables['potential_T_monthly'][:].filled(np.nan)
        #S_clim = WOA_2013_v2_clim.S_monthly
        S_clim = TS_data.variables['S_monthly'][:].filled(np.nan)

        #lon = WOA_2013_v2_clim.lon
        lon = TS_data.variables['lon'][:]
        #lat = WOA_2013_v2_clim.lat
        lat = TS_data.variables['lat'][:]
        
        #num_clim_depths = length(WOA_2013_v2_clim.depth.data)
        clim_depths =  TS_data.variables['depth'][:]
        num_clim_depths = len(clim_depths)

    """  
    # MATLAB CODE
    elif run_code == '20181203_ARGO':
        save_output_to_disk = 0
        make_figs = 0
        
        TS_clim_dir = ['/home/ifenty/data/observations/TS_climatology/WOA_2013_V2/1995-2014-merged']
        TS_clim_fname = ['WOA13_v2_TS_clim_potential_T_argo55_vertical_levels.mat'];

        cd(TS_clim_dir);
        load(TS_clim_fname);

        T_clim = WOA_2013_v2_clim_55z.potential_T_monthly;
        S_clim = WOA_2013_v2_clim_55z.S_monthly;
        
        lon = WOA_2013_v2_clim_55z.lon;
        lat = WOA_2013_v2_clim_55z.lat;
        
        num_clim_depths = length(WOA_2013_v2_clim_55z.depth);
           
    elif run_code == '20181203_97z':
            
        save_output_to_disk = 0;
        make_figs = 0;
        
        % the climatology
        TS_clim_dir = ['/home/ifenty/data/observations/TS_climatology/WOA_2013_V2/1995-2014-merged']
        TS_clim_fname = ['WOA13_v2_TS_clim_potential_T_SIO_2018_97_vertical_levels.mat'];

        cd(TS_clim_dir);
        load(TS_clim_fname);

        T_clim = WOA_2013_v2_clim_97z.potential_T_monthly;
        S_clim = WOA_2013_v2_clim_97z.S_monthly;
        
        lon = WOA_2013_v2_clim_97z.lon;
        lat = WOA_2013_v2_clim_97z.lat;
        
        num_clim_depths = length(WOA_2013_v2_clim_97z.depth);
    """

    # mesh the climatology lon and lats
    lon_woam, lat_woam = np.meshgrid(lon.data, lat.data)
    deg2rad = np.float64(np.pi/180.0)
 
    # POINTS TO USE ARE THOSE POINTS WITH VALID DATA at the surface
    # good_clim_ins=find(~isnan(squeeze(S_clim(1,1,:,:))));
    subset = S_clim[0,0].flatten(order= 'F')
    good_clim_ins = np.where(~np.isnan(subset))[0]

    lon_woam = lon_woam.flatten(order='F').astype(np.float64)
    lat_woam = lat_woam.flatten(order='F').astype(np.float64)

    X_woa, Y_woa, Z_woa = sph2cart(lon_woam[good_clim_ins]*deg2rad, lat_woam[good_clim_ins]*deg2rad, 1)
    AI = np.arange(0,X_woa.size).astype(np.float64)
    
    # these are the x,y,z coordinates of all points in the climatology
    #xyz= [X_woa(:) Y_woa(:) Z_woa(:)];
    xyz = np.column_stack((X_woa, Y_woa, Z_woa)).astype(np.float64)

    #'making scattered interpolant'
    #F  = scatteredInterpolant(xyz, AI(:),'nearest');
    """
    NOTE: Don't need to make F rn
    F = griddata(xyz, AI, (X_woa, Y_woa, Z_woa), method='nearest')
    F = F.astype(int)
    """
    """
    # verify that our little trick works in 4 parts of the earth
    for i in np.arange(1,5):
        if i == 1:
            test_lat = 56
            test_lon = -40
        if i == 2:
            test_lat = 60
            test_lon = 10
        if i == 3:
            test_lat = -60
            test_lon = -120
        if i == 4:
            test_lat = -69
            test_lon = 60
        test_x, test_y, test_z = sph2cart(test_lon*deg2rad, test_lat*deg2rad, 1)
        
        F = griddata(xyz, AI, (test_x, test_y, test_z), method='nearest')
        test_ind = F.astype(int)
        print("{} {} {} | {} {} {}".format(X_woa[test_ind], Y_woa[test_ind], Z_woa[test_ind], test_x, test_y, test_z))
        print("{} {} | {} {}".format(lat_woam[good_clim_ins[test_ind]], lon_woam[good_clim_ins[test_ind]], test_lat, test_lon))
        print("===========================")
    """
    # for ilist = 1:length(MITprofs)
    #     fprintf(['MITPROF ' num2str(ilist) '  of  '  num2str(length(MITprofs)) '\n'])
    #     MITprof = MITprofs{ilist};
    num_profs = len(MITprofs['prof_lat'])
    num_prof_depths = len(MITprofs['prof_depth'])

    # bad data = profX_flag > 0 --> 0 weight, valid climatology value.
    #  no data => profX == -9999, valid value in climatology, 0 in weight
    
    # determine the month for every profile
    #tmp = str(MITprofs['prof_YYYYMMDD'])
    #prof_month = num(tmp(:,5:6));
    prof_month = ((MITprofs['prof_YYYYMMDD'] % 10000) // 100).astype(int)

    # 'mapping profiles to x,y,z'
    MITprofs['prof_lon'] = MITprofs['prof_lon'].astype(np.float64)
    MITprofs['prof_lat'] = MITprofs['prof_lat'].astype(np.float64)
    prof_x, prof_y, prof_z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 1)
    
    # map a climatology grid index to each profile.
    #prof_clim_cell_index = F(prof_x, prof_y, prof_z);
    prof_clim_cell_index = griddata(xyz, AI, (prof_x, prof_y, prof_z), method='nearest')
    prof_clim_cell_index = prof_clim_cell_index.astype(int)
    """
    NOTE: code to check prof_x/prof_y/prof_z difference
    #prof_x = np.around(prof_x, decimals=27)
    import scipy.io as sio
    mat_contents = sio.loadmat('/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology/prof_x.mat')
    mat_contents['prof_x'] = np.squeeze(mat_contents['prof_x'])
    ticker = 0
    for a in np.arange(prof_x.size):
        if prof_x[a] != mat_contents['prof_x'][a]:
            print(prof_x[a])
            print(mat_contents['prof_x'][a])
            print("--------")
            ticker = ticker + 1
    raise Exception(ticker)
    
    """
    """
    NOTE: code to check for prof_cell... intrep difference
    import scipy.io as sio
    mat_contents = sio.loadmat('/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology/cells.mat')
    mat_contents['prof_clim_cell_index'] = np.squeeze(mat_contents['prof_clim_cell_index'])
    mat_contents['prof_clim_cell_index'] = mat_contents['prof_clim_cell_index'] - 1
    ticker = 0
    for a in np.arange(prof_clim_cell_index.size):
        if prof_clim_cell_index[a] != mat_contents['prof_clim_cell_index'][a]:
            if not (np.isnan(prof_clim_cell_index[a]) and np.isnan(mat_contents['prof_clim_cell_index'][a])):
                print(prof_clim_cell_index[a])
                print(mat_contents['prof_clim_cell_index'][a])    print(np.where(np.isnan(prof_clim_S))[0].size)
                print("{}".format(a))
                ticker = ticker + 1
                print("--------")

    raise Exception(ticker)
    """   
    # NOTE: importing from matlab the interp -> so we can continue w/ coding rest of pipeline
    import scipy.io as sio
    mat_contents = sio.loadmat('/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology/cells.mat')
    mat_contents['prof_clim_cell_index'] = np.squeeze(mat_contents['prof_clim_cell_index'])
    mat_contents['prof_clim_cell_index'] = mat_contents['prof_clim_cell_index'] - 1
    prof_clim_cell_index = mat_contents['prof_clim_cell_index']
            
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
    """
    # NOTE: final check to make sure prof_clim_T + prof_clim_S are the same as matlab
    import scipy.io as sio
    mat_contents = sio.loadmat('/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/TS Climatology/prof_clim_T_array.mat')

    for a in np.arange(32364):
        for b in np.arange(137):
            if prof_clim_T[a][b] != mat_contents['prof_clim_T'][a][b]:
                if (np.isnan(prof_clim_S[a][b]) and (not np.isnan(mat_contents['prof_clim_T'][a][b]))) or (not np.isnan(prof_clim_T[a][b]) and (np.isnan(mat_contents['prof_clim_T'][a][b]))):
                    print(prof_clim_T[a][b])
                    print(mat_contents['prof_clim_T'][a][b])
                    print("{} {}".format(a, b))
                    print("-------------")
                
                if not (np.isnan(prof_clim_T[a][b]) and np.isnan(mat_contents['prof_clim_T'][a][b])):
                    print(prof_clim_T[a][b])
                    print(mat_contents['prof_clim_T'][a][b])
                    print("{} {}".format(a, b))
                    print("-------------")
    """
    
def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    print("step 03: update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles")
    update_monthly_mean_TS_clim_WOA13v2_on_prepared_profiles(run_code, MITprofs)

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
        MITprofs = MITprof_read(file, 3)

    run_code = '20181202_NODC'
    grid_dir = "hehe"

    main(run_code, MITprofs, grid_dir)
