import argparse
import glob
import os
from tools import load_llc90_grid, load_llc270_grid, sph2cart, MITprof_read
from scipy.interpolate import griddata
import numpy as np

def make_F_llc90_ALL_INS_SURF_XYZ_to_INDEX(bathy_90, X_90, Y_90, Z_90):

    ## Prepare the nearest neighbor mapping
    # do the big mapping.
    #AI_90(1:end) = 1:length(bathy_90(:));
    AI_90 = np.arange(bathy_90.size)
    # these are the x,y,z coordinates of the 'good' cells in llc90
    # xyz= [X_90(: ) Y_90(: ) Z_90(: )];
    X_90 = X_90.flatten(order = 'F')
    Y_90 = Y_90.flatten(order = 'F')
    Z_90 = Z_90.flatten(order = 'F')
    xyz = np.column_stack((X_90, Y_90, Z_90))

    # these AI points are in no domain, they are just a list of points
    # a mapping between these x,y,z points and the AI_90 index
    # you provide an x,y,z and F_llc90_XYZ_to_INDEX provides you with an
    # index in the global llc90 file correpsonding with the nearest
    # neighbor.

    #F_llc90_ALL_INS_SURF_XYZ_to_INDEX = griddata(xyz, AI_90, xyz, method='nearest')
    #F_grid_PF_XYZ_to_INDEX = griddata(model_xyz, AI_grid_pf, (prof_x, prof_y, prof_z), method='nearest')
    F_llc90_ALL_INS_SURF_XYZ_to_INDEX = TriScatteredInterp(xyz, AI_90,'nearest')

    return F_llc90_ALL_INS_SURF_XYZ_to_INDEX

def make_F_llc270_ALL_INS_SURF_XYZ_to_INDEX(X_270, Y_270, Z_270, bathy_270, good_ins_270):
    #AI(1:end) = 1:length(bathy_270(:));
    AI_270 = np.arange(bathy_270.size)
    AI_270 = AI_270[np.unravel_index(good_ins_270, AI_270.shape, order = 'F')]
    # xyz= [X_270(good_ins_270) Y_270(good_ins_270) Z_270(good_ins_270)];
    good_ins_270_index = np.unravel_index(good_ins_270, X_270.shape, order = 'F')
    X_270 = X_270[good_ins_270_index].flatten(order = 'F')
    Y_270 = Y_270[good_ins_270_index].flatten(order = 'F')
    Z_270 = Z_270[good_ins_270_index].flatten(order = 'F')
    xyz = np.column_stack((X_270, Y_270, Z_270))

    F_llc270_ALL_INS_SURF_XYZ_to_INDEX = TriScatteredInterp(xyz, AI_270,'nearest')

    return F_llc270_ALL_INS_SURF_XYZ_to_INDEX


class TriScatteredInterp:
    """
    TriScatteredIntrep function to mimick the behavior of function in Matlab
    """
    def __init__(self, X, V, method='nearest'):
        self.X = X
        self.V = V
        self.method = method
        #self.interpolated_values = griddata(self.X, self.V, self.X, method=self.method)
    """
    Python's griddata will return an array of interpolated values for all 
    numbers.This allows 3 points to be passed and interpolated.
    """
    def __call__(self, points):
        return griddata(self.X, self.V, points, method=self.method)
    
def update_spatial_bin_index_on_prepared_profiles(run_code, MITprofs, grid_dir):

    make_figs = 0

    if run_code == 10242 or run_code == 2562:
            
        # the bin data have to be projected to the model grid;
        bin_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents/grid_llc90/sphere_point_distribution'
        bin_file_1 = os.path.join(bin_dir, 'llc090_sphere_point_n_10242_ids.bin')
        bin_file_2 =  os.path.join(bin_dir, 'llc090_sphere_point_n_02562_ids.bin')

        bin_llcN = 90

        # was stored in list?
        prof_bin_name_1 = 'prof_bin_id_a'
        prof_bin_name_2 = 'prof_bin_id_b'
        
        save_output_to_disk = 0
        num_MITprofs = len(MITprofs)
            
    """
    cd(bin_dir)
    num_bins = length(bin_file);
    for i = 1:num_bins
        fprintf(['bin ' padzero(i,2) '\n'])
        bin{i} = readbin([bin_dir bin_file{i}], [bin_llcN 13*bin_llcN 1 1],1,'real*4',0,'ieee-be');
        stats(bin{i})
    end
    """
    # read binary files
    siz = [bin_llcN, 13*bin_llcN, 1, 1]
    typ = 1
    prec ='float32'  # real*4 corresponds to float32
    skip = 0 
    mform = '>f4' # 'ieee-be' corresponds to f4
    # NOTE: if bin_llcN = 270 these files dont work lol
    with open(bin_file_1, 'rb') as fid:
        bin_1 = np.fromfile(fid, dtype=mform)
        bin_1 = bin_1.reshape((siz[0], np.prod(siz[1:])), order='F')
        bin_1 = bin_1.reshape((siz[0], siz[1], siz[2]))

    with open(bin_file_2, 'rb') as fid:
        bin_2 = np.fromfile(fid, dtype=mform)
        bin_2 = bin_2.reshape((siz[0], np.prod(siz[1:])), order='F')
        bin_2 = bin_2.reshape((siz[0], siz[1], siz[2]))

    ## Prepare the nearest neighbor mapping

    if bin_llcN  == 90:
  
        lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90 = load_llc90_grid(grid_dir)
        """
        Cant load matlab file
        cd(llc90_grid_dir)
        NOTE: llc90_grid_dir = "C:\\Users\\szswe\\Desktop\\grid_llc90"
        if exist('F_llc90_ALL_INS_SURF_XYZ_to_INDEX.mat','file')
            ['loading F']
            load('F_llc90_ALL_INS_SURF_XYZ_to_INDEX');
            ['loaded F']
        else
            make_F_llc90_ALL_INS_SURF_XYZ_to_INDEX;
        end
        """

        F = make_F_llc90_ALL_INS_SURF_XYZ_to_INDEX(bathy_90, X_90, Y_90, Z_90)
        X = X_90.flatten(order = 'F')
        Y = Y_90.flatten(order = 'F')
        Z = Z_90.flatten(order = 'F')
        lon_llc = lon_90.flatten(order = 'F')
        lat_llc = lat_90.flatten(order = 'F')
    
    if bin_llcN  == 270:
        lon_270, lat_270, blank_270, wet_ins_270_k, X_270, Y_270, Z_270, bathy_270, good_ins_270 = load_llc270_grid(grid_dir)
        F = make_F_llc270_ALL_INS_SURF_XYZ_to_INDEX(X_270, Y_270, Z_270, bathy_270, good_ins_270)
        X = X_270.flatten(order = 'F')
        Y = Y_270.flatten(order = 'F')
        Z = Z_270.flatten(order = 'F')
        lon_llc = lon_270.flatten(order = 'F')
        lat_llc = lat_270.flatten(order = 'F')
        #NOTE: not sure if deep copy is needed? check to see if we're changing these vals
 
    # verify that our little trick works in 4 parts of the earth'
    deg2rad = np.pi/180.0
    for i in range(1,5):
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
        testarr = np.asarray([test_x, test_y, test_z])
        test_ind = int(F(testarr)[0])
      
        print("=================")
        print("i = {} | test_ind = {} | lat = {} | lon = {}".format(i, test_ind, test_lat, test_lon))
        print("{} {} {} | {} {} {}".format(X[test_ind], Y[test_ind], Z[test_ind], test_x, test_y, test_z))
        print("{} {} | {} {}".format(lat_llc[test_ind], lon_llc[test_ind], test_lat, test_lon))
        print("=================")
        """
        NOTE: review this w/ ian
        for the last 2 we calculate different vals for test_ind (interpolation)
        The longs/ lat match MATLAB down vals by 0.5, PYTHON up by 0.5
        """

    ##---------------------------------------------
    ## Read and process the profile files
    ##---------------------------------------------
    #[prof_x, prof_y, prof_z] = sph2cart(MITprof.prof_lon*deg2rad, MITprof.prof_lat*deg2rad, 1);
    deg2rad = np.pi/180
    prof_x, prof_y, prof_z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 1)
    
    # map a grid index to each profile.
    prof_llcN_cell_index = F(np.column_stack((prof_x, prof_y, prof_z)))
    prof_llcN_cell_index  = prof_llcN_cell_index.astype(int)

    """
    if 'prof_gci' in MITprof:
        if unique(MITprof.prof_gci - prof_llcN_cell_index) ~= 0
            [' prof_llcN_cell index does not equal prof_gci!!!']
            break;
        else
            'prof_gci and prof_llcN_cell_index are the same'
    """  
    """
    for bin_i = 1:num_bins
        bin_tmp = bin{bin_i};
        MITprof = setfield(MITprof, prof_bin_name{bin_i}, ...
            bin_tmp(prof_llcN_cell_index));
    """
     # loop through the different geodesic bins
    bin_1 = bin_1.flatten(order = 'F')
    bin_2 = bin_2.flatten(order = 'F')
    MITprofs['prof_bin_id_a'] = bin_1[prof_llcN_cell_index]
    MITprofs['prof_bin_id_b'] = bin_2[prof_llcN_cell_index]
    """
    if save_output_to_disk
        %  Write output
        ['writing output to netcdf']
        fileOut=[output_dir '/' fDataOut{ilist}]
        fprintf('%s\n',fileOut);
        write_profile_structure_to_netcdf(MITprof_new,fileOut);
    else
        % add to MITprofs container
        MITprofs_new{ilist} = MITprof;
    end
    """

def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    
    update_spatial_bin_index_on_prepared_profiles(run_code, MITprofs, grid_dir)

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
    

    args = parser.parse_args()

    run_code = args.run_code
    grid_dir = args.grid_dir
    MITprofs_fp = args.MIT_dir

    if run_code != 90 or run_code != 270:
        raise Exception("Runcode has to be 90 or 270!")
    
    nc_files = glob.glob(os.path.join(MITprofs_fp, '*.nc'))
    if len(nc_files) == 0:
        raise Exception("Invalid NC filepath")
    for file in nc_files:
        MITprofs = MITprof_read(nc_files)

    main(run_code, MITprofs, grid_dir)