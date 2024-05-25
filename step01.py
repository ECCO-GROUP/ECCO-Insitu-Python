import argparse
import glob
import os
import numpy as np
import copy
from geopy import distance
from scipy.interpolate import griddata
from tools import MITprof_read, load_llc270_grid, load_llc90_grid, patchface3D, sph2cart

def get_profpoint_llc_ian(lon_llc, lat_llc, mask_llc, MITprof):
    """
    Finds the 'prof_point' of each profile in the MITprof object for a global LLC grid

    lon_llc, lat_llc: the XC and YC of the llc grid in compact format llc x (13* llc)
    mask_llc: a mask with 1/0 denoting whether to use a point or not in the search
    """

    deg2rad = np.pi/180.0
    llcN = lon_llc.shape[0]
    
    mask_llc_pf, faces = patchface3D(llcN,13*llcN,1,mask_llc,2)
    
    X_grid, Y_grid, Z_grid = sph2cart(lon_llc*deg2rad, lat_llc*deg2rad, 1)

    # convert X,Y,Z coords to global view
    X_grid_pf, faces =patchface3D(llcN,13*llcN,1,X_grid,2)
    Y_grid_pf, faces =patchface3D(llcN,13*llcN,1,Y_grid,2)
    Z_grid_pf, faces =patchface3D(llcN,13*llcN,1,Z_grid,2)
    
    AI_grid_pf = np.arange(0, X_grid_pf.size).reshape(X_grid_pf.shape, order = 'F')
    
    mask_llc_pf_flat = mask_llc_pf.flatten(order = 'F')
    good_ins = np.where(mask_llc_pf_flat == 1)[0]

    # make a subset of X,Y,Z and AI to include only the non-nan, not masked points
    X_grid_pf = X_grid_pf.flatten(order = 'F')
    X_grid_pf = X_grid_pf[good_ins]

    Y_grid_pf = Y_grid_pf.flatten(order='F')
    Y_grid_pf = Y_grid_pf[good_ins]

    Z_grid_pf =  Z_grid_pf.flatten(order='F')
    Z_grid_pf = Z_grid_pf[good_ins]

    AI_grid_pf = AI_grid_pf.flatten(order='F')
    AI_grid_pf = AI_grid_pf[good_ins]

    # these are the x,y,z coordinates of the 'good' cells in
    model_xyz = np.column_stack((X_grid_pf, Y_grid_pf, Z_grid_pf))
    point_lon = MITprof["prof_lon"].astype(np.float64)
    point_lat = MITprof["prof_lat"].astype(np.float64)
    
    prof_x, prof_y, prof_z = sph2cart(point_lon*deg2rad, point_lat*deg2rad, 1)

    F_grid_PF_XYZ_to_INDEX = griddata(model_xyz, AI_grid_pf, (prof_x, prof_y, prof_z), method='nearest')
    F_grid_PF_XYZ_to_INDEX = F_grid_PF_XYZ_to_INDEX.astype(int)
    
    # creating new prof_point field in dict and populating
    MITprof.update({"prof_point": F_grid_PF_XYZ_to_INDEX})

    return F_grid_PF_XYZ_to_INDEX

def get_tile_point_llc_ian(lon_llc, lat_llc, ni, nj, MITprof):
    """
    Finds the tile coordinates for the MITgcm profile package for a profile point on a llc grid
    
    lon_llc, lat_llc: lon and lat dimensions [llc x (13 * llc)]
    ni, nj: the tile size for the model
    """

    llcN = np.size(lon_llc,0)

    # conver the XC YC coordinates to patchface.
    xgrid, faces = patchface3D(llcN, llcN*13, 1, lon_llc , 2)
    ygrid, faces = patchface3D(llcN, llcN*13, 1, lat_llc , 2)

    tileCount=0
 
    #get 5 faces
    temp, xgrid = patchface3D(4*llcN,4*llcN,1,xgrid,0.5)
    temp, ygrid = patchface3D(4*llcN,4*llcN,1,ygrid,0.5)
    
    # initalize some variables
    XC11= copy.deepcopy(xgrid)
    YC11= copy.deepcopy(ygrid)
    XCNINJ= copy.deepcopy(xgrid)
    YCNINJ= copy.deepcopy(xgrid)
    iTile= copy.deepcopy(xgrid)
    jTile= copy.deepcopy(xgrid)
    tileNo= copy.deepcopy(xgrid)

    # loop through each of the 5 faces in the grid structure;
    for iF in range(len(xgrid)):
        # pull the XC and YC
        face_XC = xgrid[iF]
        face_YC = ygrid[iF]
        
        # loop through each tile - of which there are size(face_XC,1) /ni in i
        # and size(face_XC,2)/nj in j
        for ii in range(face_XC.shape[0]// ni):
            for jj in range(face_XC.shape[1] // nj):

                # accumulate tile counter
                tileCount = tileCount + 1

                # find the indicies of this particular tile in i and j
                tmp_i = np.arange(ni) + ni * ii
                tmp_j = np.arange(nj) + nj * jj

                # pull the XC and YC of this tile
                tmp_XC = face_XC[tmp_i[:, np.newaxis], tmp_j[:]]
                tmp_YC = face_YC[tmp_i[:, np.newaxis], tmp_j[:]]

                # pull the XC and YC at position 1,1 of this tile
                XC11[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_XC[0, 0]
                YC11[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_YC[0, 0]

                # pull the XC and YC at the position (end, end) of this tile
                XCNINJ[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_XC[-1, -1]
                YCNINJ[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_YC[-1, -1]

                # fill in this iTile/jTile structure at this tile points with a
                # local count of the i and j within the tile [from 1:ni] 
                # basically, this is a map of where in the tile each i,j point
                # is.  in llcv4 it is 1:30 in i and 1:30 in j            
                iTile[iF][tmp_i[:, np.newaxis], tmp_j[:]] = (np.arange(1, ni + 1) * np.ones((1, nj))).T
                jTile[iF][tmp_i[:, np.newaxis], tmp_j[:]] = np.ones((1, nj)) * np.arange(1, ni + 1)

                # This is a count of the tile
                tileNo[iF][tmp_i[:, np.newaxis], tmp_j] = tileCount * np.ones((ni,nj))

    list_in = {0: xgrid, 1: ygrid, 2: XC11, 3: YC11, 4: XCNINJ, 5: YCNINJ, 6: iTile, 7: jTile, 8: tileNo}
        
    # now take these funky structures, cast them into patchface) form, then use
    # prof point to pull the value at the profile point that we need
    for k in range(len(list_in) - 1):
        # puts this in the original llc messed up face
        temp_var = list_in.get(k) # gets list of lists 
        temp_var_list_concate = np.concatenate((temp_var[0], temp_var[1], temp_var[2], temp_var[3].T, temp_var[4].T), axis = 1)
        list_in[k], faces = patchface3D(llcN,13*llcN,1,temp_var_list_concate,3.5)

        # use the prof_point to pull the right value from whatever list_in{k}
        # is.. list_in{k} is in patchface format, from above.    
        if k == 0:
            MITprof['prof_interp_lon'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 1:
            MITprof['prof_interp_lat'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 2:    
            MITprof['prof_interp_XC11'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 3:    
            MITprof['prof_interp_YC11'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 4:    
            MITprof['prof_interp_XCNINJ'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 5:    
            MITprof['prof_interp_YCNINJ'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 6:    
            MITprof['prof_interp_i'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
        elif k == 7:    
            MITprof['prof_interp_j'] = list_in[k].flatten(order = 'F')[MITprof['prof_point']]
    
    # one last thing: "weights", which is 1 b/c we're using nearest neighbor:
    MITprof['prof_interp_weights'] = np.ones(MITprof['prof_point'].shape)

    return MITprof

def update_prof_and_tile_points_on_profiles(MITprof, grid_dir, llcN, wet_or_all):
    """
    This script updates the prof_points and tile interpolation points
    so that the MITgcm knows which grid points to use for the cost function

    Input Parameters:
        llcN: which grid to use, 90 or 270
        wet_or_all: 0 = interpolated to nearest wet point, 1 = interpolated all points, regardless of wet or dry
        MITprof: a single MITprof object
        grid_dir: directory path of grid to be read in

    Output:
        Operates on MITprofs directly 
    """
    
    ##  Read in llc grid 
    if llcN == 90:

        lon_90, lat_90, blank_90, wet_ins_90_k = load_llc90_grid(grid_dir, 1)
        # tiles are 30x30
        ni = 30
        nj = 30
        lon_llc = lon_90
        lat_llc = lat_90
        mask_llc = blank_90
        if wet_or_all == 0:
            mask_llc[np.unravel_index(wet_ins_90_k[0], mask_llc.shape, order = 'F')] = 1
        else:
            mask_llc=np.ones(blank_90.shape, order = 'F') 
    if llcN == 270:
        lon_270, lat_270, blank_270, wet_ins_270_k = load_llc270_grid(grid_dir, 1)
        ni = 30
        nj = 30  
        lon_llc = lon_270
        lat_llc = lat_270
        mask_llc = blank_270
        if wet_or_all ==0:
            mask_llc[wet_ins_270_k[1]] = 1
        else:
            mask_llc=np.ones(blank_270.shape, order = 'F')
   
    F = get_profpoint_llc_ian(lon_llc, lat_llc, mask_llc, MITprof)

    MITprof = get_tile_point_llc_ian(lon_llc, lat_llc, ni, nj, MITprof)
        
    #  Sanity Check Interpolation 
    #  if the distance between the closest mitgcm grid point and the 
    #  profile is too far then assign it a flag of 101
    #  also check to see if |lat| > 90, if so then assign flag 100.
    #  these flag values can be used later when assigning weights.

    tmp_prof_lat = copy.deepcopy(MITprof['prof_lat'])
    tmp_prof_lon = copy.deepcopy(MITprof['prof_lon'])
        
    bad_lats_over = np.where(abs(tmp_prof_lat)>90)[0]
    bad_lats_under = np.where(tmp_prof_lat<-90)[0]
        
    if bad_lats_over.size != 0 or bad_lats_under.size != 0:
        raise Exception("Step01: Bad lats found (lat>90 or lat<-90)")
    
    d = []
    for k in range(len(tmp_prof_lat)):
        d.append(distance.distance((tmp_prof_lat[k], tmp_prof_lon[k]), (MITprof['prof_interp_lat'][k], MITprof['prof_interp_lon'][k])).km)
    d = np.asarray(d)

    # distance between grid cells referenced to llcN 90
    dx = 112* 90/llcN

    # find points where distance between the profile point and the 
    # closest grid point is further than twice the distance 
    # of the square root of the area.
    ins_too_far = np.where(d / dx > 2)[0]

    # if the profile lat is > |90| call it a bad lat
    # if the distance between the profile and the nearest grid cell
    # is greater than one grid cell distance then call it a bad point
    # -- you may have to create the field prof_flag.
    if 'prof_flag' not in MITprof:
        MITprof['prof_flag'] = np.zeros(len(MITprof['prof_YYYYMMDD']))

    MITprof['prof_flag'][ins_too_far] = 101

def main(MITprof, grid_dir, llcN, wet_or_all):

    print("step01: update_prof_and_tile_points_on_profiles")
    update_prof_and_tile_points_on_profiles(MITprof, grid_dir, llcN, wet_or_all)

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
        MITprofs = MITprof_read(nc_files, 1)
    
    llcN = 90                       # Which grid to use, 90 or 270
    wet_or_all = 1                  # 0 = interpolated to nearest wet point, 1 = interpolated all points, regardless of wet or dry

    main(MITprofs, grid_dir, llcN, wet_or_all)