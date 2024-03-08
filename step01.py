import argparse
import glob
import os
import numpy as np
import copy
from geopy import distance
from scipy.interpolate import griddata
from tools import MITprof_read, load_llc270_grid, load_llc90_grid, patchface3D, sph2cart

def get_profpoint_llc_ian(lon_llc, lat_llc, mask_llc, MITprof):

    deg2rad = np.pi/180.0

    # llcN = np.size(lon_llc,1)
    llcN = lon_llc.shape[0]
    
    lon_pf, faces = patchface3D(llcN,13*llcN,1,lon_llc,2)
    lat_pf, faces  = patchface3D(llcN,13*llcN,1,lat_llc,2)
    mask_llc_pf, faces = patchface3D(llcN,13*llcN,1,mask_llc,2)
    
    X_grid, Y_grid, Z_grid = sph2cart(lon_llc*deg2rad, lat_llc*deg2rad, 1)

    # convert X,Y,Z coords to global view
    X_grid_pf, faces =patchface3D(llcN,13*llcN,1,X_grid,2)
    Y_grid_pf, faces =patchface3D(llcN,13*llcN,1,Y_grid,2)
    Z_grid_pf, faces =patchface3D(llcN,13*llcN,1,Z_grid,2)
    
    # AI_grid_pf = X_grid_pf.*0
    # AI_grid_pf(1:end) = 1:length(AI_grid_pf(:))
    AI_grid_pf = np.arange(0, X_grid_pf.size).reshape(X_grid_pf.shape, order = 'F')
    
    # good_ins = find(mask_llc_pf == 1);
    mask_llc_pf_flat = mask_llc_pf.flatten(order = 'F')
    good_ins = np.where(mask_llc_pf_flat == 1)[0]

    # make a subset of X,Y,Z and AI to include only the non-nan, not masked points
    # X_grid_pf = X_grid_pf(good_ins);
    X_grid_pf_flat = X_grid_pf.flatten(order = 'F')
    X_grid_pf = X_grid_pf_flat[good_ins]

    #Y_grid_pf = Y_grid_pf[good_ins]
    Y_grid_pf_flat = Y_grid_pf.flatten(order='F')
    Y_grid_pf = Y_grid_pf_flat[good_ins]

    #Z_grid_pf = Z_grid_pf[good_ins]
    Z_grid_pf_flat =  Z_grid_pf.flatten(order='F')
    Z_grid_pf = Z_grid_pf_flat[good_ins]

    #AI_grid_pf = AI_grid_pf[good_ins]
    AI_grid_pf_flat = AI_grid_pf.flatten(order='F')
    AI_grid_pf = AI_grid_pf_flat[good_ins]

    # these are the x,y,z coordinates of the 'good' cells in
    # model_xyz= [X_grid_pf Y_grid_pf Z_grid_pf]
    model_xyz = np.column_stack((X_grid_pf, Y_grid_pf, Z_grid_pf))
    point_lon = MITprof["prof_lon"].astype(np.float64)
    point_lat = MITprof["prof_lat"].astype(np.float64)
    
    prof_x, prof_y, prof_z = sph2cart(point_lon*deg2rad, point_lat*deg2rad, 1)

    # F_grid_PF_XYZ_to_INDEX = scatteredInterpolant(model_xyz, AI_grid_pf,'nearest')
    F_grid_PF_XYZ_to_INDEX = griddata(model_xyz, AI_grid_pf, (prof_x, prof_y, prof_z), method='nearest')
    F_grid_PF_XYZ_to_INDEX = F_grid_PF_XYZ_to_INDEX.astype(int)
    
    # creating new prof_point field in dict and populating
    MITprof.update({"prof_point": F_grid_PF_XYZ_to_INDEX})

    return F_grid_PF_XYZ_to_INDEX

def get_tile_point_llc_ian(lon_llc, lat_llc, ni, nj, MITprof):

    llcN = np.size(lon_llc,0)

    # conver the XC YC coordinates to patchface.
    xgrid, faces = patchface3D(llcN, llcN*13, 1, lon_llc , 2)
    ygrid, faces = patchface3D(llcN, llcN*13, 1, lat_llc , 2)

    tileCount=0
 
    #get 5 faces
    temp,xgrid = patchface3D(4*llcN,4*llcN,1,xgrid,0.5)
    temp,ygrid = patchface3D(4*llcN,4*llcN,1,ygrid,0.5)
    
    # initalize some variables
    XC11= copy.deepcopy(xgrid)
    YC11= copy.deepcopy(ygrid)
    XCNINJ= copy.deepcopy(xgrid)
    YCNINJ= copy.deepcopy(xgrid)
    iTile= copy.deepcopy(xgrid)
    jTile= copy.deepcopy(xgrid)
    tileNo= copy.deepcopy(xgrid)

    # loop through each of the 5 faces in the grid structure;
    # for iF=1:size(xgrid,2);
    for iF in range(len(xgrid)):
        # pull the XC and YC
        face_XC = xgrid[iF]
        face_YC = ygrid[iF]
        
        # loop through each tile - of which there are size(face_XC,1) /ni in i
        # and size(face_XC,2)/nj in j
        # for ii=1:size(face_XC,1)/ni;
        for ii in range(face_XC.shape[0]// ni):
            # for jj=1:size(face_XC,2)/nj;
            for jj in range(face_XC.shape[1] // nj):

                # accumulate tile counter
                tileCount=tileCount+1
                # find the indicies of this particular tile in i and j
                #tmp_i=[1:ni]+ni*(ii-1)
                tmp_i = np.arange(ni) + ni * ii
                #tmp_j=[1:nj]+nj*(jj-1)
                tmp_j = np.arange(nj) + nj * jj

                # pull the XC and YC of this tile
                # tmp_XC=face_XC(tmp_i,tmp_j);
                tmp_XC = face_XC[tmp_i[:, np.newaxis], tmp_j[:]]
                #tmp_YC=face_YC(tmp_i,tmp_j);
                tmp_YC = face_YC[tmp_i[:, np.newaxis], tmp_j[:]]

                # pull the XC and YC at position 1,1 of this tile
                #XC11{iF}(tmp_i,tmp_j)=tmp_XC(1,1);
                XC11[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_XC[0, 0]
                #YC11{iF}(tmp_i,tmp_j)=tmp_YC(1,1);
                YC11[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_YC[0, 0]

                # pull the XC and YC at the position (end, end) of this tile
                #XCNINJ{iF}(tmp_i,tmp_j)=tmp_XC(end,end);
                XCNINJ[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_XC[-1, -1]
                #YCNINJ{iF}(tmp_i,tmp_j)=tmp_YC(end,end);
                YCNINJ[iF][tmp_i[:, np.newaxis], tmp_j[:]] = tmp_YC[-1, -1]

                # fill in this iTile/jTile structure at this tile points with a
                # local count of the i and j within the tile [from 1:ni] 
                # basically, this is a map of where in the tile each i,j point
                # is.  in llcv4 it is 1:30 in i and 1:30 in j            
                
                # iTile{iF}(tmp_i,tmp_j)=[1:ni]'*ones(1,nj);
                iTile[iF][tmp_i[:, np.newaxis], tmp_j[:]] = (np.arange(1, ni + 1) * np.ones((1, nj))).T
                #jTile{iF}(tmp_i,tmp_j)=ones(ni,1)*[1:nj];
                jTile[iF][tmp_i[:, np.newaxis], tmp_j[:]] = np.ones((1, nj)) * np.arange(1, ni + 1)

                # This is a count of the tile
                # tileNo{iF}(tmp_i,tmp_j)=tileCount*ones(ni,nj);
                tileNo[iF][tmp_i[:, np.newaxis], tmp_j] = tileCount * np.ones((ni,nj))

    #list_out= {0: MITprof['prof_interp_lon'], 1: MITprof['prof_interp_lat'], 2: MITprof['prof_interp_XC11'], 3: MITprof['prof_interp_YC11'], 4: MITprof['prof_interp_XCNINJ'], 5: MITprof['prof_interp_YCNINJ'], 6: MITprof['prof_interp_i'], 7: MITprof['prof_interp_j']}
    list_in = {0: xgrid, 1: ygrid, 2: XC11, 3: YC11, 4: XCNINJ, 5: YCNINJ, 6: iTile, 7: jTile, 8: tileNo}
        
    # now take these funky structures, cast them into patchface) form, then use
    # prof point to pull the value at the profile point that we need
    # for k=1:size(list_out,2); 
    for k in range(len(list_in) - 1):
        # puts this in the original llc fucked up face
        # eval(['temp=' list_in{k} ';'])
        temp_var = list_in.get(k) # gets list of lists 
        # temp=[temp{1},temp{2},temp{3}, temp{4}',temp{5}'];
        temp_var_list_concate = np.concatenate((temp_var[0], temp_var[1], temp_var[2], temp_var[3].T, temp_var[4].T), axis = 1)
        #eval([list_in{k} '=patchface3D(llcN,13*llcN,1,temp,3);']);
        list_in[k], faces = patchface3D(llcN,13*llcN,1,temp_var_list_concate,3.5)

        # use the prof_point to pull the right value from whatever list_in{k}
        # is.. list_in{k} is in patchface format, from above.    
        #eval(['MITprof.prof_interp_' list_out{k} '=' list_in{k} '(MITprof.prof_point);']);
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
    # MITprof.prof_interp_weights = ones(size(MITprof.prof_point));
    MITprof['prof_interp_weights'] = np.ones(MITprof['prof_point'].shape)

    return MITprof

"""
NOTE: replaced w/ built in function
"""
def vdist(lat1, lon1, lat2, lon2):

    # reshape inputs
    keepsize = np.size(lat1)
    # lat1=lat1(:)
    lat1 = np.ravel(lat1)
    lon1 = np.ravel(lon1)
    lat2 = np.ravel(lat2)
    lon2 = np.ravel(lon2)

    # Input check:
    # if any(abs(lat1)>90 | abs(lat2)>90)
    if np.any(np.abs(lat1) > 90) or np.any(np.abs(lat2) > 90):
        raise ValueError('Input latitudes must be between -90 and 90 degrees, inclusive.')
    
    # Supply WGS84 earth ellipsoid axis lengths in meters:
    a = 6378137 # definitionally
    b = 6356752.31424518 # computed from WGS84 earth flattening coefficient
    
    # preserve true input latitudes:
    lat1tr = copy.deepcopy(lat1)
    lat2tr = copy.deepcopy(lat2)

    # convert inputs in degrees to radians:
    lat1 = lat1 * 0.0174532925199433
    lon1 = lon1 * 0.0174532925199433
    lat2 = lat2 * 0.0174532925199433
    lon2 = lon2 * 0.0174532925199433

    # correct for errors at exact poles by adjusting 0.6 millimeters:
    kidx = np.abs(np.pi/2-np.abs(lat1)) < 1e-10
    if np.any(kidx):
        lat1[kidx] = np.sign(lat1[kidx])* (np.pi/2-(1e-10))
    kidx = np.abs(np.pi/2-abs(lat2)) < 1e-10
    if np.any(kidx):
        lat2[kidx] = np.sign(lat2[kidx])*(np.pi/2-(1e-10))

    f = (a-b)/a
    U1 = np.arctan((1-f)*np.tan(lat1))
    U2 = np.arctan((1-f)*np.tan(lat2))
    lon1 = np.mod(lon1,2*np.pi)
    lon2 = np.mod(lon2,2*np.pi)
    L = np.abs(lon2-lon1)
    kidx = L > np.pi
    if np.any(kidx):
        L[kidx] = 2*np.pi - L[kidx]

    lambda_var = copy.deepcopy(L)
    lambdaold = np.zeros_like(lat1)
    itercount = 0
    #logical(1+0*lat1)
    notdone = np.ones_like(lat1, dtype=bool)
    alpha = np.zeros_like(lat1)
    sigma = np.zeros_like(lat1)
    cos2sigmam = np.zeros_like(lat1)
    C = np.zeros_like(lat1)
    warninggiven = False

    while np.any(notdone):  # force at least one execution
        itercount = itercount + 1

        if itercount > 50:
            if warninggiven == False:
                print('Essentially antipodal points encountered. Precision may be reduced slightly.')
                warninggiven = True
            lambda_var[notdone] = np.pi
            break

        lambdaold[notdone] = lambda_var[notdone]
        sinsigma = np.empty(notdone.shape)
        sinsigma[notdone] = np.sqrt((np.cos(U2[notdone]) * np.sin(lambda_var[notdone]))**2 +
                                        (np.cos(U1[notdone]) * np.sin(U2[notdone]) -
                                        np.sin(U1[notdone]) * np.cos(U2[notdone]) * np.cos(lambda_var[notdone]))**2)
        cossigma = np.empty(notdone.shape)
        cossigma[notdone] = np.sin(U1[notdone]) * np.sin(U2[notdone]) + \
                            np.cos(U1[notdone]) * np.cos(U2[notdone]) * np.cos(lambda_var[notdone])

        # eliminate rare imaginary portions at limit of numerical precision:
        sinsigma[notdone] = np.real(sinsigma[notdone])
        cossigma[notdone]= np.real(cossigma[notdone])

        sigma[notdone] = np.arctan2(sinsigma[notdone],cossigma[notdone])
        alpha[notdone] = np.arcsin(np.cos(U1[notdone]) * np.cos(U2[notdone]) *
                              np.sin(lambda_var[notdone]) / np.sin(sigma[notdone]))

        cos2sigmam[notdone] = np.cos(sigma[notdone]) - 2 * np.sin(U1[notdone]) * np.sin(U2[notdone]) / \
                        np.cos(alpha[notdone])**2
        C[notdone] = f / 16 * np.cos(alpha[notdone])**2 * (4 + f * (4 - 3 * np.cos(alpha[notdone])**2))

        lambda_var[notdone] = L[notdone] + (1 - C[notdone]) * f * np.sin(alpha[notdone]) * \
                    (sigma[notdone] + C[notdone] * np.sin(sigma[notdone]) * 
                    (cos2sigmam[notdone] + C[notdone] * np.cos(sigma[notdone]) * 
                    (-1 + 2 * cos2sigmam[notdone]**2)))

        # correct for convergence failure in the case of essentially antipodal
        # points
        if np.any(lambda_var[notdone] > np.pi):
            print(['Essentially antipodal points encountered.Precision may be reduced slightly.'])
            warninggiven = True
            lambdaold[np.where(lambda_var > np.pi)[0]] = np.pi
            lambda_var[np.where(lambda_var>np.pi)] = np.pi

        notdone = np.abs(lambda_var - lambdaold) > 1e-12
        
    u2 = np.cos(alpha)**2 * (a**2 - b**2) / b**2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltasigma = B * np.sin(sigma) * (cos2sigmam + B / 4 * (np.cos(sigma) * (-1 + 2 * cos2sigmam**2) - B / 6 * cos2sigmam * (-3 + 4 * np.sin(sigma)**2) * (-3 + 4 * cos2sigmam**2)))
    
    varargout = np.reshape(b * A * (sigma - deltasigma), keepsize)

    return varargout

     #NOTE: nowhere in the Matlab code does nargout get assigned idk
    """    
    if nargout > 1
        % From point #1 to point #2
        % correct sign of lambda for azimuth calcs:
        lambda = abs(lambda);
        kidx=sign(sin(lon2-lon1)) .* sign(sin(lambda)) < 0;
        lambda(kidx) = -lambda(kidx);
        numer = cos(U2).*sin(lambda);
        denom = cos(U1).*sin(U2)-sin(U1).*cos(U2).*cos(lambda);
        a12 = atan2(numer,denom);
        kidx = a12<0;
        a12(kidx)=a12(kidx)+2*pi;
        % from poles:
        a12(lat1tr <= -90) = 0;
        a12(lat1tr >= 90 ) = pi;
        varargout{2} = reshape(a12 * 57.2957795130823,keepsize); % to degrees
    end
    if nargout > 2
        a21=NaN*lat1;
        % From point #2 to point #1
        % correct sign of lambda for azimuth calcs:
        lambda = abs(lambda);
        kidx=sign(sin(lon1-lon2)) .* sign(sin(lambda)) < 0;
        lambda(kidx)=-lambda(kidx);
        numer = cos(U1).*sin(lambda);
        denom = sin(U1).*cos(U2)-cos(U1).*sin(U2).*cos(lambda);
        a21 = atan2(numer,denom);
        kidx=a21<0;
        a21(kidx)= a21(kidx)+2*pi;
        % backwards from poles:
        a21(lat2tr >= 90) = pi;
        a21(lat2tr <= -90) = 0;
        varargout{3} = reshape(a21 * 57.2957795130823,keepsize); % to degrees
    end
    return
    """

def update_prof_and_tile_points_on_profiles(run_code, MITprof, grid_dir):
    """
    #   run_code : a code that tells the program which options to use.
    #   MITprofs : a list of MITprof objects.  If not specified, MITprofs = [],
    #              then the model will attempt to read from the disk files
    #              with name fDataIn which have to be defined somewhere in the
    #              appropriate run_code block.
    #
    # output:
    #   
    #   MITprofs_new : a list of MITprof objects that have been updated with
    #                  prof and tile points.
    #
    """
    ## Assign parameter values based on the run_code
    if run_code == 90:
        # llc model grid
        llcN = 90
        wet_or_all = 1
        make_figs = 0
        save_output_to_disk = 0
    elif run_code == 270:    
        # llc model grid
        llcN = 270
        wet_or_all = 1
        make_figs = 0
        save_output_to_disk = 0
    else: 
        # save_output_to_disk :option to save updated MITprof objects to disk.
        # 0: do not save to disk, only return MITprofs_new
        # 1: save to disk using filenames stored in fDataOut
        save_output_to_disk = 0
        # wet_or_all
        # 0: interpolate to nearest wet point
        # 1: interpolate to all points, regardless of wet or dry
        wet_or_all = 1
        # which llc grid to use [90 or 270]
        llcN = 90
        # make_figs: whether or not to make some figures as you go (0=no, 1=yes)
        make_figs = 0
    ##---------------------------------------------
    ##  Read in llc grid 

    if llcN == 90:

        lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90, z_top_90, z_bot_90, hFacC_90, AI_90, z_cen_90 = load_llc90_grid(grid_dir)
        # tiles are 30x30
        ni = 30
        nj = 30
        nx = 90 
        ny=90
        lon_llc = lon_90
        lat_llc = lat_90
        # llc_grid_dir = llc90_grid_dir
        mask_llc = blank_90
        if wet_or_all == 0:
            # mask_llc[wet_ins_90_k{1}] = 1
            mask_llc[np.unravel_index(wet_ins_90_k[0], mask_llc.shape, order = 'F')] = 1
        else:
            #mask_llc=np.ones(blank_90.shape)
            mask_llc=np.ones(blank_90.shape, order = 'F') 
        # llc_grid_dir = llc90_grid_dir

    if llcN == 270:
        lon_270, lat_270, blank_270, wet_ins_270_k = load_llc270_grid(grid_dir)
        nx = 270 
        ny=270

        # tiles are 30x30
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
        
    """ NOTE: optional feature 
    if make_figs
        %%
        figure(i);clf;hold on;
        set(0,'DefaultTextInterpreter','none');
        plot(MITprof.prof_lon, MITprof.prof_lat,'r.')
        plot(MITprof.prof_interp_lon, MITprof.prof_interp_lat,'bo')
        grid;
        
        figure;
        set(0,'DefaultTextInterpreter','none');
        projection_code=1;
        plot_proj=0;
        makemmap_general;
        m_plot(MITprof.prof_lon, MITprof.prof_lat,'r.')
        m_plot(MITprof.prof_interp_lon, MITprof.prof_interp_lat,'bo')
        makemmap_general;
        m_grid;
    end
    """    
    #  sanity check interpolation 
    #  if the distance between the closest mitgcm grid point and the 
    #  profile is too far then assign it a flag of 101
    #  also check to see if |lat| > 90, if so then assign flag 100.
    #  these flag values can be used later when assigning weights.

    tmp_prof_lat = copy.deepcopy(MITprof['prof_lat'])
    tmp_prof_lon = copy.deepcopy(MITprof['prof_lon'])
        
    bad_lats_over = np.where(abs(tmp_prof_lat)>90)[0]
    bad_lats_under = np.where(tmp_prof_lat<-90)[0]
        
    # print(np.where(tmp_prof_lat>90)[0])
    if bad_lats_over.size != 0 or bad_lats_under.size != 0:
        raise Exception("Bad lats found (lat>90 or lat<-90)")
    
    # Replaced this line w/ distance module below
    # d = vdist(tmp_prof_lat, tmp_prof_lon, MITprof['prof_interp_lat'], MITprof['prof_interp_lon'])   
    d = []
    for k in range(len(tmp_prof_lat)):
        d.append(distance.distance((tmp_prof_lat[k], tmp_prof_lon[k]), (MITprof['prof_interp_lat'][k], MITprof['prof_interp_lon'][k])).km)
    d = np.asarray(d)

    #NOTE: distance b/w grid cells referenced to llcN 90
    dx = 112* 90/llcN

    # find points where distance between the profile point and the 
    # closest grid point is further than twice the distance 
    # of the square root of the area.
    ins_too_far = np.where(d / dx > 2)[0]

    # if the profile lat is > |90| call it a bad lat
    # if the distance between the profile and the nearest grid cell
    # is greater than one grid cell distance then call it a bad point
    # -- you may have to create the field prof_flag.
    
    #if isfield(MITprof,'prof_flag')
    if 'prof_flag' not in MITprof:
        #MITprof.prof_flag = zeros(length(MITprof.prof_YYYYMMDD),1);
        MITprof['prof_flag'] = np.zeros(len(MITprof['prof_YYYYMMDD']))

    # MITprof['prof_flag'][bad_lats] = 100
    MITprof['prof_flag'][ins_too_far] = 101

    """
    if make_figs
        if length(ins) > 0
            figure(i)
            set(0,'DefaultTextInterpreter','none');
            h1=plot(MITprof.prof_lon(ins), MITprof.prof_lat(ins),'go','MarkerSize',10)
            
            set(h1(:), 'MarkerEdgeColor','k','MarkerFaceColor','g');
            h2=plot(MITprof.prof_interp_lon(ins), MITprof.prof_interp_lat(ins),'yo','MarkerSize',10)
            set(h2(:), 'MarkerEdgeColor','k','MarkerFaceColor','y');
            
            title(['points that are too far away: MITprof #' str2num(i)])
        end
    end
        
    %% Add updated MITprof object to list
    MITprofs_new{ilist} = MITprof;
        if save_output_to_disk
        cd(output_dir)
        fprintf(['writing ' fDataOut{ilist} '\n'])
        fileOut=[output_dir '/'  fDataOut{ilist}];
        fprintf('%s\n',fileOut);
        write_profile_structure_to_netcdf(MITprof, fileOut);
    end

    """

def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
     
    update_prof_and_tile_points_on_profiles(run_code, MITprofs, grid_dir)

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
        MITprofs = MITprof_read(nc_files, 1)

    main(run_code, MITprofs, grid_dir)