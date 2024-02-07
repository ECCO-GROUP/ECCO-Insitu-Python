import os
import numpy as np
import copy
from geopy import distance
from scipy.interpolate import griddata

def patchface3D(nx, ny, nz, array_in, direction):

    faces = [] 

    if direction == 3.5: # 3 but for 2D
        # f1=array_in(1:nx, 1:nx*3 ,:);
        f1 = array_in[0:nx, 0:nx*3]
        #f2=array_in(1:nx,  nx*3+1:nx*6 ,:);
        f2 = array_in[0:nx, nx*3:nx*6]
        #f3=array_in(1:nx,  nx*6+1:nx*7 ,:);
        f3 = array_in[0:nx, nx*6:nx*7]

        #temp=array_in(1:nx,nx* 7+1:nx*10,:);
        temp = array_in[0:nx, nx*7:nx*10]
        #for k=1:nz; f4(:,:,k)=temp(:,:,k)'; end;
        f4 = temp.T 
        # temp=array_in(1:nx, nx*10+1:nx*13, :);	
        temp = array_in[0:nx, nx*10:nx*13]
        # for k=1:nz; f5(:,:,k)=temp(:,:,k)'; end;
        f5 = temp.T

        #array_out=zeros(4*nx,4*nx,nz);
        array_out = np.zeros((4 * nx, 4 * nx))

        for k in range(nz):
            # temp_out=zeros(4*nx,4*nx);
            temp_out = np.zeros((4 * nx, 4 * nx))
            #temp_out(1:3*nx,:)=[f1(:,:,k)',f2(:,:,k)',flipud(f4(:,:,k)),flipud(f5(:,:,k))];
            temp_out[:3 * nx, :] = np.hstack((f1.T, f2.T, np.flipud(f4), np.flipud(f5)))
            # temp_out(3*nx+1:4*nx,1:nx)=fliplr(f3(:,:,k));
            temp_out[3 * nx:, :nx] = np.fliplr(f3) 
            #array_out(:,:,k)=temp_out';
            array_out[:, :] = temp_out.T

        faces.append(f1)
        faces.append(f2)
        faces.append(f3)
        faces.append(f4)
        faces.append(f5)

    if direction == 2.5:
         # f1=array_in(:, 1:3 * nx, :);
        f1 = array_in[:, :3 * nx]
        # f2=array_in(: ,3 * nx+1:6 * nx, :);
        f2 = array_in[:, 3 * nx:6 * nx]
        # f3=array_in(:, 6 * nx+1:7 * nx,:);	% arctic: [nx nx]
        f3 = array_in[:, 6 * nx:7 * nx]   # arctic: [nx, nx]

        # Now the tricky part, because the grid is read in the wrong direction

        #f4a=array_in( :, 7 * nx+1:3:10 * nx,:);
        f4a = array_in[:, 7 * nx:10 * nx:3]
        # f4b=array_in(:, 7 * nx + 2:3:10 * nx + 1,:);
        f4b = array_in[:, 7 * nx + 1:10 * nx + 1:3]
        # f4c=array_in(:, 7 * nx + 3:3:10 * nx + 1,:);
        f4c = array_in[:, 7 * nx + 2:10 * nx + 1:3]
        #  f5a=array_in(:,10 * nx+1:3:13 * nx,:);
        f5a = array_in[:, 10 * nx:13 * nx:3]
        #f5b=array_in(:,  10 * nx + 2:3:13 * nx +1,:)
        f5b = array_in[:, 10 * nx + 1:13 * nx + 1:3]
        # f5c=array_in(:, 10 * nx + 3:3:13 * nx + 1,:);
        f5c = array_in[:, 10 * nx + 2:13 * nx + 1:3]

        f4 = np.zeros((3 * nx, nx))
        #print("ny: {}, nx : {}, nx : {}".format(ny, 3 * nx, nz))
        f5 = np.zeros((3 * nx, nx))

        # this loop only runs once? NZ = 1 for first 2 calls of patchface
        for k in range(nz):
            # temp = [        f4a(:, :, k)',  f4b(:, :, k)',  f4c(:, :, k)']; 
            temp = np.hstack((f4a[:, :].T, f4b[:, :].T, f4c[:, :].T))
            # f4(:,:,k)=temp' #f4[:, :, k] = temp.T
            valid_mask = np.isfinite(temp.T)
            np.copyto(f4[:, :], temp.T, where=valid_mask)

            # print(np.isnan(temp.T).any()) # NaN values present in temp
            # temp = [        f5a(:, :, k)',  f5b(:, :, k)',  f5c(:, :, k)']; 
            temp = np.hstack((f5a[:, :].T, f5b[:, :].T, f5c[:, :].T))
            # f5(:,:,k)=temp'; # f5[:, :, k] = temp.T
            valid_mask = np.isfinite(temp.T)
            np.copyto(f5[:, :], temp.T, where=valid_mask)

        # array_out=zeros(4*nx,4*nx,nz);
        array_out = np.zeros((4 * nx, 4 * nx))

        for k in range(nz):
            # temp_out=zeros(4*nx,4*nx);
            temp_out = np.zeros((4 * nx, 4 * nx))
            # temp_out(1:3*nx,:)=[f1(:,:,k)',f2(:,:,k)',flipud(f4(:,:,k)),flipud(f5(:,:,k))];    
            temp_out[:3 * nx] = np.hstack((f1[:, :].T, f2[:, :].T, np.flipud(f4[:, :]), np.flipud(f5[:, :])))
            
            # temp_out(3*nx+1:4*nx,1:nx)=fliplr(f3(:,:,k));
            #temp_var = np.fliplr(f3[:, :, k])
            #valid_mask = np.isfinite(temp_var)
            #np.copyto(temp_out[3 * nx:, :nx], temp_var, where = valid_mask)
            temp_out[3 * nx:, :nx] = np.fliplr(f3[:, :]) 

            # array_out(:,:,k)=temp_out';
            array_out[:, :] = temp_out.T

        faces.append(f1)
        faces.append(f2)
        faces.append(f3)
        faces.append(f4)
        faces.append(f5)

    if (direction == 2 or direction == 3):
        if direction == 3:
            print("uncoded")
        if direction == 2:  # [2] from MITgcm compact array
            # f1=array_in(:, 1:3 * nx, :);
            f1 = array_in[:, :3 * nx, :]
            # f2=array_in(: ,3 * nx+1:6 * nx, :);
            f2 = array_in[:, 3 * nx:6 * nx, :]
            # f3=array_in(:, 6 * nx+1:7 * nx,:);	% arctic: [nx nx]
            f3 = array_in[:, 6 * nx:7 * nx, :]   # arctic: [nx, nx]

            # Now the tricky part, because the grid is read in the wrong direction

            #f4a=array_in( :, 7 * nx+1:3:10 * nx,:);
            f4a = array_in[:, 7 * nx:10 * nx:3, :]
            # f4b=array_in(:, 7 * nx + 2:3:10 * nx + 1,:);
            f4b = array_in[:, 7 * nx + 1:10 * nx + 1:3, :]
            # f4c=array_in(:, 7 * nx + 3:3:10 * nx + 1,:);
            f4c = array_in[:, 7 * nx + 2:10 * nx + 1:3, :]
            #  f5a=array_in(:,10 * nx+1:3:13 * nx,:);
            f5a = array_in[:, 10 * nx:13 * nx:3, :]
            #f5b=array_in(:,  10 * nx + 2:3:13 * nx +1,:)
            f5b = array_in[:, 10 * nx + 1:13 * nx + 1:3, :]
            # f5c=array_in(:, 10 * nx + 3:3:13 * nx + 1,:);
            f5c = array_in[:, 10 * nx + 2:13 * nx + 1:3, :]

            f4 = np.zeros((3 * nx, nx, nz))
            #print("ny: {}, nx : {}, nx : {}".format(ny, 3 * nx, nz))
            f5 = np.zeros((3 * nx, nx, nz))

            for k in range(nz):
                # temp = [        f4a(:, :, k)',  f4b(:, :, k)',  f4c(:, :, k)']; 
                temp = np.hstack((f4a[:, :, k].T, f4b[:, :, k].T, f4c[:, :, k].T))
                # f4(:,:,k)=temp' #f4[:, :, k] = temp.T
                valid_mask = np.isfinite(temp.T)
                np.copyto(f4[:, :, k], temp.T, where=valid_mask)

                # print(np.isnan(temp.T).any()) # NaN values present in temp
                # temp = [        f5a(:, :, k)',  f5b(:, :, k)',  f5c(:, :, k)']; 
                temp = np.hstack((f5a[:, :, k].T, f5b[:, :, k].T, f5c[:, :, k].T))
                # f5(:,:,k)=temp'; # f5[:, :, k] = temp.T
                valid_mask = np.isfinite(temp.T)
                np.copyto(f5[:, :, k], temp.T, where=valid_mask)

        # array_out=zeros(4*nx,4*nx,nz);
        array_out = np.zeros((4 * nx, 4 * nx, nz))

        for k in range(nz):
            # temp_out=zeros(4*nx,4*nx);
            temp_out = np.zeros((4 * nx, 4 * nx))
            # temp_out(1:3*nx,:)=[f1(:,:,k)',f2(:,:,k)',flipud(f4(:,:,k)),flipud(f5(:,:,k))];    
            temp_out[:3 * nx, :] = np.hstack((f1[:, :, k].T, f2[:, :, k].T, np.flipud(f4[:, :, k]), np.flipud(f5[:, :, k])))
            
            # temp_out(3*nx+1:4*nx,1:nx)=fliplr(f3(:,:,k));
            #temp_var = np.fliplr(f3[:, :, k])
            #valid_mask = np.isfinite(temp_var)
            #np.copyto(temp_out[3 * nx:, :nx], temp_var, where = valid_mask)
            temp_out[3 * nx:, :nx] = np.fliplr(f3[:, :, k]) 

            # array_out(:,:,k)=temp_out';
            array_out[:, :, k] = temp_out.T

        faces.append(f1)
        faces.append(f2)
        faces.append(f3)
        faces.append(f4)
        faces.append(f5)

    # NOTE: Same as 0 but for 2D arrays
    if direction == 0.5: 
        nx=nx//4
        # f1=array_in(1:nx,1:3*nx,:)	
        f1 = array_in[:nx, :3 * nx, :]
        f1 = np.squeeze(f1)
        #f2=array_in(nx+1:2*nx,1:3*nx,:)		
        f2 = array_in[nx:2 * nx, :3 * nx, :]
        f2 = np.squeeze(f2)
        #temp=array_in(1:nx,3*nx+1:4*nx,:)
        temp = array_in[:nx, 3 * nx:4 * nx, :]
        
        # for k=1:nz; f3(:,:,k)=fliplr(temp(:,:,k)');
        f3 = np.array([np.fliplr(temp[:, :, k].T) for k in range(temp.shape[2])])
        # Remove the singleton dimension
        f3 = np.squeeze(f3)

        #temp=array_in(2 * nx+1:3*nx,1:3*nx,:)
        temp = array_in[2 * nx:3 * nx, :3 * nx, :]
        #for k=1:nz; f4(:,:,k)=temp(:,:,k)'
        f4 = np.array([temp[:, :, k].T for k in range(temp.shape[2])])
        f4 = np.squeeze(f4)

        #temp=array_in(3*nx+1:4*nx,1:3*nx,:)
        temp = array_in[3 * nx:4 * nx, :3 * nx, :]
        #for k=1:nz; f5(:,:,k)=temp(:,:,k)';
        f5 = np.array([temp[:, :, k].T for k in range(temp.shape[2])])
        f5 = np.squeeze(f5)

        f4a = f4[:nx, :]
        #f4b=f4(nx+1:2*nx,:,:);
        f4b = f4[nx:2*nx, :]	
        #f4c=f4(2*nx+1:3*nx,:,:);
        f4c = f4[2*nx:3*nx, :]

        #f5a=f5(1:nx,:,:);	
        f5a = f5[:nx, :]
        #f5b=f5(nx+1:2*nx,:,:);	
        f5b = f5[nx:2*nx, :]
        #f5c=f5(2*nx+1:3*nx,:,:);
        f5c = f5[2*nx:3*nx, :]
        
        # array_out=zeros(nx,13*nx,nz);
        array_out = np.zeros((nx,13*nx))

        #for k=1:nz
        for k in range(nz):
            # f4p=zeros(nx,3*nx);
            f4p = np.zeros((nx, 3*nx))
            # f4p(:, 3:3:end,k)=flipud(f4a(:,:,k));
            f4p[:, 2::3] = np.flipud(f4a[:, :])
            # f4p(:, 2:3:end,k)=flipud(f4b(:,:,k));
            f4p[:, 1::3] = np.flipud(f4b[:, :])
            # f4p(:, 1:3:end,k)=flipud(f4c(:,:,k));
            f4p[:, 0::3] = np.flipud(f4c[:, :])

            # f5p=zeros(nx,3*nx);
            f5p = np.zeros((nx, 3*nx))
            # f5p(:,3:3:end,k)=flipud(f5a(:,:,k));
            f5p[:, 2::3] = np.flipud(f5a[:, :])
            # f5p(:,2:3:end,k)=flipud(f5b(:,:,k));
            f5p[:, 1::3] = np.flipud(f5b[:, :])
            # f5p(:,1:3:end,k)=flipud(f5c(:,:,k));
            f5p[:, 0::3] = np.flipud(f5c[:, :])

            # array_out(1:nx, 1: 3*nx,k) = f1(:,:,k);
            array_out[0:nx, 0:3*nx] = f1[:, :]
            # array_out(1:nx, 3*nx+1: 6*nx,k) = f2(:,:,k);
            array_out[0:nx, 3*nx:6*nx] = f2[:, :]
            # array_out(1:nx, 6*nx+1: 7*nx,k) = f3(:,:,k);
            array_out[0:nx, 6*nx:7*nx] = f3[:, :]
            # array_out(1:nx, 7*nx+1:10*nx,k) = f4p;
            array_out[0:nx, 7*nx:10*nx] = f4p
            # array_out(1:nx,10*nx+1:13*nx,k) = f5p;
            array_out[0:nx, 10*nx:13*nx] = f5p

        faces.append(f1)	
        faces.append(f2)
        faces.append(f3)
        faces.append(np.flipud(f4))
        faces.append(np.flipud(f5))

    if(direction==0 or direction==1):
        if direction==0: 
            raise Exception("double check this, we coded 0.5 for 2D, this handles 3D arrays")
            nx=nx//4
            # f1=array_in(1:nx,1:3*nx,:)	
            f1 = array_in[:nx, :3 * nx, :]
            #f2=array_in(nx+1:2*nx,1:3*nx,:)		
            f2 = array_in[nx:2 * nx, :3 * nx, :]
            #temp=array_in(1:nx,3*nx+1:4*nx,:)
            temp = array_in[:nx, 3 * nx:4 * nx, :]
            
            # for k=1:nz; f3(:,:,k)=fliplr(temp(:,:,k)');
            f3 = np.array([np.fliplr(temp[:, :, k].T) for k in range(temp.shape[2])])
            # Remove the singleton dimension

            #temp=array_in(2*nx+1:3*nx,1:3*nx,:)
            temp = array_in[2 * nx:3 * nx, :3 * nx, :]
            #for k=1:nz; f4(:,:,k)=temp(:,:,k)'
            f4 = np.array([temp[:, :, k].T for k in range(temp.shape[2])])

            #temp=array_in(3*nx+1:4*nx,1:3*nx,:)
            temp = array_in[3 * nx:4 * nx, :3 * nx, :]
            #for k=1:nz; f5(:,:,k)=temp(:,:,k)';
            f5 = np.array([temp[:, :, k].T for k in range(temp.shape[2])])

        if direction == 1:
            print("uncoded")

        f4a = f4[:nx, :, :]
        #f4b=f4(nx+1:2*nx,:,:);
        f4b = f4[nx:2*nx, :, :]	
        #f4c=f4(2*nx+1:3*nx,:,:);
        f4c = f4[2*nx:3*nx, :, :]
        #f5a=f5(1:nx,:,:);	
        f5a = f5[:nx, :, :]
        #f5b=f5(nx+1:2*nx,:,:);	
        f5b = f5[nx:2*nx, :, :]
        #f5c=f5(2*nx+1:3*nx,:,:);
        f5c = f5[2*nx:3*nx, :, :]
        
        # array_out=zeros(nx,13*nx,nz);
        array_out = np.zeros((nx,13*nx, nz))

        #for k=1:nz
        for k in range(nz):
            # f4p=zeros(nx,3*nx);
            f4p = np.zeros((nx, 3*nx))
            # f4p(:,3:3:end,k)=flipud(f4a(:,:,k));
            f4p[:, 2:3:-1, k] = f4a[:, :, k]
            # f4p(:,2:3:end,k)=flipud(f4b(:,:,k));
            f4p[:, 1:3:-1, k] = f4b[:, :, k]
            # f4p(:,1:3:end,k)=flipud(f4c(:,:,k));
            f4p[:, 0:3:-1, k] = f4c[:, :, k]

            # f5p=zeros(nx,3*nx);
            f5p = np.zeros((nx, 3*nx))
            # f5p(:,3:3:end,k)=flipud(f5a(:,:,k));
            f5p[:, 2:3:-1, k] = f5a[:, :, k]
            # f5p(:,2:3:end,k)=flipud(f5b(:,:,k));
            f5p[:, 1:3:-1, k] = f5b[:, :, k]
            # f5p(:,1:3:end,k)=flipud(f5c(:,:,k));
            f5p[:, 0:3:-1, k] = f5c[:, :, k]

            # array_out(1:nx,      1: 3*nx,k) = f1(:,:,k);
            array_out[0:nx, 0:3*nx, k] = f1[:, :, k]
            # array_out(1:nx, 3*nx+1: 6*nx,k) = f2(:,:,k);
            array_out[0:nx, 3*nx:6*nx, k] = f2[:, :, k]
            # array_out(1:nx, 6*nx+1: 7*nx,k) = f3(:,:,k);
            array_out[0:nx, 6*nx:7*nx, k] = np.fliplr(f3[:, :, k])
            # array_out(1:nx, 7*nx+1:10*nx,k) = f4p;
            array_out[0:nx, 7*nx:10*nx, k] = f4p
            # array_out(1:nx,10*nx+1:13*nx,k) = f5p;
            array_out[0:nx, 10*nx:13*nx, k] = f5p

        print(array_out.shape)
        raise Exception("double check this, we coded 0.5 for 2D, this handles 3D arrays")
        """
        % same as f1=readbin('llc_001_96_288.bin',[97 289],1,'real*8');f1=f1(1:nx,1:3*nx);
        faces{1}=f1;	
        faces{2}=f2;
        faces{3}=f3; 
        faces{4}=flipud(f4);
        faces{5}=flipud(f5);
        """

    return array_out, faces

def make_llc90_cell_centers():

    delR = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.01,
     10.03, 10.11, 10.32, 10.80, 11.76, 13.42, 16.04, 19.82, 24.85,
     31.10, 38.42, 46.50, 55.00, 63.50, 71.58, 78.90, 85.15, 90.18,
     93.96, 96.58, 98.25, 99.25,100.01,101.33,104.56,111.33,122.83, 
     139.09,158.94,180.83,203.55,226.50,249.50,272.50,295.50,318.50, 
     341.50,364.50,387.50,410.50,433.50,456.50])
    
    z_top = np.concatenate(([0], np.cumsum(delR[:-1])))
    z_bot = np.cumsum(delR)
    z_cen = 0.5 * (z_top[:50] + z_bot[:50])

    return delR, z_top, z_bot, z_cen

def sph2cart(az, elev, r):

    z = r * np.sin(elev)
    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev * np.sin(az)

    return x, y, z

def load_llc90_grid():

    grootdir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'

    """
    %% BATHY
    % CODES 0 is ECCOv4 
    %       1 is ice shelf cavity
    """
    BATHY_CODE = 0

    if BATHY_CODE == 0:
        bathy_90_fname = os.path.join(grootdir, 'grid_llc90', 'bathy_eccollc_90x50_min2pts.bin')

    if BATHY_CODE == 1:
        bathy_90_fname = os.path.join('/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents', 'grid_llc90', 'bathy_with_ice_shelf','BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin')

    # 1 Good 
    # bathy_90 = readbin(bathy_90_fname,[llcN 13*llcN 1],1,'real*4',0,'ieee-be')
    llcN = 90 # in matlab
    siz = [llcN, 13*llcN, 1]
    typ = 1
    prec ='float32'  # real*4 corresponds to float32
    skip = 0 
    mform = '>f4' # 'ieee-be' corresponds to f4
    with open(bathy_90_fname, 'rb') as fid:
        bathy_90 = np.fromfile(fid, dtype=mform)
        bathy_90 = bathy_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        bathy_90 = bathy_90.reshape((siz[0], siz[1], siz[2]))
    
    #2 Good
    XC_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'XC.data')
    YC_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'YC.data')
    # lon_90 = readbin('XC.data',[llcN, 13*llcN, 1],1,'real*4',0,'ieee-be');
    with open(XC_path, 'rb') as fid:
        lon_90 = np.fromfile(fid, dtype=mform)
        # order F: populates column first instead of default Python via row
        lon_90 = lon_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        lon_90 = lon_90.reshape((siz[0], siz[1], siz[2]))
    #lat_90 = readbin('YC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(YC_path, 'rb') as fid:
        lat_90 = np.fromfile(fid, dtype=mform)
        lat_90 = lat_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        lat_90 = lat_90.reshape((siz[0], siz[1], siz[2]))
    
    # 3 Good
    XG_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'XG.data')
    YG_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'YG.data')    
    #XG_90 = readbin('XG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(XG_path, 'rb') as fid:
        XG_90 = np.fromfile(fid, dtype=mform)
        XG_90 = XG_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        XG_90 = XG_90.reshape((siz[0], siz[1], siz[2]))
    #YG_90 = readbin('YG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(YG_path, 'rb') as fid:
        YG_90 = np.fromfile(fid, dtype=mform)
        YG_90 = YG_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        YG_90 = YG_90.reshape((siz[0], siz[1], siz[2]))
    
    # 4 good
    RAC90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'RAC.data')
    #  RAC_90 = readbin('RAC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(RAC90_path, 'rb') as fid:
        RAC_90 = np.fromfile(fid, dtype=mform)
        RAC_90 = RAC_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        RAC_90 = RAC_90.reshape((siz[0], siz[1], siz[2]))

    RAC_90_pf, faces = patchface3D(llcN, llcN*13, 1, RAC_90, 2)

    # 5 good
    #RC_90 = readbin('RC.data',[1 50],1,'real*4',0,'ieee-be');
    RC90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'RC.data')
    siz = [1, 50] # [llcN, 13*llcN, 1]
    with open(RC90_path, 'rb') as fid:
        RC_90 = np.fromfile(fid, dtype=mform)
        RC_90 = RC_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        RC_90 = RC_90.reshape((siz[0], siz[1]))

    # 6 good
    #RF_90 = readbin('RF.data',[1 50],1,'real*4',0,'ieee-be');
    RF90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'RF.data')
    with open(RF90_path, 'rb') as fid:
        RF_90 = np.fromfile(fid, dtype=mform)
        # python reads 51 vals but matlab reads 50
        RF_90 = RF_90[0:siz[1]]
        RF_90 = RF_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        RF_90 = RF_90.reshape((siz[0], siz[1]))
    
    # 7 Good
    siz = [llcN, 13*llcN, 1]
    DXG_90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'DXG.data')
    #DXG_90 = readbin('DXG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(DXG_90_path, 'rb') as fid:
        DXG_90 = np.fromfile(fid, dtype=mform)
        DXG_90 = DXG_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        DXG_90 = DXG_90.reshape((siz[0], siz[1], siz[2]))

    #DYG_90 = readbin('DYG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    DXG_90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'DYG.data')
    with open(DXG_90_path, 'rb') as fid:
        DYG_90 = np.fromfile(fid, dtype=mform)
        DYG_90 = DYG_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        DYG_90 = DYG_90.reshape((siz[0], siz[1], siz[2]))

    # 8 good
    #DXC_90 = readbin('DXC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    DXC_90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'DXC.data')
    with open(DXC_90_path, 'rb') as fid:
        DXC_90 = np.fromfile(fid, dtype=mform)
        DXC_90 = DXC_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        DXC_90 = DXC_90.reshape((siz[0], siz[1], siz[2]))
    # DYC_90 = readbin('DYC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    DYC_90_path = os.path.join(grootdir, 'grid_llc90', 'no_blank', 'DYC.data')
    with open(DYC_90_path, 'rb') as fid:
        DYC_90 = np.fromfile(fid, dtype=mform)
        DYC_90 = DYC_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        DYC_90 = DYC_90.reshape((siz[0], siz[1], siz[2]))

    # 9 Good, index off by 1 to account for matlab/ Python difference
    # AI_90 = 1:length(lon_90(:));
    AI_90 = np.arange(0, lon_90.size)
    #  AI_90 = reshape(AI_90, size(lon_90));
    AI_90 = AI_90.reshape(lon_90.shape, order = 'F')

    # NOTE: ADDED CODE BC need good_ins_90
    bad_ins_90 = np.where(np.logical_and(lat_90 == 0, lon_90 == 0, bathy_90 == 0).flatten(order = 'F'))[0]
    good_ins_90 = np.setdiff1d(AI_90.flatten(order = 'F').T, bad_ins_90.flatten(order = 'F'))

    # 10 Good
    # blank_90 = bathy_90.*NaN;
    blank_90 = np.full_like(bathy_90, np.nan)
    bathy_90_pf, faces = patchface3D(llcN, llcN*13, 1, bathy_90, 2)

    # 11 Good
    delR_90, z_top_90, z_bot_90, z_cen_90 = make_llc90_cell_centers()
   
    deg2rad = np.pi/180.0
    X_90, Y_90, Z_90 = sph2cart(lon_90*deg2rad, lat_90*deg2rad, 1)

    # 12 good
    # cd([llc90_grid_dir])
    # Depth_90 = readbin('Depth.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    depth_90_path = os.path.join(grootdir, 'grid_llc90', 'Depth.data')
    with open(depth_90_path, 'rb') as fid:
        Depth_90 = np.fromfile(fid, dtype=mform)
        Depth_90 = Depth_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        Depth_90 = Depth_90.reshape((siz[0], siz[1], siz[2]))

    hFacC_90_path = os.path.join(grootdir, 'grid_llc90', 'hFacC.data')
    siz = [llcN, 13*llcN, 50]
    # hFacC_90 = readbin('hFacC.data',[llcN 13*llcN 50],1,'real*4',0,'ieee-be');
    with open(hFacC_90_path, 'rb') as fid:
        hFacC_90 = np.fromfile(fid, dtype=mform)
        # order F: populates column first instead of default Python via row
        hFacC_90 = hFacC_90.reshape((siz[0], np.prod(siz[1:])), order='F')
        hFacC_90 = hFacC_90.reshape((siz[0], siz[1], siz[2]), order='F')
    
    siz = [llcN, 13*llcN, 1]
    # 13 Good
    # need to flatten to get 1D array indices like matlab
    hFacC_90_flat = hFacC_90.flatten(order = 'F')
    # wet_ins_90 = find(hFacC_90 > 0);
    # returns a tuple, get first element of arr index
    wet_ins_90 = np.where(hFacC_90_flat > 0)[0]
    # dry_ins_90 = find(hFacC_90 == 0);
    dry_ins_90 = np.where(hFacC_90 == 0)[0]

    # 14 NOTE: problems - Python cannot have numpy arr of different sizes
    # so result is a list of numpy arrs 
    dry_ins_90_k = []
    wet_ins_90_k = []
    nan_ins_90_k = []
    for k in range(0,50):
        tmp = hFacC_90[:,:,k].flatten(order = 'F')
        dry_ins_90_k.append(np.where(tmp == 0)[0])
        wet_ins_90_k.append(np.where(tmp > 0)[0])
        nan_ins_90_k.append(np.where(np.isnan(tmp))[0])  
    """
    MATLAB:
    for k = 1:50                                # gets vertical slices
        tmp = hFacC_90(:,:,k);
        dry_ins_90_k{k} = find(tmp == 0);       # find where in tmp arr == 0
        wet_ins_90_k{k} = find(tmp > 0);
        nan_ins_90_k{k} = find(isnan(tmp));
    end
    # result: all arrays are 2D and contain the indexes of where in hFacC_90 there exists said elements
    """

    # 16 Good
    # hf0_90 = hFacC_90[:,:,1]
    hf0_90 = hFacC_90[:,:,0]
    temp = hf0_90
    temp = np.ceil(temp)

    # 17 Good
    # NOTE: problems = matlab slices different, had to recode for 2D
    # landmask_90_pf = patchface3D(llcN, llcN*13, 1, temp, 2)
    landmask_90_pf, faces = patchface3D(llcN, llcN*13, 1, temp, 2.5)

    return lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90

def make_llc270_cell_centers():

    delR = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.01,
                     10.03, 10.11, 10.32, 10.80, 11.76, 13.42, 16.04, 19.82, 24.85,
                     31.10, 38.42, 46.50, 55.00, 63.50, 71.58, 78.90, 85.15, 90.18,
                     93.96, 96.58, 98.25, 99.25, 100.01, 101.33, 104.56, 111.33,
                     122.83, 139.09, 158.94, 180.83, 203.55, 226.50, 249.50, 272.50,
                     295.50, 318.50, 341.50, 364.50, 387.50, 410.50, 433.50, 456.50])

    z_top = np.concatenate(([0], np.cumsum(delR[:-1])))
    z_bot = np.cumsum(delR)
    z_cen = 0.5 * (z_top[:50] + z_bot[:50])

    return delR, z_top, z_bot, z_cen

def load_llc270_grid():

    llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    
    llcN = 270
    siz = [llcN, 13*llcN, 1]
    typ = 1
    prec ='float32'  # real*4 corresponds to float32
    skip = 0 
    mform = '>f4' # 'ieee-be' corresponds to f4

    # 1
    bathy_270_fname = os.path.join(llc270_grid_dir, 'bathy_llc270')
    #bathy_270 = readbin(bathy_270_fname,[llcN 13*llcN 1],1,'real*4',0,'ieee-be')
    with open(bathy_270_fname, 'rb') as fid:
        bathy_270 = np.fromfile(fid, dtype=mform)
        # order F: populates column first instead of default Python via row
        bathy_270 = bathy_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        bathy_270 = bathy_270.reshape((siz[0], siz[1], siz[2]))

    #2
    lon_270_path = os.path.join(llc270_grid_dir, 'XC.data')
    lat_270_path = os.path.join(llc270_grid_dir, 'YC.data')
    #lon_270 = readbin('XC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(lon_270_path, 'rb') as fid:
        lon_270 = np.fromfile(fid, dtype=mform)
        # order F: populates column first instead of default Python via row
        lon_270 = lon_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        lon_270 = lon_270.reshape((siz[0], siz[1], siz[2]))
    # lat_270 = readbin('YC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be')
    with open(lat_270_path, 'rb') as fid:
        lat_270 = np.fromfile(fid, dtype=mform)
        # order F: populates column first instead of default Python via row
        lat_270 = lat_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        lat_270 = lat_270.reshape((siz[0], siz[1], siz[2]))
    
    #3 did ^ instead of this line
    #[lat_270, lon_270] = load_grid_fields_from_tile_files(pwd, 270);

    #4
    XG_270_path = os.path.join(llc270_grid_dir, 'XG.data')
    YG_270_path = os.path.join(llc270_grid_dir, 'YG.data')
    #XG_270 = readbin('XG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(XG_270_path, 'rb') as fid:
        XG_270 = np.fromfile(fid, dtype=mform)
        XG_270 = XG_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        XG_270 = XG_270.reshape((siz[0], siz[1], siz[2]))
    #YG_270 = readbin('YG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(YG_270_path, 'rb') as fid:
        YG_270 = np.fromfile(fid, dtype=mform)
        YG_270 = YG_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        YG_270 = YG_270.reshape((siz[0], siz[1], siz[2]))

    #5
    RAC_270_path = os.path.join(llc270_grid_dir, 'RAC.data')
    #RAC_270 = readbin('RAC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(RAC_270_path, 'rb') as fid:
        RAC_270 = np.fromfile(fid, dtype=mform)
        RAC_270 = RAC_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        RAC_270 = RAC_270.reshape((siz[0], siz[1], siz[2]))

    #6
    DXG_270_path = os.path.join(llc270_grid_dir, 'DXG.data')
    #DXG_270 = readbin('DXG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(DXG_270_path, 'rb') as fid:
        DXG_270 = np.fromfile(fid, dtype=mform)
        DXG_270 = DXG_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        DXG_270 = DXG_270.reshape((siz[0], siz[1], siz[2]))
    DYG_270_path = os.path.join(llc270_grid_dir, 'DYG.data')
    #DYG_270 = readbin('DYG.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(DYG_270_path, 'rb') as fid:
        DYG_270 = np.fromfile(fid, dtype=mform)
        DYG_270 = DYG_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        DYG_270 = DYG_270.reshape((siz[0], siz[1], siz[2]))
    
    #7
    DXC_270_path = os.path.join(llc270_grid_dir, 'DXC.data')
    DYC_270_path = os.path.join(llc270_grid_dir, 'DYC.data')
    #DXC_270 = readbin('DXC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(DXC_270_path, 'rb') as fid:
        DXC_270 = np.fromfile(fid, dtype=mform)
        DXC_270 = DXC_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        DXC_270 = DXC_270.reshape((siz[0], siz[1], siz[2]))
    #DYC_270 = readbin('DYC.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(DYC_270_path, 'rb') as fid:
        DYC_270 = np.fromfile(fid, dtype=mform)
        DYC_270 = DYC_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        DYC_270 = DYC_270.reshape((siz[0], siz[1], siz[2]))

    #8
    Depth_270_path = os.path.join(llc270_grid_dir, 'Depth.data')
    #Depth_270 = readbin('Depth.data',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(Depth_270_path, 'rb') as fid:
        Depth_270 = np.fromfile(fid, dtype=mform)
        Depth_270 = Depth_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        Depth_270 = Depth_270.reshape((siz[0], siz[1], siz[2]))

    #9
    siz = [llcN, 13*llcN, 50]
    hFacC_270_path = os.path.join(llc270_grid_dir, 'hFacC.data')
    hFacW_270_path = os.path.join(llc270_grid_dir, 'hFacW.data')
    hFacS_270_path = os.path.join(llc270_grid_dir, 'hFacS.data')
    #hFacC_270 = readbin('hFacC.data',[llcN 13*llcN 50],1,'real*4',0,'ieee-be');
    with open(hFacC_270_path, 'rb') as fid:
        hFacC_270 = np.fromfile(fid, dtype=mform)
        hFacC_270 = hFacC_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        hFacC_270 = hFacC_270.reshape((siz[0], siz[1], siz[2]), order='F')
    #hFacW_270 = readbin('hFacW.data',[llcN 13*llcN 50],1,'real*4',0,'ieee-be');
    with open(hFacW_270_path, 'rb') as fid:
        hFacW_270 = np.fromfile(fid, dtype=mform)
        hFacW_270 = hFacW_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        hFacW_270 = hFacW_270.reshape((siz[0], siz[1], siz[2]), order='F')
    #hFacS_270 = readbin('hFacS.data',[llcN 13*llcN 50],1,'real*4',0,'ieee-be');
    with open(hFacS_270_path, 'rb') as fid:
        hFacS_270 = np.fromfile(fid, dtype=mform)
        hFacS_270 = hFacS_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        hFacS_270 = hFacS_270.reshape((siz[0], siz[1], siz[2]), order='F')
    
    # 10
    siz = [llcN, 13*llcN, 1]
    basin_mask_270_path = os.path.join(llc270_grid_dir, 'basin_masks_eccollc_llc270.bin')
    #basin_mask_270 = readbin('basin_masks_eccollc_llc270.bin',[llcN 13*llcN 1],1,'real*4',0,'ieee-be');
    with open(basin_mask_270_path, 'rb') as fid:
        basin_mask_270 = np.fromfile(fid, dtype=mform)
        basin_mask_270 = basin_mask_270.reshape((siz[0], np.prod(siz[1:])), order='F')
        basin_mask_270 = basin_mask_270.reshape((siz[0], siz[1], siz[2]))

    # 11
    deg2rad = np.pi/180.0

    # 12
    # this is how we find bogus points in the compact grid.
    #bad_ins_270 = find(lat_270 ==0 & lon_270 == 0 & bathy_270 == 0);
    bad_ins_270 = np.where(np.logical_and(lat_270 == 0, lon_270 == 0, bathy_270 == 0).flatten(order = 'F'))[0]
    #AI_270 = 1:length(lon_270(:));
    #AI_270 = reshape(AI_270, size(lon_270));
    AI_270 = np.arange(lon_270.size, dtype=np.float64).reshape(lon_270.shape, order = 'F')

    # 13
    #good_ins_270 = setdiff(AI_270(:)', bad_ins_270(:));
    good_ins_270 = np.setdiff1d(AI_270.flatten(order = 'F').T, bad_ins_270.flatten(order = 'F'))
    good_ins_270 = good_ins_270.astype(int)
    
    # 14
    #[X_270, Y_270, Z_270] = sph2cart(lon_270*deg2rad, lat_270*deg2rad, 1);
    X_270, Y_270, Z_270 = sph2cart(lon_270*deg2rad, lat_270*deg2rad, 1)

    # 15
    X_270[np.unravel_index(bad_ins_270, X_270.shape, order = 'F')] = np.NaN
    Y_270[np.unravel_index(bad_ins_270, X_270.shape, order = 'F')] = np.NaN
    Z_270[np.unravel_index(bad_ins_270, X_270.shape, order = 'F')] = np.NaN
    AI_270[np.unravel_index(bad_ins_270, AI_270.shape, order = 'F')] = np.NaN
  
    lon_270[np.unravel_index(bad_ins_270, lon_270.shape, order = 'F')] = np.NaN
    lat_270[np.unravel_index(bad_ins_270, lat_270.shape, order = 'F')] = np.NaN
    
    # 17
    dry_ins_270_k = []
    wet_ins_270_k = []
    for k in range(0,50):
        tmp = hFacC_270[:,:,k].flatten(order = 'F')
        dry_ins_270_k.append(np.where(tmp == 0)[0])
        wet_ins_270_k.append(np.where(tmp > 0)[0])
    """
    for k = 1:50
        tmp = hFacC_270(:,:,k);
        dry_ins_270_k{k} = find(tmp == 0);
        wet_ins_270_k{k} = find(tmp > 0);
    """
    # 18
    # hf0_270 = hFacC_270(:,:,1);
    hf0_270 = hFacC_270[:,:,0]
    # tmp = bathy_270(wet_ins_270_k{1});
    bathy_270_flat = bathy_270.flatten(order = 'F')
    tmp = bathy_270_flat[wet_ins_270_k[0]]

    # 19
    #blank_270 = bathy_270.*NaN;
    blank_270 = np.full_like(bathy_270, np.nan)

    # 20
    #hf0_270_pf = patchface3D(llcN, llcN*13, 1, hFacC_270(:,:,1), 2);
    hf0_270_pf, faces = patchface3D(llcN, llcN*13, 1, hFacC_270[:,:,1], 2.5)
    #bathy_270_pf = patchface3D(llcN, llcN*13, 1, bathy_270, 2);
    bathy_270_pf, faces = patchface3D(llcN, llcN*13, 1, bathy_270, 2)
    RAC_270_pf, faces = patchface3D(llcN, llcN*13, 1, RAC_270 , 2)
    
    # 22
    delR_270, z_top_270, z_bot_270, z_cen_270 = make_llc270_cell_centers()
    
    return lon_270, lat_270, blank_270, wet_ins_270_k, X_270, Y_270, Z_270, bathy_270, good_ins_270

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
    point_lon = MITprof["prof_lon"]
    point_lat = MITprof["prof_lat"]

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

def update_prof_and_tile_points_on_profiles(run_code, MITprof):
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
        lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90 = load_llc90_grid()
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
        lon_270, lat_270, blank_270, wet_ins_270_k = load_llc270_grid()
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
    """
    # if MITprofs = 0 then read in data from disk!!
    NOTE: moving to NODC main pipline
    #NOTE: var created: filepath_data_in for list of file
    filepath_data_in = "should be user input"
    if len(MITprofs) == 0:
            # glob all NETCDF files
            nc_files = glob.glob(filepath_data_in, '*.nc')  
            if len(MITprofs) != 0:
                MITprofs = MITprof_read(nc_files)
            else:
                raise Exception("No NetCDF files found")
    """
                
    # Loop throgh MITprof objects and update fields

    # initialize new container for MITprofs (1 per year generally)
    # NOTE: why?
    MITprofs_new = {}

    # NOTE: contains cell arr, I passed in only one MITprof at a time
    #for ilist = 1:length(MITprofs)
        # MITprof = MITprofs{ilist};
        # in python: MITprof
    # NOTE cont: so, we can bypass loop
    """
    # NOTE: F will never exist w/o init in PYthon
        if ~exist('F') 
            tic
            [MITprof_pp, F] = ...
                get_profpoint_llc_ian(lon_llc, lat_llc, mask_llc,  MITprof);
        else
            tic
            [MITprof_pp, F] = ...
                get_profpoint_llc_ian(lon_llc, lat_llc, mask_llc,  MITprof, F);
    """
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
    if 'prof_flag' in MITprof:
        #MITprof.prof_flag = zeros(length(MITprof.prof_YYYYMMDD),1);
        MITprof['prof_flag'] = np.zeros(len(MITprof['prof_YYMMDD']))

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

def main(dest_dir, input_dir):
    
    update_prof_and_tile_points_on_profiles(90, 43)

if __name__ == '__main__':

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dest_dir", action= "store",
                        help = "The destination where output files will be stored" , dest= "dest_dir",
                        type = str, required= True)

    parser.add_argument("-i", "--input_dir", action= "store",
                        help = "The input directory of CSV files to process" , dest= "input_dir",
                        type = str, required= True)
    

    args = parser.parse_args()

    dest_dir = args.dest_dir
    input_dir = args.input_dir
    """
    fnam = os.path.join('C:\\Users\\szswe\\Desktop\\', 'grid_llc90', 'bathy_eccollc_90x50_min2pts.bin')

    update_prof_and_tile_points_on_profiles(90, 43)
    #main(dest_dir, input_dir)