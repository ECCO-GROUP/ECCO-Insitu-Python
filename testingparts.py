
import glob
import os
import netCDF4 as nc
import numpy as np
import copy
from scipy.interpolate import griddata
from NODC import MITprof_read

from step01 import load_llc90_grid, load_llc270_grid


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

def sph2cart(az, elev, r):

    z = r * np.sin(elev)
    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev * np.sin(az)

    return x, y, z

class TriScatteredInterp:
    """
    TriScatteredIntrep function to mimick the behavior of function in Matlab
    """
    def __init__(self, X, V, method='nearest'):
        self.X = X
        self.V = V
        self.method = method
        # self.interpolated_values = griddata(self.X, self.V, self.X, method=self.method)
    """
    Python's griddata will return an array of interpolated values for all 
    numbers.This allows 3 points to be passed and interpolated.
    """
    def __call__(self, points):
        return griddata(self.X, self.V, points, method=self.method)

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

def update_prof_and_tile_points_on_profiles(MITprofs):

    bin_dir = 'C:\\Users\\szswe\\Desktop\\grid_llc90\\sphere_point_distribution'
    bin_file_1 = os.path.join(bin_dir, 'llc090_sphere_point_n_10242_ids.bin')
    bin_file_2 =  os.path.join(bin_dir, 'llc090_sphere_point_n_02562_ids.bin')

    bin_llcN = 90
    siz = [bin_llcN, 13*bin_llcN, 1, 1]
    typ = 1
    prec ='float32'  # real*4 corresponds to float32
    skip = 0 
    mform = '>f4' # 'ieee-be' corresponds to f4

    with open(bin_file_1, 'rb') as fid:
        bin_1 = np.fromfile(fid, dtype=mform)
        bin_1 = bin_1.reshape((siz[0], np.prod(siz[1:])), order='F')

    with open(bin_file_2, 'rb') as fid:
        bin_2 = np.fromfile(fid, dtype=mform)
        bin_2 = bin_2.reshape((siz[0], np.prod(siz[1:])), order='F')
 
    if bin_llcN  == 90:
        
        lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90 = load_llc90_grid()
        F = make_F_llc90_ALL_INS_SURF_XYZ_to_INDEX(bathy_90, X_90, Y_90, Z_90)
        X = X_90.flatten(order = 'F')
        Y = Y_90.flatten(order = 'F') 
        Z = Z_90.flatten(order = 'F')
        lon_llc = lon_90.flatten(order = 'F')
        lat_llc = lat_90.flatten(order = 'F')

    if bin_llcN  == 270:
        lon_270, lat_270, blank_270, wet_ins_270_k, X_270, Y_270, Z_270, bathy_270, good_ins_270 = load_llc270_grid()
        
        F = make_F_llc270_ALL_INS_SURF_XYZ_to_INDEX(X_270, Y_270, Z_270, bathy_270, good_ins_270)
        X = X_270
        Y = Y_270
        Z = Z_270
        lon_llc = lon_270
        lat_llc = lat_270

    # initialize new container for MITprofs (1 per year generally)
    MITprofs_new = {}
    # fprintf(['MITPROF ' num2str(ilist) '  of  '  num2str(length(MITprofs)) '\n'])
    num_profs = len(MITprofs['prof_lat'])
    # ['mapping profiles to bin llc grid']
    deg2rad = np.pi/180
    prof_x, prof_y, prof_z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 1)
    
    ## map a grid index to each profile.
    prof_llcN_cell_index = F(np.column_stack((prof_x, prof_y, prof_z)))

    """
    if 'prof_gci' in MITprof:
        if unique(MITprof.prof_gci - prof_llcN_cell_index) ~= 0
            [' prof_llcN_cell index does not equal prof_gci!!!']
            break;
        else
            'prof_gci and prof_llcN_cell_index are the same'
    """  
    # loop through the different geodesic bins
    """
    for bin_i = 1:num_bins
        bin_tmp = bin{bin_i};
        MITprof = setfield(MITprof, prof_bin_name{bin_i}, ...
            bin_tmp(prof_llcN_cell_index));
    """
    bin_1 = bin_1.flatten(order = 'F')
    bin_2 = bin_2.flatten(order = 'F')
    MITprofs['prof_bin_id_a'] = bin_1[prof_llcN_cell_index]
    MITprofs['prof_bin_id_b'] = bin_2[prof_llcN_cell_index]

    #MITprofs_new{ilist} = MITprof;
def test_methods(MITprofs):
    
    bin_dir = 'C:\\Users\\szswe\\Desktop\\grid_llc90\\sphere_point_distribution'
    bin_file_1 = os.path.join(bin_dir, 'llc090_sphere_point_n_10242_ids.bin')
    bin_file_2 =  os.path.join(bin_dir, 'llc090_sphere_point_n_02562_ids.bin')

    bin_llcN = 90
    siz = [bin_llcN, 13*bin_llcN, 1, 1]
    typ = 1
    prec ='float32'  # real*4 corresponds to float32
    skip = 0 
    mform = '>f4' # 'ieee-be' corresponds to f4

    with open(bin_file_1, 'rb') as fid:
        bin_1 = np.fromfile(fid, dtype=mform)
        bin_1 = bin_1.reshape((siz[0], np.prod(siz[1:])), order='F')
        bin_1 = bin_1.reshape((siz[0], siz[1], siz[2]))

    with open(bin_file_2, 'rb') as fid:
        bin_2 = np.fromfile(fid, dtype=mform)
        bin_2 = bin_2.reshape((siz[0], np.prod(siz[1:])), order='F')
        bin_2 = bin_2.reshape((siz[0], siz[1], siz[2]))
 
    if bin_llcN  == 90:
        
        lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90 = load_llc90_grid()
        #F = make_F_llc90_ALL_INS_SURF_XYZ_to_INDEX(bathy_90, X_90, Y_90, Z_90)
        X = X_90.flatten(order = 'F')
        Y = Y_90.flatten(order = 'F') 
        Z = Z_90.flatten(order = 'F')
        lon_llc = lon_90.flatten(order = 'F')
        lat_llc = lat_90.flatten(order = 'F')

    if bin_llcN  == 270:
        lon_270, lat_270, blank_270, wet_ins_270_k, X_270, Y_270, Z_270, bathy_270, good_ins_270 = load_llc270_grid()
        #F = make_F_llc270_ALL_INS_SURF_XYZ_to_INDEX(X_270, Y_270, Z_270, bathy_270, good_ins_270)
        X = X_270
        Y = Y_270
        Z = Z_270
        lon_llc = lon_270
        lat_llc = lat_270

    from pyresample import kd_tree, geometry
    from pyresample.geometry import GridDefinition

    AI_90 = np.arange(bathy_90.size)
    X_90 = X_90.flatten(order = 'F')
    Y_90 = Y_90.flatten(order = 'F')
    Z_90 = Z_90.flatten(order = 'F')
    xyz = np.column_stack((X_90, Y_90, Z_90))

    data = np.column_stack((X_90, Y_90, Z_90))
    lons = np.squeeze(lon_90)
    lats = np.squeeze(lat_90)
    area_def = GridDefinition(lons=lons, lats=lats)
    swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
    result = kd_tree.resample_nearest(swath_def, data, area_def, radius_of_influence=500000, epsilon=0.01)
  
    deg2rad = np.pi/180
    prof_x, prof_y, prof_z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 1)
    area_def = GridDefinition(lons=MITprofs['prof_lon'], lats=MITprofs['prof_lat'])
    swath_def = geometry.SwathDefinition(lons=MITprofs['prof_lon'], lats=MITprofs['prof_lat'])
    data = np.column_stack((prof_x.flatten(order = 'F'), prof_y.flatten(order = 'F'), prof_z.flatten(order = 'F')))
    result = kd_tree.resample_nearest(swath_def, data, area_def, radius_of_influence=500000, epsilon=0.01)
    print(MITprofs['prof_lat'].shape)
    return

    # initialize new container for MITprofs (1 per year generally)
    MITprofs_new = {}
    # fprintf(['MITPROF ' num2str(ilist) '  of  '  num2str(length(MITprofs)) '\n'])
    num_profs = len(MITprofs['prof_lat'])
    # ['mapping profiles to bin llc grid']
    deg2rad = np.pi/180
    prof_x, prof_y, prof_z = sph2cart(MITprofs['prof_lon']*deg2rad, MITprofs['prof_lat']*deg2rad, 1)
    
    ## map a grid index to each profile.
    F = ""
    prof_llcN_cell_index = []
    for i in range(len(prof_x)):
        prof_llcN_cell_index.append(F(prof_x[i], prof_y[i], prof_z[i])[0])
    prof_llcN_cell_index = np.asarray(prof_llcN_cell_index, order = 'F')
    print(prof_llcN_cell_index)
    """
    if 'prof_gci' in MITprof:
        if unique(MITprof.prof_gci - prof_llcN_cell_index) ~= 0
            [' prof_llcN_cell index does not equal prof_gci!!!']
            break;
        else
            'prof_gci and prof_llcN_cell_index are the same'
    """  
    # loop through the different geodesic bins
    """
    for bin_i = 1:num_bins
        bin_tmp = bin{bin_i};
        MITprof = setfield(MITprof, prof_bin_name{bin_i}, ...
            bin_tmp(prof_llcN_cell_index));
    """

    #MITprofs_new{ilist} = MITprof;

def main():

    netCDF_fp_mat = "C:\\Users\\szswe\\Desktop\\ECCO_processing\\20190131_END_CHAIN\\CTD_20190131_1997.nc"
    netCDF_fp = "C:\\Users\\szswe\\Desktop\\"
    
    netCDF_files = glob.glob(os.path.join(netCDF_fp, '*.nc'))
    MITprofs = {}

    if len(netCDF_files) != 0:
        for file in netCDF_files:
            MITprofs = MITprof_read(file)
            if MITprofs != 0 :
                update_prof_and_tile_points_on_profiles(MITprofs)
                #test_methods(MITprofs)
            else:
                raise Exception("No info in NetCDF files")
    else:
        raise Exception("No NetCDF files found")

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
    main()
    #main(dest_dir, input_dir)