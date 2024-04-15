
import os
import numpy as np
import netCDF4 as nc

"""
READS THE MATLAB GENERATED FILES
"""
def MITprof_read(file, step):

    MITprofs = {}
    dataset = nc.Dataset(file)
        
    df_HHMMSS = dataset.variables['prof_HHMMSS'][:]
    MITprofs.update({"prof_HHMMSS": df_HHMMSS})

    df_YYMMDD = dataset.variables['prof_YYYYMMDD'][:]
    MITprofs.update({"prof_YYYYMMDD": df_YYMMDD})
    
    df_lat = dataset.variables['prof_lat'][:]
    MITprofs.update({"prof_lat": df_lat})
    df_lon = dataset.variables['prof_lon'][:]
    MITprofs.update({"prof_lon": df_lon})

    df_basin = dataset.variables['prof_basin'][:]
    MITprofs.update({"prof_basin": df_basin})

    df_date = dataset.variables['prof_date'][:]
    MITprofs.update({"prof_date": df_date})

    df_depth = dataset.variables['prof_depth'][:]
    MITprofs.update({"prof_depth": df_depth})
    #df_depth_f_flag = dataset.variables['prof_depth_wod_flag'][:]
    #df_depth_o_flag = dataset.variables['prof_depth_orig_flag'][:]
    #MITprofs.update({"prof_depth_wod_flag": df_depth_f_flag})
    #MITprofs.update({"prof_depth_orig_flag": df_depth_o_flag})

    df_desc = dataset.variables['prof_descr'][:] 
    MITprofs.update({"prof_descr": df_desc})
    df_point = dataset.variables['prof_point'][:]
    MITprofs.update({"prof_point": df_point})

    #=========== PROF_S VARS ===========
    df_S = dataset.variables['prof_S'][:]
    MITprofs.update({"prof_S": df_S})

    df_Sestim = dataset.variables['prof_Sestim'][:]
    MITprofs.update({"prof_Sestim": df_Sestim})

    df_S_f_flag = dataset.variables['prof_Sflag'][:]   # prof_S_wod_flag
    MITprofs.update({"prof_Sflag": df_S_f_flag}) 

    #df_S_o_flag = dataset.variables['prof_S_orig_flag'][:]
    #MITprofs.update({"prof_S_orig_flag": df_S_o_flag})

    #=========== PROF_S VARS END ===========
    
    #=========== PROF_T VARS ===========
    df_T = dataset.variables['prof_T'][:]
    MITprofs.update({"prof_T": df_T})

    df_Testim = dataset.variables['prof_Testim'][:]
    MITprofs.update({"prof_Testim": df_Sestim})

    df_T_f_flag = dataset.variables['prof_Tflag'][:]   #  prof_T_wod_flag
    MITprofs.update({"prof_Tflag": df_T_f_flag})       #  prof_Tflag

    #df_T_o_flag = dataset.variables['prof_T_orig_flag'][:]
    #MITprofs.update({"prof_T_orig_flag": df_T_o_flag}) 
    #=========== PROF_T VARS END ===========

    # NOTE: added in step 1
    if step > 1:
        df_interp_i = dataset.variables['prof_interp_i'][:]
        MITprofs.update({"prof_interp_i": df_interp_i})
        df_interp_j = dataset.variables['prof_interp_j'][:]
        MITprofs.update({"prof_interp_j": df_interp_j})
        
        df_interp_lon = dataset.variables['prof_interp_lon'][:]
        MITprofs.update({"prof_interp_lon": df_interp_lon})
        df_interp_lat = dataset.variables['prof_interp_lat'][:]
        MITprofs.update({"prof_interp_lat": df_interp_lat})
    
        df_interp_weight = dataset.variables['prof_interp_weights'][:]
        MITprofs.update({"prof_interp_weights": df_interp_weight})
        
        df_interp_XC11 = dataset.variables['prof_interp_XC11'][:]
        MITprofs.update({"prof_interp_XC11": df_interp_XC11})

        df_interp_XCNINJ = dataset.variables['prof_interp_XCNINJ'][:]
        MITprofs.update({"prof_interp_XCNINJ": df_interp_XCNINJ})

        df_interp_YC11 = dataset.variables['prof_interp_YC11'][:]
        MITprofs.update({"prof_interp_YC11": df_interp_YC11})
        
        df_interp_YCNINJ = dataset.variables['prof_interp_YCNINJ'][:]
        MITprofs.update({"prof_interp_YCNINJ": df_interp_YCNINJ})
    
    # NOTE: added in step 2
    if step > 2:
        df_bin_a = dataset.variables['prof_bin_id_a'][:]
        MITprofs.update({"prof_bin_id_a": df_bin_a})
        df_bin_b = dataset.variables['prof_bin_id_b'][:]
        MITprofs.update({"prof_bin_id_b": df_bin_b})

    # NOTE: added in step 3
    
    if step > 3:
        df_prof_Tclim = dataset.variables['prof_Tclim'][:]
        MITprofs.update({"prof_Tclim": df_prof_Tclim})
        df_prof_Sclim = dataset.variables['prof_Sclim'][:]
        MITprofs.update({"prof_Sclim": df_prof_Sclim})
    
    # NOTE: arrs are empty before they are added in step 4?
    # However, step 4 tries first to pull existing info from these arrs BEFORE populating them
    # I would check at the end of pipeline completion and ask if there is ever a senario where these following fields
    # are populated from the original CSV files
    df_Serr = dataset.variables['prof_Serr'][:]    # NOTE: empty but there is code that is translated
    MITprofs.update({"prof_Serr": df_Serr})        # but untested to populate these fields
    df_Terr = dataset.variables['prof_Terr'][:]
    MITprofs.update({"prof_Terr": df_Terr})

    df_Sweight = dataset.variables['prof_Sweight'][:]
    MITprofs.update({"prof_Sweight": df_Sweight})
    df_Tweight = dataset.variables['prof_Tweight'][:]
    MITprofs.update({"prof_Tweight": df_Tweight})
    # Note above pertains to fields above this line

    if step > 5:
        df_area_gamma = dataset.variables['prof_area_gamma'][:]
        MITprofs.update({"prof_area_gamma": df_area_gamma})
    
    # step6: updated prof_T
    if step > 7:
        df_Tweight_code = dataset.variables['prof_Tweight_code'][:]
        MITprofs.update({"prof_Tweight_code": df_Tweight_code})
        df_Sweight_code = dataset.variables['prof_Sweight_code'][:]
        MITprofs.update({"prof_Sweight_code": df_Sweight_code})

    return MITprofs

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

def load_llc270_grid(llc270_grid_dir):

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
    
    return lon_270, lat_270, blank_270, wet_ins_270_k, X_270, Y_270, Z_270, bathy_270, good_ins_270, RAC_270_pf

def sph2cart(az, elev, r):

    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev* np.sin(az)
    z = r * np.sin(elev)

    return x, y, z

def load_llc90_grid(grootdir):

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
    #deg2rad = np.int64(deg2rad)
    lon_90_64 = lon_90.astype(np.float64)
    lat_90_64 = lat_90.astype(np.float64)

    X_90, Y_90, Z_90 = sph2cart(lon_90_64*deg2rad, lat_90_64*deg2rad, 1.0)

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

    return lon_90, lat_90, blank_90, wet_ins_90_k, RAC_90_pf, bathy_90, good_ins_90, X_90, Y_90, Z_90, z_top_90, z_bot_90, hFacC_90, AI_90, z_cen_90 
