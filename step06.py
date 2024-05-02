import glob
import os
import numpy as np
from tools import MITprof_read

def sw_pres(DEPTH, LAT):

    # CHECK INPUTS
    mD,nD = DEPTH.shape
    mL,nL = LAT.shape

    if mL==1 & nL==1:
        LAT = np.ones(DEPTH.shape) * LAT

    if (mD != mL) or (nD != nL):              # DEPTH & LAT are not the same shape
        if (nD ==nL) & (mL==1):               # LAT for each column of DEPTH
            #LAT = LAT( ones(1,mD), : );      # copy LATS down each column
            LAT = np.tile(LAT[0, :], (LAT.shape[0], 1))     # s.t. dim(DEPTH)==dim(LAT)
        else:
            raise Exception('sw_pres.m:  Inputs arguments have wrong dimensions')

    Transpose = 0
    if mD == 1:  #row vector
        DEPTH = DEPTH.flatten(order = 'F')
        LAT = LAT.flatten(order = 'F')
        Transpose = 1

    DEG2RAD = np.pi/180
    X       = np.sin(np.abs(LAT)*DEG2RAD)  # convert to radians
    C1      = 5.92E-3 + X**2 * 5.25E-3
    pres    = ((1 - C1)- np.sqrt(((1-C1)**2)-(8.84E-6*DEPTH)))/4.42E-6

    if Transpose:
        pres = pres.T
    
    return pres


def sw_adtg(S,T,P):

    # CHECK S,T,P dimensions and verify consistent
    ms, ns = S.shape
    mt, nt = T.shape
    mp, np_s = P.shape
    
    # CHECK THAT S & T HAVE SAME SHAPE
    if (ms != mt) or (ns != nt):
        raise Exception('check_stp: S & T must have same dimensions')

    # CHECK OPTIONAL SHAPES FOR P
    if mp == 1 and np_s == 1:                    # P is a scalar.  Fill to size of S
        P = np.ones((ms,ns)) * P[0,0]
    elif np_s == ns and mp==1:                   # P is row vector with same cols as S
        P = np.tile(P[0, :], (P.shape[0], 1))    # Copy down each column.
    elif mp == ms and np_s ==1:                  # P is column vector
        P = np.tile(P[:, 0], (ns, 1)).T          # Copy across each row
    elif mp == ms and np_s == ns:                # PR is a matrix size(S)
        print("step6 (sw_adtg): shape ok")
    else:
        raise Exception('check_stp: P has wrong dimensions')

    mp, np_s = P.shape
    
    # IF ALL ROW VECTORS ARE PASSED THEN LET US PRESERVE SHAPE ON RETURN.
    Transpose = 0
    if mp == 1:  # row vector
        P = P.flatten(order= 'F')
        T = T.flatten(order= 'F')
        S = S.flatten(order= 'F')  
        Transpose = 1

    # BEGIN
    a0 =  3.5803E-5
    a1 = +8.5258E-6
    a2 = -6.836E-8
    a3 =  6.6228E-10

    b0 = +1.8932E-6
    b1 = -4.2393E-8

    c0 = +1.8741E-8
    c1 = -6.7795E-10
    c2 = +8.733E-12
    c3 = -5.4481E-14

    d0 = -1.1351E-10
    d1 =  2.7759E-12

    e0 = -4.6206E-13
    e1 = +1.8676E-14
    e2 = -2.1687E-16

    ADTG = a0 + (a1 + (a2 + a3 * T) *T) *T + (b0 + b1 *T) *(S-35) + ((c0 + (c1 + (c2 + c3 *T) *T) *T) + (d0 + d1 *T) *(S-35) ) *P + (e0 + (e1 + e2 *T) *T ) *P *P

    if Transpose:
        ADTG = ADTG.T

    return ADTG


def sw_ptmp(S, T, P, PR):

    # CHECK S,T,P dimensions and verify consistent
    ms, ns = S.shape
    mt, nt = T.shape
    mp, np_s = P.shape
    mpr, npr = PR.shape

    # CHECK THAT S & T HAVE SAME SHAPE
    if (ms != mt) or (ns !=nt):
        raise Exception('check_stp: S & T must have same dimensions')
  
    # CHECK OPTIONAL SHAPES FOR P
    if mp == 1 and np_s == 1:                          # P is a scalar.  Fill to size of S
        P = np.ones((ms,ns)) * P[0,0]
    elif np_s == ns and mp == 1:                       # P is row vector with same cols as S
        P = np.tile(P[0, :], (P.shape[0], 1))          # Copy down each column.
    elif mp == ms and np_s == 1:                       # P is column vector
        P = np.tile(P[:, 0], (ns, 1)).T                # Copy across each row
    elif mp == ms and np_s == ns:                      # PR is a matrix size(S)
        print("step6 (sw_ptmp): shape ok")
    else:
        raise Exception('check_stp: P has wrong dimensions')

    mp, np_s = P.shape
    
    # CHECK OPTIONAL SHAPES FOR PR
    if mpr == 1 and npr == 1:                          # PR is a scalar.  Fill to size of S
        PR = np.ones((ms,ns)) * PR[0,0]
    elif npr == ns and mpr == 1:                       # PR is row vector with same cols as S
        PR = np.tile(PR[0, :], (PR.shape[0], 1))       # Copy down each column.
    elif mpr == ms and npr == 1:                       # P is column vector
        PR = np.tile(PR[:, 0], (ns, 1)).T 
    elif mpr == ms and npr == ns:                      # PR is a matrix size(S)
        print("step6 (sw_ptmp): shape ok")
    else:
        raise Exception('check_stp: PR has wrong dimensions')

    mpr, npr = PR.shape
  
    # IF ALL ROW VECTORS ARE PASSED THEN LET US PRESERVE SHAPE ON RETURN.
    Transpose = 0
    if mp == 1:  # row vector
        P       =  P.flatten(order = 'F')
        T       =  T.flatten(order = 'F')
        S       =  S.flatten(order = 'F')
        PR      = PR.flatten(order = 'F')
        Transpose = 1

    # theta1
    del_P  = PR - P
    del_th = del_P * sw_adtg(S,T,P)
    th     = T + 0.5* del_th
    q      = del_th

    # theta2
    del_th = del_P * sw_adtg(S, th, P+ 0.5*del_P)
    th     = th + (1 - 1/np.sqrt(2)) * (del_th - q)
    q      = (2 - np.sqrt(2))*del_th + (-2+3/ np.sqrt(2))*q

    # theta3
    del_th = del_P * sw_adtg(S,th,P+0.5*del_P)
    th     = th + (1 + 1/np.sqrt(2))*(del_th - q)
    q      = (2 + np.sqrt(2))*del_th + (-2-3/np.sqrt(2))*q

    # theta4
    del_th = del_P *sw_adtg(S,th,P+del_P)
    PT     = th + (del_th - 2*q)/6

    if Transpose:
        PT = PT.T

    return PT

def update_prof_insitu_T_to_potential_T(run_code, MITprofs, grid_dir):

    # SET INPUT PARAMETERS
    fillVal=-9999
    checkVal=-9000

    # by default, do not replace missing salinity values with salinity values
    # from the climatology
    replace_missing_S_with_clim_S = 0

    if run_code == '20181202_use_clim_for_missing_S':
        save_output_to_disk = 0
        replace_missing_S_with_clim_S = 1
            
    elif run_code == 'NODC_20181029_clim_S':
        raise Exception("uncoded")
        """
        output_rootdir = ['/home/ifenty/data/observations/insitu/NODC/NODC_20180508/all/']
        rootdir = ['~/data11/ifenty/observations/insitu/20181015_NCEI/']
        

        input_dir  =  [rootdir 'step_03_sigmas'];
        output_dir =  [output_rootdir 'step_04_potential_temperature_clim_S']
        
        file_suffix_in  = ['step_03.nc'];
        file_suffix_out = ['step_04.nc']
        
        fig_dir = [output_dir '/figures']
    
        cd(input_dir);
        f = dir(['*nc']);
        
        for i = 1:length(f)
            fDataIn{i}  =   f(i).name;
            fDataOut{i} =  [f(i).name(1:end-length(file_suffix_in)) file_suffix_out];
        end
        
        replace_missing_S_with_clim_S = 1;
        """
    elif run_code == 'NODC_20181008':
        raise Exception("uncoded")
        """
        rootdir = ['/home/ifenty/data/observations/insitu/NODC/NODC_20180508/all/']
        input_dir  =  [rootdir 'step_03_sigmas'];
        output_dir =  [rootdir 'step_04_potential_temperature']
        
        file_suffix_in  = ['step_03.nc'];
        file_suffix_out = ['step_04.nc']
        
        fig_dir = [output_dir '/figures']
    
        cd(input_dir);
        f = dir(['*nc']);
        
        for i = 1:length(f)
            fDataIn{i}  =   f(i).name;
            fDataOut{i} =  [f(i).name(1:end-length(file_suffix_in)) file_suffix_out];
        end
        """
    elif run_code ==  '20181015_llc90_ITP':
        raise Exception("uncoded")
        """
        rootdir = '/home/ifenty/data/observations/insitu/ITP/201810/TO_JPL_2018/'
        input_dir  =  [rootdir 'step_03_sigmas'];
        output_dir =  [rootdir 'step_04_potential_temperature']
        
        file_suffix_in  = ['step_03.nc'];
        file_suffix_out = ['step_04.nc']
        
        fig_dir = [output_dir '/figures']
    
        cd(input_dir);
        f = dir(['*nc']);
        
        for i = 1:length(f)
            fDataIn{i}  =   f(i).name;
            fDataOut{i} =  [f(i).name(1:end-length(file_suffix_in)) file_suffix_out];
        end
        """
    elif run_code == '20181015_llc90_GOSHIP':
        """
        rootdir = '/home/ifenty/data/observations/insitu/GO-SHIP/SIO_201810/TO_JPL_2018/'
        input_dir  =  [rootdir 'step_03_sigmas'];
        output_dir =  [rootdir 'step_04_potential_temperature']
        
        file_suffix_in  = ['step_03.nc'];
        file_suffix_out = ['step_04.nc']
        
        fig_dir = [output_dir '/figures']
    
        cd(input_dir);
        f = dir(['*nc']);
        
        for i = 1:length(f)
            fDataIn{i}  =   f(i).name;
            fDataOut{i} =  [f(i).name(1:end-length(file_suffix_in)) file_suffix_out];
        end
        """
    
    lats = MITprofs['prof_lat']
    # flatten array and converted all NaN vals to fillval
    prof_T = MITprofs['prof_T'].flatten(order = 'F').filled(fillVal)
    prof_S = MITprofs['prof_S'].flatten(order = 'F').filled(fillVal)
    
    # to qualify you need to have a valid T, S 
    good_T_and_S_ins = np.where((prof_T != fillVal) & (prof_S != fillVal))[0]
    
    if replace_missing_S_with_clim_S:
        # missing_S_ins = find(prof_T ~= fillVal & prof_S == fillVal);
        missing_S_ins = np.where((prof_T != fillVal) & (prof_S == fillVal))[0]
        prof_S[missing_S_ins] = MITprofs['prof_Sclim'].ravel(order = 'F')[missing_S_ins]

    # to qualify you need to have a valid T, S  %%
    #good_T_and_S_ins = find(prof_T ~= fillVal & prof_S ~= fillVal);
    good_T_and_S_ins = np.where((prof_T != fillVal) & (prof_S != fillVal))[0]

    prof_S = prof_S.reshape(MITprofs['prof_S'].shape, order = 'F')
    prof_T = prof_T.reshape(MITprofs['prof_T'].shape, order = 'F')

    # Check to see if **all** salinty values are missing
    S_max = np.max(prof_S, axis = 1)
      
    if max(S_max) == fillVal:
        print('all S are missing, applying clim instead')
        prof_S = MITprofs['prof_Sclim']
        # to qualify you need to have a valid T, S  %%
        good_T_and_S_ins = np.where((prof_T != fillVal) & (prof_S != fillVal))[0]

    # Make empty arryas.
    prof_T_tmp = np.full_like(prof_T, np.nan)
    prof_S_tmp = np.full_like(prof_S, np.nan)

    # set values at the good T and S pairs to be the original T and S
    prof_T_tmp.ravel(order = 'F')[good_T_and_S_ins] = prof_T.ravel(order = 'F')[good_T_and_S_ins]
    prof_S_tmp.ravel(order = 'F')[good_T_and_S_ins] = prof_S.ravel(order = 'F')[good_T_and_S_ins]
    
    # define an empty ptemp;
    ptemp = np.full_like(prof_T, fillVal)
    
    if len(good_T_and_S_ins) > 0:
        # Prepare 2D matrix of pres and lats required for sw_ptmp
        depths = MITprofs['prof_depth']
        #depths_mat = repmat(depths, [1 length(lats)]);
        depths_mat = np.tile(depths, (len(lats), 1)).T

        #lats_mat = repmat(lats, [1 length(depths)])';
        lats_mat = np.tile(lats, (len(depths), 1))
        
        # calculate equivalent pressure from depth
        #pres_mat = sw_pres(depths_mat, lats_mat);
        pres_mat = sw_pres(depths_mat, lats_mat)
        pres_mat = pres_mat.T
    
        # Calc potential temperature w.r.t. to surf [pres = 0]
        ptemp = sw_ptmp(prof_S_tmp, prof_T_tmp, pres_mat, np.zeros(pres_mat.shape))
 
    else:
        print("step06: There is not a single good T and S pair to use here")
    
    ptemp = ptemp.filled(np.nan)
    # set to -9999 if there no new ptemp
    ptemp[np.isnan(ptemp)] = -9999

    MITprofs['prof_T'] = ptemp
 
    
def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    print("step06: update_prof_insitu_T_to_potential_T")
    update_prof_insitu_T_to_potential_T(run_code, MITprofs, grid_dir)

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
        MITprofs = MITprof_read(file, 6)

    run_code = '20181202_use_clim_for_missing_S'
    grid_dir = "hehe"

    main(run_code, MITprofs, grid_dir)
