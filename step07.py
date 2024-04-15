import copy
import glob
import os
import numpy as np
import numpy.ma as ma
import datetime as dt
from tools import MITprof_read
import scipy.io as sio

def myprctile(x, p, dim = None):
    """
    %PRCTILE Percentiles of a sample.
    %   Y = PRCTILE(X,P) returns percentiles of the values in X.  P is a scalar
    %   or a vector of percent values.  When X is a vector, Y is the same size
    %   as P, and Y(i) contains the P(i)-th percentile.  When X is a matrix,
    %   the i-th row of Y contains the P(i)-th percentiles of each column of X.
    %   For N-D arrays, PRCTILE operates along the first non-singleton
    %   dimension.
    %
    %   Y = PRCTILE(X,P,DIM) calculates percentiles along dimension DIM.  The
    %   DIM'th dimension of Y has length LENGTH(P).
    %
    %   Percentiles are specified using percentages, from 0 to 100.  For an N
    %   element vector X, PRCTILE computes percentiles as follows:
    %      1) The sorted values in X are taken as the 100*(0.5/N), 100*(1.5/N),
    %         ..., 100*((N-0.5)/N) percentiles.
    %      2) Linear interpolation is used to compute percentiles for percent
    %         values between 100*(0.5/N) and 100*((N-0.5)/N)
    %      3) The minimum or maximum values in X are assigned to percentiles
    %         for percent values outside that range.
    %
    %   PRCTILE treats NaNs as missing values, and removes them.
    %
    %   Examples:
    %      y = prctile(x,50); % the median of x
    %      y = prctile(x,[2.5 25 50 75 97.5]); % a useful summary of x
    %
    %   See also IQR, MEDIAN, NANMEDIAN, QUANTILE.

    %   Copyright 1993-2004 The MathWorks, Inc.
    """
    
    if  np.size(p) == 0 or len(np.where((p < 0)) | (p > 100)[0]) > 0:
        raise Exception("stats:prctile:BadPercents")

    # Figure out which dimension prctile will work along.
    sz = x.shape
    if dim == None:
        dim = np.where(sz != 1)[0][0]
        print(dim)
        if np.isempty(dim):
            dim = 1; 
        dimArgGiven = False
    else:
        raise Exception("untranslated")
        # Permute the array so that the requested dimension is the first dim.
        """
        nDimsX = ndims(x);
        perm = [dim:max(nDimsX,dim) 1:dim-1];
        x = permute(x,perm);
        % Pad with ones if dim > ndims.
        if dim > nDimsX
            sz = [sz ones(1,dim-nDimsX)];
        end
        sz = sz(perm);
        dim = 1;
        dimArgGiven = true;
        """
    """
    # If X is empty, return all NaNs.
    if isempty(x)
        if isequal(x,[]) && ~dimArgGiven
            y = nan(size(p),class(x));
        else
            szout = sz; szout(dim) = numel(p);
            y = nan(szout,class(x));
        end

    else
        % Drop X's leading singleton dims, and combine its trailing dims.  This
        % leaves a matrix, and we can work along columns.
        nrows = sz(dim);
        ncols = prod(sz) ./ nrows;
        x = reshape(x, nrows, ncols);

        x = sort(x,1);
        nonnans = ~isnan(x);

        % If there are no NaNs, do all cols at once.
        if all(nonnans(:))
            n = sz(dim);
            if isequal(p,50) % make the median fast
                if rem(n,2) % n is odd
                    y = x((n+1)/2,:);
                else        % n is even
                    y = (x(n/2,:) + x(n/2+1,:))/2;
                end
            else
                q = [0 100*(0.5:(n-0.5))./n 100]';
                xx = [x(1,:); x(1:n,:); x(n,:)];
                y = zeros(numel(p), ncols, class(x));
                y(:,:) = interp1q(q,xx,p(:));
            end

        % If there are NaNs, work on each column separately.
        else
            % Get percentiles of the non-NaN values in each column.
            y = nan(numel(p), ncols, class(x));
            for j = 1:ncols
                nj = find(nonnans(:,j),1,'last');
                if nj > 0
                    if isequal(p,50) % make the median fast
                        if rem(nj,2) % nj is odd
                            y(:,j) = x((nj+1)/2,j);
                        else         % nj is even
                            y(:,j) = (x(nj/2,j) + x(nj/2+1,j))/2;
                        end
                    else
                        q = [0 100*(0.5:(nj-0.5))./nj 100]';
                        xx = [x(1,j); x(1:nj,j); x(nj,j)];
                        y(:,j) = interp1q(q,xx,p(:));
                    end
                end
            end
        end

        % Reshape Y to conform to X's original shape and size.
        szout = sz; szout(dim) = numel(p);
        y = reshape(y,szout);
    end
    % undo the DIM permutation
    if dimArgGiven
        y = ipermute(y,perm);  
    end

    % If X is a vector, the shape of Y should follow that of P, unless an
    % explicit DIM arg was given.
    if ~dimArgGiven && isvector(x)
        y = reshape(y,size(p)); 
    end
    """

def mynanmedian(x, dim = None):
    """
    %   NANMEDIAN Median value, ignoring NaNs.
    %   M = NANMEDIAN(X) returns the sample median of X, treating NaNs as
    %   missing values.  For vector input, M is the median value of the non-NaN
    %   elements in X.  For matrix input, M is a row vector containing the
    %   median value of non-NaN elements in each column.  For N-D arrays,
    %   NANMEDIAN operates along the first non-singleton dimension.
    %
    %   NANMEDIAN(X,DIM) takes the median along the dimension DIM of X.
    %
    %   See also MEDIAN, NANMEAN, NANSTD, NANVAR, NANMIN, NANMAX, NANSUM.

    %   Copyright 1993-2004 The MathWorks, Inc.
    %   $Revision: 1.1.8.1 $  $Date: 2010/03/16 00:15:51 $
    """
    raise Exception("used np.nanmedian instead")
    if dim is None:
        y = myprctile(x, 50)
    else:
        y = myprctile(x, 50, dim)

def mynanmean(x, dim = None):
    """
    %   NANMEAN Mean value, ignoring NaNs.
    %   M = NANMEAN(X) returns the sample mean of X, treating NaNs as missing
    %   values.  For vector input, M is the mean value of the non-NaN elements
    %   in X.  For matrix input, M is a row vector containing the mean value of
    %   non-NaN elements in each column.  For N-D arrays, NANMEAN operates
    %   along the first non-singleton dimension.
    %
    %   NANMEAN(X,DIM) takes the mean along dimension DIM of X.
    %
    %   See also MEAN, NANMEDIAN, NANSTD, NANVAR, NANMIN, NANMAX, NANSUM.
    """

    # Find NaNs and set them to zero
    nans = np.isnan(x)
    x[nans] = 0
    
    if dim is None: # let sum deal with figuring out which dimension to use
        # Count up non-NaNs.
        n = np.sum(~nans, axis = 0).astype(np.float64) # first row: all 0's in python
        n[np.where(n == 0)] = np.NaN # prevent divideByZero warnings
        # Sum up non-NaNs, and divide by the number of non-NaNs.
        m = np.sum(x, axis = 0) / n
    else:
        # Count up non-NaNs.
        n = np.sum(~nans, dim).astype(np.float64)
        n[np.where(n==0)] = np.NaN # prevent divideByZero warnings
        # Sum up non-NaNs, and divide by the number of non-NaNs.
        m = np.sum(x,dim) / n

    return m

def count_profs_with_nonzero_weights(MITprofs):
    """
    careful when calling this method, the ins might behave unexpectedly b/c of intersect/ union methods used 
    """

    num_profs = len(MITprofs['prof_lon'])
    nonzero_T_ins = np.where(np.sum(MITprofs['prof_Tweight'], axis = 1) > 0)[0]
    num_nonzero_T = len(nonzero_T_ins)
    zero_weight_T_ins = np.where(np.sum(MITprofs['prof_Tweight'], axis= 1) == 0)[0]

    if 'prof_S' in MITprofs:

        nonzero_S_ins = np.where(np.sum(MITprofs['prof_Sweight'], axis = 1) > 0)[0]
        num_nonzero_S = len(nonzero_S_ins)
        zero_weight_S_ins = np.where(np.sum(MITprofs['prof_Sweight'], axis=1) == 0)[0]
        zero_weight_TS_ins = np.intersect1d(zero_weight_T_ins, zero_weight_S_ins)
        
    else:
        nonzero_S_ins = []
        num_nonzero_S = 0
        zero_weight_S_ins = []
        zero_weight_TS_ins = zero_weight_T_ins

   
    nonzero_TS_ins = np.union1d(nonzero_T_ins, nonzero_S_ins)
    num_nonzero_TS = len(nonzero_TS_ins)

    return num_nonzero_T, num_nonzero_S, num_nonzero_TS, num_profs, zero_weight_T_ins, zero_weight_S_ins, zero_weight_TS_ins, nonzero_T_ins, nonzero_S_ins, nonzero_TS_ins


def update_zero_weight_points_on_prepared_profiles(run_code, MITprofs, grid_dir):
    
    # SET INPUT PARAMETERS
    fillVal=-9999
    checkVal=-9000

    """
    % zero critera codes
    % TEST NUMBER
    % 1 : T or S weight is already zero
    % 2 : nonzero prof T or S flag
    % 3 : missing T or S
    % 4 : T or S identically zero
    % 5 : T or S outside range
    % 6;  no climatology value
    % 7:  invalid date/time
    % 8:  lat-lons outside range or at 0 N,0 E
    % 9:  high AVERAGE cost of the whole profile vs. climatology
    % 10:  high cost of a particular value vs. climatology
    % 11: test for likely bad conductivity cell.
    """

    # the total number of test
    num_tests = 11
    critera_names =  ['T or S weight already zero',
        'nonzero prof T or S flag',
        'missing T or S',
        'T or S identically zero',
        'T or S outside range',
        'no climatology value',
        'invalid date/time',
        'lat lons outside legal range or 0N, 0E',
        'high avg cost of whole prof vs. climatology',
        'high cost of a particular value vs. climatology',
        'test of likely bad conductivity cell']
    
    """
    %% MORE DETAILS
    %% TEST 4
    %   part 1: test for bad salinity sensor.  if more than half of salinity values
    %   below prof_S_subsurface_depth are prof_S_subsurface_threshold or below
    %   then flag that entire profile as bad
    %   part 2: test for individual bad T or S values.  if T or S values are outside
    %   of the prof_Tmin to prof_Tmax (or S) range, then set weight to zero

    % exclude_high_latitude_profiles_from_clim_cost : test to determines
    %          whether to keep high-lat profiles regardless of cost vs clim
    %          because clim at high lats are not reliable,

    % high_lat_cutoff : latitude poleward of which to ignore clim costs (if
    %          above flag is 1.

    % bad_profs_to_plot  :  the number of suspect profiles to plot

    %
    % plot_individual_bad_profiles : 0/1 whether to plot individual profiles

    % plot_map_bad_profiles : 0/1 whether to make a plot of bad profs locations
    """

    if run_code == '20190126':
        save_output_to_disk = 0
        zero_criteria_code = np.arange(1,12)
        
        exclude_high_latitude_profiles_from_clim_cost = 0
        
        prof_Tmin = -2
        prof_Tmax = 40
        
        prof_Smin = 20
        prof_Smax = 40
        
        prof_subsurface_min_S_threshold = [30, 34]
        prof_subsurface_min_S_threshold_depth = [50, 250]
        
        profile_avg_cost_threshold = 16
        single_datum_cost_threshold = 100
            
    elif run_code == '20190126_high_lat':
   
        save_output_to_disk = 0
        # NOTE: using 0-11 instead of 1-12
        zero_criteria_code = np.arange(1,12)
        
        # don't bother testing high latitude profiles against the
        # climatology because we don't trust the climatology.
        exclude_high_latitude_profiles_from_clim_cost = 1
        high_lat_cutoff = 60
        
        prof_Tmin = -2
        prof_Tmax = 40
    
        prof_Smin = 20
        prof_Smax = 40
        
        prof_subsurface_min_S_threshold = [30, 34]
        prof_subsurface_min_S_threshold_depth = [50, 250]
        
        profile_avg_cost_threshold = 16
        single_datum_cost_threshold = 100
            
    elif run_code ==  '20181202':
        
        save_output_to_disk = 0
        zero_criteria_code = np.arange(1,11)
        exclude_high_latitude_profiles_from_clim_cost = 0
        
        prof_Tmin = -2
        prof_Tmax = 40
        
        prof_Smin = 20
        prof_Smax = 40
        
        profile_avg_cost_threshold = 16
        single_datum_cost_threshold = 100
            
    elif run_code == '20181202_high_lat':
        
        save_output_to_disk = 0
        zero_criteria_code = np.arange(1,11)
        
        # don't bother testing high latitude profiles against the
        # climatology because we don't trust the climatology.
        exclude_high_latitude_profiles_from_clim_cost = 1
        high_lat_cutoff = 60
        
        prof_Tmin = -2
        prof_Tmax = 40
        
        prof_Smin = 20
        prof_Smax = 40
        
        profile_avg_cost_threshold = 16
        single_datum_cost_threshold = 100

    num_profs = len(MITprofs['prof_YYYYMMDD'])
    num_orig_nonzero_Tweights = len(np.where(MITprofs['prof_Tweight'])[0])
    
    if 'prof_S' in MITprofs:
        num_orig_nonzero_Sweights = len(np.where(MITprofs['prof_Sweight'])[0])
        print("START Nonzero T weights {}".format(num_orig_nonzero_Tweights))
        print("START Nonzero S weights {}".format(num_orig_nonzero_Sweights))
    else:
        print("START Nonzero T weights {}".format(num_orig_nonzero_Tweights))
        print("No S")

    
    nnt_orig, nns_orig, nnts_orig, np_orig, zwti_orig, zwsi_orig, zwtsi_orig, nzwti_orig, nzwsi_orig, nzwtsi_orig = count_profs_with_nonzero_weights(MITprofs)
    
    print(f'num profs: {np_orig:10d}\nnum nonzero T : {nnt_orig:10d}\nnum nonzero S : {nns_orig:10d}\nnum nonzero TS: {nnts_orig:10d}')

    tmp_weight_zero = np.zeros_like(MITprofs['prof_Tweight'])
    zero_T_weight_reason = copy.deepcopy(tmp_weight_zero)
  
    if 'prof_S' in MITprofs:
        zero_S_weight_reason = copy.deepcopy(tmp_weight_zero)
    
    # Make copy of MITprofs
    MITprofs_new = copy.deepcopy(MITprofs)
    for zero_critera_code_i in zero_criteria_code:
   
        print('\nCriteria : {} {}\n'.format(zero_critera_code_i, critera_names[zero_critera_code_i - 1]))
        print('------------------------------------\n')

        num_nonzero_Tweights = len(np.where(MITprofs_new['prof_Tweight'])[0])
        
        if 'prof_S' in MITprofs:
            num_nonzero_Sweights = len(np.where(MITprofs_new['prof_Sweight'])[0])
            print(f'Nonzero T weights NOW/ORIG {num_orig_nonzero_Tweights}/{num_nonzero_Tweights}\n' \
                  f'Nonzero S weights NOW/ORIG {num_nonzero_Sweights}/{num_orig_nonzero_Sweights}\n')

  
        nnt, nns, nnts, num_np, zwti, zwsi, zwtsi,nzwti, nzwsi, nzwtsi = count_profs_with_nonzero_weights(MITprofs)
        print(f'\nnum profs(n/o): {num_np:10d}\nnum nonzero T : {nnt:10d} {nnt_orig:10d}\nnum nonzero S : {nns:10d} {nns_orig:10d}\nnum nonzero TS: {nnt:10d} {nnts_orig:10d}\n')

        if zero_critera_code_i == 1: #  profiles already have zero or missing weights

            # Find indices of NaNs and non-positive values
            # ins1 = find(isnan(MITprof.prof_Tweight));
            ins1 = np.where(np.isnan(MITprofs['prof_Tweight'].flatten(order = 'F')))
            # ins2 = find(MITprof.prof_Tweight <= 0);
            ins2 = np.where(MITprofs['prof_Tweight'].flatten(order = 'F') <= 0)
            ins3 = np.union1d(ins1, ins2)

            ins3 = np.unravel_index(ins3, MITprofs['prof_Tweight'].shape, order = 'F')

            # Set values to 0 directly using boolean indexing
            MITprofs_new['prof_Tweight'][ins3] = 0   
            zero_T_weight_reason[ins3] = zero_T_weight_reason[ins3] + 2 **(zero_critera_code_i - 1)          

            # Now salinity
            if 'prof_S' in MITprofs:
                # nan weights
                # ins1 = find(isnan(MITprof.prof_Sweight));
                ins1 = np.where(np.isnan(MITprofs['prof_Sweight'].flatten(order = 'F')))
                # zero weights or negative weights
                ins2 = np.where(MITprofs['prof_Sweight'].flatten(order = 'F') <= 0)
                ins3 = np.union1d(ins1, ins2)

                ins3 = np.unravel_index(ins3, MITprofs['prof_Tweight'].shape, order = 'F')

                MITprofs_new['prof_Sweight'][ins3] = 0
                zero_S_weight_reason[ins3] = zero_S_weight_reason[ins3] + 2 ** (zero_critera_code_i - 1)
       
        if zero_critera_code_i == 2: # nonzero prof T or S flag
            
            # ['+++ Profs with nonzero T or S flags']
            ins1 = np.where(MITprofs['prof_Tflag'] > 0)
            MITprofs_new['prof_Tweight'][ins1] = 0    
            zero_T_weight_reason[ins1] = zero_T_weight_reason[ins1] + 2**(zero_critera_code_i - 1)

            if 'prof_S' in MITprofs:
                ins1 = np.where(MITprofs['prof_Sflag'] > 0)
                MITprofs_new['prof_Sweight'][ins1] = 0
                zero_S_weight_reason[ins1] = zero_S_weight_reason[ins1] + 2 ** (zero_critera_code_i -1)

        if zero_critera_code_i == 3: #  missing T or S

            ins1 = np.where(np.isnan(MITprofs['prof_T'].flatten(order = 'F')))
            ins2 = np.where(MITprofs['prof_T'].flatten(order = 'F') <= checkVal)
            ins3 = np.union1d(ins1, ins2)

            ins3 = np.unravel_index(ins3, MITprofs['prof_T'].shape, order = 'F')

            MITprofs_new['prof_T'][ins3] = fillVal
            MITprofs_new['prof_Tweight'][ins3] = 0
            zero_T_weight_reason[ins3] = zero_T_weight_reason[ins3] + 2 ** (zero_critera_code_i - 1)
     
            if 'prof_S' in MITprofs:
                ins1 = np.where(np.isnan(MITprofs['prof_S'].flatten(order = 'F')))
                ins2 = np.where(MITprofs['prof_S'].flatten(order = 'F') <= checkVal)
                ins3 = np.union1d(ins1, ins2)

                ins3 = np.unravel_index(ins3, MITprofs['prof_S'].shape, order = 'F')
                
                MITprofs_new['prof_S'][ins3] = fillVal
                MITprofs_new['prof_Sweight'][ins3] = 0
                zero_S_weight_reason[ins3] = zero_S_weight_reason[ins3] + 2**(zero_critera_code_i - 1)      
      
        if zero_critera_code_i == 4: # T or S identically zero

            ins1 = np.where(MITprofs['prof_T'] == 0)
            # if so, put fill val there, because identically
            # zero is most likely because of missing data that
            # slipped through the cracks
            MITprofs_new['prof_T'][ins1] = fillVal
            MITprofs_new['prof_Tweight'][ins1] = 0
            
            zero_T_weight_reason[ins1] = zero_T_weight_reason[ins1] + 2 ** (zero_critera_code_i - 1)
            
            if 'prof_S' in MITprofs:
                ins1 = np.where(MITprofs['prof_S'] == 0)
                
                # if so, put fill val there, because identically
                # zero is most likely because of missing data that
                # slipped through the cracks
                MITprofs_new['prof_S'][ins1] = fillVal
                MITprofs_new['prof_Sweight'][ins1] = 0
                zero_S_weight_reason[ins1] = zero_S_weight_reason[ins1] + 2**(zero_critera_code_i -1)

        if zero_critera_code_i == 5: # T or S outside some legal range
            # Profs with T or S outside legal range 
            # find those that are less than the legal min
            # but not already set to the -9999 missing data
            ins1 = np.where((MITprofs['prof_T'].flatten(order = 'F') < prof_Tmin) & (MITprofs['prof_T'].flatten(order = 'F')  > checkVal))
            ins2 = np.where((MITprofs['prof_T'].flatten(order = 'F')  > prof_Tmax) & (MITprofs['prof_T'].flatten(order = 'F')  > checkVal))
            ins3 = np.union1d(ins1, ins2)

            ins3 = np.unravel_index(ins3, MITprofs['prof_T'].shape, order = 'F')
            
            MITprofs_new['prof_Tweight'][ins3] = 0
            zero_T_weight_reason[ins3] = zero_T_weight_reason[ins3] + 2 ** (zero_critera_code_i - 1)

            if 'prof_S' in MITprofs:
                # PART 2: FIND S VALUES OUTSIDE OF RANGE.
                # find those that are less than the legal min
                # but not already set to the -9999 missing data
                ins1 = np.where((MITprofs['prof_S'].flatten(order = 'F') < prof_Smin) & (MITprofs['prof_S'].flatten(order = 'F') > checkVal))
                ins2 = np.where((MITprofs['prof_S'].flatten(order = 'F') > prof_Smax) & (MITprofs['prof_S'].flatten(order = 'F') > checkVal))
                ins3 = np.union1d(ins1, ins2)

                ins3 = np.unravel_index(ins3, MITprofs['prof_S'].shape, order = 'F')

                MITprofs_new['prof_Sweight'][ins3] = 0
                zero_S_weight_reason[ins3] = zero_S_weight_reason[ins3] + 2**(zero_critera_code_i -1)

        if zero_critera_code_i == 6: # missing climatology value
 
            ins1 = np.where(MITprofs['prof_Tclim'].flatten(order = 'F') <= checkVal)
            ins2 = np.where(np.isnan(MITprofs['prof_Tclim'].flatten(order = 'F') ))
            ins3 = np.where(MITprofs['prof_Tclim'].flatten(order = 'F')  == 0)
            ins4 = np.union1d(ins3, np.union1d(ins1, ins2))

            ins4 = np.unravel_index(ins4, MITprofs['prof_Tclim'].shape, order = 'F')

            MITprofs_new['prof_Tweight'][ins4] = 0
            zero_T_weight_reason[ins4] = zero_T_weight_reason[ins4] + 2**(zero_critera_code_i - 1)
      
            if 'prof_S' in MITprofs:
                
                ins1 = np.where(MITprofs['prof_Sclim'].flatten(order = 'F') <= checkVal)
                ins2 = np.where(np.isnan(MITprofs['prof_Sclim'].flatten(order = 'F')))
                ins3 = np.where(MITprofs['prof_Sclim'].flatten(order = 'F') == 0)
                ins4 = np.union1d(ins3, np.union1d(ins1, ins2))

                ins4 = np.unravel_index(ins4, MITprofs['prof_Sclim'].shape, order = 'F')
                
                MITprofs_new['prof_Sweight'][ins4] = 0
                zero_S_weight_reason[ins4] = zero_S_weight_reason[ins4] + 2**(zero_critera_code_i -1)

        if zero_critera_code_i == 7: # illegal dates/times

            all_years = np.floor(MITprofs['prof_YYYYMMDD']/1e4)
            years = np.unique(all_years)

            y = np.zeros(num_profs, dtype=int)
            m = np.zeros(num_profs, dtype=int)
            d = np.zeros(num_profs, dtype=int)
            
            for i in np.arange(num_profs):
                tmp = str(MITprofs['prof_YYYYMMDD'][i])
                y[i]  = int(tmp[0:4])
                m[i]  = int(tmp[4:6])
                d[i]  = int(tmp[6:8])

            todays_year = dt.datetime.now().year

            # bad years are pre 1950 and after today's year
            bad_years = np.where((y < 1950) | (y > todays_year))
            bad_mons  = np.where((m < 1)|(m > 12))
            bad_days  = np.where((d < 1) | (d > 31))
            bad_times = np.where((MITprofs['prof_HHMMSS'] < 0) | (MITprofs['prof_HHMMSS'] > 240000))

            bad_profs = np.union1d(bad_times, np.union1d(bad_days, np.union1d(bad_years, bad_mons)))

            bad_profs = np.unravel_index(bad_profs, MITprofs_new['prof_Tweight'].shape, order = 'F')

            MITprofs_new['prof_Tweight'][bad_profs, :] = 0
            zero_T_weight_reason[bad_profs] = zero_T_weight_reason[bad_profs] + 2**(zero_critera_code_i -1)
   
            if 'prof_S' in MITprofs:
                MITprofs_new['prof_Sweight'][bad_profs, :] = 0
                zero_S_weight_reason[bad_profs] = zero_S_weight_reason[bad_profs] + 2**(zero_critera_code_i -1)


        if zero_critera_code_i == 8: # lat-lon out of bounds or  0 deg N and 0 deg E
            
            lats = MITprofs['prof_lat']
            lons = MITprofs['prof_lon']
            
            outside_lat_lim_ins = np.where((lats < -90) | (lats > 90))
            outside_lon_lim_ins = np.where((lons < -180) | (lons > 180))
            zero_lat_lon_ins = np.where((lats == 0) & (lons == 0))

            bad_profs = np.union1d(outside_lat_lim_ins, np.union1d(outside_lon_lim_ins, zero_lat_lon_ins))
            bad_profs = np.unravel_index(bad_profs, MITprofs_new['prof_Tweight'].shape, order = 'F')

            MITprofs_new['prof_Tweight'][bad_profs, :] = 0
            zero_T_weight_reason[bad_profs] = zero_T_weight_reason[bad_profs] + 2**(zero_critera_code_i -1)

            if 'prof_S' in MITprofs:
                MITprofs_new['prof_Sweight'][bad_profs, :] = 0
                zero_S_weight_reason[bad_profs] = zero_S_weight_reason[bad_profs] + 2**(zero_critera_code_i -1)

        if zero_critera_code_i == 9 or zero_critera_code_i == 10: # high cost vs. climatology
            
            # T FIRST 
            tmpT = MITprofs['prof_T']
            tmpC = MITprofs['prof_Tclim']
            tmpW = MITprofs['prof_Tweight']
            
            tmpT[np.where(tmpT < checkVal)] = np.NaN
            tmpT[np.where(tmpT == 0)] = np.NaN
            
            tmpC[np.where(tmpC < checkVal)] = np.NaN
            tmpC[np.where(tmpC == 0)] = np.NaN
            
            tmpW[np.where(tmpW < 0)] = np.NaN
            
            T_cost_vs_clim = (tmpT - tmpC)**2 * tmpW

            if 'prof_S' in MITprofs:
                tmpS = MITprofs['prof_S']
                tmpC = MITprofs['prof_Sclim']
                tmpW = MITprofs['prof_Sweight']
                
                tmpS[np.where(tmpS < checkVal)] = np.NaN
                tmpS[np.where(tmpS == 0)] = np.NaN
                
                tmpC[np.where(tmpC < checkVal)] = np.NaN
                tmpC[np.where(tmpC == 0)] = np.NaN
                
                tmpW[np.where(tmpW < 0)] = np.NaN

                S_cost_vs_clim = (tmpS - tmpC)**2 * tmpW
             
            if exclude_high_latitude_profiles_from_clim_cost:
                # find all profiles that are outside of high
                # latitudes (high_lat_cutoff), e.g., -60 to 60)
                too_high = np.where(MITprofs['prof_lat'] <= -high_lat_cutoff)
                too_low  = np.where(MITprofs['prof_lat'] >= high_lat_cutoff)

                high_lat_profs_ins = np.union1d(too_high, too_low)
                high_lat_profs_ins_unraveled = np.unravel_index(high_lat_profs_ins, MITprofs_new['prof_Tweight'].shape, order = 'F')
  
                high_lat_points_mat = np.zeros(MITprofs['prof_Tweight'].shape)
                high_lat_points_mat[high_lat_profs_ins_unraveled, :] = 1

                high_lat_points_ins = np.where(high_lat_points_mat.flatten(order = 'F') == 1)
            
            if zero_critera_code_i == 9: # CHECK AVERAGE COST VS CLIM

                avg_T_costs = mynanmean(T_cost_vs_clim, 1)
                bad_profs = np.where(avg_T_costs >= profile_avg_cost_threshold)

                if exclude_high_latitude_profiles_from_clim_cost:
                    bad_profs = np.setdiff1d(bad_profs, high_lat_profs_ins)
                
                MITprofs_new['prof_Tweight'][bad_profs, :] = 0
                zero_T_weight_reason[bad_profs, :] = zero_T_weight_reason[bad_profs, :] + 2**(zero_critera_code_i -1)
               
                if 'prof_S' in MITprofs:
                
                    avg_S_costs = mynanmean(S_cost_vs_clim, 1)
                    bad_profs = np.where(avg_S_costs >= profile_avg_cost_threshold)
            
                    if exclude_high_latitude_profiles_from_clim_cost:
                        bad_profs = np.setdiff1d(bad_profs, high_lat_profs_ins)
          
                    MITprofs_new['prof_Sweight'][bad_profs, :] = 0
                    zero_S_weight_reason[bad_profs, :] = zero_S_weight_reason[bad_profs, :] + 2**(zero_critera_code_i -1)
                
            if zero_critera_code_i == 10: # individual points exceed cost threshold

                bad_prof_data = np.where(T_cost_vs_clim.flatten(order = 'F') >= single_datum_cost_threshold)
    
                if exclude_high_latitude_profiles_from_clim_cost:
                    bad_prof_data = np.setdiff1d(bad_prof_data, high_lat_points_ins[0])
                
                bad_prof_data = np.unravel_index(bad_prof_data, MITprofs_new['prof_Tweight'].shape, order = 'F')

                MITprofs_new['prof_Tweight'][bad_prof_data] = 0
                zero_T_weight_reason[bad_prof_data] = zero_T_weight_reason[bad_prof_data] + 2**(zero_critera_code_i -1)
               
                if 'prof_S' in MITprofs:

                    bad_prof_data = np.where(S_cost_vs_clim.flatten(order = 'F') >= single_datum_cost_threshold)
                    
                    if exclude_high_latitude_profiles_from_clim_cost:
                        bad_prof_data = np.setdiff1d(bad_prof_data, high_lat_points_ins[0])

                    bad_prof_data = np.unravel_index(bad_prof_data, MITprofs_new['prof_Sweight'].shape, order = 'F')
     
                    MITprofs_new['prof_Sweight'][bad_prof_data] = 0
                    zero_S_weight_reason[bad_prof_data] = zero_S_weight_reason[bad_prof_data] + 2**(zero_critera_code_i -1)

        if zero_critera_code_i == 11: # test for possible bad conducivity cell.
            if 'prof_S' in MITprofs:
                
                # PART 1, FIND BAD CONDUCTIVITY CELLS

                num_tests = len(prof_subsurface_min_S_threshold_depth)    
                # pull the salinity field
                tmpS = MITprofs_new['prof_S']
                bad_cond_ins = []

                for ii in np.arange(num_tests):

                    prof_S_sst = prof_subsurface_min_S_threshold[ii]
                    prof_S_sstd = prof_subsurface_min_S_threshold_depth[ii]
                    
                    # find profile depths that are shallower than the
                    # subsurface depth threshold
                    ssdi = np.where(np.abs(MITprofs_new['prof_depth']) <= np.abs(prof_S_sstd))
                    
                    # find data where S <= threshold
                    ins1 = np.where((tmpS <= prof_S_sst) & (MITprofs['prof_S'] > checkVal))
                    
                    # find data where S > threshold
                    ins2 = np.where((tmpS >= prof_S_sst) & (MITprofs['prof_S'] > checkVal))
                   
                    # make a mask of nans, one for each value of prof_S
                    tmp = np.full(MITprofs_new['prof_S'].shape, np.nan)
                    tmp[ins1] = 1
                    tmp[ins2] = 0
                    
                    # exclude points in the upper ocean above the
                    # prof_S_subsurface_threshold
                    tmp[:, ssdi] = np.nan
            
                    # find the median value of tmp
                    # x = mynanmedian(tmp');
                    x = np.nanmedian(tmp.T, axis = 0)

                    # identify as likely bad profiles all of those
                    # profiles where the median value of S below the
                    # treshold depth is less than the threshold salinity
                    ins3 = np.where(x == 1)

                    bad_cond_ins = np.union1d(bad_cond_ins, ins3)
                
                bad_cond_ins = bad_cond_ins.astype(int)

                # set the weights of all of those profiles to zero
                MITprofs_new['prof_Sweight'][bad_cond_ins,:] = 0
                
                # record the reason that we're zeroing it out.
                zero_S_weight_reason[bad_cond_ins,:] = zero_S_weight_reason[bad_cond_ins,:] + 2**(zero_critera_code_i -1)

                # ALSO DO T IN CASE IT IS THE T MEASURMENT
                # THAT CAUSED S TO BE CRAZY.
                # set the weights of all of those profiles to zero
                MITprofs_new['prof_Tweight'][bad_cond_ins,:] = 0
                
                # record the reason that we're zeroing it out.
                zero_T_weight_reason[bad_cond_ins,:] = zero_T_weight_reason[bad_cond_ins,:] + 2**(zero_critera_code_i -1)
            
        MITprofs_new['prof_Tweight_code'] = zero_T_weight_reason
        num_nans = np.where(np.isnan(zero_T_weight_reason))[0]
        if len(num_nans) > 0:
            raise Exception('nans found in prof t weight code')
            
        if 'prof_S' in MITprofs:
            MITprofs_new['prof_Sweight_code'] = zero_S_weight_reason
            num_nans = np.where(np.isnan(zero_S_weight_reason))[0]
            if len(num_nans) > 0:
                raise Exception('nans found in prof s weight code')

    MITprofs.update(MITprofs_new)

    """
    # NOTE: code for testing in case issues arise in future 
    mat_contents = sio.loadmat('/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/converted_to_MITprof/zero_S_wr.mat')
    s = mat_contents['zero_S_weight_reason']

    t = 0
    for i in np.arange(32364):
        for j in np.arange(137):
            if s[i, j] != zero_S_weight_reason[i, j]:
                print("i: {} j: {}".format(i, j))
                print("s {}".format(s[i, j]))
                print("pyt {}".format(zero_S_weight_reason[i, j]))
                print("============")

                print(bin(s[i, j]))
                print(bin(int(zero_S_weight_reason[i, j])))
                print("--------------------")
                t = t+1

    print(np.array_equal(s, zero_S_weight_reason))
    mat_contents = sio.loadmat('/home/sweet/Desktop/ECCO-Insitu-Ian/Original-Matlab-Dest/converted_to_MITprof/zero_T_wr.mat')
    t = mat_contents['zero_T_weight_reason']
    print(np.array_equal(t, zero_T_weight_reason))
    """


def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    print("update_zero_weight_points_on_prepared_profiles")
    update_zero_weight_points_on_prepared_profiles(run_code, MITprofs, grid_dir)

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
        MITprofs = MITprof_read(file, 7)

    run_code = '20190126_high_lat'
    grid_dir = "hehe"

    # Convert all masked arrs to non-masked types
    for keys in MITprofs.keys():
        if ma.isMaskedArray(MITprofs[keys]):
            MITprofs[keys] = MITprofs[keys].filled(np.NaN)
    

    main(run_code, MITprofs, grid_dir)