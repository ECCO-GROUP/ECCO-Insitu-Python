import glob
import os
import numpy as np

from tools import MITprof_read

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
    critera_names =  {'T or S weight already zero',
        'nonzero prof T or S flag',
        'missing T or S',
        'T or S identically zero',
        'T or S outside range',
        'no climatology value',
        'invalid date/time',
        'lat lons outside legal range or 0N, 0E',
        'high avg cost of whole prof vs. climatology',
        'high cost of a particular value vs. climatology',
        'test of likely bad conductivity cell'}
    
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


    """
    num_profs = length(MITprof.prof_YYYYMMDD);
    num_orig_nonzero_Tweights = length(find(MITprof.prof_Tweight));
    
    
    if isfield(MITprof,'prof_S')
        num_orig_nonzero_Sweights = length(find(MITprof.prof_Sweight));
        fprintf(['START Nonzero T weights ' num2str(num_orig_nonzero_Tweights) '\n' ...
            'START Nonzero S weights ' num2str(num_orig_nonzero_Sweights) '\n'] )
    else
        fprintf(['START Nonzero T weights ' num2str(num_orig_nonzero_Tweights) '\n' ...
            'No S \n'])
    end
    
    nnt_orig, nns_orig, nnts_orig, np_orig, zwti_orig, zwsi_orig, zwtsi_orig, nzwti_orig, nzwsi_orig, nzwtsi_orig = count_profs_with_nonzero_weights(MITprofs)
    fprintf('\nnum profs     : %10i \nnum nonzero T : %10i \nnum nonzero S : %10i \nnum nonzero TS: %10i \n', ...
        np_orig ,nnt_orig,nns_orig,nnts_orig)
    
    
    tmp_weight_zero = zeros(size(MITprof.prof_Tweight));
    
    zero_T_weight_reason = tmp_weight_zero;
    if isfield(MITprof,'prof_S')
        zero_S_weight_reason = tmp_weight_zero;
    end
    
    %% Make a copy of MITprof.
    MITprof_new = MITprof;
    """
    """
    for zero_critera_code_i = zero_criteria_code:
        
        fprintf(['\nCriteria : ' padzero(zero_critera_code_i,2) ' ' critera_names{zero_critera_code_i} '\n'])
        fprintf(['------------------------------------\n'])
        num_nonzero_Tweights = length(find(MITprof_new.prof_Tweight));
        
        if isfield(MITprof,'prof_S')
            num_nonzero_Sweights = length(find(MITprof_new.prof_Sweight));
            fprintf(['Nonzero T weights NOW/ORIG ' num2str(num_orig_nonzero_Tweights) '/' ...
                num2str(num_nonzero_Tweights) '\n' ...
                    'Nonzero S weights NOW/ORIG ' num2str(num_nonzero_Sweights) '/' ...
                    num2str(num_orig_nonzero_Sweights) '\n'])
        end
        
        [nnt, nns, nnts, np, zwti, zwsi, zwtsi,nzwti, nzwsi, nzwtsi] = count_profs_with_nonzero_weights(MITprof);
        fprintf('\nnum profs(n/o): %10i \nnum nonzero T : %10i %10i \nnum nonzero S : %10i %10i \nnum nonzero TS: %10i %10i\n', ...
            np, nnt, nnt_orig, nns, nns_orig, nnt, nnts_orig)
    
        switch zero_critera_code_i
                
                case 1 %  profiles already have zero or missing weights
                    %%
                    %['+++ Profs with zero or missing weights']
                    % nan weights
                    ins1 = find(isnan(MITprof.prof_Tweight));
                    % zero weights or negative weights
                    ins2 = find(MITprof.prof_Tweight <= 0);
                    
                    ins3 = union(ins1,ins2);
                    
                    %['# prof T weight is nan, is leq ZERO']
                    [length(ins1), length(ins2)];
                    
                    MITprof_new.prof_Tweight(ins3) = 0;
                    
                    zero_T_weight_reason(ins3) = ...
                        zero_T_weight_reason(ins3) + 2^(zero_critera_code_i -1);
                    
                    % Now salinity
                    if isfield(MITprof,'prof_S')
                        % nan weights
                        ins1 = find(isnan(MITprof.prof_Sweight));
                        % zero weights or negative weights
                        ins2 = find(MITprof.prof_Sweight <= 0);
                        
                        ins3 = union(ins1,ins2);
                        
                    % ['# prof S weight is nan, is leq ZERO']
                    % [length(ins1), length(ins2)]
                        
                        MITprof_new.prof_Sweight(ins3) = 0;
                        
                        zero_S_weight_reason(ins3) = ...
                            zero_S_weight_reason(ins3) + 2^(zero_critera_code_i -1);
                    end
                    
                case 2 % nonzero prof T or S flag
                    %%
                    %['+++ Profs with nonzero T or S flags']
                    clear ins*;
                    ins1 = find(MITprof.prof_Tflag > 0);
                    
                    %['# prof T flag is > 0']
                    %[length(ins1)]
                    
                    MITprof_new.prof_Tweight(ins1) = 0;
                    
                    zero_T_weight_reason(ins1) = ...
                        zero_T_weight_reason(ins1) + 2^(zero_critera_code_i -1);
                    
                    % Now salinity
                    if isfield(MITprof,'prof_S')
                        clear ins*;
                        ins1 = find(MITprof.prof_Sflag > 0);
                        
                        %['# prof S flag is > 0']
                        %[length(ins1)]
                        
                        MITprof_new.prof_Sweight(ins1) = 0;
                        
                        zero_S_weight_reason(ins1) = ...
                            zero_S_weight_reason(ins1) + 2^(zero_critera_code_i -1);
                    end
                    
                case 3 %  missing T or S
                    %%
                    %['+++ Profs with missing T or S ']
                    clear ins*;
                    
                    % is T nan or fillVal?
                    ins1 = find(isnan(MITprof.prof_T));
                    ins2 = find(MITprof.prof_T <= checkVal);
                    ins3 = union(ins1,ins2);
                    
                    %['# prof T is nan, is < -9000']
                % [length(ins1), length(ins2)]
                    
                    % if so, put fill val there (just in case we had
                    % NaN)
                    MITprof_new.prof_T(ins3) = fillVal;
                    MITprof_new.prof_Tweight(ins3) = 0;
                    
                    zero_T_weight_reason(ins3) = ...
                        zero_T_weight_reason(ins3) + 2^(zero_critera_code_i -1);
                    
                    % Now salinity
                    if isfield(MITprof,'prof_S')
                        clear ins*;
                        
                        % is S nan or fillVal?
                        ins1 = find(isnan(MITprof.prof_S));
                        ins2 = find(MITprof.prof_S <= checkVal);
                        ins3 = union(ins1,ins2);
                        
                    % ['# prof S is nan, is < -9000']
                    % [length(ins1), length(ins2)]
                        
                        % if so, put fill val there (just in case we had
                        % NaN)
                        MITprof_new.prof_S(ins3) = fillVal;
                        MITprof_new.prof_Sweight(ins3) = 0;
                        
                        zero_S_weight_reason(ins3) = ...
                            zero_S_weight_reason(ins3) + 2^(zero_critera_code_i -1);
                    end
                    
                    
                case 4 % T or S identically zero
                    %%
                    %['+++ Profs with identically zero T or S ']
                    clear ins*;
                    
                    % T is identically zero
                    ins1 = find(MITprof.prof_T == 0);
                    
                % ['# prof T is identically zero']
                % [length(ins1)]
                    
                    % if so, put fill val there, because identically
                    % zero is most likely because of missing data that
                    % slipped through the cracks
                    MITprof_new.prof_T(ins1) = fillVal;
                    MITprof_new.prof_Tweight(ins1) = 0;
                    
                    zero_T_weight_reason(ins1) = ...
                        zero_T_weight_reason(ins1) + 2^(zero_critera_code_i -1);
                    
                    % Now salinity
                    if isfield(MITprof,'prof_S')
                        %%
                        clear ins*;
                        
                        % S is identically zero
                        ins1 = find(MITprof.prof_S == 0);
                        
                    % ['# prof S is identically zero']
                    % [length(ins1)]
                        
                        % if so, put fill val there, because identically
                        % zero is most likely because of missing data that
                        % slipped through the cracks
                        MITprof_new.prof_S(ins1) = fillVal;
                        MITprof_new.prof_Sweight(ins1) = 0;
                        
                        zero_S_weight_reason(ins1) = ...
                            zero_S_weight_reason(ins1) + 2^(zero_critera_code_i -1);
                    end
                    
                case 5 % T or S outside some legal range
                    %%
                    %['+++ Profs with  T or S outside legal range']
                    clear ins*;
                    % find those that are less than the legal min
                    % but not already set to the -9999 missing data
                    ins1 = find(MITprof.prof_T < prof_Tmin & ...
                        MITprof.prof_T > checkVal);
                    
                    ins2 = find(MITprof.prof_T > prof_Tmax & ...
                        MITprof.prof_T > checkVal);
                    
                    ins3 = union(ins1,ins2);
                    
                    MITprof_new.prof_Tweight(ins3) = 0;
                    
                    zero_T_weight_reason(ins3) = ...
                        zero_T_weight_reason(ins3) + 2^(zero_critera_code_i -1);
                    
                % ['# too cold, # too warm']
                % [length(ins1) length(ins2)]
                    
                    
                    if isfield(MITprof,'prof_S')
                        clear ins*;
                        
                        % PART 2: FIND S VALUES OUTSIDE OF RANGE.
                        % find those that are less than the legal min
                        % but not already set to the -9999 missing data
                        ins1 = find(MITprof.prof_S < prof_Smin & ...
                            MITprof.prof_S > checkVal);
                        
                        ins2 = find(MITprof.prof_S > prof_Smax & ...
                            MITprof.prof_S > checkVal);
                        
                    %  ['# too fresh, # too salty']
                    %  [length(ins1) length(ins2)]
                        ins3 = union(ins1,ins2);
                        
                        MITprof_new.prof_Sweight(ins3) = 0;
                        
                        zero_S_weight_reason(ins3) = ...
                            zero_S_weight_reason(ins3) + 2^(zero_critera_code_i -1);
                    end
                    
                case 6 % missing climatology value
                    %%
                    %['+++ Profs with missing T or S clim values']
                    clear ins*;
                    
                    ins1 = find(MITprof.prof_Tclim <= checkVal);
                    ins2 = find(isnan(MITprof.prof_Tclim));
                    ins3 = find(MITprof.prof_Tclim == 0);
                    ins4 = union(union(ins1,ins2),ins3);
                    
                    length(ins4);
                    
                    MITprof_new.prof_Tweight(ins4) = 0;
                    
                % ['# T clim (3): <= checkVal, is nan, is zero']
                % [length(ins1) length(ins2) length(ins3)]
                    
                    zero_T_weight_reason(ins4) = ...
                        zero_T_weight_reason(ins4) + 2^(zero_critera_code_i -1);
                    
                    if isfield(MITprof,'prof_S')
                        clear ins*;
                        
                        ins1 = find(MITprof.prof_Sclim <= checkVal);
                        ins2 = find(isnan(MITprof.prof_Sclim));
                        ins3 = find(MITprof.prof_Sclim == 0);
                        ins4 = union(union(ins1,ins2),ins3);
                        
                    % length(ins4)
                        
                        MITprof_new.prof_Sweight(ins4) = 0;
                        
                    % ['# S clim (3): <= check, is nan, is zero']
                    % [length(ins1) length(ins2) length(ins3)]
                        
                        zero_S_weight_reason(ins4) = ...
                            zero_S_weight_reason(ins4) + 2^(zero_critera_code_i -1);
                    end
                    
                    
                case 7 % illegal dates/times
                    %%
                % ['+++ Profs with illegal dates/times']
                    clear ins*;
                    
                    all_years = floor(MITprof.prof_YYYYMMDD/1e4);
                    years = unique(all_years);
                    
                    clear y m d;
                    for i = 1:num_profs
                        tmp = num2str(MITprof.prof_YYYYMMDD(i));
                        y(i)  = str2num(tmp(1:4));
                        m(i)  = str2num(tmp(5:6));
                        d(i)  = str2num(tmp(7:8));
                    end
                    
                    % get current date
                    s = date;
                    
                    todays_year = str2num(s(end-3:end));
                    
                    % bad years are pre 1950 and after today's year
                    bad_years_a = find(y < 1950);
                    bad_years_b = find(y > todays_year);
                    bad_mons_a  = find(m < 1);
                    bad_mons_b  = find(m > 12);
                    bad_days_a  = find(d < 1);
                    bad_days_b  = find(d > 31);
                    
                    bad_years = union(bad_years_a, bad_years_b);
                    bad_mons = union(bad_mons_a, bad_mons_b);
                    bad_days = union(bad_days_a, bad_days_b);
                    
                    bad_times_A = find(MITprof.prof_HHMMSS < 0);
                    bad_times_B = find(MITprof.prof_HHMMSS > 240000);
                    
                    bad_profs = union(union(union(union(bad_years, ...
                        bad_mons), ...
                        bad_days), bad_times_A), bad_times_B);
                    
                    all_profs = 1:num_profs;
                    
                    good_profs = setdiff(all_profs, bad_profs);
                    
                % ['date/time check: # profs, # good, # bad']
                % [length(all_profs) length(good_profs) ...
                %     length(bad_profs)]
                    
                    MITprof_new.prof_Tweight(bad_profs,:) = 0;
                    zero_T_weight_reason(bad_profs) = ...
                        zero_T_weight_reason(bad_profs) + 2^(zero_critera_code_i -1);
                    
                    if isfield(MITprof,'prof_S')
                        MITprof_new.prof_Sweight(bad_profs,:) = 0;
                        zero_S_weight_reason(bad_profs) = ...
                            zero_S_weight_reason(bad_profs) + 2^(zero_critera_code_i -1);
                    end
                    
                case 8 % lat-lon out of bounds or  0 deg N and 0 deg E
                    %%
                % ['+++ Profs with illegal lats/lons']
                    
                    clear ins* bad_profs*;
                    %['check to see if lat-lon = 0 or out of bounds']
                    clear bad_profs good_profs all_profs all_ins;
                    
                    ['bad lats lons check'];
                    lats = MITprof.prof_lat;
                    lons = MITprof.prof_lon;
                    
                    outside_lat_lim_ins = ...
                        find(lats < -90 | lats > 90);
                    
                    outside_lon_lim_ins = ...
                        find(lons < -180 | lons > 180);
                    
                    zero_lat_lon_ins = find(lats == 0 & lons == 0);
                    
                    bad_profs = union(union(outside_lat_lim_ins,...
                        outside_lon_lim_ins), zero_lat_lon_ins);
                    
                    all_profs = 1:num_profs;
                    good_profs = setdiff(all_profs, bad_profs);
                    
                    ['lat/lon check: outside lat lims, outside lon lims, zero lat/lon'];
                    [length(outside_lat_lim_ins) ...
                        length(outside_lon_lim_ins) ...
                        length(zero_lat_lon_ins)];
                    
                    ['all profs, good profs, bad_profs'];
                    [length(all_profs) length(good_profs) length(bad_profs)];
                    
                    % set the weights of profiles that are out of
                    % bounds to zero.
                    
                    MITprof_new.prof_Tweight(bad_profs,:) = 0;
                    zero_T_weight_reason(bad_profs) = ...
                        zero_T_weight_reason(bad_profs) + 2^(zero_critera_code_i -1);
                    
                    if isfield(MITprof,'prof_S')
                        MITprof_new.prof_Sweight(bad_profs,:) = 0;
                        zero_S_weight_reason(bad_profs) = ...
                            zero_S_weight_reason(bad_profs) + 2^(zero_critera_code_i -1);
                    end
                    
                case {9,10} % high cost vs. climatology
                    %%
                    
                    clear ins*;
                    clear bad_profs* good_profs*;
                    
                    % T FIRST
                    tmpT = MITprof.prof_T;
                    tmpC = MITprof.prof_Tclim;
                    tmpW = MITprof.prof_Tweight;
                    
                    tmpT(find(tmpT < checkVal))=NaN;
                    tmpT(find(tmpT ==0))=NaN;
                    
                    tmpC(find(tmpC < checkVal))=NaN;
                    tmpC(find(tmpC == 0)) = NaN;
                    
                    tmpW(find(tmpW < 0))=NaN;
                    
                    T_cost_vs_clim = ...
                        (tmpT - tmpC).^2 .* tmpW;
                    
                    % S NEXT
                    if isfield(MITprof,'prof_S')
                        tmpS = MITprof.prof_S;
                        tmpC = MITprof.prof_Sclim;
                        tmpW = MITprof.prof_Sweight;
                        
                        tmpS(find(tmpS < checkVal))=NaN;
                        tmpS(find(tmpS ==0))=NaN;
                        
                        tmpC(find(tmpC < checkVal))=NaN;
                        tmpC(find(tmpC == 0)) = NaN;
                        
                        tmpW(find(tmpW < 0))=NaN;
                        
                        S_cost_vs_clim = ...
                            (tmpS - tmpC).^2 .* tmpW;
                    end
                    
                    if exclude_high_latitude_profiles_from_clim_cost
                        % findn all profiles that are outside of high
                        % latitudes (high_lat_cutoff), e.g., -60 to 60)
                        too_high = find(MITprof.prof_lat <= -high_lat_cutoff);
                        too_low  = find(MITprof.prof_lat >= high_lat_cutoff);
                        high_lat_profs_ins = union(too_high, too_low);
                        
                        high_lat_points_mat = ...
                            zeros(size(MITprof.prof_Tweight));
                        
                    % ['Number of high lat profs']
                    % length(high_lat_profs_ins)
                        
                        high_lat_points_mat(high_lat_profs_ins,:) = 1;
                        high_lat_points_ins = ...
                            find(high_lat_points_mat == 1);
                    end
                    
                    
                    if zero_critera_code_i == 9 % CHECK AVERAGE COST VS CLIM
                        %%
                        %['+++ Profs with high AVG cost vs clim']
                        clear bad*p*;
                        avg_T_costs = mynanmean(T_cost_vs_clim,2);
                        bad_profs = find(avg_T_costs >= ....
                            profile_avg_cost_threshold);
                        
                        if exclude_high_latitude_profiles_from_clim_cost
                            bad_profs = setdiff(bad_profs, ...
                                high_lat_profs_ins);
                        end
                        
                    % ['AVG cost vs clim T: num profs, bad T profs']
                    % [num_profs length(bad_profs)]
                        
                        MITprof_new.prof_Tweight(bad_profs,:) = 0;
                        zero_T_weight_reason(bad_profs,:) = ...
                            zero_T_weight_reason(bad_profs,:) + 2^(zero_critera_code_i -1);
                        
                        if isfield(MITprof,'prof_S')
                            clear bad*p*;
                            avg_S_costs = mynanmean(S_cost_vs_clim,2);
                            bad_profs = find(avg_S_costs >= ...
                                profile_avg_cost_threshold);
                            
                            if exclude_high_latitude_profiles_from_clim_cost
                                bad_profs = setdiff(bad_profs, ...
                                    high_lat_profs_ins);
                            end
                            
                        %  ['AVG cost vs clim S: num profs, bad S profs']
                        %  [num_profs length(bad_profs)]
                            
                            MITprof_new.prof_Sweight(bad_profs,:) = 0;
                            zero_S_weight_reason(bad_profs,:) = ...
                                zero_S_weight_reason(bad_profs,:) + 2^(zero_critera_code_i -1);
                        end
                    end
                    
                    if zero_critera_code_i == 10 % individual points exceed cost threshold
                        %%
                    % ['+++ POINTS with high INDIVIDUAL cost vs clim']
                        clear bad*p*;
                        
                        bad_prof_data = find(T_cost_vs_clim >= ...
                            single_datum_cost_threshold);
                        
                        if exclude_high_latitude_profiles_from_clim_cost
                            bad_prof_data = setdiff(bad_prof_data, ...
                                high_lat_points_ins);
                        end
                        
                    % ['BAD SINGLE VALUE cost vs clim T: num profs, bad T profs']
                    % [num_profs length(bad_prof_data)]
                        
                        MITprof_new.prof_Tweight(bad_prof_data) = 0;
                        zero_T_weight_reason(bad_prof_data) = ...
                            zero_T_weight_reason(bad_prof_data) + 2^(zero_critera_code_i -1);
                        
                        if isfield(MITprof,'prof_S')
                            clear bad*p*;
                            
                            bad_prof_data = find(S_cost_vs_clim >= ...
                                single_datum_cost_threshold);
                            
                            if exclude_high_latitude_profiles_from_clim_cost
                                bad_prof_data = setdiff(bad_prof_data, ...
                                    high_lat_points_ins);
                            end
                            
                        % ['BAD SINGLE VALUE cost vs clim S: num profs, bad T profs']
                        % [num_profs length(bad_prof_data)]
                            
                            MITprof_new.prof_Sweight(bad_prof_data) = 0;
                            zero_S_weight_reason(bad_prof_data) = ...
                                zero_S_weight_reason(bad_prof_data) + 2^(zero_critera_code_i -1);
                        end
                    end % 
                
                
                case 11 % test for possible bad conducivity cell.
                    %%
                    if isfield(MITprof,'prof_S')
                    % ['testing for bad conductivity cell']
                        clear ins*;
                        
                        % PART 1, FIND BAD CONDUCTIVITY CELLS
                        
                        num_tests = length(prof_subsurface_min_S_threshold_depth);
                        
                        % pull the salinity field
                        tmpS = MITprof_new.prof_S;
                        
                        bad_cond_ins = [];
                        
                        for ii = 1:num_tests
                            prof_S_sst = prof_subsurface_min_S_threshold(ii);
                            prof_S_sstd = prof_subsurface_min_S_threshold_depth(ii);
                            
                            % find profile depths that are shallower than the
                            % subsurface depth threshold
                            ssdi = find(abs(MITprof_new.prof_depth) <= abs(prof_S_sstd));
                            
                            % find data where S <= threshold
                            ins1 = find(tmpS <= prof_S_sst & MITprof.prof_S > checkVal);
                            
                            % find data where S > threshold
                            ins2 = find(tmpS >= prof_S_sst & MITprof.prof_S > checkVal);
                            
                            % make a mask of nans, one for each value of prof_S
                            tmp = zeros(size(MITprof_new.prof_S)).*NaN;
                            tmp(ins1) = 1;
                            tmp(ins2) = 0;
                            
                            % exclude points in the upper ocean above the
                            % prof_S_subsurface_threshold
                            tmp(:,ssdi) = NaN;
                            %%
                            % find the median value of tmp
                            x=mynanmedian(tmp');
                            
                            % identify as likely bad profiles all of those
                            % profiles where the median value of S below the
                            % treshold depth is less than the threshold salinity
                            ins3 = find(x == 1);
                            
                            bad_cond_ins = union(bad_cond_ins, ins3);
                            
                            
                        end
                        
                        % set the weights of all of those profiles to zero
                        MITprof_new.prof_Sweight(bad_cond_ins,:) = 0;
                        
                        % record the reason that we're zeroing it out.
                        zero_S_weight_reason(bad_cond_ins,:) = ...
                            zero_S_weight_reason(bad_cond_ins,:) + 2^(zero_critera_code_i -1);
                        
                        % ALSO DO T IN CASE IT IS THE T MEASURMENT
                        % THAT CAUSED S TO BE CRAZY.
                        % set the weights of all of those profiles to zero
                        MITprof_new.prof_Tweight(bad_cond_ins,:) = 0;
                        
                        % record the reason that we're zeroing it out.
                        zero_T_weight_reason(bad_cond_ins,:) = ...
                            zero_T_weight_reason(bad_cond_ins,:) + 2^(zero_critera_code_i -1);
                    end
                    
                    
            end % switch i
            %pause
            
        end % zero code
        %%
        MITprof_new.prof_Tweight_code = zero_T_weight_reason;
        num_nans = find(isnan(zero_T_weight_reason));
        if length(num_nans > 0)
            ['nans found in prof t weight code']
            stop
        end
        
        if isfield(MITprof,'prof_S')
            MITprof_new.prof_Sweight_code = zero_S_weight_reason;
            num_nans = find(isnan(zero_S_weight_reason));
            if length(num_nans > 0)
                ['nans found in prof s weight code']
                stop
            end
        end
        
        
        
        if length(MITprof_new.prof_YYYYMMDD) > 0
            if save_output_to_disk
                %%
                fDataOut{ilist}
                
                %  Write output
                ['writing output']
                fileOut=[output_dir '/' fDataOut{ilist}];
                fprintf('%s\n',fileOut);
                
                write_profile_structure_to_netcdf(MITprof_new,fileOut);
            else
                MITprofs_new{ilist} = MITprof_new;
            end
        else
            ['no profiles remaining!']
            pause
        end
        
    end % ilist


    fprintf(['\n---> finished updating zero weight points on prepared profiles \n'])
    """

def main(run_code, MITprofs, grid_dir):

    grid_dir = '/home/sweet/Desktop/ECCO-Insitu-Ian/Matlab-Dependents'
    #llc270_grid_dir = 'C:\\Users\\szswe\\Downloads\\grid_llc270_common-20240125T224704Z-001\\grid_llc270_common'
    
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

    main(run_code, MITprofs, grid_dir)