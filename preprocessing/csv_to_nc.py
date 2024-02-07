import argparse
import os
import glob
import csv
from types import NoneType
import netCDF4 as nc4
import datetime as dt
import numpy as np
import itertools

def createNETCDF(file, dest_dir, prof_desc, prof_HHMMSS, prof_lat, prof_lon, prof_depth, prof_S, 
                 prof_T, prof_YYMMDD, prof_depth_f_flag, prof_depth_o_flag, 
                 prof_T_f_flag, prof_T_o_flag, prof_S_f_flag, prof_S_o_flag):

    # create new netCDF file for each year - NOTE: change hardcoded CTD tag for when more data flags are added
    # rename file w: CSVFILENAME_YEAR.nc
    filename = os.path.basename(file).split('.csv')
    year = str(prof_YYMMDD[0])[:4]
    output_filename = "{}_{}.nc".format(filename[0], year)
    output_filename_path = os.path.join(dest_dir, output_filename)
    nc = nc4.Dataset(output_filename_path, 'w')
    
    # Set global attributes 
    # NOTE: Date that is processed, where (directory) of CSV data came from
    # NOTE: MD5 check - gives you string hash of file
    # filename that is doing the processing 
    nc.title = 'test title' # name of org CSV file w/ suffix
    
    nc.insitution = 'JPL'
    nc.source = 'test source' 
    nc.author = 'test author' # me
    nc.date_created = '{0} creation of netcdf file.'.format(dt.datetime.now().strftime("%Y-%m-%d"))
    nc.summary = 'test summary' # profiles translated # converted_NCEI_profile_data_for_ECCO

    # Convert list to np array type
    prof_HHMMSS_np = np.asarray(prof_HHMMSS, dtype= np.int64)
    prof_YYMMDD_np = np.asarray(prof_YYMMDD, dtype= np.int64)
    prof_lat_np = np.asarray(prof_lat, dtype= np.float64)
    prof_lon_np = np.asarray(prof_lon, dtype= np.float64)
    # Multi-dim data
    prof_desc_np = np.array(list(itertools.zip_longest(*prof_desc, fillvalue=np.NaN))).T
    prof_S_np= np.array(list(itertools.zip_longest(*prof_S, fillvalue=np.NaN))).T
    prof_T_np = np.array(list(itertools.zip_longest(*prof_T, fillvalue=np.NaN))).T
    prof_depth_np = np.array(list(itertools.zip_longest(*prof_depth, fillvalue=np.NaN))).T
    prof_depth_f_flag_np = np.array(list(itertools.zip_longest(*prof_depth_f_flag, fillvalue=np.NaN))).T
    prof_depth_o_flag_np = np.array(list(itertools.zip_longest(*prof_depth_o_flag, fillvalue=np.NaN))).T
    prof_T_f_flag_np = np.array(list(itertools.zip_longest(*prof_T_f_flag, fillvalue=np.NaN))).T
    prof_T_o_flag_np = np.array(list(itertools.zip_longest(*prof_T_o_flag, fillvalue=np.NaN))).T
    prof_S_f_flag_np = np.array(list(itertools.zip_longest(*prof_S_f_flag, fillvalue=np.NaN))).T
    prof_S_o_flag_np = np.array(list(itertools.zip_longest(*prof_S_o_flag, fillvalue=np.NaN))).T
    
    # Create dimensions for NETCDF file NOTE: can create unlimited dim, can be appended to ????????
    # Find dim for multi-dim data
    x, y = prof_T_np.shape
    one_dim_size = nc.createDimension('one_dim', len(prof_HHMMSS_np)) 
    multi_dim_size_x = nc.createDimension('dim_x', x)
    multi_dim_size_y = nc.createDimension('dim_y', y)

    # Create NETCDF variables
    prof_HHMMSS_var = nc.createVariable('prof_HHMMSS', np.int64, 'one_dim')
    prof_HHMMSS_var.units = 'Hour, Mins, Seconds' 
    prof_HHMMSS_var[:] = prof_HHMMSS_np

    prof_YYMMDD_var = nc.createVariable('prof_YYYYMMDD', np.int64, 'one_dim')
    prof_YYMMDD_var.units = 'Year, Month, Day'
    prof_YYMMDD_var[:] = prof_YYMMDD_np
    
    prof_lat_var = nc.createVariable('prof_lat', np.float64, 'one_dim')
    prof_lat_var.units = 'Decimal Degrees' # NOTE: no error checking for this parem
    prof_lat_var[:] = prof_lat_np

    prof_lon_var = nc.createVariable('prof_lon', np.float64, 'one_dim')
    prof_lon_var.units = 'Decimal Degrees'# NOTE: no error checking for this parem 
    prof_lon_var[:] = prof_lon_np

    # Create var - multi dim
    time_x, time_y = prof_desc_np.shape
    multi_dim_time_x= nc.createDimension('dim_xt', time_x)
    multi_dim_time_y= nc.createDimension('dim_yd', time_y)
    prof_desc_var = nc.createVariable('prof_descr', 'S4', ('dim_xt', 'dim_yd'))
    prof_desc_var[:] = prof_desc_np

    prof_S_var = nc.createVariable('prof_S', np.float64, ('dim_x', 'dim_y'))
    prof_S_var.units = 'N/A'
    prof_S_var[:] = prof_S_np

    prof_T_var = nc.createVariable('prof_T', np.float64, ('dim_x', 'dim_y'))
    prof_T_var.units = 'Degree C'
    prof_T_var[:] = prof_T_np

    prof_depth_var = nc.createVariable('prof_depth', np.float64, ('dim_x', 'dim_y'))
    prof_depth_var.units = 'm'
    prof_depth_var[:] = prof_depth_np

    prof_depth_f_flag_var = nc.createVariable('prof_depth_wod_flag', np.float64, ('dim_x', 'dim_y'))
    prof_depth_f_flag_var.units = 'N/A'
    prof_depth_f_flag_var[:] = prof_depth_f_flag_np

    prof_depth_o_flag_var = nc.createVariable('prof_depth_orig_flag', np.float64, ('dim_x', 'dim_y'))
    prof_depth_o_flag_var.units = 'N/A'
    prof_depth_o_flag_var[:] = prof_depth_o_flag_np

    prof_T_f_flag_var = nc.createVariable('prof_T_wod_flag', np.float64, ('dim_x', 'dim_y'))
    prof_T_f_flag_var.units = 'N/A'
    prof_T_f_flag_var[:] = prof_T_f_flag_np

    prof_T_o_flag_var = nc.createVariable('prof_T_orig_flag', np.float64, ('dim_x', 'dim_y'))
    prof_T_o_flag_var.units = 'N/A'
    prof_T_o_flag_var[:] =  prof_T_o_flag_np

    prof_S_f_flag_var = nc.createVariable('prof_S_wod_flag', np.float64, ('dim_x', 'dim_y'))
    prof_S_f_flag_var.units = 'PSS'
    prof_S_f_flag_var[:] = prof_S_f_flag_np

    prof_S_o_flag_var = nc.createVariable('prof_S_orig_flag', np.float64, ('dim_x', 'dim_y'))
    prof_S_o_flag_var.units = 'N/A'
    prof_S_o_flag_var[:] = prof_S_o_flag_np

    nc.close()

"""
MATLAB CODE NOTES - CSV to WOD13
parse_wod13_csv_v03 -> crunch_through_wod_lines -> parse_wod_profile
Takes CSV files and converts into WOD13 matlab files
    A. parse_wod13_csv_v03
    - gets csv files, asks user if they want to iterate through all files
    - uses time (tic/toc funcs) to save WOD13.m files in order to figure out if
    there are any intermediate files to process
    - calls crunch_through_wod_lines(wod_text, ...
                start_i, save_freq, debug);
    - saves file w/ timestamps (tic/toc)

    B. crunch_through_wod_lines
    - inits WOD prof structure some w/ -9999 values 
    - looks for certain markers in CSV file to indicate one profile 
    - calls parse_wod_profile(wod_text_subset, debug)
    - saves info from parse_wod_profile into WOD prof structure

    C. parse_wod_profile
    - pulls out information so WOD13 structure can be init

MATLAB CODE NOTES - WOD13 to NETCDF
transform_WOD13prof_to_MITprof -> make_wod13_cell_centers
write_profile_structure_to_netcdf

    A. transform_WOD13prof_to_MITprof
    - reads all WODprofs files, gets unique years, build big array
    for every year
    - pull out info 

    B. write_profile_structure_to_netcdf
    - checks for existence of fields, writes them to NETCDF
"""
"""
MIT_PROF STRUCT NETCDF FILE 

- prof_basin          Added later
prof_depth          Good if all files have standard depths
prof_data - NEED JULIAN DAY
- prof_descr          CAST_12245790
                    NODC_Cruise_ID GB-10536     
                    METADATA
                    COUNTRY
                    ACCESSION NUMBER  

prof_HHMMSS         Good
- prof_interp_XC11    Added later, populated w/ empty???
- prof_XCNINJ         Added later, or not 
- prof_YC11           Added later, 
- prof_YCNINJ         Added later, 
prof_lat            Good
prof_lon            Good
- prof_point          Added later, 
prof_S              Good
- prof_Serr           Added later, 
- prof_Sestim         Added later, 
- prof_Sweight        Added later, 
prof_T              
- prof_Terr           Added later, 
- prof_Testim         Added later, 
- prof_Tweight        Added later, 
 prof_YYYYMMDD       Good

- prof_depth_wod_flag  
- prof_T_wod_flag       
- prof_S_wod_flag       (named prof_Sflag in thingy)

- prof_depth_orig_flag  (npp,1:nd) = profile.depth_orig_flag;
- prof_T_orig_flag      (npp,1:nd) = profile.T_orig_flag;
- prof_S_orig_flag      (npp,1:nd) = profile.S_orig_flag;
? WODprof.file_lines(npp,:) = [station_start(num_stations) station_end(num_stations)];
- prof_instruments

"""
def parse_row_info(row, profflag_found, unique_years, foundNewYear, foundFirstProfile, one_prof_desc, one_prof_HHMMSS, one_prof_lat, one_prof_lon, 
                   one_prof_depth, one_prof_S, one_prof_T, one_prof_YYMMDD, one_prof_depth_f_flag, one_prof_depth_o_flag,
                   one_prof_T_f_flag, one_prof_T_o_flag, one_prof_S_f_flag, one_prof_S_o_flag):
    
    # Parse description parem
    if(row[0].strip() == 'CAST'):
        one_prof_desc.append("CAST: {}".format(row[2].strip()))
    if(row[0].strip() == 'NODC Cruise ID'):
        one_prof_desc.append("NODC CRUISE ID: {}".format(row[2].strip()))
    if(row[0].strip() == "Country"):
        one_prof_desc.append("COUNTRY: {}".format(row[2].strip()))
    if(row[0].strip() == "probe_type"):
        one_prof_desc.append("PROBE_TYPE: {}".format(row[2].strip()))
    if(row[0].strip() == "Institute"):
        one_prof_desc.append("INSTITUTE: {}".format(row[2].strip()))
        
    # Parse time parem
    if row[0].strip() == 'Time':
        decimal_time = float(row[2])
        hours = int(decimal_time)
        mins = int((decimal_time*60) % 60)
        seconds = int((decimal_time*3600) % 60)
        if hours < 10:
            hours = "0%d" % (hours)
        if mins < 10:
            mins = "0%d" % (mins)
        if seconds < 10:
            seconds = "0%d" % (seconds)
        one_prof_HHMMSS = "{}{}{}".format(str(hours), str(mins), str(seconds))
    
    # Parse day, year, month parem
    if row[0].strip() == 'Year':
        # Check to see if we need to make new NetCDF
        if int(row[2]) in unique_years:
            one_prof_YYMMDD = row[2]
            foundNewYear = False
        else:
            # New NETCDF file needed, found new year
            if len(unique_years) == 0:
                foundFirstProfile = True
            else:
                foundFirstProfile = False
            unique_years.add(int(row[2]))
            one_prof_YYMMDD = row[2]
            foundNewYear = True

    if row[0].strip() == 'Month':
        fillval = int(row[2])
        if fillval < 10:
            fillval = "0{}".format(fillval) 
        one_prof_YYMMDD = "%s%s" % (one_prof_YYMMDD, fillval)

    if row[0].strip() == 'Day':
        fillval = int(row[2])
        if fillval < 10:
            fillval = "0{}".format(fillval)
        one_prof_YYMMDD = int("%s%s" % (one_prof_YYMMDD, fillval))

    # Parse long/lat parem
    if row[0].strip() == 'Longitude':
        one_prof_lon = float(row[2])
        
    if row[0].strip() == 'Latitude':
        one_prof_lat = float(row[2])
              
    if profflag_found == True:
        # NOTE: do we append prof_flag?
        # Append Depth, F, O
        try:
            one_prof_depth.append(float(row[1]))
        except Exception as e:
            one_prof_depth.append(np.NaN)
        try: 
            one_prof_depth_f_flag.append(float(row[2]))
        except Exception as e:
             one_prof_depth_f_flag.append(np.NaN)
        try:
            one_prof_depth_o_flag.append(float(row[3]))
        except Exception as e:
            one_prof_depth_o_flag.append(np.NaN)
        # Append Temp, F, O
        try:
            one_prof_T.append(float(row[4]))
        except Exception as e:
            one_prof_T.append(np.NaN)
        try: 
            one_prof_T_f_flag.append(float(row[5]))
        except Exception as e:
            one_prof_T_f_flag.append(np.NaN)
        try: 
            one_prof_T_o_flag.append(float(row[6]))
        except Exception as e:
            one_prof_T_o_flag.append(np.NaN)
        # Append Salinity, F, O
        try:
            one_prof_S.append(float(row[7]))
        except Exception as e:
            one_prof_S.append(np.NaN) 
        try:
            one_prof_S_f_flag.append(float(row[8]))
        except Exception as e:
            one_prof_S_f_flag.append(np.NaN) 
        try: 
            one_prof_S_o_flag.append(float(row[9]))
        except Exception as e:
            one_prof_S_o_flag.append(np.NaN) 

    return foundNewYear, foundFirstProfile, one_prof_lon, one_prof_lat, one_prof_HHMMSS, one_prof_YYMMDD   
              
# NOTE: Organize saved NETCDF profiles by year 
def csv_processing(dest_dir, input_dir):

    # Get list of .csv files to process
    csv_files_to_process = glob.glob(os.path.join(input_dir,'*.csv'))

    # Get set of unique years
    unique_years = set()

    for file in csv_files_to_process:
        
        startOfNewProfile = False
        # Defines start of column data: salinity, temp, depth...
        profflag_found = False
        valid_profile_data = True
        missing_salinity = False

        foundNewYear = False
        foundFirstProfile = False
        orgFlagExists = False
        orgFlag = None

        prof_HHMMSS = []
        prof_desc = []
        prof_lat = [] 
        prof_lon = []
        prof_depth = []
        prof_S = []
        prof_T = []
        prof_YYMMDD = []
        prof_depth_f_flag = []
        prof_depth_o_flag = [] 
        prof_T_f_flag = []
        prof_T_o_flag = []   
        prof_S_f_flag = []
        prof_S_o_flag = [] 
        
        line_num = 0

        # Create log file for errors 
        textfile_name = os.path.join(dest_dir, "log_{}_{}.txt".format(os.path.basename(file).split('.csv')[0], dt.datetime.today().strftime('%Y-%m-%d-%H-%M')))
        with open(textfile_name, 'w') as textfile:  

            with open(file, "r") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:

                    line_num = line_num + 1

                    if row[0] == '#--------------------------------------------------------------------------------':
                        startOfNewProfile = True
                        # At the start of new profile, init data structures to parse info
                        one_prof_HHMMSS = None
                        one_prof_YYMMDD = None
                        one_prof_lat = None
                        one_prof_lon = None
                        one_prof_depth = [] 
                        one_prof_desc = []
                        one_prof_S = [] 
                        one_prof_T = [] 
                        one_prof_depth_f_flag = []
                        one_prof_depth_o_flag = [] 
                        one_prof_T_f_flag = []
                        one_prof_T_o_flag = []   
                        one_prof_S_f_flag = []
                        one_prof_S_o_flag = []   

                        profflag_found = False
                        missing_salinity = False
                        orgFlagExists = False
                        orgFlag = None
                        valid_profile_data = True

                        # Add profile line num to desc field
                        one_prof_desc.append("LINE NUM: {}".format(line_num))

                    if row[0].strip() == 'END OF VARIABLES SECTION':

                        startOfNewProfile = False
                        profflag_found = False
                        missing_salinity = False
                        orgFlagExists = False
                        orgFlag = None

                        # Check if datetime is valid, convert to Python datetime obj
                        date_time_combined = "{}{}".format(one_prof_YYMMDD, one_prof_HHMMSS)
                        try: 
                            dt.datetime.strptime(date_time_combined, '%Y%m%d%H%M%S')
                        except ValueError as e:
                            valid_profile_data = False
                        
                        # Check if lon is b/w -180 and +180
                        if one_prof_lon > 180:
                            one_prof_lon = one_prof_lon - 360
                        
                        if one_prof_lon < -180 or one_prof_lon > 180:
                            valid_profile_data = False
                        
                        if one_prof_lat < -90 or one_prof_lat > 90:
                            valid_profile_data = False

                        if one_prof_HHMMSS != None and valid_profile_data == True:    

                            prof_desc.append(one_prof_desc)
                            prof_YYMMDD.append(one_prof_YYMMDD)
                            prof_HHMMSS.append(one_prof_HHMMSS)
                            prof_lat.append(one_prof_lat)
                            prof_lon.append(one_prof_lon)
                            prof_S.append(one_prof_S)
                            prof_T.append(one_prof_T)
                            prof_depth.append(one_prof_depth)
                            prof_depth_f_flag.append(one_prof_depth_f_flag)
                            prof_depth_o_flag.append(one_prof_depth_o_flag) 
                            prof_T_f_flag.append(one_prof_T_f_flag)
                            prof_T_o_flag.append(one_prof_T_o_flag)  
                            prof_S_f_flag.append(one_prof_S_f_flag)
                            prof_S_o_flag.append(one_prof_S_o_flag)
                            
                        valid_profile_data = True

                    # Check prof-flag is valid
                    if row[0].strip() == 'Prof-Flag' and valid_profile_data == True:
                        profflag_found = True
                        depth_org = row[2].strip()
                        temp_org = row[5].strip()
                        if missing_salinity == False:
                            salinity_org = row[8].strip()
                        else:
                            salinity_org = None
                        # Origin flag verification - if values are 0, then bypass this step even if origin flags exists
                        if (orgFlagExists == True and orgFlag != None) and (depth_org != '0' and temp_org != '0' and (salinity_org != '0' and missing_salinity == False)):
                            if orgFlag == 'WOCE':
                                if depth_org != '2':
                                    valid_profile_data = False
                                if temp_org != '2':
                                    valid_profile_data = False
                                if salinity_org != '2':
                                    valid_profile_data = False
                            elif orgFlag == 'GTSPP':
                                if depth_org != '1':
                                    valid_profile_data = False
                                if temp_org != '1':
                                    valid_profile_data = False
                                if salinity_org != '1':
                                    valid_profile_data = False
                            elif orgFlag == 'GEOSECS':
                                if depth_org != '5':
                                    valid_profile_data = False
                                if temp_org != '5':
                                    valid_profile_data = False
                                if salinity_org != '5':
                                    valid_profile_data = False
                            elif orgFlag == 'CalCOFI':
                                if depth_org != '2' or depth_org != '6':
                                    valid_profile_data = False
                                if temp_org != '2' or temp_org != '6':
                                    valid_profile_data = False
                                if salinity_org != '2' or salinity_org != '6':
                                    valid_profile_data = False
                            elif orgFlag == 'Wilkes Land Expedition':
                                # NOTE: is interpolated okay? also no good flag either?
                                if depth_org != '6':
                                    valid_profile_data = False
                                if temp_org != '6':
                                    valid_profile_data = False
                                if salinity_org != '6':
                                    valid_profile_data = False
                            elif orgFlag == 'OMEX':
                                if depth_org == '1':
                                    valid_profile_data = False
                                if temp_org == '1':
                                    valid_profile_data = False
                                if salinity_org == '1':
                                    valid_profile_data = False
                            elif orgFlag == 'Accession #0000440':
                                if depth_org == '1':
                                    valid_profile_data = False
                                if temp_org == '1':
                                    valid_profile_data = False
                                if salinity_org == '1':
                                    valid_profile_data = False
                            elif orgFlag == 'Accession #0001086':
                                if depth_org == '3' or depth_org == '4':
                                    valid_profile_data = False
                                if temp_org == '3' or temp_org == '4':
                                    valid_profile_data = False
                                if (salinity_org == '3' or salinity_org == '4'):
                                    valid_profile_data = False
                            elif orgFlag == 'PMEL TAO/PIRATA database':
                                if depth_org != '1' or depth_org != '2':
                                    valid_profile_data = False
                                if temp_org != '1' or temp_org != '2':
                                    valid_profile_data = False
                                if salinity_org != '1' or salinity_org != '2':
                                    valid_profile_data = False
                            elif orgFlag == 'ARGO profiling floats':
                                if depth_org != '1' or depth_org != '2':
                                    valid_profile_data = False
                                if temp_org != '1' or temp_org != '2':
                                    valid_profile_data = False
                                if salinity_org != '1' or salinity_org != '2':
                                    valid_profile_data = False
                            elif orgFlag == 'Accession 0065693 LaTex Project':
                                if depth_org == '1':
                                    valid_profile_data = False
                                if temp_org == '1':
                                    valid_profile_data = False
                                if salinity_org == '1' and missing_salinity == False:
                                    valid_profile_data = False
                            elif orgFlag == 'INIDEP (Modified IGOSS)':
                                if depth_org != '1':
                                    valid_profile_data = False
                                if temp_org != '1':
                                    valid_profile_data = False
                                if salinity_org != '1' and missing_salinity == False:
                                    valid_profile_data = False
                            else:
                                # NOTE: textfile
                                textfile.write("Unknow Org Flag at line: {}\n".format(line_num))
                            
                            if valid_profile_data == False:
                                textfile.write("Invalid profile data - prof flag invalid at line {}\n".format(line_num))

                        else:
                            if depth_org != '0':
                                valid_profile_data = False
                            if temp_org != '0':
                                valid_profile_data = False
                            if salinity_org != '0' and missing_salinity == False:
                                valid_profile_data = False

                            if valid_profile_data == False:
                                textfile.write("Invalid profile data - prof flag invalid at line {}\n".format(line_num))
                        
                    # Check to see if all columns of data are present 
                    if row[0].strip() == 'VARIABLES':
                        if row[1].strip() != 'Depth':
                            valid_profile_data = False
                            if row[1].strip() != '':
                                if row[2].strip() != 'F':
                                    valid_profile_data = False
                                if row[3].strip() != 'O':
                                    valid_profile_data = False
                        if row[4].strip() != 'Temperatur':
                            valid_profile_data = False
                            if row[4].strip() != '':
                                if row[5].strip() != 'F':
                                    valid_profile_data = False
                                if row[6].strip() != 'O':
                                    valid_profile_data = False 
                        if row[7].strip() != 'Salinity':
                            missing_salinity = True
                            textfile.write("Profile missing salinity at line {}\n".format(line_num))
                        
                        if valid_profile_data == False:
                            textfile.write("Invalid profile data - no temp/ depth variable at line {}\n".format(line_num))
                
                    # Check to see if units are valid
                    if row[0].strip() == 'UNITS' and valid_profile_data == True:
                        if row[1].strip() != 'm':
                            valid_profile_data = False
                        if row[4].strip() != 'degrees C':
                            valid_profile_data = False
                        if row[8].strip() != 'PSS' and missing_salinity == False:
                            valid_profile_data = False
                        if valid_profile_data == False:
                            textfile.write("Invalid profile data - units are invalid {}\n".format(line_num))
                    
                    # Check to see lon and lat are valid 
                    if row[0].strip() == 'Latitude' or row[0].strip() == 'Longitude':
                        if row[3].strip() != 'decimal degrees':
                            valid_profile_data = False
                            textfile.write("Invalid profile data - lon/lat units are false {}\n".format(line_num))
                    
                    # Check if Originators flag exists
                    # NOTE INFO LINK: https://www.nodc.noaa.gov/OC5/WOD/CODES/s_96_origflagset.html
                    if row[0].strip() == 'Originators flag set to use' and row[3].strip() == 'WOD code':
                        # Validate data if it exists
                        if row[2].strip() == '1' and row[4].strip() == 'WOCE':
                            orgFlagExists = True
                            orgFlag = 'WOCE'
                        elif row[2].strip() == '3' and row[4].strip() == 'GTSPP':
                            orgFlagExists = True
                            orgFlag = 'GTSPP'
                        elif row[2].strip() == '5' and row[4].strip() == 'GEOSECS':
                            orgFlagExists = True
                            orgFlag = 'GEOSECS'
                        elif row[2].strip() == '6' and row[4].strip() == 'CalCOFI':
                            orgFlagExists = True
                            orgFlag = 'CalCOFI'
                        elif row[2].strip() == '7' and row[4].strip() == 'Wilkes Land Expedition':
                            orgFlagExists = True
                            orgFlag = 'Wilkes Land Expedition'
                        elif row[2].strip() == '8' and row[4].strip() == 'OMEX':
                            orgFlagExists = True
                            orgFlag = 'OMEX'
                        elif row[2].strip() == '9' and row[4].strip() == 'Accession #0000440':
                            orgFlagExists = True
                            orgFlag = 'Accession #0000440'
                        elif row[2].strip() == '10' and row[4].strip() == 'Accession #0001086':
                            orgFlagExists = True
                            orgFlag = 'Accession #0001086'
                        elif row[2].strip() == '11' and row[4].strip() == 'PMEL TAO/PIRATA database':
                            orgFlagExists = True
                            orgFlag = 'PMEL TAO/PIRATA database'
                        elif row[2].strip() == '12' and row[4].strip() == 'ARGO profiling floats':
                            orgFlagExists = True
                            orgFlag = 'ARGO profiling floats'
                        elif row[2].strip() == '13' and row[4].strip() == 'Accession 0065693 LaTex Project':
                            orgFlagExists = True
                            orgFlag = 'Accession 0065693 LaTex Project'
                        elif row[2].strip() == '14' and row[4].strip() == 'INIDEP (Modified IGOSS)':
                            orgFlagExists = True
                            orgFlag = 'INIDEP (Modified IGOSS)'
                        else:
                            orgFlagExists = True
                            orgFlag = 'unknown'

                    # Parse row information
                    if startOfNewProfile == True and valid_profile_data == True: 

                        foundNewYear, foundFirstProfile, one_prof_lon, one_prof_lat, one_prof_HHMMSS, one_prof_YYMMDD = parse_row_info(row, profflag_found, unique_years, foundNewYear, foundFirstProfile, one_prof_desc, one_prof_HHMMSS, one_prof_lat, one_prof_lon, 
                                                                                                                                    one_prof_depth, one_prof_S, one_prof_T, one_prof_YYMMDD, one_prof_depth_f_flag, one_prof_depth_o_flag,
                                                                                                                                        one_prof_T_f_flag, one_prof_T_o_flag, one_prof_S_f_flag, one_prof_S_o_flag)
                    # Create new NETCDF file if we have found new year                                                              
                    if foundNewYear == True and foundFirstProfile == False:
                        # Create prev years NETCDF file
                        createNETCDF(file, dest_dir, prof_desc, prof_HHMMSS, prof_lat, prof_lon, prof_depth, prof_S, prof_T, prof_YYMMDD, 
                                        prof_depth_f_flag, prof_depth_o_flag, prof_T_f_flag, prof_T_o_flag, 
                                        prof_S_f_flag, prof_S_o_flag)
                        
                        # Resets overall list for new NETCDF file
                        prof_HHMMSS = []
                        prof_lat = [] 
                        prof_lon = []
                        prof_depth = []
                        prof_S = []
                        prof_T = []
                        prof_YYMMDD = []
                        prof_depth_f_flag = []
                        prof_depth_o_flag = [] 
                        prof_T_f_flag = []
                        prof_T_o_flag = []   
                        prof_S_f_flag = []
                        prof_S_o_flag = []   
                        prof_desc = []

                        foundNewYear = False
                        
    # Catch last profile info
    if one_prof_HHMMSS != None and valid_profile_data == True:
        prof_YYMMDD.append(one_prof_YYMMDD)
        prof_HHMMSS.append(one_prof_HHMMSS)
        prof_lat.append(one_prof_lat)
        prof_lon.append(one_prof_lon)
        prof_S.append(one_prof_S)
        prof_T.append(one_prof_T)
        prof_depth.append(one_prof_depth)   
        prof_depth_f_flag.append(one_prof_depth_f_flag)
        prof_depth_o_flag.append(one_prof_depth_o_flag) 
        prof_T_f_flag.append(one_prof_T_f_flag)
        prof_T_o_flag.append(one_prof_T_o_flag)  
        prof_S_f_flag.append(one_prof_S_f_flag)
        prof_S_o_flag.append(one_prof_S_o_flag)    
        prof_desc.append(one_prof_desc)  

    createNETCDF(file, dest_dir, prof_desc, prof_HHMMSS, prof_lat, prof_lon, prof_depth, prof_S, prof_T, prof_YYMMDD, 
                prof_depth_f_flag, prof_depth_o_flag, prof_T_f_flag, prof_T_o_flag, 
                prof_S_f_flag, prof_S_o_flag)

def main(dest_dir, input_dir):
    
    csv_processing(dest_dir, input_dir)

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
    #input_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/CSV_preprocessing_Python/Small Dataset/Data"    
    #dest_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/CSV_preprocessing_Python/Small Dataset/Output"

    input_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/CSV_preprocessing_Python/Big Dataset/Data"
    dest_dir = "/home/sweet/Desktop/ECCO-Insitu-Ian/CSV_preprocessing_Python/Big Dataset/Output"
    csv_processing(dest_dir, input_dir)
    #main(dest_dir, input_dir)