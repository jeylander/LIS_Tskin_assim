#!/usr/bin/env /opt/cray/pe/python/3.11.7/bin/python
##/usr/bin/env /Users/rdcrljbe/anaconda3/bin/python

import os
import sys
#import wget
import numpy as np
import datetime
import pandas
import csv

#from ...Data_Readers.read_LVT_Timeseries_Table import get_LVT_Timeseries_stations, get_ISMN_obs
#on the HPC
sys.path.append('/p/home/eylandej/SRC_D/python/')
sys.path.append('/p/home/eylandej/SRC_D/python/Data_Readers')
sys.path.append('/p/home/eylandej/SRC_D/python/algorithms')
#on my mac
#sys.path.append('/Users/rdcrljbe/SRC/Python')
#sys.path.append('/Users/rdcrljbe/SRC/Python/Data_Readers')
#sys.path.append('/Users/rdcrljbe/SRC/Python/algorithms')
from read_LVT_Timeseries_Table import get_LVT_Timeseries_stations, get_ISMN_obs
import Read_LIS_Tsoil_data
import Read_LIS_Tskin_data
import Read_LIS_NoahMP_Tsoil_data
import Read_LIS_NoahMP_Tskin_data
#from ISMN_LIS_ISCCP_CONFIGSISMN_LIS_ISCCP_Reader_Configs get ISMN_LIS_ISCCP_CONFIGS
import Read_ISCCP_data
from ISMN_Obs_Poly_Fit import ISMN_Obs_Poly_Fit, ISMN_Obs_Poly_Fit_LayerVal

PROCESS_ISMN='True'
PROCESS_LIS='True'
PROCESS_ISCCP='True'

#######################################################################
# Set the beginning and end dates for generating the files
#######################################################################


num_years=1
year_array=[2011,2011,2012]

Beg_YYYY = year_array[0]
End_YYYY = year_array[num_years-1]

Beg_mm=1
Beg_dd=1
Beg_HH=0

End_mm=12
End_dd=31
End_HH=23



########################################################################
#   PATH AND FILE INFO
########################################################################
# on the HPC
#STATIONS_CFG="/p/home/eylandej/configs/ISMN_Stations_SWUS_2010_SHRUBLAND.csv"
#STATIONS_CFG="/p/home/eylandej/configs/ISMN_Stations_CONUS_2010_CROPLAND.csv"
#STATIONS_CFG="/p/home/eylandej/configs/ISMN_Stations_CONUS_2010_GRASSLAND.csv"
STATIONS_CFG="/p/home/eylandej/configs/ISMN_Stations_All_2013_merged.csv"

ISMN_FILE_PATH="/p/work/eylandej/ISMN"
#ISMN_DATA_PATH="/p/home/eylandej/data/ISMN/CONUS_SHRUBLAND"
#ISMN_DATA_PATH="/p/home/eylandej/data/ISMN/CONUS_CROPLAND"
#ISMN_DATA_PATH="/p/home/eylandej/data/ISMN/CONUS_GRASSLAND"
ISMN_DATA_PATH="/p/home/eylandej/data/ISMN/CONUS_ALL"

# on my mac
#STATIONS_CFG="/Users/rdcrljbe/Data/ISMN/ISMN_Stations_CONUS_2010.csv"
#ISMN_FILE_PATH="/Users/rdcrljbe/Data/ISMN"
#ISMN_DATA_PATH="/Users/rdcrljbe/Data/ISMN"

year_loop = 0
while year_loop < num_years:

    Beg_YYYY_inc = year_array[year_loop]
    End_yyyy_inc = Beg_YYYY_inc
    print ("Working on data year", Beg_YYYY_inc, "Loop number:", year_loop, year_array[year_loop])
    print ("ISMN_FILE_PATH:", ISMN_FILE_PATH)
    print (str(Beg_YYYY_inc))
    ####
    #  Compute some date parameters
    #####
    #  First, how many increments
    end_DTG=datetime.datetime(End_yyyy_inc, End_mm, End_dd, End_HH)
    beg_DTG=datetime.datetime(Beg_YYYY_inc, Beg_mm, Beg_dd, Beg_HH)
    
    #### Figure out how many days to loop over
    date_diff=end_DTG-beg_DTG
    the_hour_csv_range=date_diff.days+1
    
    #####  This array is
    #####   it's size is number of days times the number of cycles per day (8)
    the_pts_array=np.zeros(((date_diff.days+1)*8)+1, dtype=np.int32)
    
    
    ############################################################################################
    ##   THIS NEXT SECTION IS ALL ABOUT ISMN DATA
    ############################################################################################
    
    #####  This is the file we use to determine how many stations will
    #      be used in the analysis
    
    ISMN_stations_df=get_LVT_Timeseries_stations(STATIONS_CFG)
    num_stations = (len(ISMN_stations_df.index))
    
    ##### set up the number of observational soil layers to read in
    #
    soil_layers=np.zeros((9,num_stations),dtype=float)
    
    ISMN_d_end = datetime.datetime(End_yyyy_inc, End_mm, End_dd, End_HH)
    the_filename_EndDTG='{:%Y%m%d}'.format(ISMN_d_end)
    
    ISMN_d_beg = datetime.datetime(Beg_YYYY_inc, Beg_mm, Beg_dd, Beg_HH)
    the_filename_BegDTG='{:%Y%m%d}'.format(ISMN_d_beg)
    
    the_filename_DTG=the_filename_BegDTG+'_'+the_filename_EndDTG
    ISMN_FILE_POST=the_filename_DTG+'.stm'
    NEW_ISMN_FILE_PATH=ISMN_FILE_PATH+"/"+str(Beg_YYYY_inc)+"/"
    
    #  filter out the ISMN data points that don't line up with LIS timestep output and
    #  create a soil temp profile that better matches with Noah layer scheme
    
    size_good_dates=((date_diff.days+1)*8)
    FLT_ISMN_tsoil1_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_tsoil2_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_tsoil3_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_tsoil4_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_tsoil5_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    
    FLT_ISMN_depth1_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_depth2_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_depth3_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_depth4_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    FLT_ISMN_depth5_array=np.zeros((size_good_dates, num_stations), dtype=np.float64)
    
    FLT_ISMN_numobs1_array=np.zeros((size_good_dates, num_stations), dtype=np.int32)
    FLT_ISMN_numobs2_array=np.zeros((size_good_dates, num_stations), dtype=np.int32)
    FLT_ISMN_numobs3_array=np.zeros((size_good_dates, num_stations), dtype=np.int32)
    FLT_ISMN_numobs4_array=np.zeros((size_good_dates, num_stations), dtype=np.int32)
    FLT_ISMN_numobs5_array=np.zeros((size_good_dates, num_stations), dtype=np.int32)
    
    FLT_Date_Array=np.empty(size_good_dates, dtype=object)
    print (ISMN_stations_df.layer1)
    station_loop = 0
    while station_loop < num_stations:
    
            soil_layers[0,station_loop]=ISMN_stations_df.layer1[station_loop]
            soil_layers[1,station_loop]=ISMN_stations_df.layer2[station_loop]
            soil_layers[2,station_loop]=ISMN_stations_df.layer3[station_loop]
            soil_layers[3,station_loop]=ISMN_stations_df.layer4[station_loop]
            soil_layers[4,station_loop]=ISMN_stations_df.layer5[station_loop]
            soil_layers[5,station_loop]=ISMN_stations_df.layer6[station_loop]
            soil_layers[6,station_loop]=ISMN_stations_df.layer7[station_loop]
            soil_layers[7,station_loop]=ISMN_stations_df.layer8[station_loop]
    
            num_layers = len ([index for index in soil_layers[:,station_loop] if index > 0.0])
    
            layer = 0
            while layer < 9:
              the_layer=soil_layers[layer,station_loop]
              if the_layer > 0.0:
                level_string_val="{:8.6f}".format(the_layer)
                THE_ISMN_FILE=ISMN_stations_df.filename[station_loop]
                THE_ISMN_FILE=THE_ISMN_FILE.replace("*.******", level_string_val)
                THE_ISMN_FILE=NEW_ISMN_FILE_PATH+THE_ISMN_FILE+ISMN_FILE_POST
                print ('Reading: ', THE_ISMN_FILE)
                the_date_yyyymmdd, the_time_hhmm, the_station_id, lat_val, lon_val, elevation, Layer_STS_ob, ts_valid_flag=get_ISMN_obs(THE_ISMN_FILE)
                print ('before layer assignment diag 1',num_stations,station_loop, FLT_ISMN_tsoil1_array.shape, Layer_STS_ob.shape)
                print ('before layer assignment diag 2', Layer_STS_ob)
                print ('min value of array', np.min(Layer_STS_ob))
                MISSING_VALS=np.where(Layer_STS_ob < -99.0)
                print ('Missing data: ', Layer_STS_ob[MISSING_VALS])
                Layer_STS_ob = Layer_STS_ob + 273.15
    
                if Layer_STS_ob.shape[0] == size_good_dates:
                  the_date_yyyymmdd_full=the_date_yyyymmdd
                  the_time_hhmm_full=the_time_hhmm
                  if the_layer > 0.0 and the_layer < 0.10:
                      print ('Layer 1 diag 1',num_stations,station_loop, FLT_ISMN_tsoil1_array.shape, Layer_STS_ob.shape)
                      print ('Layer 1 diag 2', Layer_STS_ob)
    
                      FLT_ISMN_tsoil1_array[:,station_loop] =  Layer_STS_ob[:] + FLT_ISMN_tsoil1_array[:,station_loop]
                      FLT_ISMN_depth1_array[:,station_loop] =  the_layer
                      FLT_ISMN_numobs1_array[:,station_loop]+=1
    
                  if the_layer >= 0.10 and the_layer < 0.41:
                      FLT_ISMN_tsoil2_array[:,station_loop] =  Layer_STS_ob[:] + FLT_ISMN_tsoil2_array[:,station_loop]
                      FLT_ISMN_depth2_array[:,station_loop] =  the_layer
                      FLT_ISMN_numobs2_array[:,station_loop]+=1
                          
                  if the_layer > 0.40 and the_layer < 1.0:
                      FLT_ISMN_tsoil3_array[:,station_loop] =  Layer_STS_ob[:] + FLT_ISMN_tsoil3_array[:,station_loop]
                      FLT_ISMN_depth3_array[:,station_loop] =  the_layer
                      FLT_ISMN_numobs3_array[:,station_loop]+=1
                  if the_layer >= 1.00:
                      FLT_ISMN_tsoil4_array[:,station_loop] =  Layer_STS_ob[:] + FLT_ISMN_tsoil4_array[:,station_loop]
                      FLT_ISMN_depth4_array[:,station_loop] =  the_layer
                      FLT_ISMN_numobs4_array[:,station_loop]+=1
                else:
                  print ('*****SHORT FILE, SKIPPING******', Layer_STS_ob.shape,size_good_dates,size_good_dates*3)
    
              layer+=1
    #
            station_loop+=1
            
    FLT_ISMN_tsoil1_array[:,:]=FLT_ISMN_tsoil1_array[:,:]/FLT_ISMN_numobs1_array[:,:]
    FLT_ISMN_tsoil2_array[:,:]=FLT_ISMN_tsoil2_array[:,:]/FLT_ISMN_numobs2_array[:,:]
    FLT_ISMN_tsoil3_array[:,:]=FLT_ISMN_tsoil3_array[:,:]/FLT_ISMN_numobs3_array[:,:]
    FLT_ISMN_tsoil4_array[:,:]=FLT_ISMN_tsoil4_array[:,:]/FLT_ISMN_numobs4_array[:,:]
    
    #  This next loop will perform a polynomial 2nd order curve fit on the observations to compute the 0.25 m and .7 meter
    #   soil temperature estimates if a 1.0 meter value is available.  If the lowest layer is above the .7 meter level,
    #   then it will only compute the poly fit for the 0.25 level and assume the observation below that level represents the
    #   Noah 0.4 to 1.0 layer.
    #
    #   Method assumes 3 layers of soil temperature are observed, with a surface value at less than .1 meter depth and deeper soil
    #   temperature observation of at least 0.5 meters.
    
    print ('End of Sorting Loop')


 
    
    print ('crashes here: ', size_good_dates, len(the_date_yyyymmdd), len(the_time_hhmm))
    date_loop=0
    while date_loop < size_good_dates:
            if Layer_STS_ob.shape[0] == size_good_dates:
                FLT_Date_Array[date_loop]=the_date_yyyymmdd_full[date_loop]+' '+the_time_hhmm_full[date_loop]
            date_loop+=1
    
    
    DF_LY1=pandas.DataFrame(FLT_ISMN_tsoil1_array, columns=list( ISMN_stations_df.Stat_Name[0:num_stations]), index=FLT_Date_Array)
    DF_LY1.index.names=['Date']
    print ('Transfering ISMN Data Arrays to the DataFrame')
    DF_LY2=pandas.DataFrame(FLT_ISMN_tsoil2_array, columns=list(ISMN_stations_df.Stat_Name[0:num_stations]), index=FLT_Date_Array)
    DF_LY3=pandas.DataFrame(FLT_ISMN_tsoil3_array, columns=list(ISMN_stations_df.Stat_Name[0:num_stations]), index=FLT_Date_Array)
    DF_LY4=pandas.DataFrame(FLT_ISMN_tsoil4_array, columns=list(ISMN_stations_df.Stat_Name[0:num_stations]), index=FLT_Date_Array)
    #DF_LY5=pandas.DataFrame(FLT_ISMN_tsoil5_array, columns=list(ISMN_stations_df.Stat_Name[0:num_stations]), index=FLT_Date_Array)
    
    if (year_loop == 0):
        merged_DF_LY1=DF_LY1.copy()
        merged_DF_LY2=DF_LY2.copy()
        merged_DF_LY3=DF_LY3.copy()
        merged_DF_LY4=DF_LY4.copy()
    else:
        print ("Adding additional year to the dataframe")
        merged_DF_LY1=pandas.concat([merged_DF_LY1, DF_LY1])
        merged_DF_LY2=pandas.concat([merged_DF_LY2, DF_LY2])
        merged_DF_LY3=pandas.concat([merged_DF_LY3, DF_LY3])
        merged_DF_LY4=pandas.concat([merged_DF_LY4, DF_LY4])
        
    year_loop+=1


print ('writing ISMN Data Arrays to file')

ISMN_d_end = datetime.datetime(End_YYYY, End_mm, End_dd, End_HH)
the_filename_EndDTG='{:%Y%m%d}'.format(ISMN_d_end)

ISMN_d_beg = datetime.datetime(Beg_YYYY, Beg_mm, Beg_dd, Beg_HH)
the_filename_BegDTG='{:%Y%m%d}'.format(ISMN_d_beg)

the_filename_DTG=the_filename_BegDTG+'-'+the_filename_EndDTG

merged_DF_LY1.to_csv(ISMN_DATA_PATH+'/'+'ISMN_tsoil_Noah-L1_data_by_hour_'+the_filename_DTG+'.csv')
merged_DF_LY2.to_csv(ISMN_DATA_PATH+'/'+'ISMN_tsoil_Noah-L2_data_by_hour_'+the_filename_DTG+'.csv')
merged_DF_LY3.to_csv(ISMN_DATA_PATH+'/'+'ISMN_tsoil_Noah-L3_data_by_hour_'+the_filename_DTG+'.csv')
merged_DF_LY4.to_csv(ISMN_DATA_PATH+'/'+'ISMN_tsoil_Noah-L4_data_by_hour_'+the_filename_DTG+'.csv')
#DF_LY5.to_csv('ISMN_tsoil_L5_data_by_hour_'+the_filename_DTG+'.csv')


print ('Prepping other data')

LIS_Noah36_tsoil_pts_OL=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_NoahMP_tsoil_pts_OL=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_Noah36_tsoil_pts_DA=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_NoahMP_tsoil_pts_DA=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
ISCCP_tskin_pts=np.zeros((((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_Noah36_tskin_pts_OL=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)
LIS_NoahMP_tskin_pts_OL=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)
LIS_Noah36_tskin_pts_DA=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)
LIS_NoahMP_tskin_pts_DA=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)

LIS_Noah36_tsoil_pts_DA2=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_NoahMP_tsoil_pts_DA2=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_Noah36_tskin_pts_DA2=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)
LIS_NoahMP_tskin_pts_DA2=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)

LIS_Noah36_tsoil_pts_DA3=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_NoahMP_tsoil_pts_DA3=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_Noah36_tskin_pts_DA3=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)
LIS_NoahMP_tskin_pts_DA3=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)

LIS_Noah36_tsoil_pts_DA4=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_NoahMP_tsoil_pts_DA4=np.zeros((4,((date_diff.days+1)*8), num_stations), dtype=np.float64)
LIS_Noah36_tskin_pts_DA4=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)
LIS_NoahMP_tskin_pts_DA4=np.zeros(((date_diff.days+1)*8, num_stations), dtype=np.float64)


##########
######
######LIS_three_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_zero_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_six_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_nine_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_twelve_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_fifteen_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_eighteen_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_twtyone_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######
######LIS_Noah_MP_three_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_zero_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_six_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_nine_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_twelve_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_fifteen_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_eighteen_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_twtyone_array_tsoil1 = np.zeros((4,the_hour_csv_range, num_stations), dtype=np.float64)
######
##########
######
######LIS_three_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_zero_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_six_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_nine_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_twelve_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_fifteen_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_eighteen_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_twtyone_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######
######LIS_Noah_MP_three_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_zero_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_six_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_nine_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_twelve_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_fifteen_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_eighteen_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######LIS_Noah_MP_twtyone_array_tskin = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######
##########
######
######ISSCP_three_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_zero_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_six_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_nine_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_twelve_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_fifteen_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_eighteen_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)
######ISSCP_twtyone_array_tskin1 = np.zeros((the_hour_csv_range, num_stations), dtype=np.float64)



Stat_Lats = ISMN_stations_df.lat
Stat_Lon = ISMN_stations_df.lon

print (Stat_Lats)
print ('Array of Station Longitudes    ', Stat_Lon)

# Read in the LIS data using a loop

curr_dtg=beg_DTG
points=0
while curr_dtg <= end_DTG:

    The_YYYY=curr_dtg.year
    The_MM=curr_dtg.month
    The_DD=curr_dtg.day
    The_HH=curr_dtg.hour

    The_YYYY_str=str(The_YYYY)
    The_MM_str=str(The_MM)
    The_DD_str=str(The_DD)
    The_HH_str=str(The_HH)
    print (The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon)
    if points == 0:
        dtg_array=["%s/%s/%s %s:00" % (The_YYYY_str,The_MM_str,The_DD_str, The_HH_str)]
    else:
        dtg_array=dtg_array+["%s/%s/%s %s:00" % (The_YYYY_str,The_MM_str,The_DD_str, The_HH_str)]
    
    #*************************************
    # Get the ISCCP Temperatures
    #*************************************
    #First Convert the Longitude values to 0-360 range (stored as -180 to 180 in config file)
    Converted_Stat_Lons=Stat_Lon
    print ('before conversion', Converted_Stat_Lons, Stat_Lon, num_stations)
    station_loop = 0
    while station_loop < num_stations:
        if Stat_Lon[station_loop] < 0:
            Converted_Stat_Lons[station_loop]=Stat_Lon[station_loop]+360
        else:
            Converted_Stat_Lons[station_loop]=Stat_Lon[station_loop]
        
        station_loop+=1
	
    print ('Converted the longitudes from -180 to 180 into a 0-360 range.....', Converted_Stat_Lons)
    
    ISCCP_tskin_pts[points, :]=Read_ISCCP_data.Read_ISCCP_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Converted_Stat_Lons)

    if PROCESS_LIS == 'True':

      #*************************************
      # Get the Open Loop Data
      #*************************************    
      LIS_RESULTS_FILE_DIR="/p/work/eylandej/DIS_test2/OUTDIR"
    
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_OL_CENUS/SURFACEMODEL/"
      LIS_Noah36_tsoil_pts_OL[:, points, :]=Read_LIS_Tsoil_data.Read_LIS_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      print (LIS_Noah36_tsoil_pts_OL[:, points, :])
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_OL_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tsoil_pts_OL[:, points, :]=Read_LIS_NoahMP_Tsoil_data.Read_LIS_NoahMP_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #  Get the OpenLoop-results for TSKIN
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_OL_CENUS/SURFACEMODEL/"
      LIS_Noah36_tskin_pts_OL[points,:]=Read_LIS_Tskin_data.Read_LIS_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_OL_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tskin_pts_OL[points,:]=Read_LIS_NoahMP_Tskin_data.Read_LIS_NoahMP_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
    
      #*************************************   
      # Get the Data Assimilation Results
      #*************************************   
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA_CENUS/SURFACEMODEL/"
      LIS_Noah36_tsoil_pts_DA[:, points, :]=Read_LIS_Tsoil_data.Read_LIS_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tsoil_pts_DA[:, points, :]=Read_LIS_NoahMP_Tsoil_data.Read_LIS_NoahMP_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #  Get the Data Assimilation-results for TSKIN
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA_CENUS/SURFACEMODEL/"
      LIS_Noah36_tskin_pts_DA[points,:]=Read_LIS_Tskin_data.Read_LIS_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tskin_pts_DA[points,:]=Read_LIS_NoahMP_Tskin_data.Read_LIS_NoahMP_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      
      
      #*************************************
      # Get the Data Assimilation Results from 2nd DA run tests
      #*************************************
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA2_CENUS/SURFACEMODEL/"
      LIS_Noah36_tsoil_pts_DA2[:, points, :]=Read_LIS_Tsoil_data.Read_LIS_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA2_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tsoil_pts_DA2[:, points, :]=Read_LIS_NoahMP_Tsoil_data.Read_LIS_NoahMP_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #  Get the Data Assimilation-results for TSKIN
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA2_CENUS/SURFACEMODEL/"
      LIS_Noah36_tskin_pts_DA2[points,:]=Read_LIS_Tskin_data.Read_LIS_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA2_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tskin_pts_DA2[points,:]=Read_LIS_NoahMP_Tskin_data.Read_LIS_NoahMP_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      
      #*************************************
      # Get the Data Assimilation Results from 3rd DA run tests
      #*************************************
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA3_CENUS/SURFACEMODEL/"
      LIS_Noah36_tsoil_pts_DA3[:, points, :]=Read_LIS_Tsoil_data.Read_LIS_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA3_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tsoil_pts_DA3[:, points, :]=Read_LIS_NoahMP_Tsoil_data.Read_LIS_NoahMP_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #  Get the Data Assimilation-results for TSKIN
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA3_CENUS/SURFACEMODEL/"
      LIS_Noah36_tskin_pts_DA3[points,:]=Read_LIS_Tskin_data.Read_LIS_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA3_CENUS/SURFACEMODEL/"
      LIS_NoahMP_tskin_pts_DA3[points,:]=Read_LIS_NoahMP_Tskin_data.Read_LIS_NoahMP_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      
      
      #*************************************
      # Get the Data Assimilation Results from 4th DA run tests
      #*************************************
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA4_CENUS/SURFACEMODEL/"
      LIS_Noah36_tsoil_pts_DA4[:, points, :]=Read_LIS_Tsoil_data.Read_LIS_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA4_CENUS/SURFACEMODEL/"
      #LIS_NoahMP_tsoil_pts_DA4[:, points, :]=Read_LIS_NoahMP_Tsoil_data.Read_LIS_NoahMP_Tsoil_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #  Get the Data Assimilation-results for TSKIN
      filedir=LIS_RESULTS_FILE_DIR+"/Noah36/ISCCP_DA4_CENUS/SURFACEMODEL/"
      LIS_Noah36_tskin_pts_DA4[points,:]=Read_LIS_Tskin_data.Read_LIS_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
      #filedir=LIS_RESULTS_FILE_DIR+"/NoahMP/ISCCP_DA4_CENUS/SURFACEMODEL/"
      #LIS_NoahMP_tskin_pts_DA4[points,:]=Read_LIS_NoahMP_Tskin_data.Read_LIS_NoahMP_Tskin_data(The_YYYY, The_MM, The_DD, The_HH, Stat_Lats, Stat_Lon,filedir)
    
    
    curr_dtg=curr_dtg+datetime.timedelta(hours=3)
    the_pts_array[points]=points
    

    points+=1

print (dtg_array)

if PROCESS_LIS == 'True':

##############################################################################################################
#  Noah MP TSOIL OUTPUT TO FILES
##############################################################################################################

  # write the noah36 Open Loop Loop data to files
  print (LIS_Noah36_tsoil_pts_DA.shape, len(ISMN_stations_df.Stat_Name), len(dtg_array))
  DF_Noah36_LY1_OL=pandas.DataFrame(LIS_Noah36_tsoil_pts_OL[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY2_OL=pandas.DataFrame(LIS_Noah36_tsoil_pts_OL[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY3_OL=pandas.DataFrame(LIS_Noah36_tsoil_pts_OL[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY4_OL=pandas.DataFrame(LIS_Noah36_tsoil_pts_OL[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_Noah36_LY1_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L1_Tsoil_OL_'+the_filename_DTG+'.csv')
  DF_Noah36_LY2_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L2_Tsoil_OL_'+the_filename_DTG+'.csv')
  DF_Noah36_LY3_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L3_Tsoil_OL_'+the_filename_DTG+'.csv')
  DF_Noah36_LY4_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L4_Tsoil_OL_'+the_filename_DTG+'.csv')

  # write the noah36 Data Assimilation Loop data to files
  print (LIS_Noah36_tsoil_pts_DA.shape, len(ISMN_stations_df.Stat_Name), len(dtg_array))
  DF_Noah36_LY1_DA=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY2_DA=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY3_DA=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY4_DA=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_Noah36_LY1_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L1_Tsoil_DA_'+the_filename_DTG+'.csv')
  DF_Noah36_LY2_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L2_Tsoil_DA_'+the_filename_DTG+'.csv')
  DF_Noah36_LY3_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L3_Tsoil_DA_'+the_filename_DTG+'.csv')
  DF_Noah36_LY4_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L4_Tsoil_DA_'+the_filename_DTG+'.csv')

  # write the noah36 Data Assimilation Loop 2 data to files
  print (LIS_Noah36_tsoil_pts_DA2.shape, len(ISMN_stations_df.Stat_Name), len(dtg_array))
  DF_Noah36_LY1_DA2=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA2[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY2_DA2=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA2[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY3_DA2=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA2[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY4_DA2=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA2[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_Noah36_LY1_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L1_Tsoil_DA2_'+the_filename_DTG+'.csv')
  DF_Noah36_LY2_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L2_Tsoil_DA2_'+the_filename_DTG+'.csv')
  DF_Noah36_LY3_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L3_Tsoil_DA2_'+the_filename_DTG+'.csv')
  DF_Noah36_LY4_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L4_Tsoil_DA2_'+the_filename_DTG+'.csv')

  # write the noah36 Data Assimilation Loop 3 data to files
  print (LIS_Noah36_tsoil_pts_DA3.shape, len(ISMN_stations_df.Stat_Name), len(dtg_array))
  DF_Noah36_LY1_DA3=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA3[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY2_DA3=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA3[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY3_DA3=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA3[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY4_DA3=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA3[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_Noah36_LY1_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L1_Tsoil_DA3_'+the_filename_DTG+'.csv')
  DF_Noah36_LY2_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L2_Tsoil_DA3_'+the_filename_DTG+'.csv')
  DF_Noah36_LY3_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L3_Tsoil_DA3_'+the_filename_DTG+'.csv')
  DF_Noah36_LY4_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L4_Tsoil_DA3_'+the_filename_DTG+'.csv')

  # write the noah36 Data Assimilation Loop 4 data to files
  print (LIS_Noah36_tsoil_pts_DA4.shape, len(ISMN_stations_df.Stat_Name), len(dtg_array))
  DF_Noah36_LY1_DA4=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA4[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY2_DA4=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA4[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY3_DA4=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA4[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_LY4_DA4=pandas.DataFrame(LIS_Noah36_tsoil_pts_DA4[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_Noah36_LY1_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L1_Tsoil_DA4_'+the_filename_DTG+'.csv')
  DF_Noah36_LY2_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L2_Tsoil_DA4_'+the_filename_DTG+'.csv')
  DF_Noah36_LY3_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L3_Tsoil_DA4_'+the_filename_DTG+'.csv')
  DF_Noah36_LY4_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_L4_Tsoil_DA4_'+the_filename_DTG+'.csv')

##############################################################################################################
#  Noah MP TSOIL OUTPUT TO FILES
##############################################################################################################

  # write the noahMP Open Loop data to files

  DF_NoahMP_LY1_OL=pandas.DataFrame(LIS_NoahMP_tsoil_pts_OL[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY2_OL=pandas.DataFrame(LIS_NoahMP_tsoil_pts_OL[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY3_OL=pandas.DataFrame(LIS_NoahMP_tsoil_pts_OL[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY4_OL=pandas.DataFrame(LIS_NoahMP_tsoil_pts_OL[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_NoahMP_LY1_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L1_Tsoil_OL_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY2_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L2_Tsoil_OL_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY3_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L3_Tsoil_OL_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY4_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L4_Tsoil_OL_'+the_filename_DTG+'.csv')


  # write the noahMP Data Assimilation data to files

  DF_NoahMP_LY1_DA=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY2_DA=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY3_DA=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY4_DA=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_NoahMP_LY1_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L1_Tsoil_DA_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY2_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L2_Tsoil_DA_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY3_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L3_Tsoil_DA_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY4_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L4_Tsoil_DA_'+the_filename_DTG+'.csv')
  
  # write the noahMP Data Assimilation Run test number 2 data to files

  DF_NoahMP_LY1_DA2=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA2[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY2_DA2=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA2[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY3_DA2=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA2[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY4_DA2=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA2[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_NoahMP_LY1_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L1_Tsoil_DA2_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY2_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L2_Tsoil_DA2_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY3_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L3_Tsoil_DA2_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY4_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L4_Tsoil_DA2_'+the_filename_DTG+'.csv')
  
  # write the noahMP Data Assimilation Run test number 3 data to files

  DF_NoahMP_LY1_DA3=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA3[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY2_DA3=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA3[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY3_DA3=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA3[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_LY4_DA3=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA3[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)

  DF_NoahMP_LY1_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L1_Tsoil_DA3_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY2_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L2_Tsoil_DA3_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY3_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L3_Tsoil_DA3_'+the_filename_DTG+'.csv')
  DF_NoahMP_LY4_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L4_Tsoil_DA3_'+the_filename_DTG+'.csv')
  
  # write the noahMP Data Assimilation Run test number 4 data to files

#  DF_NoahMP_LY1_DA4=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA4[0, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, #index=dtg_array)
#  DF_NoahMP_LY2_DA4=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA4[1, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, #index=dtg_array)
#  DF_NoahMP_LY3_DA4=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA4[2, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, #index=dtg_array)
#  DF_NoahMP_LY4_DA4=pandas.DataFrame(LIS_NoahMP_tsoil_pts_DA4[3, :, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, #index=dtg_array)
#
#  DF_NoahMP_LY1_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L1_Tsoil_DA4_'+the_filename_DTG+'.csv')
#  DF_NoahMP_LY2_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L2_Tsoil_DA4_'+the_filename_DTG+'.csv')
#  DF_NoahMP_LY3_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L3_Tsoil_DA4_'+the_filename_DTG+'.csv')
#  DF_NoahMP_LY4_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_L4_Tsoil_DA4_'+the_filename_DTG+'.csv')
 
##############################################################################################################
#  Noah 36 TSKIN OUTPUT TO FILES
##############################################################################################################

  # Write the Noah 36 Skin Temperature Open Loop data to files
  DF_Noah36_TSKIN_OL=pandas.DataFrame(LIS_Noah36_tskin_pts_OL[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_TSKIN_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_Tskin_OL_'+the_filename_DTG+'.csv')

  # Write the Noah 36 Skin Temperature Data Assimilation data to files
  DF_Noah36_TSKIN_DA=pandas.DataFrame(LIS_Noah36_tskin_pts_DA[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_TSKIN_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_Tskin_DA_'+the_filename_DTG+'.csv')
  
  # Write the Noah 36 Skin Temperature Data Assimilation data to files
  DF_Noah36_TSKIN_DA2=pandas.DataFrame(LIS_Noah36_tskin_pts_DA2[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_TSKIN_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_Tskin_DA2_'+the_filename_DTG+'.csv')

  # Write the Noah 36 Skin Temperature Data Assimilation data to files
  DF_Noah36_TSKIN_DA3=pandas.DataFrame(LIS_Noah36_tskin_pts_DA3[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_TSKIN_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_Tskin_DA3_'+the_filename_DTG+'.csv')
  
    # Write the Noah 36 Skin Temperature Data Assimilation data to files
  DF_Noah36_TSKIN_DA4=pandas.DataFrame(LIS_Noah36_tskin_pts_DA4[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_Noah36_TSKIN_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_Noah36_Tskin_DA4_'+the_filename_DTG+'.csv')


##############################################################################################################
#  Noah MP TSKIN OUTPUT TO FILES
##############################################################################################################

  # Write the Noah MP Skin Temperature Open Loop data to files
  DF_NoahMP_TSKIN_OL=pandas.DataFrame(LIS_NoahMP_tskin_pts_OL[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_TSKIN_OL.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_Tskin_OL_'+the_filename_DTG+'.csv')
  
  # Write the Noah MP Skin Temperature Open Loop data to files
  DF_NoahMP_TSKIN_DA=pandas.DataFrame(LIS_NoahMP_tskin_pts_DA[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_TSKIN_DA.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_Tskin_DA_'+the_filename_DTG+'.csv')
  
    # Write the Noah MP Skin Temperature Open Loop data to files
  DF_NoahMP_TSKIN_DA2=pandas.DataFrame(LIS_NoahMP_tskin_pts_DA2[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_TSKIN_DA2.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_Tskin_DA2_'+the_filename_DTG+'.csv')
  
    # Write the Noah MP Skin Temperature Open Loop data to files
  DF_NoahMP_TSKIN_DA3=pandas.DataFrame(LIS_NoahMP_tskin_pts_DA3[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
  DF_NoahMP_TSKIN_DA3.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_Tskin_DA3_'+the_filename_DTG+'.csv')
  
    # Write the Noah MP Skin Temperature Open Loop data to files
#  DF_NoahMP_TSKIN_DA4=pandas.DataFrame(LIS_NoahMP_tskin_pts_DA4[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
#  DF_NoahMP_TSKIN_DA4.to_csv(ISMN_DATA_PATH+'/'+'LIS_NoahMP_Tskin_DA4_'+the_filename_DTG+'.csv')



# write the ISCCP data to files

DF_ISCCP_TSKIN=pandas.DataFrame(ISCCP_tskin_pts[:, :], columns=list(ISMN_stations_df.Stat_Name[0:num_stations]))#, index=dtg_array)
DF_ISCCP_TSKIN.to_csv(ISMN_DATA_PATH+'/'+'ISCCP_TSKIN_'+the_filename_DTG+'.csv')



# Read in the LIS data for this time period


#  the SCAN data needs to be grouped into layers that best represent the LIS layers.


# plot the two datasets for each lat/lon

# average bias of the soil profile for all stations

# generate a table with the correlation values for the LIS vs. the temperatures


print ('**************************')
print ('Program Ended Successfully')
print ('**************************')


