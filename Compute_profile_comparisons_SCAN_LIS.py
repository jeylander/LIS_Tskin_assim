#!/usr/bin/env /opt/local/bin/python3.12

import os
import sys
import numpy as np
import datetime
import pandas
import csv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from scipy import stats
import generate_8_panel_scatter_plots
import generate_station_profiles
import generate_contourf_plots



num_years=3
year_array=[2012, 2013, 2014]

Beg_YYYY = year_array[0]
End_yyyy = year_array[num_years-1]

#Beg_YYYY=2010
Beg_mm=1
Beg_dd=1
Beg_HH=0

#End_yyyy=2012
End_mm=12
End_dd=31
End_HH=23

EXP_NAME="DA3"

N36=True
NMP=True


end_DTG=datetime.datetime(End_yyyy, End_mm, End_dd, End_HH)
EDATE='{:%Y%m%d}'.format(end_DTG)
EDATETXT='{:%Y/%m/%d}'.format(end_DTG)
beg_DTG=datetime.datetime(Beg_YYYY, Beg_mm, Beg_dd, Beg_HH)
BGDATE='{:%Y%m%d}'.format(beg_DTG)
BGDATETXT='{:%Y/%m/%d}'.format(beg_DTG)
date_diff=end_DTG-beg_DTG
the_hour_csv_range=date_diff.days+1
the_pts_array=np.zeros(((date_diff.days+1)*8)+1, dtype=np.int32)

DataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CENUS_ALL/'
LISDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CENUS_ALL/'
ISCCPDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CENUS_ALL/'
OUT_PATH='/Users/rdcrljbe/Data/DIS_test2/output/CENUS_ALL'
ISMNPATH='/Users/rdcrljbe/Data/ISMN/Configs/'
 
stationsfilename=ISMNPATH+'ISMN_Stations_CENUS_2010.csv'
print ("Reading ISMN Stations Domain File: ", stationsfilename)
stationsfile=pandas.read_csv(stationsfilename)

num_stations=stationsfile.shape[0]
print ('number of stations = ', num_stations)
#Stat_Name=stationsfile['Station Name'][ii]

#  declare all the arrays

#  determine if we are reading in 1 year or multiple years

num_years = End_yyyy-Beg_YYYY+1
print ('Number of years to process = ', num_years)

INC_YEAR=Beg_YYYY
for year_loop in range (0, num_years, 1):
    print ('the year loop number is', year_loop, num_years)
    
    print ('the year loop number is', year_loop, num_years)
    INC_YEAR = year_array[year_loop]
    print ('the year we are processing is', INC_YEAR)
    STR_YR=str(INC_YEAR)
    
    STR_BDATE=STR_YR+'0101'
    STR_EDATE=STR_YR+'1231'

    SCAN_LEVELS=[1,2,3,4]
    LVL=0
    while LVL<=3:
        CURR_SCANfile=DataPath+'ISMN_tsoil_Noah-L'+str(SCAN_LEVELS[LVL])+'_data_by_hour_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        print ('Reading in File: ', CURR_SCANfile)
        CURR_SCANDATA=pandas.read_csv(CURR_SCANfile, index_col=0)
        num_scan_stations=CURR_SCANDATA.shape[1]
        num_scan_recs=CURR_SCANDATA.shape[0]
        print ('for year ', STR_YR, ' the number of scan records is ', num_scan_recs)
        the_stations=list(CURR_SCANDATA.columns)
        
        if LVL==0:
            SCAN_DATA_ARRAY_TEMP=np.zeros((4, num_scan_recs, num_scan_stations), dtype=np.float64)
        ii=0
        while ii<=num_scan_stations-1:
            
            SCAN_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_SCANDATA[CURR_SCANDATA.columns[ii]]
            ii+=1
        
        LVL+=1
    the_master_station_list=CURR_SCANDATA.columns.tolist()
    
    SCAN_DATA_ARRAY_TEMP=np.nan_to_num(SCAN_DATA_ARRAY_TEMP, copy=True, nan=-9999.0)
    SCAN_DATES_ARRAY_TEMP=CURR_SCANDATA.index.to_numpy()

    if year_loop == 0:
        SCAN_DATA_ARRAY=SCAN_DATA_ARRAY_TEMP
        SCAN_DATES_ARRAY=SCAN_DATES_ARRAY_TEMP
        print ("testing before the concatenation feature", SCAN_DATA_ARRAY.shape, SCAN_DATA_ARRAY_TEMP.shape)
    else:
        SCAN_DATA_ARRAY=np.concatenate((SCAN_DATA_ARRAY, SCAN_DATA_ARRAY_TEMP), axis=1)
        SCAN_DATES_ARRAY=np.concatenate((SCAN_DATES_ARRAY, SCAN_DATES_ARRAY_TEMP), axis=0)
        print ("testing the concatenation feature", SCAN_DATA_ARRAY.shape, SCAN_DATA_ARRAY_TEMP.shape)
    
        
    index_array=np.arange(num_scan_recs-1)
    

    
##############################################################################################################################
##  READ IN THE Noah 36 Soil Temperature Open Loop Run Data
##############################################################################################################################
    
    Noah_Levels=[1,2,3,4]
    LVL=0
    if (N36):
        while LVL<=3:
            CURR_LIS_N36_OL_file=LISDataPath+'LIS_Noah36_L'+str(Noah_Levels[LVL])+'_Tsoil_OL_'+STR_BDATE+'-'+STR_EDATE+'.csv'
            print ('Reading in File: ', CURR_LIS_N36_OL_file)
            CURR_LIS_N36_OL_data=pandas.read_csv(CURR_LIS_N36_OL_file, index_col=0)
            num_lis_stations=CURR_LIS_N36_OL_data.shape[1]
            num_lis_recs=CURR_LIS_N36_OL_data.shape[0]
            if LVL==0:
                LIS_Noah36_OL_DATA_ARRAY_TEMP=np.zeros((4, num_lis_recs, num_lis_stations), dtype=np.float64)
            ii=0
            while ii<=num_lis_stations-2:
                LIS_Noah36_OL_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_LIS_N36_OL_data[CURR_LIS_N36_OL_data.columns[ii]]
                ii+=1
            LVL+=1
        if year_loop == 0:
            LIS_Noah36_OL_DATA_ARRAY=LIS_Noah36_OL_DATA_ARRAY_TEMP
        else:
            LIS_Noah36_OL_DATA_ARRAY=np.concatenate((LIS_Noah36_OL_DATA_ARRAY, LIS_Noah36_OL_DATA_ARRAY_TEMP), axis=1)


        #  READING IN TSKIN DATA
        
        CURR_LIS_TSKIN_N36_OL_file=LISDataPath+'LIS_Noah36_Tskin_OL_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        print (CURR_LIS_TSKIN_N36_OL_file)
        CURR_N36_TSKIN_OL_data=pandas.read_csv(CURR_LIS_TSKIN_N36_OL_file, index_col=0)
        num_lis_stations=CURR_N36_TSKIN_OL_data.shape[1]
        num_lis_recs=CURR_N36_TSKIN_OL_data.shape[0]
        LIS_N36_TSKIN_OL_ARRAY_TEMP=np.zeros((num_lis_recs, num_lis_stations), dtype=np.float64)
        ii=0
        while ii<=num_lis_stations-2:
            LIS_N36_TSKIN_OL_ARRAY_TEMP[:,ii]=CURR_N36_TSKIN_OL_data[CURR_N36_TSKIN_OL_data.columns[ii]]
            ii+=1
        if year_loop == 0:
            LIS_N36_TSKIN_OL_ARRAY=LIS_N36_TSKIN_OL_ARRAY_TEMP
        else:
            LIS_N36_TSKIN_OL_ARRAY=np.concatenate((LIS_N36_TSKIN_OL_ARRAY, LIS_N36_TSKIN_OL_ARRAY_TEMP), axis=0)
    
        the_lis_stations=list(CURR_LIS_N36_OL_data.columns)
        curr_stations=np.array(the_lis_stations)

    ##############################################################################################################################
    ##  READ IN THE Noah 36 Soil Temperature DA Loop Run Data
    ##############################################################################################################################
    
        Noah_Levels=[1,2,3,4]
        LVL=0
        while LVL<=3:
            CURR_LIS_N36_DA_file=LISDataPath+'LIS_Noah36_L'+str(Noah_Levels[LVL])+'_Tsoil_'+EXP_NAME+'_'+STR_BDATE+'-'+STR_EDATE+'.csv'
            print ('Reading in File: ', CURR_LIS_N36_DA_file)
            CURR_LIS_N36_DA_data=pandas.read_csv(CURR_LIS_N36_DA_file, index_col=0)
            num_lis_stations=CURR_LIS_N36_DA_data.shape[1]
            num_lis_recs=CURR_LIS_N36_DA_data.shape[0]
            if LVL==0:
                LIS_Noah36_DA_DATA_ARRAY_TEMP=np.zeros((4, num_lis_recs, num_lis_stations), dtype=np.float64)
            ii=0
            while ii<=num_lis_stations-2:
                LIS_Noah36_DA_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_LIS_N36_DA_data[CURR_LIS_N36_DA_data.columns[ii]]
                ii+=1
            LVL+=1
            
        if year_loop == 0:
            LIS_Noah36_DA_DATA_ARRAY=LIS_Noah36_DA_DATA_ARRAY_TEMP
        else:
            LIS_Noah36_DA_DATA_ARRAY=np.concatenate((LIS_Noah36_DA_DATA_ARRAY, LIS_Noah36_DA_DATA_ARRAY_TEMP), axis=1)
            
      #### READING IN TSKIN DATA
        
        CURR_LIS_TSKIN_N36_DA_file=LISDataPath+'LIS_Noah36_Tskin_'+EXP_NAME+'_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        CURR_N36_TSKIN_DA_data=pandas.read_csv(CURR_LIS_TSKIN_N36_DA_file, index_col=0)
        num_lis_stations=CURR_N36_TSKIN_DA_data.shape[1]
        num_lis_recs=CURR_N36_TSKIN_DA_data.shape[0]
        LIS_N36_TSKIN_DA_ARRAY_TEMP=np.zeros((num_lis_recs, num_lis_stations), dtype=np.float64)
        ii=0
        while ii<=num_lis_stations-2:
            LIS_N36_TSKIN_DA_ARRAY_TEMP[:,ii]=CURR_N36_TSKIN_DA_data[CURR_N36_TSKIN_DA_data.columns[ii]]
            ii+=1
    
        if year_loop == 0:
            LIS_N36_TSKIN_DA_ARRAY=LIS_N36_TSKIN_DA_ARRAY_TEMP
        else:
            LIS_N36_TSKIN_DA_ARRAY=np.concatenate((LIS_N36_TSKIN_DA_ARRAY, LIS_N36_TSKIN_DA_ARRAY_TEMP), axis=0)

        
    ###################################################################################################################################################
    ##  IF NoahMP data is available READ IN THE NoahMP DA Loop Soil Temperature Data
    ###################################################################################################################################################

            
    if (NMP):
        Noah_Levels=[1,2,3,4]
        LVL=0
        while LVL<=3:
            CURR_LIS_NMP_OL_file=LISDataPath+'LIS_NoahMP_L'+str(Noah_Levels[LVL])+'_Tsoil_OL_'+STR_BDATE+'-'+STR_EDATE+'.csv'
            CURR_MP_OL_data=pandas.read_csv(CURR_LIS_NMP_OL_file, index_col=0)
            num_lis_stations=CURR_MP_OL_data.shape[1]
            num_lis_recs=CURR_MP_OL_data.shape[0]
            if LVL==0:
                LIS_NoahMP_OL_DATA_ARRAY_TEMP=np.zeros((4, num_lis_recs, num_lis_stations), dtype=np.float64)
            ii=0
            while ii<=num_lis_stations-2:
                LIS_NoahMP_OL_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_MP_OL_data[CURR_MP_OL_data.columns[ii]]
                ii+=1
            LVL+=1
        if year_loop == 0:
            LIS_NoahMP_OL_DATA_ARRAY=LIS_NoahMP_OL_DATA_ARRAY_TEMP
        else:
            LIS_NoahMP_OL_DATA_ARRAY=np.concatenate((LIS_NoahMP_OL_DATA_ARRAY, LIS_NoahMP_OL_DATA_ARRAY_TEMP), axis=1)
        ##### Read in the Skin Temperature data from the Noah and Noah MP files from open loop runs
        
        CURR_LIS_TSKIN_NMP_OL_file=LISDataPath+'LIS_NoahMP_Tskin_OL_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        CURR_MP_TSKIN_OL_data=pandas.read_csv(CURR_LIS_TSKIN_NMP_OL_file, index_col=0)
        num_lis_stations=CURR_MP_TSKIN_OL_data.shape[1]
        num_lis_recs=CURR_MP_TSKIN_OL_data.shape[0]
        LIS_MP_TSKIN_OL_ARRAY_TEMP=np.zeros((num_lis_recs, num_lis_stations), dtype=np.float64)
        ii=0
        while ii<=num_lis_stations-2:
            LIS_MP_TSKIN_OL_ARRAY_TEMP[:,ii]=CURR_MP_TSKIN_OL_data[CURR_MP_TSKIN_OL_data.columns[ii]]
            ii+=1
        if year_loop == 0:
            LIS_MP_TSKIN_OL_ARRAY=LIS_MP_TSKIN_OL_ARRAY_TEMP
        else:
            LIS_MP_TSKIN_OL_ARRAY=np.concatenate((LIS_MP_TSKIN_OL_ARRAY, LIS_MP_TSKIN_OL_ARRAY_TEMP), axis=0)

    ###################################################################################################################################################
    ##  READ IN THE NoahMP DA Loop Soil Temperature Data
    ###################################################################################################################################################
    
        Noah_Levels=[1,2,3,4]
        LVL=0
        while LVL<=3:
            CURR_LIS_NMP_DA_file=LISDataPath+'LIS_NoahMP_L'+str(Noah_Levels[LVL])+'_Tsoil_'+EXP_NAME+'_'+STR_BDATE+'-'+STR_EDATE+'.csv'
            CURR_MP_DA_data=pandas.read_csv(CURR_LIS_NMP_DA_file, index_col=0)
            num_lis_stations=CURR_MP_DA_data.shape[1]
            num_lis_recs=CURR_MP_DA_data.shape[0]
            print ('figure out if leap year is happening', num_lis_recs)
            if LVL==0:
                LIS_NoahMP_DA_DATA_ARRAY_TEMP=np.zeros((4, num_lis_recs, num_lis_stations), dtype=np.float64)
            ii=0
            while ii<=num_lis_stations-2:
                LIS_NoahMP_DA_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_MP_DA_data[CURR_MP_DA_data.columns[ii]]
                ii+=1
            LVL+=1
        if year_loop == 0:
            LIS_NoahMP_DA_DATA_ARRAY=LIS_NoahMP_DA_DATA_ARRAY_TEMP
        else:
            LIS_NoahMP_DA_DATA_ARRAY=np.concatenate((LIS_NoahMP_DA_DATA_ARRAY, LIS_NoahMP_DA_DATA_ARRAY_TEMP), axis=1)
        ##### Read in the Skin Temperature data from the Noah and Noah MP files from assimilation runs
        
        CURR_LIS_TSKIN_NMP_DA_file=LISDataPath+'LIS_NoahMP_Tskin_'+EXP_NAME+'_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        CURR_MP_TSKIN_DA_data=pandas.read_csv(CURR_LIS_TSKIN_NMP_DA_file, index_col=0)
        num_lis_stations=CURR_MP_TSKIN_DA_data.shape[1]
        num_lis_recs=CURR_MP_TSKIN_DA_data.shape[0]
        LIS_MP_TSKIN_DA_ARRAY_TEMP=np.zeros((num_lis_recs, num_lis_stations), dtype=np.float64)
        ii=0
        while ii<=num_lis_stations-2:
            LIS_MP_TSKIN_DA_ARRAY_TEMP[:,ii]=CURR_MP_TSKIN_DA_data[CURR_MP_TSKIN_DA_data.columns[ii]]
            ii+=1
    
        if year_loop == 0:
            LIS_MP_TSKIN_DA_ARRAY=LIS_MP_TSKIN_DA_ARRAY_TEMP
        else:
            LIS_MP_TSKIN_DA_ARRAY=np.concatenate((LIS_MP_TSKIN_DA_ARRAY, LIS_MP_TSKIN_DA_ARRAY_TEMP), axis=0)
    
    ###################################################################################################################################################
    #####  Read in the Skin Temperature data from the ISCCP files
    ###################################################################################################################################################
    
    CURR_ISCCP_TSKIN_file=ISCCPDataPath+'ISCCP_TSKIN_'+STR_BDATE+'-'+STR_EDATE+'.csv'
    CURR_ISCCP_TSKIN_data=pandas.read_csv(CURR_ISCCP_TSKIN_file, index_col=0)
    num_lis_stations=CURR_ISCCP_TSKIN_data.shape[1]
    num_lis_recs=CURR_ISCCP_TSKIN_data.shape[0]
    ISCCP_TSKIN_ARRAY_TEMP=np.zeros((num_lis_recs, num_lis_stations), dtype=np.float64)
    ii=0
    while ii<=num_lis_stations-2:
        ISCCP_TSKIN_ARRAY_TEMP[:,ii]=CURR_ISCCP_TSKIN_data[CURR_ISCCP_TSKIN_data.columns[ii]]
        ii+=1
    
    
    if year_loop == 0:
        ISCCP_TSKIN_ARRAY=ISCCP_TSKIN_ARRAY_TEMP
    else:
        ISCCP_TSKIN_ARRAY=np.concatenate((ISCCP_TSKIN_ARRAY, ISCCP_TSKIN_ARRAY_TEMP), axis=0)

max_num_stations=num_lis_stations

#######################################################################################################################################################
### REASSIGN ARRAYS TO OLDER ARRAY NAMES SINCE THIS CODE WAS MODIFIED
#######################################################################################################################################################


if (N36):
    # NOAH 36 OPEN LOOP ARRAYS
    LIS_DATA_ARRAY=LIS_Noah36_OL_DATA_ARRAY
    LIS_TSKIN_DATA_ARRAY=LIS_N36_TSKIN_OL_ARRAY
    # NOAH 36 DA LOOP ARRAYS
    N36DA_DATA_ARRAY=LIS_Noah36_DA_DATA_ARRAY
    N36DA_TSKIN_DATA_ARRAY=LIS_N36_TSKIN_DA_ARRAY

else:
    LIS_DATA_ARRAY=np.zeros((4, ((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)
    LIS_TSKIN_DATA_ARRAY=np.zeros((((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)
    # NOAH 36 DA LOOP ARRAYS
    N36DA_DATA_ARRAY=np.zeros((4, ((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)
    N36DA_TSKIN_DATA_ARRAY=np.zeros((((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)

if (NMP):
# NOAHMP OPEN LOOP ARRAYS
    LIS_MP_ARRAY=LIS_NoahMP_OL_DATA_ARRAY
    LIS_MP_TSKIN_ARRAY=LIS_MP_TSKIN_OL_ARRAY
    # NOAHMP DA LOOP ARRAYS
    LIS_MPDA_ARRAY=LIS_NoahMP_DA_DATA_ARRAY
    LIS_MPDA_TSKIN_ARRAY=LIS_MP_TSKIN_DA_ARRAY

else:
    # NOAHMP OPEN LOOP ARRAYS
    LIS_MP_ARRAY=np.zeros((4, ((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)
    LIS_MP_TSKIN_ARRAY=np.zeros((((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)
    # NOAHMP DA LOOP ARRAYS
    LIS_MPDA_ARRAY=np.zeros((4, ((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)
    LIS_MPDA_TSKIN_ARRAY=np.zeros((((date_diff.days+1)*8)+1, num_lis_stations), dtype=np.float64)

#loop through the years to determine how many days in each month

Total_Month_days=np.zeros((12), dtype='i4')
Winter_days=np.zeros((3), dtype='i4')
Spring_days=np.zeros((3), dtype='i4')
Summer_days=np.zeros((3), dtype='i4')
Fall_days=np.zeros((3), dtype='i4')

for year_loop in range (0, num_years, 1):
    #check the year to see if it is a leap year
    temp_year=datetime.datetime(year_array[year_loop], 1, 1)
    temp_date_check='{:%Y-%m-%d}'.format(temp_year)
    check_leap_year=pandas.DatetimeIndex([temp_date_check])
    if check_leap_year.is_leap_year == True:
        print (year_array[year_loop], 'a leap year')
        Mon_Days=np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    else:
        print (year_array[year_loop], 'not a leap year')
        Mon_Days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    
    Total_Month_days=Total_Month_days[:]+Mon_Days[:]
    Winter_days=Winter_days+[Mon_Days[11], Mon_Days[0], Mon_Days[1]]
    Spring_days=Spring_days+[Mon_Days[2], Mon_Days[3], Mon_Days[4]]
    Summer_days=Summer_days+[Mon_Days[5], Mon_Days[6], Mon_Days[7]]
    Fall_days=Fall_days+[Mon_Days[8], Mon_Days[9], Mon_Days[10]]
    
num_Winter_days=np.sum(Winter_days)
num_Spring_days=np.sum(Spring_days)
num_Summer_days=np.sum(Summer_days)
num_Fall_days=np.sum(Fall_days)
tot_num_hours=8*np.sum(Total_Month_days)
tot_num_days=np.sum(Total_Month_days)

print ('some diagnostics', tot_num_hours)

#  create the SCAN monthly and seasonal arrays

mon_SCAN_avg=np.zeros((4, 12, max_num_stations), dtype=np.float64)
Jan_SCAN=np.zeros((4, Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_SCAN=np.zeros((4, Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_SCAN=np.zeros((4, Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_SCAN=np.zeros((4, Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_SCAN=np.zeros((4, Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_SCAN=np.zeros((4, Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_SCAN=np.zeros((4, Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_SCAN=np.zeros((4, Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_SCAN=np.zeros((4, Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_SCAN=np.zeros((4, Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_SCAN=np.zeros((4, Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_SCAN=np.zeros((4, Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_SCAN=np.zeros((4, num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_SCAN=np.zeros((4, num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_SCAN=np.zeros((4, num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_SCAN=np.zeros((4, num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_SCAN=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_SCAN=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_SCAN=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_SCAN=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_SCAN=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_SCAN=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_SCAN=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_SCAN=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_SCAN=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_SCAN=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_SCAN=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_SCAN=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_SCAN=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_SCAN=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_SCAN=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_SCAN=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

#  creae the ISCCP monthly and seasonal arrays

mon_ISCCP_avg=np.zeros((12, max_num_stations), dtype=np.float64)
Jan_ISCCP=np.zeros((Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_ISCCP=np.zeros((Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_ISCCP=np.zeros((Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_ISCCP=np.zeros((Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_ISCCP=np.zeros((Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_ISCCP=np.zeros((Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_ISCCP=np.zeros((Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_ISCCP=np.zeros((Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_ISCCP=np.zeros((Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_ISCCP=np.zeros((Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_ISCCP=np.zeros((Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_ISCCP=np.zeros((Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_ISCCP=np.zeros((num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_ISCCP=np.zeros((num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_ISCCP=np.zeros((num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_ISCCP=np.zeros((num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_ISCCP=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_ISCCP=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_ISCCP=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_ISCCP=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_ISCCP=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_ISCCP=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_ISCCP=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_ISCCP=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_ISCCP=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_ISCCP=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_ISCCP=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_ISCCP=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_ISCCP=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_ISCCP=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_ISCCP=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_ISCCP=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah 36 OPEN LOOP monthly and seasonal arrays
###########################################################################################

mon_N36_avg=np.zeros((4, 12, max_num_stations), dtype=np.float64)
Jan_N36=np.zeros((4, Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_N36=np.zeros((4, Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_N36=np.zeros((4, Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_N36=np.zeros((4, Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_N36=np.zeros((4, Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_N36=np.zeros((4, Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_N36=np.zeros((4, Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_N36=np.zeros((4, Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_N36=np.zeros((4, Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_N36=np.zeros((4, Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_N36=np.zeros((4, Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_N36=np.zeros((4, Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_N36=np.zeros((4, num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_N36=np.zeros((4, num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_N36=np.zeros((4, num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_N36=np.zeros((4, num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_N36=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_N36=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_N36=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_N36=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_N36=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_N36=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_N36=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_N36=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_N36=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_N36=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_N36=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_N36=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_N36=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_N36=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_N36=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_N36=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah 36 OPEN LOOP monthly and seasonal Skin Temperature arrays
###########################################################################################

mon_N36_Tskin_avg=np.zeros((12, max_num_stations), dtype=np.float64)
Jan_N36_Tskin=np.zeros((Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_N36_Tskin=np.zeros((Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_N36_Tskin=np.zeros((Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_N36_Tskin=np.zeros((Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_N36_Tskin=np.zeros((Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_N36_Tskin=np.zeros((Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_N36_Tskin=np.zeros((Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_N36_Tskin=np.zeros((Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_N36_Tskin=np.zeros((Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_N36_Tskin=np.zeros((Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_N36_Tskin=np.zeros((Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_N36_Tskin=np.zeros((Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_N36_Tskin=np.zeros((num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_N36_Tskin=np.zeros((num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_N36_Tskin=np.zeros((num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_N36_Tskin=np.zeros((num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_N36_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_N36_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_N36_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_N36_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_N36_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_N36_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_N36_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_N36_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_N36_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_N36_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_N36_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_N36_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_N36_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_N36_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_N36_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_N36_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah 36 DA LOOP monthly and seasonal arrays for SOIL TEMPERATURE
###########################################################################################

mon_N36DA_avg=np.zeros((4, 12, max_num_stations), dtype=np.float64)
Jan_N36DA=np.zeros((4, Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_N36DA=np.zeros((4, Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_N36DA=np.zeros((4, Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_N36DA=np.zeros((4, Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_N36DA=np.zeros((4, Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_N36DA=np.zeros((4, Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_N36DA=np.zeros((4, Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_N36DA=np.zeros((4, Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_N36DA=np.zeros((4, Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_N36DA=np.zeros((4, Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_N36DA=np.zeros((4, Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_N36DA=np.zeros((4, Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_N36DA=np.zeros((4, num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_N36DA=np.zeros((4, num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_N36DA=np.zeros((4, num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_N36DA=np.zeros((4, num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_N36DA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_N36DA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_N36DA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_N36DA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_N36DA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_N36DA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_N36DA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_N36DA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_N36DA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_N36DA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_N36DA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_N36DA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_N36DA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_N36DA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_N36DA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_N36DA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah 36 monthly and seasonal Skin Temperature arrays
###########################################################################################

mon_N36DA_Tskin_avg=np.zeros((12, max_num_stations), dtype=np.float64)
Jan_N36DA_Tskin=np.zeros((Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_N36DA_Tskin=np.zeros((Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_N36DA_Tskin=np.zeros((Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_N36DA_Tskin=np.zeros((Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_N36DA_Tskin=np.zeros((Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_N36DA_Tskin=np.zeros((Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_N36DA_Tskin=np.zeros((Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_N36DA_Tskin=np.zeros((Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_N36DA_Tskin=np.zeros((Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_N36DA_Tskin=np.zeros((Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_N36DA_Tskin=np.zeros((Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_N36DA_Tskin=np.zeros((Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_N36DA_Tskin=np.zeros((num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_N36DA_Tskin=np.zeros((num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_N36DA_Tskin=np.zeros((num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_N36DA_Tskin=np.zeros((num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_N36DA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_N36DA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_N36DA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_N36DA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_N36DA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_N36DA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_N36DA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_N36DA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_N36DA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_N36DA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_N36DA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_N36DA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_N36DA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_N36DA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_N36DA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_N36DA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah MP OPEN LOOP monthly and seasonal arrays for SOIL TEMPERATURE
###########################################################################################

mon_MP_avg=np.zeros((4, 12, num_lis_stations), dtype=np.float64)
Jan_MP=np.zeros((4, Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_MP=np.zeros((4, Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_MP=np.zeros((4, Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_MP=np.zeros((4, Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_MP=np.zeros((4, Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_MP=np.zeros((4, Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_MP=np.zeros((4, Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_MP=np.zeros((4, Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_MP=np.zeros((4, Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_MP=np.zeros((4, Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_MP=np.zeros((4, Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_MP=np.zeros((4, Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_MP=np.zeros((4, num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_MP=np.zeros((4, num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_MP=np.zeros((4, num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_MP=np.zeros((4, num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_NMP=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_NMP=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_NMP=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_NMP=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_NMP=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_NMP=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_NMP=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_NMP=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_NMP=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_NMP=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_NMP=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_NMP=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_NMP=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_NMP=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_NMP=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_NMP=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah MP OPEN LOOP monthly and seasonal Skin Temperature arrays
###########################################################################################

mon_NMP_Tskin_avg=np.zeros((12, num_lis_stations), dtype=np.float64)
Jan_NMP_Tskin=np.zeros((Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_NMP_Tskin=np.zeros((Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_NMP_Tskin=np.zeros((Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_NMP_Tskin=np.zeros((Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_NMP_Tskin=np.zeros((Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_NMP_Tskin=np.zeros((Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_NMP_Tskin=np.zeros((Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_NMP_Tskin=np.zeros((Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_NMP_Tskin=np.zeros((Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_NMP_Tskin=np.zeros((Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_NMP_Tskin=np.zeros((Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_NMP_Tskin=np.zeros((Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_NMP_Tskin=np.zeros((num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_NMP_Tskin=np.zeros((num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_NMP_Tskin=np.zeros((num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_NMP_Tskin=np.zeros((num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_NMP_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_NMP_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_NMP_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_NMP_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_NMP_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_NMP_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_NMP_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_NMP_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_NMP_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_NMP_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_NMP_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_NMP_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_NMP_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_NMP_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_NMP_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_NMP_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah MP DA LOOP monthly and seasonal arrays for SOIL TEMPERATURE
###########################################################################################

mon_MPDA_avg=np.zeros((4, 12, num_lis_stations), dtype=np.float64)
Jan_MPDA=np.zeros((4, Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_MPDA=np.zeros((4, Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_MPDA=np.zeros((4, Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_MPDA=np.zeros((4, Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_MPDA=np.zeros((4, Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_MPDA=np.zeros((4, Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_MPDA=np.zeros((4, Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_MPDA=np.zeros((4, Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_MPDA=np.zeros((4, Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_MPDA=np.zeros((4, Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_MPDA=np.zeros((4, Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_MPDA=np.zeros((4, Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_MPDA=np.zeros((4, num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_MPDA=np.zeros((4, num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_MPDA=np.zeros((4, num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_MPDA=np.zeros((4, num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_NMPDA=np.zeros((4, Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_NMPDA=np.zeros((4, Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_NMPDA=np.zeros((4, Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_NMPDA=np.zeros((4, Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_NMPDA=np.zeros((4, Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_NMPDA=np.zeros((4, Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_NMPDA=np.zeros((4, Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_NMPDA=np.zeros((4, Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_NMPDA=np.zeros((4, Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_NMPDA=np.zeros((4, Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_NMPDA=np.zeros((4, Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_NMPDA=np.zeros((4, Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_NMPDA=np.zeros((4, num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_NMPDA=np.zeros((4, num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_NMPDA=np.zeros((4, num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_NMPDA=np.zeros((4, num_Fall_days, max_num_stations), dtype=np.float64)

###########################################################################################
#  create the LIS Noah MP OPEN LOOP monthly and seasonal Skin Temperature arrays
###########################################################################################

mon_NMPDA_Tskin_avg=np.zeros((12, num_lis_stations), dtype=np.float64)
Jan_NMPDA_Tskin=np.zeros((Total_Month_days[0]*8, max_num_stations), dtype=np.float64)
Feb_NMPDA_Tskin=np.zeros((Total_Month_days[1]*8, max_num_stations), dtype=np.float64)
Mar_NMPDA_Tskin=np.zeros((Total_Month_days[2]*8, max_num_stations), dtype=np.float64)
Apr_NMPDA_Tskin=np.zeros((Total_Month_days[3]*8, max_num_stations), dtype=np.float64)
May_NMPDA_Tskin=np.zeros((Total_Month_days[4]*8, max_num_stations), dtype=np.float64)
Jun_NMPDA_Tskin=np.zeros((Total_Month_days[5]*8, max_num_stations), dtype=np.float64)
Jul_NMPDA_Tskin=np.zeros((Total_Month_days[6]*8, max_num_stations), dtype=np.float64)
Aug_NMPDA_Tskin=np.zeros((Total_Month_days[7]*8, max_num_stations), dtype=np.float64)
Sep_NMPDA_Tskin=np.zeros((Total_Month_days[8]*8, max_num_stations), dtype=np.float64)
Oct_NMPDA_Tskin=np.zeros((Total_Month_days[9]*8, max_num_stations), dtype=np.float64)
Nov_NMPDA_Tskin=np.zeros((Total_Month_days[10]*8, max_num_stations), dtype=np.float64)
Dec_NMPDA_Tskin=np.zeros((Total_Month_days[11]*8, max_num_stations), dtype=np.float64)
Winter_NMPDA_Tskin=np.zeros((num_Winter_days*8, max_num_stations), dtype=np.float64)
Spring_NMPDA_Tskin=np.zeros((num_Spring_days*8, max_num_stations), dtype=np.float64)
Summer_NMPDA_Tskin=np.zeros((num_Summer_days*8, max_num_stations), dtype=np.float64)
Fall_NMPDA_Tskin=np.zeros((num_Fall_days*8, max_num_stations), dtype=np.float64)

Jan_00_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_00_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_00_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_00_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_00_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_00_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_00_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_00_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_00_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_00_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_00_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_00_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_00_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_00_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_00_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_00_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_03_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_03_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_03_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_03_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_03_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_03_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_03_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_03_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_03_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_03_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_03_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_03_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_03_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_03_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_03_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_03_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_06_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_06_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_06_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_06_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_06_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_06_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_06_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_06_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_06_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_06_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_06_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_06_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_06_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_06_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_06_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_06_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_09_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_09_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_09_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_09_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_09_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_09_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_09_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_09_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_09_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_09_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_09_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_09_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_09_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_09_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_09_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_09_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_12_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_12_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_12_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_12_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_12_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_12_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_12_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_12_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_12_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_12_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_12_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_12_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_12_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_12_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_12_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_12_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_15_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_15_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_15_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_15_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_15_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_15_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_15_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_15_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_15_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_15_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_15_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_15_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_15_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_15_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_15_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_15_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_18_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_18_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_18_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_18_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_18_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_18_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_18_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_18_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_18_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_18_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_18_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_18_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_18_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_18_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_18_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_18_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)

Jan_21_NMPDA_Tskin=np.zeros((Total_Month_days[0], max_num_stations), dtype=np.float64)
Feb_21_NMPDA_Tskin=np.zeros((Total_Month_days[1], max_num_stations), dtype=np.float64)
Mar_21_NMPDA_Tskin=np.zeros((Total_Month_days[2], max_num_stations), dtype=np.float64)
Apr_21_NMPDA_Tskin=np.zeros((Total_Month_days[3], max_num_stations), dtype=np.float64)
May_21_NMPDA_Tskin=np.zeros((Total_Month_days[4], max_num_stations), dtype=np.float64)
Jun_21_NMPDA_Tskin=np.zeros((Total_Month_days[5], max_num_stations), dtype=np.float64)
Jul_21_NMPDA_Tskin=np.zeros((Total_Month_days[6], max_num_stations), dtype=np.float64)
Aug_21_NMPDA_Tskin=np.zeros((Total_Month_days[7], max_num_stations), dtype=np.float64)
Sep_21_NMPDA_Tskin=np.zeros((Total_Month_days[8], max_num_stations), dtype=np.float64)
Oct_21_NMPDA_Tskin=np.zeros((Total_Month_days[9], max_num_stations), dtype=np.float64)
Nov_21_NMPDA_Tskin=np.zeros((Total_Month_days[10], max_num_stations), dtype=np.float64)
Dec_21_NMPDA_Tskin=np.zeros((Total_Month_days[11], max_num_stations), dtype=np.float64)
Winter_21_NMPDA_Tskin=np.zeros((num_Winter_days, max_num_stations), dtype=np.float64)
Spring_21_NMPDA_Tskin=np.zeros((num_Spring_days, max_num_stations), dtype=np.float64)
Summer_21_NMPDA_Tskin=np.zeros((num_Summer_days, max_num_stations), dtype=np.float64)
Fall_21_NMPDA_Tskin=np.zeros((num_Fall_days, max_num_stations), dtype=np.float64)



###########################################################################################
###########################################################################################

Noah_36_00_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_03_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_06_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_09_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_12_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_15_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_18_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36_21_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)

Noah_36DA_00_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_03_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_06_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_09_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_12_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_15_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_18_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_36DA_21_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)


Noah_MP_00_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_03_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_06_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_09_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_12_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_15_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_18_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MP_21_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)

Noah_MPDA_00_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_03_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_06_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_09_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_12_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_15_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_18_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
Noah_MPDA_21_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)


SCAN_00_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_03_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_06_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_09_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_12_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_15_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_18_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)
SCAN_21_Tsoil=np.zeros((4,tot_num_days, max_num_stations), dtype=np.float64)

ISCCP_00_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_03_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_06_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_09_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_12_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_15_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_18_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)
ISCCP_21_Tskin=np.zeros((tot_num_days, max_num_stations), dtype=np.float64)


Jan_Inc=0
Feb_Inc=0
Mar_Inc=0
Apr_Inc=0
May_Inc=0
Jun_Inc=0
Jul_Inc=0
Aug_Inc=0
Sep_Inc=0
Oct_Inc=0
Nov_Inc=0
Dec_Inc=0

Jan_00_Inc=0
Feb_00_Inc=0
Mar_00_Inc=0
Apr_00_Inc=0
May_00_Inc=0
Jun_00_Inc=0
Jul_00_Inc=0
Aug_00_Inc=0
Sep_00_Inc=0
Oct_00_Inc=0
Nov_00_Inc=0
Dec_00_Inc=0

Jan_03_Inc=0
Feb_03_Inc=0
Mar_03_Inc=0
Apr_03_Inc=0
May_03_Inc=0
Jun_03_Inc=0
Jul_03_Inc=0
Aug_03_Inc=0
Sep_03_Inc=0
Oct_03_Inc=0
Nov_03_Inc=0
Dec_03_Inc=0

Jan_06_Inc=0
Feb_06_Inc=0
Mar_06_Inc=0
Apr_06_Inc=0
May_06_Inc=0
Jun_06_Inc=0
Jul_06_Inc=0
Aug_06_Inc=0
Sep_06_Inc=0
Oct_06_Inc=0
Nov_06_Inc=0
Dec_06_Inc=0

Jan_09_Inc=0
Feb_09_Inc=0
Mar_09_Inc=0
Apr_09_Inc=0
May_09_Inc=0
Jun_09_Inc=0
Jul_09_Inc=0
Aug_09_Inc=0
Sep_09_Inc=0
Oct_09_Inc=0
Nov_09_Inc=0
Dec_09_Inc=0

Jan_12_Inc=0
Feb_12_Inc=0
Mar_12_Inc=0
Apr_12_Inc=0
May_12_Inc=0
Jun_12_Inc=0
Jul_12_Inc=0
Aug_12_Inc=0
Sep_12_Inc=0
Oct_12_Inc=0
Nov_12_Inc=0
Dec_12_Inc=0

Jan_15_Inc=0
Feb_15_Inc=0
Mar_15_Inc=0
Apr_15_Inc=0
May_15_Inc=0
Jun_15_Inc=0
Jul_15_Inc=0
Aug_15_Inc=0
Sep_15_Inc=0
Oct_15_Inc=0
Nov_15_Inc=0
Dec_15_Inc=0

Jan_18_Inc=0
Feb_18_Inc=0
Mar_18_Inc=0
Apr_18_Inc=0
May_18_Inc=0
Jun_18_Inc=0
Jul_18_Inc=0
Aug_18_Inc=0
Sep_18_Inc=0
Oct_18_Inc=0
Nov_18_Inc=0
Dec_18_Inc=0

Jan_21_Inc=0
Feb_21_Inc=0
Mar_21_Inc=0
Apr_21_Inc=0
May_21_Inc=0
Jun_21_Inc=0
Jul_21_Inc=0
Aug_21_Inc=0
Sep_21_Inc=0
Oct_21_Inc=0
Nov_21_Inc=0
Dec_21_Inc=0

hour_00_inc=0
hour_03_inc=0
hour_06_inc=0
hour_09_inc=0
hour_12_inc=0
hour_15_inc=0
hour_18_inc=0
hour_21_inc=0



for hour_loop in range(0, tot_num_hours-3, 1):
    print (hour_loop,tot_num_hours-1,int(SCAN_DATES_ARRAY[hour_loop][0:4]), int(SCAN_DATES_ARRAY[hour_loop][5:7]), int(SCAN_DATES_ARRAY[hour_loop][8:10]), int(SCAN_DATES_ARRAY[hour_loop][11:13]), int(SCAN_DATES_ARRAY[hour_loop][14:16]))
    SCAN_DTG= datetime.datetime(int(SCAN_DATES_ARRAY[hour_loop][0:4]), int(SCAN_DATES_ARRAY[hour_loop][5:7]), int(SCAN_DATES_ARRAY[hour_loop][8:10]), int(SCAN_DATES_ARRAY[hour_loop][11:13]), int(SCAN_DATES_ARRAY[hour_loop][14:16]))
    the_year=int(SCAN_DATES_ARRAY[hour_loop][0:4])
    the_hour=int(SCAN_DATES_ARRAY[hour_loop][11:13])
    the_month=int(SCAN_DATES_ARRAY[hour_loop][5:7])
    the_day=int(SCAN_DATES_ARRAY[hour_loop][8:10])
    
    
    if the_hour == 0:
        Noah_36_00_Tsoil[:,hour_00_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_00_Tsoil[:,hour_00_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_00_Tsoil[:,hour_00_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_00_Tsoil[:,hour_00_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_00_Tsoil[:,hour_00_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_00_Tskin[hour_00_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_00_inc == 0:
            dates_array_00=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_00=np.append(dates_array_00, datetime.date(the_year, the_month, the_day))
            
        hour_00_inc+=1
    if the_hour == 3:
        Noah_36_03_Tsoil[:,hour_03_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_03_Tsoil[:,hour_03_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_03_Tsoil[:,hour_03_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_03_Tsoil[:,hour_03_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_03_Tsoil[:,hour_03_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_03_Tskin[hour_03_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        
        if hour_03_inc == 0:
            dates_array_03=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_03=np.append(dates_array_03, datetime.date(the_year, the_month, the_day))
            
        hour_03_inc+=1
        
    if the_hour == 6:
        Noah_36_06_Tsoil[:,hour_06_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_06_Tsoil[:,hour_06_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_06_Tsoil[:,hour_06_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_06_Tsoil[:,hour_06_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_06_Tsoil[:,hour_06_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_06_Tskin[hour_06_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_06_inc == 0:
            dates_array_06=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_06=np.append(dates_array_06, datetime.date(the_year, the_month, the_day))
        hour_06_inc+=1
        
    if the_hour == 9:
        Noah_36_09_Tsoil[:,hour_09_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_09_Tsoil[:,hour_09_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_09_Tsoil[:,hour_09_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_09_Tsoil[:,hour_09_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_09_Tsoil[:,hour_09_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_09_Tskin[hour_09_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_09_inc == 0:
            dates_array_09=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_09=np.append(dates_array_09, datetime.date(the_year, the_month, the_day))
        hour_09_inc+=1
    if the_hour == 12:
        Noah_36_12_Tsoil[:,hour_12_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_12_Tsoil[:,hour_12_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_12_Tsoil[:,hour_12_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_12_Tsoil[:,hour_12_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_12_Tsoil[:,hour_12_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_12_Tskin[hour_12_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_12_inc == 0:
            dates_array_12=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_12=np.append(dates_array_12, datetime.date(the_year, the_month, the_day))
        hour_12_inc+=1
        
    if the_hour == 15:
        Noah_36_15_Tsoil[:,hour_15_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_15_Tsoil[:,hour_15_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_15_Tsoil[:,hour_15_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_15_Tsoil[:,hour_15_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_15_Tsoil[:,hour_15_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_15_Tskin[hour_15_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_15_inc == 0:
            dates_array_15=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_15=np.append(dates_array_15, datetime.date(the_year, the_month, the_day))
        hour_15_inc+=1
        
    if the_hour == 18:
        Noah_36_18_Tsoil[:,hour_18_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_18_Tsoil[:,hour_18_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_18_Tsoil[:,hour_18_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_18_Tsoil[:,hour_18_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_18_Tsoil[:,hour_18_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_18_Tskin[hour_18_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_18_inc == 0:
            dates_array_18=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_18=np.append(dates_array_18, datetime.date(the_year, the_month, the_day))
            
        hour_18_inc+=1
        
    if the_hour == 21:
        Noah_36_21_Tsoil[:,hour_21_inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Noah_MP_21_Tsoil[:,hour_21_inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Noah_36DA_21_Tsoil[:,hour_21_inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Noah_MPDA_21_Tsoil[:,hour_21_inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        SCAN_21_Tsoil[:,hour_21_inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        ISCCP_21_Tskin[hour_21_inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if hour_21_inc == 0:
            dates_array_21=datetime.date(the_year, the_month, the_day)
        else:
            dates_array_21=np.append(dates_array_21, datetime.date(the_year, the_month, the_day))
        hour_21_inc+=1
    
    
    if the_month == 1:
        Jan_SCAN[:,Jan_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Jan_N36[:,Jan_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Jan_MP[:,Jan_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Jan_N36_Tskin[Jan_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Jan_NMP_Tskin[Jan_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
        
        Jan_N36DA[:,Jan_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Jan_MPDA[:,Jan_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Jan_N36DA_Tskin[Jan_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Jan_NMPDA_Tskin[Jan_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Jan_ISCCP[Jan_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        
        if the_hour == 0:
            
            Jan_00_SCAN[:,Jan_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_00_N36[:,Jan_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_00_NMP[:,Jan_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_00_N36_Tskin[Jan_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_00_NMP_Tskin[Jan_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_00_N36DA[:,Jan_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_00_NMPDA[:,Jan_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_00_N36DA_Tskin[Jan_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_00_NMPDA_Tskin[Jan_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jan_00_ISCCP[Jan_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_00_Inc+=1
            
        if the_hour == 3:
            Jan_03_SCAN[:,Jan_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_03_N36[:,Jan_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_03_NMP[:,Jan_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_03_N36_Tskin[Jan_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_03_NMP_Tskin[Jan_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_03_N36DA[:,Jan_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_03_NMPDA[:,Jan_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_03_N36DA_Tskin[Jan_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_03_NMPDA_Tskin[Jan_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
     
            Jan_03_ISCCP[Jan_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_03_Inc+=1
            
        if the_hour == 6:
            Jan_06_SCAN[:,Jan_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_06_N36[:,Jan_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_06_NMP[:,Jan_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_06_N36_Tskin[Jan_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_06_NMP_Tskin[Jan_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_06_N36DA[:,Jan_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_06_NMPDA[:,Jan_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_06_N36DA_Tskin[Jan_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_06_NMPDA_Tskin[Jan_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Jan_06_ISCCP[Jan_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_06_Inc+=1
            
        if the_hour == 9:
            Jan_09_SCAN[:,Jan_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_09_N36[:,Jan_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_09_NMP[:,Jan_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_09_N36_Tskin[Jan_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_09_NMP_Tskin[Jan_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_09_N36DA[:,Jan_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_09_NMPDA[:,Jan_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_09_N36DA_Tskin[Jan_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_09_NMPDA_Tskin[Jan_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Jan_09_ISCCP[Jan_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_09_Inc+=1
            
        if the_hour == 12:
            Jan_12_SCAN[:,Jan_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_12_N36[:,Jan_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_12_NMP[:,Jan_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_12_N36_Tskin[Jan_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_12_NMP_Tskin[Jan_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_12_N36DA[:,Jan_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_12_NMPDA[:,Jan_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_12_N36DA_Tskin[Jan_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_12_NMPDA_Tskin[Jan_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Jan_12_ISCCP[Jan_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_12_Inc+=1
            
        if the_hour == 15:
            Jan_15_SCAN[:,Jan_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_15_N36[:,Jan_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_15_NMP[:,Jan_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_15_N36_Tskin[Jan_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_15_NMP_Tskin[Jan_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Jan_15_N36DA[:,Jan_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_15_NMPDA[:,Jan_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_15_N36DA_Tskin[Jan_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_15_NMPDA_Tskin[Jan_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Jan_15_ISCCP[Jan_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_15_Inc+=1
            
        if the_hour == 18:
            Jan_18_SCAN[:,Jan_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_18_N36[:,Jan_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_18_NMP[:,Jan_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_18_N36_Tskin[Jan_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_18_NMP_Tskin[Jan_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jan_18_N36DA[:,Jan_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_18_NMPDA[:,Jan_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_18_N36DA_Tskin[Jan_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_18_NMPDA_Tskin[Jan_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Jan_18_ISCCP[Jan_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_18_Inc+=1
            
        if the_hour == 21:
            Jan_21_SCAN[:,Jan_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jan_21_N36[:,Jan_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jan_21_NMP[:,Jan_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jan_21_N36_Tskin[Jan_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_21_NMP_Tskin[Jan_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Jan_21_N36DA[:,Jan_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jan_21_NMPDA[:,Jan_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jan_21_N36DA_Tskin[Jan_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jan_21_NMPDA_Tskin[Jan_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Jan_21_ISCCP[Jan_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jan_21_Inc+=1
        
        Jan_Inc+=1
        
    if the_month == 2:
        Feb_SCAN[:,Feb_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        
        Feb_N36[:,Feb_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Feb_MP[:,Feb_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Feb_N36_Tskin[Feb_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Feb_NMP_Tskin[Feb_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
        
        Feb_N36DA[:,Feb_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Feb_MPDA[:,Feb_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Feb_N36DA_Tskin[Feb_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Feb_NMPDA_Tskin[Feb_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Feb_ISCCP[Feb_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
        
            Feb_00_SCAN[:,Feb_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_00_N36[:,Feb_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_00_NMP[:,Feb_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_00_N36_Tskin[Feb_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_00_NMP_Tskin[Feb_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Feb_00_N36DA[:,Feb_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_00_NMPDA[:,Feb_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_00_N36DA_Tskin[Feb_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_00_NMPDA_Tskin[Feb_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_00_ISCCP[Feb_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_00_Inc+=1
            
        if the_hour == 3:
            Feb_03_SCAN[:,Feb_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_03_N36[:,Feb_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_03_NMP[:,Feb_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_03_N36_Tskin[Feb_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_03_NMP_Tskin[Feb_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Feb_03_N36DA[:,Feb_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_03_NMPDA[:,Feb_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_03_N36DA_Tskin[Feb_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_03_NMPDA_Tskin[Feb_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_03_ISCCP[Feb_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_03_Inc+=1
        if the_hour == 6:
            Feb_06_SCAN[:,Feb_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_06_N36[:,Feb_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_06_NMP[:,Feb_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_06_N36_Tskin[Feb_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_06_NMP_Tskin[Feb_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Feb_06_N36DA[:,Feb_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_06_NMPDA[:,Feb_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_06_N36DA_Tskin[Feb_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_06_NMPDA_Tskin[Feb_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_06_ISCCP[Feb_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_06_Inc+=1
            
        if the_hour == 9:
            Feb_09_SCAN[:,Feb_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_09_N36[:,Feb_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_09_NMP[:,Feb_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_09_N36_Tskin[Feb_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_09_NMP_Tskin[Feb_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Feb_09_N36DA[:,Feb_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_09_NMPDA[:,Feb_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_09_N36DA_Tskin[Feb_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_09_NMPDA_Tskin[Feb_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_09_ISCCP[Feb_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_09_Inc+=1
            
        if the_hour == 12:
            Feb_12_SCAN[:,Feb_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_12_N36[:,Feb_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_12_NMP[:,Feb_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_12_N36_Tskin[Feb_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_12_NMP_Tskin[Feb_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Feb_12_N36DA[:,Feb_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_12_NMPDA[:,Feb_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_12_N36DA_Tskin[Feb_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_12_NMPDA_Tskin[Feb_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_12_ISCCP[Feb_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_12_Inc+=1
            
        if the_hour == 15:
            Feb_15_SCAN[:,Feb_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_15_N36[:,Feb_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_15_NMP[:,Feb_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_15_N36_Tskin[Feb_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_15_NMP_Tskin[Feb_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Feb_15_N36DA[:,Feb_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_15_NMPDA[:,Feb_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_15_N36DA_Tskin[Feb_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_15_NMPDA_Tskin[Feb_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_15_ISCCP[Feb_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_15_Inc+=1
        if the_hour == 18:
            Feb_18_SCAN[:,Feb_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_18_N36[:,Feb_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_18_NMP[:,Feb_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_18_N36_Tskin[Feb_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_18_NMP_Tskin[Feb_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Feb_18_N36DA[:,Feb_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_18_NMPDA[:,Feb_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_18_N36DA_Tskin[Feb_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_18_NMPDA_Tskin[Feb_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_18_ISCCP[Feb_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_18_Inc+=1
        if the_hour == 21:
            Feb_21_SCAN[:,Feb_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Feb_21_N36[:,Feb_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Feb_21_NMP[:,Feb_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Feb_21_N36_Tskin[Feb_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_21_NMP_Tskin[Feb_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Feb_21_N36DA[:,Feb_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Feb_21_NMPDA[:,Feb_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Feb_21_N36DA_Tskin[Feb_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Feb_21_NMPDA_Tskin[Feb_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Feb_21_ISCCP[Feb_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Feb_21_Inc+=1
        Feb_Inc+=1
    if the_month == 3:
        Mar_SCAN[:,Mar_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Mar_N36[:,Mar_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Mar_MP[:,Mar_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Mar_N36_Tskin[Mar_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Mar_NMP_Tskin[Mar_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
        
        Mar_N36DA[:,Mar_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Mar_MPDA[:,Mar_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Mar_N36DA_Tskin[Mar_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Mar_NMPDA_Tskin[Mar_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Mar_ISCCP[Mar_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Mar_00_SCAN[:,Mar_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_00_N36[:,Mar_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_00_NMP[:,Mar_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_00_N36_Tskin[Mar_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_00_NMP_Tskin[Mar_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Mar_00_N36DA[:,Mar_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_00_NMPDA[:,Mar_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_00_N36DA_Tskin[Mar_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_00_NMPDA_Tskin[Mar_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_00_ISCCP[Mar_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_00_Inc+=1
        if the_hour == 3:
            Mar_03_SCAN[:,Mar_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_03_N36[:,Mar_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_03_NMP[:,Mar_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_03_N36_Tskin[Mar_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_03_NMP_Tskin[Mar_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Mar_03_N36DA[:,Mar_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_03_NMPDA[:,Mar_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_03_N36DA_Tskin[Mar_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_03_NMPDA_Tskin[Mar_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_03_ISCCP[Mar_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_03_Inc+=1
        if the_hour == 6:
            Mar_06_SCAN[:,Mar_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_06_N36[:,Mar_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_06_NMP[:,Mar_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_06_N36_Tskin[Mar_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_06_NMP_Tskin[Mar_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Mar_06_N36DA[:,Mar_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_06_NMPDA[:,Mar_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_06_N36DA_Tskin[Mar_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_06_NMPDA_Tskin[Mar_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_06_ISCCP[Mar_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_06_Inc+=1
        if the_hour == 9:
            Mar_09_SCAN[:,Mar_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_09_N36[:,Mar_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_09_NMP[:,Mar_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_09_N36_Tskin[Mar_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_09_NMP_Tskin[Mar_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Mar_09_N36DA[:,Mar_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_09_NMPDA[:,Mar_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_09_N36DA_Tskin[Mar_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_09_NMPDA_Tskin[Mar_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_09_ISCCP[Mar_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_09_Inc+=1
        if the_hour == 12:
            Mar_12_SCAN[:,Mar_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_12_N36[:,Mar_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_12_NMP[:,Mar_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_12_N36_Tskin[Mar_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_12_NMP_Tskin[Mar_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Mar_12_N36DA[:,Mar_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_12_NMPDA[:,Mar_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_12_N36DA_Tskin[Mar_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_12_NMPDA_Tskin[Mar_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_12_ISCCP[Mar_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_12_Inc+=1
        if the_hour == 15:
            Mar_15_SCAN[:,Mar_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_15_N36[:,Mar_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_15_NMP[:,Mar_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_15_N36_Tskin[Mar_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_15_NMP_Tskin[Mar_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Mar_15_N36DA[:,Mar_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_15_NMPDA[:,Mar_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_15_N36DA_Tskin[Mar_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_15_NMPDA_Tskin[Mar_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_15_ISCCP[Mar_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_15_Inc+=1
        if the_hour == 18:
            Mar_18_SCAN[:,Mar_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_18_N36[:,Mar_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_18_NMP[:,Mar_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_18_N36_Tskin[Mar_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_18_NMP_Tskin[Mar_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Mar_18_N36DA[:,Mar_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_18_NMPDA[:,Mar_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_18_N36DA_Tskin[Mar_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_18_NMPDA_Tskin[Mar_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_18_ISCCP[Mar_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_18_Inc+=1
        if the_hour == 21:
            Mar_21_SCAN[:,Mar_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Mar_21_N36[:,Mar_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Mar_21_NMP[:,Mar_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Mar_21_N36_Tskin[Mar_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_21_NMP_Tskin[Mar_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Mar_21_N36DA[:,Mar_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Mar_21_NMPDA[:,Mar_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Mar_21_N36DA_Tskin[Mar_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Mar_21_NMPDA_Tskin[Mar_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Mar_21_ISCCP[Mar_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Mar_21_Inc+=1
        Mar_Inc+=1
    if the_month == 4:
        Apr_SCAN[:,Apr_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Apr_N36[:,Apr_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Apr_MP[:,Apr_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Apr_N36_Tskin[Apr_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Apr_NMP_Tskin[Apr_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
        
        Apr_N36DA[:,Apr_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Apr_MPDA[:,Apr_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Apr_N36DA_Tskin[Apr_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Apr_NMPDA_Tskin[Apr_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Apr_ISCCP[Apr_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Apr_00_SCAN[:,Apr_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_00_N36[:,Apr_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_00_NMP[:,Apr_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_00_N36_Tskin[Apr_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_00_NMP_Tskin[Apr_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Apr_00_N36DA[:,Apr_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_00_NMPDA[:,Apr_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_00_N36DA_Tskin[Apr_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_00_NMPDA_Tskin[Apr_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Apr_00_ISCCP[Apr_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_00_Inc+=1
        if the_hour == 3:
            Apr_03_SCAN[:,Apr_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_03_N36[:,Apr_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_03_NMP[:,Apr_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_03_N36_Tskin[Apr_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_03_NMP_Tskin[Apr_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Apr_03_N36DA[:,Apr_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_03_NMPDA[:,Apr_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_03_N36DA_Tskin[Apr_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_03_NMPDA_Tskin[Apr_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_03_ISCCP[Apr_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_03_Inc+=1
        if the_hour == 6:
            Apr_06_SCAN[:,Apr_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_06_N36[:,Apr_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_06_NMP[:,Apr_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_06_N36_Tskin[Apr_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_06_NMP_Tskin[Apr_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
           
            Apr_06_N36DA[:,Apr_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_06_NMPDA[:,Apr_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_06_N36DA_Tskin[Apr_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_06_NMPDA_Tskin[Apr_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_06_ISCCP[Apr_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_06_Inc+=1
        if the_hour == 9:
            Apr_09_SCAN[:,Apr_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_09_N36[:,Apr_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_09_NMP[:,Apr_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_09_N36_Tskin[Apr_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_09_NMP_Tskin[Apr_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Apr_09_N36DA[:,Apr_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_09_NMPDA[:,Apr_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_09_N36DA_Tskin[Apr_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_09_NMPDA_Tskin[Apr_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_09_ISCCP[Apr_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_09_Inc+=1
        if the_hour == 12:
            Apr_12_SCAN[:,Apr_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_12_N36[:,Apr_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_12_NMP[:,Apr_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_12_N36_Tskin[Apr_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_12_NMP_Tskin[Apr_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Apr_12_N36DA[:,Apr_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_12_NMPDA[:,Apr_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_12_N36DA_Tskin[Apr_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_12_NMPDA_Tskin[Apr_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_12_ISCCP[Apr_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_12_Inc+=1
        if the_hour == 15:
            Apr_15_SCAN[:,Apr_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_15_N36[:,Apr_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_15_NMP[:,Apr_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_15_N36_Tskin[Apr_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_15_NMP_Tskin[Apr_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Apr_15_N36DA[:,Apr_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_15_NMPDA[:,Apr_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_15_N36DA_Tskin[Apr_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_15_NMPDA_Tskin[Apr_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_15_ISCCP[Apr_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_15_Inc+=1
        if the_hour == 18:
            Apr_18_SCAN[:,Apr_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_18_N36[:,Apr_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_18_NMP[:,Apr_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_18_N36_Tskin[Apr_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_18_NMP_Tskin[Apr_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Apr_18_N36DA[:,Apr_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_18_NMPDA[:,Apr_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_18_N36DA_Tskin[Apr_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_18_NMPDA_Tskin[Apr_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_18_ISCCP[Apr_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_18_Inc+=1
        if the_hour == 21:
            Apr_21_SCAN[:,Apr_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Apr_21_N36[:,Apr_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Apr_21_NMP[:,Apr_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Apr_21_N36_Tskin[Apr_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_21_NMP_Tskin[Apr_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Apr_21_N36DA[:,Apr_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Apr_21_NMPDA[:,Apr_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Apr_21_N36DA_Tskin[Apr_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Apr_21_NMPDA_Tskin[Apr_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

            Apr_21_ISCCP[Apr_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Apr_21_Inc+=1
        Apr_Inc+=1
    if the_month == 5:
        May_SCAN[:,May_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        May_N36[:,May_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        May_MP[:,May_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        May_N36_Tskin[May_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        May_NMP_Tskin[May_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

        May_N36DA[:,May_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        May_MPDA[:,May_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        May_N36DA_Tskin[May_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        May_NMPDA_Tskin[May_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        May_ISCCP[May_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            May_00_SCAN[:,May_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_00_N36[:,May_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_00_NMP[:,May_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_00_N36_Tskin[May_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_00_NMP_Tskin[May_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_00_N36DA[:,May_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_00_NMPDA[:,May_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_00_N36DA_Tskin[May_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_00_NMPDA_Tskin[May_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_00_ISCCP[May_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_00_Inc+=1
        if the_hour == 3:
            May_03_SCAN[:,May_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_03_N36[:,May_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_03_NMP[:,May_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_03_N36_Tskin[May_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_03_NMP_Tskin[May_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_03_N36DA[:,May_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_03_NMPDA[:,May_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_03_N36DA_Tskin[May_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_03_NMPDA_Tskin[May_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_03_ISCCP[May_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_03_Inc+=1
        if the_hour == 6:
            May_06_SCAN[:,May_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_06_N36[:,May_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_06_NMP[:,May_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_06_N36_Tskin[May_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_06_NMP_Tskin[May_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_06_N36DA[:,May_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_06_NMPDA[:,May_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_06_N36DA_Tskin[May_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_06_NMPDA_Tskin[May_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_06_ISCCP[May_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_06_Inc+=1
        if the_hour == 9:
            May_09_SCAN[:,May_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_09_N36[:,May_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_09_NMP[:,May_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_09_N36_Tskin[May_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_09_NMP_Tskin[May_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_09_N36DA[:,May_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_09_NMPDA[:,May_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_09_N36DA_Tskin[May_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_09_NMPDA_Tskin[May_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_09_ISCCP[May_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_09_Inc+=1
        if the_hour == 12:
            May_12_SCAN[:,May_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_12_N36[:,May_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_12_NMP[:,May_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_12_N36_Tskin[May_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_12_NMP_Tskin[May_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_12_N36DA[:,May_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_12_NMPDA[:,May_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_12_N36DA_Tskin[May_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_12_NMPDA_Tskin[May_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_12_ISCCP[May_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_12_Inc+=1
        if the_hour == 15:
            May_15_SCAN[:,May_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_15_N36[:,May_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_15_NMP[:,May_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_15_N36_Tskin[May_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_15_NMP_Tskin[May_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_15_N36DA[:,May_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_15_NMPDA[:,May_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_15_N36DA_Tskin[May_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_15_NMPDA_Tskin[May_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_15_ISCCP[May_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_15_Inc+=1
        if the_hour == 18:
            May_18_SCAN[:,May_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_18_N36[:,May_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_18_NMP[:,May_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_18_N36_Tskin[May_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_18_NMP_Tskin[May_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            May_18_N36DA[:,May_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_18_NMPDA[:,May_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_18_N36DA_Tskin[May_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_18_NMPDA_Tskin[May_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_18_ISCCP[May_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_18_Inc+=1
        if the_hour == 21:
            May_21_SCAN[:,May_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            May_21_N36[:,May_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            May_21_NMP[:,May_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            May_21_N36_Tskin[May_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            May_21_NMP_Tskin[May_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            May_21_N36DA[:,May_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            May_21_NMPDA[:,May_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            May_21_N36DA_Tskin[May_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            May_21_NMPDA_Tskin[May_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            May_21_ISCCP[May_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            May_21_Inc+=1
        May_Inc+=1
    if the_month == 6:
        Jun_SCAN[:,Jun_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Jun_N36[:,Jun_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Jun_MP[:,Jun_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Jun_N36_Tskin[Jun_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Jun_NMP_Tskin[Jun_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

        Jun_N36DA[:,Jun_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Jun_MPDA[:,Jun_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Jun_N36DA_Tskin[Jun_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Jun_NMPDA_Tskin[Jun_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Jun_ISCCP[Jun_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Jun_00_SCAN[:,Jun_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_00_N36[:,Jun_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_00_NMP[:,Jun_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_00_N36_Tskin[Jun_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_00_NMP_Tskin[Jun_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_00_N36DA[:,Jun_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_00_NMPDA[:,Jun_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_00_N36DA_Tskin[Jun_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_00_NMPDA_Tskin[Jun_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_00_ISCCP[Jun_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_00_Inc+=1
        if the_hour == 3:
            Jun_03_SCAN[:,Jun_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_03_N36[:,Jun_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_03_NMP[:,Jun_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_03_N36_Tskin[Jun_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_03_NMP_Tskin[Jun_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_03_N36DA[:,Jun_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_03_NMPDA[:,Jun_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_03_N36DA_Tskin[Jun_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_03_NMPDA_Tskin[Jun_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_03_ISCCP[Jun_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_03_Inc+=1
        if the_hour == 6:
            Jun_06_SCAN[:,Jun_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_06_N36[:,Jun_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_06_NMP[:,Jun_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_06_N36_Tskin[Jun_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_06_NMP_Tskin[Jun_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_06_N36DA[:,Jun_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_06_NMPDA[:,Jun_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_06_N36DA_Tskin[Jun_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_06_NMPDA_Tskin[Jun_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_06_ISCCP[Jun_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_06_Inc+=1
        if the_hour == 9:
            Jun_09_SCAN[:,Jun_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_09_N36[:,Jun_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_09_NMP[:,Jun_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_09_N36_Tskin[Jun_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_09_NMP_Tskin[Jun_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_09_N36DA[:,Jun_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_09_NMPDA[:,Jun_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_09_N36DA_Tskin[Jun_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_09_NMPDA_Tskin[Jun_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_09_ISCCP[Jun_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_09_Inc+=1
        if the_hour == 12:
            Jun_12_SCAN[:,Jun_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_12_N36[:,Jun_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_12_NMP[:,Jun_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_12_N36_Tskin[Jun_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_12_NMP_Tskin[Jun_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_12_N36DA[:,Jun_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_12_NMPDA[:,Jun_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_12_N36DA_Tskin[Jun_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_12_NMPDA_Tskin[Jun_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_12_ISCCP[Jun_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_12_Inc+=1
        if the_hour == 15:
            Jun_15_SCAN[:,Jun_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_15_N36[:,Jun_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_15_NMP[:,Jun_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_15_N36_Tskin[Jun_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_15_NMP_Tskin[Jun_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_15_N36DA[:,Jun_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_15_NMPDA[:,Jun_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_15_N36DA_Tskin[Jun_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_15_NMPDA_Tskin[Jun_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_15_ISCCP[Jun_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_15_Inc+=1
        if the_hour == 18:
            Jun_18_SCAN[:,Jun_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_18_N36[:,Jun_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_18_NMP[:,Jun_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_18_N36_Tskin[Jun_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_18_NMP_Tskin[Jun_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_18_N36DA[:,Jun_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_18_NMPDA[:,Jun_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_18_N36DA_Tskin[Jun_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_18_NMPDA_Tskin[Jun_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_18_ISCCP[Jun_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_18_Inc+=1
        if the_hour == 21:
            Jun_21_SCAN[:,Jun_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jun_21_N36[:,Jun_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jun_21_NMP[:,Jun_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jun_21_N36_Tskin[Jun_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_21_NMP_Tskin[Jun_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jun_21_N36DA[:,Jun_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jun_21_NMPDA[:,Jun_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jun_21_N36DA_Tskin[Jun_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jun_21_NMPDA_Tskin[Jun_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jun_21_ISCCP[Jun_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jun_21_Inc+=1
        Jun_Inc+=1
    if the_month == 7:
        Jul_SCAN[:,Jul_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Jul_N36[:,Jul_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Jul_MP[:,Jul_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Jul_N36_Tskin[Jul_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Jul_NMP_Tskin[Jul_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

        Jul_N36DA[:,Jul_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Jul_MPDA[:,Jul_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Jul_N36DA_Tskin[Jul_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Jul_NMPDA_Tskin[Jul_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

        Jul_ISCCP[Jul_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Jul_00_SCAN[:,Jul_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_00_N36[:,Jul_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_00_NMP[:,Jul_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_00_N36_Tskin[Jul_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_00_NMP_Tskin[Jul_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Jul_00_N36DA[:,Jul_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_00_NMPDA[:,Jul_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_00_N36DA_Tskin[Jul_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_00_NMPDA_Tskin[Jul_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Jul_00_ISCCP[Jul_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_00_Inc+=1
        if the_hour == 3:
            Jul_03_SCAN[:,Jul_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_03_N36[:,Jul_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_03_NMP[:,Jul_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_03_N36_Tskin[Jul_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_03_NMP_Tskin[Jul_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jul_03_N36DA[:,Jul_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_03_NMPDA[:,Jul_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_03_N36DA_Tskin[Jul_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_03_NMPDA_Tskin[Jul_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Jul_03_ISCCP[Jul_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_03_Inc+=1
        if the_hour == 6:
            Jul_06_SCAN[:,Jul_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_06_N36[:,Jul_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_06_NMP[:,Jul_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_06_N36_Tskin[Jul_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_06_NMP_Tskin[Jul_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jul_06_N36DA[:,Jul_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_06_NMPDA[:,Jul_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_06_N36DA_Tskin[Jul_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_06_NMPDA_Tskin[Jul_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Jul_06_ISCCP[Jul_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_06_Inc+=1
        if the_hour == 9:
            Jul_09_SCAN[:,Jul_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_09_N36[:,Jul_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_09_NMP[:,Jul_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_09_N36_Tskin[Jul_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_09_NMP_Tskin[Jul_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jul_09_N36DA[:,Jul_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_09_NMPDA[:,Jul_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_09_N36DA_Tskin[Jul_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_09_NMPDA_Tskin[Jul_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Jul_09_ISCCP[Jul_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_09_Inc+=1
        if the_hour == 12:
            Jul_12_SCAN[:,Jul_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_12_N36[:,Jul_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_12_NMP[:,Jul_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_12_N36_Tskin[Jul_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_12_NMP_Tskin[Jul_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jul_12_N36DA[:,Jul_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_12_NMPDA[:,Jul_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_12_N36DA_Tskin[Jul_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_12_NMPDA_Tskin[Jul_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Jul_12_ISCCP[Jul_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_12_Inc+=1
        if the_hour == 15:
            Jul_15_SCAN[:,Jul_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_15_N36[:,Jul_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_15_NMP[:,Jul_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_15_N36_Tskin[Jul_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_15_NMP_Tskin[Jul_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Jul_15_N36DA[:,Jul_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_15_NMPDA[:,Jul_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_15_N36DA_Tskin[Jul_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_15_NMPDA_Tskin[Jul_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jul_15_ISCCP[Jul_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_15_Inc+=1
        if the_hour == 18:
            Jul_18_SCAN[:,Jul_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_18_N36[:,Jul_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_18_NMP[:,Jul_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_18_N36_Tskin[Jul_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_18_NMP_Tskin[Jul_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Jul_18_N36DA[:,Jul_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_18_NMPDA[:,Jul_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_18_N36DA_Tskin[Jul_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_18_NMPDA_Tskin[Jul_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jul_18_ISCCP[Jul_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_18_Inc+=1
        if the_hour == 21:
            Jul_21_SCAN[:,Jul_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Jul_21_N36[:,Jul_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Jul_21_NMP[:,Jul_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Jul_21_N36_Tskin[Jul_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_21_NMP_Tskin[Jul_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Jul_21_N36DA[:,Jul_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Jul_21_NMPDA[:,Jul_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Jul_21_N36DA_Tskin[Jul_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Jul_21_NMPDA_Tskin[Jul_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Jul_21_ISCCP[Jul_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Jul_21_Inc+=1
        Jul_Inc+=1
    if the_month == 8:
        Aug_SCAN[:,Aug_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Aug_N36[:,Aug_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Aug_MP[:,Aug_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Aug_N36_Tskin[Aug_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Aug_NMP_Tskin[Aug_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
        
        Aug_N36DA[:,Aug_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Aug_MPDA[:,Aug_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Aug_N36DA_Tskin[Aug_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Aug_NMPDA_Tskin[Aug_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

        Aug_ISCCP[Aug_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Aug_00_SCAN[:,Aug_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_00_N36[:,Aug_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_00_NMP[:,Aug_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_00_N36_Tskin[Aug_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_00_NMP_Tskin[Aug_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Aug_00_N36DA[:,Aug_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_00_NMPDA[:,Aug_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_00_N36DA_Tskin[Aug_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_00_NMPDA_Tskin[Aug_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Aug_00_ISCCP[Aug_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_00_Inc+=1
        if the_hour == 3:
            Aug_03_SCAN[:,Aug_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_03_N36[:,Aug_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_03_NMP[:,Aug_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_03_N36_Tskin[Aug_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_03_NMP_Tskin[Aug_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Aug_03_N36DA[:,Aug_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_03_NMPDA[:,Aug_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_03_N36DA_Tskin[Aug_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_03_NMPDA_Tskin[Aug_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_03_ISCCP[Aug_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_03_Inc+=1
        if the_hour == 6:
            Aug_06_SCAN[:,Aug_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_06_N36[:,Aug_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_06_NMP[:,Aug_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_06_N36_Tskin[Aug_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_06_NMP_Tskin[Aug_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Aug_06_N36DA[:,Aug_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_06_NMPDA[:,Aug_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_06_N36DA_Tskin[Aug_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_06_NMPDA_Tskin[Aug_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_06_ISCCP[Aug_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_06_Inc+=1
        if the_hour == 9:
            Aug_09_SCAN[:,Aug_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_09_N36[:,Aug_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_09_NMP[:,Aug_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_09_N36_Tskin[Aug_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_09_NMP_Tskin[Aug_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Aug_09_N36DA[:,Aug_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_09_NMPDA[:,Aug_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_09_N36DA_Tskin[Aug_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_09_NMPDA_Tskin[Aug_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_09_ISCCP[Aug_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_09_Inc+=1
        if the_hour == 12:
            Aug_12_SCAN[:,Aug_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_12_N36[:,Aug_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_12_NMP[:,Aug_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_12_N36_Tskin[Aug_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_12_NMP_Tskin[Aug_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Aug_12_N36DA[:,Aug_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_12_NMPDA[:,Aug_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_12_N36DA_Tskin[Aug_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_12_NMPDA_Tskin[Aug_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_12_ISCCP[Aug_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_12_Inc+=1
        if the_hour == 15:
            Aug_15_SCAN[:,Aug_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_15_N36[:,Aug_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_15_NMP[:,Aug_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_15_N36_Tskin[Aug_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_15_NMP_Tskin[Aug_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Aug_15_N36DA[:,Aug_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_15_NMPDA[:,Aug_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_15_N36DA_Tskin[Aug_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_15_NMPDA_Tskin[Aug_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_15_ISCCP[Aug_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_15_Inc+=1
        if the_hour == 18:
            Aug_18_SCAN[:,Aug_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_18_N36[:,Aug_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_18_NMP[:,Aug_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_18_N36_Tskin[Aug_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_18_NMP_Tskin[Aug_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Aug_18_N36DA[:,Aug_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_18_NMPDA[:,Aug_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_18_N36DA_Tskin[Aug_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_18_NMPDA_Tskin[Aug_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_18_ISCCP[Aug_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_18_Inc+=1
        if the_hour == 21:
            Aug_21_SCAN[:,Aug_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Aug_21_N36[:,Aug_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Aug_21_NMP[:,Aug_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Aug_21_N36_Tskin[Aug_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_21_NMP_Tskin[Aug_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Aug_21_N36DA[:,Aug_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Aug_21_NMPDA[:,Aug_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Aug_21_N36DA_Tskin[Aug_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Aug_21_NMPDA_Tskin[Aug_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Aug_21_ISCCP[Aug_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Aug_21_Inc+=1
        Aug_Inc+=1
    if the_month == 9:
        Sep_SCAN[:,Sep_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Sep_N36[:,Sep_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Sep_MP[:,Sep_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Sep_N36_Tskin[Sep_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Sep_NMP_Tskin[Sep_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

        Sep_N36DA[:,Sep_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Sep_MPDA[:,Sep_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Sep_N36DA_Tskin[Sep_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Sep_NMPDA_Tskin[Sep_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]

        Sep_ISCCP[Sep_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Sep_00_SCAN[:,Sep_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_00_N36[:,Sep_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_00_NMP[:,Sep_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_00_N36_Tskin[Sep_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_00_NMP_Tskin[Sep_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Sep_00_N36DA[:,Sep_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_00_NMPDA[:,Sep_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_00_N36DA_Tskin[Sep_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_00_NMPDA_Tskin[Sep_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Sep_00_ISCCP[Sep_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_00_Inc+=1
        if the_hour == 3:
            Sep_03_SCAN[:,Sep_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_03_N36[:,Sep_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_03_NMP[:,Sep_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_03_N36_Tskin[Sep_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_03_NMP_Tskin[Sep_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_03_N36DA[:,Sep_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_03_NMPDA[:,Sep_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_03_N36DA_Tskin[Sep_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_03_NMPDA_Tskin[Sep_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_03_ISCCP[Sep_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_03_Inc+=1
        if the_hour == 6:
            Sep_06_SCAN[:,Sep_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_06_N36[:,Sep_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_06_NMP[:,Sep_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_06_N36_Tskin[Sep_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_06_NMP_Tskin[Sep_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_06_N36DA[:,Sep_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_06_NMPDA[:,Sep_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_06_N36DA_Tskin[Sep_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_06_NMPDA_Tskin[Sep_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_06_ISCCP[Sep_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_06_Inc+=1
        if the_hour == 9:
            Sep_09_SCAN[:,Sep_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_09_N36[:,Sep_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_09_NMP[:,Sep_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_09_N36_Tskin[Sep_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_09_NMP_Tskin[Sep_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_09_N36DA[:,Sep_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_09_NMPDA[:,Sep_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_09_N36DA_Tskin[Sep_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_09_NMPDA_Tskin[Sep_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_09_ISCCP[Sep_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_09_Inc+=1
        if the_hour == 12:
            Sep_12_SCAN[:,Sep_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_12_N36[:,Sep_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_12_NMP[:,Sep_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_12_N36_Tskin[Sep_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_12_NMP_Tskin[Sep_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_12_N36DA[:,Sep_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_12_NMPDA[:,Sep_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_12_N36DA_Tskin[Sep_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_12_NMPDA_Tskin[Sep_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_12_ISCCP[Sep_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_12_Inc+=1
        if the_hour == 15:
            Sep_15_SCAN[:,Sep_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_15_N36[:,Sep_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_15_NMP[:,Sep_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_15_N36_Tskin[Sep_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_15_NMP_Tskin[Sep_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_15_N36DA[:,Sep_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_15_NMPDA[:,Sep_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_15_N36DA_Tskin[Sep_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_15_NMPDA_Tskin[Sep_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_15_ISCCP[Sep_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_15_Inc+=1
        if the_hour == 18:
            Sep_18_SCAN[:,Sep_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_18_N36[:,Sep_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_18_NMP[:,Sep_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_18_N36_Tskin[Sep_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_18_NMP_Tskin[Sep_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_18_N36DA[:,Sep_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_18_NMPDA[:,Sep_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_18_N36DA_Tskin[Sep_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_18_NMPDA_Tskin[Sep_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_18_ISCCP[Sep_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_18_Inc+=1
        if the_hour == 21:
            Sep_21_SCAN[:,Sep_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Sep_21_N36[:,Sep_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Sep_21_NMP[:,Sep_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Sep_21_N36_Tskin[Sep_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_21_NMP_Tskin[Sep_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Sep_21_N36DA[:,Sep_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Sep_21_NMPDA[:,Sep_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Sep_21_N36DA_Tskin[Sep_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Sep_21_NMPDA_Tskin[Sep_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
         
            Sep_21_ISCCP[Sep_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Sep_21_Inc+=1
        Sep_Inc+=1
    if the_month == 10:
        Oct_SCAN[:,Oct_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Oct_N36[:,Oct_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Oct_MP[:,Oct_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Oct_N36_Tskin[Oct_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Oct_NMP_Tskin[Oct_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
        
        Oct_N36DA[:,Oct_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Oct_MPDA[:,Oct_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Oct_N36DA_Tskin[Oct_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Oct_NMPDA_Tskin[Oct_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Oct_ISCCP[Oct_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Oct_00_SCAN[:,Oct_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_00_N36[:,Oct_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_00_NMP[:,Oct_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_00_N36_Tskin[Oct_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_00_NMP_Tskin[Oct_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Oct_00_N36DA[:,Oct_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_00_NMPDA[:,Oct_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_00_N36DA_Tskin[Oct_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_00_NMPDA_Tskin[Oct_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_00_ISCCP[Oct_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_00_Inc+=1
        if the_hour == 3:
            Oct_03_SCAN[:,Oct_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_03_N36[:,Oct_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_03_NMP[:,Oct_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_03_N36_Tskin[Oct_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_03_NMP_Tskin[Oct_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_03_N36DA[:,Oct_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_03_NMPDA[:,Oct_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_03_N36DA_Tskin[Oct_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_03_NMPDA_Tskin[Oct_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_03_ISCCP[Oct_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_03_Inc+=1
        if the_hour == 6:
            Oct_06_SCAN[:,Oct_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_06_N36[:,Oct_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_06_NMP[:,Oct_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_06_N36_Tskin[Oct_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_06_NMP_Tskin[Oct_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_06_N36DA[:,Oct_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_06_NMPDA[:,Oct_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_06_N36DA_Tskin[Oct_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_06_NMPDA_Tskin[Oct_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_06_ISCCP[Oct_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_06_Inc+=1
        if the_hour == 9:
            Oct_09_SCAN[:,Oct_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_09_N36[:,Oct_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_09_NMP[:,Oct_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_09_N36_Tskin[Oct_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_09_NMP_Tskin[Oct_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_09_N36DA[:,Oct_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_09_NMPDA[:,Oct_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_09_N36DA_Tskin[Oct_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_09_NMPDA_Tskin[Oct_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_09_ISCCP[Oct_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_09_Inc+=1
        if the_hour == 12:
            Oct_12_SCAN[:,Oct_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_12_N36[:,Oct_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_12_NMP[:,Oct_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_12_N36_Tskin[Oct_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_12_NMP_Tskin[Oct_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_12_N36DA[:,Oct_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_12_NMPDA[:,Oct_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_12_N36DA_Tskin[Oct_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_12_NMPDA_Tskin[Oct_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_12_ISCCP[Oct_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_12_Inc+=1
        if the_hour == 15:
            Oct_15_SCAN[:,Oct_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_15_N36[:,Oct_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_15_NMP[:,Oct_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_15_N36_Tskin[Oct_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_15_NMP_Tskin[Oct_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_15_N36DA[:,Oct_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_15_NMPDA[:,Oct_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_15_N36DA_Tskin[Oct_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_15_NMPDA_Tskin[Oct_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_15_ISCCP[Oct_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_15_Inc+=1
        if the_hour == 18:
            Oct_18_SCAN[:,Oct_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_18_N36[:,Oct_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_18_NMP[:,Oct_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_18_N36_Tskin[Oct_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_18_NMP_Tskin[Oct_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_18_N36DA[:,Oct_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_18_NMPDA[:,Oct_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_18_N36DA_Tskin[Oct_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_18_NMPDA_Tskin[Oct_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_18_ISCCP[Oct_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_18_Inc+=1
        if the_hour == 21:
            Oct_21_SCAN[:,Oct_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Oct_21_N36[:,Oct_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Oct_21_NMP[:,Oct_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Oct_21_N36_Tskin[Oct_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_21_NMP_Tskin[Oct_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Oct_21_N36DA[:,Oct_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Oct_21_NMPDA[:,Oct_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Oct_21_N36DA_Tskin[Oct_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Oct_21_NMPDA_Tskin[Oct_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Oct_21_ISCCP[Oct_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Oct_21_Inc+=1
        Oct_Inc+=1
    if the_month == 11:
        Nov_SCAN[:,Nov_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Nov_N36[:,Nov_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Nov_MP[:,Nov_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Nov_N36_Tskin[Nov_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Nov_NMP_Tskin[Nov_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

        Nov_N36DA[:,Nov_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Nov_MPDA[:,Nov_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Nov_N36DA_Tskin[Nov_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Nov_NMPDA_Tskin[Nov_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Nov_ISCCP[Nov_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Nov_00_SCAN[:,Nov_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_00_N36[:,Nov_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_00_NMP[:,Nov_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_00_N36_Tskin[Nov_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_00_NMP_Tskin[Nov_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Nov_00_N36DA[:,Nov_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_00_NMPDA[:,Nov_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_00_N36DA_Tskin[Nov_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_00_NMPDA_Tskin[Nov_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_00_ISCCP[Nov_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_00_Inc+=1
        if the_hour == 3:
            Nov_03_SCAN[:,Nov_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_03_N36[:,Nov_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_03_NMP[:,Nov_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_03_N36_Tskin[Nov_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_03_NMP_Tskin[Nov_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_03_N36DA[:,Nov_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_03_NMPDA[:,Nov_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_03_N36DA_Tskin[Nov_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_03_NMPDA_Tskin[Nov_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_03_ISCCP[Nov_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_03_Inc+=1
        if the_hour == 6:
            Nov_06_SCAN[:,Nov_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_06_N36[:,Nov_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_06_NMP[:,Nov_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_06_N36_Tskin[Nov_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_06_NMP_Tskin[Nov_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_06_N36DA[:,Nov_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_06_NMPDA[:,Nov_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_06_N36DA_Tskin[Nov_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_06_NMPDA_Tskin[Nov_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_06_ISCCP[Nov_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_06_Inc+=1
        if the_hour == 9:
            Nov_09_SCAN[:,Nov_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_09_N36[:,Nov_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_09_NMP[:,Nov_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_09_N36_Tskin[Nov_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_09_NMP_Tskin[Nov_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_09_N36DA[:,Nov_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_09_NMPDA[:,Nov_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_09_N36DA_Tskin[Nov_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_09_NMPDA_Tskin[Nov_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_09_ISCCP[Nov_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_09_Inc+=1
        if the_hour == 12:
            Nov_12_SCAN[:,Nov_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_12_N36[:,Nov_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_12_NMP[:,Nov_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_12_N36_Tskin[Nov_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_12_NMP_Tskin[Nov_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_12_N36DA[:,Nov_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_12_NMPDA[:,Nov_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_12_N36DA_Tskin[Nov_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_12_NMPDA_Tskin[Nov_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_12_ISCCP[Nov_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_12_Inc+=1
        if the_hour == 15:
            Nov_15_SCAN[:,Nov_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_15_N36[:,Nov_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_15_NMP[:,Nov_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_15_N36_Tskin[Nov_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_15_NMP_Tskin[Nov_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_15_N36DA[:,Nov_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_15_NMPDA[:,Nov_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_15_N36DA_Tskin[Nov_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_15_NMPDA_Tskin[Nov_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_15_ISCCP[Nov_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_15_Inc+=1
        if the_hour == 18:
            Nov_18_SCAN[:,Nov_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_18_N36[:,Nov_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_18_NMP[:,Nov_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_18_N36_Tskin[Nov_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_18_NMP_Tskin[Nov_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_18_N36DA[:,Nov_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_18_NMPDA[:,Nov_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_18_N36DA_Tskin[Nov_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_18_NMPDA_Tskin[Nov_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_18_ISCCP[Nov_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_18_Inc+=1
        if the_hour == 21:
            Nov_21_SCAN[:,Nov_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Nov_21_N36[:,Nov_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Nov_21_NMP[:,Nov_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Nov_21_N36_Tskin[Nov_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_21_NMP_Tskin[Nov_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Nov_21_N36DA[:,Nov_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Nov_21_NMPDA[:,Nov_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Nov_21_N36DA_Tskin[Nov_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Nov_21_NMPDA_Tskin[Nov_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
  
            Nov_21_ISCCP[Nov_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Nov_21_Inc+=1
        Nov_Inc+=1
    if the_month == 12:
        Dec_SCAN[:,Dec_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
        Dec_N36[:,Dec_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
        Dec_MP[:,Dec_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
        Dec_N36_Tskin[Dec_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
        Dec_NMP_Tskin[Dec_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

        Dec_N36DA[:,Dec_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
        Dec_MPDA[:,Dec_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
        Dec_N36DA_Tskin[Dec_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
        Dec_NMPDA_Tskin[Dec_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
        
        Dec_ISCCP[Dec_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
        if the_hour == 0:
            Dec_00_SCAN[:,Dec_00_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_00_N36[:,Dec_00_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_00_NMP[:,Dec_00_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_00_N36_Tskin[Dec_00_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_00_NMP_Tskin[Dec_00_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Dec_00_N36DA[:,Dec_00_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_00_NMPDA[:,Dec_00_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_00_N36DA_Tskin[Dec_00_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_00_NMPDA_Tskin[Dec_00_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
            
            Dec_00_ISCCP[Dec_00_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_00_Inc+=1
        if the_hour == 3:
            Dec_03_SCAN[:,Dec_03_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_03_N36[:,Dec_03_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_03_NMP[:,Dec_03_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_03_N36_Tskin[Dec_03_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_03_NMP_Tskin[Dec_03_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Dec_03_N36DA[:,Dec_03_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_03_NMPDA[:,Dec_03_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_03_N36DA_Tskin[Dec_03_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_03_NMPDA_Tskin[Dec_03_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_03_ISCCP[Dec_03_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_03_Inc+=1
        if the_hour == 6:
            Dec_06_SCAN[:,Dec_06_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_06_N36[:,Dec_06_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_06_NMP[:,Dec_06_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_06_N36_Tskin[Dec_06_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_06_NMP_Tskin[Dec_06_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Dec_06_N36DA[:,Dec_06_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_06_NMPDA[:,Dec_06_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_06_N36DA_Tskin[Dec_06_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_06_NMPDA_Tskin[Dec_06_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_06_ISCCP[Dec_06_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_06_Inc+=1
        if the_hour == 9: 
            Dec_09_SCAN[:,Dec_09_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_09_N36[:,Dec_09_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_09_NMP[:,Dec_09_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_09_N36_Tskin[Dec_09_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_09_NMP_Tskin[Dec_09_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Dec_09_N36DA[:,Dec_09_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_09_NMPDA[:,Dec_09_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_09_N36DA_Tskin[Dec_09_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_09_NMPDA_Tskin[Dec_09_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_09_ISCCP[Dec_09_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_09_Inc+=1
        if the_hour == 12:
            Dec_12_SCAN[:,Dec_12_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_12_N36[:,Dec_12_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_12_NMP[:,Dec_12_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_12_N36_Tskin[Dec_12_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_12_NMP_Tskin[Dec_12_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Dec_12_N36DA[:,Dec_12_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_12_NMPDA[:,Dec_12_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_12_N36DA_Tskin[Dec_12_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_12_NMPDA_Tskin[Dec_12_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_12_ISCCP[Dec_12_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_12_Inc+=1
        if the_hour == 15:
            Dec_15_SCAN[:,Dec_15_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_15_N36[:,Dec_15_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_15_NMP[:,Dec_15_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_15_N36_Tskin[Dec_15_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_15_NMP_Tskin[Dec_15_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]

            Dec_15_N36DA[:,Dec_15_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_15_NMPDA[:,Dec_15_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_15_N36DA_Tskin[Dec_15_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_15_NMPDA_Tskin[Dec_15_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_15_ISCCP[Dec_15_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_15_Inc+=1
        if the_hour == 18:
            Dec_18_SCAN[:,Dec_18_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_18_N36[:,Dec_18_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_18_NMP[:,Dec_18_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_18_N36_Tskin[Dec_18_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_18_NMP_Tskin[Dec_18_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Dec_18_N36DA[:,Dec_18_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_18_NMPDA[:,Dec_18_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_18_N36DA_Tskin[Dec_18_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_18_NMPDA_Tskin[Dec_18_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_18_ISCCP[Dec_18_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_18_Inc+=1
        if the_hour == 21:
            Dec_21_SCAN[:,Dec_21_Inc,:]=SCAN_DATA_ARRAY[:,hour_loop,:]
            Dec_21_N36[:,Dec_21_Inc,:]=LIS_DATA_ARRAY[:,hour_loop,:]
            Dec_21_NMP[:,Dec_21_Inc,:]=LIS_MP_ARRAY[:,hour_loop,:]
            Dec_21_N36_Tskin[Dec_21_Inc,:]=LIS_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_21_NMP_Tskin[Dec_21_Inc,:]=LIS_MP_TSKIN_ARRAY[hour_loop,:]
            
            Dec_21_N36DA[:,Dec_21_Inc,:]=N36DA_DATA_ARRAY[:,hour_loop,:]
            Dec_21_NMPDA[:,Dec_21_Inc,:]=LIS_MPDA_ARRAY[:,hour_loop,:]
            Dec_21_N36DA_Tskin[Dec_21_Inc,:]=N36DA_TSKIN_DATA_ARRAY[hour_loop,:]
            Dec_21_NMPDA_Tskin[Dec_21_Inc,:]=LIS_MPDA_TSKIN_ARRAY[hour_loop,:]
 
            Dec_21_ISCCP[Dec_21_Inc,:]=ISCCP_TSKIN_ARRAY[hour_loop,:]
            Dec_21_Inc+=1


###########
# Create the winter arrays for SCAN data
###########

Winter_index=np.zeros((3), dtype='int32')
Winter_index[0]=Total_Month_days[11]
Winter_index[1]=Total_Month_days[11]+Total_Month_days[0]
Winter_index[2]=Total_Month_days[11]+np.sum(Total_Month_days[0:2])


Winter_00_SCAN[:,0:Winter_index[0]              ,:]=Dec_00_SCAN[:,:,:]
Winter_00_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_00_SCAN[:,:,:]
Winter_00_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_00_SCAN[:,:,:]

Winter_03_SCAN[:,0:Winter_index[0]              ,:]=Dec_03_SCAN[:,:,:]
Winter_03_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_03_SCAN[:,:,:]
Winter_03_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_03_SCAN[:,:,:]

Winter_06_SCAN[:,0:Winter_index[0]              ,:]=Dec_06_SCAN[:,:,:]
Winter_06_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_06_SCAN[:,:,:]
Winter_06_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_06_SCAN[:,:,:]

Winter_09_SCAN[:,0:Winter_index[0]              ,:]=Dec_09_SCAN[:,:,:]
Winter_09_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_09_SCAN[:,:,:]
Winter_09_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_09_SCAN[:,:,:]

Winter_12_SCAN[:,0:Winter_index[0]              ,:]=Dec_12_SCAN[:,:,:]
Winter_12_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_12_SCAN[:,:,:]
Winter_12_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_12_SCAN[:,:,:]

Winter_15_SCAN[:,0:Winter_index[0]              ,:]=Dec_15_SCAN[:,:,:]
Winter_15_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_15_SCAN[:,:,:]
Winter_15_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_15_SCAN[:,:,:]

Winter_18_SCAN[:,0:Winter_index[0]              ,:]=Dec_18_SCAN[:,:,:]
Winter_18_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_18_SCAN[:,:,:]
Winter_18_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_18_SCAN[:,:,:]

Winter_21_SCAN[:,0:Winter_index[0]              ,:]=Dec_21_SCAN[:,:,:]
Winter_21_SCAN[:,Winter_index[0]:Winter_index[1],:]=Jan_21_SCAN[:,:,:]
Winter_21_SCAN[:,Winter_index[1]:Winter_index[2],:]=Feb_21_SCAN[:,:,:]

##########################

Winter_00_ISCCP[0:Winter_index[0]              ,:]=Dec_00_ISCCP[:,:]
Winter_00_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_00_ISCCP[:,:]
Winter_00_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_00_ISCCP[:,:]

Winter_03_ISCCP[0:Winter_index[0]              ,:]=Dec_03_ISCCP[:,:]
Winter_03_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_03_ISCCP[:,:]
Winter_03_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_03_ISCCP[:,:]

Winter_06_ISCCP[0:Winter_index[0]              ,:]=Dec_06_ISCCP[:,:]
Winter_06_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_06_ISCCP[:,:]
Winter_06_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_06_ISCCP[:,:]

Winter_09_ISCCP[0:Winter_index[0]              ,:]=Dec_09_ISCCP[:,:]
Winter_09_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_09_ISCCP[:,:]
Winter_09_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_09_ISCCP[:,:]

Winter_12_ISCCP[0:Winter_index[0]              ,:]=Dec_12_ISCCP[:,:]
Winter_12_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_12_ISCCP[:,:]
Winter_12_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_12_ISCCP[:,:]

Winter_15_ISCCP[0:Winter_index[0]              ,:]=Dec_15_ISCCP[:,:]
Winter_15_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_15_ISCCP[:,:]
Winter_15_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_15_ISCCP[:,:]

Winter_18_ISCCP[0:Winter_index[0]              ,:]=Dec_18_ISCCP[:,:]
Winter_18_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_18_ISCCP[:,:]
Winter_18_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_18_ISCCP[:,:]

Winter_21_ISCCP[0:Winter_index[0]              ,:]=Dec_21_ISCCP[:,:]
Winter_21_ISCCP[Winter_index[0]:Winter_index[1],:]=Jan_21_ISCCP[:,:]
Winter_21_ISCCP[Winter_index[1]:Winter_index[2],:]=Feb_21_ISCCP[:,:]

#######
#   Winter Noah MP arrays OPEN LOOP
#######

Winter_00_NMP[:,0:Winter_index[0]              ,:]=Dec_00_NMP[:,:,:]
Winter_00_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_00_NMP[:,:,:]
Winter_00_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_00_NMP[:,:,:]

Winter_03_NMP[:,0:Winter_index[0]              ,:]=Dec_03_NMP[:,:,:]
Winter_03_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_03_NMP[:,:,:]
Winter_03_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_03_NMP[:,:,:]

Winter_06_NMP[:,0:Winter_index[0]              ,:]=Dec_06_NMP[:,:,:]
Winter_06_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_06_NMP[:,:,:]
Winter_06_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_06_NMP[:,:,:]

Winter_09_NMP[:,0:Winter_index[0]              ,:]=Dec_09_NMP[:,:,:]
Winter_09_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_09_NMP[:,:,:]
Winter_09_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_09_NMP[:,:,:]

Winter_12_NMP[:,0:Winter_index[0]              ,:]=Dec_12_NMP[:,:,:]
Winter_12_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_12_NMP[:,:,:]
Winter_12_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_12_NMP[:,:,:]

Winter_15_NMP[:,0:Winter_index[0]              ,:]=Dec_15_NMP[:,:,:]
Winter_15_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_15_NMP[:,:,:]
Winter_15_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_15_NMP[:,:,:]

Winter_18_NMP[:,0:Winter_index[0]              ,:]=Dec_18_NMP[:,:,:]
Winter_18_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_18_NMP[:,:,:]
Winter_18_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_18_NMP[:,:,:]

Winter_21_NMP[:,0:Winter_index[0]              ,:]=Dec_21_NMP[:,:,:]
Winter_21_NMP[:,Winter_index[0]:Winter_index[1],:]=Jan_21_NMP[:,:,:]
Winter_21_NMP[:,Winter_index[1]:Winter_index[2],:]=Feb_21_NMP[:,:,:]

#######
#   Winter Noah MP TSKIN arrays OPEN LOOP
#######

Winter_00_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_00_NMP_Tskin[:,:]
Winter_00_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_00_NMP_Tskin[:,:]
Winter_00_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_00_NMP_Tskin[:,:]

Winter_03_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_03_NMP_Tskin[:,:]
Winter_03_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_03_NMP_Tskin[:,:]
Winter_03_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_03_NMP_Tskin[:,:]

Winter_06_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_06_NMP_Tskin[:,:]
Winter_06_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_06_NMP_Tskin[:,:]
Winter_06_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_06_NMP_Tskin[:,:]

Winter_09_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_09_NMP_Tskin[:,:]
Winter_09_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_09_NMP_Tskin[:,:]
Winter_09_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_09_NMP_Tskin[:,:]

Winter_12_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_12_NMP_Tskin[:,:]
Winter_12_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_12_NMP_Tskin[:,:]
Winter_12_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_12_NMP_Tskin[:,:]

Winter_15_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_15_NMP_Tskin[:,:]
Winter_15_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_15_NMP_Tskin[:,:]
Winter_15_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_15_NMP_Tskin[:,:]

Winter_18_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_18_NMP_Tskin[:,:]
Winter_18_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_18_NMP_Tskin[:,:]
Winter_18_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_18_NMP_Tskin[:,:]

Winter_21_NMP_Tskin[0:Winter_index[0]              ,:]=Dec_21_NMP_Tskin[:,:]
Winter_21_NMP_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_21_NMP_Tskin[:,:]
Winter_21_NMP_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_21_NMP_Tskin[:,:]

#######
#   Winter Noah MP arrays DA LOOP
#######

Winter_00_NMPDA[:,0:Winter_index[0]              ,:]=Dec_00_NMPDA[:,:,:]
Winter_00_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_00_NMPDA[:,:,:]
Winter_00_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_00_NMPDA[:,:,:]

Winter_03_NMPDA[:,0:Winter_index[0]              ,:]=Dec_03_NMPDA[:,:,:]
Winter_03_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_03_NMPDA[:,:,:]
Winter_03_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_03_NMPDA[:,:,:]

Winter_06_NMPDA[:,0:Winter_index[0]              ,:]=Dec_06_NMPDA[:,:,:]
Winter_06_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_06_NMPDA[:,:,:]
Winter_06_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_06_NMPDA[:,:,:]

Winter_09_NMPDA[:,0:Winter_index[0]              ,:]=Dec_09_NMPDA[:,:,:]
Winter_09_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_09_NMPDA[:,:,:]
Winter_09_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_09_NMPDA[:,:,:]

Winter_12_NMPDA[:,0:Winter_index[0]              ,:]=Dec_12_NMPDA[:,:,:]
Winter_12_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_12_NMPDA[:,:,:]
Winter_12_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_12_NMPDA[:,:,:]

Winter_15_NMPDA[:,0:Winter_index[0]              ,:]=Dec_15_NMPDA[:,:,:]
Winter_15_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_15_NMPDA[:,:,:]
Winter_15_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_15_NMPDA[:,:,:]

Winter_18_NMPDA[:,0:Winter_index[0]              ,:]=Dec_18_NMPDA[:,:,:]
Winter_18_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_18_NMPDA[:,:,:]
Winter_18_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_18_NMPDA[:,:,:]

Winter_21_NMPDA[:,0:Winter_index[0]              ,:]=Dec_21_NMPDA[:,:,:]
Winter_21_NMPDA[:,Winter_index[0]:Winter_index[1],:]=Jan_21_NMPDA[:,:,:]
Winter_21_NMPDA[:,Winter_index[1]:Winter_index[2],:]=Feb_21_NMPDA[:,:,:]

#######
#   Winter Noah MP TSKIN arrays DA LOOP
#######

Winter_00_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_00_NMPDA_Tskin[:,:]
Winter_00_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_00_NMPDA_Tskin[:,:]
Winter_00_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_00_NMPDA_Tskin[:,:]

Winter_03_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_03_NMPDA_Tskin[:,:]
Winter_03_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_03_NMPDA_Tskin[:,:]
Winter_03_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_03_NMPDA_Tskin[:,:]

Winter_06_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_06_NMPDA_Tskin[:,:]
Winter_06_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_06_NMPDA_Tskin[:,:]
Winter_06_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_06_NMPDA_Tskin[:,:]

Winter_09_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_09_NMPDA_Tskin[:,:]
Winter_09_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_09_NMPDA_Tskin[:,:]
Winter_09_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_09_NMPDA_Tskin[:,:]

Winter_12_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_12_NMPDA_Tskin[:,:]
Winter_12_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_12_NMPDA_Tskin[:,:]
Winter_12_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_12_NMPDA_Tskin[:,:]

Winter_15_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_15_NMPDA_Tskin[:,:]
Winter_15_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_15_NMPDA_Tskin[:,:]
Winter_15_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_15_NMPDA_Tskin[:,:]

Winter_18_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_18_NMPDA_Tskin[:,:]
Winter_18_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_18_NMPDA_Tskin[:,:]
Winter_18_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_18_NMPDA_Tskin[:,:]

Winter_21_NMPDA_Tskin[0:Winter_index[0]              ,:]=Dec_21_NMPDA_Tskin[:,:]
Winter_21_NMPDA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_21_NMPDA_Tskin[:,:]
Winter_21_NMPDA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_21_NMPDA_Tskin[:,:]

#######
#   Winter Noah 36 arrays  OPEN LOOP
#######

Winter_00_N36[:,0:Winter_index[0]              ,:]=Dec_00_N36[:,:,:]
Winter_00_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_00_N36[:,:,:]
Winter_00_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_00_N36[:,:,:]

Winter_03_N36[:,0:Winter_index[0]              ,:]=Dec_03_N36[:,:,:]
Winter_03_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_03_N36[:,:,:]
Winter_03_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_03_N36[:,:,:]

Winter_06_N36[:,0:Winter_index[0]              ,:]=Dec_06_N36[:,:,:]
Winter_06_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_06_N36[:,:,:]
Winter_06_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_06_N36[:,:,:]

Winter_09_N36[:,0:Winter_index[0]              ,:]=Dec_09_N36[:,:,:]
Winter_09_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_09_N36[:,:,:]
Winter_09_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_09_N36[:,:,:]

Winter_12_N36[:,0:Winter_index[0]              ,:]=Dec_12_N36[:,:,:]
Winter_12_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_12_N36[:,:,:]
Winter_12_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_12_N36[:,:,:]

Winter_15_N36[:,0:Winter_index[0]              ,:]=Dec_15_N36[:,:,:]
Winter_15_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_15_N36[:,:,:]
Winter_15_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_15_N36[:,:,:]

Winter_18_N36[:,0:Winter_index[0]              ,:]=Dec_18_N36[:,:,:]
Winter_18_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_18_N36[:,:,:]
Winter_18_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_18_N36[:,:,:]

Winter_21_N36[:,0:Winter_index[0]              ,:]=Dec_21_N36[:,:,:]
Winter_21_N36[:,Winter_index[0]:Winter_index[1],:]=Jan_21_N36[:,:,:]
Winter_21_N36[:,Winter_index[1]:Winter_index[2],:]=Feb_21_N36[:,:,:]

#######
#   Winter Noah 36 TSKIN arrays  OPEN LOOP
#######

Winter_00_N36_Tskin[0:Winter_index[0]              ,:]=Dec_00_N36_Tskin[:,:]
Winter_00_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_00_N36_Tskin[:,:]
Winter_00_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_00_N36_Tskin[:,:]

Winter_03_N36_Tskin[0:Winter_index[0]              ,:]=Dec_03_N36_Tskin[:,:]
Winter_03_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_03_N36_Tskin[:,:]
Winter_03_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_03_N36_Tskin[:,:]

Winter_06_N36_Tskin[0:Winter_index[0]              ,:]=Dec_06_N36_Tskin[:,:]
Winter_06_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_06_N36_Tskin[:,:]
Winter_06_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_06_N36_Tskin[:,:]

Winter_09_N36_Tskin[0:Winter_index[0]              ,:]=Dec_09_N36_Tskin[:,:]
Winter_09_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_09_N36_Tskin[:,:]
Winter_09_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_09_N36_Tskin[:,:]

Winter_12_N36_Tskin[0:Winter_index[0]              ,:]=Dec_12_N36_Tskin[:,:]
Winter_12_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_12_N36_Tskin[:,:]
Winter_12_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_12_N36_Tskin[:,:]

Winter_15_N36_Tskin[0:Winter_index[0]              ,:]=Dec_15_N36_Tskin[:,:]
Winter_15_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_15_N36_Tskin[:,:]
Winter_15_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_15_N36_Tskin[:,:]

Winter_18_N36_Tskin[0:Winter_index[0]              ,:]=Dec_18_N36_Tskin[:,:]
Winter_18_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_18_N36_Tskin[:,:]
Winter_18_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_18_N36_Tskin[:,:]

Winter_21_N36_Tskin[0:Winter_index[0]              ,:]=Dec_21_N36_Tskin[:,:]
Winter_21_N36_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_21_N36_Tskin[:,:]
Winter_21_N36_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_21_N36_Tskin[:,:]

#######
#   Winter Noah 36 arrays DA LOOP
#######

Winter_00_N36DA[:,0:Winter_index[0]              ,:]=Dec_00_N36DA[:,:,:]
Winter_00_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_00_N36DA[:,:,:]
Winter_00_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_00_N36DA[:,:,:]

Winter_03_N36DA[:,0:Winter_index[0]              ,:]=Dec_03_N36DA[:,:,:]
Winter_03_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_03_N36DA[:,:,:]
Winter_03_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_03_N36DA[:,:,:]

Winter_06_N36DA[:,0:Winter_index[0]              ,:]=Dec_06_N36DA[:,:,:]
Winter_06_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_06_N36DA[:,:,:]
Winter_06_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_06_N36DA[:,:,:]

Winter_09_N36DA[:,0:Winter_index[0]              ,:]=Dec_09_N36DA[:,:,:]
Winter_09_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_09_N36DA[:,:,:]
Winter_09_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_09_N36DA[:,:,:]

Winter_12_N36DA[:,0:Winter_index[0]              ,:]=Dec_12_N36DA[:,:,:]
Winter_12_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_12_N36DA[:,:,:]
Winter_12_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_12_N36DA[:,:,:]

Winter_15_N36DA[:,0:Winter_index[0]              ,:]=Dec_15_N36DA[:,:,:]
Winter_15_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_15_N36DA[:,:,:]
Winter_15_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_15_N36DA[:,:,:]

Winter_18_N36DA[:,0:Winter_index[0]              ,:]=Dec_18_N36DA[:,:,:]
Winter_18_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_18_N36DA[:,:,:]
Winter_18_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_18_N36DA[:,:,:]

Winter_21_N36DA[:,0:Winter_index[0]              ,:]=Dec_21_N36DA[:,:,:]
Winter_21_N36DA[:,Winter_index[0]:Winter_index[1],:]=Jan_21_N36DA[:,:,:]
Winter_21_N36DA[:,Winter_index[1]:Winter_index[2],:]=Feb_21_N36DA[:,:,:]

#######
#   Winter Noah 36 TSKIN arrays DA LOOP
#######

Winter_00_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_00_N36DA_Tskin[:,:]
Winter_00_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_00_N36DA_Tskin[:,:]
Winter_00_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_00_N36DA_Tskin[:,:]

Winter_03_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_03_N36DA_Tskin[:,:]
Winter_03_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_03_N36DA_Tskin[:,:]
Winter_03_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_03_N36DA_Tskin[:,:]

Winter_06_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_06_N36DA_Tskin[:,:]
Winter_06_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_06_N36DA_Tskin[:,:]
Winter_06_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_06_N36DA_Tskin[:,:]

Winter_09_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_09_N36DA_Tskin[:,:]
Winter_09_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_09_N36DA_Tskin[:,:]
Winter_09_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_09_N36DA_Tskin[:,:]

Winter_12_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_12_N36DA_Tskin[:,:]
Winter_12_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_12_N36DA_Tskin[:,:]
Winter_12_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_12_N36DA_Tskin[:,:]

Winter_15_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_15_N36DA_Tskin[:,:]
Winter_15_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_15_N36DA_Tskin[:,:]
Winter_15_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_15_N36DA_Tskin[:,:]

Winter_18_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_18_N36DA_Tskin[:,:]
Winter_18_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_18_N36DA_Tskin[:,:]
Winter_18_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_18_N36DA_Tskin[:,:]

Winter_21_N36DA_Tskin[0:Winter_index[0]              ,:]=Dec_21_N36DA_Tskin[:,:]
Winter_21_N36DA_Tskin[Winter_index[0]:Winter_index[1],:]=Jan_21_N36DA_Tskin[:,:]
Winter_21_N36DA_Tskin[Winter_index[1]:Winter_index[2],:]=Feb_21_N36DA_Tskin[:,:]


#######
#   Spring SCAN arrays
#######

Spring_index=np.zeros((3), dtype='int32')
Spring_index[0]=Total_Month_days[2]
Spring_index[1]=np.sum(Total_Month_days[2:4])
Spring_index[2]=np.sum(Total_Month_days[2:5])

Spring_00_SCAN[:,0:Spring_index[0]              ,:]=Mar_00_SCAN[:,:,:]
Spring_00_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_00_SCAN[:,:,:]
Spring_00_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_00_SCAN[:,:,:]

Spring_03_SCAN[:,0:Spring_index[0]              ,:]=Mar_03_SCAN[:,:,:]
Spring_03_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_03_SCAN[:,:,:]
Spring_03_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_03_SCAN[:,:,:]

Spring_06_SCAN[:,0:Spring_index[0]              ,:]=Mar_06_SCAN[:,:,:]
Spring_06_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_06_SCAN[:,:,:]
Spring_06_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_06_SCAN[:,:,:]

Spring_09_SCAN[:,0:Spring_index[0]              ,:]=Mar_09_SCAN[:,:,:]
Spring_09_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_09_SCAN[:,:,:]
Spring_09_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_09_SCAN[:,:,:]

Spring_12_SCAN[:,0:Spring_index[0]              ,:]=Mar_12_SCAN[:,:,:]
Spring_12_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_12_SCAN[:,:,:]
Spring_12_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_12_SCAN[:,:,:]

Spring_15_SCAN[:,0:Spring_index[0]              ,:]=Mar_15_SCAN[:,:,:]
Spring_15_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_15_SCAN[:,:,:]
Spring_15_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_15_SCAN[:,:,:]

Spring_18_SCAN[:,0:Spring_index[0]              ,:]=Mar_18_SCAN[:,:,:]
Spring_18_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_18_SCAN[:,:,:]
Spring_18_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_18_SCAN[:,:,:]

Spring_21_SCAN[:,0:Spring_index[0]              ,:]=Mar_21_SCAN[:,:,:]
Spring_21_SCAN[:,Spring_index[0]:Spring_index[1],:]=Apr_21_SCAN[:,:,:]
Spring_21_SCAN[:,Spring_index[1]:Spring_index[2],:]=May_21_SCAN[:,:,:]

#######
#######

Spring_00_ISCCP[0:Spring_index[0]              ,:]=Mar_00_ISCCP[:,:]
Spring_00_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_00_ISCCP[:,:]
Spring_00_ISCCP[Spring_index[1]:Spring_index[2],:]=May_00_ISCCP[:,:]

Spring_03_ISCCP[0:Spring_index[0]              ,:]=Mar_03_ISCCP[:,:]
Spring_03_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_03_ISCCP[:,:]
Spring_03_ISCCP[Spring_index[1]:Spring_index[2],:]=May_03_ISCCP[:,:]

Spring_06_ISCCP[0:Spring_index[0]              ,:]=Mar_06_ISCCP[:,:]
Spring_06_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_06_ISCCP[:,:]
Spring_06_ISCCP[Spring_index[1]:Spring_index[2],:]=May_06_ISCCP[:,:]

Spring_09_ISCCP[0:Spring_index[0]              ,:]=Mar_09_ISCCP[:,:]
Spring_09_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_09_ISCCP[:,:]
Spring_09_ISCCP[Spring_index[1]:Spring_index[2],:]=May_09_ISCCP[:,:]

Spring_12_ISCCP[0:Spring_index[0]              ,:]=Mar_12_ISCCP[:,:]
Spring_12_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_12_ISCCP[:,:]
Spring_12_ISCCP[Spring_index[1]:Spring_index[2],:]=May_12_ISCCP[:,:]

Spring_15_ISCCP[0:Spring_index[0]              ,:]=Mar_15_ISCCP[:,:]
Spring_15_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_15_ISCCP[:,:]
Spring_15_ISCCP[Spring_index[1]:Spring_index[2],:]=May_15_ISCCP[:,:]

Spring_18_ISCCP[0:Spring_index[0]              ,:]=Mar_18_ISCCP[:,:]
Spring_18_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_18_ISCCP[:,:]
Spring_18_ISCCP[Spring_index[1]:Spring_index[2],:]=May_18_ISCCP[:,:]

Spring_21_ISCCP[0:Spring_index[0]              ,:]=Mar_21_ISCCP[:,:]
Spring_21_ISCCP[Spring_index[0]:Spring_index[1],:]=Apr_21_ISCCP[:,:]
Spring_21_ISCCP[Spring_index[1]:Spring_index[2],:]=May_21_ISCCP[:,:]

#######
#   Spring NoahMP arrays OPEN LOOP
#######

Spring_00_NMP[:,0:Spring_index[0]              ,:]=Mar_00_NMP[:,:,:]
Spring_00_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_00_NMP[:,:,:]
Spring_00_NMP[:,Spring_index[1]:Spring_index[2],:]=May_00_NMP[:,:,:]

Spring_03_NMP[:,0:Spring_index[0]              ,:]=Mar_03_NMP[:,:,:]
Spring_03_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_03_NMP[:,:,:]
Spring_03_NMP[:,Spring_index[1]:Spring_index[2],:]=May_03_NMP[:,:,:]

Spring_06_NMP[:,0:Spring_index[0]              ,:]=Mar_06_NMP[:,:,:]
Spring_06_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_06_NMP[:,:,:]
Spring_06_NMP[:,Spring_index[1]:Spring_index[2],:]=May_06_NMP[:,:,:]

Spring_09_NMP[:,0:Spring_index[0]              ,:]=Mar_09_NMP[:,:,:]
Spring_09_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_09_NMP[:,:,:]
Spring_09_NMP[:,Spring_index[1]:Spring_index[2],:]=May_09_NMP[:,:,:]

Spring_12_NMP[:,0:Spring_index[0]              ,:]=Mar_12_NMP[:,:,:]
Spring_12_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_12_NMP[:,:,:]
Spring_12_NMP[:,Spring_index[1]:Spring_index[2],:]=May_12_NMP[:,:,:]

Spring_15_NMP[:,0:Spring_index[0]              ,:]=Mar_15_NMP[:,:,:]
Spring_15_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_15_NMP[:,:,:]
Spring_15_NMP[:,Spring_index[1]:Spring_index[2],:]=May_15_NMP[:,:,:]

Spring_18_NMP[:,0:Spring_index[0]              ,:]=Mar_18_NMP[:,:,:]
Spring_18_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_18_NMP[:,:,:]
Spring_18_NMP[:,Spring_index[1]:Spring_index[2],:]=May_18_NMP[:,:,:]

Spring_21_NMP[:,0:Spring_index[0]              ,:]=Mar_21_NMP[:,:,:]
Spring_21_NMP[:,Spring_index[0]:Spring_index[1],:]=Apr_21_NMP[:,:,:]
Spring_21_NMP[:,Spring_index[1]:Spring_index[2],:]=May_21_NMP[:,:,:]

#######
#   Spring NoahMP TSKIN arrays OPEN LOOP
#######

Spring_00_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_00_NMP_Tskin[:,:]
Spring_00_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_00_NMP_Tskin[:,:]
Spring_00_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_00_NMP_Tskin[:,:]

Spring_03_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_03_NMP_Tskin[:,:]
Spring_03_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_03_NMP_Tskin[:,:]
Spring_03_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_03_NMP_Tskin[:,:]

Spring_06_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_06_NMP_Tskin[:,:]
Spring_06_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_06_NMP_Tskin[:,:]
Spring_06_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_06_NMP_Tskin[:,:]

Spring_09_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_09_NMP_Tskin[:,:]
Spring_09_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_09_NMP_Tskin[:,:]
Spring_09_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_09_NMP_Tskin[:,:]

Spring_12_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_12_NMP_Tskin[:,:]
Spring_12_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_12_NMP_Tskin[:,:]
Spring_12_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_12_NMP_Tskin[:,:]

Spring_15_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_15_NMP_Tskin[:,:]
Spring_15_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_15_NMP_Tskin[:,:]
Spring_15_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_15_NMP_Tskin[:,:]

Spring_18_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_18_NMP_Tskin[:,:]
Spring_18_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_18_NMP_Tskin[:,:]
Spring_18_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_18_NMP_Tskin[:,:]

Spring_21_NMP_Tskin[0:Spring_index[0]              ,:]=Mar_21_NMP_Tskin[:,:]
Spring_21_NMP_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_21_NMP_Tskin[:,:]
Spring_21_NMP_Tskin[Spring_index[1]:Spring_index[2],:]=May_21_NMP_Tskin[:,:]


#######
#   Spring NoahMP arrays DA LOOP
#######

Spring_00_NMPDA[:,0:Spring_index[0]              ,:]=Mar_00_NMPDA[:,:,:]
Spring_00_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_00_NMPDA[:,:,:]
Spring_00_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_00_NMPDA[:,:,:]

Spring_03_NMPDA[:,0:Spring_index[0]              ,:]=Mar_03_NMPDA[:,:,:]
Spring_03_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_03_NMPDA[:,:,:]
Spring_03_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_03_NMPDA[:,:,:]

Spring_06_NMPDA[:,0:Spring_index[0]              ,:]=Mar_06_NMPDA[:,:,:]
Spring_06_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_06_NMPDA[:,:,:]
Spring_06_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_06_NMPDA[:,:,:]

Spring_09_NMPDA[:,0:Spring_index[0]              ,:]=Mar_09_NMPDA[:,:,:]
Spring_09_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_09_NMPDA[:,:,:]
Spring_09_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_09_NMPDA[:,:,:]

Spring_12_NMPDA[:,0:Spring_index[0]              ,:]=Mar_12_NMPDA[:,:,:]
Spring_12_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_12_NMPDA[:,:,:]
Spring_12_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_12_NMPDA[:,:,:]

Spring_15_NMPDA[:,0:Spring_index[0]              ,:]=Mar_15_NMPDA[:,:,:]
Spring_15_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_15_NMPDA[:,:,:]
Spring_15_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_15_NMPDA[:,:,:]

Spring_18_NMPDA[:,0:Spring_index[0]              ,:]=Mar_18_NMPDA[:,:,:]
Spring_18_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_18_NMPDA[:,:,:]
Spring_18_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_18_NMPDA[:,:,:]

Spring_21_NMPDA[:,0:Spring_index[0]              ,:]=Mar_21_NMPDA[:,:,:]
Spring_21_NMPDA[:,Spring_index[0]:Spring_index[1],:]=Apr_21_NMPDA[:,:,:]
Spring_21_NMPDA[:,Spring_index[1]:Spring_index[2],:]=May_21_NMPDA[:,:,:]

#######
#   Spring NoahMP TSKIN arrays DA LOOP
#######

Spring_00_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_00_NMPDA_Tskin[:,:]
Spring_00_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_00_NMPDA_Tskin[:,:]
Spring_00_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_00_NMPDA_Tskin[:,:]

Spring_03_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_03_NMPDA_Tskin[:,:]
Spring_03_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_03_NMPDA_Tskin[:,:]
Spring_03_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_03_NMPDA_Tskin[:,:]

Spring_06_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_06_NMPDA_Tskin[:,:]
Spring_06_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_06_NMPDA_Tskin[:,:]
Spring_06_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_06_NMPDA_Tskin[:,:]

Spring_09_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_09_NMPDA_Tskin[:,:]
Spring_09_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_09_NMPDA_Tskin[:,:]
Spring_09_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_09_NMPDA_Tskin[:,:]

Spring_12_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_12_NMPDA_Tskin[:,:]
Spring_12_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_12_NMPDA_Tskin[:,:]
Spring_12_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_12_NMPDA_Tskin[:,:]

Spring_15_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_15_NMPDA_Tskin[:,:]
Spring_15_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_15_NMPDA_Tskin[:,:]
Spring_15_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_15_NMPDA_Tskin[:,:]

Spring_18_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_18_NMPDA_Tskin[:,:]
Spring_18_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_18_NMPDA_Tskin[:,:]
Spring_18_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_18_NMPDA_Tskin[:,:]

Spring_21_NMPDA_Tskin[0:Spring_index[0]              ,:]=Mar_21_NMPDA_Tskin[:,:]
Spring_21_NMPDA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_21_NMPDA_Tskin[:,:]
Spring_21_NMPDA_Tskin[Spring_index[1]:Spring_index[2],:]=May_21_NMPDA_Tskin[:,:]

#######
#   Spring Noah36 arrays OPEN LOOP
#######

Spring_00_N36[:,0:Spring_index[0]              ,:]=Mar_00_N36[:,:,:]
Spring_00_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_00_N36[:,:,:]
Spring_00_N36[:,Spring_index[1]:Spring_index[2],:]=May_00_N36[:,:,:]

Spring_03_N36[:,0:Spring_index[0]              ,:]=Mar_03_N36[:,:,:]
Spring_03_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_03_N36[:,:,:]
Spring_03_N36[:,Spring_index[1]:Spring_index[2],:]=May_03_N36[:,:,:]

Spring_06_N36[:,0:Spring_index[0]              ,:]=Mar_06_N36[:,:,:]
Spring_06_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_06_N36[:,:,:]
Spring_06_N36[:,Spring_index[1]:Spring_index[2],:]=May_06_N36[:,:,:]

Spring_09_N36[:,0:Spring_index[0]              ,:]=Mar_09_N36[:,:,:]
Spring_09_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_09_N36[:,:,:]
Spring_09_N36[:,Spring_index[1]:Spring_index[2],:]=May_09_N36[:,:,:]

Spring_12_N36[:,0:Spring_index[0]              ,:]=Mar_12_N36[:,:,:]
Spring_12_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_12_N36[:,:,:]
Spring_12_N36[:,Spring_index[1]:Spring_index[2],:]=May_12_N36[:,:,:]

Spring_15_N36[:,0:Spring_index[0]              ,:]=Mar_15_N36[:,:,:]
Spring_15_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_15_N36[:,:,:]
Spring_15_N36[:,Spring_index[1]:Spring_index[2],:]=May_15_N36[:,:,:]

Spring_18_N36[:,0:Spring_index[0]              ,:]=Mar_18_N36[:,:,:]
Spring_18_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_18_N36[:,:,:]
Spring_18_N36[:,Spring_index[1]:Spring_index[2],:]=May_18_N36[:,:,:]

Spring_21_N36[:,0:Spring_index[0]              ,:]=Mar_21_N36[:,:,:]
Spring_21_N36[:,Spring_index[0]:Spring_index[1],:]=Apr_21_N36[:,:,:]
Spring_21_N36[:,Spring_index[1]:Spring_index[2],:]=May_21_N36[:,:,:]

#######
#   Spring Noah36 TSKIN arrays OPEN LOOP
#######

Spring_00_N36_Tskin[0:Spring_index[0]              ,:]=Mar_00_N36_Tskin[:,:]
Spring_00_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_00_N36_Tskin[:,:]
Spring_00_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_00_N36_Tskin[:,:]

Spring_03_N36_Tskin[0:Spring_index[0]              ,:]=Mar_03_N36_Tskin[:,:]
Spring_03_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_03_N36_Tskin[:,:]
Spring_03_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_03_N36_Tskin[:,:]

Spring_06_N36_Tskin[0:Spring_index[0]              ,:]=Mar_06_N36_Tskin[:,:]
Spring_06_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_06_N36_Tskin[:,:]
Spring_06_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_06_N36_Tskin[:,:]

Spring_09_N36_Tskin[0:Spring_index[0]              ,:]=Mar_09_N36_Tskin[:,:]
Spring_09_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_09_N36_Tskin[:,:]
Spring_09_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_09_N36_Tskin[:,:]

Spring_12_N36_Tskin[0:Spring_index[0]              ,:]=Mar_12_N36_Tskin[:,:]
Spring_12_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_12_N36_Tskin[:,:]
Spring_12_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_12_N36_Tskin[:,:]

Spring_15_N36_Tskin[0:Spring_index[0]              ,:]=Mar_15_N36_Tskin[:,:]
Spring_15_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_15_N36_Tskin[:,:]
Spring_15_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_15_N36_Tskin[:,:]

Spring_18_N36_Tskin[0:Spring_index[0]              ,:]=Mar_18_N36_Tskin[:,:]
Spring_18_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_18_N36_Tskin[:,:]
Spring_18_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_18_N36_Tskin[:,:]

Spring_21_N36_Tskin[0:Spring_index[0]              ,:]=Mar_21_N36_Tskin[:,:]
Spring_21_N36_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_21_N36_Tskin[:,:]
Spring_21_N36_Tskin[Spring_index[1]:Spring_index[2],:]=May_21_N36_Tskin[:,:]

#######
#   Spring Noah36 arrays DA LOOP
#######

Spring_00_N36DA[:,0:Spring_index[0]              ,:]=Mar_00_N36DA[:,:,:]
Spring_00_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_00_N36DA[:,:,:]
Spring_00_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_00_N36DA[:,:,:]

Spring_03_N36DA[:,0:Spring_index[0]              ,:]=Mar_03_N36DA[:,:,:]
Spring_03_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_03_N36DA[:,:,:]
Spring_03_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_03_N36DA[:,:,:]

Spring_06_N36DA[:,0:Spring_index[0]              ,:]=Mar_06_N36DA[:,:,:]
Spring_06_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_06_N36DA[:,:,:]
Spring_06_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_06_N36DA[:,:,:]

Spring_09_N36DA[:,0:Spring_index[0]              ,:]=Mar_09_N36DA[:,:,:]
Spring_09_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_09_N36DA[:,:,:]
Spring_09_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_09_N36DA[:,:,:]

Spring_12_N36DA[:,0:Spring_index[0]              ,:]=Mar_12_N36DA[:,:,:]
Spring_12_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_12_N36DA[:,:,:]
Spring_12_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_12_N36DA[:,:,:]

Spring_15_N36DA[:,0:Spring_index[0]              ,:]=Mar_15_N36DA[:,:,:]
Spring_15_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_15_N36DA[:,:,:]
Spring_15_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_15_N36DA[:,:,:]

Spring_18_N36DA[:,0:Spring_index[0]              ,:]=Mar_18_N36DA[:,:,:]
Spring_18_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_18_N36DA[:,:,:]
Spring_18_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_18_N36DA[:,:,:]

Spring_21_N36DA[:,0:Spring_index[0]              ,:]=Mar_21_N36DA[:,:,:]
Spring_21_N36DA[:,Spring_index[0]:Spring_index[1],:]=Apr_21_N36DA[:,:,:]
Spring_21_N36DA[:,Spring_index[1]:Spring_index[2],:]=May_21_N36DA[:,:,:]

#######
#   Spring Noah36 TSKIN arrays DA LOOP
#######

Spring_00_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_00_N36DA_Tskin[:,:]
Spring_00_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_00_N36DA_Tskin[:,:]
Spring_00_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_00_N36DA_Tskin[:,:]

Spring_03_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_03_N36DA_Tskin[:,:]
Spring_03_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_03_N36DA_Tskin[:,:]
Spring_03_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_03_N36DA_Tskin[:,:]

Spring_06_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_06_N36DA_Tskin[:,:]
Spring_06_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_06_N36DA_Tskin[:,:]
Spring_06_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_06_N36DA_Tskin[:,:]

Spring_09_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_09_N36DA_Tskin[:,:]
Spring_09_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_09_N36DA_Tskin[:,:]
Spring_09_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_09_N36DA_Tskin[:,:]

Spring_12_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_12_N36DA_Tskin[:,:]
Spring_12_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_12_N36DA_Tskin[:,:]
Spring_12_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_12_N36DA_Tskin[:,:]

Spring_15_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_15_N36DA_Tskin[:,:]
Spring_15_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_15_N36DA_Tskin[:,:]
Spring_15_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_15_N36DA_Tskin[:,:]

Spring_18_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_18_N36DA_Tskin[:,:]
Spring_18_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_18_N36DA_Tskin[:,:]
Spring_18_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_18_N36DA_Tskin[:,:]

Spring_21_N36DA_Tskin[0:Spring_index[0]              ,:]=Mar_21_N36DA_Tskin[:,:]
Spring_21_N36DA_Tskin[Spring_index[0]:Spring_index[1],:]=Apr_21_N36DA_Tskin[:,:]
Spring_21_N36DA_Tskin[Spring_index[1]:Spring_index[2],:]=May_21_N36DA_Tskin[:,:]

#######
#   Summer SCAN arrays
#######

Summer_index=np.zeros((3), dtype='int32')
Summer_index[0]=Total_Month_days[5]
Summer_index[1]=np.sum(Total_Month_days[5:7])
Summer_index[2]=np.sum(Total_Month_days[5:8])

Summer_00_SCAN[:,0:Summer_index[0]              ,:]=Jun_00_SCAN[:,:,:]
Summer_00_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_00_SCAN[:,:,:]
Summer_00_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_00_SCAN[:,:,:]

Summer_03_SCAN[:,0:Summer_index[0]              ,:]=Jun_03_SCAN[:,:,:]
Summer_03_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_03_SCAN[:,:,:]
Summer_03_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_03_SCAN[:,:,:]

Summer_06_SCAN[:,0:Summer_index[0]              ,:]=Jun_06_SCAN[:,:,:]
Summer_06_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_06_SCAN[:,:,:]
Summer_06_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_06_SCAN[:,:,:]

Summer_09_SCAN[:,0:Summer_index[0]              ,:]=Jun_09_SCAN[:,:,:]
Summer_09_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_09_SCAN[:,:,:]
Summer_09_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_09_SCAN[:,:,:]

Summer_12_SCAN[:,0:Summer_index[0]              ,:]=Jun_12_SCAN[:,:,:]
Summer_12_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_12_SCAN[:,:,:]
Summer_12_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_12_SCAN[:,:,:]

Summer_15_SCAN[:,0:Summer_index[0]              ,:]=Jun_15_SCAN[:,:,:]
Summer_15_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_15_SCAN[:,:,:]
Summer_15_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_15_SCAN[:,:,:]

Summer_18_SCAN[:,0:Summer_index[0]              ,:]=Jun_18_SCAN[:,:,:]
Summer_18_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_18_SCAN[:,:,:]
Summer_18_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_18_SCAN[:,:,:]

Summer_21_SCAN[:,0:Summer_index[0]              ,:]=Jun_21_SCAN[:,:,:]
Summer_21_SCAN[:,Summer_index[0]:Summer_index[1],:]=Jul_21_SCAN[:,:,:]
Summer_21_SCAN[:,Summer_index[1]:Summer_index[2],:]=Aug_21_SCAN[:,:,:]

######################

Summer_00_ISCCP[0:Summer_index[0]              ,:]=Jun_00_ISCCP[:,:]
Summer_00_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_00_ISCCP[:,:]
Summer_00_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_00_ISCCP[:,:]

Summer_03_ISCCP[0:Summer_index[0]              ,:]=Jun_03_ISCCP[:,:]
Summer_03_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_03_ISCCP[:,:]
Summer_03_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_03_ISCCP[:,:]

Summer_06_ISCCP[0:Summer_index[0]              ,:]=Jun_06_ISCCP[:,:]
Summer_06_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_06_ISCCP[:,:]
Summer_06_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_06_ISCCP[:,:]

Summer_09_ISCCP[0:Summer_index[0]              ,:]=Jun_09_ISCCP[:,:]
Summer_09_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_09_ISCCP[:,:]
Summer_09_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_09_ISCCP[:,:]

Summer_12_ISCCP[0:Summer_index[0]              ,:]=Jun_12_ISCCP[:,:]
Summer_12_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_12_ISCCP[:,:]
Summer_12_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_12_ISCCP[:,:]

Summer_15_ISCCP[0:Summer_index[0]              ,:]=Jun_15_ISCCP[:,:]
Summer_15_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_15_ISCCP[:,:]
Summer_15_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_15_ISCCP[:,:]

Summer_18_ISCCP[0:Summer_index[0]              ,:]=Jun_18_ISCCP[:,:]
Summer_18_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_18_ISCCP[:,:]
Summer_18_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_18_ISCCP[:,:]

Summer_21_ISCCP[0:Summer_index[0]              ,:]=Jun_21_ISCCP[:,:]
Summer_21_ISCCP[Summer_index[0]:Summer_index[1],:]=Jul_21_ISCCP[:,:]
Summer_21_ISCCP[Summer_index[1]:Summer_index[2],:]=Aug_21_ISCCP[:,:]

#######
#   Summer NoaMP arrays OPEN LOOP
#######

Summer_00_NMP[:,0:Summer_index[0]              ,:]=Jun_00_NMP[:,:,:]
Summer_00_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_00_NMP[:,:,:]
Summer_00_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_00_NMP[:,:,:]

Summer_03_NMP[:,0:Summer_index[0]              ,:]=Jun_03_NMP[:,:,:]
Summer_03_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_03_NMP[:,:,:]
Summer_03_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_03_NMP[:,:,:]

Summer_06_NMP[:,0:Summer_index[0]              ,:]=Jun_06_NMP[:,:,:]
Summer_06_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_06_NMP[:,:,:]
Summer_06_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_06_NMP[:,:,:]

Summer_09_NMP[:,0:Summer_index[0]              ,:]=Jun_09_NMP[:,:,:]
Summer_09_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_09_NMP[:,:,:]
Summer_09_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_09_NMP[:,:,:]

Summer_12_NMP[:,0:Summer_index[0]              ,:]=Jun_12_NMP[:,:,:]
Summer_12_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_12_NMP[:,:,:]
Summer_12_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_12_NMP[:,:,:]

Summer_15_NMP[:,0:Summer_index[0]              ,:]=Jun_15_NMP[:,:,:]
Summer_15_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_15_NMP[:,:,:]
Summer_15_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_15_NMP[:,:,:]

Summer_18_NMP[:,0:Summer_index[0]              ,:]=Jun_18_NMP[:,:,:]
Summer_18_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_18_NMP[:,:,:]
Summer_18_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_18_NMP[:,:,:]

Summer_21_NMP[:,0:Summer_index[0]              ,:]=Jun_21_NMP[:,:,:]
Summer_21_NMP[:,Summer_index[0]:Summer_index[1],:]=Jul_21_NMP[:,:,:]
Summer_21_NMP[:,Summer_index[1]:Summer_index[2],:]=Aug_21_NMP[:,:,:]

#######
#   Summer NoaMP TSKIN arrays OPEN LOOP
#######

Summer_00_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_00_NMP_Tskin[:,:]
Summer_00_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_00_NMP_Tskin[:,:]
Summer_00_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_00_NMP_Tskin[:,:]

Summer_03_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_03_NMP_Tskin[:,:]
Summer_03_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_03_NMP_Tskin[:,:]
Summer_03_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_03_NMP_Tskin[:,:]

Summer_06_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_06_NMP_Tskin[:,:]
Summer_06_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_06_NMP_Tskin[:,:]
Summer_06_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_06_NMP_Tskin[:,:]

Summer_09_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_09_NMP_Tskin[:,:]
Summer_09_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_09_NMP_Tskin[:,:]
Summer_09_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_09_NMP_Tskin[:,:]

Summer_12_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_12_NMP_Tskin[:,:]
Summer_12_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_12_NMP_Tskin[:,:]
Summer_12_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_12_NMP_Tskin[:,:]

Summer_15_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_15_NMP_Tskin[:,:]
Summer_15_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_15_NMP_Tskin[:,:]
Summer_15_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_15_NMP_Tskin[:,:]

Summer_18_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_18_NMP_Tskin[:,:]
Summer_18_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_18_NMP_Tskin[:,:]
Summer_18_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_18_NMP_Tskin[:,:]

Summer_21_NMP_Tskin[0:Summer_index[0]              ,:]=Jun_21_NMP_Tskin[:,:]
Summer_21_NMP_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_21_NMP_Tskin[:,:]
Summer_21_NMP_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_21_NMP_Tskin[:,:]

#######
#   Summer NoaMP arrays DA LOOP
#######

Summer_00_NMPDA[:,0:Summer_index[0]              ,:]=Jun_00_NMPDA[:,:,:]
Summer_00_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_00_NMPDA[:,:,:]
Summer_00_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_00_NMPDA[:,:,:]

Summer_03_NMPDA[:,0:Summer_index[0]              ,:]=Jun_03_NMPDA[:,:,:]
Summer_03_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_03_NMPDA[:,:,:]
Summer_03_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_03_NMPDA[:,:,:]

Summer_06_NMPDA[:,0:Summer_index[0]              ,:]=Jun_06_NMPDA[:,:,:]
Summer_06_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_06_NMPDA[:,:,:]
Summer_06_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_06_NMPDA[:,:,:]

Summer_09_NMPDA[:,0:Summer_index[0]              ,:]=Jun_09_NMPDA[:,:,:]
Summer_09_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_09_NMPDA[:,:,:]
Summer_09_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_09_NMPDA[:,:,:]

Summer_12_NMPDA[:,0:Summer_index[0]              ,:]=Jun_12_NMPDA[:,:,:]
Summer_12_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_12_NMPDA[:,:,:]
Summer_12_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_12_NMPDA[:,:,:]

Summer_15_NMPDA[:,0:Summer_index[0]              ,:]=Jun_15_NMPDA[:,:,:]
Summer_15_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_15_NMPDA[:,:,:]
Summer_15_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_15_NMPDA[:,:,:]

Summer_18_NMPDA[:,0:Summer_index[0]              ,:]=Jun_18_NMPDA[:,:,:]
Summer_18_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_18_NMPDA[:,:,:]
Summer_18_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_18_NMPDA[:,:,:]

Summer_21_NMPDA[:,0:Summer_index[0]              ,:]=Jun_21_NMPDA[:,:,:]
Summer_21_NMPDA[:,Summer_index[0]:Summer_index[1],:]=Jul_21_NMPDA[:,:,:]
Summer_21_NMPDA[:,Summer_index[1]:Summer_index[2],:]=Aug_21_NMPDA[:,:,:]

#######
#   Summer NoaMP TSKIN arrays DA LOOP
#######

Summer_00_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_00_NMPDA_Tskin[:,:]
Summer_00_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_00_NMPDA_Tskin[:,:]
Summer_00_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_00_NMPDA_Tskin[:,:]

Summer_03_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_03_NMPDA_Tskin[:,:]
Summer_03_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_03_NMPDA_Tskin[:,:]
Summer_03_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_03_NMPDA_Tskin[:,:]

Summer_06_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_06_NMPDA_Tskin[:,:]
Summer_06_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_06_NMPDA_Tskin[:,:]
Summer_06_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_06_NMPDA_Tskin[:,:]

Summer_09_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_09_NMPDA_Tskin[:,:]
Summer_09_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_09_NMPDA_Tskin[:,:]
Summer_09_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_09_NMPDA_Tskin[:,:]

Summer_12_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_12_NMPDA_Tskin[:,:]
Summer_12_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_12_NMPDA_Tskin[:,:]
Summer_12_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_12_NMPDA_Tskin[:,:]

Summer_15_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_15_NMPDA_Tskin[:,:]
Summer_15_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_15_NMPDA_Tskin[:,:]
Summer_15_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_15_NMPDA_Tskin[:,:]

Summer_18_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_18_NMPDA_Tskin[:,:]
Summer_18_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_18_NMPDA_Tskin[:,:]
Summer_18_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_18_NMPDA_Tskin[:,:]

Summer_21_NMPDA_Tskin[0:Summer_index[0]              ,:]=Jun_21_NMPDA_Tskin[:,:]
Summer_21_NMPDA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_21_NMPDA_Tskin[:,:]
Summer_21_NMPDA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_21_NMPDA_Tskin[:,:]

#######
#   Summer Noa36 arrays OPEN LOOP
#######

Summer_00_N36[:,0:Summer_index[0]              ,:]=Jun_00_N36[:,:,:]
Summer_00_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_00_N36[:,:,:]
Summer_00_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_00_N36[:,:,:]

Summer_03_N36[:,0:Summer_index[0]              ,:]=Jun_03_N36[:,:,:]
Summer_03_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_03_N36[:,:,:]
Summer_03_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_03_N36[:,:,:]

Summer_06_N36[:,0:Summer_index[0]              ,:]=Jun_06_N36[:,:,:]
Summer_06_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_06_N36[:,:,:]
Summer_06_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_06_N36[:,:,:]

Summer_09_N36[:,0:Summer_index[0]              ,:]=Jun_09_N36[:,:,:]
Summer_09_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_09_N36[:,:,:]
Summer_09_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_09_N36[:,:,:]

Summer_12_N36[:,0:Summer_index[0]              ,:]=Jun_12_N36[:,:,:]
Summer_12_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_12_N36[:,:,:]
Summer_12_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_12_N36[:,:,:]

Summer_15_N36[:,0:Summer_index[0]              ,:]=Jun_15_N36[:,:,:]
Summer_15_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_15_N36[:,:,:]
Summer_15_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_15_N36[:,:,:]

Summer_18_N36[:,0:Summer_index[0]              ,:]=Jun_18_N36[:,:,:]
Summer_18_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_18_N36[:,:,:]
Summer_18_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_18_N36[:,:,:]

Summer_21_N36[:,0:Summer_index[0]              ,:]=Jun_21_N36[:,:,:]
Summer_21_N36[:,Summer_index[0]:Summer_index[1],:]=Jul_21_N36[:,:,:]
Summer_21_N36[:,Summer_index[1]:Summer_index[2],:]=Aug_21_N36[:,:,:]

#######
#   Summer Noa36 TSKIN arrays OPEN LOOP
#######

Summer_00_N36_Tskin[0:Summer_index[0]              ,:]=Jun_00_N36_Tskin[:,:]
Summer_00_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_00_N36_Tskin[:,:]
Summer_00_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_00_N36_Tskin[:,:]

Summer_03_N36_Tskin[0:Summer_index[0]              ,:]=Jun_03_N36_Tskin[:,:]
Summer_03_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_03_N36_Tskin[:,:]
Summer_03_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_03_N36_Tskin[:,:]

Summer_06_N36_Tskin[0:Summer_index[0]              ,:]=Jun_06_N36_Tskin[:,:]
Summer_06_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_06_N36_Tskin[:,:]
Summer_06_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_06_N36_Tskin[:,:]

Summer_09_N36_Tskin[0:Summer_index[0]              ,:]=Jun_09_N36_Tskin[:,:]
Summer_09_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_09_N36_Tskin[:,:]
Summer_09_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_09_N36_Tskin[:,:]

Summer_12_N36_Tskin[0:Summer_index[0]              ,:]=Jun_12_N36_Tskin[:,:]
Summer_12_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_12_N36_Tskin[:,:]
Summer_12_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_12_N36_Tskin[:,:]

Summer_15_N36_Tskin[0:Summer_index[0]              ,:]=Jun_15_N36_Tskin[:,:]
Summer_15_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_15_N36_Tskin[:,:]
Summer_15_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_15_N36_Tskin[:,:]

Summer_18_N36_Tskin[0:Summer_index[0]              ,:]=Jun_18_N36_Tskin[:,:]
Summer_18_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_18_N36_Tskin[:,:]
Summer_18_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_18_N36_Tskin[:,:]

Summer_21_N36_Tskin[0:Summer_index[0]              ,:]=Jun_21_N36_Tskin[:,:]
Summer_21_N36_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_21_N36_Tskin[:,:]
Summer_21_N36_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_21_N36_Tskin[:,:]

#######
#   Summer Noa36 arrays DA LOOP
#######

Summer_00_N36DA[:,0:Summer_index[0]              ,:]=Jun_00_N36DA[:,:,:]
Summer_00_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_00_N36DA[:,:,:]
Summer_00_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_00_N36DA[:,:,:]

Summer_03_N36DA[:,0:Summer_index[0]              ,:]=Jun_03_N36DA[:,:,:]
Summer_03_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_03_N36DA[:,:,:]
Summer_03_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_03_N36DA[:,:,:]

Summer_06_N36DA[:,0:Summer_index[0]              ,:]=Jun_06_N36DA[:,:,:]
Summer_06_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_06_N36DA[:,:,:]
Summer_06_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_06_N36DA[:,:,:]

Summer_09_N36DA[:,0:Summer_index[0]              ,:]=Jun_09_N36DA[:,:,:]
Summer_09_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_09_N36DA[:,:,:]
Summer_09_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_09_N36DA[:,:,:]

Summer_12_N36DA[:,0:Summer_index[0]              ,:]=Jun_12_N36DA[:,:,:]
Summer_12_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_12_N36DA[:,:,:]
Summer_12_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_12_N36DA[:,:,:]

Summer_15_N36DA[:,0:Summer_index[0]              ,:]=Jun_15_N36DA[:,:,:]
Summer_15_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_15_N36DA[:,:,:]
Summer_15_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_15_N36DA[:,:,:]

Summer_18_N36DA[:,0:Summer_index[0]              ,:]=Jun_18_N36DA[:,:,:]
Summer_18_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_18_N36DA[:,:,:]
Summer_18_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_18_N36DA[:,:,:]

Summer_21_N36DA[:,0:Summer_index[0]              ,:]=Jun_21_N36DA[:,:,:]
Summer_21_N36DA[:,Summer_index[0]:Summer_index[1],:]=Jul_21_N36DA[:,:,:]
Summer_21_N36DA[:,Summer_index[1]:Summer_index[2],:]=Aug_21_N36DA[:,:,:]

#######
#   Summer Noa36 TSKIN arrays DA LOOP
#######

Summer_00_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_00_N36DA_Tskin[:,:]
Summer_00_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_00_N36DA_Tskin[:,:]
Summer_00_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_00_N36DA_Tskin[:,:]

Summer_03_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_03_N36DA_Tskin[:,:]
Summer_03_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_03_N36DA_Tskin[:,:]
Summer_03_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_03_N36DA_Tskin[:,:]

Summer_06_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_06_N36DA_Tskin[:,:]
Summer_06_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_06_N36DA_Tskin[:,:]
Summer_06_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_06_N36DA_Tskin[:,:]

Summer_09_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_09_N36DA_Tskin[:,:]
Summer_09_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_09_N36DA_Tskin[:,:]
Summer_09_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_09_N36DA_Tskin[:,:]

Summer_12_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_12_N36DA_Tskin[:,:]
Summer_12_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_12_N36DA_Tskin[:,:]
Summer_12_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_12_N36DA_Tskin[:,:]

Summer_15_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_15_N36DA_Tskin[:,:]
Summer_15_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_15_N36DA_Tskin[:,:]
Summer_15_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_15_N36DA_Tskin[:,:]

Summer_18_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_18_N36DA_Tskin[:,:]
Summer_18_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_18_N36DA_Tskin[:,:]
Summer_18_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_18_N36DA_Tskin[:,:]

Summer_21_N36DA_Tskin[0:Summer_index[0]              ,:]=Jun_21_N36DA_Tskin[:,:]
Summer_21_N36DA_Tskin[Summer_index[0]:Summer_index[1],:]=Jul_21_N36DA_Tskin[:,:]
Summer_21_N36DA_Tskin[Summer_index[1]:Summer_index[2],:]=Aug_21_N36DA_Tskin[:,:]

#######
#   Fall SCAN arrays
#######

Fall_index=np.zeros((3), dtype='int32')
Fall_index[0]=Total_Month_days[8]
Fall_index[1]=np.sum(Total_Month_days[8:10])
Fall_index[2]=np.sum(Total_Month_days[8:11])

Fall_00_SCAN[:,0:Fall_index[0]            ,:]=Sep_00_SCAN[:,:,:]
Fall_00_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_00_SCAN[:,:,:]
Fall_00_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_00_SCAN[:,:,:]

Fall_03_SCAN[:,0:Fall_index[0]            ,:]=Sep_03_SCAN[:,:,:]
Fall_03_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_03_SCAN[:,:,:]
Fall_03_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_03_SCAN[:,:,:]

Fall_06_SCAN[:,0:Fall_index[0]            ,:]=Sep_06_SCAN[:,:,:]
Fall_06_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_06_SCAN[:,:,:]
Fall_06_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_06_SCAN[:,:,:]

Fall_09_SCAN[:,0:Fall_index[0]            ,:]=Sep_09_SCAN[:,:,:]
Fall_09_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_09_SCAN[:,:,:]
Fall_09_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_09_SCAN[:,:,:]

Fall_12_SCAN[:,0:Fall_index[0]            ,:]=Sep_12_SCAN[:,:,:]
Fall_12_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_12_SCAN[:,:,:]
Fall_12_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_12_SCAN[:,:,:]

Fall_15_SCAN[:,0:Fall_index[0]            ,:]=Sep_15_SCAN[:,:,:]
Fall_15_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_15_SCAN[:,:,:]
Fall_15_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_15_SCAN[:,:,:]

Fall_18_SCAN[:,0:Fall_index[0]            ,:]=Sep_18_SCAN[:,:,:]
Fall_18_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_18_SCAN[:,:,:]
Fall_18_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_18_SCAN[:,:,:]

Fall_21_SCAN[:,0:Fall_index[0]            ,:]=Sep_21_SCAN[:,:,:]
Fall_21_SCAN[:,Fall_index[0]:Fall_index[1],:]=Oct_21_SCAN[:,:,:]
Fall_21_SCAN[:,Fall_index[1]:Fall_index[2],:]=Nov_21_SCAN[:,:,:]

##############
##############

Fall_00_ISCCP[0:Fall_index[0]            ,:]=Sep_00_ISCCP[:,:]
Fall_00_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_00_ISCCP[:,:]
Fall_00_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_00_ISCCP[:,:]

Fall_03_ISCCP[0:Fall_index[0]            ,:]=Sep_03_ISCCP[:,:]
Fall_03_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_03_ISCCP[:,:]
Fall_03_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_03_ISCCP[:,:]

Fall_06_ISCCP[0:Fall_index[0]            ,:]=Sep_06_ISCCP[:,:]
Fall_06_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_06_ISCCP[:,:]
Fall_06_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_06_ISCCP[:,:]

Fall_09_ISCCP[0:Fall_index[0]            ,:]=Sep_09_ISCCP[:,:]
Fall_09_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_09_ISCCP[:,:]
Fall_09_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_09_ISCCP[:,:]

Fall_12_ISCCP[0:Fall_index[0]            ,:]=Sep_12_ISCCP[:,:]
Fall_12_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_12_ISCCP[:,:]
Fall_12_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_12_ISCCP[:,:]

Fall_15_ISCCP[0:Fall_index[0]            ,:]=Sep_15_ISCCP[:,:]
Fall_15_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_15_ISCCP[:,:]
Fall_15_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_15_ISCCP[:,:]

Fall_18_ISCCP[0:Fall_index[0]            ,:]=Sep_18_ISCCP[:,:]
Fall_18_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_18_ISCCP[:,:]
Fall_18_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_18_ISCCP[:,:]

Fall_21_ISCCP[0:Fall_index[0]            ,:]=Sep_21_ISCCP[:,:]
Fall_21_ISCCP[Fall_index[0]:Fall_index[1],:]=Oct_21_ISCCP[:,:]
Fall_21_ISCCP[Fall_index[1]:Fall_index[2],:]=Nov_21_ISCCP[:,:]

#######
#   Fall Noah MP arrays OPEN LOOP
#######

Fall_00_NMP[:,0:Fall_index[0]            ,:]=Sep_00_NMP[:,:,:]
Fall_00_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_00_NMP[:,:,:]
Fall_00_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_00_NMP[:,:,:]

Fall_03_NMP[:,0:Fall_index[0]            ,:]=Sep_03_NMP[:,:,:]
Fall_03_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_03_NMP[:,:,:]
Fall_03_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_03_NMP[:,:,:]

Fall_06_NMP[:,0:Fall_index[0]            ,:]=Sep_06_NMP[:,:,:]
Fall_06_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_06_NMP[:,:,:]
Fall_06_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_06_NMP[:,:,:]

Fall_09_NMP[:,0:Fall_index[0]            ,:]=Sep_09_NMP[:,:,:]
Fall_09_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_09_NMP[:,:,:]
Fall_09_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_09_NMP[:,:,:]

Fall_12_NMP[:,0:Fall_index[0]            ,:]=Sep_12_NMP[:,:,:]
Fall_12_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_12_NMP[:,:,:]
Fall_12_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_12_NMP[:,:,:]

Fall_15_NMP[:,0:Fall_index[0]            ,:]=Sep_15_NMP[:,:,:]
Fall_15_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_15_NMP[:,:,:]
Fall_15_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_15_NMP[:,:,:]

Fall_18_NMP[:,0:Fall_index[0]            ,:]=Sep_18_NMP[:,:,:]
Fall_18_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_18_NMP[:,:,:]
Fall_18_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_18_NMP[:,:,:]

Fall_21_NMP[:,0:Fall_index[0]            ,:]=Sep_21_NMP[:,:,:]
Fall_21_NMP[:,Fall_index[0]:Fall_index[1],:]=Oct_21_NMP[:,:,:]
Fall_21_NMP[:,Fall_index[1]:Fall_index[2],:]=Nov_21_NMP[:,:,:]

#######
#   Fall Noah MP TSKIN arrays OPEN LOOP
#######

Fall_00_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_00_NMP_Tskin[:,:]
Fall_00_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_00_NMP_Tskin[:,:]
Fall_00_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_00_NMP_Tskin[:,:]

Fall_03_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_03_NMP_Tskin[:,:]
Fall_03_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_03_NMP_Tskin[:,:]
Fall_03_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_03_NMP_Tskin[:,:]

Fall_06_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_06_NMP_Tskin[:,:]
Fall_06_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_06_NMP_Tskin[:,:]
Fall_06_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_06_NMP_Tskin[:,:]

Fall_09_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_09_NMP_Tskin[:,:]
Fall_09_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_09_NMP_Tskin[:,:]
Fall_09_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_09_NMP_Tskin[:,:]

Fall_12_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_12_NMP_Tskin[:,:]
Fall_12_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_12_NMP_Tskin[:,:]
Fall_12_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_12_NMP_Tskin[:,:]

Fall_15_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_15_NMP_Tskin[:,:]
Fall_15_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_15_NMP_Tskin[:,:]
Fall_15_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_15_NMP_Tskin[:,:]

Fall_18_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_18_NMP_Tskin[:,:]
Fall_18_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_18_NMP_Tskin[:,:]
Fall_18_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_18_NMP_Tskin[:,:]

Fall_21_NMP_Tskin[0:Fall_index[0]            ,:]=Sep_21_NMP_Tskin[:,:]
Fall_21_NMP_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_21_NMP_Tskin[:,:]
Fall_21_NMP_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_21_NMP_Tskin[:,:]

#######
#   Fall Noah MP arrays DA LOOP
#######

Fall_00_NMPDA[:,0:Fall_index[0]            ,:]=Sep_00_NMPDA[:,:,:]
Fall_00_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_00_NMPDA[:,:,:]
Fall_00_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_00_NMPDA[:,:,:]

Fall_03_NMPDA[:,0:Fall_index[0]            ,:]=Sep_03_NMPDA[:,:,:]
Fall_03_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_03_NMPDA[:,:,:]
Fall_03_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_03_NMPDA[:,:,:]

Fall_06_NMPDA[:,0:Fall_index[0]            ,:]=Sep_06_NMPDA[:,:,:]
Fall_06_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_06_NMPDA[:,:,:]
Fall_06_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_06_NMPDA[:,:,:]

Fall_09_NMPDA[:,0:Fall_index[0]            ,:]=Sep_09_NMPDA[:,:,:]
Fall_09_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_09_NMPDA[:,:,:]
Fall_09_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_09_NMPDA[:,:,:]

Fall_12_NMPDA[:,0:Fall_index[0]            ,:]=Sep_12_NMPDA[:,:,:]
Fall_12_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_12_NMPDA[:,:,:]
Fall_12_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_12_NMPDA[:,:,:]

Fall_15_NMPDA[:,0:Fall_index[0]            ,:]=Sep_15_NMPDA[:,:,:]
Fall_15_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_15_NMPDA[:,:,:]
Fall_15_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_15_NMPDA[:,:,:]

Fall_18_NMPDA[:,0:Fall_index[0]            ,:]=Sep_18_NMPDA[:,:,:]
Fall_18_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_18_NMPDA[:,:,:]
Fall_18_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_18_NMPDA[:,:,:]

Fall_21_NMPDA[:,0:Fall_index[0]            ,:]=Sep_21_NMPDA[:,:,:]
Fall_21_NMPDA[:,Fall_index[0]:Fall_index[1],:]=Oct_21_NMPDA[:,:,:]
Fall_21_NMPDA[:,Fall_index[1]:Fall_index[2],:]=Nov_21_NMPDA[:,:,:]

#######
#   Fall Noah MP TSKIN arrays DA LOOP
#######

Fall_00_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_00_NMPDA_Tskin[:,:]
Fall_00_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_00_NMPDA_Tskin[:,:]
Fall_00_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_00_NMPDA_Tskin[:,:]

Fall_03_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_03_NMPDA_Tskin[:,:]
Fall_03_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_03_NMPDA_Tskin[:,:]
Fall_03_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_03_NMPDA_Tskin[:,:]

Fall_06_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_06_NMPDA_Tskin[:,:]
Fall_06_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_06_NMPDA_Tskin[:,:]
Fall_06_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_06_NMPDA_Tskin[:,:]

Fall_09_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_09_NMPDA_Tskin[:,:]
Fall_09_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_09_NMPDA_Tskin[:,:]
Fall_09_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_09_NMPDA_Tskin[:,:]

Fall_12_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_12_NMPDA_Tskin[:,:]
Fall_12_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_12_NMPDA_Tskin[:,:]
Fall_12_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_12_NMPDA_Tskin[:,:]

Fall_15_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_15_NMPDA_Tskin[:,:]
Fall_15_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_15_NMPDA_Tskin[:,:]
Fall_15_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_15_NMPDA_Tskin[:,:]

Fall_18_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_18_NMPDA_Tskin[:,:]
Fall_18_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_18_NMPDA_Tskin[:,:]
Fall_18_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_18_NMPDA_Tskin[:,:]

Fall_21_NMPDA_Tskin[0:Fall_index[0]            ,:]=Sep_21_NMPDA_Tskin[:,:]
Fall_21_NMPDA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_21_NMPDA_Tskin[:,:]
Fall_21_NMPDA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_21_NMPDA_Tskin[:,:]

#######
#   Fall Noah 36 arrays OPEN LOOP
#######

Fall_00_N36[:,0:Fall_index[0]            ,:]=Sep_00_N36[:,:,:]
Fall_00_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_00_N36[:,:,:]
Fall_00_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_00_N36[:,:,:]

Fall_03_N36[:,0:Fall_index[0]            ,:]=Sep_03_N36[:,:,:]
Fall_03_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_03_N36[:,:,:]
Fall_03_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_03_N36[:,:,:]

Fall_06_N36[:,0:Fall_index[0]            ,:]=Sep_06_N36[:,:,:]
Fall_06_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_06_N36[:,:,:]
Fall_06_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_06_N36[:,:,:]

Fall_09_N36[:,0:Fall_index[0]            ,:]=Sep_09_N36[:,:,:]
Fall_09_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_09_N36[:,:,:]
Fall_09_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_09_N36[:,:,:]

Fall_12_N36[:,0:Fall_index[0]            ,:]=Sep_12_N36[:,:,:]
Fall_12_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_12_N36[:,:,:]
Fall_12_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_12_N36[:,:,:]

Fall_15_N36[:,0:Fall_index[0]            ,:]=Sep_15_N36[:,:,:]
Fall_15_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_15_N36[:,:,:]
Fall_15_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_15_N36[:,:,:]

Fall_18_N36[:,0:Fall_index[0]            ,:]=Sep_18_N36[:,:,:]
Fall_18_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_18_N36[:,:,:]
Fall_18_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_18_N36[:,:,:]

Fall_21_N36[:,0:Fall_index[0]            ,:]=Sep_21_N36[:,:,:]
Fall_21_N36[:,Fall_index[0]:Fall_index[1],:]=Oct_21_N36[:,:,:]
Fall_21_N36[:,Fall_index[1]:Fall_index[2],:]=Nov_21_N36[:,:,:]

#######
#   Fall Noah 36 arrays OPEN LOOP
#######

Fall_00_N36_Tskin[0:Fall_index[0]            ,:]=Sep_00_N36_Tskin[:,:]
Fall_00_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_00_N36_Tskin[:,:]
Fall_00_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_00_N36_Tskin[:,:]

Fall_03_N36_Tskin[0:Fall_index[0]            ,:]=Sep_03_N36_Tskin[:,:]
Fall_03_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_03_N36_Tskin[:,:]
Fall_03_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_03_N36_Tskin[:,:]

Fall_06_N36_Tskin[0:Fall_index[0]            ,:]=Sep_06_N36_Tskin[:,:]
Fall_06_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_06_N36_Tskin[:,:]
Fall_06_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_06_N36_Tskin[:,:]

Fall_09_N36_Tskin[0:Fall_index[0]            ,:]=Sep_09_N36_Tskin[:,:]
Fall_09_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_09_N36_Tskin[:,:]
Fall_09_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_09_N36_Tskin[:,:]

Fall_12_N36_Tskin[0:Fall_index[0]            ,:]=Sep_12_N36_Tskin[:,:]
Fall_12_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_12_N36_Tskin[:,:]
Fall_12_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_12_N36_Tskin[:,:]

Fall_15_N36_Tskin[0:Fall_index[0]            ,:]=Sep_15_N36_Tskin[:,:]
Fall_15_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_15_N36_Tskin[:,:]
Fall_15_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_15_N36_Tskin[:,:]

Fall_18_N36_Tskin[0:Fall_index[0]            ,:]=Sep_18_N36_Tskin[:,:]
Fall_18_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_18_N36_Tskin[:,:]
Fall_18_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_18_N36_Tskin[:,:]

Fall_21_N36_Tskin[0:Fall_index[0]            ,:]=Sep_21_N36_Tskin[:,:]
Fall_21_N36_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_21_N36_Tskin[:,:]
Fall_21_N36_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_21_N36_Tskin[:,:]

#######
#   Fall Noah 36 arrays DA LOOP
#######

Fall_00_N36DA[:,0:Fall_index[0]            ,:]=Sep_00_N36DA[:,:,:]
Fall_00_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_00_N36DA[:,:,:]
Fall_00_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_00_N36DA[:,:,:]

Fall_03_N36DA[:,0:Fall_index[0]            ,:]=Sep_03_N36DA[:,:,:]
Fall_03_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_03_N36DA[:,:,:]
Fall_03_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_03_N36DA[:,:,:]

Fall_06_N36DA[:,0:Fall_index[0]            ,:]=Sep_06_N36DA[:,:,:]
Fall_06_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_06_N36DA[:,:,:]
Fall_06_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_06_N36DA[:,:,:]

Fall_09_N36DA[:,0:Fall_index[0]            ,:]=Sep_09_N36DA[:,:,:]
Fall_09_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_09_N36DA[:,:,:]
Fall_09_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_09_N36DA[:,:,:]

Fall_12_N36DA[:,0:Fall_index[0]            ,:]=Sep_12_N36DA[:,:,:]
Fall_12_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_12_N36DA[:,:,:]
Fall_12_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_12_N36DA[:,:,:]

Fall_15_N36DA[:,0:Fall_index[0]            ,:]=Sep_15_N36DA[:,:,:]
Fall_15_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_15_N36DA[:,:,:]
Fall_15_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_15_N36DA[:,:,:]

Fall_18_N36DA[:,0:Fall_index[0]            ,:]=Sep_18_N36DA[:,:,:]
Fall_18_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_18_N36DA[:,:,:]
Fall_18_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_18_N36DA[:,:,:]

Fall_21_N36DA[:,0:Fall_index[0]            ,:]=Sep_21_N36DA[:,:,:]
Fall_21_N36DA[:,Fall_index[0]:Fall_index[1],:]=Oct_21_N36DA[:,:,:]
Fall_21_N36DA[:,Fall_index[1]:Fall_index[2],:]=Nov_21_N36DA[:,:,:]

#######
#   Fall Noah 36 arrays DA LOOP
#######

Fall_00_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_00_N36DA_Tskin[:,:]
Fall_00_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_00_N36DA_Tskin[:,:]
Fall_00_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_00_N36DA_Tskin[:,:]

Fall_03_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_03_N36DA_Tskin[:,:]
Fall_03_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_03_N36DA_Tskin[:,:]
Fall_03_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_03_N36DA_Tskin[:,:]

Fall_06_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_06_N36DA_Tskin[:,:]
Fall_06_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_06_N36DA_Tskin[:,:]
Fall_06_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_06_N36DA_Tskin[:,:]

Fall_09_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_09_N36DA_Tskin[:,:]
Fall_09_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_09_N36DA_Tskin[:,:]
Fall_09_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_09_N36DA_Tskin[:,:]

Fall_12_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_12_N36DA_Tskin[:,:]
Fall_12_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_12_N36DA_Tskin[:,:]
Fall_12_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_12_N36DA_Tskin[:,:]

Fall_15_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_15_N36DA_Tskin[:,:]
Fall_15_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_15_N36DA_Tskin[:,:]
Fall_15_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_15_N36DA_Tskin[:,:]

Fall_18_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_18_N36DA_Tskin[:,:]
Fall_18_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_18_N36DA_Tskin[:,:]
Fall_18_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_18_N36DA_Tskin[:,:]

Fall_21_N36DA_Tskin[0:Fall_index[0]            ,:]=Sep_21_N36DA_Tskin[:,:]
Fall_21_N36DA_Tskin[Fall_index[0]:Fall_index[1],:]=Oct_21_N36DA_Tskin[:,:]
Fall_21_N36DA_Tskin[Fall_index[1]:Fall_index[2],:]=Nov_21_N36DA_Tskin[:,:]


if (N36):
    ptitle='ISMN Gauge Versus LIS Noah v3.6 January Data'
    fname_out=OUT_PATH+'/ISMNvN36_JAN_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Jan_00_SCAN[0,:,:], Jan_03_SCAN[0,:,:], Jan_06_SCAN[0,:,:], Jan_09_SCAN[0,:,:], Jan_12_SCAN[0,:,:], Jan_15_SCAN[0,:,:],   Jan_18_SCAN[0,:,:], Jan_21_SCAN[0,:,:], Jan_00_N36[0,:,:], Jan_03_N36[0,:,:], Jan_06_N36[0,:,:], Jan_09_N36[0,:,:], Jan_12_N36[0,:,:], Jan_15_N36[0,:,:], Jan_18_N36[0,:,:],    Jan_21_N36[0,:,:], Jan_00_N36DA[0,:,:], Jan_03_N36DA[0,:,:], Jan_06_N36DA[0,:,:], Jan_09_N36DA[0,:,:], Jan_12_N36DA[0,:,:],
        Jan_15_N36DA[0,:,:], Jan_18_N36DA[0,:,:], Jan_21_N36DA[0,:,:], ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 February Data'
    fname_out=OUT_PATH+'/ISMNvN36_FEB_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Feb_00_SCAN[0,:,:], Feb_03_SCAN[0,:,:], Feb_06_SCAN[0,:,:], Feb_09_SCAN[0,:,:], Feb_12_SCAN[0,:,:], Feb_15_SCAN[0,:,:],   Feb_18_SCAN[0,:,:], Feb_21_SCAN[0,:,:], Feb_00_N36[0,:,:], Feb_03_N36[0,:,:], Feb_06_N36[0,:,:], Feb_09_N36[0,:,:], Feb_12_N36[0,:,:], Feb_15_N36[0,:,:], Feb_18_N36[0,:,:],    Feb_21_N36[0,:,:], Feb_00_N36DA[0,:,:], Feb_03_N36DA[0,:,:], Feb_06_N36DA[0,:,:], Feb_09_N36DA[0,:,:], Feb_12_N36DA[0,:,:], Feb_15_N36DA[0,:,:], Feb_18_N36DA[0,:,:], Feb_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 March Data'
    fname_out=OUT_PATH+'/ISMNvN36_MAR_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Mar_00_SCAN[0,:,:], Mar_03_SCAN[0,:,:], Mar_06_SCAN[0,:,:], Mar_09_SCAN[0,:,:], Mar_12_SCAN[0,:,:], Mar_15_SCAN[0,:,:],   Mar_18_SCAN[0,:,:], Mar_21_SCAN[0,:,:], Mar_00_N36[0,:,:], Mar_03_N36[0,:,:], Mar_06_N36[0,:,:], Mar_09_N36[0,:,:], Mar_12_N36[0,:,:], Mar_15_N36[0,:,:], Mar_18_N36[0,:,:],    Mar_21_N36[0,:,:], Mar_00_N36DA[0,:,:], Mar_03_N36DA[0,:,:], Mar_06_N36DA[0,:,:], Mar_09_N36DA[0,:,:], Mar_12_N36DA[0,:,:], Mar_15_N36DA[0,:,:], Mar_18_N36DA[0,:,:], Mar_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 April Data'
    fname_out=OUT_PATH+'/ISMNvN36_APR_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Apr_00_SCAN[0,:,:], Apr_03_SCAN[0,:,:], Apr_06_SCAN[0,:,:], Apr_09_SCAN[0,:,:], Apr_12_SCAN[0,:,:], Apr_15_SCAN[0,:,:],   Apr_18_SCAN[0,:,:], Apr_21_SCAN[0,:,:], Apr_00_N36[0,:,:], Apr_03_N36[0,:,:], Apr_06_N36[0,:,:], Apr_09_N36[0,:,:], Apr_12_N36[0,:,:], Apr_15_N36[0,:,:], Apr_18_N36[0,:,:],    Apr_21_N36[0,:,:], Apr_00_N36DA[0,:,:], Apr_03_N36DA[0,:,:], Apr_06_N36DA[0,:,:], Apr_09_N36DA[0,:,:], Apr_12_N36DA[0,:,:], Apr_15_N36DA[0,:,:], Apr_18_N36DA[0,:,:], Apr_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 May Data'
    fname_out=OUT_PATH+'/ISMNvN36_May_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(May_00_SCAN[0,:,:], May_03_SCAN[0,:,:], May_06_SCAN[0,:,:], May_09_SCAN[0,:,:], May_12_SCAN[0,:,:], May_15_SCAN[0,:,:],   May_18_SCAN[0,:,:], May_21_SCAN[0,:,:], May_00_N36[0,:,:], May_03_N36[0,:,:], May_06_N36[0,:,:], May_09_N36[0,:,:], May_12_N36[0,:,:], May_15_N36[0,:,:], May_18_N36[0,:,:],    May_21_N36[0,:,:], May_00_N36DA[0,:,:], May_03_N36DA[0,:,:], May_06_N36DA[0,:,:], May_09_N36DA[0,:,:], May_12_N36DA[0,:,:], May_15_N36DA[0,:,:], May_18_N36DA[0,:,:],   May_21_N36DA[0,:,:],ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 June Data'
    fname_out=OUT_PATH+'/ISMNvN36_JUN_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Jun_00_SCAN[0,:,:], Jun_03_SCAN[0,:,:], Jun_06_SCAN[0,:,:], Jun_09_SCAN[0,:,:], Jun_12_SCAN[0,:,:], Jun_15_SCAN[0,:,:],   Jun_18_SCAN[0,:,:], Jun_21_SCAN[0,:,:], Jun_00_N36[0,:,:], Jun_03_N36[0,:,:], Jun_06_N36[0,:,:], Jun_09_N36[0,:,:], Jun_12_N36[0,:,:], Jun_15_N36[0,:,:], Jun_18_N36[0,:,:],    Jun_21_N36[0,:,:], Jun_00_N36DA[0,:,:], Jun_03_N36DA[0,:,:], Jun_06_N36DA[0,:,:], Jun_09_N36DA[0,:,:], Jun_12_N36DA[0,:,:], Jun_15_N36DA[0,:,:], Jun_18_N36DA[0,:,:], Jun_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 July Data'
    fname_out=OUT_PATH+'/ISMNvN36_JUL_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Jul_00_SCAN[0,:,:], Jul_03_SCAN[0,:,:], Jul_06_SCAN[0,:,:], Jul_09_SCAN[0,:,:], Jul_12_SCAN[0,:,:], Jul_15_SCAN[0,:,:],   Jul_18_SCAN[0,:,:], Jul_21_SCAN[0,:,:], Jul_00_N36[0,:,:], Jul_03_N36[0,:,:], Jul_06_N36[0,:,:], Jul_09_N36[0,:,:], Jul_12_N36[0,:,:], Jul_15_N36[0,:,:], Jul_18_N36[0,:,:],    Jul_21_N36[0,:,:], Jul_00_N36DA[0,:,:], Jul_03_N36DA[0,:,:], Jul_06_N36DA[0,:,:], Jul_09_N36DA[0,:,:], Jul_12_N36DA[0,:,:], Jul_15_N36DA[0,:,:], Jul_18_N36DA[0,:,:], Jul_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 August Data'
    fname_out=OUT_PATH+'/ISMNvN36_AUG_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Aug_00_SCAN[0,:,:], Aug_03_SCAN[0,:,:], Aug_06_SCAN[0,:,:], Aug_09_SCAN[0,:,:], Aug_12_SCAN[0,:,:], Aug_15_SCAN[0,:,:],   Aug_18_SCAN[0,:,:], Aug_21_SCAN[0,:,:], Aug_00_N36[0,:,:], Aug_03_N36[0,:,:], Aug_06_N36[0,:,:], Aug_09_N36[0,:,:], Aug_12_N36[0,:,:], Aug_15_N36[0,:,:], Aug_18_N36[0,:,:],    Aug_21_N36[0,:,:], Aug_00_N36DA[0,:,:], Aug_03_N36DA[0,:,:], Aug_06_N36DA[0,:,:], Aug_09_N36DA[0,:,:], Aug_12_N36DA[0,:,:], Aug_15_N36DA[0,:,:], Aug_18_N36DA[0,:,:], Aug_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 September Data'
    fname_out=OUT_PATH+'/ISMNvN36_SEP_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Sep_00_SCAN[0,:,:], Sep_03_SCAN[0,:,:], Sep_06_SCAN[0,:,:], Sep_09_SCAN[0,:,:], Sep_12_SCAN[0,:,:], Sep_15_SCAN[0,:,:],   Sep_18_SCAN[0,:,:], Sep_21_SCAN[0,:,:], Sep_00_N36[0,:,:], Sep_03_N36[0,:,:], Sep_06_N36[0,:,:], Sep_09_N36[0,:,:], Sep_12_N36[0,:,:], Sep_15_N36[0,:,:], Sep_18_N36[0,:,:],    Sep_21_N36[0,:,:], Sep_00_N36DA[0,:,:], Sep_03_N36DA[0,:,:], Sep_06_N36DA[0,:,:], Sep_09_N36DA[0,:,:], Sep_12_N36DA[0,:,:], Sep_15_N36DA[0,:,:], Sep_18_N36DA[0,:,:], Sep_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 October Data'
    fname_out=OUT_PATH+'/ISMNvN36_OCT_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Oct_00_SCAN[0,:,:], Oct_03_SCAN[0,:,:], Oct_06_SCAN[0,:,:], Oct_09_SCAN[0,:,:], Oct_12_SCAN[0,:,:], Oct_15_SCAN[0,:,:],   Oct_18_SCAN[0,:,:], Oct_21_SCAN[0,:,:], Oct_00_N36[0,:,:], Oct_03_N36[0,:,:], Oct_06_N36[0,:,:], Oct_09_N36[0,:,:], Oct_12_N36[0,:,:], Oct_15_N36[0,:,:], Oct_18_N36[0,:,:],    Oct_21_N36[0,:,:], Oct_00_N36DA[0,:,:], Oct_03_N36DA[0,:,:], Oct_06_N36DA[0,:,:], Oct_09_N36DA[0,:,:], Oct_12_N36DA[0,:,:], Oct_15_N36DA[0,:,:], Oct_18_N36DA[0,:,:], Oct_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 November Data'
    fname_out=OUT_PATH+'/ISMNvN36_NOV_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Nov_00_SCAN[0,:,:], Nov_03_SCAN[0,:,:], Nov_06_SCAN[0,:,:], Nov_09_SCAN[0,:,:], Nov_12_SCAN[0,:,:], Nov_15_SCAN[0,:,:],   Nov_18_SCAN[0,:,:], Nov_21_SCAN[0,:,:], Nov_00_N36[0,:,:], Nov_03_N36[0,:,:], Nov_06_N36[0,:,:], Nov_09_N36[0,:,:], Nov_12_N36[0,:,:], Nov_15_N36[0,:,:], Nov_18_N36[0,:,:],    Nov_21_N36[0,:,:], Nov_00_N36DA[0,:,:], Nov_03_N36DA[0,:,:], Nov_06_N36DA[0,:,:], Nov_09_N36DA[0,:,:], Nov_12_N36DA[0,:,:], Nov_15_N36DA[0,:,:], Nov_18_N36DA[0,:,:], Nov_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah v3.6 December Data'
    fname_out=OUT_PATH+'/ISMNvN36_DEC_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Dec_00_SCAN[0,:,:], Dec_03_SCAN[0,:,:], Dec_06_SCAN[0,:,:], Dec_09_SCAN[0,:,:], Dec_12_SCAN[0,:,:], Dec_15_SCAN[0,:,:],   Dec_18_SCAN[0,:,:], Dec_21_SCAN[0,:,:], Dec_00_N36[0,:,:], Dec_03_N36[0,:,:], Dec_06_N36[0,:,:], Dec_09_N36[0,:,:], Dec_12_N36[0,:,:], Dec_15_N36[0,:,:], Dec_18_N36[0,:,:],    Dec_21_N36[0,:,:], Dec_00_N36DA[0,:,:], Dec_03_N36DA[0,:,:], Dec_06_N36DA[0,:,:], Dec_09_N36DA[0,:,:], Dec_12_N36DA[0,:,:], Dec_15_N36DA[0,:,:], Dec_18_N36DA[0,:,:], Dec_21_N36DA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah V3.6 Temp (K)')

####################################################################################################
#
####################################################################################################

if (NMP):
    ptitle='ISMN Gauge Versus LIS Noah MP January Data'
    fname_out=OUT_PATH+'/ISMNvNMP_JAN_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Jan_00_SCAN[0,:,:], Jan_03_SCAN[0,:,:], Jan_06_SCAN[0,:,:], Jan_09_SCAN[0,:,:], Jan_12_SCAN[0,:,:], Jan_15_SCAN[0,:,:],   Jan_18_SCAN[0,:,:], Jan_21_SCAN[0,:,:], Jan_00_NMP[0,:,:], Jan_03_NMP[0,:,:], Jan_06_NMP[0,:,:], Jan_09_NMP[0,:,:], Jan_12_NMP[0,:,:], Jan_15_NMP[0,:,:], Jan_18_NMP[0,:,:],    Jan_21_NMP[0,:,:], Jan_00_NMPDA[0,:,:], Jan_03_NMPDA[0,:,:], Jan_06_NMPDA[0,:,:], Jan_09_NMPDA[0,:,:], Jan_12_NMPDA[0,:,:], Jan_15_NMPDA[0,:,:], Jan_18_NMPDA[0,:,:], Jan_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP February Data'
    fname_out=OUT_PATH+'/ISMNvNMP_FEB_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Feb_00_SCAN[0,:,:], Feb_03_SCAN[0,:,:], Feb_06_SCAN[0,:,:], Feb_09_SCAN[0,:,:], Feb_12_SCAN[0,:,:], Feb_15_SCAN[0,:,:],   Feb_18_SCAN[0,:,:], Feb_21_SCAN[0,:,:], Feb_00_NMP[0,:,:], Feb_03_NMP[0,:,:], Feb_06_NMP[0,:,:], Feb_09_NMP[0,:,:], Feb_12_NMP[0,:,:], Feb_15_NMP[0,:,:], Feb_18_NMP[0,:,:],    Feb_21_NMP[0,:,:], Feb_00_NMPDA[0,:,:], Feb_03_NMPDA[0,:,:], Feb_06_NMPDA[0,:,:], Feb_09_NMPDA[0,:,:], Feb_12_NMPDA[0,:,:], Feb_15_NMPDA[0,:,:], Feb_18_NMPDA[0,:,:], Feb_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP March Data'
    fname_out=OUT_PATH+'/ISMNvNMP_MAR_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Mar_00_SCAN[0,:,:], Mar_03_SCAN[0,:,:], Mar_06_SCAN[0,:,:], Mar_09_SCAN[0,:,:], Mar_12_SCAN[0,:,:], Mar_15_SCAN[0,:,:],   Mar_18_SCAN[0,:,:], Mar_21_SCAN[0,:,:], Mar_00_NMP[0,:,:], Mar_03_NMP[0,:,:], Mar_06_NMP[0,:,:], Mar_09_NMP[0,:,:], Mar_12_NMP[0,:,:], Mar_15_NMP[0,:,:], Mar_18_NMP[0,:,:],    Mar_21_NMP[0,:,:], Mar_00_NMPDA[0,:,:], Mar_03_NMPDA[0,:,:], Mar_06_NMPDA[0,:,:], Mar_09_NMPDA[0,:,:], Mar_12_NMPDA[0,:,:], Mar_15_NMPDA[0,:,:], Mar_18_NMPDA[0,:,:], Mar_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP April Data'
    fname_out=OUT_PATH+'/ISMNvNMP_APR_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Apr_00_SCAN[0,:,:], Apr_03_SCAN[0,:,:], Apr_06_SCAN[0,:,:], Apr_09_SCAN[0,:,:], Apr_12_SCAN[0,:,:], Apr_15_SCAN[0,:,:],   Apr_18_SCAN[0,:,:], Apr_21_SCAN[0,:,:], Apr_00_NMP[0,:,:], Apr_03_NMP[0,:,:], Apr_06_NMP[0,:,:], Apr_09_NMP[0,:,:], Apr_12_NMP[0,:,:], Apr_15_NMP[0,:,:], Apr_18_NMP[0,:,:],    Apr_21_NMP[0,:,:], Apr_00_NMPDA[0,:,:], Apr_03_NMPDA[0,:,:], Apr_06_NMPDA[0,:,:], Apr_09_NMPDA[0,:,:], Apr_12_NMPDA[0,:,:], Apr_15_NMPDA[0,:,:], Apr_18_NMPDA[0,:,:], Apr_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP May Data'
    fname_out=OUT_PATH+'/ISMNvNMP_May_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(May_00_SCAN[0,:,:], May_03_SCAN[0,:,:], May_06_SCAN[0,:,:], May_09_SCAN[0,:,:], May_12_SCAN[0,:,:], May_15_SCAN[0,:,:],   May_18_SCAN[0,:,:], May_21_SCAN[0,:,:], May_00_NMP[0,:,:], May_03_NMP[0,:,:], May_06_NMP[0,:,:], May_09_NMP[0,:,:], May_12_NMP[0,:,:], May_15_NMP[0,:,:], May_18_NMP[0,:,:],    May_21_NMP[0,:,:], May_00_NMPDA[0,:,:], May_03_NMPDA[0,:,:], May_06_NMPDA[0,:,:], May_09_NMPDA[0,:,:], May_12_NMPDA[0,:,:], May_15_NMPDA[0,:,:], May_18_NMPDA[0,:,:], May_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP June Data'
    fname_out=OUT_PATH+'/ISMNvNMP_JUN_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Jun_00_SCAN[0,:,:], Jun_03_SCAN[0,:,:], Jun_06_SCAN[0,:,:], Jun_09_SCAN[0,:,:], Jun_12_SCAN[0,:,:], Jun_15_SCAN[0,:,:],   Jun_18_SCAN[0,:,:], Jun_21_SCAN[0,:,:], Jun_00_NMP[0,:,:], Jun_03_NMP[0,:,:], Jun_06_NMP[0,:,:], Jun_09_NMP[0,:,:], Jun_12_NMP[0,:,:], Jun_15_NMP[0,:,:], Jun_18_NMP[0,:,:],    Jun_21_NMP[0,:,:], Jun_00_NMPDA[0,:,:], Jun_03_NMPDA[0,:,:], Jun_06_NMPDA[0,:,:], Jun_09_NMPDA[0,:,:], Jun_12_NMPDA[0,:,:], Jun_15_NMPDA[0,:,:], Jun_18_NMPDA[0,:,:], Jun_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP July Data'
    fname_out=OUT_PATH+'/ISMNvNMP_JUL_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Jul_00_SCAN[0,:,:], Jul_03_SCAN[0,:,:], Jul_06_SCAN[0,:,:], Jul_09_SCAN[0,:,:], Jul_12_SCAN[0,:,:], Jul_15_SCAN[0,:,:],   Jul_18_SCAN[0,:,:], Jul_21_SCAN[0,:,:], Jul_00_NMP[0,:,:], Jul_03_NMP[0,:,:], Jul_06_NMP[0,:,:], Jul_09_NMP[0,:,:], Jul_12_NMP[0,:,:], Jul_15_NMP[0,:,:], Jul_18_NMP[0,:,:],    Jul_21_NMP[0,:,:], Jul_00_NMPDA[0,:,:], Jul_03_NMPDA[0,:,:], Jul_06_NMPDA[0,:,:], Jul_09_NMPDA[0,:,:], Jul_12_NMPDA[0,:,:], Jul_15_NMPDA[0,:,:], Jul_18_NMPDA[0,:,:], Jul_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP August Data'
    fname_out=OUT_PATH+'/ISMNvNMP_AUG_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Aug_00_SCAN[0,:,:], Aug_03_SCAN[0,:,:], Aug_06_SCAN[0,:,:], Aug_09_SCAN[0,:,:], Aug_12_SCAN[0,:,:], Aug_15_SCAN[0,:,:],   Aug_18_SCAN[0,:,:], Aug_21_SCAN[0,:,:], Aug_00_NMP[0,:,:], Aug_03_NMP[0,:,:], Aug_06_NMP[0,:,:], Aug_09_NMP[0,:,:], Aug_12_NMP[0,:,:], Aug_15_NMP[0,:,:], Aug_18_NMP[0,:,:],    Aug_21_NMP[0,:,:], Aug_00_NMPDA[0,:,:], Aug_03_NMPDA[0,:,:], Aug_06_NMPDA[0,:,:], Aug_09_NMPDA[0,:,:], Aug_12_NMPDA[0,:,:], Aug_15_NMPDA[0,:,:], Aug_18_NMPDA[0,:,:], Aug_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP September Data'
    fname_out=OUT_PATH+'/ISMNvNMP_SEP_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Sep_00_SCAN[0,:,:], Sep_03_SCAN[0,:,:], Sep_06_SCAN[0,:,:], Sep_09_SCAN[0,:,:], Sep_12_SCAN[0,:,:], Sep_15_SCAN[0,:,:],   Sep_18_SCAN[0,:,:], Sep_21_SCAN[0,:,:], Sep_00_NMP[0,:,:], Sep_03_NMP[0,:,:], Sep_06_NMP[0,:,:], Sep_09_NMP[0,:,:], Sep_12_NMP[0,:,:], Sep_15_NMP[0,:,:], Sep_18_NMP[0,:,:],    Sep_21_NMP[0,:,:], Sep_00_NMPDA[0,:,:], Sep_03_NMPDA[0,:,:], Sep_06_NMPDA[0,:,:], Sep_09_NMPDA[0,:,:], Sep_12_NMPDA[0,:,:], Sep_15_NMPDA[0,:,:], Sep_18_NMPDA[0,:,:], Sep_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP October Data'
    fname_out=OUT_PATH+'/ISMNvNMP_OCT_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Oct_00_SCAN[0,:,:], Oct_03_SCAN[0,:,:], Oct_06_SCAN[0,:,:], Oct_09_SCAN[0,:,:], Oct_12_SCAN[0,:,:], Oct_15_SCAN[0,:,:],   Oct_18_SCAN[0,:,:], Oct_21_SCAN[0,:,:], Oct_00_NMP[0,:,:], Oct_03_NMP[0,:,:], Oct_06_NMP[0,:,:], Oct_09_NMP[0,:,:], Oct_12_NMP[0,:,:], Oct_15_NMP[0,:,:], Oct_18_NMP[0,:,:],    Oct_21_NMP[0,:,:], Oct_00_NMPDA[0,:,:], Oct_03_NMPDA[0,:,:], Oct_06_NMPDA[0,:,:], Oct_09_NMPDA[0,:,:], Oct_12_NMPDA[0,:,:], Oct_15_NMPDA[0,:,:], Oct_18_NMPDA[0,:,:], Oct_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP November Data'
    fname_out=OUT_PATH+'/ISMNvNMP_NOV_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Nov_00_SCAN[0,:,:], Nov_03_SCAN[0,:,:], Nov_06_SCAN[0,:,:], Nov_09_SCAN[0,:,:], Nov_12_SCAN[0,:,:], Nov_15_SCAN[0,:,:],   Nov_18_SCAN[0,:,:], Nov_21_SCAN[0,:,:], Nov_00_NMP[0,:,:], Nov_03_NMP[0,:,:], Nov_06_NMP[0,:,:], Nov_09_NMP[0,:,:], Nov_12_NMP[0,:,:], Nov_15_NMP[0,:,:], Nov_18_NMP[0,:,:],    Nov_21_NMP[0,:,:], Nov_00_NMPDA[0,:,:], Nov_03_NMPDA[0,:,:], Nov_06_NMPDA[0,:,:], Nov_09_NMPDA[0,:,:], Nov_12_NMPDA[0,:,:], Nov_15_NMPDA[0,:,:], Nov_18_NMPDA[0,:,:], Nov_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')
    
    ptitle='ISMN Gauge Versus LIS Noah MP December Data'
    fname_out=OUT_PATH+'/ISMNvNMP_DEC_'+EXP_NAME+'.png'
    generate_8_panel_scatter_plots.generate_8_panel_scatter_plots(Dec_00_SCAN[0,:,:], Dec_03_SCAN[0,:,:], Dec_06_SCAN[0,:,:], Dec_09_SCAN[0,:,:], Dec_12_SCAN[0,:,:], Dec_15_SCAN[0,:,:],   Dec_18_SCAN[0,:,:], Dec_21_SCAN[0,:,:], Dec_00_NMP[0,:,:], Dec_03_NMP[0,:,:], Dec_06_NMP[0,:,:], Dec_09_NMP[0,:,:], Dec_12_NMP[0,:,:], Dec_15_NMP[0,:,:], Dec_18_NMP[0,:,:],    Dec_21_NMP[0,:,:], Dec_00_NMPDA[0,:,:], Dec_03_NMPDA[0,:,:], Dec_06_NMPDA[0,:,:], Dec_09_NMPDA[0,:,:], Dec_12_NMPDA[0,:,:], Dec_15_NMPDA[0,:,:], Dec_18_NMPDA[0,:,:], Dec_21_NMPDA[0,:,:],  ptitle, fname_out, max_num_stations, 'ISMN Temp (K)', 'Noah MP Temp (K)')

#  generate the vertical plot values
    
for stat_num in range (0, max_num_stations, 1):

  #if np.amax(Jan_00_SCAN[:,:,stat_num]) > 0.0:
    Station_Name=the_master_station_list[stat_num]
    print ('printing January plots')
    print (Station_Name)
    ptitle='January Soil Temperature Profiles for '.join(Station_Name)
    Station_fName=Station_Name.replace(' ','')
    Station_fName=Station_fName.replace('#','-')
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_JAN-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Jan_00_SCAN[:,:,stat_num], Jan_03_SCAN[:,:,stat_num], Jan_06_SCAN[:,:,stat_num], Jan_09_SCAN[:,:,stat_num], Jan_12_SCAN[:,:,stat_num], Jan_15_SCAN[:,:,stat_num], Jan_18_SCAN[:,:,stat_num], Jan_21_SCAN[:,:,stat_num], Jan_00_ISCCP[:,stat_num], Jan_03_ISCCP[:,stat_num], Jan_06_ISCCP[:,stat_num], Jan_09_ISCCP[:,stat_num], Jan_12_ISCCP[:,stat_num], Jan_15_ISCCP[:,stat_num], Jan_18_ISCCP[:,stat_num], Jan_21_ISCCP[:,stat_num], Jan_00_N36[:,:,stat_num],  Jan_03_N36[:,:,stat_num],  Jan_06_N36[:,:,stat_num], Jan_09_N36[:,:,stat_num],  Jan_12_N36[:,:,stat_num],  Jan_15_N36[:,:,stat_num],  Jan_18_N36[:,:,stat_num], Jan_21_N36[:,:,stat_num],  Jan_00_N36_Tskin[:,stat_num],  Jan_03_N36_Tskin[:,stat_num],  Jan_06_N36_Tskin[:,stat_num], Jan_09_N36_Tskin[:,stat_num],  Jan_12_N36_Tskin[:,stat_num],  Jan_15_N36_Tskin[:,stat_num],  Jan_18_N36_Tskin[:,stat_num], Jan_21_N36_Tskin[:,stat_num],  Jan_00_NMP[:,:,stat_num],  Jan_03_NMP[:,:,stat_num],  Jan_06_NMP[:,:,stat_num], Jan_09_NMP[:,:,stat_num],  Jan_12_NMP[:,:,stat_num],  Jan_15_NMP[:,:,stat_num],  Jan_18_NMP[:,:,stat_num], Jan_21_NMP[:,:,stat_num],  Jan_00_NMP_Tskin[:,stat_num],  Jan_03_NMP_Tskin[:,stat_num],  Jan_06_NMP_Tskin[:,stat_num], Jan_09_NMP_Tskin[:,stat_num],  Jan_12_NMP_Tskin[:,stat_num],  Jan_15_NMP_Tskin[:,stat_num],  Jan_18_NMP_Tskin[:,stat_num], Jan_21_NMP_Tskin[:,stat_num],  ptitle, fname_out)

  #if np.amax(Feb_00_SCAN[:,:,stat_num]) > 0.0:
    print ('printing February plots')
    ptitle='February Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_FEB-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Feb_00_SCAN[:,:,stat_num], Feb_03_SCAN[:,:,stat_num], Feb_06_SCAN[:,:,stat_num], Feb_09_SCAN[:,:,stat_num], Feb_12_SCAN[:,:,stat_num], Feb_15_SCAN[:,:,stat_num], Feb_18_SCAN[:,:,stat_num], Feb_21_SCAN[:,:,stat_num], Feb_00_ISCCP[:,stat_num], Feb_03_ISCCP[:,stat_num], Feb_06_ISCCP[:,stat_num], Feb_09_ISCCP[:,stat_num], Feb_12_ISCCP[:,stat_num], Feb_15_ISCCP[:,stat_num], Feb_18_ISCCP[:,stat_num], Feb_21_ISCCP[:,stat_num], Feb_00_N36[:,:,stat_num],  Feb_03_N36[:,:,stat_num],  Feb_06_N36[:,:,stat_num], Feb_09_N36[:,:,stat_num],  Feb_12_N36[:,:,stat_num],  Feb_15_N36[:,:,stat_num],  Feb_18_N36[:,:,stat_num], Feb_21_N36[:,:,stat_num],  Feb_00_N36_Tskin[:,stat_num],  Feb_03_N36_Tskin[:,stat_num],  Feb_06_N36_Tskin[:,stat_num], Feb_09_N36_Tskin[:,stat_num],  Feb_12_N36_Tskin[:,stat_num],  Feb_15_N36_Tskin[:,stat_num],  Feb_18_N36_Tskin[:,stat_num], Feb_21_N36_Tskin[:,stat_num],  Feb_00_NMP[:,:,stat_num],  Feb_03_NMP[:,:,stat_num],  Feb_06_NMP[:,:,stat_num], Feb_09_NMP[:,:,stat_num],  Feb_12_NMP[:,:,stat_num],  Feb_15_NMP[:,:,stat_num],  Feb_18_NMP[:,:,stat_num], Feb_21_NMP[:,:,stat_num],  Feb_00_NMP_Tskin[:,stat_num],  Feb_03_NMP_Tskin[:,stat_num],  Feb_06_NMP_Tskin[:,stat_num], Feb_09_NMP_Tskin[:,stat_num],  Feb_12_NMP_Tskin[:,stat_num],  Feb_15_NMP_Tskin[:,stat_num],  Feb_18_NMP_Tskin[:,stat_num], Feb_21_NMP_Tskin[:,stat_num],  ptitle, fname_out)

  #if np.amax(Mar_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='March Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_MAR-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Mar_00_SCAN[:,:,stat_num], Mar_03_SCAN[:,:,stat_num], Mar_06_SCAN[:,:,stat_num], Mar_09_SCAN[:,:,stat_num], Mar_12_SCAN[:,:,stat_num], Mar_15_SCAN[:,:,stat_num], Mar_18_SCAN[:,:,stat_num], Mar_21_SCAN[:,:,stat_num], Mar_00_ISCCP[:,stat_num], Mar_03_ISCCP[:,stat_num], Mar_06_ISCCP[:,stat_num], Mar_09_ISCCP[:,stat_num], Mar_12_ISCCP[:,stat_num], Mar_15_ISCCP[:,stat_num], Mar_18_ISCCP[:,stat_num], Mar_21_ISCCP[:,stat_num], Mar_00_N36[:,:,stat_num], Mar_03_N36[:,:,stat_num], Mar_06_N36[:,:,stat_num], Mar_09_N36[:,:,stat_num], Mar_12_N36[:,:,stat_num], Mar_15_N36[:,:,stat_num], Mar_18_N36[:,:,stat_num], Mar_21_N36[:,:,stat_num], Mar_00_N36_Tskin[:,stat_num], Mar_03_N36_Tskin[:,stat_num], Mar_06_N36_Tskin[:,stat_num], Mar_09_N36_Tskin[:,stat_num], Mar_12_N36_Tskin[:,stat_num], Mar_15_N36_Tskin[:,stat_num], Mar_18_N36_Tskin[:,stat_num], Mar_21_N36_Tskin[:,stat_num], Mar_00_NMP[:,:,stat_num], Mar_03_NMP[:,:,stat_num], Mar_06_NMP[:,:,stat_num], Mar_09_NMP[:,:,stat_num], Mar_12_NMP[:,:,stat_num], Mar_15_NMP[:,:,stat_num], Mar_18_NMP[:,:,stat_num], Mar_21_NMP[:,:,stat_num], Mar_00_NMP_Tskin[:,stat_num], Mar_03_NMP_Tskin[:,stat_num], Mar_06_NMP_Tskin[:,stat_num], Mar_09_NMP_Tskin[:,stat_num], Mar_12_NMP_Tskin[:,stat_num], Mar_15_NMP_Tskin[:,stat_num], Mar_18_NMP_Tskin[:,stat_num], Mar_21_NMP_Tskin[:,stat_num], ptitle, fname_out)

  #if np.amax(Apr_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='April Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_APR-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Apr_00_SCAN[:,:,stat_num], Apr_03_SCAN[:,:,stat_num], Apr_06_SCAN[:,:,stat_num], Apr_09_SCAN[:,:,stat_num], Apr_12_SCAN[:,:,stat_num], Apr_15_SCAN[:,:,stat_num], Apr_18_SCAN[:,:,stat_num], Apr_21_SCAN[:,:,stat_num], Apr_00_ISCCP[:,stat_num], Apr_03_ISCCP[:,stat_num], Apr_06_ISCCP[:,stat_num], Apr_09_ISCCP[:,stat_num], Apr_12_ISCCP[:,stat_num], Apr_15_ISCCP[:,stat_num], Apr_18_ISCCP[:,stat_num], Apr_21_ISCCP[:,stat_num], Apr_00_N36[:,:,stat_num], Apr_03_N36[:,:,stat_num], Apr_06_N36[:,:,stat_num], Apr_09_N36[:,:,stat_num], Apr_12_N36[:,:,stat_num], Apr_15_N36[:,:,stat_num], Apr_18_N36[:,:,stat_num], Apr_21_N36[:,:,stat_num], Apr_00_N36_Tskin[:,stat_num], Apr_03_N36_Tskin[:,stat_num], Apr_06_N36_Tskin[:,stat_num], Apr_09_N36_Tskin[:,stat_num], Apr_12_N36_Tskin[:,stat_num], Apr_15_N36_Tskin[:,stat_num], Apr_18_N36_Tskin[:,stat_num], Apr_21_N36_Tskin[:,stat_num], Apr_00_NMP[:,:,stat_num], Apr_03_NMP[:,:,stat_num], Apr_06_NMP[:,:,stat_num], Apr_09_NMP[:,:,stat_num], Apr_12_NMP[:,:,stat_num], Apr_15_NMP[:,:,stat_num], Apr_18_NMP[:,:,stat_num], Apr_21_NMP[:,:,stat_num], Apr_00_NMP_Tskin[:,stat_num], Apr_03_NMP_Tskin[:,stat_num], Apr_06_NMP_Tskin[:,stat_num], Apr_09_NMP_Tskin[:,stat_num], Apr_12_NMP_Tskin[:,stat_num], Apr_15_NMP_Tskin[:,stat_num], Apr_18_NMP_Tskin[:,stat_num], Apr_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
  #if np.amax(May_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='May Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_MAY-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(May_00_SCAN[:,:,stat_num], May_03_SCAN[:,:,stat_num], May_06_SCAN[:,:,stat_num], May_09_SCAN[:,:,stat_num], May_12_SCAN[:,:,stat_num], May_15_SCAN[:,:,stat_num], May_18_SCAN[:,:,stat_num], May_21_SCAN[:,:,stat_num], May_00_ISCCP[:,stat_num], May_03_ISCCP[:,stat_num], May_06_ISCCP[:,stat_num], May_09_ISCCP[:,stat_num], May_12_ISCCP[:,stat_num], May_15_ISCCP[:,stat_num], May_18_ISCCP[:,stat_num], May_21_ISCCP[:,stat_num], May_00_N36[:,:,stat_num],  May_03_N36[:,:,stat_num],  May_06_N36[:,:,stat_num], May_09_N36[:,:,stat_num],  May_12_N36[:,:,stat_num],  May_15_N36[:,:,stat_num],  May_18_N36[:,:,stat_num], May_21_N36[:,:,stat_num],  May_00_N36_Tskin[:,stat_num],  May_03_N36_Tskin[:,stat_num],  May_06_N36_Tskin[:,stat_num], May_09_N36_Tskin[:,stat_num],  May_12_N36_Tskin[:,stat_num],  May_15_N36_Tskin[:,stat_num],  May_18_N36_Tskin[:,stat_num], May_21_N36_Tskin[:,stat_num],  May_00_NMP[:,:,stat_num],  May_03_NMP[:,:,stat_num],  May_06_NMP[:,:,stat_num], May_09_NMP[:,:,stat_num],  May_12_NMP[:,:,stat_num],  May_15_NMP[:,:,stat_num],  May_18_NMP[:,:,stat_num], May_21_NMP[:,:,stat_num], May_00_NMP_Tskin[:,stat_num],  May_03_NMP_Tskin[:,stat_num],  May_06_NMP_Tskin[:,stat_num], May_09_NMP_Tskin[:,stat_num],  May_12_NMP_Tskin[:,stat_num],  May_15_NMP_Tskin[:,stat_num],  May_18_NMP_Tskin[:,stat_num], May_21_NMP_Tskin[:,stat_num], ptitle, fname_out)

  #if np.amax(Jun_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='June Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_JUN-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Jun_00_SCAN[:,:,stat_num], Jun_03_SCAN[:,:,stat_num], Jun_06_SCAN[:,:,stat_num], Jun_09_SCAN[:,:,stat_num], Jun_12_SCAN[:,:,stat_num], Jun_15_SCAN[:,:,stat_num], Jun_18_SCAN[:,:,stat_num], Jun_21_SCAN[:,:,stat_num], Jun_00_ISCCP[:,stat_num], Jun_03_ISCCP[:,stat_num], Jun_06_ISCCP[:,stat_num], Jun_09_ISCCP[:,stat_num], Jun_12_ISCCP[:,stat_num], Jun_15_ISCCP[:,stat_num], Jun_18_ISCCP[:,stat_num], Jun_21_ISCCP[:,stat_num], Jun_00_N36[:,:,stat_num],  Jun_03_N36[:,:,stat_num],  Jun_06_N36[:,:,stat_num], Jun_09_N36[:,:,stat_num],  Jun_12_N36[:,:,stat_num],  Jun_15_N36[:,:,stat_num],  Jun_18_N36[:,:,stat_num], Jun_21_N36[:,:,stat_num],  Jun_00_N36_Tskin[:,stat_num],  Jun_03_N36_Tskin[:,stat_num],  Jun_06_N36_Tskin[:,stat_num], Jun_09_N36_Tskin[:,stat_num],  Jun_12_N36_Tskin[:,stat_num],  Jun_15_N36_Tskin[:,stat_num],  Jun_18_N36_Tskin[:,stat_num], Jun_21_N36_Tskin[:,stat_num],  Jun_00_NMP[:,:,stat_num],  Jun_03_NMP[:,:,stat_num],  Jun_06_NMP[:,:,stat_num], Jun_09_NMP[:,:,stat_num],  Jun_12_NMP[:,:,stat_num],  Jun_15_NMP[:,:,stat_num],  Jun_18_NMP[:,:,stat_num], Jun_21_NMP[:,:,stat_num], Jun_00_NMP_Tskin[:,stat_num],  Jun_03_NMP_Tskin[:,stat_num],  Jun_06_NMP_Tskin[:,stat_num], Jun_09_NMP_Tskin[:,stat_num],  Jun_12_NMP_Tskin[:,stat_num],  Jun_15_NMP_Tskin[:,stat_num],  Jun_18_NMP_Tskin[:,stat_num], Jun_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
  #if np.amax(Jul_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='July Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_JUL-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Jul_00_SCAN[:,:,stat_num], Jul_03_SCAN[:,:,stat_num], Jul_06_SCAN[:,:,stat_num], Jul_09_SCAN[:,:,stat_num], Jul_12_SCAN[:,:,stat_num], Jul_15_SCAN[:,:,stat_num], Jul_18_SCAN[:,:,stat_num], Jul_21_SCAN[:,:,stat_num], Jul_00_ISCCP[:,stat_num], Jul_03_ISCCP[:,stat_num], Jul_06_ISCCP[:,stat_num], Jul_09_ISCCP[:,stat_num], Jul_12_ISCCP[:,stat_num], Jul_15_ISCCP[:,stat_num], Jul_18_ISCCP[:,stat_num], Jul_21_ISCCP[:,stat_num], Jul_00_N36[:,:,stat_num],  Jul_03_N36[:,:,stat_num],  Jul_06_N36[:,:,stat_num], Jul_09_N36[:,:,stat_num],  Jul_12_N36[:,:,stat_num],  Jul_15_N36[:,:,stat_num],  Jul_18_N36[:,:,stat_num], Jul_21_N36[:,:,stat_num],  Jul_00_N36_Tskin[:,stat_num],  Jul_03_N36_Tskin[:,stat_num],  Jul_06_N36_Tskin[:,stat_num], Jul_09_N36_Tskin[:,stat_num],  Jul_12_N36_Tskin[:,stat_num],  Jul_15_N36_Tskin[:,stat_num],  Jul_18_N36_Tskin[:,stat_num], Jul_21_N36_Tskin[:,stat_num], Jul_00_NMP[:,:,stat_num],  Jul_03_NMP[:,:,stat_num],  Jul_06_NMP[:,:,stat_num], Jul_09_NMP[:,:,stat_num],  Jul_12_NMP[:,:,stat_num],  Jul_15_NMP[:,:,stat_num],  Jul_18_NMP[:,:,stat_num], Jul_21_NMP[:,:,stat_num], Jul_00_NMP_Tskin[:,stat_num],  Jul_03_NMP_Tskin[:,stat_num],  Jul_06_NMP_Tskin[:,stat_num], Jul_09_NMP_Tskin[:,stat_num],  Jul_12_NMP_Tskin[:,stat_num],  Jul_15_NMP_Tskin[:,stat_num],  Jul_18_NMP_Tskin[:,stat_num], Jul_21_NMP_Tskin[:,stat_num], ptitle, fname_out)

  #if np.amax(Aug_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='August Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_AUG-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Aug_00_SCAN[:,:,stat_num], Aug_03_SCAN[:,:,stat_num], Aug_06_SCAN[:,:,stat_num], Aug_09_SCAN[:,:,stat_num], Aug_12_SCAN[:,:,stat_num], Aug_15_SCAN[:,:,stat_num], Aug_18_SCAN[:,:,stat_num], Aug_21_SCAN[:,:,stat_num], Aug_00_ISCCP[:,stat_num], Aug_03_ISCCP[:,stat_num], Aug_06_ISCCP[:,stat_num], Aug_09_ISCCP[:,stat_num], Aug_12_ISCCP[:,stat_num], Aug_15_ISCCP[:,stat_num], Aug_18_ISCCP[:,stat_num], Aug_21_ISCCP[:,stat_num], Aug_00_N36[:,:,stat_num],  Aug_03_N36[:,:,stat_num],  Aug_06_N36[:,:,stat_num], Aug_09_N36[:,:,stat_num],  Aug_12_N36[:,:,stat_num],  Aug_15_N36[:,:,stat_num],  Aug_18_N36[:,:,stat_num], Aug_21_N36[:,:,stat_num],  Aug_00_N36_Tskin[:,stat_num],  Aug_03_N36_Tskin[:,stat_num],  Aug_06_N36_Tskin[:,stat_num], Aug_09_N36_Tskin[:,stat_num],  Aug_12_N36_Tskin[:,stat_num],  Aug_15_N36_Tskin[:,stat_num],  Aug_18_N36_Tskin[:,stat_num], Aug_21_N36_Tskin[:,stat_num], Aug_00_NMP[:,:,stat_num],  Aug_03_NMP[:,:,stat_num],  Aug_06_NMP[:,:,stat_num], Aug_09_NMP[:,:,stat_num],  Aug_12_NMP[:,:,stat_num],  Aug_15_NMP[:,:,stat_num],  Aug_18_NMP[:,:,stat_num], Aug_21_NMP[:,:,stat_num], Aug_00_NMP_Tskin[:,stat_num],  Aug_03_NMP_Tskin[:,stat_num],  Aug_06_NMP_Tskin[:,stat_num], Aug_09_NMP_Tskin[:,stat_num],  Aug_12_NMP_Tskin[:,stat_num],  Aug_15_NMP_Tskin[:,stat_num],  Aug_18_NMP_Tskin[:,stat_num], Aug_21_NMP_Tskin[:,stat_num], ptitle, fname_out)

  #if np.amax(Sep_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='September Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_SEP-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Sep_00_SCAN[:,:,stat_num], Sep_03_SCAN[:,:,stat_num], Sep_06_SCAN[:,:,stat_num], Sep_09_SCAN[:,:,stat_num], Sep_12_SCAN[:,:,stat_num], Sep_15_SCAN[:,:,stat_num], Sep_18_SCAN[:,:,stat_num], Sep_21_SCAN[:,:,stat_num], Sep_00_ISCCP[:,stat_num], Sep_03_ISCCP[:,stat_num], Sep_06_ISCCP[:,stat_num], Sep_09_ISCCP[:,stat_num], Sep_12_ISCCP[:,stat_num], Sep_15_ISCCP[:,stat_num], Sep_18_ISCCP[:,stat_num], Sep_21_ISCCP[:,stat_num], Sep_00_N36[:,:,stat_num],  Sep_03_N36[:,:,stat_num],  Sep_06_N36[:,:,stat_num], Sep_09_N36[:,:,stat_num],  Sep_12_N36[:,:,stat_num],  Sep_15_N36[:,:,stat_num],  Sep_18_N36[:,:,stat_num], Sep_21_N36[:,:,stat_num],  Sep_00_N36_Tskin[:,stat_num],  Sep_03_N36_Tskin[:,stat_num],  Sep_06_N36_Tskin[:,stat_num], Sep_09_N36_Tskin[:,stat_num],  Sep_12_N36_Tskin[:,stat_num],  Sep_15_N36_Tskin[:,stat_num],  Sep_18_N36_Tskin[:,stat_num], Sep_21_N36_Tskin[:,stat_num],  Sep_00_NMP[:,:,stat_num],  Sep_03_NMP[:,:,stat_num],  Sep_06_NMP[:,:,stat_num], Sep_09_NMP[:,:,stat_num],  Sep_12_NMP[:,:,stat_num],  Sep_15_NMP[:,:,stat_num],  Sep_18_NMP[:,:,stat_num], Sep_21_NMP[:,:,stat_num], Sep_00_NMP_Tskin[:,stat_num],  Sep_03_NMP_Tskin[:,stat_num],  Sep_06_NMP_Tskin[:,stat_num], Sep_09_NMP_Tskin[:,stat_num],  Sep_12_NMP_Tskin[:,stat_num],  Sep_15_NMP_Tskin[:,stat_num],  Sep_18_NMP_Tskin[:,stat_num], Sep_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
  #if np.amax(Oct_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='October Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_OCT-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Oct_00_SCAN[:,:,stat_num], Oct_03_SCAN[:,:,stat_num], Oct_06_SCAN[:,:,stat_num], Oct_09_SCAN[:,:,stat_num], Oct_12_SCAN[:,:,stat_num], Oct_15_SCAN[:,:,stat_num], Oct_18_SCAN[:,:,stat_num], Oct_21_SCAN[:,:,stat_num], Oct_00_ISCCP[:,stat_num], Oct_03_ISCCP[:,stat_num], Oct_06_ISCCP[:,stat_num], Oct_09_ISCCP[:,stat_num], Oct_12_ISCCP[:,stat_num], Oct_15_ISCCP[:,stat_num], Oct_18_ISCCP[:,stat_num], Oct_21_ISCCP[:,stat_num], Oct_00_N36[:,:,stat_num],  Oct_03_N36[:,:,stat_num],  Oct_06_N36[:,:,stat_num], Oct_09_N36[:,:,stat_num],  Oct_12_N36[:,:,stat_num],  Oct_15_N36[:,:,stat_num],  Oct_18_N36[:,:,stat_num], Oct_21_N36[:,:,stat_num],  Oct_00_N36_Tskin[:,stat_num],  Oct_03_N36_Tskin[:,stat_num],  Oct_06_N36_Tskin[:,stat_num], Oct_09_N36_Tskin[:,stat_num],  Oct_12_N36_Tskin[:,stat_num],  Oct_15_N36_Tskin[:,stat_num],  Oct_18_N36_Tskin[:,stat_num], Oct_21_N36_Tskin[:,stat_num],  Oct_00_NMP[:,:,stat_num],  Oct_03_NMP[:,:,stat_num],  Oct_06_NMP[:,:,stat_num], Oct_09_NMP[:,:,stat_num],  Oct_12_NMP[:,:,stat_num],  Oct_15_NMP[:,:,stat_num],  Oct_18_NMP[:,:,stat_num], Oct_21_NMP[:,:,stat_num], Oct_00_NMP_Tskin[:,stat_num],  Oct_03_NMP_Tskin[:,stat_num],  Oct_06_NMP_Tskin[:,stat_num], Oct_09_NMP_Tskin[:,stat_num],  Oct_12_NMP_Tskin[:,stat_num],  Oct_15_NMP_Tskin[:,stat_num],  Oct_18_NMP_Tskin[:,stat_num], Oct_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
  #if np.amax(Nov_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='November Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_NOV-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Nov_00_SCAN[:,:,stat_num], Nov_03_SCAN[:,:,stat_num], Nov_06_SCAN[:,:,stat_num], Nov_09_SCAN[:,:,stat_num], Nov_12_SCAN[:,:,stat_num], Nov_15_SCAN[:,:,stat_num], Nov_18_SCAN[:,:,stat_num], Nov_21_SCAN[:,:,stat_num], Nov_00_ISCCP[:,stat_num], Nov_03_ISCCP[:,stat_num], Nov_06_ISCCP[:,stat_num], Nov_09_ISCCP[:,stat_num], Nov_12_ISCCP[:,stat_num], Nov_15_ISCCP[:,stat_num], Nov_18_ISCCP[:,stat_num], Nov_21_ISCCP[:,stat_num], Nov_00_N36[:,:,stat_num],  Nov_03_N36[:,:,stat_num],  Nov_06_N36[:,:,stat_num], Nov_09_N36[:,:,stat_num],  Nov_12_N36[:,:,stat_num],  Nov_15_N36[:,:,stat_num],  Nov_18_N36[:,:,stat_num], Nov_21_N36[:,:,stat_num],  Nov_00_N36_Tskin[:,stat_num],  Nov_03_N36_Tskin[:,stat_num],  Nov_06_N36_Tskin[:,stat_num], Nov_09_N36_Tskin[:,stat_num],  Nov_12_N36_Tskin[:,stat_num],  Nov_15_N36_Tskin[:,stat_num],  Nov_18_N36_Tskin[:,stat_num], Nov_21_N36_Tskin[:,stat_num],  Nov_00_NMP[:,:,stat_num],  Nov_03_NMP[:,:,stat_num],  Nov_06_NMP[:,:,stat_num], Nov_09_NMP[:,:,stat_num],  Nov_12_NMP[:,:,stat_num],  Nov_15_NMP[:,:,stat_num],  Nov_18_NMP[:,:,stat_num], Nov_21_NMP[:,:,stat_num], Nov_00_NMP_Tskin[:,stat_num],  Nov_03_NMP_Tskin[:,stat_num],  Nov_06_NMP_Tskin[:,stat_num], Nov_09_NMP_Tskin[:,stat_num],  Nov_12_NMP_Tskin[:,stat_num],  Nov_15_NMP_Tskin[:,stat_num],  Nov_18_NMP_Tskin[:,stat_num], Nov_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
  #if np.amax(Dec_00_SCAN[:,:,stat_num]) > 0.0:
    ptitle='December Soil Temperature Profiles for '+Station_Name
    fname_out=OUT_PATH+'/Soil_Temp_Profiles_Plots_DEC-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.monthly(Dec_00_SCAN[:,:,stat_num], Dec_03_SCAN[:,:,stat_num], Dec_06_SCAN[:,:,stat_num], Dec_09_SCAN[:,:,stat_num], Dec_12_SCAN[:,:,stat_num], Dec_15_SCAN[:,:,stat_num], Dec_18_SCAN[:,:,stat_num], Dec_21_SCAN[:,:,stat_num], Dec_00_ISCCP[:,stat_num], Dec_03_ISCCP[:,stat_num], Dec_06_ISCCP[:,stat_num], Dec_09_ISCCP[:,stat_num], Dec_12_ISCCP[:,stat_num], Dec_15_ISCCP[:,stat_num], Dec_18_ISCCP[:,stat_num], Dec_21_ISCCP[:,stat_num], Dec_00_N36[:,:,stat_num],  Dec_03_N36[:,:,stat_num],  Dec_06_N36[:,:,stat_num], Dec_09_N36[:,:,stat_num],  Dec_12_N36[:,:,stat_num],  Dec_15_N36[:,:,stat_num],  Dec_18_N36[:,:,stat_num], Dec_21_N36[:,:,stat_num],  Dec_00_N36_Tskin[:,stat_num],  Dec_03_N36_Tskin[:,stat_num],  Dec_06_N36_Tskin[:,stat_num], Dec_09_N36_Tskin[:,stat_num],  Dec_12_N36_Tskin[:,stat_num],  Dec_15_N36_Tskin[:,stat_num],  Dec_18_N36_Tskin[:,stat_num], Dec_21_N36_Tskin[:,stat_num],  Dec_00_NMP[:,:,stat_num],  Dec_03_NMP[:,:,stat_num],  Dec_06_NMP[:,:,stat_num], Dec_09_NMP[:,:,stat_num],  Dec_12_NMP[:,:,stat_num],  Dec_15_NMP[:,:,stat_num],  Dec_18_NMP[:,:,stat_num], Dec_21_NMP[:,:,stat_num], Dec_00_NMP_Tskin[:,stat_num],  Dec_03_NMP_Tskin[:,stat_num],  Dec_06_NMP_Tskin[:,stat_num], Dec_09_NMP_Tskin[:,stat_num],  Dec_12_NMP_Tskin[:,stat_num],  Dec_15_NMP_Tskin[:,stat_num],  Dec_18_NMP_Tskin[:,stat_num], Dec_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    
    ptitle='Seasonal Soil Temperature Profiles 00 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_00Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_00_SCAN[:,:,stat_num], Spring_00_SCAN[:,:,stat_num], Summer_00_SCAN[:,:,stat_num], Fall_00_SCAN[:,:,stat_num], Winter_00_ISCCP[:,stat_num], Spring_00_ISCCP[:,stat_num], Summer_00_ISCCP[:,stat_num], Fall_00_ISCCP[:,stat_num], Winter_00_N36[:,:,stat_num], Spring_00_N36[:,:,stat_num], Summer_00_N36[:,:,stat_num], Fall_00_N36[:,:,stat_num], Winter_00_N36_Tskin[:,stat_num], Spring_00_N36_Tskin[:,stat_num], Summer_00_N36_Tskin[:,stat_num], Fall_00_N36_Tskin[:,stat_num], Winter_00_NMP[:,:,stat_num], Spring_00_NMP[:,:,stat_num], Summer_00_NMP[:,:,stat_num], Fall_00_NMP[:,:,stat_num], Winter_00_NMP_Tskin[:,stat_num], Spring_00_NMP_Tskin[:,stat_num], Summer_00_NMP_Tskin[:,stat_num], Fall_00_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    ptitle='Seasonal Soil Temperature Profiles 03 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_03Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_03_SCAN[:,:,stat_num], Spring_03_SCAN[:,:,stat_num], Summer_03_SCAN[:,:,stat_num], Fall_03_SCAN[:,:,stat_num], Winter_03_ISCCP[:,stat_num], Spring_03_ISCCP[:,stat_num], Summer_03_ISCCP[:,stat_num], Fall_03_ISCCP[:,stat_num], Winter_03_N36[:,:,stat_num], Spring_03_N36[:,:,stat_num], Summer_03_N36[:,:,stat_num], Fall_03_N36[:,:,stat_num], Winter_03_N36_Tskin[:,stat_num], Spring_03_N36_Tskin[:,stat_num], Summer_03_N36_Tskin[:,stat_num], Fall_03_N36_Tskin[:,stat_num], Winter_03_NMP[:,:,stat_num], Spring_03_NMP[:,:,stat_num], Summer_03_NMP[:,:,stat_num], Fall_03_NMP[:,:,stat_num], Winter_03_NMP_Tskin[:,stat_num], Spring_03_NMP_Tskin[:,stat_num], Summer_03_NMP_Tskin[:,stat_num], Fall_03_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    ptitle='Seasonal Soil Temperature Profiles 06 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_06Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_06_SCAN[:,:,stat_num], Spring_06_SCAN[:,:,stat_num], Summer_06_SCAN[:,:,stat_num], Fall_06_SCAN[:,:,stat_num], Winter_06_ISCCP[:,stat_num], Spring_06_ISCCP[:,stat_num], Summer_06_ISCCP[:,stat_num], Fall_06_ISCCP[:,stat_num], Winter_06_N36[:,:,stat_num], Spring_06_N36[:,:,stat_num], Summer_06_N36[:,:,stat_num], Fall_06_N36[:,:,stat_num],  Winter_06_N36_Tskin[:,stat_num], Spring_06_N36_Tskin[:,stat_num], Summer_06_N36_Tskin[:,stat_num], Fall_06_N36_Tskin[:,stat_num], Winter_06_NMP[:,:,stat_num], Spring_06_NMP[:,:,stat_num], Summer_06_NMP[:,:,stat_num], Fall_06_NMP[:,:,stat_num], Winter_06_NMP_Tskin[:,stat_num], Spring_06_NMP_Tskin[:,stat_num], Summer_06_NMP_Tskin[:,stat_num], Fall_06_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    ptitle='Seasonal Soil Temperature Profiles 09 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_09Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_09_SCAN[:,:,stat_num], Spring_09_SCAN[:,:,stat_num], Summer_09_SCAN[:,:,stat_num], Fall_09_SCAN[:,:,stat_num], Winter_09_ISCCP[:,stat_num], Spring_09_ISCCP[:,stat_num], Summer_09_ISCCP[:,stat_num], Fall_09_ISCCP[:,stat_num], Winter_09_N36[:,:,stat_num], Spring_09_N36[:,:,stat_num], Summer_09_N36[:,:,stat_num], Fall_09_N36[:,:,stat_num],  Winter_09_N36_Tskin[:,stat_num], Spring_09_N36_Tskin[:,stat_num], Summer_09_N36_Tskin[:,stat_num], Fall_09_N36_Tskin[:,stat_num], Winter_09_NMP[:,:,stat_num], Spring_09_NMP[:,:,stat_num], Summer_09_NMP[:,:,stat_num], Fall_09_NMP[:,:,stat_num], Winter_09_NMP_Tskin[:,stat_num], Spring_09_NMP_Tskin[:,stat_num], Summer_09_NMP_Tskin[:,stat_num], Fall_09_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    ptitle='Seasonal Soil Temperature Profiles 12 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_12Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_12_SCAN[:,:,stat_num], Spring_12_SCAN[:,:,stat_num], Summer_12_SCAN[:,:,stat_num], Fall_12_SCAN[:,:,stat_num], Winter_12_ISCCP[:,stat_num], Spring_12_ISCCP[:,stat_num], Summer_12_ISCCP[:,stat_num], Fall_12_ISCCP[:,stat_num], Winter_12_N36[:,:,stat_num], Spring_12_N36[:,:,stat_num], Summer_12_N36[:,:,stat_num], Fall_12_N36[:,:,stat_num],  Winter_12_N36_Tskin[:,stat_num], Spring_12_N36_Tskin[:,stat_num], Summer_12_N36_Tskin[:,stat_num], Fall_12_N36_Tskin[:,stat_num], Winter_12_NMP[:,:,stat_num], Spring_12_NMP[:,:,stat_num], Summer_12_NMP[:,:,stat_num], Fall_12_NMP[:,:,stat_num], Winter_12_NMP_Tskin[:,stat_num], Spring_12_NMP_Tskin[:,stat_num], Summer_12_NMP_Tskin[:,stat_num], Fall_12_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    ptitle='Seasonal Soil Temperature Profiles 15 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_15Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_15_SCAN[:,:,stat_num], Spring_15_SCAN[:,:,stat_num], Summer_15_SCAN[:,:,stat_num], Fall_15_SCAN[:,:,stat_num], Winter_15_ISCCP[:,stat_num], Spring_15_ISCCP[:,stat_num], Summer_15_ISCCP[:,stat_num], Fall_15_ISCCP[:,stat_num], Winter_15_N36[:,:,stat_num], Spring_15_N36[:,:,stat_num], Summer_15_N36[:,:,stat_num], Fall_15_N36[:,:,stat_num],  Winter_15_N36_Tskin[:,stat_num], Spring_15_N36_Tskin[:,stat_num], Summer_15_N36_Tskin[:,stat_num], Fall_15_N36_Tskin[:,stat_num], Winter_15_NMP[:,:,stat_num], Spring_15_NMP[:,:,stat_num], Summer_15_NMP[:,:,stat_num], Fall_15_NMP[:,:,stat_num], Winter_15_NMP_Tskin[:,stat_num], Spring_15_NMP_Tskin[:,stat_num], Summer_15_NMP_Tskin[:,stat_num], Fall_15_NMP_Tskin[:,stat_num], ptitle, fname_out)

    ptitle='Seasonal Soil Temperature Profiles 18 UTC cycle for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_18Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_18_SCAN[:,:,stat_num], Spring_18_SCAN[:,:,stat_num], Summer_18_SCAN[:,:,stat_num], Fall_18_SCAN[:,:,stat_num], Winter_18_ISCCP[:,stat_num], Spring_18_ISCCP[:,stat_num], Summer_18_ISCCP[:,stat_num], Fall_18_ISCCP[:,stat_num], Winter_18_N36[:,:,stat_num], Spring_18_N36[:,:,stat_num], Summer_18_N36[:,:,stat_num], Fall_18_N36[:,:,stat_num],  Winter_18_N36_Tskin[:,stat_num], Spring_18_N36_Tskin[:,stat_num], Summer_18_N36_Tskin[:,stat_num], Fall_18_N36_Tskin[:,stat_num], Winter_18_NMP[:,:,stat_num], Spring_18_NMP[:,:,stat_num], Summer_18_NMP[:,:,stat_num], Fall_18_NMP[:,:,stat_num], Winter_18_NMP_Tskin[:,stat_num], Spring_18_NMP_Tskin[:,stat_num], Summer_18_NMP_Tskin[:,stat_num], Fall_18_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
    ptitle='Seasonal Soil Temperature Profiles 21 HR for '+Station_Name
    fname_out=OUT_PATH+'/Seasonal_Soil_Temp_Profiles_Plots_21Z-'+EXP_NAME+'_'+Station_fName+'.png'
    generate_station_profiles.seasonal(Winter_21_SCAN[:,:,stat_num], Spring_21_SCAN[:,:,stat_num], Summer_21_SCAN[:,:,stat_num], Fall_21_SCAN[:,:,stat_num], Winter_21_ISCCP[:,stat_num], Spring_21_ISCCP[:,stat_num], Summer_21_ISCCP[:,stat_num], Fall_21_ISCCP[:,stat_num], Winter_21_N36[:,:,stat_num], Spring_21_N36[:,:,stat_num], Summer_21_N36[:,:,stat_num], Fall_21_N36[:,:,stat_num],  Winter_21_N36_Tskin[:,stat_num], Spring_21_N36_Tskin[:,stat_num], Summer_21_N36_Tskin[:,stat_num], Fall_21_N36_Tskin[:,stat_num], Winter_21_NMP[:,:,stat_num], Spring_21_NMP[:,:,stat_num], Summer_21_NMP[:,:,stat_num], Fall_21_NMP[:,:,stat_num], Winter_21_NMP_Tskin[:,stat_num], Spring_21_NMP_Tskin[:,stat_num], Summer_21_NMP_Tskin[:,stat_num], Fall_21_NMP_Tskin[:,stat_num], ptitle, fname_out)
    
#
    ptitle='Contour Plot Soil Temperature Difference Profiles for '+Station_Name
    fname_pre=OUT_PATH+'/CountourPlot_Soil_Temp_difference_plots_'+EXP_NAME+'_'+Station_fName

    if (stat_num == 0):
        print (Noah_MPDA_00_Tsoil[0,:,stat_num])
    print ('we are here', stat_num)
    generate_contourf_plots.plot_8_panel(Noah_36_00_Tsoil[:,:,stat_num], Noah_36_03_Tsoil[:,:,stat_num], Noah_36_06_Tsoil[:,:,stat_num], Noah_36_09_Tsoil[:,:,stat_num], Noah_36_12_Tsoil[:,:,stat_num], Noah_36_15_Tsoil[:,:,stat_num], Noah_36_18_Tsoil[:,:,stat_num], Noah_36_21_Tsoil[:,:,stat_num], Noah_36DA_00_Tsoil[:,:,stat_num], Noah_36DA_03_Tsoil[:,:,stat_num], Noah_36DA_06_Tsoil[:,:,stat_num], Noah_36DA_09_Tsoil[:,:,stat_num], Noah_36DA_12_Tsoil[:,:,stat_num], Noah_36DA_15_Tsoil[:,:,stat_num], Noah_36DA_18_Tsoil[:,:,stat_num], Noah_36DA_21_Tsoil[:,:,stat_num],
        Noah_MP_00_Tsoil[:,:,stat_num], Noah_MP_03_Tsoil[:,:,stat_num], Noah_MP_06_Tsoil[:,:,stat_num], Noah_MP_09_Tsoil[:,:,stat_num], Noah_MP_12_Tsoil[:,:,stat_num], Noah_MP_15_Tsoil[:,:,stat_num], Noah_MP_18_Tsoil[:,:,stat_num], Noah_MP_21_Tsoil[:,:,stat_num],
        Noah_MPDA_00_Tsoil[:,:,stat_num], Noah_MPDA_03_Tsoil[:,:,stat_num], Noah_MPDA_06_Tsoil[:,:,stat_num], Noah_MPDA_09_Tsoil[:,:,stat_num], Noah_MPDA_12_Tsoil[:,:,stat_num], Noah_MPDA_15_Tsoil[:,:,stat_num], Noah_MPDA_18_Tsoil[:,:,stat_num], Noah_MPDA_21_Tsoil[:,:,stat_num],
        SCAN_00_Tsoil[:,:,stat_num], SCAN_03_Tsoil[:,:,stat_num], SCAN_06_Tsoil[:,:,stat_num], SCAN_09_Tsoil[:,:,stat_num], SCAN_12_Tsoil[:,:,stat_num], SCAN_15_Tsoil[:,:,stat_num], SCAN_18_Tsoil[:,:,stat_num], SCAN_21_Tsoil[:,:,stat_num], ptitle, fname_pre, tot_num_days, dates_array_00)

print ('done')
# create seasonal averages of the profiles
