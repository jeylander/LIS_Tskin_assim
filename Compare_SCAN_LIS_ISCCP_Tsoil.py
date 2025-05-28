#!/usr/bin/env /opt/local/bin/python3.12

import os
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
import numpy.ma as ma
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


num_years=3
year_array=[2013, 2014, 2015]

Beg_YYYY = year_array[0]
End_yyyy = year_array[num_years-1]


#Beg_YYYY=2013
Beg_mm=1
Beg_dd=1
Beg_HH=0

#End_yyyy=2015
End_mm=12
End_dd=31
End_HH=23

end_DTG=datetime.datetime(End_yyyy, End_mm, End_dd, End_HH)
EDATE='{:%Y%m%d}'.format(end_DTG)
EDATETXT='{:%Y/%m/%d}'.format(end_DTG)
beg_DTG=datetime.datetime(Beg_YYYY, Beg_mm, Beg_dd, Beg_HH)
BGDATE='{:%Y%m%d}'.format(beg_DTG)
BGDATETXT='{:%Y/%m/%d}'.format(beg_DTG)
date_diff=end_DTG-beg_DTG
print ('Date Difference', date_diff.days+1)
num_years = End_yyyy-Beg_YYYY+1
print ('Number of years', num_years)
the_hour_csv_range=(date_diff.days+1)*8
the_pts_array=np.zeros((((date_diff.days+1)*8)+1)*num_years, dtype=np.int32)

EXP_NAME="DA"

N36=True
NMP=False

DataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CENUS_ALL/'
ISMNPATH='/Users/rdcrljbe/Data/ISMN/Configs/'
LISDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CENUS_ALL/'
ISCCPDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CENUS_ALL/'
img_out_path='/Users/rdcrljbe/Data/DIS_test2/output/CENUS_ALL/'

#check output directory to make sure it exists
if not os.path.exists(img_out_path):
    print ('creating output directory')
    os.mkdir(img_out_path)

INC_YEAR=Beg_YYYY

for year_loop in range (0, num_years, 1):

    print ('the year loop number is', year_loop, num_years)
    INC_YEAR = year_array[year_loop]
    print ('the year we are processing is', INC_YEAR)
    STR_YR=str(INC_YEAR)
    
    STR_BDATE=STR_YR+'0101'
    STR_EDATE=STR_YR+'1231'
    
    SCAN_LEVELS=[1,2,3,4]
    LVL=0
    
#  declare all the arrays

    while LVL<=3:
        CURR_SCANfile=DataPath+'ISMN_tsoil_Noah-L'+str(SCAN_LEVELS[LVL])+'_data_by_hour_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        print (DataPath+'ISMN_tsoil_Noah-L'+str(SCAN_LEVELS[LVL])+'_data_by_hour_'+STR_BDATE+'-'+STR_EDATE+'.csv')
        CURR_SCANDATA=pandas.read_csv(CURR_SCANfile, index_col=0)
        num_scan_stations=CURR_SCANDATA.shape[1]
        num_scan_recs=CURR_SCANDATA.shape[0]
        if LVL==0:
            SCAN_DATA_ARRAY_TEMP=np.zeros((4, num_scan_recs, num_scan_stations), dtype=np.float64)
        ii=0
        while ii<=num_scan_stations-2:
            
            SCAN_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_SCANDATA[CURR_SCANDATA.columns[ii]]
            ii+=1
        
        LVL+=1
        
    
    SCAN_DATA_ARRAY_TEMP=np.nan_to_num(SCAN_DATA_ARRAY_TEMP, copy=True, nan=-9999.0)

    if year_loop == 0:
        SCAN_DATA_ARRAY=SCAN_DATA_ARRAY_TEMP
        print ("testing before the concatenation feature", SCAN_DATA_ARRAY.shape, SCAN_DATA_ARRAY_TEMP.shape)
    else:
        SCAN_DATA_ARRAY=np.concatenate((SCAN_DATA_ARRAY, SCAN_DATA_ARRAY_TEMP), axis=1)
        print ("testing the concatenation feature", SCAN_DATA_ARRAY.shape, SCAN_DATA_ARRAY_TEMP.shape)
        
        print ('pre diag', SCAN_DATA_ARRAY_TEMP.shape)
    ################################################
    #  Read in all the Open Loop Data
    ################################################
        
    Noah_Levels=[1,2,3,4]
    LVL=0
    if (N36):
        while LVL<=3:
            CURR_LIS_N36_OL_file=LISDataPath+'LIS_Noah36_L'+str(Noah_Levels[LVL])+'_Tsoil_OL_'+STR_BDATE+'-'+STR_EDATE+'.csv'
            print (CURR_LIS_N36_OL_file)
            CURR_LIS_N36_OL_data=pandas.read_csv(CURR_LIS_N36_OL_file, index_col=0)
            num_lis_stations=CURR_LIS_N36_OL_data.shape[1]
            num_lis_recs=CURR_LIS_N36_OL_data.shape[0]
            if LVL==0:
                LIS_Noah36_OL_DATA_ARRAY_TEMP=np.zeros((4, num_lis_recs, num_lis_stations), dtype=np.float64)
            ii=0
            while ii<=num_lis_stations-2:
                LIS_Noah36_OL_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_LIS_N36_OL_data[CURR_LIS_N36_OL_data.columns[ii+1]]
                ii+=1
            LVL+=1
        if year_loop == 0:
            LIS_Noah36_OL_DATA_ARRAY=LIS_Noah36_OL_DATA_ARRAY_TEMP
        else:
            LIS_Noah36_OL_DATA_ARRAY=np.concatenate((LIS_Noah36_OL_DATA_ARRAY, LIS_Noah36_OL_DATA_ARRAY_TEMP), axis=1)
        
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
    
        ################################################
        #  Read in all the Noah 36 DA Loop Data
        ################################################
    
        Noah_Levels=[1,2,3,4]
        LVL=0
        while LVL<=3:
            CURR_LIS_N36_DA_file=LISDataPath+'LIS_Noah36_L'+str(Noah_Levels[LVL])+'_Tsoil_'+EXP_NAME+'_'+STR_BDATE+'-'+STR_EDATE+'.csv'
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
            
            
    ################################################
    #  IF NoahMP data is available, read it in
    ################################################
            
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
        ################################################
        #  Read in all the DA Loop Data
        ################################################
    
        Noah_Levels=[1,2,3,4]
        LVL=0
        while LVL<=3:
            CURR_LIS_NMP_DA_file=LISDataPath+'LIS_NoahMP_L'+str(Noah_Levels[LVL])+'_Tsoil_'+EXP_NAME+'_'+STR_BDATE+'-'+STR_EDATE+'.csv'
            CURR_MP_DA_data=pandas.read_csv(CURR_LIS_NMP_DA_file, index_col=0)
            num_lis_stations=CURR_MP_DA_data.shape[1]
            num_lis_recs=CURR_MP_DA_data.shape[0]
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
        
        
            
    #####  Read in the Skin Temperature data from the ISCCP files
    
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

print (ISCCP_TSKIN_ARRAY.shape, LIS_Noah36_OL_DATA_ARRAY.shape)

##############################################################
## TSOIL Data Arrays for LIS Noah 36 and Noah MP, Open Loop
##############################################################

if (N36):
    print (the_hour_csv_range, the_hour_csv_range/8)
    LIS_three_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_zero_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_six_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_nine_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_twelve_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_fifteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_eighteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_twtyone_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    
    LIS_three_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_zero_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_six_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_nine_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_twelve_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_fifteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_eighteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    LIS_twtyone_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    
    ##############################################################
    ## TSOIL Data Arrays for LIS Noah 36 and Noah MP, Open Loop
    ##############################################################

    N36_DA_three_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_zero_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_six_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_nine_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_twelve_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_fifteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_eighteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_twtyone_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    
    N36_DA_three_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_zero_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_six_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_nine_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_twelve_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_fifteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_eighteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    N36_DA_twtyone_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)

if (NMP):

    MP_three_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_zero_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_six_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_nine_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_twelve_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_fifteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_eighteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_twtyone_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    
    
    MP_DA_three_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_zero_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_six_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_nine_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_twelve_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_fifteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_eighteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_twtyone_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)

    ##############################################################
    ## TSKIN Data Arrays for LIS Noah 36 and Noah MP, Open Loop
    ##############################################################

    MP_three_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_zero_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_six_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_nine_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_twelve_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_fifteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_eighteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_twtyone_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)

    ##############################################################
    ## TSKIN Data Arrays for LIS Noah 36 and Noah MP, DA Loop
    ##############################################################

    MP_DA_three_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_zero_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_six_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_nine_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_twelve_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_fifteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_eighteen_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
    MP_DA_twtyone_array_skintemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)

##############################################################
## TSKIN Data Arrays for station observation locations
##############################################################

SCAN_three_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_zero_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_six_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_nine_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_twelve_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_fifteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_eighteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
SCAN_twtyone_array_soiltemp = np.zeros((4,int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)

##############################################################
## TSKIN Data Arrays for ISCCP locations
##############################################################

ISSCP_three_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_zero_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_six_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_nine_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_twelve_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_fifteen_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_eighteen_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)
ISSCP_twtyone_array_soiltemp = np.zeros((int(the_hour_csv_range/8)+1, num_scan_stations), dtype=np.float64)

the_zero_hour_csv_index=0
the_three_hour_csv_index=0
the_six_hour_csv_index=0
the_nine_hour_csv_index=0
the_twlv_hour_csv_index=0
the_fftn_hour_csv_index=0
the_eithn_hour_csv_index=0
the_twtone_hour_csv_index=0

points=0
curr_dtg=beg_DTG
while curr_dtg <= end_DTG:

    The_YYYY=curr_dtg.year
    The_MM=curr_dtg.month
    The_DD=curr_dtg.day
    The_HH=curr_dtg.hour

    The_YYYY_str=str(The_YYYY)
    The_MM_str=str(The_MM)
    The_DD_str=str(The_DD)
    The_HH_str=str(The_HH)

    if points == 0:
        dtg_array=datetime.datetime(The_YYYY,The_MM,The_DD, The_HH)
    else:
        dtg_array=np.append(dtg_array, datetime.datetime(The_YYYY,The_MM,The_DD, The_HH))

    curr_dtg=curr_dtg+datetime.timedelta(hours=3)
    the_pts_array[points]=points

    if The_HH == 0 :
        if (N36):
            LIS_zero_array_soiltemp[:,the_zero_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            N36_DA_zero_array_soiltemp[:,the_zero_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            LIS_zero_array_skintemp[the_zero_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_zero_array_skintemp[the_zero_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]
        
        if (NMP):
            MP_zero_array_soiltemp[:,the_zero_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_DA_zero_array_soiltemp[:,the_zero_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_zero_array_skintemp[the_zero_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_zero_array_skintemp[the_zero_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_zero_array_soiltemp[:,the_zero_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_zero_array_soiltemp[the_zero_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if (the_zero_hour_csv_index == 0):
            dates_array_00=datetime.date(The_YYYY, The_MM, The_DD)
        else:
            dates_array_00=np.append(dates_array_00, datetime.date(The_YYYY, The_MM, The_DD))
            
        the_zero_hour_csv_index+=1

    elif The_HH == 3:
        if (N36):
            LIS_three_array_soiltemp[:,the_three_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            N36_DA_three_array_soiltemp[:,the_three_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            LIS_three_array_skintemp[the_three_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_three_array_skintemp[the_three_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]
            
        if (NMP):
            MP_three_array_soiltemp[:,the_three_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_DA_three_array_soiltemp[:,the_three_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_three_array_skintemp[the_three_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_three_array_skintemp[the_three_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_three_array_soiltemp[:,the_three_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_three_array_soiltemp[the_three_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_three_hour_csv_index+=1

    elif The_HH == 6:
        if (N36):
            LIS_six_array_soiltemp[:,the_six_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            LIS_six_array_skintemp[the_six_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_six_array_soiltemp[:,the_six_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            N36_DA_six_array_skintemp[the_six_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]

        if (NMP):
            MP_six_array_soiltemp[:,the_six_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_six_array_skintemp[the_six_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_six_array_skintemp[the_six_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
            MP_DA_six_array_soiltemp[:,the_six_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
        
        SCAN_six_array_soiltemp[:,the_six_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_six_array_soiltemp[the_six_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_six_hour_csv_index+=1

    elif The_HH == 9:
        if (N36):
            LIS_nine_array_soiltemp[:,the_nine_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            LIS_nine_array_skintemp[the_nine_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_nine_array_soiltemp[:,the_nine_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            N36_DA_nine_array_skintemp[the_nine_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]

        if (NMP):
            MP_nine_array_soiltemp[:,the_nine_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_DA_nine_array_soiltemp[:,the_nine_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_nine_array_skintemp[the_nine_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_nine_array_skintemp[the_nine_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_nine_array_soiltemp[:,the_nine_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_nine_array_soiltemp[the_nine_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_nine_hour_csv_index+=1

    elif The_HH == 12:
        if (N36):
            LIS_twelve_array_soiltemp[:,the_twlv_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            LIS_twelve_array_skintemp[the_twlv_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_twelve_array_skintemp[the_twlv_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]
            N36_DA_twelve_array_soiltemp[:,the_twlv_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
       
       
        if (NMP):
            MP_twelve_array_soiltemp[:,the_twlv_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_twelve_array_skintemp[the_twlv_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_twelve_array_soiltemp[:,the_twlv_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_DA_twelve_array_skintemp[the_twlv_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_twelve_array_soiltemp[:,the_twlv_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_twelve_array_soiltemp[the_twlv_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_twlv_hour_csv_index+=1

    elif The_HH == 15:
        if (N36):
            LIS_fifteen_array_soiltemp[:,the_fftn_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            LIS_fifteen_array_skintemp[the_fftn_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_fifteen_array_soiltemp[:,the_fftn_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            N36_DA_fifteen_array_skintemp[the_fftn_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]

        if (NMP):
            MP_fifteen_array_soiltemp[:,the_fftn_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_DA_fifteen_array_soiltemp[:,the_fftn_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_fifteen_array_skintemp[the_fftn_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_fifteen_array_skintemp[the_fftn_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_fifteen_array_soiltemp[:,the_fftn_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_fifteen_array_soiltemp[the_fftn_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_fftn_hour_csv_index+=1

    elif The_HH == 18:
        if (N36):
            LIS_eighteen_array_soiltemp[:,the_eithn_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            N36_DA_eighteen_array_soiltemp[:,the_eithn_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            N36_DA_eighteen_array_skintemp[the_eithn_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]
            LIS_eighteen_array_skintemp[the_eithn_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
        
        if (NMP):
            MP_eighteen_array_soiltemp[:,the_eithn_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_DA_eighteen_array_soiltemp[:,the_eithn_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_eighteen_array_skintemp[the_eithn_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_eighteen_array_skintemp[the_eithn_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_eighteen_array_soiltemp[:,the_eithn_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_eighteen_array_soiltemp[the_eithn_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_eithn_hour_csv_index+=1

    elif The_HH == 21:
        if (N36):
            LIS_twtyone_array_soiltemp[:,the_twtone_hour_csv_index,:]=LIS_Noah36_OL_DATA_ARRAY[:,points,:]
            LIS_twtyone_array_skintemp[the_twtone_hour_csv_index,:]=LIS_N36_TSKIN_OL_ARRAY[points,:]
            N36_DA_twtyone_array_soiltemp[:,the_twtone_hour_csv_index,:]=LIS_Noah36_DA_DATA_ARRAY[:,points,:]
            N36_DA_twtyone_array_skintemp[the_twtone_hour_csv_index,:]=LIS_N36_TSKIN_DA_ARRAY[points,:]

        if (NMP):
            MP_twtyone_array_soiltemp[:,the_twtone_hour_csv_index,:]=LIS_NoahMP_OL_DATA_ARRAY[:,points,:]
            MP_DA_twtyone_array_soiltemp[:,the_twtone_hour_csv_index,:]=LIS_NoahMP_DA_DATA_ARRAY[:,points,:]
            MP_twtyone_array_skintemp[the_twtone_hour_csv_index,:]=LIS_MP_TSKIN_OL_ARRAY[points,:]
            MP_DA_twtyone_array_skintemp[the_twtone_hour_csv_index,:]=LIS_MP_TSKIN_DA_ARRAY[points,:]
        
        SCAN_twtyone_array_soiltemp[:,the_twtone_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_twtyone_array_soiltemp[the_twtone_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_twtone_hour_csv_index+=1

    points+=1

index_array_cycles=np.arange(the_hour_csv_range)
print ('index_array_cycles', the_hour_csv_range, index_array_cycles)

#  Generate graphs of the SCAN vs. Noah 36 Open Loop datasets
figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))

if (N36):
    print ('diag here', SCAN_DATA_ARRAY.shape, num_scan_recs, LIS_Noah36_OL_DATA_ARRAY.shape)
    #temp_array_y=SCAN_DATA_ARRAY[0,0:num_scan_recs,:]
    temp_array_y=SCAN_DATA_ARRAY[0,:,:]
    temp_array_x=LIS_Noah36_OL_DATA_ARRAY[0,:,:]
    
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 1 r value ...', r_value)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(LIS_Noah36_OL_DATA_ARRAY[0,:,:], SCAN_DATA_ARRAY[0,:,:], marker='o')
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].set_xlabel('Noah 3.6 Open Loop Layer 1 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='12')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    #compute the SCAN layer-2 comparisons to LIS
    
    #SCAN_L2=(SCAN_DATA_ARRAY[1,:,:] + SCAN_DATA_ARRAY[2,:,:])/2
    SCAN_L2=SCAN_DATA_ARRAY[1,:,:]
    temp_array_y=SCAN_L2[:,:]
    temp_array_x=LIS_Noah36_OL_DATA_ARRAY[1,:,:]
    print (temp_array_x.shape, temp_array_y.shape, np.max(temp_array_x), np.max(temp_array_y), np.max(SCAN_DATA_ARRAY[0,:,:]), SCAN_DATA_ARRAY.shape)
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 2 r value ...', r_value)
    
    #fig, ax=plt.subplots(constrained_layout=True, figsize=(15,10))
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(LIS_Noah36_OL_DATA_ARRAY[1,:,:], SCAN_L2[:,:], marker='o')
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].set_xlabel('Noah 3.6 Open Loop Layer 2 (10-40 cm) Soil Temperatures', labelpad=5, fontsize='10')
    axes[1,0].set_ylabel('ISMN Avg 4,8-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    #compute the SCAN layer-2 comparisons to LIS
    
    #SCAN_L3=(SCAN_DATA_ARRAY[3,:,:] + SCAN_DATA_ARRAY[4,:,:])/2
    SCAN_L3=SCAN_DATA_ARRAY[2,:,:]
    temp_array_y=SCAN_L3[:,:]
    temp_array_x=LIS_Noah36_OL_DATA_ARRAY[2,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 3 r value ...', r_value)
    
    #fig, ax=plt.subplots(constrained_layout=True, figsize=(15,10))
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(LIS_Noah36_OL_DATA_ARRAY[2,:,:], SCAN_L3[:,:], marker='o')
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].set_xlabel('Noah 3.6 Open Loop Layer 3 (40-100 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[0,1].set_title('LIS Layer 3 (40-100 cm) Soil Temperatures',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN Avg 20,40-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    
    # Layer 4 LIS/NOAH versus 40-inch SCAN
    #temp_array_y=SCAN_DATA_ARRAY[4,0:num_scan_recs,:]
    temp_array_y=SCAN_DATA_ARRAY[3,:,:]
    temp_array_x=LIS_Noah36_OL_DATA_ARRAY[3,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 4 r value ...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(LIS_Noah36_OL_DATA_ARRAY[3,:,:], SCAN_DATA_ARRAY[3,:,:], marker='o')
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].set_xlabel('Noah 3.6 Open Loop Layer 4 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[1,1].set_title('LIS Layer 4 (0-10 cm) Soil Temperatures',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 40-Inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[1,1].set_xticklabels(rotation=70)
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah 3.6 Open Loop vs. ISMN Soil Temperature Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_36_OL_vs_ISMN_Data'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)

    ################################################################################
    #  Generate graphs of the SCAN vs. Noah 36 Data Assimilation Loop datasets
    ################################################################################
    
    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    
    temp_array_y=SCAN_DATA_ARRAY[0,:,:]
    temp_array_x=LIS_Noah36_DA_DATA_ARRAY[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 1 r value ...', r_value)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(LIS_Noah36_DA_DATA_ARRAY[0,:,:], SCAN_DATA_ARRAY[0,:,:], marker='o')
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].set_xlabel('Noah 3.6 DA Loop Layer 1 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='12')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #compute the SCAN layer-2 comparisons to LIS
    
    #SCAN_L2=(SCAN_DATA_ARRAY[1,:,:] + SCAN_DATA_ARRAY[2,:,:])/2
    SCAN_L2=SCAN_DATA_ARRAY[1,:,:]
    temp_array_y=SCAN_L2[:,:]
    temp_array_x=LIS_Noah36_DA_DATA_ARRAY[1,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 2 r value ...', r_value)
    
    #fig, ax=plt.subplots(constrained_layout=True, figsize=(15,10))
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(LIS_Noah36_DA_DATA_ARRAY[1,:,:], SCAN_L2[:,:], marker='o')
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].set_xlabel('Noah 3.6 DA Loop Layer 2 (10-40 cm) Soil Temperatures', labelpad=5, fontsize='10')
    axes[1,0].set_ylabel('ISMN Avg 4,8-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    #compute the SCAN layer-2 comparisons to LIS
    
    #SCAN_L3=(SCAN_DATA_ARRAY[3,:,:] + SCAN_DATA_ARRAY[4,:,:])/2
    SCAN_L3=SCAN_DATA_ARRAY[2,:,:]
    temp_array_y=SCAN_L3[:,:]
    temp_array_x=LIS_Noah36_DA_DATA_ARRAY[2,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 3 r value ...', r_value)
    
    #fig, ax=plt.subplots(constrained_layout=True, figsize=(15,10))
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(LIS_Noah36_DA_DATA_ARRAY[2,:,:], SCAN_L3[:,:], marker='o')
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].set_xlabel('Noah 3.6 DA Loop Layer 3 (40-100 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[0,1].set_title('LIS Layer 3 (40-100 cm) Soil Temperatures',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN Avg 20,40-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    
    # Layer 4 LIS/NOAH versus 40-inch SCAN
    #temp_array_y=SCAN_DATA_ARRAY[4,:,:]
    temp_array_y=SCAN_DATA_ARRAY[3,:,:]
    temp_array_x=LIS_Noah36_DA_DATA_ARRAY[3,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 4 r value ...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(LIS_Noah36_DA_DATA_ARRAY[3,:,:], SCAN_DATA_ARRAY[3,:,:], marker='o')
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].set_xlabel('Noah 3.6 DA Loop Layer 4 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[1,1].set_title('LIS Layer 4 (0-10 cm) Soil Temperatures',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 40-Inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[1,1].set_xticklabels(rotation=70)
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah 3.6 DA Loop vs. ISMN Soil Temperature Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_36_'+EXP_NAME+'_vs_ISMN_Data'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)


if (NMP):
    ################################################################################
    #  Generate graphs of the SCAN vs. Noah MP Open Loop datasets
    ################################################################################
    
    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    temp_array_y=SCAN_DATA_ARRAY[0,:,:]
    temp_array_x=LIS_NoahMP_OL_DATA_ARRAY[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 1 r value ...', r_value)
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(LIS_NoahMP_OL_DATA_ARRAY[0,:,:], SCAN_DATA_ARRAY[0,:,:], marker='o')
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].set_xlabel('NoahMP Open Loop Layer 1 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[0,0].set_xticklabels(rotation=70)
    
    #compute the SCAN layer-2 comparisons to LIS
    
    #fig, ax=plt.subplots(constrained_layout=True, figsize=(15,10))
    SCAN_L2=SCAN_DATA_ARRAY[1,:,:]
    temp_array_y=SCAN_L2[:,:]
    temp_array_x=LIS_NoahMP_OL_DATA_ARRAY[1,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 2 r value ...', r_value)
    
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(LIS_NoahMP_OL_DATA_ARRAY[1,:,:], SCAN_L2[:,:], marker='o')
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].set_xlabel('NoahMP Open Loop Layer 2 (10-40 cm) Soil Temperatures', labelpad=5, fontsize='10')
    axes[1,0].set_ylabel('ISMN Avg 4,8-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    #SCAN_L3=(SCAN_DATA_ARRAY[3,:,:] + SCAN_DATA_ARRAY[4,:,:])/2
    SCAN_L3=SCAN_DATA_ARRAY[2,:,:]
    temp_array_y=SCAN_L3[:,:]
    temp_array_x=LIS_NoahMP_OL_DATA_ARRAY[2,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 3 r value ...', r_value)
    
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(LIS_NoahMP_OL_DATA_ARRAY[2,:,:], SCAN_L3[:,:], marker='o')
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].set_xlabel('NoahMP Open Loop Layer 3 (40-100 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[0,1].set_title('LIS Layer 3 (40-100 cm) Soil Temperatures',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN Avg 20,40-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    # Layer 4 LIS/NOAH versus 40-inch SCAN
    #temp_array_y=SCAN_DATA_ARRAY[4,:,:]
    temp_array_y=SCAN_DATA_ARRAY[3,:,:]
    temp_array_x=LIS_NoahMP_OL_DATA_ARRAY[3,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 4 r value ...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(LIS_NoahMP_OL_DATA_ARRAY[3,:,:], SCAN_DATA_ARRAY[3,:,:], marker='+')
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].set_xlabel('NoahMP Open Loop Layer 4 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[1,1].set_title('LIS Layer 4 (0-10 cm) Soil Temperatures',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 40-Inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah MP Open Loop vs. ISMN Soil Temperature Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_MP_OL_vs_ISMN_Data'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)
    
    ################################################################################
    #  Generate graphs of the SCAN vs. Noah MP DA Loop datasets
    ################################################################################
    
    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    temp_array_y=SCAN_DATA_ARRAY[0,:,:]
    temp_array_x=LIS_NoahMP_DA_DATA_ARRAY[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 1 r value ...', r_value)
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(LIS_NoahMP_DA_DATA_ARRAY[0,:,:], SCAN_DATA_ARRAY[0,:,:], marker='o')
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].set_xlabel('NoahMP DA Loop Layer 1 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[0,0].set_xticklabels(rotation=70)
    
    #compute the SCAN layer-2 comparisons to LIS
    
    #fig, ax=plt.subplots(constrained_layout=True, figsize=(15,10))
    SCAN_L2=SCAN_DATA_ARRAY[1,:,:]
    temp_array_y=SCAN_L2[:,:]
    temp_array_x=LIS_NoahMP_DA_DATA_ARRAY[1,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 2 r value ...', r_value)
    
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(LIS_NoahMP_DA_DATA_ARRAY[1,:,:], SCAN_L2[:,:], marker='o')
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].set_xlabel('NoahMP DA Loop Layer 2 (10-40 cm) Soil Temperatures', labelpad=5, fontsize='10')
    axes[1,0].set_ylabel('ISMN Avg 4,8-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    #SCAN_L3=(SCAN_DATA_ARRAY[3,:,:] + SCAN_DATA_ARRAY[4,:,:])/2
    SCAN_L3=SCAN_DATA_ARRAY[2,:,:]
    temp_array_y=SCAN_L3[:,:]
    temp_array_x=LIS_NoahMP_DA_DATA_ARRAY[2,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 3 r value ...', r_value)
    
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(LIS_NoahMP_DA_DATA_ARRAY[2,:,:], SCAN_L3[:,:], marker='o')
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].set_xlabel('NoahMP DA Loop Layer 3 (40-100 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[0,1].set_title('LIS Layer 3 (40-100 cm) Soil Temperatures',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN Avg 20,40-inch Probe Temperature (K)',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    # Layer 4 LIS/NOAH versus 40-inch SCAN
    #temp_array_y=SCAN_DATA_ARRAY[4,:,:]
    temp_array_y=SCAN_DATA_ARRAY[3,:,:]
    temp_array_x=LIS_NoahMP_DA_DATA_ARRAY[3,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=250.0) & (temp_array_y <= 350.0) & (temp_array_x > 250.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('the Layer 4 r value ...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(LIS_NoahMP_DA_DATA_ARRAY[3,:,:], SCAN_DATA_ARRAY[3,:,:], marker='+')
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].set_xlabel('NoahMP DA Loop Layer 4 (0-10 cm) Soil Temperatures', labelpad=5, fontsize='10')
    #axes[1,1].set_title('LIS Layer 4 (0-10 cm) Soil Temperatures',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 40-Inch Probe Temperature (K) ',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah MP DA Loop vs. ISMN Soil Temperature Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_MP_'+EXP_NAME+'_vs_ISMN_Data'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)


if (N36):

    ##################################################################################################################
    #####      NOAH 36 OPEN LOOP VERSUS OBSERVATIONS
    #####
    ##################################################################################################################
    ##################################################################################################################
    #####  PLOT THE 00 HOUR Top Layer COMPARISON Noah 36 for OPEN LOOP
    ##################################################################################################################
    
    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    temp_array_y=SCAN_zero_array_soiltemp[0,:,:]
    temp_array_x=LIS_zero_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 00 HR analysis is...', r_value)
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(LIS_zero_array_soiltemp[0,:,:], SCAN_zero_array_soiltemp[0,:,:], marker='+')
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].set_xlabel('Noah 3.6 Layer 1 (0-10 cm) Soil Temperatures 00 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 00 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[0,0].set_xticklabels(rotation=70)
    
    
    ##################################################################################################################
    #####  PLOT THE 06 HOUR Top Layer COMPARISON Noah 36 for OPEN LOOP
    ##################################################################################################################
    
    temp_array_y=SCAN_six_array_soiltemp[0,:,:]
    temp_array_x=LIS_six_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 06 HR analysis is...', r_value)
    
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(LIS_six_array_soiltemp[0,:,:], SCAN_six_array_soiltemp[0,:,:], marker='+')
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].set_xlabel('Noah 3.6 Layer 1 (0-10 cm) Soil Temperatures 06 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 06 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 12 HOUR Top Layer COMPARISON Noah 36 for OPEN LOOP
    ##################################################################################################################
    
    temp_array_y=SCAN_twelve_array_soiltemp[0,:,:]
    temp_array_x=LIS_twelve_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 12 HR analysis is...', r_value)
    
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(LIS_twelve_array_soiltemp[0,:,:], SCAN_twelve_array_soiltemp[0,:,:], marker='+')
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].set_xlabel('Noah 3.6 Layer 1 (0-10 cm) Soil Temperatures 12 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 12 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 18 HOUR Top Layer COMPARISON Noah 36 for OPEN LOOP
    ##################################################################################################################
    
    temp_array_y=SCAN_eighteen_array_soiltemp[0,:,:]
    temp_array_x=LIS_eighteen_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value ...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(LIS_eighteen_array_soiltemp[0,:,:], SCAN_eighteen_array_soiltemp[0,:,:], marker='+')
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].set_xlabel('Noah 3.6 Layer 1 (0-10 cm) Soil Temperatures 18 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 18 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah 3.6 Open Loop 0-10 cm vs. ISMN \n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_36_TOP_OL_vs_ISMN_Data_cycles'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)


##################################################################################################################
#####      NOAH 36 DATA ASSIMILATION GROUP VERSUS OBSERVATIONS
#####
##################################################################################################################
##################################################################################################################
#####  PLOT THE 00 HOUR Top Layer COMPARISON Noah 36 for DA LOOP
##################################################################################################################

    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    temp_array_y=SCAN_zero_array_soiltemp[0,:,:]
    temp_array_x=N36_DA_zero_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value Noah 36 DA Loop 00 HR..', r_value)
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(N36_DA_zero_array_soiltemp[0,:,:], SCAN_zero_array_soiltemp[0,:,:], marker='+')
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].set_xlabel('Noah 3.6 DA Loop Layer 1 (0-10 cm) Soil Temperatures 00 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 00 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[0,0].set_xticklabels(rotation=70)
    
    ##################################################################################################################
    #####  PLOT THE 06 HOUR Top Layer COMPARISON Noah 36 for DA LOOP
    ##################################################################################################################
    
    temp_array_y=SCAN_six_array_soiltemp[0,:,:]
    temp_array_x=N36_DA_six_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value Noah 36 DA Loop 06 HR..', r_value)
    
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(N36_DA_six_array_soiltemp[0,:,:], SCAN_six_array_soiltemp[0,:,:], marker='+')
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].set_xlabel('Noah 3.6 DA Loop Layer 1 (0-10 cm) Soil Temperatures 06 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 06 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 12 HOUR Top Layer COMPARISON Noah 36 for DA LOOP
    ##################################################################################################################
    
    temp_array_y=SCAN_twelve_array_soiltemp[0,:,:]
    temp_array_x=N36_DA_twelve_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value Noah 36 DA Loop 12 HR..', r_value)
    
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(N36_DA_twelve_array_soiltemp[0,:,:], SCAN_twelve_array_soiltemp[0,:,:], marker='+')
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].set_xlabel('Noah 3.6 Layer 1 (0-10 cm) Soil Temperatures 12 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 12 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 18 HOUR Top Layer COMPARISON Noah 36 for DA LOOP
    ##################################################################################################################
    
    temp_array_y=SCAN_eighteen_array_soiltemp[0,:,:]
    temp_array_x=N36_DA_eighteen_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value Noah 36 DA Loop 18 HR...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(N36_DA_eighteen_array_soiltemp[0,:,:], SCAN_eighteen_array_soiltemp[0,:,:], marker='+')
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].set_xlabel('Noah 3.6 DA Loop Layer 1 (0-10 cm) Soil Temperatures 18 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 18 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[1,1].legend(loc='upper right', borderaxespad=0.)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah 3.6 0-10 cm vs. ISMN Soil Temperature Data \n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_36_TOPLYR_'+EXP_NAME+'_vs_ISMN_Data_cycles'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)
 

if (NMP):
    ##################################################################################################################
    #####   NOAH-MP OPEN LOOP VERSUS GROUND OBSERVATIONS
    #####
    ##################################################################################################################
    ##################################################################################################################
    #####  PLOT THE 00 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    temp_array_y=SCAN_zero_array_soiltemp[0,:,:]
    temp_array_x=MP_zero_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 00 HR Noah MP Open Loop analysis is...', r_value)
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(MP_zero_array_soiltemp[0,:,:], SCAN_zero_array_soiltemp[0,:,:], marker='+')
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].set_xlabel('Noah MP Layer 1 (0-10 cm) Soil Temperatures 00 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 00 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[0,0].set_xticklabels(rotation=70)
    
    ##################################################################################################################
    #####  PLOT THE 06 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    temp_array_y=SCAN_six_array_soiltemp[0,:,:]
    temp_array_x=MP_six_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 06 HR NOAH MP Open Loop analysis is...', r_value)
    
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(MP_six_array_soiltemp[0,:,:], SCAN_six_array_soiltemp[0,:,:], marker='+')
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].set_xlabel('Noah MP Layer 1 (0-10 cm) Soil Temperatures 06 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 06 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 12 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    temp_array_y=SCAN_twelve_array_soiltemp[0,:,:]
    temp_array_x=MP_twelve_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 12 HR Noah MP Open Loop analysis is...', r_value)
    
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(MP_twelve_array_soiltemp[0,:,:], SCAN_twelve_array_soiltemp[0,:,:], marker='+')
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].set_xlabel('Noah MP Layer 1 (0-10 cm) Soil Temperatures 12 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 12 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 18 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    temp_array_y=SCAN_eighteen_array_soiltemp[0,:,:]
    temp_array_x=MP_eighteen_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 18 HR Noah MP Open Loop analysis is...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(MP_eighteen_array_soiltemp[0,:,:], SCAN_eighteen_array_soiltemp[0,:,:], marker='+')
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].set_xlabel('Noah MP Layer 1 (0-10 cm) Soil Temperatures 18 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 18 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah MP Open Loop 0-10 cm vs. USDA SCAN \n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_MP_TOPLYR_OL_vs_ISMN_Data_cycles'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)
    
    
    ##################################################################################################################
    #####   NOAH-MP DATA ASSIMILATION VERSUS GROUND OBSERVATIONS
    #####
    ##################################################################################################################
    ##################################################################################################################
    #####  PLOT THE 00 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    figure, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    
    temp_array_y=SCAN_zero_array_soiltemp[0,:,:]
    temp_array_x=MP_DA_zero_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 00 HR Noah MP DA analysis is...', r_value)
    
    axes[0,0].set_ylim(240, 340)
    axes[0,0].set_xlim(240, 340)
    axes[0,0].scatter(MP_DA_zero_array_soiltemp[0,:,:], SCAN_zero_array_soiltemp[0,:,:], marker='+')
    axes[0,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,0].set_xlabel('Noah MP DA Loop Layer 1 (0-10 cm) Soil Temperatures 00 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 00 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    #axes[0,0].set_xticklabels(rotation=70)
    
    ##################################################################################################################
    #####  PLOT THE 06 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    temp_array_y=SCAN_six_array_soiltemp[0,:,:]
    temp_array_x=MP_DA_six_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 06 HR NOAH MP analysis is...', r_value)
    
    axes[1,0].set_ylim(240, 340)
    axes[1,0].set_xlim(240, 340)
    axes[1,0].scatter(MP_DA_six_array_soiltemp[0,:,:], SCAN_six_array_soiltemp[0,:,:], marker='+')
    axes[1,0].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,0].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,0].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,0].set_xlabel('Noah MP DA Loop Layer 1 (0-10 cm) Soil Temperatures 06 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,0].set_ylabel('ISMN 2-inch Probe Temperature (K) 06 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,0].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,0].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 12 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    temp_array_y=SCAN_twelve_array_soiltemp[0,:,:]
    temp_array_x=MP_DA_twelve_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 12 HR Noah MP analysis is...', r_value)
    
    axes[0,1].set_ylim(240, 340)
    axes[0,1].set_xlim(240, 340)
    axes[0,1].scatter(MP_DA_twelve_array_soiltemp[0,:,:], SCAN_twelve_array_soiltemp[0,:,:], marker='+')
    axes[0,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[0,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[0,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[0,1].set_xlabel('Noah MP DA Loop Layer 1 (0-10 cm) Soil Temperatures 12 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[0,1].set_ylabel('SCAN 2-inch Probe Temperature (K) 12 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[0,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[0,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    ##################################################################################################################
    #####  PLOT THE 18 HOUR Top Layer COMPARISON
    ##################################################################################################################
    
    temp_array_y=SCAN_eighteen_array_soiltemp[0,:,:]
    temp_array_x=MP_DA_eighteen_array_soiltemp[0,:,:]
    new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
    x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
    y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    regressor=LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred=regressor.predict(X_test)
    
    model = sm.OLS(y_pred, y_test)
    thersquared=str(round(model.fit().rsquared,5))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('RMS Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    new_x=x.reshape(x.shape[0])
    print (new_x.shape, y.shape)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_x, y)
    print ('this r value for the 18 HR Noah MP analysis is...', r_value)
    
    axes[1,1].set_ylim(240, 340)
    axes[1,1].set_xlim(240, 340)
    axes[1,1].scatter(MP_DA_eighteen_array_soiltemp[0,:,:], SCAN_eighteen_array_soiltemp[0,:,:], marker='+')
    axes[1,1].plot(X_test, y_pred, color='red', linewidth=2)
    axes[1,1].plot([273.15,273.15],[240,340], '-',color="green")
    axes[1,1].plot([240,340],[273.15,273.15], '-',color="green")
    axes[1,1].set_xlabel('Noah MP DA Loop Layer 1 (0-10 cm) Soil Temperatures 18 HR', labelpad=5, fontsize='10')
    #axes[0,0].set_title('',  fontsize='xx-large')
    axes[1,1].set_ylabel('ISMN 2-inch Probe Temperature (K) 18 HR',labelpad=5, fontsize='10')
    label=str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    axes[1,1].text(241,320, 'RMS Error:'+label, fontsize=12, color='blue')
    axes[1,1].text(241,315, 'R$^2$:'+str(round(r_value,3)), fontsize=12, color='blue')
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.20, wspace=0.25)
    plt.suptitle('LIS Noah MP DA Loop 0-10 cm vs. ISMN \n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_out_path+'LIS_Noah_MP_TOPLYR_'+EXP_NAME+'_vs_ISMN_Data_cycles'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot Top Layer COMPARISON
##################################################################################################################

station_number=0
while station_number < num_scan_stations-1:
    Scat_min=240
    Scat_max=340
    
    min_cor=260
    max_cor=350
    if (N36):
        filtered_x=ma.masked_outside(SCAN_DATA_ARRAY[0,:,station_number],min_cor,350)
        filtered_y_36_OL=ma.masked_outside(LIS_Noah36_OL_DATA_ARRAY[0,:,station_number],min_cor,350)
        filtered_y_36_DA=ma.masked_outside(LIS_Noah36_DA_DATA_ARRAY[0,:,station_number],min_cor,350)
        mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
        mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_DATA_ARRAY[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_DATA_ARRAY[0,:,station_number],mask=mask_36_DA).compressed()
        filtered_y_final_36_OL=ma.masked_array(LIS_Noah36_OL_DATA_ARRAY[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_y_final_36_DA=ma.masked_array(LIS_Noah36_DA_DATA_ARRAY[0,:,station_number],mask=mask_36_DA).compressed()
        
        print (filtered_x_final_OL.shape, filtered_x_final_DA.shape, filtered_y_final_36_OL.shape, filtered_y_final_36_DA.shape, LIS_Noah36_DA_DATA_ARRAY.shape, LIS_Noah36_OL_DATA_ARRAY.shape)
       
        if (filtered_y_final_36_OL.shape[0] > 100):
            slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_36_OL)
            #compute the bias
            filtered_diff=filtered_y_final_36_OL-filtered_x_final_OL
            temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_36_DA.shape[0] > 100):
            slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_36_DA)
            #compute the bias
            filtered_diff=filtered_y_final_36_DA-filtered_x_final_DA
            temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    
    if (NMP):
        filtered_x=ma.masked_outside(SCAN_DATA_ARRAY[0,:,station_number],min_cor,350)
        filtered_y_MP_OL=ma.masked_outside(LIS_NoahMP_OL_DATA_ARRAY[0,:,station_number],min_cor,350)
        filtered_y_MP_DA=ma.masked_outside(LIS_NoahMP_DA_DATA_ARRAY[0,:,station_number],min_cor,350)
        mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
        mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_DATA_ARRAY[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_DATA_ARRAY[0,:,station_number],mask=mask_MP_DA).compressed()
        filtered_y_final_MP_OL=ma.masked_array(LIS_NoahMP_OL_DATA_ARRAY[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_y_final_MP_DA=ma.masked_array(LIS_NoahMP_DA_DATA_ARRAY[0,:,station_number],mask=mask_MP_DA).compressed()
        
        if (filtered_y_final_MP_OL.shape[0] > 100):
            slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_MP_OL)
            #compute the bias
            filtered_diff=filtered_y_final_MP_OL-filtered_x_final_OL
            temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_MP_DA.shape[0] > 100):
            slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_MP_DA)
            #compute the bias
            filtered_diff=filtered_y_final_MP_DA-filtered_x_final_DA
            temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    
    fig=plt.figure(figsize=(15,12)) #axes=plt.subplots(nrows=4, figsize=(15,12))
    ax1=fig.add_subplot(211)
    ax1.set_ylim(Scat_min, Scat_max)
    ax1.set_xlim(0,the_hour_csv_range-1)
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    
    if (N36):
        ax1.plot(index_array_cycles, LIS_Noah36_OL_DATA_ARRAY[0,:,station_number], marker='+', color='c', label="LIS 3.6 OL Data", markersize=3)
        ax1.plot(index_array_cycles, LIS_Noah36_DA_DATA_ARRAY[0,:,station_number], marker='.', color='r', alpha=.5, label="LIS 3.6 DA Data", markersize=3)
        ax1.plot(index_array_cycles, ISCCP_TSKIN_ARRAY[:,station_number], lw=0.5, linestyle='-', color='y', label="ISCCP Data")
        ax1.scatter(index_array_cycles, SCAN_DATA_ARRAY[0,:,station_number], color='b', label="ISMN Data", marker='.', s=2)
        ax1.legend(loc='upper right', borderaxespad=0.)
        ax1.set_xlabel('date')
        ax1.set_ylabel('Temp (K)')
        if (filtered_y_final_36_OL.shape[0] > 100):
            ax1.text(30, 330, "N36 OL Bias: "+str(round(temp_bias_36_OL,1))+"  RMS: "+str(round(r_value_36_OL,1)), ha="left", va="center", size=10, bbox=bbox_props)
        if (filtered_y_final_36_DA.shape[0] > 100):
            ax1.text(30, 325, "N36 DA Bias:"+str(round(temp_bias_36_DA,1))+"  RMS: "+str(round(r_value_36_DA,1)), ha="left", va="center", size=10, bbox=bbox_props)
     
        ax1_ins = inset_axes(ax1, width="100%", height="10%", loc="lower left", borderpad=0)
        secay1 = ax1_ins.secondary_yaxis('right')
        ax1_ins.set_ylim(-5, 5)
        ax1_ins.set_xlim(0,the_hour_csv_range-1)
        ax1_ins.plot(index_array_cycles, LIS_Noah36_DA_DATA_ARRAY[0,:,station_number] - LIS_Noah36_OL_DATA_ARRAY[0,:,station_number] ,   marker='+', color='black', label="N36 DA - OL Data", markersize=2)
        ax1_ins.axes.yaxis.set_visible(False)
        ax1_ins.axes.xaxis.set_visible(False)
        ax1_ins.axis('off')
        secay1.set_ylabel('Diff (K)',labelpad=5, fontsize='large')
        ax1.set_xticks(index_array_cycles[::300])
        ax1.set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in dtg_array[::300]], rotation=45, fontsize=7.0)
   
    if (NMP):
        ax2=fig.add_subplot(212)
        ax2.set_ylim(240, 340)
        ax2.set_xlim(0,the_hour_csv_range-1)
        ax2.plot(index_array_cycles, LIS_NoahMP_OL_DATA_ARRAY[0,:,station_number], marker='+', color='c', label="MP OL Data", markersize=3)
        ax2.plot(index_array_cycles, LIS_NoahMP_DA_DATA_ARRAY[0,:,station_number], marker='.', color='r', alpha=.5, label="MP DA Data", markersize=3)
        ax2.plot(index_array_cycles, ISCCP_TSKIN_ARRAY[:,station_number], lw=0.5, linestyle='-', color='y', label="ISCCP Data")
        ax2.scatter(index_array_cycles, SCAN_DATA_ARRAY[0,:,station_number], color='b', marker='.', s=2, label="ISMN Data")
        ax2.legend(loc='upper right', borderaxespad=0.)
        ax2.set_xlabel('date')
        ax2.set_ylabel('Temp (K)')
        if (filtered_y_final_MP_OL.shape[0] > 100):
            ax2.text(30, 330, "MP OL Bias:"+str(round(temp_bias_MP_OL,1))+"  RMS: "+str(round(r_value_MP_OL,1)), ha="left", va="center", size=10, bbox=bbox_props)
        if (filtered_y_final_MP_DA.shape[0] > 100):
            ax2.text(30, 325, "MP DA Bias:"+str(round(temp_bias_MP_DA,1))+"  RMS: "+str(round(r_value_MP_DA,1)), ha="left", va="center", size=10, bbox=bbox_props)
        
        inset_bounds = [0,0,index_array_cycles,Scat_max]
        ax2_ins = inset_axes(ax2, width="100%", height="10%", loc="lower left", borderpad=0)
        secay2 = ax2_ins.secondary_yaxis('right')
        ax2_ins.set_ylim(-5, 5)
        ax2_ins.set_xlim(0,the_hour_csv_range-1)
        ax2_ins.plot(index_array_cycles, LIS_NoahMP_DA_DATA_ARRAY[0,:,station_number] - LIS_NoahMP_OL_DATA_ARRAY[0,:,station_number], marker='+',  color='black', label="MP DA - OL Data", markersize=2)
        ax2_ins.axes.yaxis.set_visible(False)
        ax2_ins.axes.xaxis.set_visible(False)
        ax2_ins.axis('off')
        secay2.set_ylabel('Diff (K)',labelpad=5, fontsize='large')
        ax2.set_xticks(index_array_cycles[::300])
        ax2.set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in dtg_array[::300]], rotation=45, fontsize=7.0)
        
    img_fname_pre=img_out_path+CURR_SCANDATA.columns[station_number]
    
    if (N36 and NMP):
        plt.suptitle(str(CURR_SCANDATA.columns[station_number])+' LIS Noah & NoahMP Open Loop vs. ISMN Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    elif (N36):
        plt.suptitle(str(CURR_SCANDATA.columns[station_number])+' LIS Noah Open Loop vs. ISMN Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    elif (NMP):
        plt.suptitle(str(CURR_SCANDATA.columns[station_number])+' LIS NoahMP Open Loop vs. ISMN Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    
    plt.savefig(img_fname_pre+'_LIS_Noah_36-MP_TOPLYR_OL-'+EXP_NAME+'_vs_ISMN_Data_year'+BGDATE+'-'+EDATE+'.png')
    plt.close(fig)
    station_number+=1

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot Top Layer COMPARISON
##################################################################################################################

station_number=0
while station_number < num_scan_stations-1:
    print ('Station number =', station_number, 'number of stations total =', num_scan_stations)
    Scat_min=240
    Scat_max=340
    
    min_cor=260
    max_cor=350
##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
##################################################################################################################
    if (N36):
        filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_OL=ma.masked_outside(LIS_zero_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_DA=ma.masked_outside(N36_DA_zero_array_soiltemp[0,:,station_number],min_cor,350)
        mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
        mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
        filtered_y_final_36_OL=ma.masked_array(LIS_zero_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_y_final_36_DA=ma.masked_array(N36_DA_zero_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
    
        if (filtered_y_final_36_OL.shape[0] > 100):
            slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_36_OL)
            #compute the bias
            filtered_diff=filtered_y_final_36_OL-filtered_x_final_OL
            temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_36_DA.shape[0] > 100):
            slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_36_DA)
            #compute the bias
            filtered_diff=filtered_y_final_36_DA-filtered_x_final_DA
            temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    
    if (NMP):
        filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_OL=ma.masked_outside(MP_zero_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_DA=ma.masked_outside(MP_DA_zero_array_soiltemp[0,:,station_number],min_cor,350)
        mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
        mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_y_final_MP_OL=ma.masked_array(MP_zero_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_y_final_MP_DA=ma.masked_array(MP_DA_zero_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
    
        if (filtered_y_final_MP_OL.shape[0] > 100):
            slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_MP_OL)
            #compute the bias
            filtered_diff=filtered_y_final_MP_OL-filtered_x_final_OL
            temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_MP_DA.shape[0] > 100):
            slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_MP_DA)
            #compute the bias
            filtered_diff=filtered_y_final_MP_DA-filtered_x_final_DA
            temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    
    
    num_days_to_plot =int(the_hour_csv_range/8.0)+1
    print (num_days_to_plot)
    num_days_index = np.arange(num_days_to_plot)
    print (num_days_index.shape)
    figure=plt.figure(figsize=(12,12), )
    ax1=figure.add_subplot(411)
    figure.subplots_adjust(right=0.8)
    ax1.set_ylim(240, 350)
    ax1.set_xlim(0,num_days_to_plot-1)
    ax1.plot(num_days_index, SCAN_zero_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dashed', label='ISMN 2-in')
    if (N36):
        ax1.plot(num_days_index, LIS_zero_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah 3.6 LY1')
        ax1.plot(num_days_index, N36_DA_zero_array_soiltemp[0,:,station_number], lw=0.5, linestyle='dotted', label='Noah 3.6 DA LY1')
    
    if (NMP):
        ax1.plot(num_days_index, MP_zero_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah MP LY1')
        ax1.plot(num_days_index, MP_DA_zero_array_soiltemp[0,:,station_number], lw=0.5, linestyle='dotted', label='Noah MP DA LY1')
   
    ax1.plot(num_days_index, ISSCP_zero_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    ax1.grid()
    ax1.set_xticks(num_days_index[::90])
    ax1.set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in dates_array_00[::90]], rotation=45, fontsize=7.0)
    ax1.set_ylabel('Temp (K)')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    ax1.text(10, 330, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    ax1.text(-0.1, 0.9, 'A)', transform=ax1.transAxes, size=20, weight='bold')
    if (N36):
        ax1.text(1.08, 0.1, "N36 OL Bias 00 UTC: "+str(round(temp_bias_36_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax1.transAxes)
        ax1.text(1.08, 0.2, "N36 DA Bias 00 UTC: "+str(round(temp_bias_36_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax1.transAxes)
        ax1.text(1.08, 0.3, "N36 OL RMS 00 UTC: "+str(round(r_value_36_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax1.transAxes)
        ax1.text(1.08, 0.4, "N36 DA RMS 00 UTC: "+str(round(r_value_36_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax1.transAxes)
    if (NMP):
        ax1.text(1.08, .5, "MP OL Bias 00 UTC:"+str(round(temp_bias_MP_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax1.transAxes)
        ax1.text(1.08, .6, "MP DA Bias 00 UTC:"+str(round(temp_bias_MP_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax1.transAxes)
        ax1.text(1.08, .7, "MP OL RMS 00 UTC: "+str(round(r_value_MP_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax1.transAxes)
        ax1.text(1.08, .8, "MP DA RMS 00 UTC: "+str(round(r_value_MP_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax1.transAxes)
    #ax1.legend(loc='upper right', edgecolor="none", borderaxespad=0.)

    if (N36):
        ax1_ins = inset_axes(ax1, width="100%", height="10%", loc="lower left", borderpad=0)
        secay1 = ax1_ins.secondary_yaxis('right')
        ax1_ins.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax1_ins.plot(num_days_index,  N36_DA_zero_array_soiltemp[0,:,station_number] - LIS_zero_array_soiltemp[0,:,station_number], marker='.', color='black', label="N36 DA - OL Data", markersize=2)
        ax1_ins.axes.yaxis.set_visible(False)
        ax1_ins.axes.xaxis.set_visible(False)
        ax1_ins.set_xlim(0,num_days_to_plot-1)
        ax1_ins.spines['top'].set_color('g')
        ax1_ins.spines['right'].set_color('g')
        secay1.tick_params(axis='y', colors='g', labelsize=8)
        ax1_ins.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay1.set_ylabel('N36 OL-DA',labelpad=5, fontsize='small', color='g', ha='left')
    
    if (NMP):
        ax1_ins2 = inset_axes(ax1, width="100%", height="10%", loc="upper left", borderpad=0)
        secay1_2 = ax1_ins2.secondary_yaxis('right')
        ax1_ins2.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax1_ins2.plot(num_days_index, MP_DA_zero_array_soiltemp[0,:,station_number] - MP_zero_array_soiltemp[0,:,station_number], marker='.', color='blue', label="MP DA - OL Data", markersize=2)
        ax1_ins2.axes.yaxis.set_visible(False)
        ax1_ins2.axes.xaxis.set_visible(False)
        ax1_ins2.set_xlim(0,num_days_to_plot-1)
        ax1_ins2.spines['top'].set_color('g')
        ax1_ins2.spines['right'].set_color('g')
        secay1_2.tick_params(axis='y', colors='g', labelsize=8)
        ax1_ins2.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay1_2.set_ylabel('MP OL-DA',labelpad=5, fontsize='small', color='g', ha='center')

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################
    if (N36):
        filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_OL=ma.masked_outside(LIS_six_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_DA=ma.masked_outside(N36_DA_six_array_soiltemp[0,:,station_number],min_cor,350)
        mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
        mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
        filtered_y_final_36_OL=ma.masked_array(LIS_six_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_y_final_36_DA=ma.masked_array(N36_DA_six_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
    
        if (filtered_y_final_36_OL.shape[0] > 100):
            slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_36_OL)
            #compute the bias
            filtered_diff=filtered_y_final_36_OL-filtered_x_final_OL
            temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_36_DA.shape[0] > 100):
            slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_36_DA)
            #compute the bias
            filtered_diff=filtered_y_final_36_DA-filtered_x_final_DA
            temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    if (NMP):
        filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_OL=ma.masked_outside(MP_six_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_DA=ma.masked_outside(MP_DA_six_array_soiltemp[0,:,station_number],min_cor,350)
        mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
        mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
        filtered_y_final_MP_OL=ma.masked_array(MP_six_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_y_final_MP_DA=ma.masked_array(MP_DA_six_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
    
        if (filtered_y_final_MP_OL.shape[0] > 100):
            slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_MP_OL)
            #compute the bias
            filtered_diff=filtered_y_final_MP_OL-filtered_x_final_OL
            temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_MP_DA.shape[0] > 100):
            slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_MP_DA)
            #compute the bias
            filtered_diff=filtered_y_final_MP_DA-filtered_x_final_DA
            temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
            
    ax2=figure.add_subplot(412)
    ax2.set_ylim(240, 350)
    ax2.set_xlim(0,num_days_to_plot-1)
    ax2.plot(num_days_index, SCAN_six_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dashed', label='ISMN 2-in')
    if (N36):
        ax2.plot(num_days_index, LIS_six_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah 3.6 LY1')
        ax2.plot(num_days_index, N36_DA_six_array_soiltemp[0,:,station_number], lw=0.5, linestyle='dotted', label='Noah 3.6 DA LY1')
    
    if (NMP):
        ax2.plot(num_days_index, MP_six_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah MP LY1')
        ax2.plot(num_days_index, MP_DA_six_array_soiltemp[0,:,station_number], lw=0.5, linestyle='dotted', label='Noah MP DA LY1')
        
    ax2.plot(num_days_index, ISSCP_six_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    ax2.grid()
    if (N36):
        ax2.text(1.08, 0.1, "N36 OL Bias 06 UTC: "+str(round(temp_bias_36_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax2.transAxes)
        ax2.text(1.08, 0.2, "N36 DA Bias 06 UTC: "+str(round(temp_bias_36_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax2.transAxes)
        ax2.text(1.08, 0.3, "N36 OL RMS 06 UTC: "+str(round(r_value_36_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax2.transAxes)
        ax2.text(1.08, 0.4, "N36 DA RMS 06 UTC: "+str(round(r_value_36_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax2.transAxes)
    if (NMP):
        ax2.text(1.08, .5, "MP OL Bias 06 UTC:"+str(round(temp_bias_MP_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax2.transAxes)
        ax2.text(1.08, .6, "MP DA Bias 06 UTC:"+str(round(temp_bias_MP_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax2.transAxes)
        ax2.text(1.08, .7, "MP OL RMS 06 UTC: "+str(round(r_value_MP_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax2.transAxes)
        ax2.text(1.08, .8, "MP DA RMS 06 UTC: "+str(round(r_value_MP_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax2.transAxes)
    ax2.set_ylabel('Temp (K)')
    ax2.set_xticks(num_days_index[::90])
    ax2.set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in dates_array_00[::90]], rotation=45, fontsize=7.0)
    ax2.text(-0.1, 0.9, 'B)', transform=ax2.transAxes, size=20, weight='bold')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    ax2.text(10, 330, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    #ax2.legend(loc='upper right', edgecolor="none", borderaxespad=0.)
    if (N36):
        ax2_ins = inset_axes(ax2, width="100%", height="10%", loc="lower left", borderpad=0)
        secay2_1 = ax2_ins.secondary_yaxis('right')
        ax2_ins.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax2_ins.plot(num_days_index, N36_DA_six_array_soiltemp[0,:,station_number] - LIS_six_array_soiltemp[0,:,station_number], marker='.', color='black', label="N36 DA - OL Data", markersize=2)
        ax2_ins.axes.yaxis.set_visible(False)
        ax2_ins.axes.xaxis.set_visible(False)
        ax2_ins.set_xlim(0,num_days_to_plot-1)
        ax2_ins.spines['top'].set_color('g')
        ax2_ins.spines['right'].set_color('g')
        secay2_1.tick_params(axis='y', colors='g', labelsize=8)
        ax2_ins.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay2_1.set_ylabel('OL-DA Diff (K)',labelpad=5, fontsize='small', color='g', ha='left')
    
    if (NMP):
        ax2_ins2 = inset_axes(ax2, width="100%", height="10%", loc="upper left", borderpad=0)
        secay2_2 = ax2_ins2.secondary_yaxis('right')
        ax2_ins2.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax2_ins2.plot(num_days_index, MP_DA_six_array_soiltemp[0,:,station_number] - MP_six_array_soiltemp[0,:,station_number], marker='.', color='blue', label="MP DA - OL Data", markersize=2)
        ax2_ins2.axes.yaxis.set_visible(False)
        ax2_ins2.axes.xaxis.set_visible(False)
        ax2_ins2.set_xlim(0,num_days_to_plot-1)
        ax2_ins2.spines['top'].set_color('g')
        ax2_ins2.spines['right'].set_color('g')
        secay2_2.tick_params(axis='y', colors='g', labelsize=8)
        ax2_ins2.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay2_2.set_ylabel('MP OL-DA',labelpad=5, fontsize='small', color='g', ha='center')

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################
    if (N36):
        filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_OL=ma.masked_outside(LIS_twelve_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_DA=ma.masked_outside(N36_DA_twelve_array_soiltemp[0,:,station_number],min_cor,350)
        mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
        mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
        filtered_y_final_36_OL=ma.masked_array(LIS_twelve_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_y_final_36_DA=ma.masked_array(N36_DA_twelve_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
    
        if (filtered_y_final_36_OL.shape[0] > 100):
            slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_36_OL)
            #compute the bias
            filtered_diff=filtered_y_final_36_OL-filtered_x_final_OL
            temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_36_DA.shape[0] > 100):
            slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_36_DA)
            #compute the bias
            filtered_diff=filtered_y_final_36_DA-filtered_x_final_DA
            temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    if (NMP):
        filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_OL=ma.masked_outside(MP_twelve_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_DA=ma.masked_outside(MP_DA_twelve_array_soiltemp[0,:,station_number],min_cor,350)
        mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
        mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
        filtered_y_final_MP_OL=ma.masked_array(MP_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_y_final_MP_DA=ma.masked_array(MP_DA_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
    
        if (filtered_y_final_MP_OL.shape[0] > 100):
            slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_MP_OL)
            #compute the bias
            filtered_diff=filtered_y_final_MP_OL-filtered_x_final_OL
            temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_MP_DA.shape[0] > 100):
            slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_MP_DA)
            #compute the bias
            filtered_diff=filtered_y_final_MP_DA-filtered_x_final_DA
            temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
            
    ax3=figure.add_subplot(413)
    ax3.set_ylim(240, 350)
    ax3.set_xlim(0,num_days_to_plot-1)
    ax3.plot(num_days_index, SCAN_twelve_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dashed', label='ISMN 2-in')
    if (N36):
        ax3.plot(num_days_index, LIS_twelve_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah OL 3.6 LY1')
        ax3.plot(num_days_index, N36_DA_twelve_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dotted', label='Noah DA 3.6 LY1')
    if (NMP):
        ax3.plot(num_days_index, MP_twelve_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah OL MP LY1')
        ax3.plot(num_days_index, MP_DA_twelve_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dotted', label='Noah DA MP LY1')
        
    ax3.plot(num_days_index, ISSCP_twelve_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    ax3.grid()
    ax3.set_xticks(num_days_index[::90])
    ax3.set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in dates_array_00[::90]], rotation=45, fontsize=7.0)
    ax3.set_ylabel('Temp (K)')
    ax3.text(-0.1, 0.9, 'C)', transform=ax3.transAxes, size=20, weight='bold')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    ax3.text(10, 330, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    if (N36):
        ax3.text(1.08, 0.1, "N36 OL Bias 12 UTC: "+str(round(temp_bias_36_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax3.transAxes)
        ax3.text(1.08, 0.3, "N36 DA Bias 12 UTC: "+str(round(temp_bias_36_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax3.transAxes)
        ax3.text(1.08, 0.2, "N36 OL RMS 12 UTC: "+str(round(r_value_36_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax3.transAxes)
        ax3.text(1.08, 0.4, "N36 DA RMS 12 UTC: "+str(round(r_value_36_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax3.transAxes)
    if (NMP):
        ax3.text(1.08, .5, "MP OL Bias 12 UTC:"+str(round(temp_bias_MP_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax3.transAxes)
        ax3.text(1.08, .7, "MP DA Bias 12 UTC:"+str(round(temp_bias_MP_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax3.transAxes)
        ax3.text(1.08, .6, "MP OL RMS 12 UTC: "+str(round(r_value_MP_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax3.transAxes)
        ax3.text(1.08, .8, "MP DA RMS 12 UTC: "+str(round(r_value_MP_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax3.transAxes)
    #ax3.legend(loc='upper right', edgecolor="none", borderaxespad=0.)
    
    if (N36):
        ax3_ins = inset_axes(ax3, width="100%", height="10%", loc="lower left", borderpad=0)
        secay3 = ax3_ins.secondary_yaxis('right')
        ax3_ins.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax3_ins.plot(num_days_index, N36_DA_twelve_array_soiltemp[0,:,station_number] - LIS_twelve_array_soiltemp[0,:,station_number], marker='.', color='black', label="N36 DA - OL Data", markersize=2)
        ax3_ins.axes.yaxis.set_visible(False)
        ax3_ins.axes.xaxis.set_visible(False)
        ax3_ins.set_xlim(0,num_days_to_plot-1)
        ax3_ins.spines['top'].set_color('g')
        ax3_ins.spines['right'].set_color('g')
        secay3.tick_params(axis='y', colors='g', labelsize=8)
        ax3_ins.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay3.set_ylabel('OL-DA Diff (K)',labelpad=5, fontsize='small', color='g', ha='left')
   
    if (NMP):
        ax3_ins2 = inset_axes(ax3, width="100%", height="10%", loc="upper left", borderpad=0)
        secay3_2 = ax3_ins2.secondary_yaxis('right')
        ax3_ins2.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax3_ins2.plot(num_days_index, MP_DA_twelve_array_soiltemp[0,:,station_number] - MP_twelve_array_soiltemp[0,:,station_number], marker='.', color='blue', label="MP DA - OL Data", markersize=2)
        ax3_ins2.axes.yaxis.set_visible(False)
        ax3_ins2.axes.xaxis.set_visible(False)
        ax3_ins2.set_xlim(0,num_days_to_plot-1)
        ax3_ins2.spines['top'].set_color('g')
        ax3_ins2.spines['right'].set_color('g')
        secay3_2.tick_params(axis='y', colors='g', labelsize=8)
        ax3_ins2.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay3_2.set_ylabel('MP OL-DA',labelpad=5, fontsize='small', color='g', ha='center')

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################
    if (N36):
        filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_OL=ma.masked_outside(LIS_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_36_DA=ma.masked_outside(N36_DA_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
        mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
        mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
        filtered_y_final_36_OL=ma.masked_array(LIS_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
        filtered_y_final_36_DA=ma.masked_array(N36_DA_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
    
        if (filtered_y_final_36_OL.shape[0] > 100):
            slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_36_OL)
            #compute the bias
            filtered_diff=filtered_y_final_36_OL-filtered_x_final_OL
            temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_36_DA.shape[0] > 100):
            slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_36_DA)
            #compute the bias
            filtered_diff=filtered_y_final_36_DA-filtered_x_final_DA
            temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    if (NMP):
        filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_OL=ma.masked_outside(MP_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
        filtered_y_MP_DA=ma.masked_outside(MP_DA_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
        mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
        mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
        filtered_x_final_OL=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_x_final_DA=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
        filtered_y_final_MP_OL=ma.masked_array(MP_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
        filtered_y_final_MP_DA=ma.masked_array(MP_DA_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
    
        if (filtered_y_final_MP_OL.shape[0] > 100):
            slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final_OL, filtered_y_final_MP_OL)
            #compute the bias
            filtered_diff=filtered_y_final_MP_OL-filtered_x_final_OL
            temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
        if (filtered_y_final_MP_DA.shape[0] > 100):
            slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final_DA, filtered_y_final_MP_DA)
            #compute the bias
            filtered_diff=filtered_y_final_MP_DA-filtered_x_final_DA
            temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    
    ax4=figure.add_subplot(414)
    ax4.set_ylim(240, 350)
    ax4.set_xlim(0,num_days_to_plot-1)
    ax4.text(10, 330, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    ax4.plot(num_days_index, SCAN_eighteen_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dashed', label='ISMN 2-in')
    if (N36):
        ax4.plot(num_days_index, LIS_eighteen_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah OL 3.6 LY1')
        ax4.plot(num_days_index, N36_DA_eighteen_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dotted', label='Noah 3.6 DA LY1')
    if (NMP):
        ax4.plot(num_days_index, MP_eighteen_array_soiltemp[0,:,station_number], lw=1.0, linestyle='solid', label='Noah OL MP LY1')
        ax4.plot(num_days_index, MP_DA_eighteen_array_soiltemp[0,:,station_number], lw=1.0, linestyle='dotted', label='Noah MP DA LY1')
    ax4.plot(num_days_index, ISSCP_eighteen_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    ax4.grid()
    ax4.set_xticks(num_days_index[::90])
    ax4.set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in dates_array_00[::90]], rotation=45, fontsize=7.0)
    ax4.set_xlabel('Day of Year')
    ax4.set_ylabel('Temp (K)')
    ax4.text(-0.1, 0.9, 'D)', transform=ax4.transAxes, size=20, weight='bold')
    #bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    #ax4.legend(loc='upper right', edgecolor="none", borderaxespad=0.)
    if (N36):
        ax4.text(1.08, 0.1, "N36 OL Bias 18 UTC: "+str(round(temp_bias_36_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax4.transAxes)
        ax4.text(1.08, 0.2, "N36 DA Bias 18 UTC: "+str(round(temp_bias_36_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax4.transAxes)
        ax4.text(1.08, 0.3, "N36 OL RMS 18 UTC: "+str(round(r_value_36_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax4.transAxes)
        ax4.text(1.08, 0.4, "N36 DA RMS 18 UTC: "+str(round(r_value_36_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax4.transAxes)
    if (NMP):
        ax4.text(1.08, .5, "MP OL Bias 18 UTC:"+str(round(temp_bias_MP_OL,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax4.transAxes)
        ax4.text(1.08, .6, "MP DA Bias 18 UTC:"+str(round(temp_bias_MP_DA,1)), ha="left", va="center", size=9, bbox=bbox_props, transform=ax4.transAxes)
        ax4.text(1.08, .7, "MP OL RMS 18 UTC: "+str(round(r_value_MP_OL,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax4.transAxes)
        ax4.text(1.08, .8, "MP DA RMS 18 UTC: "+str(round(r_value_MP_DA,1)), ha="left", va="center", size=9, color='blue', bbox=bbox_props, transform=ax4.transAxes)
    
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=6)
    
    if (N36):
        ax4_ins = inset_axes(ax4, width="100%", height="10%", loc="lower left", borderpad=0)
        secay4 = ax4_ins.secondary_yaxis('right')
    
        ax4_ins.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax4_ins.plot(num_days_index, N36_DA_eighteen_array_soiltemp[0,:,station_number] - LIS_eighteen_array_soiltemp[0,:,station_number], marker='.', color='black',   label="N36 DA - OL Data", markersize=2)
        ax4_ins.axes.yaxis.set_visible(False)
        ax4_ins.axes.xaxis.set_visible(False)
        ax4_ins.set_xlim(0,num_days_to_plot-1)
        ax4_ins.spines['top'].set_color('g')
        ax4_ins.spines['right'].set_color('g')
        ax4_ins.set_facecolor('#eafff5')
        secay4.tick_params(axis='y', colors='g', labelsize=8)
        #ax4_ins.axis('off')
        secay4.set_ylabel('OL-DA Diff (K)',labelpad=5, fontsize='small', color='g', ha='left')
    
    if (NMP):
        ax4_ins2 = inset_axes(ax4, width="100%", height="10%", loc="upper left", borderpad=0)
        secay4_2 = ax4_ins2.secondary_yaxis('right')
        ax4_ins2.set_ylim(-5.0, 5.0)
        #ax1_ins.set_xlim(0,index_array_cycles)
        ax4_ins2.plot(num_days_index, MP_DA_eighteen_array_soiltemp[0,:,station_number] - MP_eighteen_array_soiltemp[0,:,station_number], marker='.', color='blue',     label="MP OL - DA Data", markersize=2)
        ax4_ins2.axes.yaxis.set_visible(False)
        ax4_ins2.axes.xaxis.set_visible(False)
        ax4_ins2.set_xlim(0,num_days_to_plot-1)
        ax4_ins2.spines['top'].set_color('g')
        ax4_ins2.spines['right'].set_color('g')
        secay4_2.tick_params(axis='y', colors='g', labelsize=8)
        ax4_ins2.set_facecolor('#eafff5')
        #ax4_ins.axis('off')
        secay4_2.set_ylabel('MP OL-DA',labelpad=5, fontsize='small', color='g', ha='center')
    
    plt.suptitle(CURR_SCANDATA.columns[station_number]+' LIS Noah vs. ISMN Observations Valid \n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    
    img_fname_pre=img_out_path+CURR_SCANDATA.columns[station_number]
    img_fname_end='-LIS_Noah_36-MP_TOP_'+EXP_NAME+'_vs_ISMN_Data_year'+BGDATE+'-'+EDATE+'.png'
    plt.savefig(img_fname_pre+img_fname_end)
    plt.close(figure)


##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 1 soil temps for each station
##################################################################################################################

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ##################################################################################################################
    Scat_min=240
    Scat_max=340
    
    min_cor=260
    max_cor=350
    
    figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))
    
    filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 00 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[0,0].set_ylim(Scat_min, Scat_max)
        axes[0,0].set_xlim(Scat_min, Scat_max)
        axes[0,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[0,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[0,0].grid()
        axes[0,0].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,3)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-14, Scat_max-20, "BIAS:"+str(round(temp_bias,3)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-14, Scat_max-29, "(ISSCP-ISMN)", ha="right", va="center",size=8)
        axes[0,0].text(Scat_min-14, Scat_max-45, "R:"+str(round(r_value,3)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_max-14, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
        #axes[0,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 03Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 03 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[1,0].set_ylim(Scat_min, Scat_max)
        axes[1,0].set_xlim(Scat_min, Scat_max)
        axes[1,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[1,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[1,0].grid()
        axes[1,0].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
        axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,3)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-14, Scat_max-20, "BIAS:"+str(round(temp_bias,3)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-14, Scat_max-29, "(ISSCP-ISMN)", ha="right", va="center",size=8)
        axes[1,0].text(Scat_min-14, Scat_max-45, "R:"+str(round(r_value,3)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_max-14, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
  
    #axes[1,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 06 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[2,0].set_ylim(Scat_min, Scat_max)
        axes[2,0].set_xlim(Scat_min, Scat_max)
        axes[2,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[2,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[2,0].grid()
        axes[2,0].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
        axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,3)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-14, Scat_max-20, "BIAS:"+str(round(temp_bias,3)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-14, Scat_max-29, "(ISSCP-ISMN)", ha="right", va="center", size=8)
        axes[2,0].text(Scat_min-14, Scat_max-45, "R:"+str(round(r_value,3)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_max-14, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[2,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 09Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 09 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[3,0].set_ylim(Scat_min, Scat_max)
        axes[3,0].set_xlim(Scat_min, Scat_max)
        axes[3,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[3,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[3,0].grid()
        axes[3,0].set_ylabel('ISCCP Skin T (K)')
        axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-14, Scat_max-20, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-14, Scat_max-29, "(ISSCP-ISMN)", ha="right", va="center", size=8)
        axes[3,0].text(Scat_min-14, Scat_max-45, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_max-14, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[3,0].legend(loc='upper right', borderaxespad=0.)
        
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 12 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
    
        axes[0,1].set_ylim(Scat_min, Scat_max)
        axes[0,1].set_xlim(Scat_min, Scat_max)
        axes[0,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[0,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[0,1].grid()
        axes[0,1].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+5, Scat_max-20, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+5, Scat_max-29, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[0,1].text(Scat_max+5, Scat_max-45, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+5, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[0,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 15Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 15 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[1,1].set_ylim(Scat_min, Scat_max)
        axes[1,1].set_xlim(Scat_min, Scat_max)
        axes[1,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[1,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[1,1].grid()
        axes[1,1].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+5, Scat_max-20, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+5, Scat_max-29, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[1,1].text(Scat_max+5, Scat_max-45, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+5, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[1,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
    
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 18 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[2,1].set_ylim(Scat_min, Scat_max)
        axes[2,1].set_xlim(Scat_min, Scat_max)
        axes[2,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[2,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[2,1].grid()
        axes[2,1].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+5, Scat_max-20, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+5, Scat_max-29, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[2,1].text(Scat_max+5, Scat_max-45, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+5, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[2,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 21Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[0,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)

        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 21 UTC ISSCP vs. ISMN LY1 '+ CURR_SCANDATA.columns[station_number]+' is...', r_value)
    
        axes[3,1].set_ylim(Scat_min, Scat_max)
        axes[3,1].set_xlim(Scat_min, Scat_max)
        axes[3,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[3,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[3,1].grid()
        axes[3,1].set_ylabel('ISCCP Skin T (K)')
        axes[3,1].set_xlabel('ISMN 5cm Tsoil (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+5, Scat_max-20, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+5, Scat_max-29, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[3,1].text(Scat_max+5, Scat_max-45, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+5, Scat_max-60, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)

    #axes[3,1].legend(loc='upper right', borderaxespad=0.)
        
    plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp \n  Station ID -'+CURR_SCANDATA.columns[station_number], fontsize=18)
    
    img_fname_pre=img_out_path+CURR_SCANDATA.columns[station_number]
    plt.savefig(img_fname_pre+'_ISCCPvsISMN5cm_'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)
    
    
##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 2 soil temps for each station
##################################################################################################################
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ##################################################################################################################

    figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))
    
    filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 00 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[0,0].set_ylim(Scat_min, Scat_max)
        axes[0,0].set_xlim(Scat_min, Scat_max)
        axes[0,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[0,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[0,0].grid()
        axes[0,0].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
        axes[0,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
        #axes[0,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 03Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 03 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[1,0].set_ylim(Scat_min, Scat_max)
        axes[1,0].set_xlim(Scat_min, Scat_max)
        axes[1,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[1,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[1,0].grid()
        axes[1,0].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center",size=6)
        axes[1,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
  
    #axes[1,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 06 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[2,0].set_ylim(Scat_min, Scat_max)
        axes[2,0].set_xlim(Scat_min, Scat_max)
        axes[2,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[2,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[2,0].grid()
        axes[2,0].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)",ha="right", va="center", size=6)
        axes[2,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[2,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 09Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 09 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[3,0].set_ylim(Scat_min, Scat_max)
        axes[3,0].set_xlim(Scat_min, Scat_max)
        axes[3,0].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[3,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[3,0].grid()
        axes[3,0].set_ylabel('ISCCP Skin T (K)')
        axes[3,0].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
        axes[3,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[3,0].legend(loc='upper right', borderaxespad=0.)
        
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 12 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
    
        axes[0,1].set_ylim(Scat_min, Scat_max)
        axes[0,1].set_xlim(Scat_min, Scat_max)
        axes[0,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[0,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[0,1].grid()
        axes[0,1].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[0,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[0,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 15Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 15 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[1,1].set_ylim(Scat_min, Scat_max)
        axes[1,1].set_xlim(Scat_min, Scat_max)
        axes[1,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[1,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[1,1].grid()
        axes[1,1].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[1,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[1,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
    
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 18 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
        
        axes[2,1].set_ylim(Scat_min, Scat_max)
        axes[2,1].set_xlim(Scat_min, Scat_max)
        axes[2,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[2,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[2,1].grid()
        axes[2,1].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[2,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[2,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 21Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[1,:,station_number],min_cor,350)
    filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,station_number],min_cor,350)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)

        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 21 UTC ISSCP vs. ISMN LY2 '+CURR_SCANDATA.columns[station_number]+' is...', r_value)
    
        axes[3,1].set_ylim(Scat_min, Scat_max)
        axes[3,1].set_xlim(Scat_min, Scat_max)
        axes[3,1].scatter(filtered_x_final, filtered_y_final, marker='+')
        axes[3,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[3,1].grid()
        axes[3,1].set_ylabel('ISCCP Skin T (K)')
        axes[3,1].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[3,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    #axes[3,1].legend(loc='upper right', borderaxespad=0.)
        
    plt.suptitle('ISCCP Skin Temperature vs. ISMN 10cm Soil Temp', fontsize=18)
    
    img_fname_pre=img_out_path+CURR_SCANDATA.columns[station_number]
    plt.savefig(img_fname_pre+'_ISCCPvsISMN10cm_'+BGDATE+'-'+EDATE+'.png')
    plt.close(figure)
    station_number+=1
    
##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 1 soil temps for each station
##################################################################################################################


figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))

filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 00 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[0,0].set_ylim(Scat_min, Scat_max)
axes[0,0].set_xlim(Scat_min, Scat_max)
axes[0,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[0,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-14, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-14, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_max-14, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[0,0].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 03 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[1,0].set_ylim(Scat_min, Scat_max)
axes[1,0].set_xlim(Scat_min, Scat_max)
axes[1,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[1,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-14, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-14, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_max-14, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[1,0].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 06 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[2,0].set_ylim(Scat_min, Scat_max)
axes[2,0].set_xlim(Scat_min, Scat_max)
axes[2,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[2,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-14, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-14, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_max-14, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[2,0].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 09 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[3,0].set_ylim(Scat_min, Scat_max)
axes[3,0].set_xlim(Scat_min, Scat_max)
axes[3,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[3,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-14, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-14, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_max-14, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[3,0].legend(loc='upper right', borderaxespad=0.)
    
    
##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 12 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[0,1].set_ylim(Scat_min, Scat_max)
axes[0,1].set_xlim(Scat_min, Scat_max)
axes[0,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[0,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[0,1].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 15 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[1,1].set_ylim(Scat_min, Scat_max)
axes[1,1].set_xlim(Scat_min, Scat_max)
axes[1,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[1,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[1,1].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 18 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[2,1].set_ylim(Scat_min, Scat_max)
axes[2,1].set_xlim(Scat_min, Scat_max)
axes[2,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[2,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[2,1].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[0,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 21 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[3,1].set_ylim(Scat_min, Scat_max)
axes[3,1].set_xlim(Scat_min, Scat_max)
axes[3,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[3,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 5 cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
    
plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp', fontsize=18)

img_fname_pre=img_out_path
plt.savefig(img_out_path+'ISMN5cm_All_Stations_'+BGDATE+'-'+EDATE+'.png')
plt.close(figure)


##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 1 soil temps for each station
##################################################################################################################


figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))


filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 00 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[0,0].set_ylim(Scat_min, Scat_max)
axes[0,0].set_xlim(Scat_min, Scat_max)
axes[0,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[0,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[0,0].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 03 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[1,0].set_ylim(240, Scat_max)
axes[1,0].set_xlim(240, Scat_max)
axes[1,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[1,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[1,0].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 06 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[2,0].set_ylim(240, Scat_max)
axes[2,0].set_xlim(240, Scat_max)
axes[2,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[2,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[2,0].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 09 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[3,0].set_ylim(240, Scat_max)
axes[3,0].set_xlim(240, Scat_max)
axes[3,0].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[3,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R:"+str(round(r_value,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "Slope:"+str(round(slope,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_max-12, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[3,0].legend(loc='upper right', borderaxespad=0.)
    
    
##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 12 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[0,1].set_ylim(240, Scat_max)
axes[0,1].set_xlim(240, Scat_max)
axes[0,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[0,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[0,1].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 15 UTC ISSCP vs. ISMN LY1 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[1,1].set_ylim(240, Scat_max)
axes[1,1].set_xlim(240, Scat_max)
axes[1,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[1,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[1,1].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 18 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[2,1].set_ylim(240, Scat_max)
axes[2,1].set_xlim(240, Scat_max)
axes[2,1].scatter(filtered_x_final, filtered_y_final, marker='+')
axes[2,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)
#axes[2,1].legend(loc='upper right', borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[1,:,:],min_cor,350)
filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,:],min_cor,350)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,:],mask=mask).compressed()
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 21 UTC ISSCP vs. ISMN LY2 is...', r_value)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

axes[3,1].set_ylim(240, Scat_max)
axes[3,1].set_xlim(240, Scat_max)
#axes[3,1].scatter(ISSCP_twtyone_array_soiltemp[:,:], SCAN_twtyone_array_soiltemp[1,:,:], marker='+')
axes[3,1].scatter(filtered_x_final, filtered_y_final, marker='+')
#axes[3,1].plot(X_test, y_pred, color='red', linewidth=2)
axes[3,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R:"+str(round(r_value,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "Slope:"+str(round(slope,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-55, "p-value:"+str(round(p_value,2)), ha="left", va="center", size=10)

axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
#axes[3,1].legend(loc='upper right', borderaxespad=0.)
    
plt.suptitle('ISCCP Skin Temperature vs. ISMN 10cm Soil Temp', fontsize=18)

img_fname_pre=img_out_path
plt.savefig(img_out_path+'ISCCPvsISMN10cm_All_Stations_'+BGDATE+'-'+EDATE+'.png')
plt.close(figure)





#plt.show()

