import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import netCDF4 as NC
import datetime
import calendar
import math

def Read_ISCCP_data(inDTG_YYYY, inDTG_mm, inDTG_dd, inDTG_hh, the_lat_point, the_lon_point):

    def find_nearest(array, value):
        idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
        return idx

    num_stations=np.shape(the_lat_point)[0]
    filedir="/Volumes/Data8/ISCCP/"
 #   filedir="/p/cwfs/eylandej/ISCCP/"
    filepre="ISCCP.HXG.v01r00.GLOBAL."
    filepost="00.GPC.10KM.EQ0.10.nc"
    
    d = datetime.datetime(inDTG_YYYY, inDTG_mm, inDTG_dd, inDTG_hh)
    theyyyymmddhh='{:%Y.%m.%d.%H}'.format(d)
    theyyyymmdd='{:%Y%m%d}'.format(d)
    thefilename=filedir+theyyyymmdd+'/'+filepre+theyyyymmddhh+filepost
    print (thefilename)
    NCdata = NC.Dataset(thefilename, 'r')

    tskin_data=NCdata.variables['itmp'][:]
    lon_data=NCdata.variables['lon'][:]
    lat_data=NCdata.variables['lat'][:]
    tmp_tab_data=NCdata.variables['tmptab'][:]
    clouds=NCdata.variables['cloud'][:]


    NCdata.close()

    numrows=np.shape(tskin_data)[0]
    numcols=np.shape(tskin_data)[1]
    print (numcols)
    temp_array_lon = np.zeros((numcols, numrows), dtype=np.float64)
    temp_array_lat = np.zeros((numcols, numrows), dtype=np.float64)
    the_tskin_at_point = np.zeros((num_stations), dtype=np.float64)
   
    for num_stations_loop in range(num_stations):
        nearest_lat=find_nearest(lat_data, the_lat_point[num_stations_loop])
        nearest_lon=find_nearest(lon_data, the_lon_point[num_stations_loop])
        

        the_tmp_loopup=tskin_data[nearest_lat, nearest_lon]
        cloud_val=clouds[nearest_lat, nearest_lon]
        if (the_tmp_loopup > 0.0 and the_tmp_loopup < 255.0 and cloud_val == 0):
            the_tskin_at_point[num_stations_loop]=tmp_tab_data[the_tmp_loopup]
        else:
            the_tskin_at_point[num_stations_loop]=np.nan
        #print (the_tskin_at_point)
    

    return the_tskin_at_point
