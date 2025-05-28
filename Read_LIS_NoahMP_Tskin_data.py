import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.cm import get_cmap
import netCDF4 as NC
import datetime
import calendar
import math

def Read_LIS_NoahMP_Tskin_data(inDTG_YYYY, inDTG_mm, inDTG_dd, inDTG_hh, the_lat_point, the_lon_point, filedir):

    def find_nearest(array, value):
        idx = np.nanargmin(np.abs(value - array))
        return idx
    num_stations=np.shape(the_lat_point)[0]

    #filedir="/p/cwfs/eylandej/OPENLOOP/NoahMP/SURFACEMODEL/"
    filepre="LIS_HIST_"
    filepost="00.d01.nc"

    d = datetime.datetime(inDTG_YYYY, inDTG_mm, inDTG_dd, inDTG_hh)
    theyyyymmddhh='{:%Y%m%d%H}'.format(d)
    theyyyymm='{:%Y%m}'.format(d)

    thefilename=filedir+theyyyymm+'/'+filepre+theyyyymmddhh+filepost
    print ('Reading Noah MP Skin Temperature from file...', thefilename)
    NCdata = NC.Dataset(thefilename, 'r')

    tskin_data=NCdata.variables['AvgSurfT_inst'][:]
    lon_data=NCdata.variables['lon'][:]
    lat_data=NCdata.variables['lat'][:]

    NCdata.close()

    numrows=np.shape(tskin_data)[0]
    numcols=np.shape(tskin_data)[1]
    
    temp_array_lon = np.zeros((numcols), dtype=np.float64)
    temp_array_lat = np.zeros((numrows), dtype=np.float64)
    the_tskin_at_point = np.zeros((num_stations), dtype=np.float64)
    
    for x in range (0, numcols-1):
        temp_array_lon[x]=np.max(lon_data[:,x])


    for y in range (0, numrows-1):
        temp_array_lat[y]=np.max(lat_data[y,:])


    num_stations_loop=0
    for num_stations_loop in range(num_stations):
        nearest_lat=find_nearest(temp_array_lat, the_lat_point[num_stations_loop])
        nearest_lon=find_nearest(temp_array_lon, the_lon_point[num_stations_loop])
        the_tskin_at_point[num_stations_loop]=tskin_data[nearest_lat, nearest_lon]


    return the_tskin_at_point


