import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.cm import get_cmap
import netCDF4 as NC
import datetime
import calendar
import math

def Read_LIS_NoahMP_Tsoil_data(inDTG_YYYY, inDTG_mm, inDTG_dd, inDTG_hh, the_lat_point, the_lon_point, filedir):

    def find_nearest(array, value):
        idx = np.nanargmin(np.abs(value - array))
        return idx
    #  figure out how many stations we have to read in
    num_stations=np.shape(the_lat_point)[0]
    
    #filedir="/p/cwfs/eylandej/OPENLOOP/NoahMP/SURFACEMODEL/"
    #filedir="/Volumes/DataDrive/DIS_test1/NoahMP/ISCCP_OL_AFRICA/SURFACEMODEL/"
    filepre="LIS_HIST_"
    filepost="00.d01.nc"

    d = datetime.datetime(inDTG_YYYY, inDTG_mm, inDTG_dd, inDTG_hh)
    theyyyymmddhh='{:%Y%m%d%H}'.format(d)
    theyyyymm='{:%Y%m}'.format(d)

    thefilename=filedir+theyyyymm+'/'+filepre+theyyyymmddhh+filepost
    print ('Reading in file...', thefilename)
    
    NCdata = NC.Dataset(thefilename, 'r')

    tsoil_data=NCdata.variables['SoilTemp_inst'][:]
    lon_data=NCdata.variables['lon'][:]
    lat_data=NCdata.variables['lat'][:]

    NCdata.close()

    numrows=np.shape(tsoil_data)[1]
    numcols=np.shape(tsoil_data)[2]

    temp_array_lon = np.zeros((numcols), dtype=np.float64)
    temp_array_lat = np.zeros((numrows), dtype=np.float64)
    the_tsoil_at_point = np.zeros((4,num_stations), dtype=np.float64)

    for x in range (0, numcols-1):
        temp_array_lon[x]=np.max(lon_data[:,x])

    for y in range (0, numrows-1):
        temp_array_lat[y]=np.max(lat_data[y,:])
    
    num_stations_loop=0
    for num_stations_loop in range(num_stations):
        nearest_lat=find_nearest(temp_array_lat, the_lat_point[num_stations_loop])
        nearest_lon=find_nearest(temp_array_lon, the_lon_point[num_stations_loop])

        the_tsoil_at_point[:,num_stations_loop]=tsoil_data[:,nearest_lat, nearest_lon]

    return the_tsoil_at_point



