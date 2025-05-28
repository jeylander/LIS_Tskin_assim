import os
import numpy as np
import datetime
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from scipy import stats

def monthly(stat_data1, stat_data2, stat_data3, stat_data4, stat_data5, stat_data6, stat_data7, stat_data8, SAT_data1, SAT_data2, SAT_data3, SAT_data4, SAT_data5, SAT_data6, SAT_data7, SAT_data8, m1_data1, m1_data2, m1_data3, m1_data4, m1_data5, m1_data6, m1_data7, m1_data8, m1_data1_skin, m1_data2_skin, m1_data3_skin, m1_data4_skin, m1_data5_skin, m1_data6_skin, m1_data7_skin, m1_data8_skin, m2_data1, m2_data2, m2_data3, m2_data4, m2_data5, m2_data6, m2_data7, m2_data8, m2_data1_skin, m2_data2_skin, m2_data3_skin, m2_data4_skin, m2_data5_skin, m2_data6_skin, m2_data7_skin, m2_data8_skin, ptitle, fname_out):

    
    #for  graph_loop in range (0, max_num_stations, 1):

    ###############################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ###############################################################################################


        #  Generate the layer average
        plot_minT=260
        plot_maxT=320
        Noah_layer_depths=[.05, .25, .70, 1.5 ]
        Noah_tskin_tsoil_depths=[0,0.05,0.25,0.70,1.5]
        #SCAN_layer_depths=[.05, .10, .20, 0.5, 1.02 ]
        SCAN_layer_depths=[.05, .25, 0.7, 1.02 ]
        tskin_depth=[0]
        array_size=stat_data1.shape[1]
        m1_size=m1_data1.shape[1]
        m2_size=m2_data1.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data1=np.zeros((4),dtype='float64')
        temp_m1_data1   =np.zeros((4),dtype='float64')
        temp_m2_data1   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL1=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data1_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data1_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data1_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data1_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data1 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data1[SAT_skin_temp_addresses])
        
        for num_layer in range (0,4,1):
            temp_stat_data_all[:]=stat_data1[num_layer,:]
            temp_stat_data11=np.where(temp_stat_data_all > 0.01)
            temp_stat_data1[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data11])
            if num_layer < 4:
                temp_m1_data_all[:]=m1_data1[num_layer,:]
                temp_m1_data11=np.where(temp_m1_data_all > 0.01)
                temp_m1_data1[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data11])
                
                temp_m2_data_all[:]=m2_data1[num_layer,:]
                temp_m2_data11=np.where(temp_m2_data_all > 0.01)
                temp_m2_data1[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data11])
        
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data1[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data1[:]
        SCAN_ISCCP_TSKIN_TSOIL1[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL1[1:5]=temp_stat_data1[:]
        
        figure, axes=plt.subplots(ncols=4,nrows=2,  figsize=(15,8))
        axes[0,0].set_ylim(2.0, 0.0)
        axes[0,0].set_xlim([plot_minT, plot_maxT])
        axes[0,0].plot([273.15,273.15],[0,2.0], '-', lw=0.5, color="blue")
        axes[0,0].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[0,0].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[0,0].plot(temp_stat_data1,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[0,0].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[0,0].grid()
        axes[0,0].set_ylabel('Layer Depth (m)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(plot_minT+2, 1.8, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,0].legend(loc='lower right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 03Z Top Layer COMPARISON
    ##################################################################################################################

        array_size=stat_data2.shape[1]
        m1_size=m1_data2.shape[1]
        m2_size=m2_data2.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data2 =np.zeros((4),dtype='float64')
        temp_m1_data2   =np.zeros((4),dtype='float64')
        temp_m2_data2   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data2_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data2_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data2_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data2_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data2 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data2[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data2[num_layer,:]
               temp_stat_data22=np.where(temp_stat_data_all > 0.01)
               temp_stat_data2[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data22])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data2[num_layer,:]
                   temp_m1_data22=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data2[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data22])
                   
                   temp_m2_data_all[:]=m2_data2[num_layer,:]
                   temp_m2_data22=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data2[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data22])

        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data2[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data2[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data2[:]

        axes[0,1].set_ylim(2.0, 0.0)
        axes[0,1].set_xlim(plot_minT, plot_maxT)
        axes[0,1].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[0,1].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[0,1].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[0,1].plot(temp_stat_data2,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[0,1].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[0,1].grid()
        #axes[0,1].set_ylabel('Temp (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(plot_minT+2, 1.8, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,1].legend(loc='lower right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################

        array_size=stat_data3.shape[1]
        m1_size=m1_data3.shape[1]
        m2_size=m2_data3.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data3 =np.zeros((4),dtype='float64')
        temp_m1_data3   =np.zeros((4),dtype='float64')
        temp_m2_data3   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data3_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data3_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data3_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data3_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data3 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data3[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data3[num_layer,:]
               temp_stat_data33=np.where(temp_stat_data_all > 0.01)
               temp_stat_data3[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data33])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data3[num_layer,:]
                   temp_m1_data33=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data3[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data33])
                   
                   temp_m2_data_all[:]=m2_data3[num_layer,:]
                   temp_m2_data33=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data3[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data33])
                   
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data3[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data3[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data3[:]

        axes[0,2].set_ylim(2.0, 0.0)
        axes[0,2].set_xlim(plot_minT, plot_maxT)
        axes[0,2].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[0,2].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[0,2].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[0,2].plot(temp_stat_data3,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[0,2].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[0,2].grid()
        #axes[0,2].set_xlabel('Temp (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,2].text(plot_minT+2, 1.8, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,2].legend(loc='lower right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 09Z Top Layer COMPARISON
    ##################################################################################################################

        array_size=stat_data4.shape[1]
        m1_size=m1_data4.shape[1]
        m2_size=m2_data4.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data4 =np.zeros((4),dtype='float64')
        temp_m1_data4   =np.zeros((4),dtype='float64')
        temp_m2_data4   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data4_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data4_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data4_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data4_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data4 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data4[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data4[num_layer,:]
               temp_stat_data44=np.where(temp_stat_data_all > 0.01)
               temp_stat_data4[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data44])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data4[num_layer,:]
                   temp_m1_data44=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data4[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data44])
                   
                   temp_m2_data_all[:]=m2_data4[num_layer,:]
                   temp_m2_data44=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data4[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data44])

        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data4[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data4[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data4[:]

        axes[0,3].set_ylim(2.0, 0.0)
        axes[0,3].set_xlim(plot_minT, plot_maxT)
        axes[0,3].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[0,3].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[0,3].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[0,3].plot(temp_stat_data4,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[0,3].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[0,3].grid()
        #axes[0,3].set_ylabel('Layer')
        #axes[0,3].set_xlabel('Temp (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,3].text(plot_minT+2, 1.8, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,3].legend(loc='lower right', borderaxespad=0.)
        
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################
        
        array_size=stat_data5.shape[1]
        m1_size=m1_data5.shape[1]
        m2_size=m2_data5.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data5 =np.zeros((4),dtype='float64')
        temp_m1_data5   =np.zeros((4),dtype='float64')
        temp_m2_data5   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data5_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data5_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data5_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data5_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data5 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data5[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data5[num_layer,:]
               temp_stat_data55=np.where(temp_stat_data_all > 0.01)
               temp_stat_data5[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data55])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data5[num_layer,:]
                   temp_m1_data55=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data5[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data55])
                   
                   temp_m2_data_all[:]=m2_data5[num_layer,:]
                   temp_m2_data55=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data5[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data55])

        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data5[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data5[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data5[:]

        axes[1,0].set_ylim(2.0, 0.0)
        axes[1,0].set_xlim(plot_minT, plot_maxT)
        axes[1,0].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[1,0].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[1,0].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[1,0].plot(temp_stat_data5,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[1,0].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[1,0].grid()
        axes[1,0].set_xlabel('Temp (K)')
        axes[1,0].set_ylabel('Layer Depth (m)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,0].text(plot_minT+2, 1.8, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,0].legend(loc='lower right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 15Z Top Layer COMPARISON
    ##################################################################################################################
        
        array_size=stat_data6.shape[1]
        m1_size=m1_data6.shape[1]
        m2_size=m2_data6.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data6 =np.zeros((4),dtype='float64')
        temp_m1_data6   =np.zeros((4),dtype='float64')
        temp_m2_data6   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data6_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data6_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data6_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data6_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data6 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data6[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data6[num_layer,:]
               temp_stat_data66=np.where(temp_stat_data_all > 0.01)
               temp_stat_data6[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data66])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data6[num_layer,:]
                   temp_m1_data66=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data6[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data66])
                   
                   temp_m2_data_all[:]=m2_data6[num_layer,:]
                   temp_m2_data66=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data6[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data66])

        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data6[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data6[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data6[:]



        axes[1,1].set_ylim(2.0, 0.0)
        axes[1,1].set_xlim(plot_minT, plot_maxT)
        axes[1,1].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[1,1].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[1,1].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[1,1].plot(temp_stat_data6,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[1,1].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[1,1].grid()
        #axes[1,1].set_ylabel('Layer')
        axes[1,1].set_xlabel('Temp (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(plot_minT+2, 1.8, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,1].legend(loc='lower right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

        array_size=stat_data7.shape[1]
        m1_size=m1_data7.shape[1]
        m2_size=m2_data7.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data7 =np.zeros((4),dtype='float64')
        temp_m1_data7   =np.zeros((4),dtype='float64')
        temp_m2_data7   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data7_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data7_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data7_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data7_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data7 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data7[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data7[num_layer,:]
               temp_stat_data77=np.where(temp_stat_data_all > 0.01)
               temp_stat_data7[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data77])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data7[num_layer,:]
                   temp_m1_data77=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data7[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data77])
                   
                   temp_m2_data_all[:]=m2_data7[num_layer,:]
                   temp_m2_data77=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data7[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data77])
       
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data7[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data7[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data7[:]

                   
        axes[1,2].set_ylim(2.0, 0.0)
        axes[1,2].set_xlim(plot_minT, plot_maxT)
        axes[1,2].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[1,2].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[1,2].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[1,2].plot(temp_stat_data7,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[1,2].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[1,2].grid()
        #axes[1,2].set_ylabel('Layer')
        axes[1,2].set_xlabel('Temp (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,2].text(plot_minT+2, 1.80, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,2].legend(loc='lower right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 121 Top Layer COMPARISON
    ##################################################################################################################
        
        array_size=stat_data8.shape[1]
        m1_size=m1_data8.shape[1]
        m2_size=m2_data8.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data8 =np.zeros((4),dtype='float64')
        temp_m1_data8   =np.zeros((4),dtype='float64')
        temp_m2_data8   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        SCAN_ISCCP_TSKIN_TSOIL=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data8_skin > 0.1)
        m1_skin_temp=np.nanmean(m1_data8_skin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data8_skin > 0.1)
        m2_skin_temp=np.nanmean(m2_data8_skin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(SAT_data8 > 260.0)
        SAT_skin_temp=np.nanmean(SAT_data8[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data8[num_layer,:]
               temp_stat_data88=np.where(temp_stat_data_all > 0.01)
               temp_stat_data8[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data88])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data8[num_layer,:]
                   temp_m1_data88=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data8[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data88])
                   
                   temp_m2_data_all[:]=m2_data8[num_layer,:]
                   temp_m2_data88=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data8[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data88])
        
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data8[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data8[:]
        SCAN_ISCCP_TSKIN_TSOIL[0]=SAT_skin_temp
        SCAN_ISCCP_TSKIN_TSOIL[1:5]=temp_stat_data8[:]

        axes[1,3].set_ylim(2.0, 0.0)
        axes[1,3].set_xlim(plot_minT, plot_maxT)
        axes[1,3].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[1,3].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[1,3].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[1,3].plot(temp_stat_data8,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', label='ISMN', color="green")
        axes[1,3].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[1,3].grid()
        axes[1,3].set_xlabel('Temp (K)')
        #axes[1,3].set_ylabel('Layer')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,3].text(plot_minT+2, 1.8, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,3].legend(loc='lower right', borderaxespad=0.)
        
        plt.suptitle(ptitle, fontsize=18)
        plt.savefig(fname_out)
        plt.close(figure)
        #station_number+=1


def seasonal(stat_data1, stat_data2, stat_data3, stat_data4, Sat_data1, Sat_data2, Sat_data3, Sat_data4, m1_data1, m1_data2, m1_data3, m1_data4, m1_data1_tskin, m1_data2_tskin, m1_data3_tskin, m1_data4_tskin, m2_data1, m2_data2, m2_data3, m2_data4, m2_data1_tskin, m2_data2_tskin, m2_data3_tskin, m2_data4_tskin, ptitle, fname_out):


        plot_minT=260
        plot_maxT=320
        
        Noah_layer_depths=[.05, .25, .70, 1.5 ]
        SCAN_layer_depths=[.05, .25, 0.7, 1.02 ]
        Noah_tskin_tsoil_depths=[0,0.05,0.25,0.70,1.5]
        
        figure, axes=plt.subplots(ncols=2,nrows=2,  figsize=(15,8))
        
        array_size=stat_data1.shape[1]
        m1_size=m1_data1.shape[1]
        m2_size=m2_data1.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data1 =np.zeros((4),dtype='float64')
        temp_m1_data1   =np.zeros((4),dtype='float64')
        temp_m2_data1   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data1_tskin > 0.1)
        m1_skin_temp=np.nanmean(m1_data1_tskin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data1_tskin > 0.1)
        m2_skin_temp=np.nanmean(m2_data1_tskin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(Sat_data1 > 260.0)
        SAT_skin_temp=np.nanmean(Sat_data1[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data1[num_layer,:]
               temp_stat_data11=np.where(temp_stat_data_all > 0.01)
               temp_stat_data1[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data11])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data1[num_layer,:]
                   temp_m1_data11=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data1[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data11])
                   
                   temp_m2_data_all[:]=m2_data1[num_layer,:]
                   temp_m2_data11=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data1[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data11])

        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data1[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data1[:]

        axes[0,0].set_ylim(2.0, 0.0)
        axes[0,0].set_xlim(plot_minT, plot_maxT)
        axes[0,0].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[0,0].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[0,0].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[0,0].plot(temp_stat_data1,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[0,0].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")

        axes[0,0].grid()
        axes[0,0].set_xlabel('Temperature')
        axes[0,0].set_ylabel('Layer Depth (m)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(plot_minT+2, 1.8, "Winter", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,0].legend(loc='lower right', borderaxespad=0.)
        
    
        array_size=stat_data2.shape[1]
        m1_size=m1_data2.shape[1]
        m2_size=m2_data2.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data2 =np.zeros((4),dtype='float64')
        temp_m1_data2   =np.zeros((4),dtype='float64')
        temp_m2_data2   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data2_tskin > 0.1)
        m1_skin_temp=np.nanmean(m1_data2_tskin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data2_tskin > 0.1)
        m2_skin_temp=np.nanmean(m2_data2_tskin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(Sat_data2 > 260.0)
        SAT_skin_temp=np.nanmean(Sat_data2[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data2[num_layer,:]
               temp_stat_data22=np.where(temp_stat_data_all > 0.01)
               temp_stat_data2[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data22])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data2[num_layer,:]
                   temp_m1_data22=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data2[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data22])
                   
                   temp_m2_data_all[:]=m2_data2[num_layer,:]
                   temp_m2_data22=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data2[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data22])
        
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data2[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data2[:]
        
        axes[1,0].set_ylim(2.0, 0.0)
        axes[1,0].set_xlim(plot_minT, plot_maxT)
        axes[1,0].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[1,0].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o',markersize=8.0, color="blue", label='Noah 3.6')
        axes[1,0].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[1,0].plot(temp_stat_data2,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[1,0].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[1,0].grid()
        axes[1,0].set_xlabel('Temperature')
        axes[1,0].set_ylabel('Layer Depth (m)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,0].text(plot_minT+2, 1.8, "Spring", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,0].legend(loc='lower right', borderaxespad=0.)
        
        
        
        
        array_size=stat_data3.shape[1]
        m1_size=m1_data3.shape[1]
        m2_size=m2_data3.shape[1]
        
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data3 =np.zeros((4),dtype='float64')
        temp_m1_data3   =np.zeros((4),dtype='float64')
        temp_m2_data3   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data3_tskin > 0.1)
        m1_skin_temp=np.nanmean(m1_data3_tskin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data3_tskin > 0.1)
        m2_skin_temp=np.nanmean(m2_data3_tskin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(Sat_data3 > 260.0)
        SAT_skin_temp=np.nanmean(Sat_data3[SAT_skin_temp_addresses])
           
        for num_layer in range (0,4,1):
               temp_stat_data_all[:]=stat_data3[num_layer,:]
               temp_stat_data33=np.where(temp_stat_data_all > 0.01)
               temp_stat_data3[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data33])
               if num_layer < 4:
                   temp_m1_data_all[:]=m1_data3[num_layer,:]
                   temp_m1_data33=np.where(temp_m1_data_all > 0.01)
                   temp_m1_data3[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data33])
                   
                   temp_m2_data_all[:]=m2_data3[num_layer,:]
                   temp_m2_data33=np.where(temp_m2_data_all > 0.01)
                   temp_m2_data3[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data33])
                   
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data3[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data3[:]
        
        axes[0,1].set_ylim(2.0, 0.0)
        axes[0,1].set_xlim(plot_minT, plot_maxT)
        axes[0,1].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[0,1].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[0,1].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[0,1].plot(temp_stat_data3,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN')
        axes[0,1].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[0,1].grid()
        #axes[0,1].set_xlabel('Temperature')
        #axes[0,1].set_ylabel('Layer')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(plot_minT+2, 1.8, "Summer", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,1].legend(loc='lower right', borderaxespad=0.)
        
        
        array_size=stat_data4.shape[1]
        m1_size=m1_data4.shape[1]
        m2_size=m2_data4.shape[1]
               
        temp_stat_data_all=np.zeros((array_size),dtype='float64')
        temp_m1_data_all=np.zeros((m1_size),dtype='float64')
        temp_m2_data_all=np.zeros((m2_size),dtype='float64')
        temp_stat_data4 =np.zeros((4),dtype='float64')
        temp_m1_data4   =np.zeros((4),dtype='float64')
        temp_m2_data4   =np.zeros((4),dtype='float64')
        tskin_tsoil_m1=np.zeros((5), dtype='float64')
        tskin_tsoil_m2=np.zeros((5), dtype='float64')
        
        m1_skin_temp_addresses = np.where(m1_data4_tskin > 0.1)
        m1_skin_temp=np.nanmean(m1_data4_tskin[m1_skin_temp_addresses])
        m2_skin_temp_addresses = np.where(m2_data4_tskin > 0.1)
        m2_skin_temp=np.nanmean(m2_data4_tskin[m2_skin_temp_addresses])
        SAT_skin_temp_addresses = np.where(Sat_data4 > 260.0)
        SAT_skin_temp=np.nanmean(Sat_data4[SAT_skin_temp_addresses])
        
        for num_layer in range (0,4,1):
            temp_stat_data_all[:]=stat_data4[num_layer,:]
            temp_stat_data44=np.where(temp_stat_data_all > 0.01)
            temp_stat_data4[num_layer]=np.nanmean(temp_stat_data_all[temp_stat_data44])
            if num_layer < 4:
                temp_m1_data_all[:]=m1_data4[num_layer,:]
                temp_m1_data44=np.where(temp_m1_data_all > 0.01)
                temp_m1_data4[num_layer]=np.nanmean(temp_m1_data_all[temp_m1_data44])
        
                temp_m2_data_all[:]=m2_data4[num_layer,:]
                temp_m2_data44=np.where(temp_m2_data_all > 0.01)
                temp_m2_data4[num_layer]=np.nanmean(temp_m2_data_all[temp_m2_data44])
    
        tskin_tsoil_m1[0]=m1_skin_temp
        tskin_tsoil_m1[1:5]=temp_m1_data4[:]
        tskin_tsoil_m2[0]=m2_skin_temp
        tskin_tsoil_m2[1:5]=temp_m2_data4[:]
        
        axes[1,1].set_ylim(2.0, 0.0)
        axes[1,1].set_xlim(plot_minT, plot_maxT)
        axes[1,1].plot([273.15,273.15],[0,2.0], '-',color="blue")
        axes[1,1].plot(tskin_tsoil_m1,Noah_tskin_tsoil_depths, lw=1.0, linestyle='dashed', marker='o', markersize=8.0, color="blue", label='Noah 3.6')
        axes[1,1].plot(tskin_tsoil_m2,Noah_tskin_tsoil_depths, lw=1.0, linestyle='solid', marker='.', markersize=8.0, color="red", label='Noah MP')
        axes[1,1].plot(temp_stat_data4,SCAN_layer_depths, lw=1.0, linestyle='solid', marker='*', markersize=8.0, label='ISMN', color="green")
        axes[1,1].plot(SAT_skin_temp,0.02, lw=1.0, linestyle='solid', marker='d', label='ISCCP', color="orange")
        axes[1,1].grid()
        axes[1,1].set_xlabel('Temperature (K)')
        #axes[1,1].set_ylabel('Layer Depth (m)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(plot_minT+2, 1.8, "Fall", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,1].legend(loc='lower right', borderaxespad=0.)
        
        plt.suptitle(ptitle, fontsize=18)
        plt.savefig(fname_out)
        plt.close(figure)

