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
from matplotlib.dates import DateFormatter

def plot_8_panel(fdata1, fdata2, fdata3, fdata4, fdata5, fdata6, fdata7, fdata8, fcdata1, fcdata2, fcdata3, fcdata4, fcdata5, fcdata6, fcdata7, fcdata8,
                mdata1, mdata2, mdata3, mdata4, mdata5, mdata6, mdata7, mdata8, mcdata1, mcdata2, mcdata3, mcdata4, mcdata5, mcdata6, mcdata7, mcdata8,
                sdata1, sdata2, sdata3, sdata4, sdata5, sdata6, sdata7, sdata8, ptitle, fname_out, total_num_recs, date_array):

    panel_labels=['00 UTC', '03 UTC' ,'06 UTC', '09 UTC', '12 UTC', '15 UTC' ,'18 UTC', '21 UTC' ]

    ################################################################################################
    ################################################################################################
    #  This section creates time contour plots
    ################################################################################################
    ################################################################################################


    #[ 5cm, 10cm, 20cm, 50, 101.6 - depths in cm] - SCAN DEPTHS
    #[0-10cm, 10-40cm, 40-100cm, 100-200cm ] Noah Depths
            
    Temp_SCAN_4LYR1=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR2=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR3=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR4=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR5=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR6=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR7=np.zeros((4, total_num_recs), dtype=np.float64)
    Temp_SCAN_4LYR8=np.zeros((4, total_num_recs), dtype=np.float64)
    print (Temp_SCAN_4LYR1.shape, sdata1.shape)
    
    Temp_SCAN_4LYR1[0,:]=sdata1[0,:]
    Temp_SCAN_4LYR1[1,:]=sdata1[1,:]  #(sdata1[1,:] + sdata1[2,:])/2
    Temp_SCAN_4LYR1[2,:]=sdata1[2,:]
    Temp_SCAN_4LYR1[3,:]=sdata1[3,:]
    
    Temp_SCAN_4LYR2[0,:]=sdata2[0,:]
    Temp_SCAN_4LYR2[1,:]=sdata2[1,:] #  (sdata2[1,:] + sdata2[2,:])/2
    Temp_SCAN_4LYR2[2,:]=sdata2[2,:]
    Temp_SCAN_4LYR2[3,:]=sdata2[3,:]

    Temp_SCAN_4LYR3[0,:]=sdata3[0,:]
    Temp_SCAN_4LYR3[1,:]=sdata3[1,:] # (sdata3[1,:] + sdata3[2,:])/2
    Temp_SCAN_4LYR3[2,:]=sdata3[2,:]
    Temp_SCAN_4LYR3[3,:]=sdata3[3,:]

    Temp_SCAN_4LYR4[0,:]=sdata4[0,:]
    Temp_SCAN_4LYR4[1,:]=sdata4[1,:] # (sdata4[1,:] + sdata4[2,:])/2
    Temp_SCAN_4LYR4[2,:]=sdata4[2,:]
    Temp_SCAN_4LYR4[3,:]=sdata4[3,:]

    Temp_SCAN_4LYR5[0,:]=sdata5[0,:]
    Temp_SCAN_4LYR5[1,:]=sdata5[1,:] # (sdata5[1,:] + sdata5[2,:])/2
    Temp_SCAN_4LYR5[2,:]=sdata5[2,:]
    Temp_SCAN_4LYR5[3,:]=sdata5[3,:]

    Temp_SCAN_4LYR6[0,:]=sdata6[0,:]
    Temp_SCAN_4LYR6[1,:]=sdata6[1,:] # (sdata6[1,:] + sdata6[2,:])/2
    Temp_SCAN_4LYR6[2,:]=sdata6[2,:]
    Temp_SCAN_4LYR6[3,:]=sdata6[3,:]
    
    Temp_SCAN_4LYR7[0,:]=sdata7[0,:]
    Temp_SCAN_4LYR7[1,:]=sdata7[1,:] # (sdata7[1,:] + sdata7[2,:])/2
    Temp_SCAN_4LYR7[2,:]=sdata7[2,:]
    Temp_SCAN_4LYR7[3,:]=sdata7[3,:]
    
    Temp_SCAN_4LYR8[0,:]=sdata8[0,:]
    Temp_SCAN_4LYR8[1,:]=sdata8[1,:] # (sdata8[1,:] + sdata8[2,:])/2
    Temp_SCAN_4LYR8[2,:]=sdata8[2,:]
    Temp_SCAN_4LYR8[3,:]=sdata8[3,:]

    #SCAN-NoahMP_OL
    SCANmNoahMP1=Temp_SCAN_4LYR1[:,:]-mdata1[:,:]
    SCANmNoahMP2=Temp_SCAN_4LYR2[:,:]-mdata2[:,:]
    SCANmNoahMP3=Temp_SCAN_4LYR3[:,:]-mdata3[:,:]
    SCANmNoahMP4=Temp_SCAN_4LYR4[:,:]-mdata4[:,:]
    SCANmNoahMP5=Temp_SCAN_4LYR5[:,:]-mdata5[:,:]
    SCANmNoahMP6=Temp_SCAN_4LYR6[:,:]-mdata6[:,:]
    SCANmNoahMP7=Temp_SCAN_4LYR7[:,:]-mdata7[:,:]
    SCANmNoahMP8=Temp_SCAN_4LYR8[:,:]-mdata8[:,:]
    
    #SCAN-NoahMP_DA
    SCANmNoahMP1_DA=Temp_SCAN_4LYR1[:,:]-mcdata1[:,:]
    SCANmNoahMP2_DA=Temp_SCAN_4LYR2[:,:]-mcdata2[:,:]
    SCANmNoahMP3_DA=Temp_SCAN_4LYR3[:,:]-mcdata3[:,:]
    SCANmNoahMP4_DA=Temp_SCAN_4LYR4[:,:]-mcdata4[:,:]
    SCANmNoahMP5_DA=Temp_SCAN_4LYR5[:,:]-mcdata5[:,:]
    SCANmNoahMP6_DA=Temp_SCAN_4LYR6[:,:]-mcdata6[:,:]
    SCANmNoahMP7_DA=Temp_SCAN_4LYR7[:,:]-mcdata7[:,:]
    SCANmNoahMP8_DA=Temp_SCAN_4LYR8[:,:]-mcdata8[:,:]


    #SCAN-Noah36_OL
    SCANmNoah361=Temp_SCAN_4LYR1[:,:]-fdata1[:,:]
    SCANmNoah362=Temp_SCAN_4LYR2[:,:]-fdata2[:,:]
    SCANmNoah363=Temp_SCAN_4LYR3[:,:]-fdata3[:,:]
    SCANmNoah364=Temp_SCAN_4LYR4[:,:]-fdata4[:,:]
    SCANmNoah365=Temp_SCAN_4LYR5[:,:]-fdata5[:,:]
    SCANmNoah366=Temp_SCAN_4LYR6[:,:]-fdata6[:,:]
    SCANmNoah367=Temp_SCAN_4LYR7[:,:]-fdata7[:,:]
    SCANmNoah368=Temp_SCAN_4LYR8[:,:]-fdata8[:,:]
    
    #SCAN-Noah36_DA
    SCANmNoah361_DA=Temp_SCAN_4LYR1[:,:]-fcdata1[:,:]
    SCANmNoah362_DA=Temp_SCAN_4LYR2[:,:]-fcdata2[:,:]
    SCANmNoah363_DA=Temp_SCAN_4LYR3[:,:]-fcdata3[:,:]
    SCANmNoah364_DA=Temp_SCAN_4LYR4[:,:]-fcdata4[:,:]
    SCANmNoah365_DA=Temp_SCAN_4LYR5[:,:]-fcdata5[:,:]
    SCANmNoah366_DA=Temp_SCAN_4LYR6[:,:]-fcdata6[:,:]
    SCANmNoah367_DA=Temp_SCAN_4LYR7[:,:]-fcdata7[:,:]
    SCANmNoah368_DA=Temp_SCAN_4LYR8[:,:]-fcdata8[:,:]
    
    #Noah36 OPEN LOOP VERSUS DA LOOP
    DAcompareNoah361=fdata1[:,:]-fcdata1[:,:]
    DAcompareNoah362=fdata2[:,:]-fcdata2[:,:]
    DAcompareNoah363=fdata3[:,:]-fcdata3[:,:]
    DAcompareNoah364=fdata4[:,:]-fcdata4[:,:]
    DAcompareNoah365=fdata5[:,:]-fcdata5[:,:]
    DAcompareNoah366=fdata6[:,:]-fcdata6[:,:]
    DAcompareNoah367=fdata7[:,:]-fcdata7[:,:]
    DAcompareNoah368=fdata8[:,:]-fcdata8[:,:]
    
    #NoahMP OPEN LOOP VERSUS DA LOOP
    DAcompareNoahMP1=mdata1[:,:]-mcdata1[:,:]
    DAcompareNoahMP2=mdata2[:,:]-mcdata2[:,:]
    DAcompareNoahMP3=mdata3[:,:]-mcdata3[:,:]
    DAcompareNoahMP4=mdata4[:,:]-mcdata4[:,:]
    DAcompareNoahMP5=mdata5[:,:]-mcdata5[:,:]
    DAcompareNoahMP6=mdata6[:,:]-mcdata6[:,:]
    DAcompareNoahMP7=mdata7[:,:]-mcdata7[:,:]
    DAcompareNoahMP8=mdata8[:,:]-mcdata8[:,:]
    
    temp1=mdata1[:,:]-fdata1[:,:]
    temp2=mdata2[:,:]-fdata2[:,:]
    temp3=mdata3[:,:]-fdata3[:,:]
    temp4=mdata4[:,:]-fdata4[:,:]
    temp5=mdata5[:,:]-fdata5[:,:]
    temp6=mdata6[:,:]-fdata6[:,:]
    temp7=mdata7[:,:]-fdata7[:,:]
    temp8=mdata8[:,:]-fdata8[:,:]

    #create the monthly averages of the profiles
    increments = np.arange(0,total_num_recs)
    
    heights=np.array([-0.05,-0.25,-0.70,-1.5])
    Tincs=[-10,-8,-6,-4,-2,-1,0,1,2,4,6,8,10]
    #print (increments.shape, heights.shape, SCANmNoah361.shape)

    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights, SCANmNoah361[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    
    Z2=axes[0,1].contourf(increments,heights, SCANmNoah362[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, SCANmNoah363[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)
    
    Z4=axes[1,1].contourf(increments,heights, SCANmNoah364[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z5=axes[2,0].contourf(increments,heights, SCANmNoah365[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, SCANmNoah366[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, SCANmNoah367[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, SCANmNoah368[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nISMN Soil Temperature minus Noah 3.6 Soil Temperature', fontsize=18)
    fname_full=fname_out+'SCANminusNoah36_OL.png'

    plt.savefig(fname_full)
    plt.close(figure)
    
    
    
    ######################################################################
    ### OBS MINUS DA Plot
    ######################################################################

    
    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights, SCANmNoah361_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    
    Z2=axes[0,1].contourf(increments,heights, SCANmNoah362_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, SCANmNoah363_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)
    
    Z4=axes[1,1].contourf(increments,heights, SCANmNoah364_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z5=axes[2,0].contourf(increments,heights, SCANmNoah365_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, SCANmNoah366_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, SCANmNoah367_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, SCANmNoah368_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nISMN Soil Temperature minus Noah 3.6 Soil Temperature DA', fontsize=18)
    fname_full=fname_out+'SCANminusNoah36_DA.png'

    plt.savefig(fname_full)
    plt.close(figure)
    
    ######################################################################
    ### Next Plot
    ######################################################################
    
    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights, SCANmNoahMP1[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    Z2=axes[0,1].contourf(increments,heights, SCANmNoahMP2[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, SCANmNoahMP3[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)

    Z4=axes[1,1].contourf(increments,heights, SCANmNoahMP4[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
 
    Z5=axes[2,0].contourf(increments,heights, SCANmNoahMP5[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, SCANmNoahMP6[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, SCANmNoahMP7[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, SCANmNoahMP8[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nISMN Soil Temperature minus Noah MP Soil Temperature', fontsize=18)
    fname_full=fname_out+'SCANminusNoahMP_OL.png'
    plt.savefig(fname_full)
    plt.close(figure)

    ######################################################################
    ### Next Plot
    ######################################################################
    
    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights, SCANmNoahMP1_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    Z2=axes[0,1].contourf(increments,heights, SCANmNoahMP2_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, SCANmNoahMP3_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)

    Z4=axes[1,1].contourf(increments,heights, SCANmNoahMP4_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
 
    Z5=axes[2,0].contourf(increments,heights, SCANmNoahMP5_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, SCANmNoahMP6_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, SCANmNoahMP7_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, SCANmNoahMP8_DA[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nISMN Soil Temperature minus Noah MP Soil Temperature', fontsize=18)
    fname_full=fname_out+'SCANminusNoahMP_DA.png'
    plt.savefig(fname_full)
    plt.close(figure)

    ######################################################################
    ### Next Plot
    ######################################################################
    
    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights, temp1[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    Z2=axes[0,1].contourf(increments,heights, temp2[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, temp3[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)
    
    Z4=axes[1,1].contourf(increments,heights, temp4[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z5=axes[2,0].contourf(increments,heights, temp5[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, temp6[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, temp7[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, temp8[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nNoah 36 minus Noah MP Soil Temperature', fontsize=18)
    fname_full=fname_out+'Noah36minusNoahMP.png'
    plt.savefig(fname_full)
    plt.close(figure)
    
    
    ######################################################################
    ### DA Comparison Plot #1
    ######################################################################
    
    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights,DAcompareNoah361[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    Z2=axes[0,1].contourf(increments,heights, DAcompareNoah362[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, DAcompareNoah363[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)
    
    Z4=axes[1,1].contourf(increments,heights, DAcompareNoah364[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z5=axes[2,0].contourf(increments,heights, DAcompareNoah365[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, DAcompareNoah366[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, DAcompareNoah367[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, DAcompareNoah368[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nNoah 3.6 Open Loop minus Noah 3.6 DA Soil Temperature', fontsize=18)
    fname_full=fname_out+'Noah36OLminusNoah36DA.png'
    plt.savefig(fname_full)
    plt.close(figure)
    
    ######################################################################
    ### DA Comparison Plot #2
    ######################################################################
    
    figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(12,12))
    Z1=axes[0,0].contourf(increments,heights,DAcompareNoahMP1[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    axes[0,0].set_xticks(increments[::90])
    axes[0,0].set_xticklabels([])
    plt.subplots_adjust(right=.88)
    pos=axes[0,0].get_position()
    axes[0,0].set_ylabel('Soil Depth (meters)')
    axes[0,0].set_title(panel_labels[0],color='blue', fontsize=12.0)
    
    Z2=axes[0,1].contourf(increments,heights, DAcompareNoahMP2[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[0,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[0,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[0,1].set_xticks(increments[::90])
    axes[0,1].set_xticklabels([])
    pos=axes[0,1].get_position()
    axes[0,1].set_title(panel_labels[1],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z2,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z3=axes[1,0].contourf(increments,heights, DAcompareNoahMP3[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[1,0].set_ylabel('Soil Depth (meters)')
    axes[1,0].set_xticks(increments[::90])
    axes[1,0].set_xticklabels([])
    pos=axes[1,0].get_position()
    axes[1,0].set_title(panel_labels[2],color='blue', fontsize=12.0)
    
    Z4=axes[1,1].contourf(increments,heights, DAcompareNoahMP4[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[1,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[1,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[1,1].get_position()
    axes[1,1].set_xticks(increments[::90])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_title(panel_labels[3],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z4,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z5=axes[2,0].contourf(increments,heights, DAcompareNoahMP5[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[2,0].set_ylabel('Soil Depth (meters)')
    axes[2,0].set_xticks(increments[::90])
    axes[2,0].set_xticklabels([])
    pos=axes[2,0].get_position()
    axes[2,0].set_title(panel_labels[4],color='blue', fontsize=12.0)
    
    Z6=axes[2,1].contourf(increments,heights, DAcompareNoahMP6[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[2,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[2,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[2,1].get_position()
    axes[2,1].set_xticks(increments[::90])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_title(panel_labels[5],color='blue', fontsize=12.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z6,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    
    Z7=axes[3,0].contourf(increments,heights, DAcompareNoahMP7[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,0].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,0].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    axes[3,0].set_ylabel('Soil Depth (meters)')
    pos=axes[3,0].get_position()
    axes[3,0].set_xticks(increments[::90])
    axes[3,0].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    axes[3,0].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    axes[3,0].set_title(panel_labels[6],color='blue', fontsize=12.0)
    
    Z8=axes[3,1].contourf(increments,heights, DAcompareNoahMP8[:,:], cmap='jet_r', levels=Tincs,extend='both')
    axes[3,1].plot([0, total_num_recs], [-0.1, -0.1], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-0.4, -0.4], color='black',linestyle='dashed', linewidth=1)
    axes[3,1].plot([0, total_num_recs], [-1.0, -1.0], color='black',linestyle='dashed', linewidth=1)
    plt.subplots_adjust(right=.88)
    pos=axes[3,1].get_position()
    axes[3,1].set_title(panel_labels[7],color='blue', fontsize=12.0)
    axes[3,1].set_xticks(increments[::90])
    axes[3,1].set_xticklabels([datetime.datetime.strftime(ii,"%m-%y") for ii in date_array[::90]], rotation=45, fontsize=7.0)
    cbar_ax=figure.add_axes([pos.x1+0.01,pos.y0,0.015,pos.height])
    cbar=plt.colorbar(Z8,cax=cbar_ax)
    cbar.ax.set_ylabel("Temperature Difference")
    axes[3,1].set_xlabel('Date (mm - yy)', color='black', fontsize=12.0)
    plt.suptitle(ptitle+'\nNoah MP Open Loop minus Noah MP DA Soil Temperature', fontsize=18)
    fname_full=fname_out+'NoahMPOLminusNoahMPDA.png'
    plt.savefig(fname_full)
    plt.close(figure)



