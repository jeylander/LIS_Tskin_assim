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

def generate_8_panel_scatter_plots(fdata1, fdata2, fdata3, fdata4, fdata5, fdata6, fdata7, fdata8, sdata1, sdata2, sdata3, sdata4, sdata5, sdata6, sdata7, sdata8,
                            tdata1, tdata2, tdata3, tdata4, tdata5, tdata6, tdata7, tdata8, ptitle, fname_out, max_num_stations, Xlabel, Ylabel):

    
    #for  graph_loop in range (0, max_num_stations, 1):

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ##################################################################################################################
        
        figure, axes=plt.subplots(ncols=2,nrows=4,  figsize=(15,8))
        axes[0,0].set_ylim(240, 340)
        axes[0,0].set_xlim(240, 340)
        axes[0,0].scatter(fdata1[:,:], sdata1[:,:], marker='+')
        
        
        temp_array_y = sdata1[:,:]
        temp_array_x = fdata1[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[0,0].plot(X_test, y_pred, color='black', linewidth=2)
        
        temp_array_y = tdata1[:,:]
        temp_array_x = fdata1[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[0,0].scatter(fdata1[:,:], tdata1[:,:], marker='.', color='red')
        axes[0,0].plot(X_test, y_pred, color='green', linewidth=2)
        #slope, intercept, r_value, p_value, std_err = stats.linregress(fdata1[:,:], tdata1[:,:])
        #axes[0,0].axline(xy1=(0, intercept), slope=slope, label=f'$y = {slope:.1f}x {intercept:+.1f}$')
        axes[0,0].grid()
        axes[0,0].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(242, 320, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[0,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################

        axes[1,0].set_ylim(240, 340)
        axes[1,0].set_xlim(240, 340)
        axes[1,0].scatter(fdata2[:,:], sdata2[:,:], marker='+')
        temp_array_y = sdata2[:,:]
        temp_array_x = fdata2[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[1,0].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[1,0].scatter(fdata2[:,:], tdata2[:,:], marker='.', color='red')
        temp_array_y = tdata2[:,:]
        temp_array_x = fdata2[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[1,0].plot(X_test, y_pred, color='black', linewidth=2)
        axes[1,0].grid()
        axes[1,0].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,0].text(242, 320, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[1,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################

        axes[2,0].set_ylim(240, 340)
        axes[2,0].set_xlim(240, 340)
        axes[2,0].scatter(fdata3[:,:], sdata3[:,:], marker='+')
        temp_array_y = sdata3[:,:]
        temp_array_x = fdata3[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[2,0].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[2,0].scatter(fdata3[:,:], tdata3[:,:], marker='.', color='red')
        temp_array_y = tdata3[:,:]
        temp_array_x = fdata3[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[2,0].plot(X_test, y_pred, color='black', linewidth=2)
        axes[2,0].grid()
        axes[2,0].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,0].text(242, 320, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[2,0].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

        axes[3,0].set_ylim(240, 340)
        axes[3,0].set_xlim(240, 340)
        axes[3,0].scatter(fdata4[:,:], sdata4[:,:], marker='+')
        temp_array_y = sdata4[:,:]
        temp_array_x = fdata4[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[3,0].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[3,0].scatter(fdata4[:,:], tdata4[:,:], marker='.', color='red')
        temp_array_y = tdata4[:,:]
        temp_array_x = fdata4[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[3,0].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[3,0].grid()
        axes[3,0].set_xlabel(Xlabel)
        axes[3,0].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,0].text(242, 320, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[3,0].legend(loc='upper right', borderaxespad=0.)
        
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ##################################################################################################################

        axes[0,1].set_ylim(240, 340)
        axes[0,1].set_xlim(240, 340)
        axes[0,1].scatter(fdata5[:,:], sdata5[:,:], marker='+')
        temp_array_y = sdata5[:,:]
        temp_array_x = fdata5[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[0,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[0,1].scatter(fdata5[:,:], tdata5[:,:], marker='.', color='red')
        temp_array_y = tdata5[:,:]
        temp_array_x = fdata5[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[0,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[0,1].grid()
        axes[0,1].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(242, 320, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[0,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################

        axes[1,1].set_ylim(240, 340)
        axes[1,1].set_xlim(240, 340)
        axes[1,1].scatter(fdata6[:,:], sdata6[:,:], marker='+')
        temp_array_y = sdata6[:,:]
        temp_array_x = fdata6[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[1,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[1,1].scatter(fdata6[:,:], tdata6[:,:], marker='.', color='red')
        temp_array_y = tdata6[:,:]
        temp_array_x = fdata6[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[1,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[1,1].grid()
        axes[1,1].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(242, 320, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[1,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################

        axes[2,1].set_ylim(240, 340)
        axes[2,1].set_xlim(240, 340)
        axes[2,1].scatter(fdata7[:,:], sdata7[:,:], marker='+')
        temp_array_y = sdata7[:,:]
        temp_array_x = fdata7[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[2,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[2,1].scatter(fdata7[:,:], tdata7[:,:], marker='.', color='red')
        temp_array_y = tdata7[:,:]
        temp_array_x = fdata7[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[2,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[2,1].grid()
        axes[2,1].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,1].text(242, 320, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[2,1].legend(loc='upper right', borderaxespad=0.)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

        axes[3,1].set_ylim(240, 340)
        axes[3,1].set_xlim(240, 340)
        axes[3,1].scatter(fdata8[:,:], sdata8[:,:], marker='+')
        temp_array_y = sdata8[:,:]
        temp_array_x = fdata8[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        axes[3,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[3,1].scatter(fdata8[:,:], tdata8[:,:], marker='.', color='red')
        temp_array_y = tdata8[:,:]
        temp_array_x = fdata8[:,:]
        new_temp_array_y=temp_array_y[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        new_temp_array_x=temp_array_x[(temp_array_y>=220.0) & (temp_array_y <= 350.0) & (temp_array_x > 220.0)]
        x=new_temp_array_x.reshape(new_temp_array_x.shape[0],1)
        y=new_temp_array_y.reshape(new_temp_array_y.shape[0])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred=regressor.predict(X_test)
        
        axes[3,1].plot(X_test, y_pred, color='black', linewidth=2)
        
        axes[3,1].grid()
        axes[3,1].set_xlabel(Xlabel)
        axes[3,1].set_ylabel(Ylabel)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,1].text(242, 320, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        #axes[3,1].legend(loc='upper right', borderaxespad=0.)
        
        plt.suptitle(ptitle, fontsize=18)
        plt.savefig(fname_out)
        plt.close(figure)
        #station_number+=1

