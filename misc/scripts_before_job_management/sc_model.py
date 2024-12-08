#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:43:30 2020

@author: zacharie
"""

import sys
import os
import errno
import re
from datetime import datetime, timedelta, date
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as opti
from scipy.stats import mstats
import shutil
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from pyproj import Proj, transform
import glob
import random
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from math import sqrt
from matplotlib.ticker import PercentFormatter
from sc_utils import *
from sklearn.metrics import confusion_matrix



def calibration(p,out,perc_cal):


                
    path_outputs = p["path_outputs"]
    dataSetDir = os.path.join(path_outputs,out)
    path_tifs = os.path.join(dataSetDir,"TIFS")


    path_cal = os.path.join(dataSetDir,"CALIBRATION")
     
    NDSIALL = []
    FSCALL = []

    shutil.rmtree(path_cal, ignore_errors=True)
    mkdir_p(path_cal)

    f= open(os.path.join(path_cal,out + "_CALIBRATION_SUMMARY.txt"),"w")
    f.write("\nDates :")
    nb_dates = 0
    
        
        
    for d in sorted(os.listdir(path_tifs)):
        date = getDateFromStr(d)
        if date == '' : continue
        print(date)
        path_tifs_date = os.path.join(path_tifs,d)
        
        
        epsgs = {}
        for tif in os.listdir(path_tifs_date) :
            epsg = getEpsgFromStr(tif)
            if epsg == '': continue
            if epsg not in epsgs :
                epsgs[epsg] = []
                
        tiles = []
        for tif in os.listdir(path_tifs_date) :
            epsg = getEpsgFromStr(tif)
            tile = getTileFromStr(tif)
            if epsg == '' or tile == '': continue
            if tile not in epsgs[epsg]:
                epsgs[epsg].append(tile)      
                
        
        
        for epsg in epsgs :
            for tile in epsgs[epsg]:
                g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                FSCALL.append(BandReadAsArray(g_FSC.GetRasterBand(1)).flatten())
                g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                NDSIALL.append(BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten())
                
        f.write("\n      " + d)
        nb_dates += 1
    
    print("Eliminate Nodata pixels")
    NDSIALL = np.hstack(NDSIALL)
    FSCALL = np.hstack(FSCALL)  
    cond1 = np.where((FSCALL != 255) & (~np.isnan(FSCALL)) & (~np.isinf(FSCALL)))
    NDSIALL = NDSIALL[cond1]
    FSCALL = FSCALL[cond1]
    
    cond2 = np.where( (NDSIALL != 255) & (~np.isnan(NDSIALL)) & (~np.isinf(NDSIALL)))
    FSCALL = FSCALL[cond2]
    NDSIALL = NDSIALL[cond2]
    

        
    if len(FSCALL) < 2 : 
        f.close()
        shutil.rmtree(path_cal, ignore_errors=True)
        print("ERROR calibrateModel : dataSet too small")
        return False


    NDSI_train, NDSI_test, FSC_train, FSC_test = train_test_split(NDSIALL, FSCALL, test_size=perc_cal)

    
    #CALIBRATION
    print("CALIBRATION")
    fun = lambda x: sqrt(mean_squared_error(0.5*np.tanh(x[0]*NDSI_train+x[1])+0.5,FSC_train))
    
    model = opti.minimize(fun,(3.0,-1.0),method = 'Nelder-Mead')#method = 'Nelder-Mead')

    a = model.x[0]
    b = model.x[1]
    success = model.success
    rmse_cal = model.fun
    print("CALIBRATION SUCCESS : ",success)
    print("CALIBRATION RMSE : ",rmse_cal)
    
    

    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("CALIBRATION WITH " + out,size = 16)
    # set up subplot grid
    gridspec.GridSpec(2,2)
    
    # 2D histo de calibration
    ax = plt.subplot2grid((2,2), (0,0))
    
    plt.title("CALIBRATION WITH THE TRAINING SET",size = 14,y=1.08)
    plt.ylabel('FSC ('+out+')',size = 14)
    plt.xlabel('NDSI (S2)',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(NDSI_train,FSC_train,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'), norm=LogNorm())
    
    n = np.arange(min(NDSI_train),1.01,0.01)
    
    line = 0.5*np.tanh(a*n+b) +  0.5

    plt.plot(n, line, 'r')
    


    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  



    # VALIDATION
    
    # prediction of FSC from testing NDSI
    FSC_pred = 0.5*np.tanh(a*NDSI_test+b) +  0.5
    

    # error
    er_FSC = FSC_pred - FSC_test

    # absolute error
    abs_er_FSC = abs(er_FSC)

    # mean error
    m_er_FSC = np.mean(er_FSC)

    # absolute mean error
    abs_m_er_FSC = np.mean(abs_er_FSC)

    #root mean square error
    rmse_FSC = sqrt(mean_squared_error(FSC_pred,FSC_test))

    #correlation
    corr_FSC = mstats.pearsonr(FSC_pred,FSC_test)[0]

    #standard deviation
    stde_FSC = np.std(er_FSC)


    #correlation, erreur moyenne, ecart-type, rmse

    # 2D histo de validation
    ax = plt.subplot2grid((2,2), (0,1))
    
    plt.title("VALIDATION WITH THE TESTING SET",size = 14,y=1.08)
    plt.ylabel('Predicted FSC (S2)',size = 14)
    plt.xlabel('FSC ('+out+')',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(FSC_test,FSC_pred,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    slope, intercept, r_value, p_value, std_err = mstats.linregress(FSC_test,FSC_pred) 
    n = np.array([min(FSC_test),1.0])
    line = slope * n + intercept

    plt.plot(n, line, 'b')#, label='y = {:.2f}x + {:.2f}\ncorr={:.2f} rmse={:.2f}'.format(slope,intercept,corr_FSC,rmse_FSC))
    plt.plot(n, n, 'g')#, label='y = 1.0x + 0.0')

    #plt.legend(fontsize=10,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  

    # 1D histo de residus
    ax = plt.subplot2grid((2,2), (1,0),rowspan=1, colspan=2)
    plt.title("FSC RESIDUALS",size = 14,y=1.08)
    plt.ylabel('Percent of data points',size = 14)
    plt.xlabel('predicted FSC (S2) - FSC ('+out+')',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    xticks = np.arange(-1.0, 1.1, 0.1)
    plt.xticks(xticks)
    plt.hist(er_FSC,bins=40,weights=np.ones(len(er_FSC)) / len(er_FSC))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.grid(True) 


    # fit subplots & save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_cal,'PLOT_CAL_' + out + '.png'))
    plt.close(fig)




    minFSC = min(FSC_test)
    
    k = 1.0
    j = 1.1

    list_var_FSC_box = [np.var(er_FSC[np.where((FSC_test>= k) & (FSC_test <= j))])]
    list_var_labels_box = ["[1]"]
    j = round(j - 0.1,1)
    k = round(k - 0.1,1)

    while j > minFSC: 

        list_var_FSC_box.insert(0,np.var(er_FSC[np.where((FSC_test >= k) & (FSC_test < j))]))
        list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(k) +"\n"+ "{0:.1f}".format(j) +" [")
        j = round(j - 0.1,1)
        k = round(k - 0.1,1)
        


    # Plot figure with subplots 
    fig = plt.figure()
    #st = fig.suptitle("FSC RESIDUALS",size = 16)
    gridspec.GridSpec(1,2)
    
    
    # boxplot avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
    #plt.title('ODK FSC/NDSI',size = 14,y=1.08)
    plt.ylabel('Variance',size = 14)
    plt.xlabel('FSC',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.bar(list_var_labels_box,list_var_FSC_box)


    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_cal,'PLOT_CAL_RESIDUAL_VAR_' + out + '.png'))
    plt.close(fig)

    f.write("\n")
    f.write("\nCALIBRATION" )
    f.write("\n Number of 20x20m data points : " + str(len(NDSI_train)))
    f.write("\n lin. reg. NDSI on FSC : 0.5*tanh(a*NDSI+b)+0.5 : a = " + str(a) + " ; b = " + str(b))
    f.write("\n root mean square err. : " + str(rmse_cal))

        
    f.write("\n")
    
    f.write("\nVALIDATION" )
    f.write("\n  Number of 20x20m data points : " + str(len(NDSI_test)))
    f.write("\n  corr. coef. : " + str(corr_FSC))
    f.write("\n  std. err. : " + str(stde_FSC))
    f.write("\n  mean err. : " + str(m_er_FSC))
    f.write("\n  abs. mean err. : " + str(abs_m_er_FSC))
    f.write("\n  root mean square err. : " + str(rmse_FSC))

    f.close()

    results = open(os.path.join(path_cal,"CALIBRATION_PARAMS.txt"),"w")
    results.write("a b\n")
    results.write(str(a) + " " + str(b))
    results.close()


    


    return True





def evaluation(p,out_cal,out_val,eval_modes,eval_res):
    
    
    
    path_outputs = p["path_outputs"]
    calDataSetDir = os.path.join(path_outputs,out_cal)
    path_eval = os.path.join(calDataSetDir,"EVALUATION")
    evalDataSetDir = os.path.join(path_outputs,out_val)
    path_tifs = os.path.join(evalDataSetDir,"TIFS")
    path_eval_dir = os.path.join(path_eval,out_val)
    path_params = os.path.join(calDataSetDir,"CALIBRATION","CALIBRATION_PARAMS.txt")

    a = 0
    b = 0
    with open(path_params, "r") as params :
        line = params.readline()
        line = params.readline()
        ab = line.split()
        a = float(ab[0])
        b = float(ab[1])

    mkdir_p(path_eval_dir)
    
    
    FSC_pred_all = []
    NDSI_test_all = []
    FSC_test_all = []
    list_NDSI_test = []
    list_FSC_test = []
    list_FSC_pred = []
    list_dates_test = []
    dates = []
    for d in sorted(os.listdir(path_tifs)):
        date = getDateFromStr(d)
        if date == '' : continue
        print(date)
        path_tifs_date = os.path.join(path_tifs,d)
            
            
        epsgs = {}
        for tif in os.listdir(path_tifs_date) :
            epsg = getEpsgFromStr(tif)
            if epsg != "" and epsg not in epsgs :
                epsgs[epsg] = []
                
        tiles = []
        for tif in os.listdir(path_tifs_date) :
            epsg = getEpsgFromStr(tif)
            tile = getTileFromStr(tif)
            if epsg == '' or tile == '': continue
            if tile not in epsgs[epsg]:
                epsgs[epsg].append(tile)      
                
        
        NDSI_test_1date = []
        FSC_test_1date = []
        FSC_pred_1date = []
        for epsg in epsgs :
            for tile in epsgs[epsg]:
                g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                NDSI = BandReadAsArray(g_NDSI.GetRasterBand(1))
                
                FSC_pred = 0.5*np.tanh(a*NDSI+b) +  0.5
                FSC_pred[NDSI == 255] = 255
                g_FSC_pred = gdal.Translate('',g_NDSI,format= 'MEM')
                g_FSC_pred.GetRasterBand(1).WriteArray(FSC_pred)
                if eval_res > 20:
                    g_FSC = gdal.Warp('',g_FSC,format= 'MEM',resampleAlg="average",xRes= eval_res,yRes= eval_res)
                    g_FSC_pred = gdal.Warp('',g_FSC_pred,format= 'MEM',resampleAlg="average",xRes= eval_res,yRes= eval_res)
                    
                    g_NDSI = gdal.Warp('',g_NDSI,format= 'MEM',resampleAlg="average",xRes= eval_res,yRes= eval_res)
                    
                    
                
                NDSI = BandReadAsArray(g_NDSI.GetRasterBand(1))
                FSC = BandReadAsArray(g_FSC.GetRasterBand(1))
                FSC_pred = BandReadAsArray(g_FSC_pred.GetRasterBand(1))
                NDSI = NDSI.flatten()
                FSC = FSC.flatten()
                FSC_pred = FSC_pred.flatten()
                cond1 = np.where((FSC <= 1) & (~np.isnan(FSC)) & (~np.isinf(FSC)))
                NDSI = NDSI[cond1]
                FSC_pred = FSC_pred[cond1]
                FSC = FSC[cond1]
                cond2 = np.where( (NDSI <= 1) & (~np.isnan(NDSI)) & (~np.isinf(NDSI)))
                FSC = FSC[cond2]
                FSC_pred = FSC_pred[cond2] 
                NDSI = NDSI[cond2] 
                cond3 = np.where( (FSC_pred <= 1) & (~np.isnan(FSC_pred)) & (~np.isinf(FSC_pred)))
                FSC = FSC[cond2]
                NDSI = NDSI[cond2] 
                FSC_pred = FSC_pred[cond2] 
                if len(FSC) > 0 :
                    FSC_test_1date.append(FSC)
                    FSC_pred_1date.append(FSC_pred)
                    NDSI_test_1date.append(NDSI)
        if len(FSC_test_1date) > 0 :
            FSC_test_all.append(np.hstack(FSC_test_1date))
            FSC_pred_all.append(np.hstack(FSC_pred_1date))
            NDSI_test_all.append(np.hstack(NDSI_test_1date))
            dates.append(date.strftime("%Y-%m-%d"))
            
    
    if len(FSC_test_all) == 0 :
        print("WARNING evaluateModel : no data could be extracted")
        return False



    list_NDSI_test = NDSI_test_all
    list_FSC_test = FSC_test_all
    list_FSC_pred = FSC_pred_all
    list_dates_test = dates
    
    

        
    f= open(os.path.join(path_eval_dir,out_cal+"_EVAL_WITH_"+out_val+".txt"),"w")
    f.write("\nCalibration dataset : " + out_cal)
    f.write("\nModel : FSC = 0.5*tanh(a*NDSI+b) +  0.5 with :")
    f.write("\n        a = " + str(a) + " b = " + str(b))
    f.write("\nEvaluation dataSet : " + out_val)
    f.write("\nNB of "+out_val+" dates : " + str(len(list_dates_test)))
    
    if "all" in eval_modes :
        f.write("\n\n\n")
        f.write("EVALUATION FOR THE PERIOD")
        f = evaluation_all(f,a,b,list_NDSI_test,list_FSC_test,list_FSC_pred,list_dates_test,path_eval_dir,out_cal,out_val)
        
    if "average" in eval_modes :
        #f.write("\n\n\n")
        #f.write("EVALUATION WITH AVERAGE FSC FOR EACH DATE")
        #f = evaluation_average(f,a,b,list_NDSI_test,list_FSC_test,list_FSC_pred,list_dates_test,path_eval_dir,out_cal,out_val)
        timeLapseEval(p,out_cal,out_val)
                
    if "separate" in eval_modes :
        f.write("\n\n\n")
        f.write("EVALUATION FOR EACH DATE")
        f = evaluation_separate(f,a,b,list_NDSI_test,list_FSC_test,list_FSC_pred,list_dates_test,path_eval_dir,out_cal,out_val)
    


        

                  
    #close txt file
    f.close()





def evaluation_separate(f,a,b,list_NDSI_test,list_FSC_test,list_FSC_pred,list_dates_test,path_eval_dir,out_cal,out_val):
    
    for i in np.arange(len(list_NDSI_test)):
        NDSI_test = list_NDSI_test[i]
        FSC_test = list_FSC_test[i]
        FSC_pred =  list_FSC_pred[i]
        date_test = list_dates_test[i]
        path_eval_date_dir = os.path.join(path_eval_dir,date_test)
        mkdir_p(path_eval_date_dir)
        # VALIDATION
  

        # error
        er_FSC = FSC_pred - FSC_test

        # absolute error
        abs_er_FSC = abs(er_FSC)

        # mean error
        m_er_FSC = np.mean(er_FSC)

        # absolute mean error
        abs_m_er_FSC = np.mean(abs_er_FSC)

        #root mean square error
        rmse_FSC = sqrt(mean_squared_error(FSC_pred,FSC_test))

        #correlation
        corr_FSC = mstats.pearsonr(FSC_pred,FSC_test)[0]

        #standard deviation
        stde_FSC = np.std(er_FSC)


        # Plot figure with subplots 
        fig = plt.figure()
        st = fig.suptitle("EVALUATION OF "+out_cal+" CALIBRATION WITH "+out_val,size = 16)
        # set up subplot grid
        gridspec.GridSpec(2,2)


        # 2D histos de FSC vs NDSI
        ax = plt.subplot2grid((2,2), (0,0))
        plt.title("FSC/NDSI TESTING SET",size = 14,y=1.08)
        plt.ylabel('FSC (' +out_val+')' ,size = 14)
        plt.xlabel('NDSI (S2)',size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.hist2d(NDSI_test,FSC_test,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
        n = np.arange(min(NDSI_test),1.01,0.01)
        line = 0.5*np.tanh(a*n+b) +  0.5
        plt.plot(n, line, 'r', label='Predicted FSC (S2)')
        plt.legend(fontsize=10,loc='upper left')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=12)
        ratio = 1
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio) 

        # 2D histos de validation
        ax = plt.subplot2grid((2,2), (0,1))
        plt.title("VALIDATION WITH THE TESTING SET",size = 14,y=1.08)
        plt.ylabel('predicted FSC (S2)',size = 14)
        plt.xlabel('FSC (' +out_val+')' ,size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.hist2d(FSC_test,FSC_pred,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
        slope, intercept, r_value, p_value, std_err = mstats.linregress(FSC_test,FSC_pred) 
        n = np.array([min(FSC_test),1.0])
        line = slope * n + intercept
        plt.plot(n, line, 'b', label='correlation={:.2f}\nrmse={:.2f}'.format(corr_FSC,rmse_FSC))
        plt.plot(n, n, 'g')#, label='y = 1.0x + 0.0')
        plt.legend(fontsize=10,loc='upper left')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=12)
        ratio = 1
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()    
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  


        # 1D histo de residus
        ax = plt.subplot2grid((2,2), (1,0),rowspan=1, colspan=2)
        plt.title("FSC RESIDUALS")
        plt.ylabel('percent of data points',size = 14)
        plt.xlabel('FSC (S2) - FSC ('+out_val+')',size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)    
        xticks = np.arange(-1.0, 1.1, 0.1)
        plt.xticks(xticks)
        plt.hist(er_FSC,bins=40,weights=np.ones(len(er_FSC)) / len(er_FSC))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.grid(True) 


        # fit subplots & save fig
        fig.tight_layout()
        fig.set_size_inches(w=16,h=10)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(path_eval_date_dir,out_cal+"_EVAL_WITH_"+out_val+"_"+date_test+'.png'))
        plt.close(fig)
        
        
        
        minFSC = min(FSC_test)
        
        k = 1.0
        j = 1.1

        list_var_FSC_box = [np.var(er_FSC[np.where((FSC_test>= k) & (FSC_test <= j))])]
        list_var_labels_box = ["[1]"]
        j = round(j - 0.1,1)
        k = round(k - 0.1,1)

        while j > minFSC: 
 
            list_var_FSC_box.insert(0,np.var(er_FSC[np.where((FSC_test >= k) & (FSC_test < j))]))
            list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(k) +"\n"+ "{0:.1f}".format(j) +" [")
            j = round(j - 0.1,1)
            k = round(k - 0.1,1)
           


        # Plot figure with subplots 
        fig = plt.figure()
        gridspec.GridSpec(1,2)
        
        
        # boxplot avec FSC = 0 et FSC = 1
        ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
        plt.title('Variance of FSC (S2) - FSC ('+out_val+')',size = 14,y=1.08)
        plt.ylabel('Variance',size = 14)
        plt.xlabel('FSC ('+out_val+')',size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.bar(list_var_labels_box,list_var_FSC_box)


        # fit subplots and save fig
        fig.tight_layout()
        fig.set_size_inches(w=16,h=10)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(path_eval_date_dir,out_cal+"_EVAL_WITH_"+out_val+"_RESIDUAL_VAR_"+date_test+'.png'))
        plt.close(fig)
        
        f.write("\n\n")
    
        f.write("\nEVALUATION WITH " + date_test )
        f.write("\n  Number of 20x20m data points : " + str(len(NDSI_test)))
        f.write("\n  corr. coef. : " + str(corr_FSC))
        f.write("\n  std. err. : " + str(stde_FSC))
        f.write("\n  mean err. : " + str(m_er_FSC))
        f.write("\n  abs. mean err. : " + str(abs_m_er_FSC))
        f.write("\n  root mean square err. : " + str(rmse_FSC))
        
    return f
    
    
    
    
def evaluation_all(f,a,b,list_NDSI_test,list_FSC_test,list_FSC_pred,list_dates_test,path_eval_dir,out_cal,out_val):
    
    
    
    
    NDSI_test = np.hstack(list_NDSI_test)
    FSC_test = np.hstack(list_FSC_test)
    FSC_pred =  np.hstack(list_FSC_pred)
    dates_test = list_dates_test[0]+"_"+list_dates_test[-1]
    
    

    # VALIDATION

  
    # error
    er_FSC = FSC_pred - FSC_test

    # absolute error
    abs_er_FSC = abs(er_FSC)

    # mean error
    m_er_FSC = np.mean(er_FSC)

    # absolute mean error
    abs_m_er_FSC = np.mean(abs_er_FSC)

    #root mean square error
    rmse_FSC = sqrt(mean_squared_error(FSC_pred,FSC_test))

    #correlation
    corr_FSC = mstats.pearsonr(FSC_pred,FSC_test)[0]

    #standard deviation
    stde_FSC = np.std(er_FSC)


    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("EVALUATION OF "+out_cal+" CALIBRATION WITH "+out_val,size = 16)
    # set up subplot grid
    gridspec.GridSpec(2,2)


    # 2D histos de FSC vs NDSI
    ax = plt.subplot2grid((2,2), (0,0))
    plt.title("FSC/NDSI TESTING SET",size = 14,y=1.08)
    plt.ylabel('FSC (' +out_val+')' ,size = 14)
    plt.xlabel('NDSI (S2)',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(NDSI_test,FSC_test,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    n = np.arange(min(NDSI_test),1.01,0.01)
    line = 0.5*np.tanh(a*n+b) +  0.5
    plt.plot(n, line, 'r', label='Predicted FSC (S2)')
    plt.legend(fontsize=10,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio) 
    # 2D histos de validation
    ax = plt.subplot2grid((2,2), (0,1))
    plt.title("VALIDATION WITH THE TESTING SET",size = 14,y=1.08)
    plt.ylabel('predicted FSC (S2)',size = 14)
    plt.xlabel('FSC (' +out_val+')' ,size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(FSC_test,FSC_pred,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    slope, intercept, r_value, p_value, std_err = mstats.linregress(FSC_test,FSC_pred) 
    n = np.array([min(FSC_test),1.0])
    line = slope * n + intercept
    plt.plot(n, line, 'b', label='correlation={:.2f}\nrmse={:.2f}'.format(corr_FSC,rmse_FSC))
    plt.plot(n, n, 'g')#, label='y = 1.0x + 0.0')
    plt.legend(fontsize=10,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()    
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  


    # 1D histo de residus
    ax = plt.subplot2grid((2,2), (1,0),rowspan=1, colspan=2)
    plt.title("FSC RESIDUALS")
    plt.ylabel('percent of data points',size = 14)
    plt.xlabel('FSC (S2) - FSC ('+out_val+')',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)    
    xticks = np.arange(-1.0, 1.1, 0.1)
    plt.xticks(xticks)
    plt.hist(er_FSC,bins=40,weights=np.ones(len(er_FSC)) / len(er_FSC))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.grid(True) 


    # fit subplots & save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,out_cal+"_EVAL_WITH_"+out_val+"_"+dates_test+'.png'))
    plt.close(fig)
        
        
        
    minFSC = min(FSC_test)
       
    k = 1.0
    j = 1.1

    list_var_FSC_box = [np.var(er_FSC[np.where((FSC_test>= k) & (FSC_test <= j))])]
    list_var_labels_box = ["[1]"]
    j = round(j - 0.1,1)
    k = round(k - 0.1,1)

    while j > minFSC: 
 
        list_var_FSC_box.insert(0,np.var(er_FSC[np.where((FSC_test >= k) & (FSC_test < j))]))
        list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(k) +"\n"+ "{0:.1f}".format(j) +" [")
        j = round(j - 0.1,1)
        k = round(k - 0.1,1)
           


    # Plot figure with subplots 
    fig = plt.figure()
    gridspec.GridSpec(1,2)
        
        
    # boxplot avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
    plt.title('Variance of FSC (S2) - FSC ('+out_val+')',size = 14,y=1.08)
    plt.ylabel('Variance',size = 14)
    plt.xlabel('FSC ('+out_val+')',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.bar(list_var_labels_box,list_var_FSC_box)


    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
        
    fig.savefig(os.path.join(path_eval_dir,out_cal+"_EVAL_WITH_"+out_val+"_RESIDUAL_VAR_"+dates_test+'.png'))
    plt.close(fig)
    
    f.write("\n\n")

    f.write("\nEVALUATION WITH " + dates_test )
    f.write("\n  Number of 20x20m data points : " + str(len(NDSI_test)))
    f.write("\n  corr. coef. : " + str(corr_FSC))
    f.write("\n  std. err. : " + str(stde_FSC))
    f.write("\n  mean err. : " + str(m_er_FSC))
    f.write("\n  abs. mean err. : " + str(abs_m_er_FSC))
    f.write("\n  root mean square err. : " + str(rmse_FSC))
    return f     
    









def evaluation_average(f,a,b,list_NDSI_test,list_FSC_test,list_FSC_pred,list_dates_test,path_eval_dir,out_cal,out_val):
    

    list_NDSI_avg_test = []
    list_FSC_avg_test = []
    list_FSC_avg_pred = []
    print(len(list_NDSI_test))
    print(len(list_FSC_test))
    print(len(list_FSC_pred))
    print(len(list_dates_test))
    
    
    for i in np.arange(len(list_dates_test)):
        list_NDSI_avg_test.append(np.average(list_NDSI_test[i]))
        list_FSC_avg_test.append(np.average(list_FSC_test[i]))
        list_FSC_avg_pred.append(np.average(list_FSC_pred[i]))
    
    print(len(list_NDSI_avg_test))
    print(len(list_FSC_avg_test))
    print(len(list_FSC_avg_pred))
    print(len(list_dates_test))    


    NDSI_avg_test = np.hstack(list_NDSI_avg_test)
    FSC_avg_test = np.hstack(list_FSC_avg_test)
    FSC_avg_pred = np.hstack(list_FSC_avg_pred)
    dates_test = np.hstack(list_dates_test)
    dates_test = dates_test[0]+"_"+dates_test[-1]
    

    print(NDSI_avg_test.shape())
    print(FSC_avg_test.shape())
    print(FSC_avg_pred.shape())
    print(dates_test.shape())
    
    # VALIDATION


    # error
    er_FSC = FSC_avg_pred - FSC_avg_test

    # absolute error
    abs_er_FSC = abs(er_FSC)

    # mean error
    m_er_FSC = np.mean(er_FSC)

    # absolute mean error
    abs_m_er_FSC = np.mean(abs_er_FSC)

    #root mean square error
    rmse_FSC = sqrt(mean_squared_error(FSC_avg_pred,FSC_avg_test))

    #correlation
    corr_FSC = mstats.pearsonr(FSC_avg_pred,FSC_avg_test)[0]

    #standard deviation
    stde_FSC = np.std(er_FSC)
    
    
    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("AVERAGE S2 AND "+out_val+" WITH "+out_cal+" CALIBRATION")
    # set up subplot grid
    gridspec.GridSpec(1,2)
    # prepare for evaluation scatterplot
    ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
    
    #plt.ylabel('FSC',size = 14)
    plt.xlabel('Dates',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)

    plt.plot([], [], ' ', label='correlation : {:.2f}\nrmse : {:.2f}'.format(corr_FSC,rmse_FSC))
    plt.plot(dates_test,FSC_avg_pred,'-o', label='predicted FSC (S2)')
    plt.plot(dates_test,FSC_avg_test,'-o', label='FSC ({:s})'.format(out_val))
    plt.scatter(days,data_percent,color = 'red', label='percent of valid '+out_val+' pixels')
    
    plt.legend(fontsize=10,loc='upper left')


    f.write("\n")  
    f.write("\n  Number of dates : " + str(len(NDSI_avg_test)))
    f.write("\n  Total number of 20x20m pixels : " + str(nb_pixel_total))
    f.write("\n  Number of 20x20m pixels per date : " + str(nb_pixel_total/len(NDSI_avg_test)))
    f.write("\n  Covered surface per date (m2) : " + str(20*20*nb_pixel_total/len(NDSI_avg_test)))
    f.write("\n  corr. coef. : " + str(corr_FSC))
    f.write("\n  std. err. (MB): " + str(stde_FSC))
    f.write("\n  mean err. : " + str(m_er_FSC))
    f.write("\n  abs. mean err. : " + str(abs_m_er_FSC))
    f.write("\n  root mean square err. : " + str(rmse_FSC))
   


    
    # fit subplots & save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,"PLOT_AVERAGE_"+out_cal+"_EVAL_WITH_"+out_val+"_"+dates_test+".png"))
    plt.close(fig)

    


    return f
    
    
    
    
def evalFSCWithDP(out_cal,out_val,p):
    path_outputs = p["path_outputs"]

    calDataSetDir = os.path.join(path_outputs,out_cal)
    path_eval = os.path.join(calDataSetDir,"EVALUATION")
    title = "EVAL_" + out_cal + "_WITH_"+out_val
    
    DPPoints = os.path.join(path_outputs,out_val,"LIST.txt")


    path_eval_dir = os.path.join(path_eval,title)
    shutil.rmtree(path_eval_dir, ignore_errors=True)

    mkdir_p(path_eval_dir)



    path_params = os.path.join(calDataSetDir,"CALIBRATION","CALIBRATION_PARAMS.txt")
    a = 0
    b = 0
    with open(path_params, "r") as params :
        line = params.readline()
        line = params.readline()
        ab = line.split()
        a = float(ab[0])
        b = float(ab[1])








    f= open(os.path.join(path_eval_dir,title + ".txt"),"w")
    f.write("\nCalibration dataset :" + out_cal )
    f.write("\nModel : FSC = 0.5*tanh(a*NDSI+b) +  0.5 :")
    f.write("\n        a = " + str(a) + " b = " + str(b))
    f.write("\nEvaluation dataSets : \n" + out_val )





    dict_FSC = {}
    dict_products = {}



    print("####################################################")
    print("Recuperation of data points")
    #on recupere les donnees odk
    with open(DPPoints, "r") as datapoints :
        line = datapoints.readline()
        line = datapoints.readline()
        while line :
            point = line.split()
            date = point[0]
            latitude = point[1]
            longitude = point[2]
            fsc = float(point[3])
            L2A_product = point[6]
            tcd = float(point[8])


                
            if date not in dict_products.keys() :
                dict_products[date] = []

            dict_products[date].append([latitude,longitude,fsc,tcd,L2A_product])
                
            line = datapoints.readline()
                


   

    #on compare ODK et L2A
    list_NDSI = []
    list_FSC = []
    list_TCD = []
    for date in dict_products :
        for point in dict_products[date] :

            lat = point[0]
            lon = point[1]
            fsc = point[2]
            tcd = point[3]
            L2A_product = point[4]

            f_NDSI = glob.glob(os.path.join(L2A_product,'*NDSI*'))[0]

            

            
            try:
                NDSI = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_NDSI, lon, lat)).read())
            except ValueError:
                continue

            
            if np.isnan(NDSI) or np.isinf(NDSI) : continue
            
            list_NDSI.append(NDSI)
            list_FSC.append(fsc)
            list_TCD.append(tcd)

        

    #on affiche les lists
    print("####################################################")
    print("\nDATA POINTS:")

    for i in np.arange(len(list_NDSI)) :
        print("NDSI = ",list_NDSI[i],"FSC = ",list_FSC[i],"TCD = ",list_TCD[i])

    print("####################################################")
    print("Calculation of NDSI-FSC relation and model evaluation")

    #on calcul et affiche la relation FSC-NDSI et l evaluation des parametres a et b

    NDSI_test = np.asarray(list_NDSI)
    FSC_test = np.asarray(list_FSC)
    TCD = np.asarray(list_TCD)


    TOC_FSC_pred = 0.5*np.tanh(a*NDSI_test+b) +  0.5
    OG_FSC_pred = TOC_FSC_pred/((100.0 - TCD)/100.0)
    OG_FSC_pred[OG_FSC_pred > 1] = 1
    OG_FSC_pred[np.isinf(OG_FSC_pred)] = 1
    FSC_test_t = FSC_test[TCD > 0]
    OG_FSC_pred_t = OG_FSC_pred[TCD > 0]

    



    #TOC
    # error
    TOC_er_FSC = TOC_FSC_pred - FSC_test
    # absolute error
    TOC_abs_er_FSC = abs(TOC_er_FSC)
    # mean error
    TOC_m_er_FSC = np.mean(TOC_er_FSC)
    # absolute mean error
    TOC_abs_m_er_FSC = np.mean(TOC_abs_er_FSC)
    #root mean square error
    TOC_rmse_FSC = sqrt(mean_squared_error(TOC_FSC_pred,FSC_test))
    #correlation
    TOC_corr_FSC = mstats.pearsonr(TOC_FSC_pred,FSC_test)[0]
    #standard deviation
    TOC_stde_FSC = np.std(TOC_er_FSC)
 
    #OG
    # error
    OG_er_FSC = OG_FSC_pred - FSC_test
    # absolute error
    OG_abs_er_FSC = abs(OG_er_FSC)
    # mean error
    OG_m_er_FSC = np.mean(OG_er_FSC)
    # absolute mean error
    OG_abs_m_er_FSC = np.mean(OG_abs_er_FSC)
    #root mean square error
    OG_rmse_FSC = sqrt(mean_squared_error(OG_FSC_pred,FSC_test))
    #correlation
    OG_corr_FSC = mstats.pearsonr(OG_FSC_pred,FSC_test)[0]
    #standard deviation
    OG_stde_FSC = np.std(OG_er_FSC)



    #OG Tree Only
    # error
    OG_er_FSC_t = OG_FSC_pred_t - FSC_test_t
    # absolute error
    OG_abs_er_FSC_t = abs(OG_er_FSC_t)
    # mean error
    OG_m_er_FSC_t = np.mean(OG_er_FSC_t)
    # absolute mean error
    OG_abs_m_er_FSC_t = np.mean(OG_abs_er_FSC_t)
    #root mean square error
    OG_rmse_FSC_t = sqrt(mean_squared_error(OG_FSC_pred_t,FSC_test_t))
    #correlation
    OG_corr_FSC_t = mstats.pearsonr(OG_FSC_pred_t,FSC_test_t)[0]
    #standard deviation
    OG_stde_FSC_t = np.std(OG_er_FSC_t)


    


    minNDSI = min(NDSI_test)
    list_FSC_box = [FSC_test[np.where((NDSI_test >= 0.8) & (NDSI_test <= 1))]]
    list_labels_box = ["[ 0.8\n1 ]"]
    j = 0.8
    while minNDSI < j : 
        i = round(j - 0.2,1)
        list_FSC_box.insert(0,FSC_test[np.where((NDSI_test >= i) & (NDSI_test < j))])
        list_labels_box.insert(0,"[ "+ "{0:.1f}".format(i) +"\n"+ "{0:.1f}".format(j) +" [")
        j = round(j - 0.2,1)
        






    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("FSC / NDSI",size = 16)
    gridspec.GridSpec(1,3)

    # 2D histo pour FSC vs NDSI
    
    ax = plt.subplot2grid((1,3), (0,2))
    #plt.title('ODK FSC/NDSI',size = 14,y=1.08)
    plt.ylabel('Testing FSC',size = 14)
    plt.xlabel('Testing NDSI',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(NDSI_test,FSC_test,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    n = np.arange(min(NDSI_test),1.01,0.01)
    line = 0.5*np.tanh(a*n+b) +  0.5
    plt.plot(n, line, 'r')#, label='Predicted TOC FSC')
    #plt.legend(fontsize=6,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12) 
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio) 

    # boxplot avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((1,3), (0,0),rowspan=1, colspan=2)
    #plt.title('ODK FSC/NDSI',size = 14,y=1.08)
    plt.ylabel('0 <= FSC <= 1',size = 14)
    plt.xlabel('NDSI',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.boxplot(list_FSC_box,labels = list_labels_box)


    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,out_val+'_ANALYSIS.png'))
    plt.close(fig)






    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("EVALUATION",size = 16)
    gridspec.GridSpec(2,3)

    # 2D histo pour TOC evaluation
    ax = plt.subplot2grid((2,3), (0,2))

    plt.title('TOC FSC EVALUATION',size = 14,y=1.08)
    plt.ylabel('Predicted TOC FSC',size = 14)
    plt.xlabel('Testing FSC',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(FSC_test,TOC_FSC_pred,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    slope, intercept, r_value, p_value, std_err = mstats.linregress(FSC_test,TOC_FSC_pred) 
    n = np.array([min(FSC_test),1.0])
    line = slope * n + intercept
    plt.plot(n, line, 'b')#, label='y = {:.2f}x + {:.2f}\ncorr={:.2f} rmse={:.2f}'.format(slope,intercept,TOC_corr_FSC,TOC_rmse_FSC))
    plt.plot(n, n, 'g')#, label='y = 1.0x + 0.0')
    #plt.legend(fontsize=6,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12) 
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  


    # 1D histo de TOC residus
    ax = plt.subplot2grid((2,3), (0,0),rowspan=1, colspan=2)
    plt.title("TOC FSC RESIDUALS",size = 14,y=1.08)
    plt.ylabel('Percent of data points',size = 14)
    plt.xlabel('FSC pred - test',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    xticks = np.arange(-1.0, 1.1, 0.1)
    plt.xticks(xticks)
    plt.hist(TOC_er_FSC,bins=40,weights=np.ones(len(TOC_er_FSC)) / len(TOC_er_FSC))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.grid(True) 


    # 2D histo pour OG evaluation
    ax = plt.subplot2grid((2,3), (1,2))

    plt.title('OG FSC EVALUATION',size = 14,y=1.08)
    plt.ylabel('Predicted OG FSC',size = 14)
    plt.xlabel('Testing FSC',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(FSC_test,OG_FSC_pred,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    slope, intercept, r_value, p_value, std_err = mstats.linregress(FSC_test,OG_FSC_pred) 
    n = np.array([min(FSC_test),1.0])
    line = slope * n + intercept
    plt.plot(n, line, 'b')#, label='y = {:.2f}x + {:.2f}\ncorr={:.2f} rmse={:.2f}'.format(slope,intercept,OG_corr_FSC,OG_rmse_FSC))
    plt.plot(n, n, 'g')#, label='y = 1.0x + 0.0')
    #plt.legend(fontsize=6,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12) 
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  


    # 1D histo de OG residus
    ax = plt.subplot2grid((2,3), (1,0),rowspan=1, colspan=2)
    plt.title("OG FSC RESIDUALS",size = 14,y=1.08)
    plt.ylabel('Percent of data points',size = 14)
    plt.xlabel('FSC pred - test',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    xticks = np.arange(-1.0, 1.1, 0.1)
    plt.xticks(xticks)
    plt.hist(OG_er_FSC,bins=40,weights=np.ones(len(OG_er_FSC)) / len(OG_er_FSC))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.grid(True) 



    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,title + '.png'))
    plt.close(fig)


    

    minFSC = min(FSC_test)
    
    i = 1.0
    j = 1.1

    list_var_FSC_box = [np.var(TOC_er_FSC[np.where((FSC_test>= i) & (FSC_test <= j))])]
    
    list_var_labels_box = ["[1]"]
    j = round(j - 0.1,1)
    i = round(i - 0.1,1)
    
    

    while j > minFSC: 

    
        list_var_FSC_box.insert(0,np.var(TOC_er_FSC[np.where((FSC_test >= i) & (FSC_test < j))]))
        
        
        list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(i) +"\n"+ "{0:.1f}".format(j) +" [")
        j = round(j - 0.1,1)
        i = round(i - 0.1,1)
        

    

    # Plot figure with subplots 
    fig = plt.figure()
    
    gridspec.GridSpec(1,2)
    
    
    # boxplot avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
    #plt.title('ODK FSC/NDSI',size = 14,y=1.08)
    plt.ylabel('Variance',size = 14)
    plt.xlabel('FSC',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.bar(list_var_labels_box,list_var_FSC_box)


    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,out_val+'_RESIDUES_ANALYSIS.png'))
    plt.close(fig)
    
    
    

    f.write("\n")
    f.write("\nEVALUATION OF TOC FSC" )
    f.write("\n  Number of data points : " + str(len(FSC_test)))
    f.write("\n  corr. coef. : " + str(TOC_corr_FSC))
    f.write("\n  std. err. : " + str(TOC_stde_FSC))
    f.write("\n  mean err. : " + str(TOC_m_er_FSC))
    f.write("\n  abs. mean err. : " + str(TOC_abs_m_er_FSC))
    f.write("\n  root mean square err. : " + str(TOC_rmse_FSC))
    f.write("\n")
    f.write("\nEVALUATION OF OG FSC" )
    f.write("\n  Number of data points : " + str(len(FSC_test)))
    f.write("\n  corr. coef. : " + str(OG_corr_FSC))
    f.write("\n  std. err. : " + str(OG_stde_FSC))
    f.write("\n  mean err. : " + str(OG_m_er_FSC))
    f.write("\n  abs. mean err. : " + str(OG_abs_m_er_FSC))
    f.write("\n  root mean square err. : " + str(OG_rmse_FSC))
    f.write("\n")
    f.write("\nEVALUATION OF OG FSC with only pixels with TCD > 0" )
    f.write("\n  Number of data points : " + str(len(FSC_test_t)))
    f.write("\n  corr. coef. : " + str(OG_corr_FSC_t))
    f.write("\n  std. err. : " + str(OG_stde_FSC_t))
    f.write("\n  mean err. : " + str(OG_m_er_FSC_t))
    f.write("\n  abs. mean err. : " + str(OG_abs_m_er_FSC_t))
    f.write("\n  root mean square err. : " + str(OG_rmse_FSC_t))

    f.close()


    return True
    
    
    
    
    
    
    
    
    
  
def evalSCAWithDP(out_cal,out_val,p):
    min_snw_depth = 0
    path_outputs = p["path_outputs"]

    calDataSetDir = os.path.join(path_outputs,out_cal)
    path_eval = os.path.join(calDataSetDir,"EVALUATION")
    title = "EVAL_SCA_" + out_cal + "_WITH_"+out_val
    
    DPPoints = os.path.join(path_outputs,out_val,"LIST.txt")


    path_eval_dir = os.path.join(path_eval,title)
    shutil.rmtree(path_eval_dir, ignore_errors=True)

    mkdir_p(path_eval_dir)



    path_params = os.path.join(calDataSetDir,"CALIBRATION","CALIBRATION_PARAMS.txt")
    a = 0
    b = 0
    with open(path_params, "r") as params :
        line = params.readline()
        line = params.readline()
        ab = line.split()
        a = float(ab[0])
        b = float(ab[1])




    dict_SNW = {}
    dict_products = {}



    print("####################################################")
    print("Recuperation of data points")
    list_FSC_OG = []
    list_FSC_TOC = []
    list_NDSI = []
    list_SNW = []
    list_TCD = []
    list_lat = []
    list_lon = []
    list_date = []
    f= open(os.path.join(path_eval_dir,title + "_data.txt"),"w")
    f.write("date dateLIS tile lat lon sca fsctoc fscog ndsi tcd acc decal")
    with open(DPPoints, "r") as datapoints :
        line = datapoints.readline()
        line = datapoints.readline()
        while line :
            point = line.split()
            date = point[0]
            lat = point[1]
            lon = point[2]
            snw = str(point[3])
            acc = int(point[4])
            decal = int(point[5])
            L2A_product = point[6]
            tile = point[7]
            tcd = float(point[8])
            dateLIS = getDateFromStr(L2A_product).strftime("%Y-%m-%d")


            f_FSC_TOC = glob.glob(os.path.join(L2A_product,'*FSCTOC*'))[0]
            f_FSC_OG = glob.glob(os.path.join(L2A_product,'*FSCOG*'))[0]
            f_NDSI = glob.glob(os.path.join(L2A_product,'*NDSI*'))[0]
            
            try:
                NDSI = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_NDSI, lon, lat)).read())
                FSC_TOC = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_FSC_TOC, lon, lat)).read())
                FSC_OG = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_FSC_OG, lon, lat)).read())
            except ValueError:
                continue
            
            f.write("\n{} {} {} {} {} {} {} {} {} {} {}".format(date,dateLIS,tile,lat,lon,snw,FSC_TOC,FSC_OG,NDSI,tcd,acc,decal))
            if "TRACE" in snw :
            	snw = 0
            elif "PLUS" in snw :
                snw = 1
            elif float(snw.split("_")[0]) > min_snw_depth :
                snw = 1
            else : 
            	snw = 0
            print("{} {} {} {} {} {} {} {} {} {} {}".format(date,dateLIS,tile,lat,lon,snw,FSC_TOC,FSC_OG,NDSI,tcd,acc,decal))
            
            list_FSC_OG.append(FSC_OG)
            list_FSC_TOC.append(FSC_TOC)
            list_NDSI.append(NDSI)
            list_SNW.append(snw)
            list_TCD.append(tcd)
            list_lat.append(lat)
            list_lon.append(lon)
            list_date.append(date)


            line = datapoints.readline()
    f.close()
    #on affiche les lists
    print("####################################################")
    print("\nDATA POINTS:")
    if len(list_SNW) == 0: 
        print ("no points available!")
        return False
    for i in np.arange(len(list_SNW)) :
        print("NDSI = ",list_NDSI[i],"FSC TOC = ",list_FSC_TOC[i],"FSC OG = ",list_FSC_OG[i],"SNW = ",list_SNW[i],"TCD = ",list_TCD[i])

    #print("####################################################")
    #print("Prediction of FSC relation and model evaluation")

    #on calcul et affiche la relation FSC-NDSI et l evaluation des parametres a et b

    NDSI = np.asarray(list_NDSI)
    TCD = np.asarray(list_TCD)
    SNW = np.asarray(list_SNW)
    FSC_TOC = np.asarray(list_FSC_TOC)
    FSC_OG = np.asarray(list_FSC_OG)
    FSC_OG_bi = np.asarray(list_FSC_OG).astype(int)
    FSC_OG_bi[FSC_OG_bi > 0]  = 1
    
    

    list_NDSI_box = [NDSI[SNW == 0],NDSI[SNW == 1]]
    list_FSC_TOC_box = [FSC_TOC[SNW == 0],FSC_TOC[SNW == 1]]
    list_FSC_OG_box = [FSC_OG[SNW == 0],FSC_OG[SNW == 1]]
    list_labels_box = ["[ 0 ]","[ 1 ]"]
    
    cm = confusion_matrix(SNW,FSC_OG_bi,labels=[0,1])


    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("LIS EVALUATION WITH ("+out_val+')',size = 16)
    gridspec.GridSpec(2,2)

    # boxplot NDSI / SCA
    ax = plt.subplot2grid((2,2), (0,0),rowspan=1, colspan=1)
    plt.title('NDSI (LIS) / SCA ('+out_val+')',size = 14,y=1.08)
    plt.ylabel('NDSI',size = 14)
    plt.xlabel('SCA',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.boxplot(list_NDSI_box,labels = list_labels_box)

    # confusion matrix SCA (LIS)/ SCA (out_val)
    ax = plt.subplot2grid((2,2), (1,0),rowspan=1, colspan=1)
    plt.title('SCA (LIS)/SCA ('+out_val+')',size = 14,y=1.08)
    plt.ylabel('SCA ('+out_val+')',size = 14)
    plt.xlabel('SCA (LIS)',size = 14)
    plot_confusion_matrix(cm,[0,1],normalize=False)



    # boxplot FSC TOC/SCA
    ax = plt.subplot2grid((2,2), (0,1),rowspan=1, colspan=1)
    plt.grid()
    plt.title('FSC TOC (LIS) / SCA ('+out_val+')',size = 14,y=1.08)
    plt.ylabel('FSC',size = 14)
    plt.xlabel('SCA',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.boxplot(list_FSC_TOC_box,labels = list_labels_box)
    
    # boxplot FSC OG/SCA
    ax = plt.subplot2grid((2,2), (1,1),rowspan=1, colspan=1)
    plt.grid()
    plt.title('FSC OG (LIS) / SCA ('+out_val+')',size = 14,y=1.08)
    plt.ylabel('FSC',size = 14)
    plt.xlabel('SCA',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.boxplot(list_FSC_OG_box,labels = list_labels_box)

    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=8,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,"EVAL_SCA_" + out_cal + "_WITH_"+out_val+'.png'))
    plt.close(fig)



    #kappa
    kappa = cohen_kappa_score(SNW,FSC_OG_bi)
    # Accuracy
    accuracy = accuracy_score(SNW,FSC_OG_bi)
    # F1
    F1 = f1_score(SNW,FSC_OG_bi,average='weighted',labels = [0,1])
    #print('TOTO')
    F1_no_snow = f1_score(SNW,FSC_OG_bi,average='binary',pos_label=0)
    F1_snow = f1_score(SNW,FSC_OG_bi,average='binary',pos_label=1)


    f= open(os.path.join(path_eval_dir,title + ".txt"),"w")
    f.write("\nCalibration dataset :" + out_cal )
    f.write("\nModel : FSC = 0.5*tanh(a*NDSI+b) +  0.5 :")
    f.write("\n        a = " + str(a) + " b = " + str(b))
    f.write("\nEvaluation dataSets : \n" + out_val )

    f.write("\n")
    f.write("\nEVALUATION WITH BINARY FSC OG" )
    f.write("\n  confusion matrix : ")
    f.write("\n  {}\n".format(cm))
    f.write("\n  kappa : " + str(kappa))
    f.write("\n  accuracy : " + str(accuracy))
    f.write("\n  mean F1 weighted : " + str(F1))
    f.write("\n  F1 snow : " + str(F1_snow))
    f.write("\n  F1_no_snow: " + str(F1_no_snow))
    f.write("\n")


    f.close()

   
    

    


    return True
    
    
    
    
    
    
def timeLapseEval(p,out_cal,out_val):
    
    path_outputs = p["path_outputs"]
    calDataSetDir = os.path.join(path_outputs,out_cal)
    path_eval = os.path.join(calDataSetDir,"EVALUATION")
    
    

    path_params = os.path.join(calDataSetDir,"CALIBRATION","CALIBRATION_PARAMS.txt")
    a = 0
    b = 0
    with open(path_params, "r") as params :
        line = params.readline()
        line = params.readline()
        ab = line.split()
        a = float(ab[0])
        b = float(ab[1])


    path_eval_dir = os.path.join(path_eval,out_val)
    

    mkdir_p(path_eval_dir)

    f= open(os.path.join(path_eval_dir,"TIMELAPSE_"+out_cal+"_EVAL_WITH_"+out_val + ".txt"),"w")
    f.write("\nCalibration dataset :" + out_cal)
    f.write("\nModel : FSC = 0.5*tanh(a*NDSI+b) +  0.5 with :")
    f.write("\n        a = " + str(a) + " b = " + str(b))
    f.write("\nEvaluation dataset :" + out_val)
    

    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle("TIMELAPSE S2 AND "+out_val+" WITH "+out_cal+" CALIBRATION")
    # set up subplot grid
    gridspec.GridSpec(1,2)
    # prepare for evaluation scatterplot
    ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
    
    #plt.ylabel('FSC',size = 14)
    plt.xlabel('Dates',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    



    nb_pixel_total = 0

    evalDataSetDir = os.path.join(path_outputs,out_val)
    path_tifs = os.path.join(evalDataSetDir,"TIFS")

    NDSI_avg_test = []
    FSC_avg_test = []
    FSC_avg_pred = []

    data_percent = []
    days = []
    c = 0
    for d in sorted(os.listdir(path_tifs)):
        date = getDateFromStr(d)
        if date == '' : continue
        print(date)
        path_tifs_date = os.path.join(path_tifs,d)
            
            
        epsgs = {}
        for tif in os.listdir(path_tifs_date) :
            epsg = getEpsgFromStr(tif)
            if epsg == '': continue
            if epsg not in epsgs :
                epsgs[epsg] = []
            
        tiles = []
        for tif in os.listdir(path_tifs_date) :
            epsg = getEpsgFromStr(tif)
            tile = getTileFromStr(tif)
            if epsg == '' or tile == '': continue
            if tile not in epsgs[epsg]:
                epsgs[epsg].append(tile)      
                
        FSC_d = []
        NDSI_d = []
        SEB_d = []
        g_input_FSC = gdal.Open(os.path.join(path_tifs_date,"INPUT_FSC.tif"))
        g_input_FSC  = gdal.Warp('',g_input_FSC ,format= 'MEM',resampleAlg='average',xRes= 20,yRes= 20)
        INPUT_FSC_d = BandReadAsArray(g_input_FSC.GetRasterBand(1)).flatten()
        INPUT_FSC_d = INPUT_FSC_d[INPUT_FSC_d <= 1.0]
        for epsg in epsgs :
            for tile in epsgs[epsg]:
                g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                FSC_d.append(BandReadAsArray(g_FSC.GetRasterBand(1)).flatten())
                g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                NDSI_d.append(BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten())
                g_SEB = gdal.Open(os.path.join(path_tifs_date,"INPUT_SEB_" + tile + "_EPSG-" + epsg + ".tif"))
                SEB_d.append(BandReadAsArray(g_SEB.GetRasterBand(1)).flatten())
            
    
    
        NDSI_d = np.hstack(NDSI_d)
        FSC_d = np.hstack(FSC_d)  
        SEB_d = np.hstack(SEB_d) 
        d_FSC_size = len(INPUT_FSC_d)
        
        cond1 = np.where((FSC_d != 255) & (~np.isnan(FSC_d)) & (~np.isinf(FSC_d)))
        NDSI_d = NDSI_d[cond1]
        FSC_d = FSC_d[cond1]
        SEB_d = SEB_d[cond1]
        
        
        
        cond2 = np.where( (NDSI_d != 255) & (~np.isnan(NDSI_d)) & (~np.isinf(NDSI_d)))
        FSC_d = FSC_d[cond2]
        NDSI_d = NDSI_d[cond2]
        SEB_d = SEB_d[cond2]
        
        
        
        
        
        d_data_size = len(FSC_d)
        
        
        if len(FSC_d) == 0 : continue
        d_data_percent = float(d_data_size) / float(d_FSC_size)
        print(d_data_percent)
        nb_pixel_total = nb_pixel_total + len(FSC_d)
        FSC_avg_test.append(np.average(FSC_d))
        #NDSI_avg = np.average(NDSI_d)
        
        cond3 = np.where( (SEB_d == 0 ) )
        
        FSC_pred_d = 0.5*np.tanh(a*NDSI_d+b) +  0.5
        c1 = len(FSC_pred_d[cond3])
        c2 = len(FSC_pred_d)
        print(c1,c2)
        c = c + c1
        
        
        FSC_pred_d[cond3] = 0
        
        FSC_avg_pred.append(np.average(FSC_pred_d))
        


        #NDSI_test.append(NDSI_d)
        #FSC_avg_test.append(FSC_avg)
        
        data_percent.append(d_data_percent)
        days.append(date)
    print(c)
    #NDSI_test = np.hstack(NDSI_test)
    FSC_avg_test = np.hstack(FSC_avg_test)
    FSC_avg_pred = np.hstack(FSC_avg_pred)
    days = np.hstack(days)
           
    



    # VALIDATION


    # error
    er_FSC = FSC_avg_pred - FSC_avg_test

    # absolute error
    abs_er_FSC = abs(er_FSC)

    # mean error
    m_er_FSC = np.mean(er_FSC)

    # absolute mean error
    abs_m_er_FSC = np.mean(abs_er_FSC)

    #root mean square error
    rmse_FSC = sqrt(mean_squared_error(FSC_avg_pred,FSC_avg_test))

    #correlation
    corr_FSC = mstats.pearsonr(FSC_avg_pred,FSC_avg_test)[0]

    #standard deviation
    stde_FSC = np.std(er_FSC)

    plt.plot([], [], ' ', label='correlation : {:.2f}\nrmse : {:.2f}'.format(corr_FSC,rmse_FSC))
    plt.plot(days,FSC_avg_pred,'-o', label='predicted FSC (S2)')
    plt.plot(days,FSC_avg_test,'-o', label='FSC ({:s})'.format(out_val))
    plt.scatter(days,data_percent,color = 'red', label='percent of valid '+out_val+' pixels')
    
    plt.legend(fontsize=10,loc='upper left')




    f.write("\n")  
    f.write("\n  Number of dates : " + str(len(FSC_avg_test)))
    f.write("\n  Total number of 20x20m pixels : " + str(nb_pixel_total))
    f.write("\n  Number of 20x20m pixels per date : " + str(nb_pixel_total/len(FSC_avg_test)))
    f.write("\n  Covered surface per date (m2) : " + str(20*20*nb_pixel_total/len(FSC_avg_test)))
    f.write("\n  corr. coef. : " + str(corr_FSC))
    f.write("\n  std. err. (MB): " + str(stde_FSC))
    f.write("\n  mean err. : " + str(m_er_FSC))
    f.write("\n  abs. mean err. : " + str(abs_m_er_FSC))
    f.write("\n  root mean square err. : " + str(rmse_FSC))
   


    
    # fit subplots & save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,"PLOT_TIMELAPSE_"+out_cal+"_EVAL_WITH_"+out_val+".png"))
    plt.close(fig)

    #close txt file
    f.close()


    return True

