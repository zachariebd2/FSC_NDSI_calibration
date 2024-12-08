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
import sc_utils
from sklearn.metrics import confusion_matrix
import csv
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


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
    
    
    
    
    
    
    


def displaySCAWithDP(path_eval_dir,category_name,category_data,true_data,test_data,label_orientation = 0,graphs=["CM","COUNT","KAPPA","F1"],category_distribution = "discrete",xsteps = 10,xticks_type = "interval",ext = "",min_size = 0, xaxe_name = ""):
    if xaxe_name == "": xaxe_name = category_name
    list_kappa = []
    list_acc = []
    list_TP_p = []
    list_FN_p = []
    list_FP_p = []
    list_TN_p = []
    list_TP = []
    list_FN = []
    list_FP = []
    list_TN = []
    list_total = []
    list_snow_true = []
    list_no_snow_true = []
    list_snow_true_p = []
    list_no_snow_true_p = []
    list_snow_pred = []
    list_no_snow_pred = []

   

    v = category_data
    txt_v = category_name


############################################################################################################################

    if category_distribution == "discrete":
        list_labels_box = []
        for c in v:
            if c not in list_labels_box: 
                if len(true_data[np.where(v == c)]) > min_size: 
                    list_labels_box.append(c)
        list_labels_box = sorted(list_labels_box)
        for c in list_labels_box:
            SNW_bi_v = true_data[np.where(v == c)]
            FSC_OG_bi_v = test_data[np.where(v == c)]
            kappa,acc,TP, FP, TN, FN, total = sc_utils.getCM2Dmetrics(SNW_bi_v,FSC_OG_bi_v,1,0)
            list_kappa.append(kappa)
            list_acc.append(acc)
            list_TP_p.append(TP/total*100)
            list_FN_p.append(FN/total*100)
            list_FP_p.append(FP/total*100)
            list_TN_p.append(TN/total*100)
            list_TP.append(TP)
            list_FN.append(FN)
            list_FP.append(FP)
            list_TN.append(TN)
            list_snow_true.append(TP+FN)
            list_no_snow_true.append(TN+FP)
            list_snow_true_p.append((TP+FN)/total*100)
            list_no_snow_true_p.append((TN+FP)/total*100)
            list_snow_pred.append(TP+FP)
            list_no_snow_pred.append(TN+FN)
            list_total.append(total)
            print(c,TP+FN,total)



    elif category_distribution == "continue":


        
        max_v = int(ceil(max(v)))
        min_v = int(floor(min(v)))
        step_v = int(ceil(float(max_v - min_v)/xsteps))
        #print("step", step_v)
        step_order = math.floor(math.log(abs(step_v), 10))
        step_v = int(floor(step_v/(10**(step_order)))*(10**(step_order)))
        b = max_v
        a = max_v - step_v
        
        a_order = math.floor(math.log(abs(a), 10))
        if a_order >= step_order:
            a = int(round(a/(10**(step_order)))*(10**(step_order)))
        cond1 = np.where((v >= a ) & (v <= b))
        SNW_bi_v = true_data[cond1]
        if len(SNW_bi_v) > min_size:
            FSC_OG_bi_v = test_data[cond1]
            kappa, acc, TP, FP, TN, FN, total = sc_utils.getCM2Dmetrics(SNW_bi_v,FSC_OG_bi_v,1,0)
            list_kappa = [kappa]
            list_acc = [acc]
            list_TP_p = [TP/total*100]
            list_FN_p = [FN/total*100]
            list_FP_p = [FP/total*100]
            list_TN_p = [TN/total*100]
            list_TP = [TP]
            list_FN = [FN]
            list_FP = [FP]
            list_TN = [TN]
            list_snow_true = [TP+FN]
            list_no_snow_true = [TN+FP]
            list_snow_true_p = [(TP+FN)/total*100]
            list_no_snow_true_p = [(TN+FP)/total*100]
            list_snow_pred = [TP+FP]
            list_no_snow_pred = [TN+FN]
            list_total = [total]
            list_labels_box = ["[{} ; {}]".format(a,b)]
            print("[{} ; {}]".format(a,b),TP+FN,total)
            #print("[{} ; {}]".format(a,b))
        b = a
        a = b - step_v
        #print("a ",a)
        #a_order = math.floor(math.log(abs(a), 10))
        #print("a order ", a_order)
        #print("step order ", step_order)
        #if a_order >= step_order:
        #    a = int(round(a/(10**(step_order)))*(10**(step_order)))
        #print("a ",a)
        
        while min_v < b : 

            if a < min_v : a = min_v

            cond2 = []
            if xticks_type == "interval":
                cond2 = np.where((v >= a) & (v < b))
            elif xticks_type == "tomax":
                cond2 = np.where((v >= a))

            SNW_bi_v = true_data[cond2]
            #print("nb ",len(SNW_bi_v))
            if len(SNW_bi_v) > min_size:
                #print("step",b,len(SNW_bi_v))
                FSC_OG_bi_v = test_data[cond2]
                kappa,acc, TP, FP, TN, FN, total = sc_utils.getCM2Dmetrics(SNW_bi_v,FSC_OG_bi_v,1,0)
                list_kappa.insert(0,kappa)
                list_acc.insert(0,acc)
                list_TP_p.insert(0,TP/total*100)
                list_FN_p.insert(0,FN/total*100)
                list_FP_p.insert(0,FP/total*100)
                list_TN_p.insert(0,TN/total*100)
                list_TP.insert(0,TP)
                list_FN.insert(0,FN)
                list_FP.insert(0,FP)
                list_TN.insert(0,TN)
                list_snow_true.insert(0,TP+FN)
                list_no_snow_true.insert(0,TN+FP)
                list_snow_true_p.insert(0,(TP+FN)/total*100)
                list_no_snow_true_p.insert(0,(TN+FP)/total*100)
                list_snow_pred.insert(0,TP+FP)
                list_no_snow_pred.insert(0,TN+FN)
                list_total.insert(0,total)
                if xticks_type == "interval":
                    list_labels_box.insert(0,"["+ "{}".format(a) +" ; "+ "{}".format(b) +"[")
                    #print("["+ "{}".format(a) +" ; "+ "{}".format(b) +"[")
                    
                elif xticks_type == "tomax":
                    list_labels_box.insert(0,"["+ "{}".format(a) +" ; "+ "{}".format(max_v) +"]")
                    #print("["+ "{}".format(a) +" ; "+ "{}".format(max_v) +"]")
            b = a
            a = b - step_v
            #print("a ",a)
            #a_order = math.floor(math.log(abs(a), 10))
            #print("a_order ",a_order)
            #if a_order >= step_order:
            #    a = int(round(a/(10**(step_order)))*(10**(step_order)))
            #print("a ",a)
            

    nx = np.arange(len(list_labels_box))

    # if "KAPPA" in graphs:
        # fig = plt.figure()
        # #plt.suptitle('KAPPA SCORES / '+txt_v,size = 14)
        # plt.ylabel('Kappa',size = 14)
        # plt.xlabel(txt_v + " "+ ext,size = 14)
        # plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # plt.bar(nx,list_kappa,0.4, align='center')
        # plt.xticks(nx,list_labels_box,rotation = label_orientation)
        # fig.tight_layout()
        # #fig.set_size_inches(w=8,h=10)
        # fig.savefig(os.path.join(path_eval_dir,txt_v+'_KAPPA.png'))
        # plt.close(fig)
   

    # if "CM" in graphs:
        # fig = plt.figure()
        # #plt.suptitle('CONFUSION MATRIX OF '+txt_v,size = 14)
        # plt.ylabel('Points (%)',size = 14)
        # plt.xlabel(txt_v + " "+ ext,size = 14)
        # plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # width = 0.2
        # plt.bar(nx-0.2,list_TP_p, width,label = "TP", align='center')
        # plt.bar(nx,list_FN_p, width,label = "FN", align='center')
        # plt.bar(nx+0.2,list_FP_p, width,label = "FP", align='center')
        # plt.xticks(nx,list_labels_box,rotation = label_orientation)
        # plt.legend()
        # fig.tight_layout()
        # #fig.set_size_inches(w=8,h=10)
        # fig.savefig(os.path.join(path_eval_dir,txt_v+'_CM.png'))
        # plt.close(fig)


    # if "COUNT" in graphs:
        # fig = plt.figure()
        # #plt.suptitle(txt_v + " PIXEL COUNT ",size = 14)
        # ax = plt.subplot2grid((1,1), (0,0),rowspan=1, colspan=1)
        # plt.ylabel('Points',size = 14)
        # plt.xlabel(txt_v + " "+ ext,size = 14)
        # plt.tick_params(axis = 'both', which = 'major', labelsize = 12)

        # plt.bar(nx,list_no_snow_true, label = "No Snow (in situ)", align='center',color = 'g')
        # bars = plt.bar(nx,list_snow_true, label = "Snow (in situ)", align='center',bottom=list_no_snow_true,color ='b')
        # #plt.bar(nx-0.18,list_no_snow_pred, width, label = "No Snow (LIS)", align='center',color = 'g', hatch = '\\')
        # #bars = plt.bar(nx-0.18,list_snow_pred, width, label = "Snow (LIS)", align='center',bottom=list_no_snow_pred,color ='b', hatch = '\\')
        # plt.xticks(nx,list_labels_box,rotation = label_orientation)
        # plt.legend()
        # fig.tight_layout()
        # #fig.set_size_inches(w=8,h=10)
        # fig.savefig(os.path.join(path_eval_dir,txt_v+'_COUNT.png'))
        # plt.close(fig)
        
        
    
    
    
    fig, (ax0, ax1,ax2) = plt.subplots(nrows=3, sharex=True)
    
    label_size = 12
    width = 0.6
    ax0.bar(nx,list_kappa,width, align='center')
    ax0.set_ylabel('Kappa', size = label_size)
    ax0.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)) 
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax0.yaxis.grid()
    ax0.set_axisbelow(True)
    
    width = 0.15
    ax1.bar(nx-(1.5*width),list_TP, width,label = "TP", align='center')
    ax1.bar(nx-(0.5*width),list_FN, width,label = "FN", align='center')
    ax1.bar(nx+(0.5*width),list_FP, width,label = "FP", align='center')
    ax1.bar(nx+(1.5*width),list_TN, width,label = "TN", align='center')
    ax1.set_yscale('log',basey = 2)
    ax1.legend(ncol=4,framealpha = 0.2,prop={'size': 9})
    ax1.set_ylabel('Points (log2)', size = label_size)
    #order = int(math.ceil(math.log(101, 2)))
    #ax1.set_yticks(2 ** np.arange(order+1))
    #locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=order + 1)
    #ax1.yaxis.set_minor_locator(locmin)
    #ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.yaxis.grid()
    ax1.set_axisbelow(True)
    
    
    
    width = 0.3
    ax2.bar(nx-(0.5*width),list_snow_true,width, label = "Snow",color ='b')
    ax2.bar(nx+(0.5*width),list_no_snow_true,width, label = "No Snow",color = 'g')

    order = int(math.ceil(math.log(max(list_total), 2)))
    ax2.set_yscale('log',basey=2)
    
    #ax2.set_yticks(2 ** np.arange(order+1))
    
    ax2.legend(ncol=2,framealpha = 0.2,prop={'size': 9})
    ax2.set_ylabel('Points (log2)', size = label_size)
    ax2.set_xlabel(xaxe_name + " "+ ext, size = label_size)
    ax2.set_xticks(nx)
    ax2.set_xticklabels(list_labels_box,rotation = label_orientation,ha="right")
    ax2.yaxis.grid()
    ax2.set_axisbelow(True)
    #locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=order+1)
    #ax2.yaxis.set_minor_locator(locmin)
    #ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.align_ylabels()
    fig.tight_layout(h_pad=-0.1)
    fig.savefig(os.path.join(path_eval_dir,txt_v+'_COMBINE.png'))
    plt.close(fig)
    
    
    



    return True
    
  
def evalSCAWithDP(out_cal,out_val,p):
    

    path_outputs=p["path_outputs"]
    calDataSetDir = os.path.join(path_outputs,out_cal)
    path_eval = os.path.join(calDataSetDir,"EVALUATION")
    DPPoints = os.path.join(path_outputs,out_val,"datasets","list_points_datasets.csv")
    path_eval_dir = os.path.join(path_eval,out_val)
    shutil.rmtree(path_eval_dir, ignore_errors=True)
    mkdir_p(path_eval_dir)
        

    covertypes={
        '0':'nd',
        '111':'Closed forest',
        '112':'Closed forest',
        '113':'Closed forest',
        '114':'Closed forest',
        '115':'Closed forest',
        '116':'Closed forest',
        '121':'Open forest',
        '122':'Open forest',
        '123':'Open forest',
        '124':'Open forest',
        '125':'Open forest',
        '126':'Open forest',
        '20':'Shrubs',
        '30':'Herb. veg.',
        '90':'Herb. wetland',
        '100':'Moss/lichen',
        '60':'Bare/sparse',
        '40':'Cropland',
        '50':'Urban',
        '70':'Snow/Ice',
        '80':'Water',
        '200':'Open sea'
    }


#open the file a first time for threshold definition and QCFLAGS evaluation##########################################################
    print("PARSE DATA FOR QCFLAGS AND DEPTH THRESHOLD")
    list_QC_FLAGS = []
    list_FSC_OG = []
    list_SNW = []
    list_FSC_OG_noflags = []
    list_SNW_noflags = []
    nb_points_final = 0
    nb_fsc100 = 0
    nb_fscnd = 0
    nb_fsc0 = 0
    stations_f = open(os.path.join(path_eval_dir,"stations_used.csv"),'w')
    stations_w = csv.writer(stations_f)
    log_f = open(os.path.join(path_eval_dir,"log_datapoints0.txt"),'w')
    log_f.write("date tile lat lon B3 B4 B11 FSCOG")
    stations_w.writerow(["lat","lon"])
    list_stations = []
    list_countries = []
    with open(DPPoints, "r") as datapoints :
        reader = csv.DictReader(datapoints)
        for point in reader:
            
            snw = float(point["depth"])
            decal = point["decal"]
            QC_FLAG = format(int(point["QCFLAGS"]),'08b')
            OG = float(point["FSCOG"])
            date = point["date_input"]
            lat = point["lat"]
            lon = point["lon"]
            date_lis = point["date_lis"]
            tile = point["tile"]
            green = float(point["green"])
            red = float(point["red"])
            mir = float(point["mir"])
            country = point["country"]

            

            #filter
            cond = False
            
            if cond : continue
            
            if OG > 100:
                nb_fscnd = nb_fscnd + 1
            
            if green <= 0 and red <= 0 and mir <= 0:
                if OG > 0 and OG <= 100 : 
                    nb_fsc100 = nb_fsc100 + 1
                    #print("date lis:",date_lis)
                    #print("green B3",green,"red B4",red,"swir B11",mir)
                    #print("tile",tile,"lat",lat,"lon",lon)
                    #print("fscog",OG)
                if OG == 0:
                    nb_fsc0 = nb_fsc0 + 1
            
                log_f.write("\n{} {} {} {} {} {} {} {}".format(date_lis,tile,lat,lon,green,red,mir,OG))
            
            if [lat,lon] not in list_stations:
                list_stations.append([lat,lon])
                stations_w.writerow([lat,lon])
            if country not in list_countries and country != "NaN" and country != "":
                list_countries.append(country)
            list_SNW.append(snw)
            list_FSC_OG.append(OG)
            list_QC_FLAGS.append(QC_FLAG)

            if QC_FLAG == "00000000":
                list_SNW_noflags.append(snw)
                list_FSC_OG_noflags.append(OG)
    stations_f.close()
    print("fsc 0:",nb_fsc0,"fsc snow:",nb_fsc100,"fsc nd:",nb_fscnd)
    log_f.close()

    if len(list_SNW) == 0: 
        print ("no points available!")
        return 

    QC_FLAGS =np.asarray(list_QC_FLAGS)
    SNW = np.asarray(list_SNW)
    FSC_OG = np.asarray(list_FSC_OG)
    SNW_noflags = np.asarray(list_SNW_noflags)
    FSC_OG_noflags = np.asarray(list_FSC_OG_noflags)

  
##############################DEPTH AND FSC THRESHOLDS AT DECAL = 0 and QC FLAGS = 0######################################################
    print("CALCULATE QCFLAGS AND DEPTH THRESHOLDS")
    kappas = []
    accs = []
    metrics = []
    depth_thresholds = [] 
    fsc_thresholds = [] 
    SNW_bi = SNW.copy()
    FSC_OG_bi = FSC_OG.copy()

    kappas_noflags = []
    accs_noflags = []
    metrics_noflags = []
    depth_thresholds_noflags = [] 
    fsc_thresholds_noflags = [] 
    SNW_bi_noflags = SNW_noflags.copy()
    FSC_OG_bi_noflags = FSC_OG_noflags.copy()

    depth_ran = np.arange(0,11,1)
    size_depth = len(depth_ran)
    fsc_ran = np.arange(0,1,1)
    size_fsc = len(fsc_ran)

    for depth_threshold in depth_ran:
        for fsc_threshold in fsc_ran:
            cond1 = np.where(SNW > depth_threshold)
            cond2 = np.where(SNW <= depth_threshold)
            SNW_bi[cond2] = 0 
            SNW_bi[cond1] = 1 
            cond1 = np.where(FSC_OG > fsc_threshold)
            cond2 = np.where(FSC_OG <= fsc_threshold)
            FSC_OG_bi[cond2] = 0 
            FSC_OG_bi[cond1] = 1
            kappa,acc, TP, FP, TN, FN, total = sc_utils.getCM2Dmetrics(SNW_bi,FSC_OG_bi,1,0)
            accs.append(acc)
            kappas.append(kappa)
            metrics.append([TP,FP,TN,FN,total])

            cond1 = np.where(SNW_noflags > depth_threshold)
            cond2 = np.where(SNW_noflags <= depth_threshold)
            SNW_bi_noflags[cond2] = 0 
            SNW_bi_noflags[cond1] = 1 
            cond1 = np.where(FSC_OG_noflags > fsc_threshold)
            cond2 = np.where(FSC_OG_noflags <= fsc_threshold)
            FSC_OG_bi_noflags[cond2] = 0 
            FSC_OG_bi_noflags[cond1] = 1
            kappa,acc, TP, FP, TN, FN, total = sc_utils.getCM2Dmetrics(SNW_bi_noflags,FSC_OG_bi_noflags,1,0)
            accs_noflags.append(acc)
            kappas_noflags.append(kappa)
            metrics_noflags.append([TP,FP,TN,FN,total])

            depth_thresholds.append(depth_threshold)
            fsc_thresholds.append(fsc_threshold)

    
    best_kappa = max(kappas)
    best_kappa_index = kappas.index(best_kappa)
    best_metrics = metrics[best_kappa_index]
    best_TP = best_metrics[0]
    best_FP = best_metrics[1]
    best_TN = best_metrics[2]
    best_FN = best_metrics[3]
    best_precision = best_TP/(best_TP + best_FP)
    best_recall = best_TP/(best_TP + best_FN)
    best_depth_threshold = depth_thresholds[best_kappa_index]
    best_fsc_threshold = fsc_thresholds[best_kappa_index]
    best_acc = accs[best_kappa_index]

    best_kappa_noflags = max(kappas_noflags)
    best_kappa_index_noflags = kappas_noflags.index(best_kappa_noflags)
    best_metrics_noflags = metrics_noflags[best_kappa_index_noflags]
    best_TP_noflags = best_metrics_noflags[0]
    best_FP_noflags = best_metrics_noflags[1]
    best_TN_noflags = best_metrics_noflags[2]
    best_FN_noflags = best_metrics_noflags[3]
    best_precision_noflags = best_TP_noflags/(best_TP_noflags + best_FP_noflags)
    best_recall_noflags = best_TP_noflags/(best_TP_noflags + best_FN_noflags)
    best_depth_threshold_noflags = depth_thresholds[best_kappa_index_noflags]
    best_fsc_threshold_noflags = fsc_thresholds[best_kappa_index_noflags]
    best_acc_noflags = accs_noflags[best_kappa_index_noflags]
   
    # depth_thresholds2D = np.reshape(depth_thresholds,(size_depth,size_fsc))
    # fsc_thresholds2D = np.reshape(fsc_thresholds,(size_depth,size_fsc))
    # kappas2D = np.reshape(kappas,(size_depth,size_fsc))

    # # Plot figure with subplots 
    # fig = plt.figure()

    # plt.title('KAPPA SCORES',size = 14,y=1.08)
    # plt.ylabel('FSC THRESHOLD',size = 14)
    # plt.xlabel('DEPTH THRESHOLD',size = 14)
    # plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    # #plt.plot(list_threshold,list_F1_snow, label = "Snow F1")
    # plt.contourf(depth_thresholds2D,fsc_thresholds2D,kappas2D,cmap=plt.cm.get_cmap('plasma'))
    # plt.colorbar()
    
    # # fit subplots and save fig
    # fig.tight_layout()
    # fig.set_size_inches(w=8,h=10)
    # #st.set_y(0.95)
    # fig.subplots_adjust(top=0.85)
    # fig.savefig(os.path.join(path_eval_dir,'DEPTH_FSC_THRESHOLDS_EVALUATION.png'))
    # plt.close(fig)
    

    # Plot figure with subplots 
    print("PLOT QCFLAGS AND DEPTH TRESHOLD")
    fig = plt.figure()
    gridspec.GridSpec(2,2)

    plt.rcParams.update({'mathtext.default':  'regular' })
    ax = plt.subplot2grid((2,2), (0,0),rowspan=2, colspan=1)
    plt.ylabel('Kappa scores')
    plt.xlabel('$HS_{0}$ (cm)')
    
    plt.plot(depth_thresholds,kappas,label = "QC filter off")
    plt.plot(depth_thresholds,kappas_noflags,label = "QC filter on")
    plt.xticks(depth_ran)
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    ax.yaxis.grid()
    plt.legend()

    ax = plt.subplot2grid((2,2), (0,1))
    plt.title("QC filter off")
    plt.ylabel('FSC')
    plt.xlabel('In-Situ')
    cm = np.array([[best_TP,best_FP],[best_FN,best_TN]])
    names = ["snow","no snow"]
    plot_confusion_matrix(cm,names)

    ax = plt.subplot2grid((2,2), (1,1))
    plt.title("QC filter on")
    plt.ylabel('FSC')
    plt.xlabel('In-Situ')
    cm = np.array([[best_TP_noflags,best_FP_noflags],[best_FN_noflags,best_TN_noflags]])
    names = ["snow","no snow"]
    plot_confusion_matrix(cm,names)
    

    # fit subplots and save fig
    fig.tight_layout()
    #fig.set_size_inches(w=16,h=10)
    #st.set_y(0.95)
    #fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_eval_dir,'DEPTH_THRESHOLDS_QCFLAGS.png'))
    plt.close(fig)


    f= open(os.path.join(path_eval_dir,"dataset_summary.txt"),"w")
 
    f.write("\nFLAGS NON FILTERED")
    f.write("\nnb of stations : " + str(len(list_stations)))
    f.write("\nnb of countries : " + str(len(list_countries))+"\n")
    for country in list_countries:
        f.write(country+" ; ")
    f.write("\nnb of points : " + str(len(SNW)))
    f.write("\nsnow at depth : >"+str(best_depth_threshold))
    f.write("\nsnow at FSC : >"+str(best_fsc_threshold))
    f.write("\nnb of TP: " +str(best_TP))
    f.write("\nnb of TN: "+str(best_TN))
    f.write("\nnb of FP: "+str(best_FP))
    f.write("\nnb of FN: "+str(best_FN))
    f.write("\ntotal: "+str(best_metrics[4]))
    f.write("\nnb of LIS snow/no-snow: "+str(best_TP+best_FP)+"/"+str(best_TN+best_FN))
    f.write("\nnb of insitu snow/no-snow: "+str(best_TP+best_FN)+"/"+str(best_TN+best_FP))
    f.write("\nKappa: "+str(best_kappa))
    f.write("\nKappa at depth 0: "+str(kappas[0]))
    f.write("\nprecision: "+str(best_precision))
    f.write("\nrecall: "+str(best_recall))
    f.write("\naccuracy: "+str(best_acc))
    f.write("\naccuracy: at depth 0: "+str(accs[0]))
    f.write("\nFLAGS FILTERED")
    f.write("\nnb of points : " + str(len(SNW_noflags)))
    f.write("\nsnow at depth : >"+str(best_depth_threshold_noflags))
    f.write("\nsnow at FSC : >"+str(best_fsc_threshold_noflags))
    f.write("\nnb of TP: " +str(best_TP_noflags))
    f.write("\nnb of TN: "+str(best_TN_noflags))
    f.write("\nnb of FP: "+str(best_FP_noflags))
    f.write("\nnb of FN: "+str(best_FN_noflags))
    f.write("\ntotal: "+str(best_metrics_noflags[4]))
    f.write("\nnb of LIS snow/no-snow: "+str(best_TP_noflags+best_FP_noflags)+"/"+str(best_TN_noflags+best_FN_noflags))
    f.write("\nnb of insitu snow/no-snow: "+str(best_TP_noflags+best_FN_noflags)+"/"+str(best_TN_noflags+best_FP_noflags))
    f.write("\nKappa: "+str(best_kappa_noflags))
    f.write("\nKappa at depth 0: "+str(kappas_noflags[0]))
    f.write("\nprecision: "+str(best_precision_noflags))
    f.write("\nrecall: "+str(best_recall_noflags))
    f.write("\naccuracy: "+str(best_acc_noflags))
    f.write("\naccuracy: at depth 0: "+str(accs_noflags[0]))



    #stratifier qcflags
    print("BINARY FOR STRATIFICATION QCFLAGS")
    SNW_bi = SNW.copy()
    FSC_OG_bi = FSC_OG.copy()
    SNW_bi[SNW_bi <= best_depth_threshold] = 0
    SNW_bi[SNW_bi > best_depth_threshold] = 1
    FSC_OG_bi[FSC_OG_bi <= best_fsc_threshold] = 0
    FSC_OG_bi[FSC_OG_bi > best_fsc_threshold] = 1
    print("QCFLAGS STRATIFICATION")
    OK = displaySCAWithDP(path_eval_dir,"QCFLAGS",QC_FLAGS,SNW_bi,FSC_OG_bi,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="discrete",xsteps =10,xticks_type ="interval",ext = "",min_size =0)





    #open the file a second time for stratification without flags##########################################################
    print("SPARSE DATA FOR STRATIFICATION")
    
    list_FSC_OG_noflags = []
    list_SNW_noflags = []
    list_NDSI_noflags = []
    list_TCD_noflags = []
    list_lat_noflags = []
    list_decal_noflags = []
    list_lon_noflags = []
    list_date_noflags = []
    list_alt_noflags = []
    list_slp_noflags = []
    list_country_noflags = []
    list_cover_noflags = []

    with open(DPPoints, "r") as datapoints :
        reader = csv.DictReader(datapoints)
        for point in reader:
            
            date = point["date_input"]
            lat = point["lat"]
            lon = point["lon"]
            snw = point["depth"]
            date_lis = point["date_lis"]
            decal = point["decal"]
            acc = point["acc"]
            tile = point["tile"]
            tcd = float(point["TCD"])
            green = float(point["green"])
            red = float(point["red"])
            mir = float(point["mir"])
            QC_FLAG = format(int(point["QCFLAGS"]),'08b')
            OG = float(point["FSCOG"])
            alt = point["alt"]
            slp = point["slp"]
            country = point["country"]
            cover = covertypes.get(point["cover"])
            
            if country == "Bosnia and Herzegovina":
                country = "Bos. & Herz."
            elif country == "Czech Republic":
                country = "Czech Rep."
            elif country == "United Kingdom":
                country = "UK"
            

            #filter
            cond = False
            cond = cond or country == "NaN"
            cond = cond or tcd > 100
            cond = cond or alt == "NaN"
            cond = cond or slp == "-9999" or slp == "NaN"
            cond = cond or alt == "None" or alt == "-32767"
            #cond = cond or green + mir == 0 
            cond = cond or QC_FLAG != "00000000"
            cond = cond or decal != "0"
    
 

            
            if cond : continue
            
            
            if "TRACE" in snw :
                snw = "0"
            elif "_" in snw :
                snw = snw.split("_")[-1]
            
            if green + mir == 0 :
                NDSI = 0
            else:
                NDSI = ((green - mir)/(green + mir))*100
                

           
            
            list_SNW_noflags.append(float(snw))
            list_FSC_OG_noflags.append(OG)
            list_NDSI_noflags.append(NDSI)
            list_TCD_noflags.append(float(tcd))
            list_lat_noflags.append(float(lat))
            list_decal_noflags.append(int(decal))
            list_lon_noflags.append(float(lon))
            list_date_noflags.append(date)
            list_alt_noflags.append(float(alt))
            list_slp_noflags.append(float(slp))
            list_country_noflags.append(country)
            list_cover_noflags.append(cover)

 
    if len(list_SNW) == 0: 
        print ("no points available!")
        return 

    NDSI_noflags = np.asarray(list_NDSI_noflags).flatten()
    DATES_noflags = np.asarray(list_date_noflags).flatten()
    DECALS_noflags = np.asarray(list_decal_noflags).flatten()
    TCD_noflags = np.asarray(list_TCD_noflags).flatten()
    ALT_noflags = np.asarray(list_alt_noflags).flatten()
    SLP_noflags = np.asarray(list_slp_noflags).flatten()
    LAT_noflags  = np.asarray(list_lat_noflags).flatten()
    LON_noflags  = np.asarray(list_lon_noflags).flatten()
    COUNTRY_noflags = np.asarray(list_country_noflags).flatten()
    COVER_noflags= np.asarray(list_cover_noflags).flatten()
    SNW_noflags = np.asarray(list_SNW_noflags).flatten()
    FSC_OG_noflags = np.asarray(list_FSC_OG_noflags).flatten()
    
    MONTHS_noflags = []
    for m in range(len(DATES_noflags)):
        MONTHS_noflags.append(sc_utils.getDateFromStr(DATES_noflags[m]).strftime("%Y-%m"))
    MONTHS_noflags = np.asarray(MONTHS_noflags).flatten()




    ##############################STRATIFICATION AT BEST DEPTH THRESHOLDS AND DECAL = 0 AND FLAGS = 0######################################################
    print("BINARY FOR STRATIFICATION")
    SNW_bi_noflags = SNW_noflags.copy()
    FSC_OG_bi_noflags = FSC_OG_noflags.copy()
    SNW_bi_noflags[SNW_bi_noflags <= best_depth_threshold_noflags] = 0
    SNW_bi_noflags[SNW_bi_noflags > best_depth_threshold_noflags] = 1
    FSC_OG_bi_noflags[FSC_OG_bi_noflags <= best_fsc_threshold_noflags] = 0
    FSC_OG_bi_noflags[FSC_OG_bi_noflags > best_fsc_threshold_noflags] = 1
    
    print("STRATIFICATION")
    print("TCD")
    OK = displaySCAWithDP(path_eval_dir,"TCD",TCD_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="continue",xsteps =10,xticks_type ="interval",ext ="(%)",min_size =0,xaxe_name = "Tree Cover Density")
    print("MONTHS")
    OK = displaySCAWithDP(path_eval_dir,"Months",MONTHS_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="discrete",xsteps =10,xticks_type ="interval",ext = "",min_size =0)
    print("NDSI")
    OK = displaySCAWithDP(path_eval_dir,"NDSI",NDSI_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="continue",xsteps =10,xticks_type ="interval",ext = "",min_size =0)
    print("COUNTRIES")
    OK = displaySCAWithDP(path_eval_dir,"Countries",COUNTRY_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="discrete",xsteps =10,xticks_type ="interval",ext = "",min_size =100)
    print("SLOPES")
    OK = displaySCAWithDP(path_eval_dir,"Slope",SLP_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="continue",xsteps =10,xticks_type ="interval",ext ="(°)",min_size =0)
    print("ELEVATIONS")
    OK = displaySCAWithDP(path_eval_dir,"Elevation",ALT_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="continue",xsteps =10,xticks_type ="interval",ext ="(m a.s.l)",min_size =0)
    print("LATITUDES")
    OK = displaySCAWithDP(path_eval_dir,"Latitude",LAT_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="continue",xsteps =10,xticks_type ="interval",ext ="(°)",min_size =0)
    print("LONGITUDES")
    OK = displaySCAWithDP(path_eval_dir,"Longitude",LON_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="continue",xsteps =10,xticks_type ="interval",ext ="(°)",min_size =0)
    print("COVERS")
    OK = displaySCAWithDP(path_eval_dir,"Covers",COVER_noflags,SNW_bi_noflags,FSC_OG_bi_noflags,label_orientation =45,graphs=["CM","COUNT","KAPPA"],category_distribution ="discrete",xsteps =10,xticks_type ="interval",ext = "",min_size =0)
    
    f.write("\nFLAGS FILTERED AND STRATIFICATION")
    f.write("\nnb of points : " + str(len(SNW_noflags)))
    f.write("\nsnow at depth : >"+str(best_depth_threshold_noflags))
    f.write("\nsnow at FSC : >"+str(best_fsc_threshold_noflags))
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

