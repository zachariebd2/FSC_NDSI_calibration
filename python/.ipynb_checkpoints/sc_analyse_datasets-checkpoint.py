#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:21:17 2020

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
from math import sqrt
from matplotlib.ticker import PercentFormatter
from sc_utils import *



def analyse_period(p,out):
    path_outputs = p["path_outputs"]
    dataSetDir = os.path.join(path_outputs,out)
    path_tifs = os.path.join(dataSetDir,"TIFS")
    path_plots = os.path.join(dataSetDir,"PLOTS")
    
    
    print("Start plotting the periode")
    NDSIALL = []
    FSCALL = []
    NDSIALL2 = []
    FSCALL2 = []

    sorted_dates = sorted(os.listdir(path_tifs))
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]
    
    path_plots_date = os.path.join(path_plots,start_date + "_" + end_date)
    mkdir_p(path_plots_date)
    
    f= open(os.path.join(path_plots_date,"INFO.txt"),"w")
    f.write("\nDates :")
    nb_dates = 0
    
   
        
    for d in sorted_dates:
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
    
    cond3 = np.where((FSCALL != 0) & (FSCALL != 1))
    NDSIALL2 = NDSIALL[cond3]
    FSCALL2 = FSCALL[cond3]
        
    if len(FSCALL2) < 2 : 
        f.close()
        shutil.rmtree(path_plots_date, ignore_errors=True)
        return False
    f.write("\nNumber of dates : " + str(nb_dates))
        

    print("Create plots")
    minNDSI = min(NDSIALL)
    print("minNDSI : " +str(minNDSI))

    list_FSC_box = [FSCALL[np.where((NDSIALL >= 0.8) & (NDSIALL <= 1))]]
    list_labels_box = ["[ 0.8\n1 ]"]
    b = 0.8
    while minNDSI < b : 
        if b > -1:
            a = round(b - 0.2,1)
        else:
            a = minNDSI
        list_FSC_box.insert(0,FSCALL[np.where((NDSIALL >= a) & (NDSIALL < b))])
        list_labels_box.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
        b = a
        

    minNDSI2 = min(NDSIALL2)
    print("minNDSI2 : " +str(minNDSI2))

    list_FSC_box2 = [FSCALL2[np.where((NDSIALL2 >= 0.8) & (NDSIALL2 <= 1))]]
    list_labels_box2 = ["[ 0.8\n1 ]"]
    b = 0.8
    while minNDSI2 < b : 
        if b > -1:
            a = round(b - 0.2,1)
        else:
            a = minNDSI2
        list_FSC_box2.insert(0,FSCALL2[np.where((NDSIALL2 >= a) & (NDSIALL2 < b))])
        list_labels_box2.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
        b = a  

    
    
    # Plot figure with subplots 
    fig = plt.figure()
    st = fig.suptitle(out + " : FSC / NDSI FOR THE PERIOD " + start_date + " - " + end_date,size = 14)
    # set up subplot grid
    gridspec.GridSpec(2,3)
    
    # 2D histo avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((2,3), (0,2))
    
    
    plt.ylabel('0 <= FSC <= 1',size = 14)
    plt.xlabel('NDSI',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(NDSIALL,FSCALL,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'), norm=LogNorm())
    
    if NDSIALL != [] and FSCALL != [] :
        slopeA, interceptA, r_valueA, p_valueA, std_errA = mstats.linregress(NDSIALL,FSCALL) 
        slopeB, interceptB, r_valueB, p_valueB, std_errB = mstats.linregress(FSCALL,NDSIALL) 
        n = np.array([minNDSI,1.0])
        lineA = slopeA*n+interceptA
        lineB = (n-interceptB)/slopeB
        plt.plot(n, lineA, 'g')#, label='MA: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(slopeA,interceptA,r_valueA,std_errA))
        plt.plot(n, lineB, 'r')#, label='MB: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(1/slopeB,-interceptB/slopeB,r_valueB,std_errB))
        #plt.legend(fontsize=6,loc='upper left')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  
    
    # 2D histo sans FSC = 0 et FSC = 1
    ax = plt.subplot2grid((2,3), (1,2))
    
    
    plt.ylabel('0 < FSC < 1',size = 14)
    plt.xlabel('NDSI',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.hist2d(NDSIALL2,FSCALL2,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
    if NDSIALL2 != []  and FSCALL2 != [] : 
        slopeA2, interceptA2, r_valueA2, p_valueA2, std_errA2 = mstats.linregress(NDSIALL2,FSCALL2) 
        slopeB2, interceptB2, r_valueB2, p_valueB2, std_errB2 = mstats.linregress(FSCALL2,NDSIALL2) 
        n = np.array([minNDSI2,1.0])
        lineA = slopeA2*n+interceptA2
        lineB = (n-interceptB2)/slopeB2
        plt.plot(n, lineA, 'g')#, label='MA: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(slopeA2,interceptA2,r_valueA2,std_errA2))
        plt.plot(n, lineB, 'r')#, label='MB: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(1/slopeB2,-interceptB2/slopeB2,r_valueB2,std_errB2))
        #plt.legend(fontsize=6,loc='upper left')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  



    # boxplot avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((2,3), (0,0),rowspan=1, colspan=2)
    plt.title('ANALYSIS WITH 0 <= FSC <= 1',size = 14,y=1.08)
    plt.ylabel('0 <= FSC <= 1',size = 14)
    plt.xlabel('NDSI',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.boxplot(list_FSC_box,labels = list_labels_box)
    

    
    # boxplot sans FSC = 0 et FSC = 1
    ax = plt.subplot2grid((2,3), (1,0),rowspan=1, colspan=2)
    plt.title('ANALYSIS WITH 0 < FSC < 1',size = 14,y=1.08)
    plt.ylabel('0 < FSC < 1',size = 14)
    plt.xlabel('NDSI',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.boxplot(list_FSC_box2,labels = list_labels_box2)

    # fit subplots & save fig
    fig.tight_layout()
    fig.set_size_inches(w=16,h=10)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_plots_date,'PLOT_FSC_NDSI_' + out + '_' + start_date + "_" + end_date + '.png'))
    plt.close(fig)
    
    
    f.write("\nFor  0 <= FSC <= 1 : " )
    f.write("\n  Number of data points : " + str(len(NDSIALL)))
    if NDSIALL != [] and FSCALL != [] :
        f.write("\n lin. reg. FSC on NDSI (MA): FSC = aNDSI + b : a = " + str(slopeA) + " ; b = " + str(interceptA))
        f.write("\n  std. err. (MA): " + str(std_errA))
        f.write("\n lin. reg. NDSI on FSC (MB): FSC = aNDSI + b : a = " + str(1/slopeB) + " ; b = " + str(-interceptB/slopeB))
        f.write("\n  std. err. (MB): " + str(std_errB))
        f.write("\n  corr. coef. : " + str(r_valueA))
        
    
    f.write("\nFor  0 < FSC < 1 : " )
    f.write("\n  Number of data points : " + str(len(NDSIALL2)))
    if NDSIALL2 != [] and FSCALL2 != [] :
        f.write("\n lin. reg. FSC on NDSI (MA): FSC = aNDSI + b : a = " + str(slopeA2) + " ; b = " + str(interceptA2))
        f.write("\n  std. err. (MA): " + str(std_errA2))
        f.write("\n lin. reg. NDSI on FSC (MB): FSC = aNDSI + b : a = " + str(1/slopeB2) + " ; b = " + str(-interceptB2/slopeB2))
        f.write("\n  std. err. (MB): " + str(std_errB2))
        f.write("\n  corr. coef. : " + str(r_valueA2))
    f.close()

    print ("\n plotting finished")
    NDSI = None
    FSC = None
    NDSIALL = None
    FSCALL = None
    NDSIALL2 = None
    FSCALL2 = None
    
    return True
    


    #plt.title("VALIDATION WITH THE TESTING SET",size = 14,y=1.08)
    #plt.ylabel('Predicted FSC',size = 14)
    #plt.xlabel('Testing FSC',size = 14)
    #plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=12)       


def analyse_dates(p,out):
    
    path_outputs = p["path_outputs"]
    print("Start plotting each date")
    dataSetDir = os.path.join(path_outputs,out)
    path_tifs = os.path.join(dataSetDir,"TIFS")
    path_plots = os.path.join(dataSetDir,"PLOTS")
        
        
    for d in sorted(os.listdir(path_tifs)):
        date = getDateFromStr(d)
        if date == '' : continue
        print(date)
        path_tifs_date = os.path.join(path_tifs,d)
        path_plots_date = os.path.join(path_plots,d)
        mkdir_p(path_plots_date)
        FSC = []
        NDSI = []  
        
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
            print("tile =",tile)
            if epsg == '' or tile == '': continue
            if tile not in epsgs[epsg]:
                epsgs[epsg].append(tile)  
                
        print("epsgs",epsgs)
        
        for epsg in epsgs :
            print(epsg)
            for tile in epsgs[epsg]:
                print(tile,epsg)
                g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                print(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                FSC.append(BandReadAsArray(g_FSC.GetRasterBand(1)).flatten())
                g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                print(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                NDSI.append(BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten())
                
                


    
        print("Eliminate Nodata pixels")
        NDSI = np.hstack(NDSI)
        FSC = np.hstack(FSC)  
        cond1 = np.where((FSC != 255) & (~np.isnan(FSC)) & (~np.isinf(FSC)))
        NDSI = NDSI[cond1]
        FSC = FSC[cond1]
        
        cond2 = np.where( (NDSI != 255) & (~np.isnan(NDSI)) & (~np.isinf(NDSI)))
        FSC = FSC[cond2]
        NDSI = NDSI[cond2]
        
        cond3 = np.where((FSC != 0) & (FSC != 1))
        NDSI2 = NDSI[cond3]
        FSC2 = FSC[cond3]            
        if len(FSC2) < 2 : 
            print("Not enough available pixels")
            continue
        

        f = open(os.path.join(path_plots_date,"INFO.txt"),"w")
        f.write("\nDate : " + d)
        f.write("\nProjections of FSC inputs : ")
        for epsg in epsgs :  f.write("\n                   " + epsg)
        
        
        
        
        minNDSI = min(NDSI)
        print("minNDSI : " +str(minNDSI))

        list_FSC_box = [FSC[np.where((NDSI >= 0.8) & (NDSI <= 1))]]
        list_labels_box = ["[ 0.8\n1 ]"]
        b = 0.8
        while minNDSI < b : 
            if b > -1:
                a = round(b - 0.2,1)
            else:
                a = minNDSI
            list_FSC_box.insert(0,FSC[np.where((NDSI >= a) & (NDSI < b))])
            list_labels_box.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
            b = a
            

        minNDSI2 = min(NDSI2)
        print("minNDSI2 : " +str(minNDSI2))

        list_FSC_box2 = [FSC2[np.where((NDSI2 >= 0.8) & (NDSI2 <= 1))]]
        list_labels_box2 = ["[ 0.8\n1 ]"]
        b = 0.8
        while minNDSI2 < b : 
            if b > -1:
                a = round(b - 0.2,1)
            else:
                a = minNDSI2
            list_FSC_box2.insert(0,FSC2[np.where((NDSI2 >= a) & (NDSI2 < b))])
            list_labels_box2.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
            b = a         
        
        

        
        # Plot figure with subplots 
        fig = plt.figure()
        st = fig.suptitle(out + " : FSC / NDSI FOR " + date.strftime("%Y/%m/%d"))
        gridspec.GridSpec(2,3)
    
        # 2D histo avec FSC = 0 et FSC = 1
        ax = plt.subplot2grid((2,3), (0,2))
        slopeA, interceptA, r_valueA, p_valueA, std_errA = mstats.linregress(NDSI,FSC) 
        slopeB, interceptB, r_valueB, p_valueB, std_errB = mstats.linregress(FSC,NDSI)
        
        plt.ylabel('0 <= FSC <= 1')
        plt.xlabel('NDSI')
        plt.hist2d(NDSI,FSC,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'), norm=LogNorm())
        n = np.array([minNDSI,1.0])
        lineA = slopeA*n+interceptA
        lineB = (n-interceptB)/slopeB
        plt.plot(n, lineA, 'g', label='MA: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(slopeA,interceptA,r_valueA,std_errA))
        plt.plot(n, lineB, 'r', label='MB: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(1/slopeB,-interceptB/slopeB,r_valueB,std_errB))
        plt.legend(fontsize=6,loc='upper left')
        plt.colorbar()
        ratio = 1
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  
        
        # 2D histo sans FSC = 0 et FSC = 1
        ax = plt.subplot2grid((2,3), (1,2))
        slopeA2, interceptA2, r_valueA2, p_valueA2, std_errA2 = mstats.linregress(NDSI2,FSC2) 
        slopeB2, interceptB2, r_valueB2, p_valueB2, std_errB2 = mstats.linregress(FSC2,NDSI2) 
        
        plt.ylabel('0 < FSC < 1')
        plt.xlabel('NDSI')
        plt.hist2d(NDSI2,FSC2,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
        n = np.array([minNDSI2,1.0])
        lineA = slopeA2*n+interceptA2
        lineB = (n-interceptB2)/slopeB2
        plt.plot(n, lineA, 'g', label='MA: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(slopeA2,interceptA2,r_valueA2,std_errA2))
        plt.plot(n, lineB, 'r', label='MB: a={:.2f} b={:.2f}\ncorr={:.2f} std_err={:.3f}'.format(1/slopeB2,-interceptB2/slopeB2,r_valueB2,std_errB2))
        
        
        plt.legend(fontsize=6,loc='upper left')
        plt.colorbar()
        ratio = 1
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  
    
    
    
        # boxplot avec FSC = 0 et FSC = 1
        ax = plt.subplot2grid((2,3), (0,0),rowspan=1, colspan=2)
        plt.title('ANALYSIS WITH 0 <= FSC <= 1')
        plt.ylabel('0 <= FSC <= 1')
        plt.xlabel('NDSI')
        plt.boxplot(list_FSC_box,labels = list_labels_box)
        
    
        
        # boxplot sans FSC = 0 et FSC = 1
        ax = plt.subplot2grid((2,3), (1,0),rowspan=1, colspan=2)
        plt.title('ANALYSIS WITH 0 < FSC < 1')
        plt.ylabel('0 < FSC < 1')
        plt.xlabel('NDSI')
        plt.boxplot(list_FSC_box2,labels = list_labels_box2)
    
        # fit subplots & save fig
        fig.tight_layout()
        
        fig.set_size_inches(w=16,h=10)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(path_plots_date,'PLOT_FSC_NDSI_'  + out + '_' + date.strftime("%Y-%m-%d") + '.png'))
        plt.close(fig)
        
        

        f.write("\nFor  0 <= FSC <= 1 : " )
        f.write("\n  Number of data points : " + str(len(NDSI)))
        if NDSI != [] and FSC != [] :
            f.write("\n lin. reg. FSC on NDSI (MA): FSC = aNDSI + b : a = " + str(slopeA) + " ; b = " + str(interceptA))
            f.write("\n  std. err. (MA): " + str(std_errA))
            f.write("\n lin. reg. NDSI on FSC (MB): FSC = aNDSI + b : a = " + str(1/slopeB) + " ; b = " + str(-interceptB/slopeB))
            f.write("\n  std. err. (MB): " + str(std_errB))
            f.write("\n  corr. coef. : " + str(r_valueA))
            
        
        f.write("\nFor  0 < FSC < 1 : " )
        f.write("\n  Number of data points : " + str(len(NDSI2)))
        if NDSI2 != [] and FSC2 != [] :
            f.write("\n lin. reg. FSC on NDSI (MA): FSC = aNDSI + b : a = " + str(slopeA2) + " ; b = " + str(interceptA2))
            f.write("\n  std. err. (MA): " + str(std_errA2))
            f.write("\n lin. reg. NDSI on FSC (MB): FSC = aNDSI + b : a = " + str(1/slopeB2) + " ; b = " + str(-interceptB2/slopeB2))
            f.write("\n  std. err. (MB): " + str(std_errB2))
            f.write("\n  corr. coef. : " + str(r_valueA2))
        f.close()
        
        
    print ("\n plotting finished")
    NDSI = None
    FSC = None
    NDSI2 = None
    FSC2 = None
    
    return True





def make_quicklooks(p,out):
    
    path_outputs = p["path_outputs"]
    p_cmp = os.path.join("palettes/palette_cmp.txt")
    p_fsc = os.path.join("palettes/palette_FSC.txt")
    
    dataSetDir = os.path.join(path_outputs,out)
    path_tifs = os.path.join(dataSetDir,"TIFS")
    path_qckls = os.path.join(dataSetDir,"QUICKLOOKS")

    nb_dates = 0
    for date in sorted(os.listdir(path_tifs)):
        print(date)
        path_tifs_date = os.path.join(path_tifs,date)
        path_qckls_date = os.path.join(path_qckls,date)
        mkdir_p(path_qckls_date)
        
        #we get a list of tiles for each epsg
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
                
       
        #create input FSC quicklook
        f_FSC_i = os.path.join(path_tifs_date,"INPUT_FSC.tif")
        os.system("gdaldem color-relief " + f_FSC_i + " " + p_fsc + " " + os.path.join(path_qckls_date,"INPUT_FSC.tif"))
        gdal.Translate(os.path.join(path_qckls_date,"INPUT_FSC.png"),os.path.join(path_qckls_date,"INPUT_FSC.tif"),format= 'PNG', width=800,outputType = gdal.GDT_Byte)
        os.remove(os.path.join(path_qckls_date,"INPUT_FSC.tif"))
        #for each epsg 
        for epsg in epsgs:
            
            #create resampled FSC quicklook for each projection
            f_FSC_r = os.path.join(path_tifs_date,"RESAMPLED_FSC_EPSG-" + epsg + ".tif")
            os.system("gdaldem color-relief " + f_FSC_r + " " + p_fsc + " " + os.path.join(path_qckls_date,"RESAMPLED_FSC_EPSG-" + epsg + ".tif"))
            gdal.Translate(os.path.join(path_qckls_date,"RESAMPLED_FSC_EPSG-" + epsg + ".png"),os.path.join(path_qckls_date,"RESAMPLED_FSC_EPSG-" + epsg + ".tif"),format= 'PNG', width=800,outputType = gdal.GDT_Byte)
            os.remove(os.path.join(path_qckls_date,"RESAMPLED_FSC_EPSG-" + epsg + ".tif"))

            
            for tile in epsgs[epsg]:
                
                ##create compo quicklook
                #f_COMPO = os.path.join(path_tifs_date,"INPUT_COMPO_"+tile+"_EPSG-"+epsg+".tif")
                #gdal.Translate(os.path.join(path_qckls_date,"INPUT_COMPO_"+tile+"_EPSG-"+epsg+".png"),f_COMPO,format= 'PNG', width=800,outputType = gdal.GDT_Byte)
                

                
                #create FSC output quiclook
                f_FSC = os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif")
                os.system("gdaldem color-relief " + f_FSC + " " + p_fsc + " " + os.path.join(path_qckls_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                gdal.Translate(os.path.join(path_qckls_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".png"),os.path.join(path_qckls_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"),format= 'PNG', width=800,outputType = gdal.GDT_Byte)
                os.remove(os.path.join(path_qckls_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                
                #create NDSI output quiclook
                f_NDSI = os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif")
                os.system("gdaldem color-relief " + f_NDSI + " " + p_fsc + " " + os.path.join(path_qckls_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                gdal.Translate(os.path.join(path_qckls_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".png"),os.path.join(path_qckls_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"),format= 'PNG', width=800,outputType = gdal.GDT_Byte)
                os.remove(os.path.join(path_qckls_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                
                
                
                # #create snow difference quicklook
                
                
                # g_FSC = gdal.Open(f_FSC_r)
                # g_SEB = gdal.Open(os.path.join(path_tifs_date,"INPUT_SEB_"+tile+"_EPSG-"+epsg+".tif"))
                
                # minx, maxy, maxx, miny = getOverlapCoords(g_FSC,g_SEB)
                # g_FSC = gdal.Translate('',g_FSC,format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
                # g_SEB = gdal.Translate('',g_SEB,format= 'MEM',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32) 
                # g_CMP = g_FSC
                # #valeurs dans FSC : [0-1] pour la neige (et non-neige) , 9999 pour noData
                # #valeurs dans SEB : 100 pour la neige, 0 pour non-neige, 205 pour nuage, 254 pour nodata         

                # SEB = BandReadAsArray(g_SEB.GetRasterBand(1))  
                # cond = np.where((SEB != 100) & (SEB != 0))
                # SEB[cond] = np.nan
                # cond = np.where(SEB == 100)
                # SEB[cond] = 1


                
                # #valeurs dans FSC : [0-1] pour la neige (et non-neige) , 9999 pour noData
                # #valeurs dans SEB : 1 pour la neige, 0 pour non neige, nan pour noData   
                

                # FSC = BandReadAsArray(g_CMP.GetRasterBand(1))

                # cond = np.where((FSC > 0) & (FSC <= 1))
                # FSC[cond] = 2
                # FSC[FSC == 9999] = np.nan

                # #valeurs dans FSC : 2 pour la neige, 0 pour non neige, nan pour nodata
                # #valeurs dans SEB : 1 pour la neige, 0 pour non neige, nan pour noData
                
                # CMP = (SEB + FSC)
                
                # #cond = np.where((CMP != 1) & (CMP != 2))

                # #CMP[cond] = np.nan
                # g_CMP.GetRasterBand(1).WriteArray(CMP)
                # gdal.Translate(os.path.join(path_tifs_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"),g_CMP,format= 'GTiff',noData = 9999)
                # os.system("gdaldem color-relief " + os.path.join(path_tifs_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif") + " " + p_cmp + " " + os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                # gdal.Translate(os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".png"),os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"),format= 'PNG', width=800,outputType = gdal.GDT_Byte)
                # os.remove(os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                
                
                # #gdal.Translate(os.path.join(path_results_date,"S2-FSC_" + tile +"_" + epsg +".png"),g_CMP,format= 'PNG', width=1000,outputType = gdal.GDT_Byte, scaleParams=[[0,2,1,255]],noData = "0")
                
                
                

    
    return True
