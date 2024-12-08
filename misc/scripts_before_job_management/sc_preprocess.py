#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:08:25 2020

@author: zacharie
"""


import sys
import os
import errno
import re
import copy
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






def makeDataSets(d,p,s):
    
    source = s["source"]
    out = s["out"]
    tiles = s["tiles"]
    epsg = s["epsg"]
    resample = s["resample"]
    selection = s["selection"]
    manual = s["manual"]
    snw = s["snw"]
    nosnw = s["nosnw"]
    nodt = s["nodt"]
    start_date = d["start_date"]
    end_date = d["end_date"]
    path_outputs = p["path_outputs"]
    path_inputs = p["path_inputs"]
    path_LIS = p["path_LIS"]
    nb_shift_days = d["nb_shift_days"]

    list_products = {}
    

    if manual != "" and not os.path.isfile(manual):
        print("ERROR snowcover : manual selection file path unspecified")
        return False 
    elif manual != "":
        with open(manual, "r") as f :
            line = f.readline()
            date = None
            while line :
                l = line.split()
                s = l[0]
                if s == "FSC" : 
                   pFSC = l[1] 
                   date = getDateFromStr(pFSC)
                   list_products[date] = [pFSC,[]]
                elif s == "LIS" :
                   tile = l[1]
                   epsg = l[2]
                   pLIS = l[3]
                   list_products[date][1].append(tile,epsg,pLIS)
                line = f.readline()
    else :
        #search overlapping tiles
        if tiles == [] :
            print("Searching for overlapping tiles")
            tiles = searchTiles(source,epsg,path_inputs,path_LIS)
            if tiles == []:
                print("ERROR makeDataSet : no tiles found")
                return False
        print("tiles: ",tiles)
        #select FSC and L2A products
        print("Selecting FSC and L2A products")
        list_products = selectProducts(start_date,end_date,source,epsg,tiles,selection,path_inputs,path_LIS,nb_shift_days)
        if list_products == {} :
            print("ERROR selectProducts : no products found")
            return False

    #display products
    print("nb of FSC products = " + str(len(list_products)))
    for FSC_date in list_products :
        f_FSC = list_products[FSC_date][0]
        l_L2A = list_products[FSC_date][1]
        print("DATE = " + str(FSC_date) + "\n     FSC = " + str(f_FSC) + "\n     nb of L2A tiles = " + str(len(l_L2A)) )
        for tile , epsgL2A, L2A  in l_L2A : 
            print("     " + tile + " : " + L2A)

    #produce datasets
    success = products2DataSets(start_date,end_date,source,out,epsg,list_products,resample,snw,nosnw,nodt,path_outputs)

    return success




    
def searchTiles(source,epsg,path_inputs,path_LIS):

    tiles_overlap = []
    
    path_FSC_dir = os.path.join(path_inputs,source)
    pxs = []
    for FSC_product in os.listdir(path_FSC_dir):
        f_FSC = os.path.join(path_FSC_dir,FSC_product)
        g_FSC = gdal.Open(f_FSC)
        px = g_FSC.GetGeoTransform()[0]
        if px not in pxs :
            print("\n")
            print("Check tiles for FSC file",FSC_product)
            pxs.append(px)
            # we get/set the FSC projection system
            
            if epsg != "" :
                g_FSC = gdal.Warp('',g_FSC,format= 'MEM',srcSRS="EPSG:" + epsg)
            # we check each S2 tiles for overlaps
            for tile in os.listdir(path_LIS) :
                path_tile = os.path.join(path_LIS,tile)
                path_NDSI = ""
                print ("Check tile : " + tile)
                try:
                    path_year = os.path.join(path_tile,os.listdir(path_tile)[-1])
                    path_month = os.path.join(path_year,os.listdir(path_year)[-1])
                    path_day = os.path.join(path_month,os.listdir(path_month)[-1])
                    path_product = os.path.join(path_day,os.listdir(path_day)[-1])
                    print("path_product",path_product)
                    path_NDSI = glob.glob(os.path.join(path_product,'*NDSI*'))[0]
                    
                except (OSError,IndexError) as exc:  # Python >2.5
                    print("access not permitted!")
                    continue
                        
                print("Check overlapping with LIS NDSI product " + path_NDSI)
                g_L2A = gdal.Open(path_NDSI)
                if isOverlapping(g_L2A,g_FSC) :
                    print("Overlap present")
                    if tile not in tiles_overlap :
                        tiles_overlap.append(tile)
            print("\n")
    return tiles_overlap
            






def selectProducts(start_date,end_date,source,epsg,tiles,selection,path_inputs,path_LIS,nb_shift_days):

    list_products = {}

    # We create a list of the FSC products (with paths)
    
    path_FSC_dir = os.path.join(path_inputs,source)
    if os.path.isdir(path_FSC_dir):
        list_FSC_products = getListDateDecal(start_date,end_date,path_FSC_dir,0,"")
    if list_FSC_products == [] :
        print ("ERROR selectProducts : No FSC product found for source " + source + " in directory " + path_FSC_dir)
        return list_products
        
    for tile in tiles :
            
        print("Check tile : " + tile)
        path_LIS_dir = os.path.join(path_LIS,tile)
        if os.path.isdir(path_LIS_dir):
            list_LIS_products = getListDateDecal(start_date,end_date,path_LIS_dir,nb_shift_days,"/FSC_*/")
        if list_LIS_products == [] :
            print ("No LIS product found for tile " + tile + " in directory " + path_LIS_dir)
                    
        g_tile = gdal.Open(glob.glob(os.path.join(list_LIS_products[0],'*NDSI*'))[0])
        
        for f_FSC in list_FSC_products :
            
            g_FSC = gdal.Open(f_FSC)

            if epsg != "" :
                g_FSC = gdal.Warp('',g_FSC,format= 'MEM',srcSRS="EPSG:" + epsg )                

            dateFSC = getDateFromStr(f_FSC)
            minx, maxy, maxx, miny = getOverlapCoords(g_FSC,g_tile)  
            if minx == None and maxy == None: continue
            
            
            
            epsgLIS = ""
            LIS = ""
            ind = nb_shift_days + 1
            NDR1 = 100
            NDR2 = 100
            for LIS_product in list_LIS_products:
                dateLIS = getDateFromStr(LIS_product)
                lag = dateLIS- dateFSC
                if abs(lag.days) >  nb_shift_days : continue
                
                g_LIS = gdal.Translate('',glob.glob(os.path.join(LIS_product,'*NDSI*'))[0],format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
                bandLIS  = BandReadAsArray(g_LIS.GetRasterBand(1))
                

                cond = np.where((bandLIS < 0) | (bandLIS  > 100) | np.isnan(bandLIS))
                NDR2 = (float(len(bandLIS[cond])) / float(np.size(bandLIS))) * 100
                
                            
                check = None
                
                if "cleanest" in selection:  
                    
                    check =  NDR2 < 100 and ((abs(NDR2 - NDR1) < 0.0001 and abs(lag.days) < ind) or (NDR1 - NDR2 >= 0.0001))
                else :
                    check = NDR2 < 100 and abs(lag.days) < ind
                
                if check :
                    #print("\n")
                    print("date FSC",dateFSC,"date LIS",dateLIS)
                    print("NoDataRatio1 = ",NDR1,"NoDataRatio2 = ",NDR2,"lag = ",lag.days)
                    #print("\n")
                    ind = abs(lag.days)
                    LIS= LIS_product
                    NDR1 = NDR2
                    epsgLIS = (gdal.Info(g_LIS, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
                else : print("date rejetee")
                    
                    
            if LIS == "" : continue
            
            print("Chosen LIS : " + LIS)
            
            if dateFSC not in list_products.keys() :
                list_products[dateFSC] = [f_FSC,[]]
                list_products[dateFSC][1].append([tile,epsgLIS,LIS])
            else :
                
                list_products[dateFSC][1].append([tile,epsgLIS,LIS])
            
            
    return list_products






def products2DataSets(start_date,end_date,source,out,epsgFSC,list_products,resample,snw,nosnw,nodt,path_outputs):
    
    

    shutil.rmtree(os.path.join(path_outputs,out),ignore_errors=True)
    dataSetDir = os.path.join(path_outputs,out,"TIFS")
    mkdir_p(dataSetDir)
    nb_results = 0

    #DEFINITION DE LA PERIODE D'ANALYSE##########################################
    #On prend une date de debut et une date de fin
    print("\nPeriode d'analyse : " + start_date + "-" + end_date)


    #POUR CHAQUE DATE:##########################################
    for dateFSC in list_products :
 
        epsg = ""
        
        nd = 100000
        dir_tifs_date = os.path.join(dataSetDir,dateFSC.strftime("%Y-%m-%d"))
        mkdir_p(dir_tifs_date)
   
    
        f_FSC = list_products[dateFSC][0]
        l_LIS = list_products[dateFSC][1]
        
        print("\nCalcul pour : " + dateFSC.strftime("%Y-%m-%d"))
        print(f_FSC)
        print(l_LIS)
        
        
        # we get the FSC projection system
       
        if epsgFSC == "" :
            epsg = (gdal.Info(f_FSC, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
        else :
            epsg = epsgFSC
        print(epsg)
        

        
        # On ouvre, converti et re-echantillonne le FSC
        
        print("\nConversion des valeurs FSC")
        #on change les valeurs FSC
        g_FSC_o = None
        g_FSC_o = gdal.Warp('',f_FSC,format= 'MEM',outputType = gdal.GDT_Float32)
        
        g_FSC_o = gdal.Warp('',g_FSC_o,format= 'MEM', dstNodata = 9999)
        g_FSC_o = gdal.Translate('',g_FSC_o,format= 'MEM',noData = nd)

        
        a_FSC = BandReadAsArray(g_FSC_o.GetRasterBand(1))
        if len(nodt) > 0 :
            for nData in nodt :
                cond = np.where((a_FSC == nData) | (np.isnan(a_FSC)))
                a_FSC[cond] = 9999
        if len(nosnw) > 0 :
            for noSnow in nosnw :
                a_FSC[a_FSC == noSnow] = 0
        if len(snw) > 0 :
            for snow in snw :
                a_FSC[a_FSC == snow] = 1
        g_FSC_o.GetRasterBand(1).WriteArray(a_FSC)
        a_FSC = None
        
        gdal.Translate(os.path.join(dir_tifs_date,"INPUT_FSC.tif"),g_FSC_o,format= 'GTiff',noData = 9999)
        

        
        print("\nTraitement des tuiles")
        
       

        l_g_FSC = {}
        

        
        # On prepare un FSC reprojete pour chaque projection
        for tile , epsgS2 , LIS_product in  l_LIS: 
            
            if epsgS2 not in l_g_FSC.keys():
                g_FSC = gdal.Warp('',g_FSC_o,format= 'MEM',srcSRS="EPSG:" + epsg,dstSRS="EPSG:" + epsgS2,resampleAlg=resample,xRes= 20,yRes= 20)
                a_FSC = BandReadAsArray(g_FSC.GetRasterBand(1))
                cond = np.where(a_FSC > 1 | np.isnan(a_FSC) | np.isinf(a_FSC))
                a_FSC[cond ] = 255
                g_FSC.GetRasterBand(1).WriteArray(a_FSC)
                l_g_FSC[epsgS2] = g_FSC
                print("resnodata999",str(len(a_FSC[a_FSC == 255])))
                print("resnodataNAN",str(len(a_FSC[np.isnan(a_FSC)])))
                gdal.Translate(os.path.join(dir_tifs_date,"RESAMPLED_FSC_EPSG-"+epsgS2+".tif"),g_FSC,format= 'GTiff',noData = 255)
                g_FSC = None
                
        
        print(l_g_FSC)

        
        for tile , epsgS2 , LIS_product in  sorted(l_LIS,key=lambda l:l[1]) : 
            
            print(tile,epsgS2,LIS_product)
            f_TOC = glob.glob(os.path.join(LIS_product,'*FSCTOC.tif'))[0]
            f_OG = glob.glob(os.path.join(LIS_product,'*FSCOG.tif'))[0]
            f_QCTOC = glob.glob(os.path.join(LIS_product,'*QCTOC.tif'))[0]
            f_QCOG = glob.glob(os.path.join(LIS_product,'*QCOG.tif'))[0]
            f_QC = glob.glob(os.path.join(LIS_product,'*QCFLAGS.tif'))[0]
            f_NDSI = glob.glob(os.path.join(LIS_product,'*NDSI.tif'))[0]

            print(f_TOC)
    
            #If there is a file missing, we skip to the next tile
            if f_TOC == "" or f_OG == "" or f_QCTOC == "" or f_QCOG  == "" or f_QC  == "" or f_NDSI == "" :  continue
            
            
            # On calcul les coord de overlap dans la projection LIS
            print("\nCalcul coordonnees de chevauchement")
            g_TOC = gdal.Open(f_TOC)
            print(l_g_FSC[epsgS2])
            print(isOverlapping(l_g_FSC[epsgS2],g_TOC))
            minx, maxy, maxx, miny = getOverlapCoords(l_g_FSC[epsgS2],g_TOC)
            print (minx)
            print (maxy)
            print (maxx)
            print (miny)
            if minx == None and maxy == None: continue
            
            #on decoupe les fichiers LIS
            print("\nDecoupage du FSCTOC")
            g_TOC = gdal.Translate(os.path.join(dir_tifs_date,"INPUT_FSCTOC_" + tile + "_EPSG-" + epsgS2 + ".tif"),g_TOC,format= 'GTiff',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32)
            print("\nDecoupage du FSCOG")
            g_OG = gdal.Translate(os.path.join(dir_tifs_date,"INPUT_FSCOG_" + tile + "_EPSG-" + epsgS2 + ".tif"),f_OG,format= 'GTiff',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32)
            print("\nDecoupage du QCTOC")
            g_QCTOC = gdal.Translate(os.path.join(dir_tifs_date,"INPUT_QCTOC_" + tile + "_EPSG-" + epsgS2 + ".tif"),f_QCTOC,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
            print("\nDecoupage du QCOG")
            g_QCOG = gdal.Translate(os.path.join(dir_tifs_date,"INPUT_QCOG_" + tile + "_EPSG-" + epsgS2 + ".tif"),f_QCOG,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
            print("\nDecoupage du QC")
            gdal.Translate(os.path.join(dir_tifs_date,"INPUT_QC_" + tile + "_EPSG-" + epsgS2 + ".tif"),f_QC,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
            print("\nDecoupage du NDSI")
            g_NDSI = gdal.Translate(os.path.join(dir_tifs_date,"INPUT_NDSI_" + tile + "_EPSG-" + epsgS2 + ".tif"),f_NDSI,format= 'GTiff',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32)
                   
            #on decoupe une copie de FSC
            g_FSC = gdal.Translate('',l_g_FSC[epsgS2],format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
            
            
            #on extrait les bandes LIS
            TOC = BandReadAsArray(g_TOC.GetRasterBand(1))
            OG = BandReadAsArray(g_OG.GetRasterBand(1))
            QCTOC = BandReadAsArray(g_QCTOC.GetRasterBand(1))
            QCOG = BandReadAsArray(g_QCOG.GetRasterBand(1))
            NDSI = BandReadAsArray(g_NDSI.GetRasterBand(1))
            #on extrait la bande FSC         
            FSC = BandReadAsArray(g_FSC.GetRasterBand(1))
            

            #On remplace les pixels non utilisables par des nodatas
            
            cond = np.where((TOC == 255) | (TOC == 205) | (FSC == 255) | np.isnan(FSC) | np.isinf(FSC))
            FSC[cond] = 255
            TOC[cond] = 255
            OG[cond] = 255
            QCTOC[cond] = 255
            QCOG[cond] = 255
            NDSI[cond] = 255
            
            NDSI[NDSI <= 100] = NDSI[NDSI <= 100] /100.0
            TOC[TOC <= 100] = TOC[TOC <= 100] /100.0
            OG[OG <= 100] = OG[OG <= 100] /100.0
            
            g_TOC.GetRasterBand(1).WriteArray(TOC)
            g_OG.GetRasterBand(1).WriteArray(OG)
            g_QCTOC.GetRasterBand(1).WriteArray(QCTOC)
            g_QCOG.GetRasterBand(1).WriteArray(QCOG)
            g_NDSI.GetRasterBand(1).WriteArray(NDSI)
            g_FSC.GetRasterBand(1).WriteArray(FSC)
            

            gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),g_FSC,format= 'GTiff',noData = 255)
            gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),g_NDSI,format= 'GTiff',noData = 255)
            gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_FSCTOC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),g_TOC,format= 'GTiff',noData = 255)
            gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_FSCOG_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),g_OG,format= 'GTiff',noData = 255)
            gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_QCTOC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),g_QCTOC,format= 'GTiff',noData = 255)
            gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_QCOG_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),g_QCOG,format= 'GTiff',noData = 255)           
            
            for proj in l_g_FSC :
                g_FSC_m = gdal.Warp('',g_FSC,format= 'MEM',dstSRS="EPSG:" + proj,xRes= 20,yRes= 20)

                MASK = BandReadAsArray(g_FSC_m.GetRasterBand(1))

                MASK[MASK != 255 ] = 255
                
                g_FSC_m.GetRasterBand(1).WriteArray(MASK)
                
                
                l_g_FSC[proj] = gdal.Warp('',[l_g_FSC[proj],g_FSC_m],format= 'MEM')

            

        nb_results += 1


    
    if nb_results > 0 :
       print("\nNumber of processed dates:" + str(nb_results))
       return True
    else :
       print("\nERROR products2DataSets : No date processed")
       return False    
       
       
       
       
       
       
       
       
       
def processDP(DP,p,d):
    source = DP["source"]
    out = DP["out"]
    epsg = DP["epsg"]
    filter_acc = DP["filter_acc"]
    filter_snw = DP["filter_snw"]
    max_acc = DP["max_acc"]
    path_outputs = p["path_outputs"]
    path_inputs = p["path_inputs"]
    path_LIS = p["path_LIS"]
    start_date = d["start_date"]
    end_date = d["end_date"]
    nb_shift_days = d["nb_shift_days"]
    path_tree = p["path_tree"]

    dataSetDir = os.path.join(path_outputs,out)
    DPList = os.path.join(dataSetDir,"LIST.txt")
    DPInfos = os.path.join(dataSetDir,"INFOS.txt")


    DPdir = os.path.join(path_inputs,source)
    shutil.rmtree(dataSetDir, ignore_errors=True)
    mkdir_p(dataSetDir)

    dict_tile = {}
    dict_products = {}
    dict_FSC = {}


    print("####################################################")
    print("Recuperation of data points")
    #on recupere les donnees odk
    for f in os.listdir(DPdir) :    
        with open(os.path.join(DPdir,f), "r") as datapoints :
            line = datapoints.readline()
            line = datapoints.readline()
            while line :
                point = line.split()
                date = point[0]
                latitude = point[1]
                longitude = point[2]
                accuracy = point[3]
                snow = point[4]
                if filter_acc == True and float(accuracy) > max_acc  : 
                    line = datapoints.readline()
                    continue
                
                if date not in dict_FSC.keys() :
                    dict_FSC[date] = []
                    dict_FSC[date].append([latitude,longitude,snow,accuracy])
                else :
                    dict_FSC[date].append([latitude,longitude,snow,accuracy])
                
                line = datapoints.readline()
                


    print("####################################################")
    print("Search of tiles and L2A rasters")
    #on trouve les tuiles et rasters correspondants
    p_val = 0
    p_a = 0
    for date in dict_FSC :
        #print("check date: ",date)
        list_points = dict_FSC[date]
        dateFSC = getDateFromStr(date)


        for point in list_points :
            
            lat = point[0]
            lon = point[1]
            snow = point[2]
            acc = point[3]
            #print(" check point : ",lat,lon)


        
            decalP = nb_shift_days + 1    
            tileP = ""
            p_L2AP = ""   

            for tile in os.listdir(path_LIS) :
                path_tile = os.path.join(path_LIS,tile)
                path_FSCTOC = ""
               
                try:
                    path_year = os.path.join(path_tile,os.listdir(path_tile)[0])
                 
                    path_month = os.path.join(path_year,os.listdir(path_year)[-1])
               
                    path_day = os.path.join(path_month,os.listdir(path_month)[-1])
             
                    path_product = os.path.join(path_day,os.listdir(path_day)[-1])
                   
                    path_FSCTOC = glob.glob(os.path.join(path_product,'*FSCTOC.tif'))[0]
             
                except (OSError,IndexError) as exc:  # Python >2.5
                    continue
                    
                #print("date:",date,"tile:",tile,"coord:",lon+"_"+lat,"valide:",p_val,"accepte:",p_a)
                
                if not isCoordInside(gdal.Open(path_FSCTOC),lon,lat,epsg) : continue
                
                
                pixel = os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (path_FSCTOC, lon, lat)).read()
                
                #print("date:",date,"tile:",tile,"coord:",lon+"_"+lat,"pixel",pixel,"valide:",p_val,"accepte:",p_a)
                
                
                try:
                    int(pixel)
                    
                except ValueError:
                    continue
                
                #LIS_products = glob.glob(os.path.join(path_tile,'**','FSC_*/'))
                LIS_products = getListDateDecal(start_date,end_date,path_tile,nb_shift_days,"/FSC_*/")

                for LIS_product in LIS_products:
                    dateL2A = getDateFromStr(LIS_product)
                    decal = dateL2A - dateFSC
                    if abs(decal.days) >= decalP : continue

                    f_L2A = glob.glob(os.path.join(LIS_product,'*FSCTOC*'))[0]
                    pixel = int(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_L2A, lon, lat)).read())

                    if pixel > 100 or np.isnan(pixel) or np.isinf(pixel) or (pixel == 0 and filter_snw == True): continue

                    decalP = abs(decal.days)
                    p_L2AP = LIS_product
                    tileP = tile
                    p_val = p_val + 1
                    #print("date:",date,"tile:",tile,"coord:",lon+"_"+lat,"pixel",pixel,"valide:",p_val,"accepte:",p_a)

            if p_L2AP == "":
                #print("  point rejete")
                #print("date:",date,"tile:",tile,"coord:",lon+"_"+lat,"pixel",pixel,"valide:",p_val,"accepte:",p_a)
                continue
            else :
                #print("  point accepte")
                p_a = p_a + 1
                #print("date:",date,"tile:",tile,"coord:",lon+"_"+lat,"pixel",pixel,"valide:",p_val,"accepte:",p_a)

            f_L2AP = os.path.basename(p_L2AP)

            #we get the tree mask value of the pixel
            forest = int(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (path_tree, lon, lat)).read())


            if dateFSC not in dict_products.keys() :
                dict_products[dateFSC] = []
                dict_products[dateFSC].append([lat,lon,snow,acc,decalP,p_L2AP,f_L2AP,tileP,forest])
            else :
                dict_products[dateFSC].append([lat,lon,snow,acc,decalP,p_L2AP,f_L2AP,tileP,forest])
                
          
    f_DPList= open(DPList,"w")
    f_DPInfos= open(DPInfos,"w")
    #on affiche le dict
    print("####################################################")
    print("\n")
    nb_points = 0
    f_DPList.write("date lat lon snow acc decal L2A tile forest")
    for date in dict_products :
        print(date)
        
        for point in dict_products[date] :
            print ("TILE : ",point[7])
            if point[7] not in dict_tile.keys():
                dict_tile[point[7]] = [1,0]
            else :
                dict_tile[point[7]][0] = dict_tile[point[7]][0] + 1
            if point[8] > 0 :
                dict_tile[point[7]][1] = dict_tile[point[7]][1] + 1
            print ("L2A product : ",point[6])
            print("lat = ",point[0],"lon = ",point[1],"snow = ",point[2],"acc = ",point[3],"decal = ",point[4],"forest value = ",point[8])
            f_DPList.write("\n"+date.strftime("%Y-%m-%d")+" "+str(point[0])+" "+str(point[1])+" "+str(point[2])+" "+str(point[3])+" "+str(point[4])+" "+point[5]+" "+point[7]+" "+str(point[8]))
            nb_points = nb_points + 1
        print("\n")
    print("nb of points = ",nb_points)

    #on affiche le nombre de points par tuile
    for tile in dict_tile :
        line = "TILE : " + tile + " ; NB of points : " + str(dict_tile[tile][0]) + " ; NB of points in forest : " + str(dict_tile[tile][1])
        print(line)
        f_DPInfos.write("\n" + line)

    f_DPInfos.write("\nTOTAL NB OF POINTS : " + str(nb_points))

    f_DPList.close()
    f_DPInfos.close()

    return True








