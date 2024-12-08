#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:26:44 2020

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
import sc_utils






class snowcover:
    def __init__(self,start_date = "",
                      end_date = "", 
                      date_format = "",
                      nb_shift_days = 0,
                      path_palettes = "",
                      path_outputs = "",
                      path_inputs = "",
                      path_LIS = "",
                      path_tree = ""):

        self.start_date = start_date
        self.end_date = end_date
        self.date_format = date_format
        self.nb_shift_days = nb_shift_days
        self.path_palettes = path_palettes
        self.path_outputs = path_outputs
        self.path_inputs = path_inputs
        self.path_LIS = path_LIS
        self.path_tree = path_tree

        if self.end_date == "": 
            self.end_date = self.start_date
        elif self.getDateFromStr(self.start_date) == '' or self.getDateFromStr(self.end_date) == '':
            print("ERROR snowcover : error in input dates")
            return None
        elif self.path_inputs == "":
            print("ERROR snowcover : inputs directory unspecified")
            return None
        elif not os.path.isdir(self.path_inputs):
            print("ERROR snowcover :",self.path_inputs,"is not a directory")
            return None
        elif self.path_outputs == "":
            print("ERROR snowcover : outputs directory unspecified")
            return None
        elif self.path_LIS == "":
            print("ERROR snowcover : LIS products directory unspecified")
            return None
        elif not os.path.isdir(self.path_LIS):
            print("ERROR snowcover :",self.path_LIS,"is not a directory")
            return None
        elif self.path_palettes != "" and not os.path.isdir(self.path_palettes):
            print("ERROR snowcover : color palettes directory unspecified")
            return None
        elif self.path_tree != "" and not os.path.isfile(self.path_tree):
            print("ERROR snowcover : tree mask path unspecified")
            return None




        



    
    
    

        


    def makeDataSets(self,out = "",source = "",epsg = "",resample = "average",snw = [],nosnw = [],nodt = [], tiles = [],selection = "closest", manual = ""):

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
                       date = self.getDateFromStr(pFSC)
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
                tiles = self.searchTiles(source,epsg)
                if tiles == []:
                    print("ERROR makeDataSet : no tiles found")
                    return False
            print("tiles: ",tiles)
            #select FSC and L2A products
            print("Selecting FSC and L2A products")
            list_products = self.selectProducts(source,epsg,tiles,selection)
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
        success = self.products2DataSets(source,out,epsg,list_products,resample,snw,nosnw,nodt)

        return success




        
    def searchTiles(self,source,epsg):

        tiles_overlap = []
        
        path_FSC_dir = os.path.join(self.path_inputs,"FSC",source)
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
                for tile in os.listdir(self.path_LIS) :
                    if not os.path.isdir(os.path.join(self.path_LIS,tile)) : continue
                    print ("Check tile : " + tile)
                    try:
                        L2A_product = os.listdir(os.path.join(self.path_LIS,tile))[-1]
                    except OSError as exc:  # Python >2.5
                        if exc.errno == errno.EACCES:
                            continue
                        else:
                            raise   
                    print("Check overlapping with L2A file " + L2A_product)
                    f_L2A = os.path.join(self.path_LIS,tile,L2A_product,"swir_band_extracted.tif")
                    g_L2A = gdal.Open(f_L2A)
                    if self.isOverlapping(g_L2A,g_FSC) :
                        print("Overlap present")
                        if tile not in tiles_overlap :
                            tiles_overlap.append(tile)
                print("\n")
        return tiles_overlap
                






    def selectProducts(self,source,epsg,tiles,selection):



        list_products = {}



        # We create a list of the FSC products (with paths)
        
        path_FSC_dir = os.path.join(self.path_inputs,"FSC",source)
        if os.path.isdir(path_FSC_dir):
            list_FSC_products = self.getListDateDecal(self.start_date,self.end_date,path_FSC_dir,0)
        if list_FSC_products == [] :
            print ("ERROR selectProducts : No FSC product found for source " + source + " in directory " + path_FSC_dir)
            return list_products
            
                
                
            
        for tile in tiles :
                
            print("Check tile : " + tile)
            path_L2A_dir = os.path.join(self.path_LIS,tile)
            if os.path.isdir(path_L2A_dir):
                list_L2A_products = self.getListDateDecal(self.start_date,self.end_date,path_L2A_dir,self.nb_shift_days)
            if list_L2A_products == [] :
                print ("No L2A product found for tile " + tile + " in directory " + path_L2A_dir)
                        
            

            L2A_product = glob.glob(os.path.join(self.path_LIS,tile,'*SENTINEL*'))[0]
            f_tile = os.path.join(L2A_product,"LIS_PRODUCTS","LIS_SEB.TIF")
            g_tile = gdal.Open(f_tile)
            
            

            
            
            for f_FSC in list_FSC_products :
                
                
                g_FSC = gdal.Open(f_FSC)

                if epsg != "" :
                    g_FSC = gdal.Warp('',g_FSC,format= 'MEM',srcSRS="EPSG:" + epsg )                
    
                dateFSC = self.getDateFromStr(f_FSC)
                minx, maxy, maxx, miny = self.getOverlapCoords(g_FSC,g_tile)  
                if minx == None and maxy == None: continue
                
                
                
                epsgL2A = ""
                L2A = ""
                ind = self.nb_shift_days + 1
                NDR1 = 100
                NDR2 = 100
                for L2A_product in list_L2A_products:
                    if "SENTINEL" not in L2A_product : continue
                    dateL2A = self.getDateFromStr(L2A_product)
                    lag = dateL2A - dateFSC
                    if abs(lag.days) >  self.nb_shift_days : continue
                    
                    f_L2A = os.path.join(L2A_product,"LIS_PRODUCTS","LIS_SEB.TIF")
                    g_L2A = gdal.Translate('',f_L2A,format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
                    bandL2A = BandReadAsArray(g_L2A.GetRasterBand(1))
                    

                    
                    NDR2 = (float(len(bandL2A[bandL2A == 205]) + len(bandL2A[bandL2A == 254])) / float(np.size(bandL2A))) * 100
                    
                                
                    check = None
                    
                    if "cleanest" in selection :   
                        check =  NDR2 < 100 and ((abs(NDR2 - NDR1) < 0.0001 and abs(lag.days) < ind) or (NDR1 - NDR2 >= 0.0001))
                    else :
                        check = abs(lag.days) < ind
                    
                    if check :
                        #print("\n")
                        print("date FSC",dateFSC,"date L2A",dateL2A)
                        print("NoDataRatio1 = ",NDR1,"NoDataRatio2 = ",NDR2,"lag = ",lag.days)
                        #print("\n")
                        ind = abs(lag.days)
                        L2A = L2A_product
                        NDR1 = NDR2
                        epsgL2A = (gdal.Info(g_L2A, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
                    else : print("date rejetee")
                        
                        
                if L2A == "" : continue
                
                print("Chosen L2A : " + L2A)
                
                if dateFSC not in list_products.keys() :
                    list_products[dateFSC] = [f_FSC,[]]
                    list_products[dateFSC][1].append([tile,epsgL2A,L2A])
                else :
                    
                    list_products[dateFSC][1].append([tile,epsgL2A,L2A])
                
                
        return list_products





   
    def products2DataSets(self,source,out,epsg,list_products,resample,snw,nosnw,nodt):
        
        

        
        dataSetDir = os.path.join(self.path_outputs,out,"TIFS")
        shutil.rmtree(dataSetDir,ignore_errors=True)
        self.mkdir_p(dataSetDir)
        nb_results = 0
    
        #DEFINITION DE LA PERIODE D'ANALYSE##########################################
        #On prend une date de debut et une date de fin
        print("\nPeriode d'analyse : " + self.start_date + "-" + self.end_date)

    
        #POUR CHAQUE DATE:##########################################
        for dateFSC in list_products :

            nd = 100000
            dir_tifs_date = os.path.join(dataSetDir,dateFSC.strftime(self.date_format))
            self.mkdir_p(dir_tifs_date)
       
        
            f_FSC = list_products[dateFSC][0]
            l_L2A = list_products[dateFSC][1]
            
            print("\nCalcul pour : " + dateFSC.strftime(self.date_format))
            
            
            
            # we get the FSC projection system
           
            if epsg == "" :
                epsg = (gdal.Info(f_FSC, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
           
            
            

            
            # On ouvre, converti et re-echantillonne le FSC
            
            print("\nConversion des valeurs FSC")
            #on change les valeurs FSC
            g_FSC_o = None
            g_FSC_o = gdal.Warp('',f_FSC,format= 'MEM',outputType = gdal.GDT_Float32)
            
            g_FSC_o = gdal.Warp('',g_FSC_o,format= 'MEM', dstNodata = 9999)
            g_FSC_o = gdal.Translate('',g_FSC_o,format= 'MEM',noData = nd)



            
            
            
            #g_FSC_o = gdal.Translate('',g_FSC_o,format= 'MEM',noData = None)
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
            for tile , epsgS2 , L2A_product in  l_L2A : 
                
                if epsgS2 not in l_g_FSC.keys():
                    g_FSC = gdal.Warp('',g_FSC_o,format= 'MEM',srcSRS="EPSG:" + epsg,dstSRS="EPSG:" + epsgS2,resampleAlg=resample,xRes= 20,yRes= 20)
                    a_FSC = BandReadAsArray(g_FSC.GetRasterBand(1))
                    a_FSC[np.isnan(a_FSC)] = 9999
                    a_FSC[a_FSC > 1] = 9999
                    g_FSC.GetRasterBand(1).WriteArray(a_FSC)
                    a_FSC = None
                    #g_FSC = gdal.Warp('',g_FSC,format= 'MEM',dstNodata = 9999)
                    l_g_FSC[epsgS2] = g_FSC
                    a_FSC = BandReadAsArray(g_FSC.GetRasterBand(1))
                    print("resnodata999",str(len(a_FSC[a_FSC == 9999])))
                    print("resnodataNAN",str(len(a_FSC[np.isnan(a_FSC)])))
                    gdal.Translate(os.path.join(dir_tifs_date,"RESAMPLED_FSC_EPSG-"+epsgS2+".tif"),g_FSC,format= 'GTiff',noData = 9999)
                    
                    g_FSC = None
                    
                    


            
            for tile , epsgS2 , L2A_product in  sorted(l_L2A,key=lambda l:l[1]) : 
                
                
                # We look for the red, green & swir bands tiff files + mask
                f_green = ""
                f_swir = ""
                f_red = ""
                f_MSK = ""
                f_compo = ""
    
        
                for f in os.listdir(L2A_product) :
                    if ("green_band_resampled.tif" in f) :
                        f_green = os.path.join(L2A_product,f)
                    elif ("red_band_resampled.tif" in f) :
                        f_red = os.path.join(L2A_product,f)
                    elif ("swir_band_extracted.tif" in f) :
                        f_swir = os.path.join(L2A_product,f)
                    elif ("LIS_PRODUCTS" in f) :
                        if os.path.isfile(os.path.join(L2A_product,f,"LIS_SEB.TIF")):
                            f_msk = os.path.join(L2A_product,f,"LIS_SEB.TIF")
                        if os.path.isfile(os.path.join(L2A_product,f,"LIS_COMPO.TIF")):
                            f_compo = os.path.join(L2A_product,f,"LIS_COMPO.TIF")

    
        
                #If there is a file missing, we skip to the next tile
                if f_green == "" or f_red == "" or f_swir == "" or f_msk == "": continue
                
                
                # On calcul les coord de overlap dans la projection L2A
                print("\nCalcul coordonnees de chevauchement")
                g_msk = gdal.Open(f_msk)
                minx, maxy, maxx, miny = self.getOverlapCoords(l_g_FSC[epsgS2],g_msk)
                
                
                #on decoupe les fichiers L2A
                #on decoupe le masque 
                print("\nDecoupage du masque")
                g_msk= gdal.Translate(os.path.join(dir_tifs_date,"INPUT_SEB_" + tile + "_EPSG-" + epsgS2 + ".tif"),g_msk,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
                print("\nDecoupage du compo")
                gdal.Translate(os.path.join(dir_tifs_date,"INPUT_COMPO_" + tile + "_EPSG-" + epsgS2 + ".tif"),f_compo,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
                
                #on load et decoupe l'image bande verte, 
                print("\nDecoupage bande verte")
                g_green = gdal.Translate('',f_green,format= 'MEM',projWin = [minx, maxy, maxx, miny])
                #on load et decoupe l'image bande rouge
                print("\nDecoupage bande rouge")
                g_red = gdal.Translate('',f_red,format= 'MEM',projWin = [minx, maxy, maxx, miny])        
                #on load et decoupe l'image bande swir
                print("\nDecoupage bande IR")
                g_swir= gdal.Translate('',f_swir,format= 'MEM',projWin = [minx, maxy, maxx, miny])                   
                
                
                #on decoupe une copie de FSC
                g_FSC_c = gdal.Translate('',l_g_FSC[epsgS2],format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
                
                #on produit un raster avec les memes conditions
                raster = g_FSC_c.copy()
                
                
                #on calcul les NDSI pour le chevauchement
                print("\nCalcul des NDSI")
                bandV = BandReadAsArray(g_green.GetRasterBand(1))
                g_green = None
                bandIR = BandReadAsArray(g_swir.GetRasterBand(1))
                g_swir = None
                bandR = BandReadAsArray(g_red.GetRasterBand(1))
                g_red = None           
                #on extrait la bande neige
                MSK = BandReadAsArray(g_msk.GetRasterBand(1))
                g_msk = None  
                #on extrait la bande FSC         
                FSC = BandReadAsArray(g_FSC_c.GetRasterBand(1))
                

                
                #On calcul les NDSI
                a = (bandV - bandIR).astype(float)
                b = (bandV + bandIR).astype(float)
                NDSI = a/b
                

                
                #On remplace les pixels non utilisables par des nodatas
                
                cond1 = np.where((MSK != 100) )
                NDSI[cond1] = 9999
                FSC[cond1] = 9999  
                MSK = None
                cond2 = np.where(FSC > 1 | np.isnan(FSC) | np.isinf(FSC))
                NDSI[cond2] = 9999
                FSC[cond2] = 9999  
                cond3 = np.where(np.isnan(NDSI) | np.isinf(NDSI))
                FSC[cond3] = 9999  
                NDSI[cond3] = 9999
                

                       
                
                cond5 = np.where((NDSI < 0) | (NDSI > 1))
                FSC[cond5] = 9999 
                NDSI[cond5] = 9999            
                
            
                
                
                raster.GetRasterBand(1).WriteArray(NDSI)

                gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),raster,format= 'GTiff',noData = 9999)
                raster.GetRasterBand(1).WriteArray(FSC)

                gdal.Translate(os.path.join(dir_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif"),raster,format= 'GTiff',noData = 9999)
                
                
                
                for proj in l_g_FSC :
                    
                    
                    g_FSC_m = gdal.Warp('',g_FSC_c,format= 'MEM',dstSRS="EPSG:" + proj,xRes= 20,yRes= 20)
                    NODATA = BandReadAsArray(g_FSC_m.GetRasterBand(1))
                    #NODATA[np.isnan(NODATA)] = 9999
                    #NODATA[NODATA != 9999] = 9999
                    condnd = np.where((NODATA != nd) & (~np.isnan(NODATA)))
                    NODATA[condnd] = 9999
                    g_FSC_m.GetRasterBand(1).WriteArray(NODATA)
                    g_FSC = gdal.Warp('',[l_g_FSC[proj],g_FSC_m],format= 'MEM')
                    l_g_FSC[proj] = g_FSC
                    a_FSC = None
                    g_FSC = None
                

            nb_results += 1
    
    
        
        if nb_results > 0 :
           print("\nNumber of processed dates:" + str(nb_results))
           return True
        else :
           print("\nERROR products2DataSets : No date processed")
           return False    
                
                
                

     
                
    def calibrateModel(self,out = "",perc_cal = 0.0):




        dataSetDir = os.path.join(self.path_outputs,out)
        path_tifs = os.path.join(dataSetDir,"TIFS")
        if out == "":
            print("\nERROR calibrateModel : directory unspecified ")
            return False
        if not os.path.isdir(path_tifs):
            print("\nERROR calibrateModel : no datasets found for " + out)
            return False

        path_cal = os.path.join(dataSetDir,"CALIBRATION")
         
        NDSIALL = []
        FSCALL = []

        shutil.rmtree(path_cal, ignore_errors=True)
        self.mkdir_p(path_cal)

        f= open(os.path.join(path_cal,out + "_CALIBRATION_SUMMARY.txt"),"w")
        f.write("\nDates :")
        nb_dates = 0
        
            
            
        for d in sorted(os.listdir(path_tifs)):
            date = self.getDateFromStr(d)
            if date == '' : continue
            print(date)
            path_tifs_date = os.path.join(path_tifs,d)
            
            
            epsgs = {}
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                if epsg == '': continue
                if epsg not in epsgs :
                    epsgs[epsg] = []
                    
            tiles = []
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                tile = self.getTileFromStr(tif)
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
        cond1 = np.where((FSCALL != 9999) & (~np.isnan(FSCALL)) & (~np.isinf(FSCALL)))
        NDSIALL = NDSIALL[cond1]
        FSCALL = FSCALL[cond1]
        
        cond2 = np.where( (NDSIALL != 9999) & (~np.isnan(NDSIALL)) & (~np.isinf(NDSIALL)))
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
        plt.ylabel('Training FSC',size = 14)
        plt.xlabel('Training NDSI',size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.hist2d(NDSI_train,FSC_train,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'), norm=LogNorm())
        
        n = np.arange(min(NDSI_train),1.01,0.01)
        
        line = 0.5*np.tanh(a*n+b) +  0.5

        plt.plot(n, line, 'r')#, label='FSC=0.5*tanh(a*NDSI+b)+0.5\na={:.2f} b={:.2f}\nRMSE={:.2f}'.format(a,b,rmse_cal))
        #plt.legend(fontsize=10,loc='upper left')

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
        plt.ylabel('Predicted FSC',size = 14)
        plt.xlabel('Testing FSC',size = 14)
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
        plt.xlabel('FSC pred - test',size = 14)
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
        
        k = 0.95
        j = 1.0

        list_var_FSC_box = [np.var(FSC_pred[np.where((FSC_test>= k) & (FSC_test <= j))])]
        list_var_labels_box = ["[ 0.8\n1 ]"]
        j = j - 0.05
        k = k - 0.05

        while j > minFSC: 
 
            list_var_FSC_box.insert(0,np.var(FSC_pred[np.where((FSC_test >= k) & (FSC_test < j))]))
            list_var_labels_box.insert(0,"[ "+ "{0:.2f}".format(k) +"\n"+ "{0:.2f}".format(j) +" [")
            j = j - 0.05
            k = k - 0.05
            if j == 0.0 or k < 0.0 : break


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





    def evaluateModel(self,out_cal = "",out_val = "",sep_eval = False):
        

        calDataSetDir = os.path.join(self.path_outputs,out_cal)
        path_eval = os.path.join(calDataSetDir,"EVALUATION")
        evalDataSetDir = os.path.join(self.path_outputs,out_val)
        path_tifs = os.path.join(evalDataSetDir,"TIFS")
        path_eval_dir = os.path.join(path_eval,out_val)
        path_params = os.path.join(calDataSetDir,"CALIBRATION","CALIBRATION_PARAMS.txt")
        if out_cal == "":
            print("\nERROR evaluateModel : calibration directory unspecified ")
            return False
        if out_val == "":
            print("\nERROR evaluateModel : evaluation directory unspecified ")
            return False


        self.mkdir_p(path_eval_dir)
        
        a = 0
        b = 0
        with open(path_params, "r") as params :
            line = params.readline()
            line = params.readline()
            ab = line.split()
            a = float(ab[0])
            b = float(ab[1])

        f= open(os.path.join(path_eval_dir,out_cal+"_EVAL_WITH_"+out_val+".txt"),"w")
        f.write("\nCalibration dataset : " + out_cal)
        f.write("\nModel : FSC = 0.5*tanh(a*NDSI+b) +  0.5 with :")
        f.write("\n        a = " + str(a) + " b = " + str(b))
        f.write("\nEvaluation dataSet : " + out_val)
        

        NDSI_test_all = []
        FSC_test_all = []
        list_NDSI_test = []
        list_FSC_test = []
        dates = []
        for d in sorted(os.listdir(path_tifs)):
            date = self.getDateFromStr(d)
            if date == '' : continue
            print(date)
            path_tifs_date = os.path.join(path_tifs,d)
                
                
            epsgs = {}
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                if epsg != "" and epsg not in epsgs :
                    epsgs[epsg] = []
                    
            tiles = []
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                tile = self.getTileFromStr(tif)
                if epsg == '' or tile == '': continue
                if tile not in epsgs[epsg]:
                    epsgs[epsg].append(tile)      
                    
            
            NDSI_test_1date = []
            FSC_test_1date = []
            for epsg in epsgs :
                for tile in epsgs[epsg]:
                    g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    NDSI = BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten()
                    FSC = BandReadAsArray(g_FSC.GetRasterBand(1)).flatten()
                    cond1 = np.where((FSC != 9999) & (~np.isnan(FSC)) & (~np.isinf(FSC)))
                    NDSI = NDSI[cond1]
                    FSC = FSC[cond1]
                    cond2 = np.where( (NDSI != 9999) & (~np.isnan(NDSI)) & (~np.isinf(NDSI)))
                    FSC = FSC[cond2]
                    NDSI = NDSI[cond2]  
                    if len(FSC) > 0 :
                        FSC_test_1date.append(FSC)
                        NDSI_test_1date.append(NDSI)
            if len(FSC_test_1date) > 0 :
                FSC_test_all.append(np.hstack(FSC_test_1date))
                NDSI_test_all.append(np.hstack(NDSI_test_1date))
                dates.append(date.strftime(self.date_format))
                
        
        if len(FSC_test_all) == 0 :
            print("WARNING evaluateModel : no data could be extracted")
            return False


        if sep_eval:
            list_NDSI_test = NDSI_test_all
            list_FSC_test = FSC_test_all
            dates_test = dates
        else :
            list_NDSI_test = [np.hstack(NDSI_test_all)]
            list_FSC_test = [np.hstack(FSC_test_all)]
            dates_test = [dates[0]+"_"+dates[-1]]
            
        f.write("\nNB of dates : " + str(len(dates_test)))
        for i in np.arange(len(list_NDSI_test)):
            NDSI_test = list_NDSI_test[i]
            FSC_test = list_FSC_test[i]


            # VALIDATION
        
            # prediction of FSC from testing NDSI
            FSC_pred =  0.5*np.tanh(a*NDSI_test+b) +  0.5

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
            st = fig.suptitle("EVALUATION OF "+out_cal+" WITH "+out_val,size = 16)
            # set up subplot grid
            gridspec.GridSpec(2,2)
    
  
            # 2D histos de FSC vs NDSI
            ax = plt.subplot2grid((2,2), (0,0))
            plt.title("FSC/NDSI TESTING SET",size = 14,y=1.08)
            plt.ylabel('Testing FSC',size = 14)
            plt.xlabel('Testing NDSI',size = 14)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
            plt.hist2d(NDSI_test,FSC_test,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
            n = np.arange(min(NDSI_test),1.01,0.01)
            line = 0.5*np.tanh(a*n+b) +  0.5
            plt.plot(n, line, 'r')#, label='Predicted FSC')
            #plt.legend(fontsize=10,loc='upper left')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=12)
            ratio = 1
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()
            ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio) 

            # 2D histos de validation
            ax = plt.subplot2grid((2,2), (0,1))
            plt.title("VALIDATION WITH THE TESTING SET",size = 14,y=1.08)
            plt.ylabel('Predicted FSC',size = 14)
            plt.xlabel('Testing FSC',size = 14)
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
            plt.title("FSC RESIDUALS")
            plt.ylabel('Percent of data points',size = 14)
            plt.xlabel('FSC pred - test',size = 14)
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
            fig.savefig(os.path.join(path_eval_dir,out_cal+"_EVAL_WITH_"+out_val+"_"+dates_test[i]+'.png'))
            plt.close(fig)
            
            
            
            minFSC = min(FSC_test)
            
            k = 0.8
            j = 1.0
    
            list_var_FSC_box = [np.var(FSC_pred[np.where((FSC_test>= k) & (FSC_test <= j))])]
            list_var_labels_box = ["[ 0.8\n1 ]"]
            j = j - 0.2
            k = k - 0.2
    
            while j > minFSC: 
     
                list_var_FSC_box.insert(0,np.var(FSC_pred[np.where((FSC_test >= k) & (FSC_test < j))]))
                list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(k) +"\n"+ "{0:.1f}".format(j) +" [")
                j = j - 0.2
                k = k - 0.2
                if j == 0.0 or k < 0.0 : break
    
    
            # Plot figure with subplots 
            fig = plt.figure()
            st = fig.suptitle("FSC RESIDUALS",size = 16)
            gridspec.GridSpec(1,2)
            
            
            # boxplot avec FSC = 0 et FSC = 1
            ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
            #plt.title('ODK FSC/NDSI',size = 14,y=1.08)
            plt.ylabel('FSC TOC residal variance',size = 14)
            plt.xlabel('FSC TOC test',size = 14)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
            plt.bar(list_var_labels_box,list_var_FSC_box)
    
    
            # fit subplots and save fig
            fig.tight_layout()
            fig.set_size_inches(w=16,h=10)
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)
            fig.savefig(os.path.join(path_eval_dir,out_cal+"_EVAL_WITH_"+out_val+"_RESIDUAL_VAR_"+dates_test[i]+'.png'))
            plt.close(fig)
            
            


            
            f.write("\n\n")
        
            f.write("\nEVALUATION WITH " + dates_test[i] )
            f.write("\n  Number of 20x20m data points : " + str(len(NDSI_test)))
            f.write("\n  corr. coef. : " + str(corr_FSC))
            f.write("\n  std. err. : " + str(stde_FSC))
            f.write("\n  mean err. : " + str(m_er_FSC))
            f.write("\n  abs. mean err. : " + str(abs_m_er_FSC))
            f.write("\n  root mean square err. : " + str(rmse_FSC))

        f.close()


        return True


    
    
    

    def plotPeriod(self,out=""):
        
        dataSetDir = os.path.join(self.path_outputs,out)
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
        self.mkdir_p(path_plots_date)
        
        f= open(os.path.join(path_plots_date,"INFO.txt"),"w")
        f.write("\nDates :")
        nb_dates = 0
        
       
            
        for d in sorted_dates:
            date = self.getDateFromStr(d)
            if date == '' : continue
            print(date)
            path_tifs_date = os.path.join(path_tifs,d)
            
            
            epsgs = {}
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                if epsg == '': continue
                if epsg not in epsgs :
                    epsgs[epsg] = []
                    
            tiles = []
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                tile = self.getTileFromStr(tif)
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
        cond1 = np.where((FSCALL != 9999) & (~np.isnan(FSCALL)) & (~np.isinf(FSCALL)))
        NDSIALL = NDSIALL[cond1]
        FSCALL = FSCALL[cond1]
        
        cond2 = np.where( (NDSIALL != 9999) & (~np.isnan(NDSIALL)) & (~np.isinf(NDSIALL)))
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
        list_FSC_box = [FSCALL[np.where((NDSIALL >= 0.8) & (NDSIALL <= 1))]]
        list_labels_box = ["[ 0.8\n1 ]"]
        b = 0.8
        while minNDSI < b : 
            a = b - 0.2
            list_FSC_box.insert(0,FSCALL[np.where((NDSIALL >= a) & (NDSIALL < b))])
            list_labels_box.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
            b = b - 0.2
            

        minNDSI2 = min(NDSIALL2)
        list_FSC_box2 = [FSCALL2[np.where((NDSIALL2 >= 0.8) & (NDSIALL2 <= 1))]]
        list_labels_box2 = ["[ 0.8\n1 ]"]
        b = 0.8
        while minNDSI2 < b : 
            a = b - 0.2
            list_FSC_box2.insert(0,FSCALL2[np.where((NDSIALL2 >= a) & (NDSIALL2 < b))])
            list_labels_box2.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
            b = b - 0.2      

        
        
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


    def plotEachDates(self,out=""):

        print("Start plotting each date")
        dataSetDir = os.path.join(self.path_outputs,out)
        path_tifs = os.path.join(dataSetDir,"TIFS")
        path_plots = os.path.join(dataSetDir,"PLOTS")
            
            
        for d in sorted(os.listdir(path_tifs)):
            date = self.getDateFromStr(d)
            if date == '' : continue
            print(date)
            path_tifs_date = os.path.join(path_tifs,d)
            path_plots_date = os.path.join(path_plots,d)
            self.mkdir_p(path_plots_date)
            FSC = []
            NDSI = []  
            
            epsgs = {}
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                if epsg == '': continue
                if epsg not in epsgs :
                    epsgs[epsg] = []
                    
            tiles = []
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                tile = self.getTileFromStr(tif)
                if epsg == '' or tile == '': continue
                if tile not in epsgs[epsg]:
                    epsgs[epsg].append(tile)  
                    
                    
            
            for epsg in epsgs :
                for tile in epsgs[epsg]:
                    g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    FSC.append(BandReadAsArray(g_FSC.GetRasterBand(1)).flatten())
                    g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    NDSI.append(BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten())
                    
                    


        
            print("Eliminate Nodata pixels")
            NDSI = np.hstack(NDSI)
            FSC = np.hstack(FSC)  
            cond1 = np.where((FSC != 9999) & (~np.isnan(FSC)) & (~np.isinf(FSC)))
            NDSI = NDSI[cond1]
            FSC = FSC[cond1]
            
            cond2 = np.where( (NDSI != 9999) & (~np.isnan(NDSI)) & (~np.isinf(NDSI)))
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
            list_FSC_box = [FSC[np.where((NDSI >= 0.8) & (NDSI <= 1))]]
            list_labels_box = ["[ 0.8\n1 ]"]
            b = 0.8
            while minNDSI < b : 
                a = b - 0.2
                list_FSC_box.insert(0,FSC[np.where((NDSI >= a) & (NDSI < b))])
                list_labels_box.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
                b = b - 0.2
                

            minNDSI2 = min(NDSI2)
            list_FSC_box2 = [FSC2[np.where((NDSI2 >= 0.8) & (NDSI2 <= 1))]]
            list_labels_box2 = ["[ 0.8\n1 ]"]
            b = 0.8
            while minNDSI2 < b : 
                a = b - 0.2
                list_FSC_box2.insert(0,FSC2[np.where((NDSI2 >= a) & (NDSI2 < b))])
                list_labels_box2.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
                b = b - 0.2         
            
            

            
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
            fig.savefig(os.path.join(path_plots_date,'PLOT_FSC_NDSI_'  + out + '_' + date.strftime(self.date_format) + '.png'))
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
    




    def createQuickLooks(self,out = ""):
        
        
        p_cmp = os.path.join(self.path_palettes,"palette_cmp.txt")
        p_fsc = os.path.join(self.path_palettes,"palette_FSC.txt")
        
        dataSetDir = os.path.join(self.path_outputs,out)
        path_tifs = os.path.join(dataSetDir,"TIFS")
        path_qckls = os.path.join(dataSetDir,"QUICKLOOKS")

        nb_dates = 0
        for date in sorted(os.listdir(path_tifs)):
            print(date)
            path_tifs_date = os.path.join(path_tifs,date)
            path_qckls_date = os.path.join(path_qckls,date)
            self.mkdir_p(path_qckls_date)
            
            #we get a list of tiles for each epsg
            epsgs = {}
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                if epsg == '': continue
                if epsg not in epsgs :
                    epsgs[epsg] = []
                    
            tiles = []
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                tile = self.getTileFromStr(tif)
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
                    
                    #create compo quicklook
                    f_COMPO = os.path.join(path_tifs_date,"INPUT_COMPO_"+tile+"_EPSG-"+epsg+".tif")
                    gdal.Translate(os.path.join(path_qckls_date,"INPUT_COMPO_"+tile+"_EPSG-"+epsg+".png"),f_COMPO,format= 'PNG', width=800,outputType = gdal.GDT_Byte)
                    

                    
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
                    
                    
                    
                    #create snow difference quicklook
                    
                    
                    g_FSC = gdal.Open(f_FSC_r)
                    g_SEB = gdal.Open(os.path.join(path_tifs_date,"INPUT_SEB_"+tile+"_EPSG-"+epsg+".tif"))
                    
                    minx, maxy, maxx, miny = self.getOverlapCoords(g_FSC,g_SEB)
                    g_FSC = gdal.Translate('',g_FSC,format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
                    g_SEB = gdal.Translate('',g_SEB,format= 'MEM',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32) 
                    g_CMP = g_FSC
                    #valeurs dans FSC : [0-1] pour la neige (et non-neige) , 9999 pour noData
                    #valeurs dans SEB : 100 pour la neige, 0 pour non-neige, 205 pour nuage, 254 pour nodata         

                    SEB = BandReadAsArray(g_SEB.GetRasterBand(1))  
                    cond = np.where((SEB != 100) & (SEB != 0))
                    SEB[cond] = np.nan
                    cond = np.where(SEB == 100)
                    SEB[cond] = 1


                    
                    #valeurs dans FSC : [0-1] pour la neige (et non-neige) , 9999 pour noData
                    #valeurs dans SEB : 1 pour la neige, 0 pour non neige, nan pour noData   
                    

                    FSC = BandReadAsArray(g_CMP.GetRasterBand(1))

                    cond = np.where((FSC > 0) & (FSC <= 1))
                    FSC[cond] = 2
                    FSC[FSC == 9999] = np.nan

                    #valeurs dans FSC : 2 pour la neige, 0 pour non neige, nan pour nodata
                    #valeurs dans SEB : 1 pour la neige, 0 pour non neige, nan pour noData
                    
                    CMP = (SEB + FSC)
                    
                    #cond = np.where((CMP != 1) & (CMP != 2))

                    #CMP[cond] = np.nan
                    g_CMP.GetRasterBand(1).WriteArray(CMP)
                    gdal.Translate(os.path.join(path_tifs_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"),g_CMP,format= 'GTiff',noData = 9999)
                    os.system("gdaldem color-relief " + os.path.join(path_tifs_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif") + " " + p_cmp + " " + os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    gdal.Translate(os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".png"),os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"),format= 'PNG', width=800,outputType = gdal.GDT_Byte)
                    os.remove(os.path.join(path_qckls_date,"SNOW_DIFF_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    
                    
                    #gdal.Translate(os.path.join(path_results_date,"S2-FSC_" + tile +"_" + epsg +".png"),g_CMP,format= 'PNG', width=1000,outputType = gdal.GDT_Byte, scaleParams=[[0,2,1,255]],noData = "0")
                    
                    
                    

        
        return True





        
        
    def timeLapseEval(self,out_cal = "",out_val = ""):
        

        calDataSetDir = os.path.join(self.path_outputs,out_cal)
        path_eval = os.path.join(calDataSetDir,"EVALUATION")
        
        

        path_params = os.path.join(calDataSetDir,"CALIBRATION","CALIBRATION_PARAMS.txt")
        a = 0
        b = 0
        with open(path_params, "r") as params :
            line = params.readline()
            line = params.readline()
            ab = line.split()
            a = ab[0]
            b = ab[1]


        path_eval_dir = os.path.join(path_eval,out_val)
        

        self.mkdir_p(path_eval_dir)

        f= open(os.path.join(path_eval_dir,"TIMELAPSE_"+out_cal+"_EVAL_WITH_"+out_val + ".txt"),"w")
        f.write("\nCalibration dataset :" + out_cal)
        f.write("\nModel : FSC = 0.5*tanh(a*NDSI+b) +  0.5 with :")
        f.write("\n        a = " + str(a) + " b = " + str(b))
        f.write("\nEvaluation dataset :" + out_val)
        

        # Plot figure with subplots 
        fig = plt.figure()
        st = fig.suptitle(title)
        # set up subplot grid
        gridspec.GridSpec(1,2)
        # prepare for evaluation scatterplot
        ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
        
        plt.ylabel('FSC predicted - evaluation',size = 14)
        plt.xlabel('Dates',size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        



        nb_pixel_total = 0

        evalDataSetDir = os.path.join(self.path_outputs,out_val)
        path_tifs = os.path.join(evalDataSetDir,"TIFS")

        NDSI_avg_test = []
        FSC_avg_test = []
        days = []
        
        for d in sorted(os.listdir(path_tifs)):
            date = self.getDateFromStr(d)
            if date == '' : continue
            print(date)
            path_tifs_date = os.path.join(path_tifs,d)
                
                
            epsgs = {}
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                if epsg == '': continue
                if epsg not in epsgs :
                    epsgs[epsg] = []
                
            tiles = []
            for tif in os.listdir(path_tifs_date) :
                epsg = self.getEpsgFromStr(tif)
                tile = self.getTileFromStr(tif)
                if epsg == '' or tile == '': continue
                if tile not in epsgs[epsg]:
                    epsgs[epsg].append(tile)      
                    
            FSC_d = []
            NDSI_d = []
            
            for epsg in epsgs :
                for tile in epsgs[epsg]:
                    g_FSC = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    FSC_d.append(BandReadAsArray(g_FSC.GetRasterBand(1)).flatten())
                    g_NDSI = gdal.Open(os.path.join(path_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsg + ".tif"))
                    NDSI_d.append(BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten())
                    
                
        
        
            NDSI_d = np.hstack(NDSI_d)
            FSC_d = np.hstack(FSC_d)  
            cond1 = np.where((FSC_d != 9999) & (~np.isnan(FSC_d)) & (~np.isinf(FSC_d)))
            NDSI_d = NDSI_d[cond1]
            FSC_d = FSC_d[cond1]
            cond2 = np.where( (NDSI_d != 9999) & (~np.isnan(NDSI_d)) & (~np.isinf(NDSI_d)))
            FSC_d = FSC_d[cond2]
            NDSI_d = NDSI_d[cond2]
            if len(FSC_d) == 0 : continue
               
            nb_pixel_total = nb_pixel_total + len(FSC_d)
            FSC_avg = np.average(FSC_d)
            NDSI_avg = np.average(NDSI_d)
    

            NDSI_avg_test.append(NDSI_avg)
            FSC_avg_test.append(FSC_avg)
            days.append(date)

        NDSI_avg_test = np.hstack(NDSI_avg_test)
        FSC_avg_test = np.hstack(FSC_avg_test)
        days = np.hstack(days)
               
        



        # VALIDATION
    
        # prediction of FSC from testing NDSI
        FSC_avg_pred = 0.5*np.tanh(a*NDSI_avg_test+b) +  0.5

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




        plt.scatter(days,er_FSC, label='{:s}; rmse : {:.2f}'.format(out_val,rmse_FSC))
        plt.legend(fontsize=10,loc='upper left')




        f.write("\n")  
        f.write("\n  Number of datess : " + str(len(NDSI_avg_test)))
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
        fig.savefig(os.path.join(path_eval_dir,"PLOT_TIMELAPSE_"+out_cal+"_EVAL_WITH_"+out_val+".png"))
        plt.close(fig)

        #close txt file
        f.close()


        return True








    def processODK(self,out = "",source = "",filter_acc = True,filter_snw = True,max_acc = 5):



        dataSetDir = os.path.join(self.path_outputs,out)
        odkList = os.path.join(dataSetDir,"ODKLIST.txt")
        odkInfos = os.path.join(dataSetDir,"ODKINFOS.txt")


        dirODK = os.path.join(self.path_inputs,"FSC",source)
        shutil.rmtree(dataSetDir, ignore_errors=True)
        self.mkdir_p(dataSetDir)

        dict_tile = {}
        dict_products = {}
        dict_FSC = {}


        print("####################################################")
        print("Recuperation of ODK data")
        #on recupere les donnees odk
        for f in os.listdir(dirODK) :    
            with open(os.path.join(dirODK,f), "r") as ODK :
                line = ODK.readline()
                line = ODK.readline()
                while line :
                    point = line.split()
                    date = point[0]
                    latitude = point[1]
                    longitude = point[2]
                    accuracy = point[3]
                    fsc = point[4]
                    if filter_acc == True and float(accuracy) > max_acc  : 
                        line = ODK.readline()
                        continue
                    
                    if date not in dict_FSC.keys() :
                        dict_FSC[date] = []
                        dict_FSC[date].append([latitude,longitude,fsc,accuracy])
                    else :
                        dict_FSC[date].append([latitude,longitude,fsc,accuracy])
                    
                    line = ODK.readline()
                    


        print("####################################################")
        print("Search of tiles and L2A rasters")
        #on trouve les tuiles et rasters correspondants
        for date in dict_FSC :
            print("check date: ",date)
            list_points = dict_FSC[date]
            dateFSC = self.getDateFromStr(date)


            for point in list_points :
                
                lat = point[0]
                lon = point[1]
                fsc = point[2]
                acc = point[3]
                print(" check point : ",lat,lon)


            
                decalP = self.nb_shift_days + 1    
                tileP = ""
                p_L2AP = ""   

                for tile in os.listdir(self.path_LIS) :
                    #print("check tile : ",tile)
                    path_tile = os.path.join(self.path_LIS,tile)
                    if not os.path.isdir(path_tile): continue
                    try:
                        L2A_product = os.listdir(path_tile)[-1]
                    except OSError as exc:  # Python >2.5
                        if exc.errno == errno.EACCES:
                            continue
                        else:
                            raise   
                    L2A_product = os.path.join(path_tile,L2A_product)
                    f_L2A = os.path.join(L2A_product,"LIS_PRODUCTS","LIS_SEB.TIF")
                

                    pixel = os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_L2A, lon, lat)).read()

                    try:
                        int(pixel)
                        
                    except ValueError:
                        continue

                    L2A_products = glob.glob(os.path.join(path_tile,'*SENTINEL*'))



                    for L2A_product in L2A_products :
                        dateL2A = self.getDateFromStr(L2A_product)
                        decal = dateL2A - dateFSC
                        if abs(decal.days) >= decalP : continue

                        f_L2A = os.path.join(L2A_product,"LIS_PRODUCTS","LIS_SEB.TIF")
                        pixel = int(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_L2A, lon, lat)).read())

                        if  filter_snw == True and pixel != 100: continue

                        decalP = abs(decal.days)
                        p_L2AP = L2A_product
                        tileP = tile


                if p_L2AP == "":
                    print("  point rejete")
                    continue
                else :
                    print("  point accepte")

                f_L2AP = os.path.basename(p_L2AP)

                #we check if pixel is in tree region
                forest = int(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (self.path_tree, lon, lat)).read())


                if dateFSC not in dict_products.keys() :
                    dict_products[dateFSC] = []
                    dict_products[dateFSC].append([lat,lon,fsc,acc,decalP,p_L2AP,f_L2AP,tileP,forest])
                else :
                    dict_products[dateFSC].append([lat,lon,fsc,acc,decalP,p_L2AP,f_L2AP,tileP,forest])
                    
              
        f_odkList= open(odkList,"w")
        f_odkInfos= open(odkInfos,"w")
        #on affiche le dict
        print("####################################################")
        print("\n")
        nb_points = 0
        f_odkList.write("date lat lon fsc acc decal L2A tile forest")
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
                print("lat = ",point[0],"lon = ",point[1],"fsc = ",point[2],"acc = ",point[3],"decal = ",point[4],"forest value = ",point[8])
                f_odkList.write("\n"+date.strftime(self.date_format)+" "+str(point[0])+" "+str(point[1])+" "+str(point[2])+" "+str(point[3])+" "+str(point[4])+" "+point[5]+" "+point[7]+" "+str(point[8]))
                nb_points = nb_points + 1
            print("\n")
        print("nb of points = ",nb_points)

        #on affiche le nombre de points par tuile
        for tile in dict_tile :
            line = "TILE : " + tile + " ; NB of points : " + str(dict_tile[tile][0]) + " ; NB of points in forest : " + str(dict_tile[tile][1])
            print(line)
            f_odkInfos.write("\n" + line)

        f_odkInfos.write("\nTOTAL NB OF POINTS : " + str(nb_points))

        f_odkList.close()
        f_odkInfos.close()

        return True







    def evalWithODK(self,out_cal = "",out_val = ""):


        calDataSetDir = os.path.join(self.path_outputs,out_cal)
        path_eval = os.path.join(calDataSetDir,"EVALUATION")
        title = "EVAL_" + out_cal + "_WITH_ODK"
        
        odkPoints = os.path.join(self.path_outputs,out_val,"ODKLIST.txt")


        path_eval_dir = os.path.join(path_eval,title)
        shutil.rmtree(path_eval_dir, ignore_errors=True)

        self.mkdir_p(path_eval_dir)



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
        print("Recuperation of ODK data")
        #on recupere les donnees odk
        with open(odkPoints, "r") as ODK :
            line = ODK.readline()
            line = ODK.readline()
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
                    
                line = ODK.readline()
                    


       

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

            
                # We look for the red, green and swir bands tiff files + mask
                f_green = ""
                f_swir = ""
                f_red = ""
                f_mask = ""


                for fp in os.listdir(L2A_product) :
                    if ("green_band_resampled.tif" in fp) :
                        f_green = os.path.join(L2A_product,fp)
                    elif ("red_band_resampled.tif" in fp) :
                        f_red = os.path.join(L2A_product,fp)
                    elif ("swir_band_extracted.tif" in fp) :
                        f_swir = os.path.join(L2A_product,fp)



                #If there is a file missing, we skip to the next point
                if f_green == "" or f_red == "" or f_swir == "" : continue
                

                #We get the corresponding pixel from each band to calculate a NDSI pixel
                green = 0
                red = 0
                swir = 0
                NDSI = 0
                

                
                try:
                    
                    green = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_green, lon, lat)).read())
                    red = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_red, lon, lat)).read())
                    swir = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_swir, lon, lat)).read())
                    
                except ValueError:
                    continue


                    
                    
                    
                NDSI = (green - swir)/(green + swir)


                
                if np.isnan(NDSI) or np.isinf(NDSI) : continue
                
                list_NDSI.append(NDSI)
                list_FSC.append(fsc)
                list_TCD.append(tcd)

            

        #on affiche les lists
        print("####################################################")
        print("\nODK POINTS:")

        for i in arange(len(list_NDSI)) :
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
            i = j - 0.2
            list_FSC_box.insert(0,FSC_test[np.where((NDSI_test >= i) & (NDSI_test < j))])
            list_labels_box.insert(0,"[ "+ "{0:.1f}".format(i) +"\n"+ "{0:.1f}".format(j) +" [")
            j = j - 0.2
            

           




        # Plot figure with subplots 
        fig = plt.figure()
        st = fig.suptitle("ODK FSC / NDSI",size = 16)
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
        fig.savefig(os.path.join(path_eval_dir,'ODK_ANALYSIS.png'))
        plt.close(fig)






        # Plot figure with subplots 
        fig = plt.figure()
        st = fig.suptitle("ODK EVALUATION",size = 16)
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
        
        i = 0.8
        j = 1.0

        list_var_FSC_box = [np.var(TOC_er_FSC[np.where((FSC_test>= i) & (FSC_test <= j))])]
        list_var_labels_box = ["[ 0.8\n1 ]"]
        j = j - 0.2
        i = i - 0.2

        while j > minFSC: 
 
            list_var_FSC_box.insert(0,np.var(TOC_er_FSC[np.where((FSC_test >= i) & (FSC_test < j))]))
            list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(i) +"\n"+ "{0:.1f}".format(j) +" [")
            j = j - 0.2
            i = i - 0.2
            if j == 0.0 or i < 0.0 : break


        # Plot figure with subplots 
        fig = plt.figure()
        st = fig.suptitle("FSC RESIDUALS",size = 16)
        gridspec.GridSpec(1,2)
        
        
        # boxplot avec FSC = 0 et FSC = 1
        ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
        #plt.title('ODK FSC/NDSI',size = 14,y=1.08)
        plt.ylabel('FSC TOC residues variance',size = 14)
        plt.xlabel('FSC TOC test',size = 14)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.bar(list_var_labels_box,list_var_FSC_box)


        # fit subplots and save fig
        fig.tight_layout()
        fig.set_size_inches(w=16,h=10)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(path_eval_dir,'ODK_RESIDUES_ANALYSIS.png'))
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


        return TOC_rmse_FSC
