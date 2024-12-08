
import sys
import os
import errno
import re
import copy
from datetime import datetime, timedelta, date
import glob
import sc_utils
import argparse
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
import csv









def selectProducts(list_selected_lis_path,nb_shift_days,selection,list_overlapping_tiles_path,FSC_path,dateFSC,epsg):


    #List of selected LIS products
    selected_LIS_products = []
    #extract tiles
    tiles = []
    with open(list_overlapping_tiles_path,'r') as list_overlapping_tiles_file:
        reader = csv.DictReader(list_overlapping_tiles_file)
        for row in reader:
            tile = row["tile"]
            path = row["path"]
            tiles.append[tile,path]

    for tile, tile_path in tiles :
            
        print("Check tile : " + tile)
        
        if os.path.isdir(tile_path):
            list_LIS_products = glob.glob(os.path.join(tile_path,'**' + "/FSC_*"), recursive=True)
        if list_LIS_products == [] :
            print ("No LIS product found for tile " + tile + " in directory " + tile_path)
                    
        g_tile = gdal.Open(glob.glob(os.path.join(list_LIS_products[0],"*NDSI*"))[0])
        
    
        
        g_FSC = gdal.Open(FSC_path)

        if epsg != "" :
            g_FSC = gdal.Warp('',g_FSC,format= 'MEM',srcSRS="EPSG:" + epsg )                

        
        minx, maxy, maxx, miny = sc_utils.getOverlapCoords(g_FSC,g_tile)  
        if minx == None or maxy == None or maxx == None or miny == None : continue
        
        
        epsgLIS = ""
        LIS = ""
        ind = nb_shift_days + 1
        NDR1 = 100
        NDR2 = 100
        for LIS_product in list_LIS_products:
            dateLIS = sc_utils.getDateFromStr(LIS_product)
            lag = dateLIS - dateFSC
            if abs(lag.days) >  nb_shift_days : continue
            
            g_LIS = gdal.Translate('',glob.glob(os.path.join(LIS_product,"*NDSI*"))[0],format= 'MEM',projWin = [minx, maxy, maxx, miny]) 
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
            else : 
                print("date rejetee")
                
                
        if LIS == "" : continue

        print("Chosen LIS : " + LIS)
        selected_LIS_products.append(LIS)


    with open(list_selected_lis_path,'a') as list_selected_lis_file:
        writer = csv.writer(list_selected_lis_file)
        out = [FSC_path]
        for i in selected_LIS_products :
            out.append(i)
        writer.writerow(out)
    







def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-list_selected_lis_path', action='store', default="", dest='list_selected_lis_path')
    parser.add_argument('-nb_shift_days', action='store',default=0,type=int, dest='nb_shift_days')
    parser.add_argument('-selection', action='store',default="", dest='selection')
    parser.add_argument('-list_overlapping_tiles_path', action='store', default="", dest='list_overlapping_tiles_path')
    parser.add_argument('-FSC_path', action='store', default="", dest='FSC_path')
    parser.add_argument('-date', action='store', default="", dest='date')
    parser.add_argument('-epsg', action='store', default="", dest='epsg')

    list_selected_lis_path = parser.parse_args().list_selected_lis_path
    nb_shift_days = parser.parse_args().nb_shift_days
    selection = parser.parse_args().selection
    list_overlapping_tiles_path = parser.parse_args().list_overlapping_tiles_path
    FSC_path = parser.parse_args().FSC_path
    date = parser.parse_args().date
    epsg = parser.parse_args().epsg


    selectProducts(list_selected_lis_path,nb_shift_days,selection,list_overlapping_tiles_path,FSC_path,date,epsg)
    


if __name__ == '__main__':
    main()