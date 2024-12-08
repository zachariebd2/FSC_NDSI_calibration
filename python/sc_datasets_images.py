
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









def make_datasets(FSC_path,LIS_products_paths,output_tifs_path,nodt,nosnw,snw,resample,epsgFSC):



    
    epsg = ""
    dateFSC = sc_utils.getDateFromStr(FSC_path)
    nd = 100000
    dir_tifs_date = os.path.join(output_tifs_path,dateFSC.strftime("%Y-%m-%d"))
    sc_utils.mkdir_p(dir_tifs_date)

    dict_datasets = {}
    dict_datasets["FSC"] = {}
    dict_datasets["FSC"]["original"]  = []
    dict_datasets["FSC"]["resampled"]  = []
    dict_datasets["FSC"]["overlap"]  = []
    dict_datasets["FSC"]["output"]  = []
    dict_datasets["LIS"] = {}
    dict_datasets["LIS"]["original"]  = []
    dict_datasets["LIS"]["overlap"]  = []
    dict_datasets["LIS"]["output"]  = []



    f_FSC = FSC_path
    l_LIS = LIS_products_paths.split()

    
    
    # we get the FSC projection system
   
    if epsgFSC == "" :
        epsg = (gdal.Info(f_FSC, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    else :
        epsg = epsgFSC

    

    
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
    dict_datasets["FSC"]["original"] = [epsg,os.path.join(dir_tifs_date,"INPUT_FSC.tif")]

    
    print("\nTraitement des tuiles")
    
   

    l_g_FSC = {}
    

    LIS_products = []
    # On prepare un FSC reprojete pour chaque projection
    for LIS_product in  l_LIS: 
        f_S2 = glob.glob(os.path.join(LIS_product,'*NDSI.tif'))[0]
        epsgS2 = (gdal.Info(f_S2, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
        tile = sc_utils.getTileFromStr(f_S2)
        LIS_products.append(tile,epsgS2,LIS_product)
        
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
            dict_datasets["FSC"]["resampled"].append([epsgS2,os.path.join(dir_tifs_date,"RESAMPLED_FSC_EPSG-"+epsgS2+".tif")])
            
    
    print(l_g_FSC)

    
    for tile , epsgS2 , LIS_product in  sorted(LIS_products,key=lambda l:l[1]) : 
        
        print(tile,epsgS2,LIS_product)
        f_TOC = glob.glob(os.path.join(LIS_product,'*FSCTOC.tif'))[0]
        f_OG = glob.glob(os.path.join(LIS_product,'*FSCOG.tif'))[0]
        f_QCTOC = glob.glob(os.path.join(LIS_product,'*QCTOC.tif'))[0]
        f_QCOG = glob.glob(os.path.join(LIS_product,'*QCOG.tif'))[0]
        f_QC = glob.glob(os.path.join(LIS_product,'*QCFLAGS.tif'))[0]
        f_NDSI = glob.glob(os.path.join(LIS_product,'*NDSI.tif'))[0]
        dict_datasets["LIS"]["original"].append([epsgS2,tile,f_TOC,f_OG,f_NDSI])

        print(f_TOC)

        #If there is a file missing, we skip to the next tile
        if f_TOC == "" or f_OG == "" or f_QCTOC == "" or f_QCOG  == "" or f_QC  == "" or f_NDSI == "" :  continue
        
        
        # On calcul les coord de overlap dans la projection LIS
        print("\nCalcul coordonnees de chevauchement")
        g_TOC = gdal.Open(f_TOC)
        print(l_g_FSC[epsgS2])
        print(sc_utils.isOverlapping(l_g_FSC[epsgS2],g_TOC))
        minx, maxy, maxx, miny = sc_utils.getOverlapCoords(l_g_FSC[epsgS2],g_TOC)
        print (minx)
        print (maxy)
        print (maxx)
        print (miny)
        if minx == None and maxy == None: continue
        
        #on decoupe les fichiers LIS
        print("\nDecoupage du FSCTOC")
        f_TOC_d = os.path.join(dir_tifs_date,"INPUT_FSCTOC_" + tile + "_EPSG-" + epsgS2 + ".tif")
        g_TOC = gdal.Translate(f_TOC_d,g_TOC,format= 'GTiff',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32)
        print("\nDecoupage du FSCOG")
        f_OG_d = os.path.join(dir_tifs_date,"INPUT_FSCOG_" + tile + "_EPSG-" + epsgS2 + ".tif")
        g_OG = gdal.Translate(f_OG_d,f_OG,format= 'GTiff',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32)
        print("\nDecoupage du QCTOC")
        f_QCTOC_d = os.path.join(dir_tifs_date,"INPUT_QCTOC_" + tile + "_EPSG-" + epsgS2 + ".tif")
        g_QCTOC = gdal.Translate(f_QCTOC_d,f_QCTOC,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
        print("\nDecoupage du QCOG")
        f_QCOG_d = os.path.join(dir_tifs_date,"INPUT_QCOG_" + tile + "_EPSG-" + epsgS2 + ".tif")
        g_QCOG = gdal.Translate(f_QCOG_d,f_QCOG,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
        print("\nDecoupage du QC")
        f_QC_d = os.path.join(dir_tifs_date,"INPUT_QC_" + tile + "_EPSG-" + epsgS2 + ".tif")
        gdal.Translate(f_QC_d,f_QC,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
        print("\nDecoupage du NDSI")
        f_NDSI_d = os.path.join(dir_tifs_date,"INPUT_NDSI_" + tile + "_EPSG-" + epsgS2 + ".tif")
        g_NDSI = gdal.Translate(f_NDSI_d,f_NDSI,format= 'GTiff',projWin = [minx, maxy, maxx, miny],outputType = gdal.GDT_Float32)
               
        #on decoupe une copie de FSC
        f_FSC_d = os.path.join(dir_tifs_date,"INPUT_RESAMPLED_FSC_" + tile + "_EPSG-" + epsgS2 + ".tif")
        g_FSC = gdal.Translate(f_FSC_d,l_g_FSC[epsgS2],format= 'GTiff',projWin = [minx, maxy, maxx, miny]) 
        dict_datasets["FSC"]["overlap"].append([epsgS2,tile,f_FSC_d])
        dict_datasets["LIS"]["overlap"].append([epsgS2,tile,f_TOC_d,f_OG_d,f_NDSI_d])

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
        


        f_TOC_f = os.path.join(dir_tifs_date,"OUTPUT_FSCTOC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif")
        f_OG_f = os.path.join(dir_tifs_date,"OUTPUT_FSCOG_tile-" + tile + "_EPSG-" + epsgS2 + ".tif")
        f_QCTOC_f = os.path.join(dir_tifs_date,"OUTPUT_QCTOC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif")
        f_QCOG_f = os.path.join(dir_tifs_date,"OUTPUT_QCOG_tile-" + tile + "_EPSG-" + epsgS2 + ".tif")
        f_NDSI_f = os.path.join(dir_tifs_date,"OUTPUT_NDSI_tile-" + tile + "_EPSG-" + epsgS2 + ".tif")
        f_FSC_f = os.path.join(dir_tifs_date,"OUTPUT_FSC_tile-" + tile + "_EPSG-" + epsgS2 + ".tif")

        gdal.Translate(f_FSC_f,g_FSC,format= 'GTiff',noData = 255)
        gdal.Translate(f_NDSI_f,g_NDSI,format= 'GTiff',noData = 255)
        gdal.Translate(f_TOC_f,g_TOC,format= 'GTiff',noData = 255)
        gdal.Translate(f_OG_f,g_OG,format= 'GTiff',noData = 255)
        gdal.Translate(f_QCTOC_f ,g_QCTOC,format= 'GTiff',noData = 255)
        gdal.Translate(f_QCOG_f,g_QCOG,format= 'GTiff',noData = 255)   
        dict_datasets["FSC"]["output"].append([epsgS2,tile,f_FSC_f])
        dict_datasets["LIS"]["output"].append([epsgS2,tile,f_TOC_f,f_OG_f,f_NDSI_f])
        
        for proj in l_g_FSC :
            g_FSC_m = gdal.Warp('',g_FSC,format= 'MEM',dstSRS="EPSG:" + proj,xRes= 20,yRes= 20)

            MASK = BandReadAsArray(g_FSC_m.GetRasterBand(1))

            MASK[MASK != 255 ] = 255
            
            g_FSC_m.GetRasterBand(1).WriteArray(MASK)
            
            
            l_g_FSC[proj] = gdal.Warp('',[l_g_FSC[proj],g_FSC_m],format= 'MEM')

        

    dict_datasets_path = os.path.join(dir_tifs_date,"dict_datasets.json")
    dict_datasets_file = open(dict_datasets_path, "w+")
    dict_datasets_file.write(json.dumps(dict_datasets, indent=3))
    dict_datasets_file.close()

    

       





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-FSC_path', action='store', default="", dest='FSC_path')
    parser.add_argument('-LIS_products_paths', action='store', default="", dest='LIS_products_paths')
    parser.add_argument('-output_tifs_path', action='store',default="",type=int, dest='output_tifs_path')
    parser.add_argument('-nodt', action='store', default=0,type=int, dest='nodt')
    parser.add_argument('-nosnw', action='store', default=0,type=int, dest='nosnw')
    parser.add_argument('-snw', action='store', default=0,type=int, dest='snw')
    parser.add_argument('-resample', action='store', default="", dest='resample')
    parser.add_argument('-epsg', action='store', default="", dest='epsg')

    FSC_path = parser.parse_args().FSC_path
    LIS_products_paths = parser.parse_args().LIS_products_paths
    output_tifs_path = parser.parse_args().output_tifs_path
    nodt = parser.parse_args().nodt
    nosnw = parser.parse_args().nosnw
    snw = parser.parse_args().snw
    resample = parser.parse_args().resample
    epsg = parser.parse_args().epsg


    make_datasets(FSC_path,LIS_products_paths,output_tifs_path,nodt,nosnw,snw,resample,epsg)
    


if __name__ == '__main__':
    main()