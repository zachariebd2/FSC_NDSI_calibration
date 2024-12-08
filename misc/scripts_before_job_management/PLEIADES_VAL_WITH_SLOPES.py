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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sc_utils import *


path_PLEIADES = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/PLEIADES/"
path_products = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/PLEIADES/TIFS/"
path_slope = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/T31TCH_SLOPE/SRTM_31TCH_20m_slope_co.tif"
path_res = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/slope_res"

path_params = os.path.join(path_PLEIADES,"CALIBRATION","CALIBRATION_PARAMS.txt")

a = 0
b = 0
with open(path_params, "r") as params :
    line = params.readline()
    line = params.readline()
    ab = line.split()
    a = float(ab[0])
    b = float(ab[1])
        
        
        
        

for date in os.listdir(path_products):
    print(date)

    path_date = os.path.join(path_products,date)
    path_FSC = os.path.join(path_date,"OUTPUT_FSC_tile-T31TCH_EPSG-32631.tif")
    path_FSC2 = os.path.join(path_date,"OUTPUT2_FSC_tile-T31TCH_EPSG-32631.tif")
    path_NDSI = os.path.join(path_date,"OUTPUT_NDSI_tile-T31TCH_EPSG-32631.tif")
    path_NDSI2 = os.path.join(path_date,"OUTPUT2_NDSI_tile-T31TCH_EPSG-32631.tif")
    path_slope2 = os.path.join(path_date,"OUTPUT2_SLOPE_tile-T31TCH_EPSG-32631.tif")
    path_PRED = os.path.join(path_date,"OUTPUT_PRED_tile-T31TCH_EPSG-32631.tif")
    g_NDSI = gdal.Open(path_NDSI)
    g_FSC = gdal.Open(path_FSC)
    g_slope = gdal.Open(path_slope)

    minx, maxy, maxx, miny = getOverlapCoords(g_NDSI,g_slope)
    
    gdal.Translate(path_FSC2,path_FSC,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
    gdal.Translate(path_NDSI2,path_NDSI,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
    gdal.Translate(path_slope2,path_slope,format= 'GTiff',projWin = [minx, maxy, maxx, miny])
    

        
    os.system("gdal_calc.py "+" -A "+path_NDSI2+" --outfile="+path_PRED+" --calc=\"0.5*tanh(("+str(a)+"*A)+("+str(b)+"))+0.5\" --NoDataValue=9999 ")
    
    g_NDSI = gdal.Open(path_NDSI2)
    g_FSC = gdal.Open(path_FSC2)
    g_PRED = gdal.Open(path_PRED)
    g_SLOPE = gdal.Open(path_slope2)
    
    NDSI = BandReadAsArray(g_NDSI.GetRasterBand(1)).flatten()
    FSC = BandReadAsArray(g_FSC.GetRasterBand(1)).flatten()
    PRED = BandReadAsArray(g_PRED.GetRasterBand(1)).flatten()
    SLOPE = BandReadAsArray(g_SLOPE.GetRasterBand(1)).flatten()

    cond1 = np.where((FSC <= 1) & (~np.isnan(FSC)) & (~np.isinf(FSC)))
    NDSI = NDSI[cond1]
    PRED = PRED[cond1]
    SLOPE = SLOPE[cond1]
    FSC = FSC[cond1]

    cond2 = np.where((NDSI <= 1) & (~np.isnan(NDSI)) & (~np.isinf(NDSI)))
    PRED = PRED[cond2]
    SLOPE = SLOPE[cond2]
    FSC = FSC[cond2]
    NDSI = NDSI[cond2]

    cond3 = np.where((PRED <= 1) & (~np.isnan(PRED)) & (~np.isinf(PRED)))
    SLOPE = SLOPE[cond3]
    FSC = FSC[cond3]
    NDSI = NDSI[cond3]   
    PRED = PRED[cond3] 

    
    ERR = PRED - FSC
    

    
    minSLOPE = 0
    
    k = 40
    j = 50
    print("toto")
    #list_var_FSC_box = [np.std(ERR[np.where((SLOPE>= k) & (SLOPE <= j))])]
    list_var_FSC_box = [sqrt(mean_squared_error(PRED[np.where((SLOPE>= k) & (SLOPE <= j))],FSC[np.where((SLOPE>= k) & (SLOPE <= j))]))]
    print("toto")
    list_var_labels_box = ["[40.0\n50.0]"]
    j = j - 10
    k = k - 10
    


    while j > minSLOPE: 

        #list_var_FSC_box.insert(0,np.std(ERR[np.where((SLOPE >= k) & (SLOPE < j))]))
        list_var_FSC_box.insert(0,sqrt(mean_squared_error(PRED[np.where((SLOPE>= k) & (SLOPE <= j))],FSC[np.where((SLOPE>= k) & (SLOPE <= j))])))
        
        list_var_labels_box.insert(0,"[ "+ "{0:.1f}".format(k) +"\n"+ "{0:.1f}".format(j) +" [")
        j = j - 10
        k = k - 10
        


    
    # Plot figure with subplots 
    fig = plt.figure()
    #st = fig.suptitle("FSC RESIDUALS",size = 16)
    gridspec.GridSpec(1,2)
    
    
    # boxplot avec FSC = 0 et FSC = 1
    ax = plt.subplot2grid((1,2), (0,0),rowspan=1, colspan=2)
    #plt.title('FSC RMSE VS SLOPE',size = 14,y=1.08)
    plt.ylabel('RMSE',size = 14)
    plt.xlabel('SLOPE',size = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.bar(list_var_labels_box,list_var_FSC_box)


    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=8,h=5)
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(path_res,'PLOT_RMSE_'+date+'_PLEIADES_SLOPE.png'))
    plt.close(fig)
    

    
    
