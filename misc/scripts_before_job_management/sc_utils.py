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
import itertools


def getListDateDecal(start_d,end_d,directory,decal,text):
        lo=[]
        li = []
        try:
            #li = os.listdir(directory)
            li = glob.glob(os.path.join(directory,'**' + text), recursive=True)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EACCES:
                return lo
            else:
                raise       
        for i in sorted(li):
            date = getDateFromStr(i) 
            if date == '' : continue
            if (date >= getDateFromStr(start_d) - timedelta(days = decal)) and (date <= getDateFromStr(end_d) + timedelta(days = decal)) :
                lo.append(i)
        return lo


def mkdir_p(dos):
    try:
        os.makedirs(dos)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dos):
            pass
        else:
            raise


def getDateFromStr(N):
    sepList = ["","-","_"]
    date = ''
    for s in sepList :
        found = re.search('\d{4}'+ s +'\d{2}'+ s +'\d{2}', N)
        if found != None :
           date = datetime.strptime(found.group(0), '%Y'+ s +'%m'+ s +'%d').date()
           break
    return date



def getTileFromStr(N):

    tile = ''
    found = re.search('\d{2}' +'[A-Z]{3}', N)
    if found != None : tile = found.group(0)
       
    return tile
    
def getEpsgFromStr(N):
    
    epsg = ''
    found = re.search('\d{5}', N)
    if found != None : epsg = found.group(0)
       
    return str(epsg)


def getOverlapCoords(G1,G2):
    
    epsg1 = (gdal.Info(G1, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    epsg2 = (gdal.Info(G2, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    
    GT1 = G1.GetGeoTransform()
    minx1 = GT1[0]
    maxy1 = GT1[3]
    maxx1 = minx1 + GT1[1] * G1.RasterXSize
    miny1 = maxy1 + GT1[5] * G1.RasterYSize
    
    GT2 = G2.GetGeoTransform()
    minx2 = GT2[0]
    maxy2 = GT2[3]
    maxx2 = minx2 + GT2[1] * G2.RasterXSize
    miny2 = maxy2 + GT2[5] * G2.RasterYSize
    
    if epsg1 not in epsg2 :
        minx1 , miny1 = reproject(epsg1,epsg2,minx1,miny1)
        maxx1 , maxy1 = reproject(epsg1,epsg2,maxx1,maxy1)
    
    
    minx3 = max(minx1,minx2)
    maxy3 = min(maxy1,maxy2)
    maxx3 = min(maxx1,maxx2)
    miny3 = max(miny1,miny2)   
    
    # no intersection 
    if (minx3 > maxx3 or miny3 > maxy3) : 
        return None,None,None,None
    
    return minx3, maxy3, maxx3, miny3
    
    
def isOverlapping(G1,G2):
    
    epsg1 = (gdal.Info(G1, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    epsg2 = (gdal.Info(G2, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    
    GT1 = G1.GetGeoTransform()
    minx1 = GT1[0]
    maxy1 = GT1[3]
    maxx1 = minx1 + GT1[1] * G1.RasterXSize
    miny1 = maxy1 + GT1[5] * G1.RasterYSize
    
    GT2 = G2.GetGeoTransform()
    minx2 = GT2[0]
    maxy2 = GT2[3]
    maxx2 = minx2 + GT2[1] * G2.RasterXSize
    miny2 = maxy2 + GT2[5] * G2.RasterYSize
    
    if epsg1 not in epsg2 :
        minx1 , miny1 = reproject(epsg1,epsg2,minx1,miny1)
        maxx1 , maxy1 = reproject(epsg1,epsg2,maxx1,maxy1)
    
    minx3 = max(minx1,minx2)
    maxy3 = min(maxy1,maxy2)
    maxx3 = min(maxx1,maxx2)
    miny3 = max(miny1,miny2)   
    
    # no intersection 
    if (minx3 > maxx3 or miny3 > maxy3) : 
        return False
    else:
        return True
    


    
def isInside(Gbig,Gsmall):
    
    epsgbig = (gdal.Info(Gbig, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    epsgsmall = (gdal.Info(Gsmall, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    

    
    GTbig = Gbig.GetGeoTransform()
    minxbig = GTbig[0]
    maxybig = GTbig[3]
    maxxbig = minxbig + GTbig[1] * Gbig.RasterXSize
    minybig = maxybig + GTbig[5] * Gbig.RasterYSize
    
    GTsmall = Gsmall.GetGeoTransform()
    minxsmall = GTsmall[0]
    maxysmall = GTsmall[3]
    maxxsmall = minxsmall + GTsmall[1] * Gsmall.RasterXSize
    minysmall = maxysmall + GTsmall[5] * Gsmall.RasterYSize
    
    if epsgbig not in epsgsmall :
        minxsmall , minysmall = reproject(epsgsmall,epsgbig,minxsmall,minysmall)
        maxxsmall , maxysmall = reproject(epsgsmall,epsgbig,maxxsmall,maxysmall)
    
    if minxbig <= minxsmall and maxxbig >= maxxsmall and minybig <= minysmall and maxybig >=maxysmall :
        return True
    else :
        return False



def reproject(inEPSG,outEPSG,x1,y1):
    
    inProj = Proj(init='EPSG:' + inEPSG)
    outProj = Proj(init='EPSG:'+ outEPSG)
    x2,y2 = transform(inProj,outProj,x1,y1)
    
    return x2, y2



def isCoordInside(Gbig,x,y,cEPSG):
    
    epsgbig = (gdal.Info(Gbig, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    epsgsmall = cEPSG
    

    
    GTbig = Gbig.GetGeoTransform()
    minxbig = GTbig[0]
    maxybig = GTbig[3]
    maxxbig = minxbig + GTbig[1] * Gbig.RasterXSize
    minybig = maxybig + GTbig[5] * Gbig.RasterYSize
    
 
    
    if epsgbig not in epsgsmall :
        x,y = reproject(epsgsmall,epsgbig,x,y)
        
    
    if minxbig <= x and maxxbig >= x and minybig <= y and maxybig >= y :
        return True
    else :
        return False




def plot_confusion_matrix(cm,target_names,cmap=None,normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    if cmap is None:
        cmap = plt.get_cmap('Blues')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    
