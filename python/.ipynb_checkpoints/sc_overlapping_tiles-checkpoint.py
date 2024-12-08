
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
import getpass




def searchOverlappingTileImage(list_overlapping_tiles_path,list_inputs_coverage_path,LIS_product_path,tile,path_LIS):
    
    #print("lis_product:",LIS_product_path)
    #print(os.access(LIS_product_path,os.R_OK))
    #print(os.listdir(LIS_product_path)[0])
    #g_lis = gdal.Open(os.path.join(LIS_product_path,os.listdir(LIS_product_path)[0]))
    g_lis = gdal.Open(LIS_product_path)
    with open(list_inputs_coverage_path,'r') as list_inputs_coverage_file:
        reader = csv.DictReader(list_inputs_coverage_file)
        for row in reader:
            f_fsc = row["file"]
            epsg = row["epsg"]
            g_fsc = gdal.Open(f_fsc)
            g_fsc = gdal.Warp('',g_fsc,format= 'MEM',srcSRS="EPSG:" + epsg)
            if sc_utils.isOverlapping(g_fsc,g_lis) : 
                list_overlapping_tiles_file = open(list_overlapping_tiles_path,'a')
                writer = csv.writer(list_overlapping_tiles_file)
                writer.writerow([tile,os.path.join(path_LIS,tile)])
                list_overlapping_tiles_file.close()
                break



def searchOverlappingTilePoint(list_overlapping_tiles_path,list_inputs_coverage_path,LIS_product_path,tile,path_LIS):
    
    g_lis = gdal.Open(LIS_product_path)
    with open(list_inputs_coverage_path,'r') as list_inputs_coverage_file:
        reader = csv.DictReader(list_inputs_coverage_file)
        for row in reader:
            lat = row['lat']
            lon = row['lon']
            if sc_utils.isCoordInside(g_lis,lon,lat) : 
                list_overlapping_tiles_file = open(list_overlapping_tiles_path,'a')
                writer = csv.writer(list_overlapping_tiles_file)
                writer.writerow([tile,os.path.join(path_LIS,tile)])
                list_overlapping_tiles_file.close()
                break





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-list_overlapping_tiles_path', action='store', default="", dest='list_overlapping_tiles_path')
    parser.add_argument('-list_inputs_coverage_path', action='store',default="", dest='list_inputs_coverage_path')
    parser.add_argument('-LIS_product_path', action='store',default="", dest='LIS_product_path')
    parser.add_argument('-tile', action='store', default="", dest='tile')
    parser.add_argument('-path_LIS', action='store', default="", dest='path_LIS')
    parser.add_argument('-type_data', action='store', default="", dest='type_data')

    list_overlapping_tiles_path = parser.parse_args().list_overlapping_tiles_path
    list_inputs_coverage_path = parser.parse_args().list_inputs_coverage_path
    LIS_product_path = parser.parse_args().LIS_product_path
    tile = parser.parse_args().tile
    path_LIS = parser.parse_args().path_LIS
    type_data = parser.parse_args().type_data

    if type_data == "image":
        searchOverlappingTileImage(list_overlapping_tiles_path,list_inputs_coverage_path,LIS_product_path,tile,path_LIS)
    elif type_data == "point":
        searchOverlappingTilePoint(list_overlapping_tiles_path,list_inputs_coverage_path,LIS_product_path,tile,path_LIS)


if __name__ == '__main__':
    main()
