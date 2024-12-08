
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
import sc_utils
import csv






def getListImagesCoverage(list_fsc_path,list_inputs_coverage_path,epsg):


    list_coverage_coord = []
    with open(list_fsc_path, "r") as list_fsc_file :
        reader = csv.DictReader(list_fsc_file)
        for row in reader:
            f_fsc = row["file"]
            g_fsc = gdal.Open(f_fsc)
            if epsg == "" :
                epsg = (gdal.Info(f_fsc, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
            coords = list(sc_utils.getCoords(g_fsc))
            if [epsg,coords[0],coords[1],coords[2],coords[3]] not in list_coverage_coord : 
                list_coverage_coord.append([epsg,coords[0],coords[1],coords[2],coords[3]])


    with open(list_inputs_coverage_path,'w') as list_inputs_coverage_file:
        writer = csv.writer(list_inputs_coverage_file)
        writer.writerow(["file","epsg","minx","maxy","maxx","miny"])
        for i in list_coverage_coord:
            writer.writerow([f_fsc,i[0],i[1],i[2],i[3],i[4]])



def getListPointsCoverage(list_input_points_path,list_input_coordinates_path):

    list_coord = []
    with open(list_input_points_path,'r') as list_input_points_file:
        reader = csv.DictReader(list_input_points_file)
        for row in reader:
            lon = row["lon"]
            lat = row["lat"]
            if [lat,lon] not in list_coord:
                list_coord.append([lat,lon])

    with open(list_input_coordinates_path,'w') as list_input_coordinates_file:
        writer = csv.writer(list_input_coordinates_file)
        writer.writerow(["lat","lon"])
        for coord in list_coord:
            writer.writerow(coord)





