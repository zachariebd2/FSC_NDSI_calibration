import sys
import os
import errno
import re
import copy
from datetime import datetime, timedelta, date
import glob
import sc_utils
import argparse
import csv





def searchFSC(start_date,end_date,source,path_inputs,list_fsc_path):
    path_FSC_dir = os.path.join(path_inputs,source)
    list_FSC_products = sorted(sc_utils.getListDateDecal(start_date,end_date,path_FSC_dir,0,""))
    with open(list_fsc_path,"w") as list_fsc_file :
        writer = csv.writer(list_fsc_file)
        writer.writerow(["date","file"])
        for fsc_file in list_FSC_products:
            date = sc_utils.getDateFromStr(fsc_file).strftime("%Y-%m-%d")
            writer.writerow([date,fsc_file])


def searchPoints(start_date,end_date,source,path_inputs,list_input_points_path):
    
    list_points = []
    params = []
    path_FSC_dir = os.path.join(path_inputs,source)
    for i in glob.glob(os.path.join(path_FSC_dir,'*')):
        with open(i,'r') as csv_points_file:
            reader = csv.DictReader(csv_points_file)
            if len(params) == 0 : params = reader.fieldnames
            for row in reader:
                date = sc_utils.getDateFromStr(row["date"])
                if (date >= sc_utils.getDateFromStr(start_date)) and (date <= sc_utils.getDateFromStr(end_date)):
                    list_points.append([row[key] for key in params])

    with open(list_input_points_path,"w") as list_input_points_file:
        writer = csv.writer(list_input_points_file)
        writer.writerow(params)
        for point in sorted(list_points,key=lambda l:l[0]):
            writer.writerow(point)




def searchAvailableTiles(path_LIS,out,path_outputs,list_available_tiles_path):
    list_available_tiles_file = open(list_available_tiles_path,"w")
    list_restricted_tiles_path = os.path.join(path_outputs,out,"list_restricted_tiles.txt")
    list_restricted_tiles_file = open(list_restricted_tiles_path,"w")
    writer_available = csv.writer(list_available_tiles_file)
    writer_restricted = csv.writer(list_restricted_tiles_file)


    writer_available.writerow(["tile","product"])
    writer_restricted.writerow(["tile","product"])
    for tile in os.listdir(path_LIS) :
        if sc_utils.getTileFromStr(tile) == "" : continue
        path_tile = os.path.join(path_LIS,tile)
        try:
            path_year = os.path.join(path_tile,os.listdir(path_tile)[-1])
            path_month = os.path.join(path_year,os.listdir(path_year)[0])
            path_day = os.path.join(path_month,os.listdir(path_month)[0])
            path_product = os.path.join(path_day,os.listdir(path_day)[0])
            path_NDSI = glob.glob(os.path.join(path_product,'*NDSI*'))[0]
        except (OSError,IndexError) as exc:  # Python >2.5
            print("access not permitted!")
            writer_restricted.writerow([tile,path_product])
            continue
        writer_available.writerow([tile,path_product])
    list_available_tiles_file.close()
    list_restricted_tiles_file.close()


