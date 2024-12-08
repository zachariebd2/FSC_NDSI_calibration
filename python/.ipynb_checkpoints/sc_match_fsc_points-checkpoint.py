
import sys
import os
import errno
import re
import copy
from datetime import datetime, timedelta, date
import glob
import sc_utils
import argparse
import numpy as np
import csv
import getpass




def matchFSCPoints(matched_points_tiles_dir,list_input_points_path,list_points_datasets_path,snw_type,nb_shift_days):
    
    print("extract input point")

    print(list_input_points_path)
    #extract input points
    dict_input_points = {}
    with open(list_input_points_path,'r') as list_input_points_file:
        reader = csv.DictReader(list_input_points_file)
        for row in reader:
       
            date = row["date"]
            lat = row["lat"]
            lon = row["lon"]
            value = row[snw_type]
            acc = row["acc"]
            if lat not in dict_input_points:
                dict_input_points[lat] = {}
                dict_input_points[lat][lon] = {}
            elif lon not in dict_input_points[lat]:
                dict_input_points[lat][lon] = {}
            
            dict_input_points[lat][lon][date] = [value,acc]
            

    print("extract LIS points")
    print(matched_points_tiles_dir)
    #extract lis points
    dict_lis_points = {}
    for tile in os.listdir(matched_points_tiles_dir):
        #print("tile",tile)
        list_lis_points_path = os.path.join(matched_points_tiles_dir,tile,"list_points_FSCOG.csv")
        with open(list_lis_points_path,'r') as list_lis_points_file:
            reader = csv.DictReader(list_lis_points_file)
            for row in reader:
      
                list_dates = row["dates"].split()
                lat = row["lat"]
                lon = row["lon"]
                alt = row["alt"]
                slp = row["slp"]
                country = row["country"]
                cover = row["cover"]
                list_fsc = row["fscog"].split()
                list_qcflags = row["qcflags"].split()
                TCD = row["TCD"]
                if lat not in dict_lis_points:
                    dict_lis_points[lat] = {}
                if lon not in dict_lis_points[lat]:
                    dict_lis_points[lat][lon] = {}
                if tile not in dict_lis_points[lat][lon]:
                    dict_lis_points[lat][lon][tile] = {}
                for i in range(len(list_dates)):
                    dict_lis_points[lat][lon][tile][list_dates[i]] = [list_fsc[i],list_qcflags[i],TCD,alt,country,slp,cover]


    print("extract MAJA points")

    print(matched_points_tiles_dir)
    #extract lis points
    dict_maja_points = {}
    for tile in os.listdir(matched_points_tiles_dir):
        #print("tile",tile)
        list_maja_points_path = os.path.join(matched_points_tiles_dir,tile,"list_points_MAJA.csv")
        with open(list_maja_points_path,'r') as list_maja_points_file:
            reader = csv.DictReader(list_maja_points_file)
            for row in reader:
           
                list_dates = row["dates"].split()
                lat = row["lat"]
                lon = row["lon"]
                list_green = row["green"].split()
                list_red = row["red"].split()
                list_mir = row["mir"].split()

                if lat not in dict_maja_points:
                    dict_maja_points[lat] = {}
                if lon not in dict_maja_points[lat]:
                    dict_maja_points[lat][lon] = {}
                if tile not in dict_maja_points[lat][lon]:
                    dict_maja_points[lat][lon][tile] = {}
                for i in range(len(list_dates)):
                    dict_maja_points[lat][lon][tile][list_dates[i]] = [list_green[i],list_red[i],list_mir[i]]



    print("match points")
  
    #match input and lis points
    list_inputs_lis_points = []
    for lat in dict_input_points.keys():
        if lat not in dict_lis_points: continue
        for lon in dict_input_points[lat]:
            if lon not in dict_lis_points[lat]: continue
            
            for datei in dict_input_points[lat][lon]:
                final_decal = nb_shift_days + 1
                final_date_lis = ""
                final_fsc = 0
                final_qcflags = 0
                final_tcd = 0
                final_alt = 0
                final_slp = 0
                final_country = ""
                final_cover=""
                final_tile = ""
                value = dict_input_points[lat][lon][datei][0]
                acc = dict_input_points[lat][lon][datei][1]
                date_input = sc_utils.getDateFromStr(datei)
                #TODO: faire une liste de dates possibles (+/- nb_shift_day) et chercher directement ces dates au lieu de tout chercher
                for tile in dict_lis_points[lat][lon]:
                    for datel in dict_lis_points[lat][lon][tile]:
                        fsc = dict_lis_points[lat][lon][tile][datel][0]
                        qcflags = dict_lis_points[lat][lon][tile][datel][1]
                   
                        tcd = dict_lis_points[lat][lon][tile][datel][2]
                        alt = dict_lis_points[lat][lon][tile][datel][3]
                        country = dict_lis_points[lat][lon][tile][datel][4]
                        slp = dict_lis_points[lat][lon][tile][datel][5]
                        cover = dict_lis_points[lat][lon][tile][datel][6]

                        date_lis = sc_utils.getDateFromStr(datel)
                        decal = abs((date_input - date_lis).days)
                        if (decal < final_decal) and (int(fsc) != 205) and (int(fsc) != 255):
                            final_decal = decal
                            final_date_lis = datel
                            final_fsc = fsc
                            final_alt = alt
                            final_slp = slp
                            final_country = country
                            final_qcflags = qcflags
                            final_cover = cover
                            final_tcd = tcd
                            final_tile = tile
                if final_date_lis != "":
                    red = "-9999"
                    green = "-9999"
                    mir = "-9999"
                    if lat in dict_maja_points:
                        if lon in dict_maja_points[lat]:
                            if final_tile in dict_maja_points[lat][lon]:
                                if final_date_lis in dict_maja_points[lat][lon][final_tile]:
                                    green = dict_maja_points[lat][lon][final_tile][final_date_lis][0]
                                    red = dict_maja_points[lat][lon][final_tile][final_date_lis][1]
                                    mir = dict_maja_points[lat][lon][final_tile][final_date_lis][2]
                    list_inputs_lis_points.append([lat,lon,final_alt,final_slp,final_country,final_cover,date_input.strftime("%Y%m%d"),value,acc,final_tile,str(final_decal),final_date_lis,final_fsc,final_qcflags,final_tcd,green,red,mir])

    #write output file
    print("write output")
    print(list_points_datasets_path)
    with open(list_points_datasets_path,'w') as list_points_datasets_file:
        writer = csv.writer(list_points_datasets_file)
        writer.writerow(["lat","lon","alt","slp","country","cover","date_input",snw_type,"acc","tile","decal","date_lis","FSCOG","QCFLAGS","TCD","green","red","mir"])
        for point in list_inputs_lis_points:
            writer.writerow(point)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-matched_points_tiles_dir', action='store', default="", dest='matched_points_tiles_dir')
    parser.add_argument('-list_input_points_path', action='store',default="", dest='list_input_points_path')
    parser.add_argument('-list_points_datasets_path', action='store',default="", dest='list_points_datasets_path')
    parser.add_argument('-snw_type', action='store',default="", dest='snw_type')
    parser.add_argument('-nb_shift_days', action='store',default=4,type=int, dest='nb_shift_days')

    matched_points_tiles_dir = parser.parse_args().matched_points_tiles_dir
    list_input_points_path = parser.parse_args().list_input_points_path
    list_points_datasets_path = parser.parse_args().list_points_datasets_path
    snw_type = parser.parse_args().snw_type
    nb_shift_days = parser.parse_args().nb_shift_days

    matchFSCPoints(matched_points_tiles_dir,list_input_points_path,list_points_datasets_path,snw_type,nb_shift_days)



if __name__ == '__main__':
    main()
