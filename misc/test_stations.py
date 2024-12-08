import json
import sys
import os
import errno
import re
import csv
from datetime import datetime
from datetime import date as libdate
import glob
import numpy as np

f_in_total = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/FSC/BD_SYNOP_ALPS/combi_monde_alpes_dp.csv"
f_in_SYNOP = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/FSC/BDALPS/alpes_dp.csv"
f_in_ALPS = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/FSC/SYNOP/mondefrance_dp.csv"
f_total = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/BD_SYNOP_ALPS/datasets/list_points_datasets.csv"
f_SYNOP = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/BDALPS/datasets/list_points_datasets.csv"
f_ALPS = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/SYNOP/datasets/list_points_datasets.csv"
list_total = []
list_SYNOP = []
list_ALPS = []
with open(f_total,'r') as f : 
    reader = csv.DictReader(f)
    for row in reader:
        lat= row["lat"]
        lon = row["lon"]
        coords = [str(round(float(lat),4)),str(round(float(lon),4)),lat,lon]
        if coords not in list_total:
            list_total.append(coords)
    print("bd_total",len(list_total))
with open(f_SYNOP,'r') as f : 
    reader = csv.DictReader(f)
    for row in reader:
        lat= row["lat"]
        lon = row["lon"]
        coords = [str(round(float(lat),4)),str(round(float(lon),4)),lat,lon]
        if coords not in list_SYNOP:
            list_SYNOP.append(coords)
    print("bd_synop",len(list_SYNOP))
with open(f_ALPS,'r') as f : 
    reader = csv.DictReader(f)
    for row in reader:
        lat= row["lat"]
        lon = row["lon"]
        coords = [str(round(float(lat),4)),str(round(float(lon),4)),lat,lon]
        if coords not in list_ALPS:
            list_ALPS.append(coords)
    print("bd_alps",len(list_ALPS))

print("synop + alps",len(list_SYNOP)+len(list_ALPS))

for coords in list_total:

    if coords[0:2] not in [synop[0:2] for synop in list_SYNOP] and coords[0:2] not in [alps[0:2] for alps in list_ALPS]:
        print(coords)
        
        


