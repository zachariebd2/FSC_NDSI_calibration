import json
import sys
import os
import errno
import re
from datetime import datetime, timedelta, date


def getDateFromStr(N):
    sepList = ["","-","_","/"]
    date = ''
    for s in sepList :
        found = re.search('\d{4}'+ s +'\d{2}'+ s +'\d{2}', N)
        if found != None :
           date = datetime.strptime(found.group(0), '%Y'+ s +'%m'+ s +'%d').date()
           break
    return date

gjson_file = open("cso-data.geojson")
gjson = json.load(gjson_file)

txt_file = open("cso_dp.txt","w")
txt_file.write("date Latitude Longitude Accuracy neige")

for POINTS in gjson["features"]:
    date = getDateFromStr(POINTS["properties"]["timestamp"])
    lon = POINTS["geometry"]["coordinates"][0]
    lat = POINTS["geometry"]["coordinates"][1]
    depth = float(POINTS["properties"]["depth"])
    if depth > 0:
        txt_file.write("\n"+date.strftime("%Y-%m-%d")+" "+str(lat)+" "+str(lon)+" "+"0"+" "+"1")
    elif depth == 0:
        txt_file.write("\n"+date.strftime("%Y-%m-%d")+" "+str(lat)+" "+str(lon)+" "+"0"+" "+"0")
    
    
    
    
    
    
    
    
    
    
    
    
    
