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
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.stats import mstats
import shutil
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm






def getDateFromStr(N):
    sepList = ["","-","_","/"]
    date = ''
    for s in sepList :
        found = re.search('\d{4}'+ s +'\d{2}'+ s +'\d{2}', N)
        if found != None :
           date = datetime.strptime(found.group(0), '%Y'+ s +'%m'+ s +'%d').date()
           break
    return date



def mkdir_p(dos):
    try:
        os.makedirs(dos)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dos):
            pass
        else:
            raise



path_ODK = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/snowcover/INPUTS/FSC/ODK"
path_LIS = "/work/OT/siaa/Theia/Neige/PRODUITS_NEIGE_LIS_develop_1.5"
name_ODK = "odk_all.txt"


path_results = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/snowcover/OUTPUTS/res/ODK"
mkdir_p(path_results)


f_ODK = os.path.join(path_ODK,name_ODK)

nb_shift_days = 4
dict_FSC = {}

list_NDSI = []
list_FSC = []

print("####################################################")
print("Recuperation of ODK data")
#on recupere les donnees odk
with open(f_ODK, "r") as ODK :
    line = ODK.readline()
    line = ODK.readline()
    while line :
        point = line.split()
        date = point[0]
        latitude = point[1]
        longitude = point[2]
        accuracy = point[3]
        fsc = point[4]
        
        
                        
        if date not in dict_FSC.keys() :
            dict_FSC[date] = [[],"",""]
            dict_FSC[date][0].append([latitude,longitude,accuracy,fsc])
        else :
            dict_FSC[date][0].append([latitude,longitude,accuracy,fsc])
            
        line = ODK.readline()
            


print("####################################################")
print("Search of tiles and L2A rasters")
#on trouve les tuiles et rasters correspondants
for date in dict_FSC :
    
    list_points = dict_FSC[date][0]
    point = list_points[0]
    lat = point[0]
    lon = point[1]
    
    
    for tile in os.listdir(path_LIS) :
        
        
        
        
        try:
            L2A_product = os.listdir(os.path.join(path_LIS,tile))[-1]
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EACCES:
                continue
            else:
                raise   
                
        
        L2A_product = os.path.join(path_LIS,tile,L2A_product)
        
        f_L2A = os.path.join(L2A_product,"red_band_extracted.tif")
        

        pixel = os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_L2A, lon, lat)).read()

        try:
            int(pixel)
            
        except ValueError:
            continue
       
       
        
        dict_FSC[date][1] = tile
        
        
        dateFSC = getDateFromStr(date)
        list_L2A = os.listdir(os.path.join(path_LIS,tile))
        
        
        
        L2A = ""
        ind = nb_shift_days + 1
        CP1 = 100
        CP2 = 100
        for L2A_product in list_L2A:
            if "SENTINEL" not in L2A_product : continue
            dateL2A = getDateFromStr(L2A_product)
            decal = dateL2A - dateFSC
            if abs(decal.days) > nb_shift_days : continue
            L2A_product = os.path.join(path_LIS,tile,L2A_product)
            f_mtd = os.path.join(L2A_product,"LIS_PRODUCTS","LIS_METADATA.XML")
            tree = ET.parse(f_mtd)
            root = tree.getroot()
            
            #print("MTD : ",f_mtd)
            for GIL in root.findall('Global_Index_List'):
                for QI in GIL.findall('QUALITY_INDEX'):
                    if QI.get('name') == "CloudPercent" :
                        CP2 = float(QI.text)
            


            if CP2 < 100 and ((abs(CP2 - CP1) < 1 and abs(decal.days) < ind) or (CP1 - CP2 >= 1)) :
                #print("date FSC",dateFSC,"date L2A",dateL2A)
                #print("CP1 = ",CP1,"CP2 = ",CP2,"decal = ",decal.days)
                #print("\n")
                ind = abs(decal.days)
                L2A = L2A_product
                CP1 = CP2
                
        if L2A == "" : continue        
        dict_FSC[date][2] = L2A
        
        


#on affiche le dict
print("####################################################")
print("\n")
for date in dict_FSC :
    print(date)
    print ("TILE : ",dict_FSC[date][1])
    print ("L2A product : ",dict_FSC[date][2])
    for point in dict_FSC[date][0] :
        print("lat = ",point[0],"lon = ",point[1],"acc = ",point[2],"fsc = ",point[3])
    print("\n")

print("####################################################")
print("Comparison NDSI and FSC")

#on compare ODK et L2A
for date in dict_FSC :
    l_points = dict_FSC[date][0]
    L2A_product = dict_FSC[date][2]
    if L2A_product == "" : continue
    
    # We look for the red, green and swir bands tiff files + mask
    f_green = ""
    f_swir = ""
    f_red = ""
    f_mask = ""


    for f in os.listdir(L2A_product) :
        if ("green_band_resampled.tif" in f) :
            f_green = os.path.join(L2A_product,f)
        elif ("red_band_resampled.tif" in f) :
            f_red = os.path.join(L2A_product,f)
        elif ("swir_band_extracted.tif" in f) :
            f_swir = os.path.join(L2A_product,f)
        elif ("LIS_PRODUCTS" in f) :
            if os.path.isfile(os.path.join(L2A_product,f,"LIS_SEB.TIF")):
                f_mask = os.path.join(L2A_product,f,"LIS_SEB.TIF")


    #If there is a file missing, we skip to the next date
    if f_green == "" or f_red == "" or f_swir == "" or f_mask == "": continue
    
    
    #for each ODK point:
    for point in l_points :
        lat = point[0]
        lon = point[1]
        acc = point[2]
        fsc = point[3]
    
    
    
        #We get the corresponding pixel from each band to calculate a NDSI pixel
        green = 0
        red = 0
        swir = 0
        mask = 0
        NDSI = 0
        
        try:
            
            mask = int(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_mask, lon, lat)).read())
            
        except ValueError:
            continue     
            
        
        if mask == 205 or mask == 254 or mask == 0 : continue
        
        
        try:
            
            green = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_green, lon, lat)).read())
            red = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_red, lon, lat)).read())
            swir = float(os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' % (f_swir, lon, lat)).read())
            
        except ValueError:
            continue
            
            
            
        NDSI = (green - swir)/(green + swir)
        
        if NDSI < -1 or NDSI > 1 : continue
        
        list_NDSI.append(NDSI)
        list_FSC.append(float(fsc))
    

#on affiche les lists
print("####################################################")
print("\n")

for i in arange(len(list_NDSI)) :
    print("NDSI = ",list_NDSI[i],"FSC = ",list_FSC[i])

print("####################################################")
print("Calculation of NDSI-FSC relation")

#on calcul et affiche la relation FSC-NDSI

NDSI = np.asarray(list_NDSI)
FSC = np.asarray(list_FSC)
NDSI2 = NDSI[np.where((FSC > 0) & (FSC < 1))]
FSC2 = FSC[np.where((FSC > 0) & (FSC < 1))]




minNDSI = min(NDSI)
list_FSC_box = [FSC[np.where((NDSI >= 0.8) & (NDSI <= 1))]]
list_labels_box = ["[ 0.8\n1 ]"]
b = 0.8
while minNDSI < b : 
    a = b - 0.2
    list_FSC_box.insert(0,FSC[np.where((NDSI >= a) & (NDSI < b))])
    list_labels_box.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
    b = b - 0.2
    

minNDSI2 = min(NDSI2)
list_FSC_box2 = [FSC2[np.where((NDSI2 >= 0.8) & (NDSI2 <= 1))]]
list_labels_box2 = ["[ 0.8\n1 ]"]
b = 0.8
while minNDSI2 < b : 
    a = b - 0.2
    list_FSC_box2.insert(0,FSC2[np.where((NDSI2 >= a) & (NDSI2 < b))])
    list_labels_box2.insert(0,"[ "+ "{0:.1f}".format(a) +"\n"+ "{0:.1f}".format(b) +" [")
    b = b - 0.2         




# Plot figure with subplots 
fig = plt.figure()
st = fig.suptitle("FSC / NDSI")
gridspec.GridSpec(2,3)

# 2D histo avec FSC = 0 et FSC = 1
ax = plt.subplot2grid((2,3), (0,2))
slope, intercept, r_value, p_value, std_err = mstats.linregress(NDSI,FSC) 

plt.ylabel('0 <= FSC <= 1')
plt.xlabel('NDSI')
plt.hist2d(NDSI,FSC,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'), norm=LogNorm())
n = np.array([minNDSI,1.0])
line = slope*n+intercept
plt.plot(n, line, 'g', label='FSC={:.2f}NDSI+{:.2f}\ncorr={:.2f}'.format(slope,intercept,r_value))
plt.legend(fontsize=6,loc='upper left')
plt.colorbar()
ratio = 1
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  

# 2D histo sans FSC = 0 et FSC = 1
ax = plt.subplot2grid((2,3), (1,2))
slope2, intercept2, r_value2, p_value2, std_err2 = mstats.linregress(NDSI2,FSC2) 

plt.ylabel('0 < FSC < 1')
plt.xlabel('NDSI')
plt.hist2d(NDSI2,FSC2,bins=(40, 40), cmap=plt.cm.get_cmap('plasma'),norm=LogNorm())
n = np.array([minNDSI2,1.0])
line = slope2*n+intercept2
plt.plot(n, line, 'g', label='FSC={:.2f}NDSI+{:.2f}\ncorr={:.2f}'.format(slope2,intercept2,r_value2))
plt.legend(fontsize=6,loc='upper left')
plt.colorbar()
ratio = 1
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)  



# boxplot avec FSC = 0 et FSC = 1
ax = plt.subplot2grid((2,3), (0,0),rowspan=1, colspan=2)
plt.title('ANALYSIS WITH 0 <= FSC <= 1')
plt.ylabel('0 <= FSC <= 1')
plt.xlabel('NDSI')
plt.boxplot(list_FSC_box,labels = list_labels_box)



# boxplot sans FSC = 0 et FSC = 1
ax = plt.subplot2grid((2,3), (1,0),rowspan=1, colspan=2)
plt.title('ANALYSIS WITH 0 < FSC < 1')
plt.ylabel('0 < FSC < 1')
plt.xlabel('NDSI')
plt.boxplot(list_FSC_box2,labels = list_labels_box2)

# fit subplots and save fig
fig.tight_layout()

fig.set_size_inches(w=16,h=10)
st.set_y(0.95)
fig.subplots_adjust(top=0.85)
fig.savefig(os.path.join(path_results,'FSC_NDSI_'  + name_ODK + '.png'))
plt.close(fig)

print("plotting finished")


