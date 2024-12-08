
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
#import pylab as plt








# dossier des inputs de test
d_input = "INPUT_TESTS"

# noms inputs
f_scf = "fake_scf.asc"
f_greenS2 = "fake_green.asc"
f_swirS2 = "fake_swir.asc"
f_maskC = "fake_mask.asc"


# test de gdal warp pour fake_scf (lambert et res=25 -> utm et res=20)
g_scf = gdal.Open(os.path.join(d_input, f_scf))

print("\n ################# SCF LAMBERT 93 RES= 20: \n")
print(np.matrix(BandReadAsArray(g_scf.GetRasterBand(1))))



G1 = gdal.Warp('',g_scf,format= 'MEM',srcSRS="EPSG:2154",dstSRS="EPSG:32631",resampleAlg="near",xRes=25,yRes=25)
print("\n ################# SCF UTM 31 RES= 25: \n")
print(np.matrix(BandReadAsArray(G1.GetRasterBand(1))))

g_scf_25 = gdal.Warp('',g_scf,format= 'MEM',resampleAlg="near",xRes=25,yRes=25)



print("\n ################# SCF LAMBERT 93 RES= 25: \n")
print(np.matrix(BandReadAsArray(g_scf_25.GetRasterBand(1))))

g_scf_25 = None
G1 = None

# on prepare les donnees geo de scf pour le decoupage des bandes sentinels

geoTransform = g_scf.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * g_scf.RasterXSize
miny = maxy + geoTransform[5] * g_scf.RasterYSize


#   Decoupage de la bande verte    
g_greenS2 = gdal.Open(os.path.join(d_input, f_greenS2))
g_greenS2_geo = gdal.Translate('',g_greenS2,format= 'MEM',projWin = [minx, maxy, maxx, miny])
print("\n ################# VERT LAMBERT 93 RES= 20 DECOUPE: \n")
print(np.matrix(BandReadAsArray(g_greenS2_geo.GetRasterBand(1))))
g_greenS2 = None

#   Decoupage de la bande swir    
g_swirS2 = gdal.Open(os.path.join(d_input, f_swirS2))
g_swirS2_geo = gdal.Translate('',g_swirS2,format= 'MEM',projWin = [minx, maxy, maxx, miny])
print("\n ################# SWIR LAMBERT 93 RES= 20 DECOUPE: \n")
print(np.matrix(BandReadAsArray(g_swirS2_geo.GetRasterBand(1))))
g_swirS2 = None

#   Decoupage du masque nuage    
g_maskC = gdal.Open(os.path.join(d_input, f_maskC))
g_maskC_geo = gdal.Translate('',g_maskC,format= 'MEM',projWin = [minx, maxy, maxx, miny])
print("\n ################# MASK LAMBERT 93 RES= 20 DECOUPE: \n")
print(np.matrix(BandReadAsArray(g_maskC_geo.GetRasterBand(1))))
g_maskC = None


#   on obtient les bandes verte et IR en Array

bandV = BandReadAsArray(g_greenS2_geo.GetRasterBand(1))
bandIR = BandReadAsArray(g_swirS2_geo.GetRasterBand(1))

#   On calcul les NDSI  (NDSI)

a = (bandV - bandIR)
b = (bandV + bandIR)
NDSI = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
bandV = None
bandIR = None
g_greenS2_geo = None
g_swirS2_geo = None

print("\n ################# NDSI LAMBERT 93 RES= 20 DECOUPE: \n")
print(np.matrix(NDSI))

#   ETAPE 5 MASQUAGE #####################

FSC = BandReadAsArray(g_scf.GetRasterBand(1))

MASKC = BandReadAsArray(g_maskC_geo.GetRasterBand(1))
MASKC[MASKC != 0] = True
MASKC[MASKC == 0] = False
MASKC[FSC == -99999] = True
#MASKC = np.ma.array(MASKC)
#MASKC = np.ma.masked_where(MASKC != 0 , MASKC) 
#MASKC = np.ma.masked_where(FSC == -99999 , MASKC) 


print("\n ################# FSC LAMBERT 93 RES= 20 DECOUPE: \n")
print(np.matrix(FSC))
print("\n ################# MASK LAMBERT 93 RES= 20 DECOUPE: \n")
print(np.matrix(MASKC))

g_scf = None
g_maskC_geo = None


np.save('TEMP/FSC_one_array', FSC)
np.save('TEMP/NDSI_one_array', NDSI)
np.save('TEMP/mask_one_array', MASKC)


#coef = np.ma.corrcoef(FSC_masked_ALL.flatten(),NDSI.flatten())

#print("\n coeff de correlation: " + str(coef))


#   ETAPE 6 CREATION ET AFFICHAGE FICHIER TABLEAUX POUR UNE DATE#####################
#   On produit un graph FSC/NDSI avec correlation et on l'enregistre dans resultats


#plt.plot(NDSI.flatten(), FSC_masked_ALL.flatten() ,'ko')
#plt.grid()
#plt.title('Rapport FSC/NDSI')
#plt.xlabel('NDSI')
#plt.ylabel('FSC')
#plt.savefig(os.path.join(d_input,'NDSI_SDF_res_TEST.png'))



