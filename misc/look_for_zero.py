
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










# test de gdal warp pour fake_scf (lambert et res=25 -> utm et res=20)
g = gdal.Open("SENTINEL2A_20161201-103456-971_L2A_T31TGL_D_V1-1_FRE_B11.tif")






b = BandReadAsArray(g.GetRasterBand(1))


zero = b[b == 0]

nozero = b[b != 0]


print("\nNombre de zero: " + str(len(zero)))
print("\nNombre de non zero: " + str(len(nozero)))
