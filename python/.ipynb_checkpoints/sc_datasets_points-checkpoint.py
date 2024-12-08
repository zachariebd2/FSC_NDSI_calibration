
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





def make_datasets(nb_shift_days,tile_path,date,valid_products_path):
    

    list_products = []
    LIS_products = sc_utils.getListDateDecal(date,date,tile_path,nb_shift_days,"/FSC_*/")
    
    for LIS_product in LIS_products:
        dateLIS = sc_utils.getDateFromStr(LIS_product)
        decal = dateLIS - sc_utils.getDateFromStr(date)
        decal = abs(decal.days)
        list_products.append([dateLIS.strftime("%Y-%m-%d"),str(decal),LIS_product])

    with open(valid_products_path,'w') as valid_products_file :
        writer = csv.writer(valid_products_file)
        writer.writerow(["date","decal","LIS_product"])
        for dateLIS , decal , LIS_product in  sorted(list_products,key=lambda l:l[1]) : 
            writer.writerow([dateLIS,decal,LIS_product])



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-nb_shift_days', action='store', default=0,type=int, dest='nb_shift_days')
    parser.add_argument('-date', action='store', default="", dest='date')
    parser.add_argument('-valid_products_path', action='store', default="", dest='valid_products_path')
    parser.add_argument('-tile_path', action='store', default="", dest='tile_path')

    nb_shift_days = parser.parse_args().nb_shift_days
    tile_path = parser.parse_args().tile_path
    date = parser.parse_args().date
    valid_products_path = parser.parse_args().valid_products_path



    make_datasets(nb_shift_days,tile_path,date,valid_products_path)
    


if __name__ == '__main__':
    main()
