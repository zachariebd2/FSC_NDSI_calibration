import os
import sys
import os.path as op
import json
import csv
import logging
import subprocess
from datetime import datetime, timedelta
from libamalthee import Amalthee

def str_to_datetime(date_string, format="%Y%m%d"):
    """ Return the datetime corresponding to the input string
    """
    logging.debug(date_string)
    return datetime.strptime(date_string, format)

def datetime_to_str(date, format="%Y%m%d"):
    """ Return the datetime corresponding to the input string
    """
    logging.debug(date)
    return date.strftime(format)
    

class prepare_data_for_snow_annual_map():
    def __init__(self, params):
        logging.info("Init snow_multitemp")
        self.raw_params = params

        self.tile_id = params.get("tile_id")
        self.date_start = str_to_datetime(params.get("date_start"), "%d/%m/%Y")
        self.date_stop = str_to_datetime(params.get("date_stop"), "%d/%m/%Y")
        self.date_margin = timedelta(days=params.get("date_margin", 0))
        self.output_dir = str(params.get("output_dir"))
        self.nbThreads = params.get("nbThreads", None)



    def run(self):
        logging.info('Process tile:' + self.tile_id + '...')
        
        search_start_date = self.date_start - self.date_margin
        search_stop_date = self.date_stop + self.date_margin
        
        parameters = {"processingLevel": "LEVEL2A", "location":str(self.tile_id)}
        amalthee_theia = Amalthee('theia')
        amalthee_theia.search("SENTINEL2",
                              datetime_to_str(search_start_date, "%Y-%m-%d"),
                              datetime_to_str(search_stop_date, "%Y-%m-%d"),
                              parameters,
                              nthreads = self.nbThreads)

        nb_products = amalthee_theia.products.shape[0]
        logging.info('There is ' + str(nb_products) + ' products for the current request')

        amalthee_theia.fill_datalake()
        amalthee_theia.create_links(self.output_dir)
        



def main():
    params = {"tile_id":"T31TGL",
              "date_start":"01/01/2019",
              "date_stop":"15/07/2019",
              "date_margin":15,
              "output_dir":"/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/S2",
              "nbThreads":5}

    prepare_data_for_snow_annual_map_app = prepare_data_for_snow_annual_map(params)
    prepare_data_for_snow_annual_map_app.run()



if __name__== "__main__":
    # Set logging level and format.
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=\
        '%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s')
    main()
