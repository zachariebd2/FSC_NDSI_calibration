import os
import sys
import os.path as op
import json
import csv
import copy
import logging
import subprocess
import time
from datetime import datetime, timedelta
from libamalthee import Amalthee
from dateutil import relativedelta

amalthee = Amalthee('theia')

amalthee.show_collections()
amalthee.show_params("Snow")
parameters = {"location": "T32TNS", "platform":"SENTINEL2A"}


amalthee.search("SENTINEL2","2019-01-01","2019-07-14",parameters,nthreads = 2)

print("\n " + str(amalthee.products))

amalthee.check_datalake()

print("\n " + str(amalthee.products))

amalthee.fill_datalake()





time.sleep(600)

amalthee.check_datalake()

print("\n " + str(amalthee.products))

amalthee.create_links("./swiss_data")




