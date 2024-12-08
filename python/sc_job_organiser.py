
import sys
import os
import errno
import re
import copy
from datetime import datetime, timedelta, date
import glob
import sc_utils
import argparse
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
import logging
import subprocess


def call_subprocess(process_list):
    """ Run subprocess and write to stdout and stderr
    """
    logging.info("Running: " + " ".join(process_list))
    process = subprocess.Popen(
        process_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = process.communicate()
    
    logging.info(out)
    sys.stderr.write(str(err))
    return out




def makeJobs(nb_jobs,max_J,params,sh_script,precedent_job_id):
    
    print("organizing job arrays:")
    
    print("   nb of jobs to do =",nb_jobs)
    if nb_jobs  == 0:
        print("    no jobs to do, end of process")
        exit()
    elif nb_jobs == 1:
        print("    only one job to do")
    else:
        print("    jobs will run in groups of ",max_J)
        
    
    
    nb_job_array = float(nb_jobs)/float(max_J)
    int_jobs = int(nb_job_array // 1) + 1
    job_id = precedent_job_id
    command_A = ["qsub",
                       "-V",
                       "-j",
                       "oe",
                       "-o",
                       "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/logs",
                       "-e",
                       "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/logs",
                       "-v",
                       params]
                       
                       
    
    
    print("starting sending jobs")
    for i in range(0,int_jobs):
        command_Out = []
        command_J = []
        
        if job_id == "" : command_Out = [sh_script]
        else : command_Out = ["-W", "depend=afterany:"+job_id ,sh_script]
           
           
        command_J = command_A + command_Out
        
        
        if nb_jobs == 1:
            print("sending one job")
        else :
            job_array = ""
            a = 1 + (i)*max_J
            b = (i+1)*max_J
            if b > nb_jobs : job_array = str(a)+"-"+str(nb_jobs)
            else: job_array = str(a)+"-"+str(b)
            print("sending jobs ", job_array)
            command_J.insert(1, "-J")
            command_J.insert(2, job_array )
        
        #print(" ".join(command_J))
        job_id = call_subprocess(command_J)
        job_id = str( job_id ).split("\'")[1].split(".")[0]
        
    return job_id



