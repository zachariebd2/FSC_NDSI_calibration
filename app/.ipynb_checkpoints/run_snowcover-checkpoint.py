#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:26:44 2020

@author: zacharie
"""

import sys
import os
import errno
import re
import json


import sc_make_datasets as smd
import sc_analyse_datasets as sad
import sc_model as sm
import sc_utils 













def process_data():
    
    j_data = open("/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/parameters/sc_parameters.json")
    data = json.load(j_data)
    d = data["dates"]
    p = data["paths"]
    
    
    
    #MAKE DATASETS########################################################################
    for s in data["sets"]:
        if 0 not in s["make_set"] and len(s["make_set"]) != 0 :
            print("making datasets for "+s["out"])
            job_id = smd.makeDataSetsFromImages(d,p,s)
            

    
    for DP in data["DP"]:
        if 0 not in DP["make_set"] and len(DP["make_set"]) != 0 :
            print("making datasets for "+DP["out"])
            job_id = smd.makeDatasetsFromPoints(DP,p,d)
            
    #DATASETS ANALYSIS####################################################################
    for s in data["sets"]:
        if s["plt_dates"] :
            print("plotting each dates for "+s["out"])
            out = s["out"]
            sad.analyse_dates(p,out)
        if s["plt_periode"] :
            print("plotting period for "+s["out"])
            out = s["out"]
            sad.analyse_period(p,out)
        if s["quicklooks"] :
            print("making quicklooks for "+s["out"])
            out = s["out"]
            sad.make_quicklooks(p,out)
    
    #CALIBRATION########################################################################
    for s in data["sets"]:
        if s["calibrate"] :
            print("calibrating from "+s["out"])
            source = s["source"]
            out = s["out"]
            perc_cal = s["perc_cal"]
            sm.calibration(p,out,perc_cal)
    
    
    
    #EVALUATION#########################################################################
    for c in data["sets"]:
        if c["eval_target"] :
            out_cal = c["out"]
            for e in data["sets"]:
                out_val = e["out"]
                if e["eval_cal"] and (out_cal != out_val):
                    print("evaluating "+out_cal+" with "+out_val)
                    eval_modes = e["eval_modes"]
                    eval_res = e["eval_res"]
                    sm.evaluation(p,out_cal,out_val,eval_modes,eval_res)
                # if e["time_lapse"] and (out_cal != out_val):
                    # print("timelapse evaluation of "+out_cal+" with "+out_val)
                    # timeLapseEval(out_cal = out_cal,
                                      # out_val = out_val)
    
            for DP in data["DP"]:
                if DP["eval_cal"] :
                    out_val = DP["out"]
                    snw_type = DP["snw_type"]
                    print("evaluating "+out_cal+" with "+out_val)
                    if snw_type == "fsc": sm.evalFSCWithDP(out_cal,out_val,p)
                    if snw_type == "depth": sm.evalSCAWithDP(out_cal,out_val,p)
                


def process_params():
    
    #we check if there is a file as input argument
    if len(sys.argv) != 2 :
        print("run_snowcover only accepts a parameter json file as input (1 argument)")
        exit
    elif not os.path.isfile(sys.argv[1]):
        print("Input parameter file",sys.argv[1],"not found")
        exit
    
    
    #we load the input file and the parameter file
    j_inputs = open(sys.argv[1])
    inputs = json.load(j_inputs)
    j_data = open("/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/parameters/sc_parameters.json","w+")
    data = inputs
    
    #CHECK PARAMETERS##################################################################################
    
    start_date = inputs["dates"]["start_date"]
    end_date = inputs["dates"]["end_date"]
    path_outputs = inputs["paths"]["path_outputs"]
    path_inputs = inputs["paths"]["path_inputs"]
    path_LIS = inputs["paths"]["path_LIS"]
    path_tree = inputs["paths"]["path_tree"]
    
    print("START DATE :",start_date)
    print("END DATE :",end_date)
    print("INPUTS PATH :",path_inputs)
    print("OUTPUTS PATH :",path_outputs)
    print("LIS PATH :",path_LIS)
    print("TREE PATH :",path_tree)
    
    if end_date == "": 
        end_date = start_date
    if sc_utils.getDateFromStr(start_date) == '' or sc_utils.getDateFromStr(end_date) == '':
        sys.exit("ERROR snowcover : error in input dates")
    if not os.path.isdir(path_inputs) :
        sys.exit("ERROR snowcover : " + path_inputs + " not a directory")
    if not os.path.isdir(path_LIS) :
        sys.exit("ERROR snowcover : " + path_LIS + " not a directory")
    if not os.path.isfile(path_tree) :
        sys.exit("ERROR snowcover : " + path_tree + " not a file")    
        
    for s in inputs["sets"]:
        print("FSC SOURCE DIRECTORY:",s["source"])
        print("    Output directory :",s["out"])
        print("    Make FSC/NDSI datasets; starting step : ", s["make_set"])
        print("    Target of evaluation : ", s["eval_target"])
        print("    Used for calibration : ", s["calibrate"])
        print("    Used for evaluation : ", s["eval_cal"])
        print("    Evaluation modes : ", s["eval_modes"])
        print("    Evaluation resolutions : ", s["eval_res"])
        print("    Plot each date : ", s["plt_dates"])
        print("    Plot periode : ", s["plt_periode"])
        print("    Make quicklook : ", s["quicklooks"])
        p_source = os.path.join(path_inputs,s["source"])
        if not os.path.isdir(p_source) :
            sys.exit("ERROR snowcover : " + p_source + " not a directory")
        
        
    for DP in inputs["DP"]:
        print("FSC SOURCE DIRECTORY:",DP["source"])
        print("    Output directory :",DP["out"])
        print("    Apply accuracy filter : ", DP["filter_acc"])
        print("    Apply snow mask filter : ", DP["filter_snw"])
        print("    Make FSC/NDSI datasets; starting step : ", DP["make_set"])
        print("    Used for evaluation : ", DP["eval_cal"])
        p_source = os.path.join(path_inputs,DP["source"])
        if not os.path.isdir(p_source) :
            sys.exit("ERROR snowcover : " + p_source + " not a directory")
    
    
    
    data["dates"]["end_date"] = end_date

    j_data.write(json.dumps(data, indent=3))
    
    
    j_inputs.close()
    j_data.close()




def main():
    
    process_params()
    process_data()
    
    
    print("end of process")
    
    
    




if __name__ == '__main__':
    main()



