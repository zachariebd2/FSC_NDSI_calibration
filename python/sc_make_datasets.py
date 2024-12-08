#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:08:25 2020

@author: zacharie
"""


import sys
import os
import errno
import re
import sc_utils
import sc_parse_inputs
import sc_parse_input_points
import sc_inputs_coverage
import sc_job_organiser
import csv
import shutil
import sc_datasets_points










def makeDataSetsFromImages(d,p,s):
    
    source = s["source"]
    out = s["out"]
    epsg = str(s["epsg"])
    resample = s["resample"]
    selection = s["selection"]
    snw = s["snw"]
    nosnw = s["nosnw"]
    nodt = s["nodt"]
    start_date = d["start_date"]
    end_date = d["end_date"]
    path_outputs = p["path_outputs"]
    path_inputs = p["path_inputs"]
    path_LIS = p["path_LIS"]
    nb_shift_days = str(d["nb_shift_days"])
    max_J = s["max_J"]
    steps = s["make_set"]
    print("making datasets from: ",source," to: ",out)
    print("steps: ",steps)
    #text file paths
    list_fsc_path = os.path.join(path_outputs,out,"list_fsc_products.csv")
    list_available_tiles_path = os.path.join(path_outputs,out,"list_available_tiles.csv")
    list_inputs_coverage_path = os.path.join(path_outputs,out,"list_fsc_coverage.csv")
    list_overlapping_tiles_path = os.path.join(path_outputs,out,"list_overlapping_tiles.csv")
    list_selected_lis_path = os.path.join(path_outputs,out,"list_selected_lis.csv")
    dict_datasets_path = os.path.join(path_outputs,out,"dict_datasets.json")

    sc_utils.mkdir_p(os.path.join(path_outputs,out))
    job_id = ""
    
    if 1 in steps:
        print("step 1:")
        #step 1: get list of FSC scenes and of available S2 tiles
        print("  parsing input FSC")
        sc_parse_inputs.searchFSC(start_date,end_date,source,path_inputs,list_fsc_path)
        print("  parsing tiles")
        sc_parse_inputs.searchAvailableTiles(path_LIS,out,path_outputs,list_available_tiles_path)

    
    if 2 in steps:
        print("step 2:")
        #step 2: get list of different FSC coverages and search tiles overlapping with FSC products
        print("  getting list of FSC coverages")
        sc_inputs_coverage.getListImagesCoverage(list_fsc_path,list_inputs_coverage_path, epsg)
    
        #search tiles overlapping with FSC products
        print("  finding overlapping tiles")
        list_overlapping_tiles_file = open(list_overlapping_tiles_path,'w')
        writer = csv.writer(list_overlapping_tiles_file)
        writer.writerow(["tile","path"])
        list_overlapping_tiles_file.close()
        nb_jobs = len(open(list_available_tiles_path).readlines(  )) - 1
        print("  number of jobs: ",nb_jobs)
        sh_script = "sc_overlapping_tiles.sh"
        precedent_job_id = job_id
        params = "list_overlapping_tiles_path=\""+list_overlapping_tiles_path+"\",list_inputs_coverage_path=\""+list_inputs_coverage_path+"\",list_available_tiles_path=\""+list_available_tiles_path+"\",path_LIS=\""+path_LIS+"\",type_data=\""+"image"+"\""
        job_id = sc_job_organiser.makeJobs(nb_jobs,max_J,params,sh_script,precedent_job_id)
        print("  job id: ",job_id)
        

    if 3 in steps:
        print("step 3:")
        print("  selecting LIS products")
        #step 3: select LIS products corresponding to the fsc products
        list_selected_lis_file = open(list_selected_lis_path,'w')
        writer = csv.writer(list_selected_lis_file)
        writer.writerow(["FSC","LIS_products"])
        list_selected_lis_file.close()
        nb_jobs = len(open(list_fsc_path).readlines(  )) - 1
        print("  number of jobs: ",nb_jobs)
        sh_script = "sc_select_products.sh"
        precedent_job_id = job_id
        params = "list_overlapping_tiles_path=\""+list_overlapping_tiles_path+"\",list_fsc_path=\""+list_fsc_path+"\",nb_shift_days=\""+nb_shift_days+"\",list_selected_lis_path=\""+list_selected_lis_path+"\",selection=\""+selection+"\",epsg=\""+epsg+"\""
        job_id = sc_job_organiser.makeJobs(nb_jobs,max_J,params,sh_script,precedent_job_id)
        print("  job id: ",job_id)

    if 4 in steps:
        print("step 4:")
        print("  making FSC/NDSI datasets")
        #step 4: make NDSI and FSC datasets from input products
        output_tifs_path = os.path.join(path_outputs,out,"TIFS")
        shutil.rmtree(output_tifs_path,ignore_errors=True)
        sc_utils.mkdir_p(output_tifs_path)
        nb_jobs = len(open(list_fsc_path).readlines(  )) - 1
        print("  number of jobs: ",nb_jobs)
        sh_script = "sc_datasets_images.sh"
        precedent_job_id = job_id
        params = "resample=\""+resample+"\",snw=\""+str(snw)+"\",nosnw=\""+str(nosnw)+"\",nodt=\""+str(nodt)+"\",list_selected_lis_path=\""+list_selected_lis_path+"\",output_tifs_path=\""+output_tifs_path+"\",epsg=\""+epsg+"\""
        job_id = sc_job_organiser.makeJobs(nb_jobs,max_J,params,sh_script,precedent_job_id)
        print("  job id: ",job_id)

    return job_id




       
def makeDatasetsFromPoints(DP,p,d):
    source = DP["source"]
    out = DP["out"]
    epsg = str(DP["epsg"])
    path_outputs = p["path_outputs"]
    path_inputs = p["path_inputs"]
    path_LIS = p["path_LIS"]
    path_DEM = p["path_DEM"]
    path_MAJA = p["path_MAJA"]
    start_date = d["start_date"]
    end_date = d["end_date"]
    nb_shift_days = d["nb_shift_days"]
    path_tree = p["path_tree"]
    path_countries = p["path_countries"]
    path_landcover = p["path_landcover"]
    snw_type = DP["snw_type"]
    max_J = DP["max_J"]
    steps = DP["make_set"]


    #text file paths
    datasets_path = os.path.join(path_outputs,out,"datasets")
    list_input_points_path = os.path.join(datasets_path,"list_input_points.csv")
    list_available_tiles_path = os.path.join(datasets_path,"list_available_tiles.csv")
    list_restricted_tiles_path = os.path.join(datasets_path,"list_restricted_tiles.txt")
    list_input_coordinates_path = os.path.join(datasets_path,"list_input_coordinates.csv")
    matched_points_tiles_dir = os.path.join(datasets_path,"matched_points_tiles")
    list_points_datasets_path = os.path.join(datasets_path,"list_points_datasets.csv")
    job_id = ""
    
    sc_utils.mkdir_p(datasets_path)
    
    if 1 in steps:

        sc_parse_input_points.searchPoints(start_date,end_date,source,path_inputs,list_input_points_path)
        sc_parse_input_points.searchAvailableTiles(path_LIS,out,path_outputs,list_available_tiles_path,list_restricted_tiles_path)
        sc_parse_input_points.getListCoordinates(list_input_points_path,list_input_coordinates_path,path_countries,epsg)
        
    
    if 2 in steps:
        shutil.rmtree(matched_points_tiles_dir, ignore_errors=True)
        nb_jobs = len(open(list_available_tiles_path).readlines( )) - 1
        
        sh_script = "sc_match_tiles_points.sh"
        precedent_job_id = job_id
        params = "matched_points_tiles_dir=\""+matched_points_tiles_dir+"\",list_input_coordinates_path=\""+list_input_coordinates_path+"\",list_available_tiles_path=\""+list_available_tiles_path+"\",path_LIS=\""+path_LIS+"\",path_tree=\""+path_tree+"\",path_DEM=\""+path_DEM+"\",path_landcover=\""+path_landcover+"\",path_MAJA=\""+path_MAJA+"\""
        job_id = sc_job_organiser.makeJobs(nb_jobs,max_J,params,sh_script,precedent_job_id)
        
    if 3 in steps:
        
        nb_jobs = 1
        sh_script = "sc_match_fsc_points.sh"
        precedent_job_id = job_id
        params = "matched_points_tiles_dir=\""+matched_points_tiles_dir+"\",list_input_points_path=\""+list_input_points_path+"\",list_points_datasets_path=\""+list_points_datasets_path+"\",snw_type=\""+snw_type+"\",nb_shift_days=\""+str(nb_shift_days)+"\""
        job_id = sc_job_organiser.makeJobs(nb_jobs,max_J,params,sh_script,precedent_job_id)
        
        
    return job_id
