#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:26:44 2020

@author: zacharie
"""
import json
from numpy import arange



json_path = 'snowcover_params.json'

source = []
out = []
make_set = []
plt_dates = []
plt_periode = []
quicklooks = []
epsg = []
resample = []
snw = []
nosnw = []
nodt = []
selection = []
calibrate = []
eval_target = []
eval_cal = []
perc_cal = []
eval_modes = []
eval_res = []
max_J = []





source_dp = []
out_dp = []
filter_acc_dp = []
filter_snw_dp = []
make_set_dp = []
eval_cal_dp = []
max_acc_dp = []
snw_type_dp = []
epsg_dp = []
max_J_dp = []

#DATES
start_date = "2017-09-01"
end_date = "2018-08-31"
nb_shift_days = 0
#PATHS
path_outputs = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/"
path_inputs = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/FSC/"
path_DEM = "/work/OT/siaa/Theia/Neige/DEM"
path_countries = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/DATA/countries"
path_landcover = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/DATA/landcover/PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif"
#path_LIS = "/work/OT/siaa/Theia/Neige/PRODUITS_NEIGE_LIS_develop_1.5/"
path_LIS = "/work/datalake/S2-L2B-COSIMS/data/Snow/FSC"
path_MAJA = "/work/datalake/S2-L2A-MAJA/data"
path_tree = "/work/OT/siaa/Theia/Neige/CoSIMS/data/tree_cover_density/original_tiling/TCD_2015_020m_eu_03035_d05_full.tif"
#PLEIADES
source.append("PLEIADES")
out.append("PLEIADES")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("")
resample.append("average")
snw.append([1])
nosnw.append([2])
nodt.append([])
selection.append("closest")
calibrate.append(False)
eval_target.append(True)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)


#PLEIADES2 
source.append("PLEIADES2")
out.append("FINSE")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("")
resample.append("average")
snw.append([100])
nosnw.append([0,1,2])
nodt.append([-9999])
selection.append("cleanest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["separate","all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)


#SPOT67
source.append("SPOT67")
out.append("SPOT67")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("")
resample.append("average")
snw.append([2])
nosnw.append([1])
nodt.append([0])
selection.append("cleanest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all","separate","average"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)

#IZAS
source.append("IZAS")
out.append("IZAS")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("25830")
resample.append("average")
snw.append([1])
nosnw.append([0])
nodt.append([])
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)

#DISCHMEX
source.append("DISCHMEX")
out.append("DISCHMEX")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("21781")
resample.append("average")
snw.append([1])
nosnw.append([0])
nodt.append([])
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)

#WEISSSEE
source.append("WEISSSEE")
out.append("WEISSSEE")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("31254")
resample.append("average")
snw.append([1])
nosnw.append([0])
nodt.append([])
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)

#CAMSNOW
source.append("CAMSNOW")
out.append("CAMSNOW")
make_set.append([0])
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
epsg.append("2154")
resample.append("near")
snw.append([])
nosnw.append([])
nodt.append([-99999])
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
max_J.append(50)

#DATA POINTS
#ODK
source_dp.append("ODK")
out_dp.append("ODK")
filter_acc_dp.append(True)
filter_snw_dp.append(True)
make_set_dp.append([0])
eval_cal_dp.append(False)
epsg_dp.append("4326")
max_acc_dp.append(5)
snw_type_dp.append("fsc")
max_J_dp.append(50)

#CSO
source_dp.append("CSO")
out_dp.append("CSO")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
make_set_dp.append([])
eval_cal_dp.append(False)
max_acc_dp.append(5)
snw_type_dp.append("depth")
epsg_dp.append("4326")
max_J_dp.append(2000)

#METEOFRANCE
source_dp.append("METEOFRANCE")
out_dp.append("METEOFRANCE")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
make_set_dp.append([])
eval_cal_dp.append(False)
max_acc_dp.append(5)
snw_type_dp.append("depth")
epsg_dp.append("4326")
max_J_dp.append(1000)

#mondefrance
source_dp.append("SYNOP")
out_dp.append("SYNOP")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
make_set_dp.append([])
eval_cal_dp.append(False)
max_acc_dp.append(5)
snw_type_dp.append("depth")
epsg_dp.append("4326")
max_J_dp.append(2000)

#alpe
source_dp.append("BDALPS")
out_dp.append("BDALPS")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
make_set_dp.append([])
eval_cal_dp.append(False)
max_acc_dp.append(5)
snw_type_dp.append("depth")
epsg_dp.append("4326")
max_J_dp.append(2000)


#bd_alpes_monde
source_dp.append("BD_SYNOP_ALPS")
out_dp.append("BD_SYNOP_ALPS")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
make_set_dp.append([])
eval_cal_dp.append(True)
max_acc_dp.append(5)
snw_type_dp.append("depth")
epsg_dp.append("4326")
max_J_dp.append(2000)

# ###########################################################################################
# JSON

jsonFile = open(json_path, "w+")
data = {}
##DATES
data["dates"] = {"start_date" : start_date,"end_date" : end_date,"nb_shift_days" : nb_shift_days}

##PATHS
data["paths"] = {"path_outputs" : path_outputs,"path_inputs" : path_inputs,"path_LIS" : path_LIS,"path_tree" : path_tree,"path_DEM" : path_DEM,"path_countries" : path_countries,"path_landcover" : path_landcover,"path_MAJA" : path_MAJA}

##SETS
data["sets"] = []
data["DP"] = []
for i in arange(len(source)):
    data["sets"].append({"source" : source[i],"out" : out[i],"max_J" : max_J[i],"epsg" : epsg[i],"resample" : resample[i], "snw" : snw[i], "nosnw" : nosnw[i] , "nodt" : nodt[i], "selection" : selection[i],"make_set" : make_set[i], "plt_dates" : plt_dates[i], "plt_periode" : plt_periode[i], "quicklooks" : quicklooks[i], "calibrate" : calibrate[i], "eval_target" : eval_target[i], "eval_cal" : eval_cal[i], "perc_cal" : perc_cal[i], "eval_modes" : eval_modes[i], "eval_res" : eval_res[i]})

##DP
for j in arange(len(source_dp)):
    data["DP"].append({"source" : source_dp[j],"out" : out_dp[j], "make_set" : make_set_dp[j],"max_J" : max_J_dp[j],"filter_acc" : filter_acc_dp[j],"filter_snw" : filter_snw_dp[j],"eval_cal" : eval_cal_dp[j], "max_acc" : max_acc_dp[j],"snw_type" : snw_type_dp[j],"epsg" : epsg_dp[j]})
jsonFile.write(json.dumps(data, indent=3))
##WRITE JSON



jsonFile.close()

















