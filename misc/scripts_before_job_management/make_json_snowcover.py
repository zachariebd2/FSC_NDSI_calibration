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
tiles = []
epsg = []
resample = []
snw = []
nosnw = []
nodt = []
manual = []
selection = []
calibrate = []
eval_target = []
eval_cal = []
perc_cal = []
eval_modes = []
eval_res = []


source_dp = []
out_dp = []
filter_acc_dp = []
filter_snw_dp = []
do_filter_dp = []
eval_cal_dp = []
max_acc_dp = []
snw_type_dp = []
epsg_dp = []


#DATES
start_date = "2015-06-01"
end_date = "2021-07-01"
nb_shift_days = 4
#PATHS
path_outputs = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/OUTPUTS/"
path_inputs = "/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover/INPUTS/FSC/"
#path_LIS = "/work/OT/siaa/Theia/Neige/PRODUITS_NEIGE_LIS_develop_1.5/"
path_LIS = "/work/datalake/S2-L2B-COSIMS/data/Snow/FSC"
path_tree = "/work/OT/siaa/Theia/Neige/CoSIMS/data/tree_cover_density/original_tiling/TCD_2015_020m_eu_03035_d05_full.tif"
#PLEIADES
source.append("PLEIADES")
out.append("PLEIADES")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append(["31TCH"])
epsg.append("")
resample.append("average")
snw.append([1])
nosnw.append([2])
nodt.append([])
manual.append("")
selection.append("closest")
calibrate.append(False)
eval_target.append(True)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)

#PLEIADES2 
source.append("PLEIADES2")
out.append("FINSE")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append(['32VMN', '32VLN', '33WXR', '34WDA'])
epsg.append("")
resample.append("average")
snw.append([100])
nosnw.append([0,1,2])
nodt.append([-9999])
manual.append("")
selection.append("cleanest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["separate","all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)

# source.append("PLEIADES2")
# out.append("PLEIADES_ABINSKO_cleanest")
# make_set.append(True)
# plt_dates.append(True)
# plt_periode.append(True)
# quicklooks.append(True)
# tiles.append(['32VMN', '32VLN', '33WXR', '34WDA'])
# epsg.append("")
# resample.append("average")
# snw.append([100])
# nosnw.append([0,1,2])
# nodt.append([-9999])
# manual.append("")
# selection.append("cleanest")
# calibrate.append(False)
# eval_target.append(False)
# eval_cal.append(True)
# perc_cal.append(0.4)
# eval_modes.append(["separate","all"])  #"all" and/or "separate" and/or "average"
# eval_res.append(20)

#SPOT67
source.append("SPOT67")
out.append("SPOT67")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append(["32TLS","32TLR","32TLQ","31TGK","31TGL","31TGM"])
epsg.append("")
resample.append("average")
snw.append([2])
nosnw.append([1])
nodt.append([0])
manual.append("")
selection.append("cleanest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all","separate","average"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
#IZAS
source.append("IZAS")
out.append("IZAS")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append(["30TYN"])
epsg.append("25830")
resample.append("average")
snw.append([1])
nosnw.append([0])
nodt.append([])
manual.append("")
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
#DISCHMEX
source.append("DISCHMEX")
out.append("DISCHMEX")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append([])
epsg.append("21781")
resample.append("average")
snw.append([1])
nosnw.append([0])
nodt.append([])
manual.append("")
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
#WEISSSEE
source.append("WEISSSEE")
out.append("WEISSSEE")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append([])
epsg.append("31254")
resample.append("average")
snw.append([1])
nosnw.append([0])
nodt.append([])
manual.append("")
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
#CAMSNOW
source.append("CAMSNOW")
out.append("CAMSNOW")
make_set.append(False)
plt_dates.append(False)
plt_periode.append(False)
quicklooks.append(False)
tiles.append([])
epsg.append("2154")
resample.append("near")
snw.append([])
nosnw.append([])
nodt.append([-99999])
manual.append("")
selection.append("closest")
calibrate.append(False)
eval_target.append(False)
eval_cal.append(False)
perc_cal.append(0.4)
eval_modes.append(["all"])  #"all" and/or "separate" and/or "average"
eval_res.append(20)
#DATA POINTS
#ODK
source_dp.append("ODK")
out_dp.append("ODK")
filter_acc_dp.append(True)
filter_snw_dp.append(True)
do_filter_dp.append(False)
eval_cal_dp.append(False)
epsg_dp.append("4326")
max_acc_dp.append(5)
snw_type_dp.append("fsc")

#CSO
source_dp.append("CSO")
out_dp.append("CSO")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
do_filter_dp.append(False)
eval_cal_dp.append(True)
max_acc_dp.append(5)
snw_type_dp.append("sca")
epsg_dp.append("4326")

#METEOFRANCE
source_dp.append("METEOFRANCE")
out_dp.append("METEOFRANCE")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
do_filter_dp.append(False)
eval_cal_dp.append(True)
max_acc_dp.append(5)
snw_type_dp.append("sca")
epsg_dp.append("4326")

# mondefrance
source_dp.append("mondefrance")
out_dp.append("mondefrance")
filter_acc_dp.append(False)
filter_snw_dp.append(False)
do_filter_dp.append(False)
eval_cal_dp.append(False)
max_acc_dp.append(5)
snw_type_dp.append("sca")
epsg_dp.append("4326")

############################################################################################
#JSON

jsonFile = open(json_path, "w+")
data = {}
##DATES
data["dates"] = {"start_date" : start_date,"end_date" : end_date,"nb_shift_days" : nb_shift_days}

##PATHS
data["paths"] = {"path_outputs" : path_outputs,"path_inputs" : path_inputs,"path_LIS" : path_LIS,"path_tree" : path_tree}

##SETS
data["sets"] = []
data["DP"] = []
for i in arange(len(source)):
    data["sets"].append({"source" : source[i],"out" : out[i],"tiles" : tiles[i],"epsg" : epsg[i],"resample" : resample[i], "snw" : snw[i], "nosnw" : nosnw[i] , "nodt" : nodt[i], "manual" : manual[i], "selection" : selection[i],"make_set" : make_set[i], "plt_dates" : plt_dates[i], "plt_periode" : plt_periode[i], "quicklooks" : quicklooks[i], "calibrate" : calibrate[i], "eval_target" : eval_target[i], "eval_cal" : eval_cal[i], "perc_cal" : perc_cal[i], "eval_modes" : eval_modes[i], "eval_res" : eval_res[i]})

##DP
for j in arange(len(source_dp)):
    data["DP"].append({"source" : source_dp[j],"out" : out_dp[j], "do_filter" : do_filter_dp[j],"filter_acc" : filter_acc_dp[j],"filter_snw" : filter_snw_dp[j],"eval_cal" : eval_cal_dp[j], "max_acc" : max_acc_dp[j],"snw_type" : snw_type_dp[j],"epsg" : epsg_dp[j]})
jsonFile.write(json.dumps(data, indent=3))
##WRITE JSON



jsonFile.close()


















