#!/bin/bash

path_fsc="/datalake/S2-L2B-COSIMS/data/Snow/FSC"
path_MAJA="/datalake/S2-L2A-MAJA/data"
tile="31TCH"

arrFSC=()

arrRED=()
arrGREEN=()
arrMIR=()
arrDateFSC=()
arrDateMAJA=()



for dir in $path_fsc/$tile/*/*/*; do
    fsc_dirs=($dir/*)
    fsc_dir="${fsc_dirs[0]}"
    arrFSC+=($fsc_dir/*FSCOG*)
    date=$(ls $fsc_dir/*FSCOG* | cut -d_ -f2 | cut -dT -f1)
    arrDateFSC+=($date)
done


for dir in $path_MAJA/$tile/*/*/*; do
    maja_dirs=($dir/*)
    maja_dir="${maja_dirs[0]}"
    arrRED+=($maja_dir/*FRE_B4*)
    arrGREEN+=($maja_dir/*FRE_B3*)
    arrMIR+=($maja_dir/*FRE_B11*)
    date=$(ls $maja_dir/*FRE_B4* | cut -d_ -f2 | cut -d- -f1)
    arrDateMAJA+=($date)
done


echo "FSC: " ${#arrFSC[@]}
echo "RED: " ${#arrRED[@]}
echo "GREEN: " ${#arrGREEN[@]}
echo "MIR: " ${#arrMIR[@]}
echo "DATE FSC : " ${#arrDateFSC[@]}
echo "DATE MAJA : " ${#arrDateMAJA[@]}

A=${arrDateFSC[@]};
B=${arrDateMAJA[@]};
if [ "$A" == "$B" ] ; then
    echo "Date arrays are the same" ;
fi

for i in $(seq 1 ${#arrDateFSC[@]}); do
    echo "${arrDateFSC[$i]}" "${arrDateMAJA[$i]}"
done

#echo ${arrDateFSC[*]}
