#!/bin/bash
#PBS -N do_datasets_points_job
#PBS -j oe
#PBS -l select=1:ncpus=4:mem=20000mb
#PBS -l walltime=00:59:00
#PBS -m e
#PBS -k n


# Store the directory in  PBS_TMP_DIR

PBS_TMPDIR=$TMPDIR
# Unset TMPDIR env variable which conflicts with openmpi
unset TMPDIR

##module load lis/1.6




readarray -t array_products < $list_input_points_path
# use the PBS_ARRAY_INDEX variable to distribute jobs in parallel (bash indexing is zero-based)
line=""
echo "PBS_ARRAY_INDEX IS " ${PBS_ARRAY_INDEX}
if [ -z $PBS_ARRAY_INDEX ]; then
    line="${array_products[1]}"
else
    line="${array_products[${PBS_ARRAY_INDEX}]}"
fi

#line="${array_products[0]}"
echo "line = " $line
IFS=',' read -ra params <<< "$line"
date=${params[0]}
lat=${params[1]}
lon=${params[2]}
value=${params[3]}
acc=${params[4]}




echo "date = " $date
echo "FSC_path = " $FSC_path

dir=$(PWD)

python ${dir}/sc_datasets_points.py   -nb_shift_days $nb_shift_days\
                                      -lat $lat\
                                      -lon $lon\
                                      -value $value\
                                      -acc $acc\
                                      -list_overlapping_tiles_path $list_overlapping_tiles_path\
                                      -list_points_datasets_path $list_points_datasets_path\
                                      -date $date\
                                      -path_tree $path_tree
