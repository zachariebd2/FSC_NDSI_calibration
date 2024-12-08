#!/bin/bash
#PBS -N do_datasets_images_job
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


nblines=$(cat $list_selected_lis_path | wc -l)

readarray -t array_products < $list_selected_lis_path
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
if [ -z $line ]; then
	echo "one or more FSC product was not selected in the selection step. End of job"
else
    IFS=',' read -ra params <<< "$line"
    FSC_path=${params[0]}
    LIS_products_paths=${params[@]: 1}



    echo "date = " $date
    echo "FSC_path = " $FSC_path

    dir=$(PWD)

    python ${dir}/sc_datasets_images.py   -FSC_path $FSC_path\
                                          -LIS_products_paths $LIS_products_paths\
                                          -output_tifs_path $output_tifs_path\
                                          -nodt $nodt\
                                          -nosnw $nosnw\
                                          -snw $snw\
                                          -epsg $epsg\
                                          -resample $resample
    
fi

