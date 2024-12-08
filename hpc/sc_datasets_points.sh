#!/bin/bash
#PBS -N do_datasets_points_job
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=200mb
#PBS -l walltime=00:05:00
#PBS -m e



# Store the directory in  PBS_TMP_DIR

PBS_TMPDIR=$TMPDIR/job_${PBS_ARRAY_INDEX}
# Unset TMPDIR env variable which conflicts with openmpi
unset TMPDIR

##module load lis/1.6


dirsnow=/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover
#dos2unix $list_input_points_path
list_input_points_path=$(echo "$list_input_points_path" | tr -d '\r')
#dos2unix $list_overlapping_tiles_path
list_overlapping_tiles_path=$(echo "$list_overlapping_tiles_path" | tr -d '\r')

# get lists content as arrays: point and tiles
readarray -t array_points < $list_input_points_path
readarray -t array_tiles < $list_overlapping_tiles_path


#get the point values corresponding to the job
# use the PBS_ARRAY_INDEX variable to distribute jobs in parallel (bash indexing is zero-based)
line=""
echo "PBS_ARRAY_INDEX IS " ${PBS_ARRAY_INDEX}
if [ -z $PBS_ARRAY_INDEX ]; then
    line="${array_points[1]}"
else
    line="${array_points[${PBS_ARRAY_INDEX}]}"
fi
#line="${array_products[0]}"
#echo "line = " $line
IFS=',' read -ra params <<< "$line"
datep=${params[0]}
lat=${params[1]}
lon=${params[2]}
value=${params[3]}
acc=${params[4]}

#for each tile,check if overlap and find the product 
#with the smallest the time gap and a clean pixel


final_decal=$(($nb_shift_days + 1))
final_product=""
final_dateLIS=""
final_tile=""
final_NDSI=0
final_TOC=0
final_OG=0
final_tree=0

for tile_line in "${array_tiles[@]:1}"
do 
   
    #get tile path
    IFS=',' read -ra tile_params <<< "$tile_line"
    tile_name=${tile_params[0]}
    #echo tile $tile_name
    tile_path=${tile_params[1]}
    #currentDate=`date`
    #echo start of overlap $currentDate
   
    #check if overlap
    #mtd_path=$(find $tile_path -type f -name *MTD.xml -print -quit)
    ndsi_path=$(find $tile_path -type f -name *NDSI* -print -quit)
    NDSI=$(gdallocationinfo -valonly -wgs84 $ndsi_path $lon $lat)
    #echo "NDSI=" $NDSI
    #currentDate=`date`
    #echo end of overlap $currentDate
    if ! [[ "$NDSI" =~ ^[0-9]+$ ]];then
        #echo no overlap
        continue
    fi
   
    #set output file name for list of valid products
    valid_products_path=$PBS_TMPDIR/$tile_name/valid_products.csv
    mkdir -p $PBS_TMPDIR/$tile_name
    #currentDate=`date`
    #echo beforepython $currentDate
    #get products inside time gap and write in valid_products.csv
    python ${dirsnow}/sc_datasets_points.py    -nb_shift_days $nb_shift_days\
                                               -valid_products_path $valid_products_path\
                                               -tile_path $tile_path\
                                               -date $datep
    #currentDate=`date`
    #echo afterpython $currentDate
    #for each valid product (already sorted), check the pixel 
    #and keep the first good product
    #valid_products_path=$(echo "$valid_products_path" | tr -d '\r')
    dos2unix $valid_products_path
    readarray -t array_products < $valid_products_path
    for product_line in "${array_products[@]:1}"
    do 
        IFS=',' read -ra product_params <<< "$product_line"
        product_date=${product_params[0]}
        product_decal=${product_params[1]}
        product_path=${product_params[2]}
        #echo p_date $product_date
        #echo p_decal $product_decal
        #echo p_path $product_path
        NDSI_path=$(find $product_path -type f -name *NDSI*)
        NDSI_name=$(basename ${NDSI_path})
        #echo n_path $NDSI_path
        #echo n_name $NDSI_name
        #tmp_NDSI_path=$PBS_TMPDIR/$tile_name/$NDSI_name
        #cp $NDSI_path $tmp_NDSI_path
        NDSI=$(gdallocationinfo -valonly -wgs84 $NDSI_path $lon $lat)
        #NDSI="|${NDSI//[$'\t\r\n']}|"
        #echo "NDSI = " $NDSI
        if (($NDSI >= 0 && $NDSI <= 100)); then
            if (($product_decal < $final_decal)); then 
                TOC_path=$(find $product_path -type f -name *FSCTOC*)
                OG_path=$(find $product_path -type f -name *FSCOG*)
                TOC=$(gdallocationinfo -valonly -wgs84 $TOC_path $lon $lat)
                #TOC="|${TOC//[$'\t\r\n']}|"
                OG=$(gdallocationinfo -valonly -wgs84 $OG_path $lon $lat)
                #OG="|${OG//[$'\t\r\n']}|"
                tree=$(gdallocationinfo -valonly -wgs84 $path_tree $lon $lat)
                #tree="|${tree//[$'\t\r\n']}|"
                final_decal=$product_decal
                final_product=$product_path
                final_tile=$tile_name
                final_dateLIS=$product_date
                final_NDSI=$NDSI
                final_TOC=$TOC
                final_OG=$OG
                final_tree=$tree
                echo "$datep,$lat,$lon,$value,$acc,$final_decal,$final_product,$final_tile,$final_tree,$final_TOC,$final_OG,$final_NDSI"
                if (($product_decal == 0)); then 
                    rm -rf $PBS_TMPDIR/$tile_name
                    break 2
                fi
            fi
        fi

    done
    rm -rf $PBS_TMPDIR/$tile_name
   
done



if [ -n "$final_product" ];then
    echo "$datep,$lat,$lon,$value,$acc,$final_decal,$final_product,$final_tile,$final_tree,$final_TOC,$final_OG,$final_NDSI">> $list_points_datasets_path
fi

