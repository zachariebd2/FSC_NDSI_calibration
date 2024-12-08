#!/bin/bash
#PBS -N do_match_tiles_job
#PBS -l select=1:ncpus=4:mem=20000mb
#PBS -l walltime=03:59:00





# Store the directory in  PBS_TMP_DIR
#P#B#S -j oe
PBS_TMPDIR=$TMPDIR
# Unset TMPDIR env variable which conflicts with openmpi
unset TMPDIR




dos2unix $list_available_tiles_path
readarray -t array_products < $list_available_tiles_path
# use the PBS_ARRAY_INDEX variable to distribute jobs in parallel (bash indexing is zero-based)
line=""
echo "PBS_ARRAY_INDEX IS " ${PBS_ARRAY_INDEX}
if [ -z $PBS_ARRAY_INDEX ]; then
    line="${array_products[1]}"
else
    line="${array_products[${PBS_ARRAY_INDEX}]}"
fi

echo "line = " $line
IFS=',' read -ra params <<< "$line"
tile=${params[0]}
LIS_product_path=${params[1]}
test_path=$(find $LIS_product_path -type f -name *QCTOC*)
test_name=$(basename ${test_path})

TEMP_JOB_DIR=$PBS_TMPDIR/${tile}
echo "TEMP_JOB_DIR:" $TEMP_JOB_DIR
mkdir $TEMP_JOB_DIR



cp $LIS_product_path/$test_name $TEMP_JOB_DIR/$test_name

TEMP_TEST_PATH=$TEMP_JOB_DIR/$test_name


#dem path
dem_path=$(find $path_DEM -type f -name *$tile*ALT_R2.TIF)
#slope path
slope_path=$(find $path_DEM -type f -name *$tile*SLP_R2.TIF)
#land cover path
landcover_path=$path_landcover

#vrt path
fsc_vrt_path=$TEMP_JOB_DIR/${tile}_FSCOG.vrt
qcf_vrt_path=$TEMP_JOB_DIR/${tile}_QCFLAGS.vrt
mir_vrt_path=$TEMP_JOB_DIR/${tile}_mir.vrt
red_vrt_path=$TEMP_JOB_DIR/${tile}_red.vrt
green_vrt_path=$TEMP_JOB_DIR/${tile}_green.vrt
#temp result path
fsc_csv_path=$TEMP_JOB_DIR/${tile}_FSCOG.csv
maja_csv_path=$TEMP_JOB_DIR/${tile}_MAJA.csv

eu_dem_path="/work/OT/siaa/Theia/Neige/CoSIMS/data/EU-DEM/original_tiling/10m"

dos2unix $list_input_coordinates_path
readarray -t array_coords < $list_input_coordinates_path
for coord_line in "${array_coords[@]:1}"
do 
    #get tile path
    IFS=',' read -ra latlon <<< "$coord_line"
    lat=${latlon[0]}
    #echo tile $tile_name
    lon=${latlon[1]}
    country=${latlon[2]}
    TEST=$(gdallocationinfo -valonly -wgs84 $TEMP_TEST_PATH $lon $lat)
    if  [[ "$TEST" =~ ^[0-9]+$ ]];then
        if [ ! -f $fsc_vrt_path ];then
            list_fsc_files=$(ls  $path_LIS/$tile/*/*/*/*/*FSCOG*tif)
            list_green_files=$(ls  $path_MAJA/$tile/*/*/*/*/*FRE_B3*tif)
            list_red_files=$(ls  $path_MAJA/$tile/*/*/*/*/*FRE_B4*tif)
            list_mir_files=$(ls  $path_MAJA/$tile/*/*/*/*/*FRE_B11*tif)
            list_qcf_files=$(ls  $path_LIS/$tile/*/*/*/*/*QCFLAGS*tif)
            list_dates=$(ls $list_fsc_files | cut -d_ -f2 | cut -dT -f1)
            list_dates_maja=$(ls $list_green_files | cut -d_ -f2 | cut -d- -f1)
            gdalbuildvrt -q -separate $fsc_vrt_path $list_fsc_files
            gdalbuildvrt -q -separate $green_vrt_path $list_green_files
            gdalbuildvrt -q -separate $red_vrt_path $list_red_files
            gdalbuildvrt -q -separate $mir_vrt_path $list_mir_files
            gdalbuildvrt -q -separate $qcf_vrt_path $list_qcf_files
            echo "lat,lon,alt,slp,country,cover,dates,fscog,qcflags,TCD" | tr -d '\n' >  $fsc_csv_path
            echo "lat,lon,dates,green,red,mir" | tr -d '\n' >  $maja_csv_path
        fi
        list_fsc_values=$( eval gdallocationinfo -valonly -wgs84 $fsc_vrt_path $lon $lat)
        list_qcf_values=$( eval gdallocationinfo -valonly -wgs84 $qcf_vrt_path $lon $lat)
        list_green_values=$( eval gdallocationinfo -valonly -wgs84 $green_vrt_path $lon $lat)
        list_red_values=$( eval gdallocationinfo -valonly -wgs84 $red_vrt_path $lon $lat)
        list_mir_values=$( eval gdallocationinfo -valonly -wgs84 $mir_vrt_path $lon $lat)
        TCD_value=$( eval gdallocationinfo -valonly -wgs84 $path_tree $lon $lat)
        cover=$( eval gdallocationinfo -valonly -wgs84 $landcover_path $lon $lat)
        #alt=$( eval gdallocationinfo -valonly -wgs84 $dem_path $lon $lat)
        alt="NaN"
        #slp=$( eval gdallocationinfo -valonly -wgs84 $slope_path $lon $lat)
        slp="NaN"
        if  [[ ! "$alt" =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] || [[ ! "$slp" =~ ^[+-]?[0-9]+\.?[0-9]*$ ]];then
            alt="NaN"
            slp="NaN"
            for eu_dem_file in $eu_dem_path/*.tif;do
                alt_eu_dem=$( eval gdallocationinfo -valonly -wgs84 $eu_dem_file $lon $lat)
                if  [[ "$alt_eu_dem" =~ ^[+-]?[0-9]+\.?[0-9]*$  ]];then
                    alt=$alt_eu_dem
                    temp_slope_tif=$TEMP_JOB_DIR/tmp_slp.tif
                    gdaldem slope $eu_dem_file $temp_slope_tif -s 111120
                    slp=$( eval gdallocationinfo -valonly -wgs84 $temp_slope_tif $lon $lat)
                    break
                fi
            done
        fi
            
        

        
        echo >> $fsc_csv_path
        echo "$lat,$lon,$alt,$slp,$country,$cover,$list_dates,$list_fsc_values,$list_qcf_values,$TCD_value" | tr '\n' ' '  >>  $fsc_csv_path
        echo >> $maja_csv_path
        echo "$lat,$lon,$list_dates_maja,$list_green_values,$list_red_values,$list_mir_values" | tr '\n' ' '  >>  $maja_csv_path
        
    fi

done

if [  -f $fsc_csv_path ];then
    mkdir -p $matched_points_tiles_dir/$tile
    cp $fsc_csv_path $matched_points_tiles_dir/$tile/list_points_FSCOG.csv
    cp $maja_csv_path $matched_points_tiles_dir/$tile/list_points_MAJA.csv
fi

rm -rf TEMP_JOB_DIR
