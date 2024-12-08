#!/bin/bash
#PBS -N do_match_fsc_job
#PBS -l select=1:ncpus=4:mem=20000mb
#PBS -l walltime=01:59:00




#P#B#S -j oe
# Store the directory in  PBS_TMP_DIR

PBS_TMPDIR=$TMPDIR
# Unset TMPDIR env variable which conflicts with openmpi
unset TMPDIR

list_points_datasets_path_temp=$PBS_TMPDIR/output.csv

dirsnow=/work/OT/siaa/Theia/Neige/CoSIMS/zacharie/TOOLS/snowcover

python ${dirsnow}/sc_match_fsc_points.py    -matched_points_tiles_dir $matched_points_tiles_dir\
                                            -list_input_points_path $list_input_points_path\
                                            -list_points_datasets_path $list_points_datasets_path_temp\
                                            -snw_type $snw_type\
                                            -nb_shift_days $nb_shift_days

dos2unix $list_points_datasets_path_temp

mv $list_points_datasets_path_temp $list_points_datasets_path
