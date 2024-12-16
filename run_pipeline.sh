#!/bin/bash

nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
        -with-tower \
        -profile singularity \
        -resume \
        --with-tower \
        --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/20752a7.csv

#check=1
#while [[ $check -ne 0 ]]
#do   
#    nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
#        -with-tower \
#        -profile singularity \
#        -resume \
#        --with-tower \
#        --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/20752a7.csv 
#done

#check=1
#while [[ $check -ne 0 ]]
#do
#    nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
#        -with-tower \
#        -profile singularity \
#        -resume \
#        --with-tower \
#        --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/sample_sheet.csv \
#        --crop_size_affine 2000\
#        --overlap_size_affine 900 \
#        --crop_size_diffeo 2000 \
#        --overlap_size_diffeo 200
#done

# TEST
#nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
#        -with-tower \
#        -resume \
#        -profile singularity \
#        --with-tower \
#        --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/sample_sheet.csv \
#        --crop_size_affine 2000\
#        --overlap_size_affine 900 \
#        --crop_size_diffeo 2000 \
#        --overlap_size_diffeo 200 

