#!/bin/bash
    
nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
    -with-tower \
    -profile singularity \
    -resume \
    --with-tower \
    --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/samples_19S7.csv \

#nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
#    -with-tower \
#    -profile singularity \
#    -resume \
#    --with-tower \
#    --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/sample_sheet.csv \
#    --crop_size_affine 2000\
#    --overlap_size_affine 900 \
#    --crop_size_diffeo 2000 \
#    --overlap_size_diffeo 200 

  
