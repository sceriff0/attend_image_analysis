#!/bin/bash

# TEST
nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
        -profile singularity \
        --with-tower \
        --test true \
        --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/sample_sheet.csv \
        --log_file /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/LOG.log \
        --crop_size_affine 2000 \
        --overlap_size_affine 900 \
        --crop_size_diffeo 2000 \
        --overlap_size_diffeo 800 \
        -resume 

