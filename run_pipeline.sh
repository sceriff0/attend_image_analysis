#!/bin/bash
    
nextflow run /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/main.nf \
    -with-tower \
    -profile singularity \
    -resume \
    --with-tower \
    --input /hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/samples_19S7.csv \

  
