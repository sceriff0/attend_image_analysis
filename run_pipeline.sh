#!/bin/bash
    
nextflow run main.nf \
    -resume \
    --with-tower \
    --input samples_19S7.csv \

  