process conversion {
    cpus 20
    memory "20G"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    tag "ome_tiff"
    
    input:
    tuple val(patient_id), path(image)

    script:
    """
    bfconvert \
        -noflat \
        -bigtiff \
        -tilex 512 \
        -tiley 512 \
        -pyramid-resolutions 3 \
        -pyramid-scale 2 \
        "$image" \
        "$patient_id".ome.tiff
    """
}

