process conversion {
    cpus 2
    memory "2G"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    publishDir ${params.output_dir}
    tag "ome_tiff"
    
    input:
    tuple val(patient_id), path(image)

    script:
    """
    if [[ $image != "null.h5" ]]; then
        bfconvert \
            -noflat \
            -bigtiff \
            -tilex 512 \
            -tiley 512 \
            -pyramid-resolutions 3 \
            -pyramid-scale 2 \
            ${image} \
            ${patient_id}.ome.tiff
    fi
    """
}

