process conversion {
    cpus 2
    memory '2 GB'
    tag { "${patient_id}_ome_tiff" }
    time '48h'
    
    publishDir "${params.outdir}/${patient_id}/ome.tiff", mode: 'copy'

    input:
    tuple val(patient_id), path(image)

    output:
    tuple val(patient_id), path("*.ome.tiff")

    script:
    """
    echo "\$(date): Starting image conversion for ${image}..." >> ${params.log_file}

    name="${image.baseName}"

    bfconvert \
        -noflat \
        -bigtiff \
        -tilex ${params.tilex} \
        -tiley ${params.tiley} \
        -pyramid-resolutions ${params.pyramid_resolutions} \
        -pyramid-scale ${params.pyramid_scale} \
        "\${image}" "\${name}.ome.tiff"

    echo "\$(date): Completed image conversion for ${image}" >> ${params.log_file}
    """
}
