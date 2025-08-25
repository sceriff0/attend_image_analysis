process conversion {
    cpus 2
    memory "2G"
    tag "ome_tiff"
    time "48.h"
    
    publishDir "${params.outdir}/${patient_id}/ome.tiff", mode: 'copy'


    input:
    tuple val(patient_id), path(image)


    output:
    tuple val(patient_id), path("*ome.tiff")


    script:
    """
    # Log the start of the script
    echo "\$(date): Starting image conversion process..." >> ${params.log_file}

    for file in $image; do
        # Get the base name and output directory path
        name="\${file%.*}"

        
        bfconvert \
            -noflat \
            -bigtiff \
            -tilex ${params.tilex} \
            -tiley ${params.tiley} \
            -pyramid-resolutions ${params.pyramid_resolutions} \
            -pyramid-scale ${params.pyramid_scale} \
            \${file} \
            \${name}.ome.tiff
        
    
    done

    # Log the end of the script
    echo "\$(date): Image conversion process completed." >> ${params.log_file}
    """
}

