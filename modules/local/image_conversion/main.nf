process conversion {
    cpus 2
    memory "2G"
    tag "ome_tiff"
    publishDir "${params.outdir}/${patient_id}/results", mode: 'copy'


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
        output="${params.outdir}/\${name}.ome.tiff"

        # Check if the output file already exists
        if [ ! -f \${output} ]; then
            # Log the start of the conversion
            echo "\$(date): Converting \${file} to \${output}..." >> ${params.log_file}
            
            bfconvert \
                -noflat \
                -bigtiff \
                -tilex 512 \
                -tiley 512 \
                -pyramid-resolutions 3 \
                -pyramid-scale 2 \
                \${file} \
                \${name}.ome.tiff
            
            # Log the completion of the conversion
            echo "\$(date): Conversion of \${file} to \${output} completed." >> ${params.log_file}
        else
            # Log if the file already exists
            echo "\$(date): File \${output} already exists, skipping conversion." >> ${params.log_file}
        fi
    done

    # Log the end of the script
    echo "\$(date): Image conversion process completed." >> ${params.log_file}
    """
}

