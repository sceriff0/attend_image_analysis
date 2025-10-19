process check_new_channels{
    cpus 1
    memory { 1.GB }
    tag "check_channels"

    input:
        tuple val(patient_id), path(files), val(metadata)
    output:
        tuple val(patient_id), path(files), val(metadata), path("channels*")
 
    script:
    """
        check_new_channels.py \
            --patient_id $patient_id \
            --nd2_files "$files" \
            --ome_tiff_image "${params.outdir}/${patient_id}.ome.tiff" \
            --optional_channels "${params.optional_channels}" \
            --log_file "${params.log_file}"
    """ 
}