process split_channels{
    cpus 1
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "split_channels"

    input:
        tuple val(patient_id), path(image), val(is_fixed)
    output:
        tuple val(patient_id), path(image), path("*tiff"), val(is_fixed)
 
    script:
    """
        split_channels.py \
            --patient_id "$patient_id" \
            --image "$image" \
            --log_file ${params.log_file}
    """ 
}