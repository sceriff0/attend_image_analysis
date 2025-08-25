process get_metadata{
    cpus 1
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "metadata"

    input:
        tuple val(patient_id), path(image_files)
    output:
        tuple val(patient_id), path("*.pkl")
 
    script:
    """
        get_metadata.py \
            --image_files "$image_files" \
            --patient_id "$patient_id" \
            --log_file "${params.log_file}" \
            --pixel_microns ${params.pixel_microns}
    """ 
}