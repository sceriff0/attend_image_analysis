process get_metadata{
    cpus 1
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "metadata"

    input:
        tuple val(patient_id), path(image_files), path(channels, stageAs: "?/*")
    output:
        tuple val(patient_id), path(channels), path("*.pkl")
 
    script:
    """
        get_metadata.py \
            --image_files "$image_files" \
            --channels "$channels" \
            --patient_id "$patient_id" \
            --log_file "${params.log_file}"
    """ 
}