process get_metadata{
    cpus 1
    conda "${params.conda_dir}"
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "metadata"

    input:
        tuple val(patient_id), path(nd2_files), path(channels, stageAs: "?/*")
    output:
        tuple val(patient_id), path(channels), path("*.pkl")
 
    script:
    """
        get_metadata.py \
            --nd2_files "$nd2_files" \
            --channels "$channels" \
            --patient_id "$patient_id" \
            --log_file "${params.log_file}"
    """ 
}