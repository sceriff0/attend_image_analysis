process stacking {
    cpus 2
    memory { task.memory + 10 * task.attempt}
    conda "${params.conda_dir}"
    tag "stacking"
    
    input:
    tuple val(patient_id), path(channels, stageAs: "?/*"), path(metadata)

    output:
    tuple val(patient_id), path("*tiff")

    script:
    """
    stacking.py \
        --patient_id "$patient_id" \
        --channels "$channels" \
        --n_crops ${params.n_crops} \
        --metadata "$metadata" \
        --log_file "${params.log_file}"
    """
}