process stacking {
    cpus 2
    memory 500.GB
    tag "stacking"
    
    input:
    tuple val(patient_id), path(channels, stageAs: "?/*")

    output:
    tuple val(patient_id), path("*tiff")

    script:
    """
    stacking.py \
        --patient_id "$patient_id" \
        --channels "$channels" \
        --n_crops ${params.n_crops} \
        --log_file "${params.log_file}"
    """
}
