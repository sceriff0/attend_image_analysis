process deduplicate_files {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    
    input:
    tuple val(patient_id), path(tiff, stageAs: "?/*")

    output:
    tuple val(patient_id), path("*tiff")

    script:
    """
    deduplicate_files.py \
        --patient_id $patient_id \
        --images $tiff \
        --log_file ${params.log_file} 
    """
}
