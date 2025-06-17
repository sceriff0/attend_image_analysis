process pipex_preprocessing {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "basicpy_preprocessing"
    
    input:
    tuple val(patient_id), path(image), path(tiff), val(is_fixed)

    output:
    tuple val(patient_id), path(image), path("preprocessing_input/preprocessed/*h5"), val(is_fixed)

    script:
    """
    preprocessing.py \
        --patient_id ${patient_id} \
        --image ${image} \
        --channels ${tiff} \
        --output_dir ./preprocessing_input/preprocessed \
        --log_file "${params.log_file}"
    """
}