/*
    Zero-pad images
*/

process get_padding{
    cpus 1
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "get_padding"

    input:
        tuple val(patient_id), path(files), val(is_fixed)
    output:
        tuple val(patient_id), path("pad.txt")
 
    script:
    """
        get_padding.py \
            --input "$files" \
            --log_file ${params.log_file}
    """ 
}

process apply_padding{
    cpus 10
    maxRetries = 3
    memory { task.memory + 10 * task.attempt} 
    tag "apply_padding"

    input:
        tuple val(patient_id), path(img), val(is_fixed), path(padding)
    output:
        tuple val(patient_id), path("padded*"), val(is_fixed)
 
    script:
    """ 
        mkdir -p tmp
        apply_padding.py \
            --image $img \
            --padding $padding \
            --log_file "${params.log_file}"
    """
}

