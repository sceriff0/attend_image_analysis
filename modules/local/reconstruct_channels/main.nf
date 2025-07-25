process reconstruct_channels {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_preprocessing"
    
    input:
    tuple val(patient_id), path(image), path(tif_crops), val(is_fixed)

    output:
    tuple val(patient_id), path(image), path('*tif'), val(is_fixed)

    script:
    """
    reconstruct_channel.py \
        --patient_id $patient_id \
        --image $image \
        --crops $tif_crops \
        --is_fixed $is_fixed \
        --log_file "${params.log_file}"
    """
}