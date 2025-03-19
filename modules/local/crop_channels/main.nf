process crop_channels {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_preprocessing"
    
    input:
    tuple val(patient_id), path(image), path(tiff), val(is_fixed)

    output:
    tuple val(patient_id), path(image), path("*tiff"), val(is_fixed)

    script:
    """
    crop_channels.py \
        --patient_id $patient_id \
        --image $image \
        --channel $tiff \
        --is_fixed $is_fixed \
        --crop_size ${params.crop_size_preproc} \
        --overlap_size ${params.overlap_size_preproc} \
        --log_file ${params.log_file}
    """
}