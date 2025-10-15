/*
    Register images with respect to a predefined fixed image
*/

process affine{
    cpus 5
    maxRetries = 3
    memory { task.memory + 10 * task.attempt}
    tag "affine"

    /*container "docker://bolt3x/attend_image_analysis:v2.3"*/

    input:
        tuple val(patient_id), path(moving), path(fixed), path(channels_to_register)
    output:
        tuple val(patient_id), path(moving), path(fixed), path("*pkl"), path(channels_to_register)
 
    script:
    """
    echo "\$(date) Memory allocated to process "affine": ${task.memory}" >> ${params.log_file}

        affine.py \
            --patient_id $patient_id \
            --channels_to_register $channels_to_register \
            --moving_image $moving \
            --fixed_image $fixed \
            --crop_size_affine ${params.crop_size_affine} \
            --overlap_size_affine ${params.overlap_size_affine} \
            --crop_size_diffeo ${params.crop_size_diffeo} \
            --overlap_size_diffeo ${params.overlap_size_diffeo} \
            --log_file "${params.log_file}"
    """
}
 
process diffeomorphic{
    cpus 4
    maxRetries = 3
    memory { 15.GB * task.attempt }
    array { task.array }
    time '10m'
    tag "diffeomorphic"
    
    container = params.use_gpu ? "docker://bolt3x/attend_image_analysis:v2.4" : null
    clusterOptions = params.use_gpu ? '--gres=gpu:nvidia_h200:1' : null
    /*container "docker://bolt3x/attend_image_analysis:debug_diffeo"*/

    input:
        tuple val(patient_id), path(moving), path(fixed), path(crop), path(channels_to_register)
    output:
        tuple val(patient_id), 
            path(moving), 
            path(fixed), 
            path("qc*"), 
            path("registered*"), 
            path(channels_to_register),
            path("debug_diffeo*", optional: true)
 
    script:
    """
    echo "\$(date) Queue: "diffeomorphic": ${task.queue}" >> ${params.log_file}
    echo "\$(date) Array directive value: "diffeomorphic": ${task.array}" >> ${params.log_file}

        diffeomorphic.py \
            --channels_to_register $channels_to_register \
            --crop_image $crop \
            --moving_image $moving \
            --use_gpu ${params.use_gpu} \
            --debug ${params.debug} \
            --log_file "${params.log_file}"
    """
}
