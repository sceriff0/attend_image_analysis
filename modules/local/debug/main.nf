process debug_diffeo {
    cpus 2
    memory '10.GB'
    tag "debug_diffeo"
    container 'docker://bolt3x/attend_image_analysis:debug_diffeo'

    publishDir "${params.debug_dir}/diffeo", mode: 'copy'
    
    input:
    tuple path(crop), path(mapping)
    
    output:
    path "${crop.baseName}_IoU.txt"
    
    script:
    """
    debug_diffeo.py \
        --crop ${crop} \
        --mapping ${mapping} \
        --model-dir "${params.segmentation_model_dir}" \
        --model-name "${params.segmentation_model}" \
        --output_dir ${params.debug_dir}/diffeo/ \
        --log_file ${params.log_file} \
        --output_file ${crop.baseName}_IoU.txt
    """
}