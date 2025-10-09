process debug_diffeo {
    cpus 2
    memory '10.GB'
    tag "debug_diffeo"
    container 'docker://bolt3x/attend_image_analysis:v2.4'
    
    input:
    tuple path(crop), path(mapping)
    
    output:
    tuple path(crop), path(mapping)
    
    script:
    """
    mkdir -p ${params.debug_dir}/diffeo
    echo "Crop files: ${crop}" >> ${params.debug_dir}/diffeo/debug.log
    echo "Mapping files: ${mapping}" >> ${params.debug_dir}/diffeo/debug.log

    debug_diffeo.py \
        --crop ${crop} \
        --mapping ${mapping} \
        --output_dir ${params.debug_dir}/diffeo/ \
        --log_file ${params.log_file}
    """
}

process debug_segmentation {
    cpus 2
    memory '10.GB'
    tag "debug_segmentation"
    container 'docker://bolt3x/attend_image_analysis:segmentation_gpu'
    
    input:
    path segmentation_mask
    
    output:
    path segmentation_mask
    
    script:
    """
    mkdir -p ${params.debug_dir}/segmentation
    echo "Segmentation mask file: ${segmentation_mask}" >> ${params.debug_dir}/segmentation/debug.log
    """
}

process debug_quantification {
    cpus 2
    memory '10.GB'
    tag "debug_quantification"
    container 'docker://bolt3x/attend_image_analysis:phenotyping'
    
    input:
    path phenotypes_data
    
    output:
    path phenotypes_data
    
    script:
    """
    mkdir -p ${params.debug_dir}/quantification
    echo "Phenotypes data file: ${phenotypes_data}" >> ${params.debug_dir}/quantification/debug.log
    """
}