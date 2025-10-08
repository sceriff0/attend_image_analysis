process debug_diffeo {
    cpus 2
    memory 10.GB
    tag "debug_diffeo"
    container 'docker://bolt3x/attend_image_analysis:v2.4'
    
    input:
    path crop, 
    path mapping
    
    output:
    tuple path(crop), path(mapping)
    
    script:
    """
    echo "Crop files: ${crop}" >> ${params.debug_dir}/diffeo/debug.log
    echo "Mapping files: ${mapping}" >> ${params.debug_dir}/diffeo/debug.log
    """
}

process debug_segmentation {
    cpus 2
    memory 10.GB
    tag "debug_segmentation"
    container 'docker://bolt3x/attend_image_analysis:segmentation_gpu'
    
    input:
    path segmentation_mask
    
    output:
    path(segmentation_mask)
    
    script:
    """
    echo "Segmentation mask file: ${segmentation_mask}" >> ${params.debug_dir}/segmentation/debug.log
    """
}   

process debug_quantification {
    cpus 2
    memory 10.GB
    tag "debug_quantification"
    container 'docker://bolt3x/attend_image_analysis:phenotyping'
    
    input:
    path phenotypes_data
    
    output:
    path(phenotypes_data)
    
    script:
    """
    echo "Phenotypes data file: ${phenotypes_data}" >> ${params.debug_dir}/quantification/debug.log
    """
}