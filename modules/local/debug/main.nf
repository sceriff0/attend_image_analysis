process debug_diffeo {
    cpus 2
    memory 10.GB
    tag "debug_diffeo"
    container 'docker://bolt3x/attend_image_analysis:v2.4'
    
    input:
    tuple val(patient_id), path(crop), path(mapping)
    
    output:
    tuple val(patient_id), path(crop), path(mapping)
    
    script:
    """
    echo "Patient: ${patient_id}" >> ${params.debug_dir}/diffeo/debug.log
    echo "Crop files: ${crop}" >> ${params.debug_dir}/diffeo/debug.log
    echo "Mapping files: ${mapping}" >> ${params.debug_dir}/diffeo/debug.log
    """
}

process debug_segmentation {
    cpus 2
    memory 10.GB
    tag "debug_segmentation"
    container 'docker://bolt3x/attend_image_analysis:v2.4'
    
    input:
    tuple val(patient_id), path(segmentation_mask)
    
    output:
    tuple val(patient_id), path(segmentation_mask)
    
    script:
    """
    echo "Patient: ${patient_id}" >> ${params.debug_dir}/segmentation/debug.log
    echo "Segmentation mask files: ${segmentation_mask}" >> ${params.debug_dir}/segmentation/debug.log
    """
}

process debug_quantification {
    cpus 2
    memory 10.GB
    tag "debug_quantification"
    container 'docker://bolt3x/attend_image_analysis:v2.4'
    
    input:
    tuple val(patient_id), path(phenotypes_data)
    
    output:
    tuple val(patient_id), path(phenotypes_data)
    
    script:
    """
    echo "Patient: ${patient_id}" >> ${params.debug_dir}/quantification/debug.log
    echo "Phenotypes data files: ${phenotypes_data}" >> ${params.debug_dir}/quantification/debug.log
    """
}