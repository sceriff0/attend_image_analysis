process collect_channels{
    cpus 1
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "collect_channels"

    input:
        // tuple val(patient_id), val(images), val(tiffs), val(is_fixed)
        tuple val(patient_id), path(images, stageAs: "?/*"), path(preprocessed, stageAs: "?/*"), val(is_fixed)
        
        
    output:
        tuple val(patient_id), path(preprocessed), val(is_fixed), path('*csv')
 
    script:
    """
        collect_channels.py \
            --patient_id $patient_id \
            --image $images \
            --channels $preprocessed \
            --is_fixed $is_fixed \
            --log_file ${params.log_file}
    """ 
}