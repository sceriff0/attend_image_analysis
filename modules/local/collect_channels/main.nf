process collect_channels{
    cpus 1
    maxRetries = 3
    memory { 1.GB * task.attempt }
    tag "collect_channels"

    input:
        tuple val(patient_id), path(image, stageAs: "?/*"), path(tif, stageAs: "?/*"), val(is_fixed)
        
        
    output:
        tuple val(patient_id), path('*h5'), val(is_fixed), path('*csv')
 
    script:
    """
        collect_channels.py \
            --patient_id $patient_id \
            --image $image \
            --channels $tif \
            --is_fixed $is_fixed \
            --log_file ${params.log_file}
    """ 
}