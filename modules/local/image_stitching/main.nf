process stitching{
    cpus 1
    maxRetries = 3
    memory { 10.GB }
    publishDir "*.jpg", mode: 'copy'
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    
    input:
        tuple val(patient_id), path(moving), path(fixed), path(dapi_crops), path(crops)
    output:
        tuple val(patient_id), path("registered_${patient_id}*")
 
    script:
    """
        stitching.py \
            --patient_id $patient_id \
            --dapi_crops $dapi_crops \
            --crops $crops \
            --crop_size ${params.crop_size_diffeo} \
            --overlap_size ${params.overlap_size_diffeo} \
            --fixed $fixed \
            --moving $moving 
    """
}
