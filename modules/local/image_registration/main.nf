/*
    Register images with respect to a predefined fixed image
*/

process affine{
    cpus 2
    maxRetries = 3
    memory { 70.GB }
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    input:
        tuple val(patient_id), path(moving), path(fixed)
    output:
        tuple val(patient_id), path(moving), path(fixed), path("*pkl")
 
    script:
    """
        affine.py \
            --moving $moving \
            --fixed $fixed \
            --crop_size_affine ${params.crop_size_affine} \
            --overlap_size_affine ${params.overlap_size_affine} \
            --crop_size_diffeo ${params.crop_size_diffeo} \
            --overlap_size_diffeo ${params.overlap_size_diffeo}
    """
}
 

process diffeomorphic{
    cpus 1
    maxRetries = 3
    memory { 2.GB * task.attempt }
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    input:
        tuple val(patient_id), path(moving), path(fixed), path(crop)
    output:
        tuple val(patient_id), path(moving), path(fixed), path("registered*")
 
    script:
    """
        diffeomorphic.py \
            --crop_image $crop \
            --moving_image $moving
    """
}