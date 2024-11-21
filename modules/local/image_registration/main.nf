/*
    Register images with respect to a predefined fixed image
*/

process affine{
    cpus 2
    memory "80G"
    input:
        tuple val(patient_id), path(moving), path(fixed)
    output:
        tuple val(patient_id), path(moving), path(fixed), path("*pkl")

    script:
    """
        affine.py --moving $moving --fixed $fixed --crop_size ${params.crop_size_affine} --overlap_size ${params.overlap_size_affine}
    """
}


process diffeomorphic{
    cpus 1
    memory "5G"
    input:
        tuple val(patient_id), path(moving), path(fixed), path(crop)
    output:
        tuple val(patient_id), path(moving), path(fixed), path("registered_${crop}")

    script:
    """
        diffeomorphic.py --crop_image $crop
    """
}
