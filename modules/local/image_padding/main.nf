/*
    Zero-pad images
*/

process get_padding{
    cpus 1
    memory "1G"
    input:
        tuple val(patient_id), path(files), val(metadata)
    output:
        tuple val(patient_id), path("pad.txt")

    script:
    """
        get_padding.py --input "$files"
    """ 
}

process apply_padding{
    cpus 2
    memory "40G"
    input:
        tuple val(patient_id), path(img), val(metadata), path(padding)
    output:
        tuple val(patient_id), path("${img.simpleName}.h5"), val(metadata)

    script:
    """
        apply_padding.py --image $img --padding $padding
    """
}
