/*
    Zero-pad images
*/

process get_padding{
    cpus 1
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    maxRetries = 3
    memory { 1.GB * task.attempt }
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
    maxRetries = 3
    memory { 20.GB } 
    // memory { 80.GB }
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    input:
        tuple val(patient_id), path(img), val(metadata), path(padding)
    output:
        tuple val(patient_id), path("${img.simpleName}.h5"), val(metadata)
 
    script:
    """
        apply_padding.py --image $img --padding $padding
    """
}
