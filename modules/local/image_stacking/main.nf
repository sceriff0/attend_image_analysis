process stacking {
    cpus 20
    memory "20G"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    tag "image_stacking"
    
    input:
    tuple val(patient_id), path(fixed), path(registered), path(metadata)

    output:
    tuple val(patient_id), path("*tiff")

    script:
    """
    stacking.py \
        --patient_id "$patient_id" \
        --fixed "$fixed" \
        --registered "$registered" \
        --metadata "$metadata" 
    """
}