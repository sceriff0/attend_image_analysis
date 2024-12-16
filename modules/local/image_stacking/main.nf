process stacking {
    cpus 2
    memory { 100.GB }
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    tag "image_stacking"
    
    input:
    tuple val(patient_id), path(channels, stageAs: "?/*"), path(metadata)

    output:
    tuple val(patient_id), path("*tiff")

    script:
    """
    stacking.py \
        --patient_id "$patient_id" \
        --channels "$channels" \
        --metadata "$metadata" 
    """
}