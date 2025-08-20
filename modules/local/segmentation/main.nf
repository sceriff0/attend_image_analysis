process segmentation{
    cpus 1
    maxRetries = 3
    memory 300.GB
    time 48.h
    publishDir "${params.outdir}/${patient_id}/segmentation", mode: 'copy'
    container "docker://yinxiu/attend_seg:v0.0"
    tag "segmentation"

    input:
        tuple val(patient_id), path(dapi)
    output:
        // tuple val(patient_id), path("registered_${patient_id}*h5"), emit: "h5"
        tuple val(patient_id), path("positions.pkl"), path("segmentation_mask.npy"), emit: "segmentation"

    script:
    """
        segmentation.py \
        --dapi-file "$dapi" \
        --model-dir "${params.segmentation_model_dir}" \
        --model-name "${params.segmentation_model}" \
        --overlap "${params.segmentation_overlap}" \
        --output-dir "./" \
        --verbose
    """
}
