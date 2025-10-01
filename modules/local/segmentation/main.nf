process segmentation{
    cpus 2
    maxRetries = 3
    memory 90.GB
    publishDir "${params.outdir}/${patient_id}/segmentation", mode: 'copy', pattern: "*.{pkl,npy}"
    container "docker://bolt3x/attend_image_analysis:segmentation_gpu"
    time '20m'
    clusterOptions = '--gres=gpu:nvidia_h200:1'

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
        
        # Verify expected output files exist
        if [ ! -f "positions.pkl" ]; then
            echo "ERROR: positions.pkl not created by segmentation.py" >&2
            exit 1
        fi
        
        if [ ! -f "segmentation_mask.npy" ]; then
            echo "ERROR: segmentation_mask.npy not created by segmentation.py" >&2
            exit 1
        fi
        
        echo "Segmentation completed successfully"
    """
}
