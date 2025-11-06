process quantification{
    cpus 8
    maxRetries = 3
    memory 250.GB
    time 48.h
    publishDir "${params.outdir}/${patient_id}/quantification", mode: 'copy'
    clusterOptions = params.use_gpu ? '--gres=gpu:nvidia_h200:1' : null
    container "docker://bolt3x/attend_image_analysis:quantification_gpu" //"docker://yinxiu/attend_quant:v0.0" 
    tag "quantification"

    input:
        tuple val(patient_id), path(markers), path(positions_file), path(mask_file)
    output:
        // tuple val(patient_id), path("registered_${patient_id}*h5"), emit: "h5"
        tuple val(patient_id), path("*segmentation_markers_data_FULL.csv"), path(mask_file), emit: "quantification"

    script:
    """
        mkdir tmp
        for file in $markers; do
            ln -s \$(readlink -f \$file) tmp/\$(basename \$file)
        done

        quantification_gpu.py \
        --patient_id ${patient_id} \
        --indir tmp \
        --mask_file ${mask_file} \
        --positions_file ${positions_file} \
        --outdir .

        # rm crop*

    """
}
