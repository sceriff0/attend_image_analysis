/*
    Zero-pad images
*/

process pad_image {
    cpus 2
    memory "80G"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    publishDir "${params.input_dir}", mode: "copy"
    // container "docker://yinxiu/bftools:latest"
    tag "conversion_h5"

    input:
    tuple val(patient_id),
        val(cycle_id),
        val(fixed_image_path),
        val(input_path),
        val(output_path)
    
    output:
    tuple val(patient_id),
        val(cycle_id),
        val(fixed_image_path),
        val(input_path),
        val(output_path)

    script:
    """
    pad_image.py \
        --patient-id "${patient_id}" \
        --input-dir "${params.input_dir}" \
        --input-path "${input_path}" \
        --crop-width-x 1000 \
        --crop-width-y 1000 \
        --logs-dir "${params.logs_dir}" 

    pad_image.py \
        --patient-id "${patient_id}" \
        --input-dir "${params.input_dir}" \
        --input-path "${fixed_image_path}" \
        --crop-width-x 1000 \
        --crop-width-y 1000 \
        --logs-dir "${params.logs_dir}" 
    """
}

