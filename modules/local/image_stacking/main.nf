process stack_images {
    cpus 20
    memory "100G"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    // cpus 32
    // memory "170G"
    // errorStrategy 'retry'
    // maxRetries = 1
    // memory { 80.GB * task.attempt }
    publishDir "${params.output_dir_stack}", mode: "copy"
    // container "docker://tuoprofilo/toolname:versione"
    tag "image_stacking"
    
    input:
    tuple val(patient_id),
        val(fixed_image_path),
        val(input_path),
        val(output_path)

    output:
     tuple val(patient_id),
        val(fixed_image_path),
        val(input_path),
        val(output_path)
    
    script:
    """
    stack_images.py \
        --output-dir "${params.output_dir_stack}" \
        --fixed-image-path "${fixed_image_path}" \
        --logs-dir "${params.logs_dir}" 
    """
}