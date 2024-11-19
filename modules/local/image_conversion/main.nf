/*
    Convert images to h5 or ome_tiff
*/

process convert_to_h5 {
    cpus 20
    memory "50G"
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
    convert_to_h5.py \
        --work-dir "${params.work_dir}" \
        --input-path "${input_path}" \
        --crop-width-x 1000 \
        --crop-width-y 1000 \
        --logs-dir "${params.logs_dir}" 
    """
}

process convert_to_ome_tiff {
    memory "1G"
    cpus 1
    publishDir "${params.output_dir_conv}", mode: "copy"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    // container "docker://yinxiu/bftools:latest"
    tag "conversion_ome"

    input:
    tuple val(patient_id),
        val(cycle_id),
        val(fixed_image_path),
        val(input_path),
        val(output_path)

    script:
    """
    if [ ! -f "${output_path}" ]; then
        bfconvert -noflat -bigtiff \
            -tilex "${params.tilex}" \
            -tiley "${params.tiley}" \
            -pyramid-resolutions "${params.pyramid_resolutions}" \
            -pyramid-scale "${params.pyramid_scale}" \
            "${input_path}" "${output_path}"
    fi
    """
}
