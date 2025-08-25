process phenotyping{
    cpus 1
    maxRetries = 3
    memory 300.GB
    time 48.h
    publishDir "${params.outdir}/${patient_id}/phenotyping", mode: 'copy'
    container "docker://yinxiu/attend_seg:v0.0"
    tag "phenotyping"

    input:
        tuple val(patient_id), path(cell_quantification), path(segmentation_mask)
    output:
        // tuple val(patient_id), path("registered_${patient_id}*h5"), emit: "h5"
        tuple val(patient_id), path("phenotypes_data.csv"), path("phenotypes_mask.tiff"), emit: "phenotyping"

    script:
    """
        phenotyping.py \
            --cell_data "${cell_quantification}" \
            --segmentation_mask "${segmentation_mask}" \
            --output_dir "./"
    """
}
