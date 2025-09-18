process pipex_preprocessing {
    cpus 10
    memory { 300.GB }
    tag "pipex_preprocessing"
    container 'docker://alech00/attend_image_analysis:v2.1'
    
    input:
    tuple val(patient_id), path(image), path(tiff), val(is_fixed)

    output:
    tuple val(patient_id), path(image), path("preprocessing_input/preprocessed/*h5"), val(is_fixed)

    script:
    """
    preprocessing.py \
        --patient_id ${patient_id} \
        --image ${image} \
        --channels ${tiff} \
        --output_dir ./preprocessing_input/preprocessed \
        --log_file "${params.log_file}"
    """
}