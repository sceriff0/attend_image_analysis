process preprocess_dapi {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    
    input:
    tuple val(patient_id), path(tiff)

    output:
    tuple val(patient_id), path("*tiff")

    script:
    """
    dapi_preprocessing.py \
        --patient_id $patient_id \
        --images $tiff \
        --log_file ${params.log_file} 
    """
}


process pipex_segmentation {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    publishDir "${params.outdir}/${patient_id}/segmentation", mode: 'copy'
    tag "pipex_segmentation"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path(tiff)
    output:
    // tuple val(patient_id), path("analysis/*")

    tuple val(patient_id), 
        path("segmentation_input/*DAPI.tiff"),
        path("segmentation_input/analysis/cell_data.csv"), 
        path("segmentation_input/analysis/quality_control"),
        path("segmentation_input/analysis/segmentation_binary_mask.tif"), 
        path("segmentation_input/analysis/segmentation_data.npy"), 
        path("segmentation_input/analysis/segmentation_mask.tif"), 
        path("segmentation_input/analysis/segmentation_mask_show.jpg")

    script:
    """
    export PIPEX_MAX_RESOLUTION=90000

    echo "\$(date) Segmentation input files:" >> ${params.log_file}

    mkdir -p ./segmentation_input

    channels=""
    for file in $tiff; do
        chname=`basename \$file | sed 's/.tiff//g' | sed 's/registered_//g' | cut -d'_' -f2-`
        channels+=" \$chname"

        cp \$file ./segmentation_input
    done
    
    channels=`echo \$channels | sed 's/ /,/g'`
    echo "\$(date): pipex_segmentation: Channels to be quantified: \$channels" >> ${params.log_file}

    echo "\$(date) Performing image segmentation..." >> ${params.log_file}
    

    ##############################################
    ##############################################
    ##############################################
    # Segmentation step

    python -u -W ignore /pipex/segmentation.py \
        -data=./segmentation_input \
        -nuclei_marker=DAPI \
        -nuclei_diameter=20 \
        -nuclei_expansion=10 \
        -measure_markers=\$channels

    echo "\$(date) Image segmentation done." >> ${params.log_file}
    """
}