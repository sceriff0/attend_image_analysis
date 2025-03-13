process pipex_preprocessing {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_preprocessing"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path(tiff)

    output:
    tuple val(patient_id), path("preprocessing_input/preprocessed"), path(tiff)

    script:
    """
    echo "\$(date): Starting pipex_preprocessing process..." >> ${params.log_file}

    mkdir -p ./preprocessing_input

    export PIPEX_MAX_RESOLUTION=90000

    channels=""
    for file in $tiff; do
        chname=`basename \$file | sed 's/.tiff//g' | sed 's/registered_//g' | cut -d'_' -f2-`
        channels+=" \$chname"
        cp \$file ./preprocessing_input
    done
    channels=`echo \$channels | sed 's/ /,/g'`

    echo "\$(date): pipex_segmentation: Channels to be preprocessed: \$channels" >> ${params.log_file}

    echo "\$(date): pipex_segmentation: Input files: \$channels" >> ${params.log_file}
    ls ./preprocessing_input  >> ${params.log_file}


    ##############################################
    ##############################################
    ##############################################
    # Preprocessing step

    python -u -W ignore /pipex/preprocessing.py \
        -data=./preprocessing_input \
        -preprocess_markers=\$channels \
        -otsu_threshold_levels=0
    """
}