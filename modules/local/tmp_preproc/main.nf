process pipex_preprocessing {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_preprocessing"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path(image), path(tiff), val(is_fixed)

    output:
    tuple val(patient_id), path(image), path("preprocessing_input/preprocessed/*tif"), val(is_fixed)

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

    cd ./preprocessing_input/preprocessed

    # Combine image basename to output tif file name to create output file
    for file in *tif; do
        mv \$file `basename ${image}_\$file`
    done
    """
}