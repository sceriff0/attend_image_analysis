process pipex_preprocessing {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_preprocessing"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path(tiff)

    output:
    tuple val(patient_id), path("preprocessed"), path(tiff)

    script:
    """
    echo "\$(date): Starting pipex_preprocessing process..." >> ${params.log_file}

    export PIPEX_MAX_RESOLUTION=90000
    channels=""
    for tiff in $tiff; do
        chname=`basename \$tiff | sed 's/.tiff//g' | sed 's/registered_//g' | cut -d'_' -f2-`
        channels+=" \$chname"
    done
    channels=`echo \$channels | sed 's/ /,/g'`

    echo "\$(date): pipex_preprocessing: Channels to be preprocessed: \$channels" >> ${params.log_file}

    echo "\$(date) Preprocessing images..." >> ${params.log_file}
    python -u -W ignore /pipex/preprocessing.py -data=./ -preprocess_markers=\$channels -otsu_threshold_levels=0
    echo "\$(date) Preprocessing images done." >> ${params.log_file}
    """
}


process pipex_segmentation {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_segmentation"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path("preprocessed/*tif"), path(tiff) 
    output:
    tuple val(patient_id), path("analysis/*")

    script:
    """
    export PIPEX_MAX_RESOLUTION=90000

    channels=""
    for tiff in $tiff; do
        chname=`basename \$tiff | sed 's/.tiff//g' | sed 's/registered_//g' | cut -d'_' -f2-`
        channels+=" \$chname"
    done
    channels=`echo \$channels | sed 's/ /,/g'`
    echo "\$(date): pipex_segmentation: Channels to be quantified: \$channels" >> ${params.log_file}

    echo "\$(date) Performing image segmentation..." >> ${params.log_file}
    echo "\$(date) Segmentation input files:" >> ${params.log_file}
    ls -l ./data >> ${params.log_file}
    python -u -W ignore /pipex/segmentation.py -data=./ -nuclei_marker=DAPI -nuclei_diameter=20 -nuclei_expansion=10 -measure_markers=\$channels
    echo "\$(date) Image segmentation done." >> ${params.log_file}
    """
}