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
    export PIPEX_MAX_RESOLUTION=90000
    channels=""
    for tiff in $tiff; do
        chname=`basename \$tiff | sed 's/.tiff//g' | sed 's/registered_//g' | cut -d'_' -f2-`
        channels+=" \$chname"
    done
    channels=`echo \$channels | sed 's/ /,/g'`
    python -u -W ignore /pipex/preprocessing.py -data=./ -preprocess_markers=\$channels -otsu_threshold_levels=0
    """
}


process pipex_segmentation {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    tag "pipex_segmentation"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path("preprocessed"), path(tiff) 
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
    python -u -W ignore /pipex/segmentation.py -data=./ -nuclei_marker=DAPI -nuclei_diameter=20 -nuclei_expansion=10 -measure_markers=\$channels
    """
}