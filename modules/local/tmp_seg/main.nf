process pipex_segmentation {
    //cpus 2
    //memory { task.memory + 10 * task.attempt}
    publishDir "${params.outdir}/${patient_id}/segmentation", mode: 'copy'
    tag "pipex_segmentation"
    container "docker://yinxiu/pipex:latest"
    
    input:
    tuple val(patient_id), path(preprocessed), path(tiff) 
    output:
    tuple val(patient_id), path("preprocessed/analysis/*")

    tuple val(patient_id), 
        path("preprocessed/analysis/cell_data.csv"), 
        path("preprocessed/analysis/quality_control"),
        path("preprocessed/analysis/segmentation_binary_mask.tif"), 
        path("preprocessed/analysis/segmentation_data.npy"), 
        path("preprocessed/analysis/segmentation_mask.tif"), 
        path("preprocessed/analysis/segmentation_mask_show.jpg")

    script:
    """
    export PIPEX_MAX_RESOLUTION=90000

    echo "\$(date) INPUT DATA FOLDER (from preprocessing) ${preprocessed}: " >> ${params.log_file}
    echo "\$(date) Segmentation input files:" >> ${params.log_file}
    ls -l ./${preprocessed} >> ${params.log_file}

    channels=""
    for tiff in $tiff; do
        chname=`basename \$tiff | sed 's/.tiff//g' | sed 's/registered_//g' | cut -d'_' -f2-`
        channels+=" \$chname"
    done
    
    channels=`echo \$channels | sed 's/ /,/g'`
    echo "\$(date): pipex_segmentation: Channels to be quantified: \$channels" >> ${params.log_file}

    echo "\$(date) Performing image segmentation..." >> ${params.log_file}
    

    ##############################################
    ##############################################
    ##############################################
    # Segmentation step

    python -u -W ignore /pipex/segmentation.py \
        -data=./${preprocessed} \
        -nuclei_marker=DAPI \
        -nuclei_diameter=20 \
        -nuclei_expansion=10 \
        -measure_markers=\$channels

    echo "\$(date) Image segmentation done." >> ${params.log_file}
    """
}