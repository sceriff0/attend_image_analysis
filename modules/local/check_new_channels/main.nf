process check_new_channels{
    cpus 1
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    maxRetries = 3
    memory { 1.GB * task.attempt }
    input:
        tuple val(patient_id), path(files), val(metadata)
    output:
        tuple val(patient_id), path(files), val(metadata), path("channels*")
 
    script:
    """
        check_new_channels.py \
            --patient_id $patient_id \
            --nd2_files "$files" \
            --ome_tiff_image "${params.output_dir}/${patient_id}.ome.tiff"
    """ 
}