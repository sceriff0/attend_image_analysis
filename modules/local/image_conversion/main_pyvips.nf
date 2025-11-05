/*
 * Image conversion using pyvips (5-20× faster than bfconvert)
 *
 * This process converts TIFF images to pyramidal OME-TIFF format
 * with multi-threaded processing and efficient memory usage.
 */

process conversion {
    cpus 8
    memory '100 GB'
    tag { "${patient_id}_ome_tiff" }
    time '4h'  // Reduced from 48h due to pyvips speed

    publishDir "${params.outdir}/${patient_id}/ome.tiff", mode: 'copy'

    input:
    tuple val(patient_id), path(image)

    output:
    tuple val(patient_id), path("*.ome.tiff")

    script:
    def compression = params.compression ?: 'lzw'
    def quality = params.jpeg_quality ?: 90
    """
    echo "\$(date): Starting pyvips conversion for ${image}..." >> ${params.log_file}

    # Set number of threads for pyvips
    export VIPS_CONCURRENCY=${task.cpus}

    convert_image_pyvips.py \
        "${image}" \
        "${image.baseName}.ome.tiff" \
        --tile-size ${params.tilex} \
        --pyramid-levels ${params.pyramid_resolutions} \
        --pyramid-scale ${params.pyramid_scale} \
        --compression ${compression} \
        --jpeg-quality ${quality}

    echo "\$(date): Completed pyvips conversion for ${image}" >> ${params.log_file}
    """
}


/*
 * Alternative: Original bfconvert (kept for compatibility)
 * Uncomment this and comment out the above to use bfconvert instead
 */

// process conversion {
//     cpus 8              // Increased from 2
//     memory '32 GB'      // Increased from 2 GB
//     tag { "${patient_id}_ome_tiff" }
//     time '8h'          // Reduced from 48h with more resources
//
//     publishDir "${params.outdir}/${patient_id}/ome.tiff", mode: 'copy'
//
//     input:
//     tuple val(patient_id), path(image)
//
//     output:
//     tuple val(patient_id), path("*.ome.tiff")
//
//     script:
//     """
//     echo "\$(date): Starting bfconvert conversion for ${image}..." >> ${params.log_file}
//
//     name="${image.baseName}"
//     image="${image}"
//
//     bfconvert \
//         -noflat \
//         -bigtiff \
//         -compression LZW \
//         -tilex ${params.tilex} \
//         -tiley ${params.tiley} \
//         -pyramid-resolutions ${params.pyramid_resolutions} \
//         -pyramid-scale ${params.pyramid_scale} \
//         "\$image" \
//         "\${name}.ome.tiff"
//
//     echo "\$(date): Completed bfconvert conversion for ${image}" >> ${params.log_file}
//     """
// }
