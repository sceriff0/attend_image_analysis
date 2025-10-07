process conversion {
    cpus 2
    memory "2G"
    tag "${patient_id}-${image.baseName}"
    time "48.h"
    publishDir "${params.outdir}/${patient_id}/ome.tiff", mode: 'copy'
    
    input:
    tuple val(patient_id), path(image)  // Single file now
    
    output:
    tuple val(patient_id), path("*.ome.tiff")
    
    script:
    def name = image.baseName
    """
    echo "\$(date): Starting conversion of ${image}..." >> ${params.log_file}
    
    bfconvert \
        -noflat \
        -bigtiff \
        -tilex ${params.tilex} \
        -tiley ${params.tiley} \
        -pyramid-resolutions ${params.pyramid_resolutions} \
        -pyramid-scale ${params.pyramid_scale} \
        ${image} \
        ${name}.ome.tiff
    
    echo "\$(date): Completed conversion of ${image}." >> ${params.log_file}
    """
}
