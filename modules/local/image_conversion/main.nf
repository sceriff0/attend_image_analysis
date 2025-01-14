// process conversion {
//     cpus 2
//     memory "2G"
//     conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
//     publishDir "${params.output_dir}", mode: 'copy'
//     tag "ome_tiff"
//     
//     input:
//     tuple val(patient_id), path(image)
// 
//     script:
//     """
//     if [[ $image != "null.tiff" ]]; then
//         bfconvert \
//             -noflat \
//             -bigtiff \
//             -tilex 512 \
//             -tiley 512 \
//             -pyramid-resolutions 3 \
//             -pyramid-scale 2 \
//             "${image}" \
//             "${name}".ome.tiff
//     fi
//     """
// }


// USE THIS 

process conversion {
    cpus 2
    memory "2G"
    conda '/hpcnfs/scratch/DIMA/chiodin/miniconda3'
    tag "ome_tiff"
    publishDir "*.ome.tiff", mode: 'copy'


    input:
    tuple val(patient_id), path(image)


    output:
    tuple val(patient_id), path("*ome.tiff")


    script:
    """
    for file in $image; do
        name=\$(basename \${file})
        bfconvert \
            -noflat \
            -bigtiff \
            -tilex 512 \
            -tiley 512 \
            -pyramid-resolutions 3 \
            -pyramid-scale 2 \
            \${file} \
            \${name}.ome.tiff
    done
    """
}
