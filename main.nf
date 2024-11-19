#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PIPELINE WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { convert_to_ome_tiff } from './modules/local/image_conversion/main.nf'
include { affine_registration } from './modules/local/image_registration/main.nf' 
include { diffeomorphic_registration } from './modules/local/image_registration/main.nf'
include { apply_mappings } from './modules/local/image_registration/main.nf'
include { export_image_1 } from './modules/local/export_image/main.nf'
include { export_image_2 } from './modules/local/export_image/main.nf'
include { stack_dapi_crops } from './modules/local/image_stacking/main.nf'
include { pad_image } from './modules/local/image_padding/main.nf'
// include { stack_images } from './modules/local/image_stacking/main.nf'

def parse_csv(csv_file_path) {
    channel
        .fromPath(csv_file_path)
        .splitCsv(header: true)
        .map { row ->
            return [
                row.patient_id,           // Patient identifier
                row.image,           // Patient identifier
                row.fixed,           // Patient identifier
            ]
        }
}

process get_padding{
    cpus 1
    memory "1G"
    input:
        tuple val(patient_id), path(files), val(metadata)
    output:
        tuple val(patient_id), path("pad.txt")

    script:
    """
        get_padding.py --input "$files"
    """ 
}

process apply_padding{
    cpus 2
    memory "40G"
    input:
        tuple val(patient_id), path(img), val(metadata), path(padding)
    output:
        tuple val(patient_id), path("${img.simpleName}.h5"), val(metadata)

    script:
    """
        apply_padding.py --image $img --padding $padding
    """
}


process affine{
    cpus 2
    memory "50G"
    input:
        tuple val(patient_id), path(moving), path(fixed)
    output:
        tuple val(patient_id), path("*pkl"), path(fixed)

    script:
    """
        affine.py --moving $moving --fixed $fixed --crop_size ${params.crop_size} --overlap_size ${params.overlap_size}
    """
}


process diffeomorphic{
    cpus 1
    memory "5G"
    input:
        tuple val(patient_id), path(fixed), path(crop)
    output:
        tuple val(patient_id), path(fixed), path("registered_${crop}")

    script:
    """
        diffeomorphic.py --crop_image $crop
    """
}

workflow {

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        PARSE CSV INPUT
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    input_ch = parse_csv(params.input)
    get_padding(input_ch.groupTuple())
    joined_channel = input_ch.combine(get_padding.out, by:0)
    apply_padding(joined_channel)
    
    moving_fixed_ch = apply_padding.out.groupTuple().flatMap { tuple ->
            def patient = tuple[0]       // Patient ID
            def records = tuple[1]      // List of records for the patient
            def fix = tuple[2]

            // Find the file associated with the `true` value
            for (int i = 0; i < records.size(); i++){
                if(fix[i] == "true"){
                    trueFile = records[i]
                    break
                }
            }
            

            // Map each record to the new structure
            records.collect { record ->
                [patient, record, trueFile] 
                }
        }.filter { tuple ->
            tuple[1..-1].unique().size() == tuple[1..-1].size() // Check for uniqueness in the list of files
        }
    
    affine(moving_fixed_ch)
    crops_data = affine.out.map { it ->
                def patient_id = it[0]
                def crops_paths = it[1]  // Paths to *.pkl files
                def fixed_image = it[2]
                
                return crops_paths.collect { crops_path ->                    
                    return [patient_id, fixed_image, crops_path]
                }
            } 
            .flatMap { it }
    diffeomorphic(crops_data)
    collapsed = diffeomorphic.out.groupTuple()


    

    // get_diffeomorphic from crop

    // apply_diffeomorphic to crop

    // combine

    // Prepare parameters from parsed CSV data
//  params_parsed = parsed_lines.map { row ->
//      tuple(
//          row.patient_id,
//          row.cycle_id,
//          row.fixed_image_path,
//          row.input_path,
//          row.output_path
//      )
//  }


//  // convert_to_h5(params_parsed)
//  pad_image(params_parsed)
//  affine_registration(pad_image.out)
//  export_image_1(affine_registration.out)
//  stack_dapi_crops(export_image_1.out)

//  crops_data = stack_dapi_crops.out
//          .map { it ->
//              def patient_id = it[0]
//              def cycle_id = it[1]
//              def fixed_image_path = it[2]
//              def input_path = it[3]
//              def output_path = it[4]
//              def crops_paths = it[5]  // Paths to *.pkl files
//              
//              return crops_paths.collect { crops_path ->                    
//                  return [patient_id, cycle_id, fixed_image_path, input_path, output_path, crops_path]
//              }
//          } 
//          .flatMap { it }

//  diffeomorphic_registration(crops_data)

//  unique_diffeo_out = diffeomorphic_registration.out
//      .toList()
//      .map { tuples ->
//          tuples.unique() 
//      }
//      .flatMap()

//  apply_mappings(unique_diffeo_out)

//  export_image_2(apply_mappings.out)

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        IMAGE STACKING
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    // stack_images(export_image_2.out)

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        CONVERSION TO OME.TIFF
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    // input_conv = convert_to_ome_tiff(
    //     stack_images.out
    //         .combine(params_conv)
    // )
}
