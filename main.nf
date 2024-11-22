#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PIPELINE WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { get_padding } from './modules/local/image_padding/main.nf'
include { apply_padding } from './modules/local/image_padding/main.nf'
include { affine } from './modules/local/image_registration/main.nf' 
include { diffeomorphic} from './modules/local/image_registration/main.nf'
include { stitching } from './modules/local/image_stitching/main.nf'
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
                def moving_image = it[1]
                def fixed_image = it[2]
                def crops_paths = it[3]  // Paths to *.pkl files

                return crops_paths.collect { crops_path ->                    
                    return [patient_id, moving_image, fixed_image, crops_path]
                }
            } 
            .flatMap { it }
    diffeomorphic(crops_data)
    collapsed = diffeomorphic.out.map{
        return [it[0], it[1].getName(), it[1], it[2], it[3]]
    }.groupTuple(by:1).map{
        return [it[0][0], it[2][0], it[3][0], it[4]]
    }
    //collapsed.view()
    stitching(collapsed)

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        IMAGE STACKING
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    // stack_images(stitching.out)

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
