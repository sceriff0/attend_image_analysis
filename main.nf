#!/usr/bin/env nextflow

nextflow.enable.dsl=2

WorkflowMain.initialise(workflow, params, log)

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PIPELINE WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { get_padding } from './modules/local/image_padding/main.nf'
include { get_metadata } from './modules/local/image_metadata/main.nf'
include { apply_padding } from './modules/local/image_padding/main.nf'
include { affine } from './modules/local/image_registration/main.nf' 
include { diffeomorphic} from './modules/local/image_registration/main.nf'
include { stitching } from './modules/local/image_stitching/main.nf'
include { stacking } from './modules/local/image_stacking/main.nf'
include { conversion } from './modules/local/image_conversion/main.nf'
include { quality_control } from './modules/local/quality_control/main.nf'
include { check_new_channels } from './modules/local/check_new_channels/main.nf'
include { pipex_preprocessing; pipex_segmentation } from './modules/local/pipex/main.nf'


def parse_csv(csv_file_path) {
    channel
        .fromPath(csv_file_path)
        .splitCsv(header: true)
        .map { row ->
            return [
                row.patient_id,      // Patient identifier
                row.image,           // Path to image
                row.fixed,           // Boolean: true if the image is the fixed one
            ]
        }
}

workflow {
    input_ch = parse_csv(params.input)
    grouped_input = input_ch.groupTuple()

    check_new_channels(grouped_input)

    get_padding(grouped_input)

    joined_channel = input_ch.combine(get_padding.out, by:0)
    
    apply_padding(joined_channel)

    moving_fixed_ch = apply_padding.out.groupTuple().flatMap { tuple ->
            def patient = tuple[0]       // Patient ID
            def records = tuple[1]       // List of records for the patient
            def fixed = tuple[2]

            // Find the file associated with the `true` value
            for (int i = 0; i < records.size(); i++){
                if(fixed[i] == "true"){
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

    affine_input = moving_fixed_ch.combine(
        check_new_channels.out.map { it ->
            def patient_id = it[0]
            def channels_to_register = it[3]
            
            return [patient_id, channels_to_register] 
        }, 
        by: 0
    )

    affine(affine_input)

    crops_data = affine.out.map { it ->
        def patient_id = it[0]
        def moving_image = it[1]
        def fixed_image = it[2]
        def crops_paths = it[3] // Paths to *.pkl files
        def channels_to_register = it[4]

        return crops_paths.collect { crops_path ->                    
            return [patient_id, moving_image, fixed_image, crops_path, channels_to_register]
        }
    } 
    .flatMap { it }

    diffeomorphic(crops_data)

    collapsed = diffeomorphic.out.map{
        def patient_id = it[0]
        def moving = it[1]
        def fixed = it[2]
        registered_dapi = it[3]
        def registered_crop = it[4]
        def channels_to_register = it[5]

        return [patient_id, moving.getName(), moving, fixed, registered_dapi, registered_crop, channels_to_register]
    }.groupTuple(by:1).map{
        return [it[0][0], it[2][0], it[3][0], it[4], it[5]]
    }

    stitching(collapsed)
    quality_control(collapsed)

    grouped_stitching_out = stitching.out.h5.groupTuple()

    stitching_out = grouped_stitching_out.map { entry ->
        def channels = entry[1].flatten()
        return [
            entry[0],
            channels
        ]
    }

    meta_input = grouped_input.combine(stitching_out, by: 0).map{
        return [it[0], it[1], it[3]]
    }

    //get_metadata(meta_input) 

    //metadata_out = get_metadata.out.groupTuple().map{
    //    return [it[0], it[1][0], it[2]]
    //}

    //stacking(metadata_out)

    //conversion(stacking.out)

    // dev
    pipex_preprocessing(stitching.out.tiff)
    pipex_segmentation(pipex_preprocessing.out)
}
