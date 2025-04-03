#!/usr/bin/env nextflow

nextflow.enable.dsl=2

WorkflowMain.initialise(workflow, params, log)

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PIPELINE WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { split_channels } from './modules/local/split_channels/main.nf'
include { crop_channels } from './modules/local/crop_channels/main.nf'
include { reconstruct_channels } from './modules/local/reconstruct_channels/main.nf'
include { collect_channels } from './modules/local/collect_channels/main.nf'
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
include { pipex_preprocessing } from './modules/local/preprocessing/main.nf'
include { pipex_membrane_segmentation } from './modules/local/segmentation/main.nf'
include { pipex_nuclei_segmentation } from './modules/local/segmentation/main.nf'
include { preprocess_dapi } from './modules/local/segmentation/main.nf'
include { deduplicate_files } from './modules/local/deduplicate_files/main.nf'
include { create_membrane_channel } from './modules/local/create_membrane_channel/main.nf'
include { segmentation_quality_control as nuclei_segmentation_quality_control } from './modules/local/quality_control/main.nf'
include { segmentation_quality_control as membrane_segmentation_quality_control } from './modules/local/quality_control/main.nf'



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

def parse_csv2(csv_file_path) {
    csv_file_path
        .splitCsv(header: true)
        .map { row ->
            return [
                row.patient_id,      // Patient identifier
                row.image,           // Path to image
                row.fixed,           // Boolean: true if the image is the fixed one
            ]
        }
}

workflow preprocessing {
    take:
    parsed_csv_ch

    main:
    split_channels(parsed_csv_ch)

    crop_channels_input = split_channels.out.map { it ->
        def patient_id = it[0]
        def image = it[1]
        def tiff_channels = it[2]
        def is_fixed = it[3] 

        tiff_channels.collect { tiff ->
            return [patient_id, image, tiff, is_fixed]
        }
    }
    .flatMap { it }

    crop_channels(crop_channels_input)

    preproc_input = crop_channels.out

    pipex_preprocessing(preproc_input)

    preprocessed = pipex_preprocessing.out

    reconstruct_channels(preprocessed)

    reconstructed_channels_ch = reconstruct_channels.out

    collect_channels_input = reconstructed_channels_ch
    .groupTuple(by: 0)
    .collect()

    collect_channels(collect_channels_input)

    csv_files =  collect_channels.out.map { it ->
        def csv_files = it[3]

        return csv_files
    }

    preprocessed_parsed_csv_ch = parse_csv2(csv_files)

    emit: preprocessed_parsed_csv_ch
}

workflow {
    parsed_csv_ch = parse_csv(params.input)

    if (params.preprocessing) {
        updated_parsed_csv_ch = preprocessing(parsed_csv_ch)
    } else {
        updated_parsed_csv_ch = parsed_csv_ch
    }

    grouped_input = updated_parsed_csv_ch.groupTuple()

    check_new_channels(grouped_input)
 
    get_padding(grouped_input)

    joined_channel = updated_parsed_csv_ch.combine(get_padding.out, by:0)
    
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
        def registered_dapi = it[3]
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

    get_metadata(meta_input) 

    metadata_out = get_metadata.out.groupTuple().map{
        return [it[0], it[1][0], it[2]]
    }

    stacking(metadata_out)

    conversion(stacking.out)
 
    duplicated_ch = stitching.out.tiff
        .groupTuple()
        .map { tuple ->
            def patient_id = tuple[0]
            def tiff_files = tuple[1].flatten()

            return [patient_id, tiff_files]
        }

    deduplicate_files(duplicated_ch)

    deduplicate_files.out

    create_membrane_channel(deduplicate_files.out)

    preprocess_dapi_input = create_membrane_channel.out.map{ it -> 
        def patient_id = it[0]
        def markers = it[1]
        def membrane_marker = it[2]        

        return [patient_id, [markers, membrane_marker].flatten()]
    }

    preprocess_dapi(preprocess_dapi_input)

    pipex_segmentation_input = preprocess_dapi.out

    pipex_membrane_segmentation(pipex_segmentation_input)
    pipex_nuclei_segmentation(pipex_segmentation_input)

    membrane_segmentation_quality_control_input = pipex_membrane_segmentation.out.map { it ->
            def patient_id = it[0]
            def dapi = it[1]
            def cell_data = it[2]
            def quality_control = it[3]
            def segmentation_binary_mask  = it[4]
            def segmentation_data  = it[5]
            def segmentation_mask  = it[6]
            def segmentation_mask_show = it[7]
            def type = 'membrane'

            return [patient_id, dapi, segmentation_mask, type]
    }

    nuclei_segmentation_quality_control_input =  pipex_nuclei_segmentation.out.map { it ->
            def patient_id = it[0]
            def dapi = it[1]
            def cell_data = it[2]
            def quality_control = it[3]
            def segmentation_binary_mask  = it[4]
            def segmentation_data  = it[5]
            def segmentation_mask  = it[6]
            def segmentation_mask_show = it[7]
            def type = 'nuclei'

            return [patient_id, dapi, segmentation_mask, type]
    } 

    membrane_segmentation_quality_control(membrane_segmentation_quality_control_input)
    nuclei_segmentation_quality_control(nuclei_segmentation_quality_control_input)
}
