docker buildx build --platform linux/amd64 -t alech00/stardist_segmentation:v1.0 --push .

#singularity build stardist_training.sif docker://alech00/stardist_segmentation:v1.0
