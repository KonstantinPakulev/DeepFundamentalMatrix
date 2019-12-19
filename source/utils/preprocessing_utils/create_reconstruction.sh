#!/usr/bin/env bash

colmap_folder="/usr/bin"

dataset_path="/home/konstantin/personal/DeepFundamentalMatrix/south-building"
#image_path="${dataset_path}/images"
#database_path="${dataset_path}/reconstruction.db"

"${colmap_folder}/colmap" model_converter \
        --input_path="${dataset_path}/sparse" \
        --output_path="${dataset_path}/sparse/0" \
        --output_type="BIN"

#nohup xvfb-run -s "+extension GLX -screen 0 1024x768x24" \
#    "${colmap_folder}/colmap" automatic_reconstructor \
#    --workspace_path="${dataset_path}" \
#    --image_path="${image_path}" \
#     --gpu_index==1&

#xvfb-run -s "+extension GLX -screen 0 1024x768x24" \
#    "${colmap_folder}/colmap" feature_extractor \
#    --database_path "${database_path}" \
#    --image_path "${image_path}" \
#        --ImageReader.camera_model RADIAL \
#        --ImageReader.single_camera 1 \
#        --SiftExtraction.use_gpu 1 \
#        --SiftExtraction.gpu_index 1 \
#        --SiftExtraction.estimate_affine_shape 0 \

#nohup xvfb-run -s "+extension GLX -screen 0 1024x768x24" \
#    "${colmap_folder}/colmap" exhaustive_matcher \
#    --database_path "${database_path}" \
#    --SiftExtraction.use_gpu 1 \
#    --SiftExtraction.gpu_index 1 \
#    --SiftMatching.cross_check 0 &
