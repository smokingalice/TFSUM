#!/bin/bash
dataset_dir="/data01/TFSUM/datasets/QFVS/QFVS_dataset"

# Set paths
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/annotations.txt"

python src/clipframe/extract_qfvs.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file"
