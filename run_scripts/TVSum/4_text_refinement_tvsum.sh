#!/bin/bash

TVsum_dir="/data01/TFSUM/datasets/TVSum"

root_path="${TVsum_dir}/frames"
annotationfile_path="${TVsum_dir}/annotations/annotations.txt"
batch_size=64
frame_interval=16
fps=30
clip_duration=10
num_samples=10
num_neighbors=1

index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"


echo "Processing index: $index_name"




captions_dir_template="$TVsum_dir/captions/raw/Salesforce/{}/"
index_dir="$TVsum_dir/index/${index_name}/index_flat_ip/"
output_dir="${TVsum_dir}/captions/clean/$index_name/"
python -m src.models.image_text_refinement \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "$output_dir" \
    --captions_dir_template "${captions_dir_template}" \
    --index_dir "${index_dir}" \
    --fps "$fps" \
    --clip_duration "$clip_duration" \
    --num_samples "$num_samples" \
    --num_neighbors "$num_neighbors"
