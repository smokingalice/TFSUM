#!/bin/bash


TVsum_dir="/data01/TFSUM/datasets/TVSum"

root_path="${TVsum_dir}/frames"
annotationfile_path="${TVsum_dir}/annotations/annotations.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
index_dim=1024
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
#index_name="flan-t5-xl-coco"

captions_dir="${TVsum_dir}/captions/summary/keyfram/${llm_model_name}/$index_name/"
output_dir="${TVsum_dir}/index/summary/keyfram/${llm_model_name}/${index_name}/index_flat_ip/"
python -m src.models.make_summary_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
