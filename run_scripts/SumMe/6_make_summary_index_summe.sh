#!/bin/bash

SumMe_dir="/data01/TFSUM/datasets/SumMe"

# Set paths
root_path="${SumMe_dir}/frames"
annotationfile_path="${SumMe_dir}/annotations/annotations.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
index_dim=1024
index_name="opt-6.7b-coco"

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

captions_dir="${SumMe_dir}/captions/summary/keyfram/${llm_model_name}/${index_name}/"
output_dir="${SumMe_dir}/index/summary/keyfram/${llm_model_name}/${index_name}/index_flat_ip/"
python -m src.models.make_summary_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
