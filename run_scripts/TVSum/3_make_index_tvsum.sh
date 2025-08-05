#!/bin/bash
TVsum_dir="/data01/TFSUM/datasets/TVSum"

root_path="${TVsum_dir}/frames"
annotationfile_path="${TVsum_dir}/annotations/annotations.txt"
batch_size=256
frame_interval=16
index_dim=1024

cap_model_names=(
    "$TVsum_dir/captions/raw/Salesforce/blip2-opt-6.7b-coco/"
    "$TVsum_dir/captions/raw/Salesforce/blip2-opt-6.7b/"
    "$TVsum_dir/captions/raw/Salesforce/blip2-flan-t5-xxl/"
    "$TVsum_dir/captions/raw/Salesforce/blip2-flan-t5-xl/"
    "$TVsum_dir/captions/raw/Salesforce/blip2-flan-t5-xl-coco/"
)

cap_model_names_str=$(IFS=' '; echo "${cap_model_names[*]}")

names=""
IFS='/' read -ra components <<< "$cap_model_names_str"
for component in "${components[@]}"; do
    if [[ "$component" =~ ^blip2- ]]; then
        names+="${component#blip2-}+"
    fi
done


names=${names%+}

echo "Creating index for $names"



index_name="index_flat_ip"
output_dir="${TVsum_dir}/index/${names}/${index_name}/"

python -m src.models.make_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "${output_dir}" \
    --captions_dirs $cap_model_names_str
