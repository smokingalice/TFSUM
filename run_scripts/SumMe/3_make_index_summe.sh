#!/bin/bash

SumMe_dir="/data01/TFSUM/datasets/SumMe"

root_path="${SumMe_dir}/frames"
annotationfile_path="${SumMe_dir}/annotations/annotations.txt"
batch_size=256
frame_interval=16
index_dim=1024

cap_model_names=(
    "$SumMe_dir/captions/raw/Salesforce/blip2-opt-6.7b-coco/"
    #"$SumMe_dir/captions/raw/Salesforce/blip2-opt-6.7b/"
    #"$SumMe_dir/captions/raw/Salesforce/blip2-flan-t5-xxl/"
    #"$SumMe_dir/captions/raw/Salesforce/blip2-flan-t5-xl/"
    #"$SumMe_dir/captions/raw/Salesforce/blip2-flan-t5-xl-coco/"
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
output_dir="${SumMe_dir}/index/${names}/${index_name}/"
# shellcheck disable=SC2086 # We want to pass a list of strings
python -m src.models.make_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "${output_dir}" \
    --captions_dirs $cap_model_names_str
