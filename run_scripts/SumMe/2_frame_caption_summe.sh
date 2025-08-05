#!/bin/bash

SumMe_dir="/data01/TFSUM/datasets/SumMe"


root_path="${SumMe_dir}/frames"
annotationfile_path="${SumMe_dir}/annotations/annotations.txt"
batch_size=128
frame_interval=1
force_download=True

pretrained_model_names=(
    "Salesforce/blip2-opt-6.7b-coco"
    #"Salesforce/blip2-opt-6.7b"
    #"Salesforce/blip2-flan-t5-xxl"
    #"Salesforce/blip2-flan-t5-xl"
    #"Salesforce/blip2-flan-t5-xl-coco"
)


if [ "$SLURM_ARRAY_TASK_ID" -ge "${#pretrained_model_names[@]}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is out of range"
  exit 1
fi


pretrained_model_name="${pretrained_model_names[$SLURM_ARRAY_TASK_ID]}"
echo "Processing model: $pretrained_model_name"

output_dir="${SumMe_dir}/captions/raw/${pretrained_model_name}/"


mkdir -p "$output_dir"


echo "Starting captioning process..."
python -m src.models.image_captioner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --pretrained_model_name "$pretrained_model_name" \
    --output_dir "$output_dir"

echo "Captioning process completed for model: $pretrained_model_name"
