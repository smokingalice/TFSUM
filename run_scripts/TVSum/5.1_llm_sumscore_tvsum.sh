#!/bin/bash

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1


TVsum_dir="/data01/TFSUM/datasets/TVSum"

root_path="${TVsum_dir}/frames"
annotationfile_path="${TVsum_dir}/annotations/annotations.txt"
llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=16

context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."
format_prompt="Please provide the response as a single number chosen from the list: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Do not include any other explanation or text. Only respond with one number from the list. The response should be enclosed in square brackets, starting with '[' and ending with ']'."

summary_prompt="My goal is to extract a summary for an entire video. Here, I’m providing a clip from a specific time frame. Please describe this segment and highlight the main key points."

index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
#index_name="flan-t5-xxl"
echo "Processing index: $index_name"

captions_dir="$TVsum_dir/captions/clean/$index_name/"


dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}

dir_name=$(printf "%s_%03d_%s" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$dir_name")



source "$VENV_DIR/bin/activate"

output_scores_dir="${TVsum_dir}/scores/raw/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${TVsum_dir}/captions/summary/keyfram/${llm_model_name}/$index_name/"


torchrun \
    --master_port=29510\
    --nproc_per_node 2 --nnodes 1 -m src.models.llm_summary_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --summary_prompt "$summary_prompt" \
    --output_summary_dir "$output_summary_dir" \
    --captions_dir "$captions_dir" \
    --ckpt_dir libs/llama/llama-2-13b-chat/ \
    --tokenizer_path libs/llama/tokenizer.model

torchrun \
    --master_port=29510\
    --nproc_per_node 2 --nnodes 1 -m src.models.llm_summary_scorer  \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_summary_dir "$output_summary_dir" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --output_scores_dir "$output_scores_dir" \
    --ckpt_dir libs/llama/llama-2-13b-chat/ \
    --tokenizer_path libs/llama/tokenizer.model \
    --score_summary
