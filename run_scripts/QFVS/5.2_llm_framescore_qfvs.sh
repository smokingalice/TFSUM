#!/bin/bash


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1


QFVS_dir="/data01/TFSUM/datasets/TVSum"
root_path="${QFVS_dir}/frames"
annotationfile_path="${QFVS_dir}/annotations/annotations.txt"

llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=16

context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."
format_prompt="Please provide the response as a single number chosen from the list: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Do not include any other explanation or text. Only respond with one number from the list. The response should be enclosed in square brackets, starting with '[' and ending with ']'."


index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
#index_name="opt-6.7b-coco"
captions_dir="${QFVS_dir}/captions/clean/${index_name}/"
output_scores_dir="${QFVS_dir}/scores/raw/keyfram/${llm_model_name}/${index_name}/from_clean_caption/"
output_summary_dir="${QFVS_dir}/captions/summary/keyfram/${llm_model_name}/${index_name}/"



torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    -m src.models.llm_frame_score  \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --captions_dir "$captions_dir" \
    --output_scores_dir "$output_scores_dir" \
    --output_summary_dir "$output_summary_dir" \
    --ckpt_dir libs/llama/llama-2-13b-chat/ \
    --tokenizer_path libs/llama/tokenizer.model \
    --score_summary