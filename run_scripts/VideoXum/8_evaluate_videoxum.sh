#!/bin/bash

VideoXum_dir="/data01/TFSUM/datasets/VideoXum"
annotation_file="${VideoXum_dir}/annotations/test_videoxum.json"

llm_model_name="llama-2-13b-chat"
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
exp_id="_000"
frame_interval=16
num_neighbors=10
lambda_value=0.7

context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."

dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

scores_dir="${VideoXum_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
output_dir="${VideoXum_dir}/results/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
raw_caption_scores_dir="${VideoXum_dir}/scores/raw/keyfram/${llm_model_name}/${index_name}/from_clean_caption"
similarity_dir="${VideoXum_dir}/similarity/clean_summary/keyfram/${llm_model_name}/${index_name}/"
captions_dir="${VideoXum_dir}/captions/clean_summary/keyfram/${llm_model_name}/$index_name/"

python -m src.evaluate_videoxum \
    --annotation_file "$annotation_file" \
    --scores_dir "$scores_dir" \
    --similarity_dir "$similarity_dir" \
    --captions_dir "$captions_dir" \
    --output_dir "$output_dir" \
    --raw_caption_scores_dir "$raw_caption_scores_dir" \
    --frame_interval "$frame_interval" \
    --num_neighbors "$num_neighbors" \
    --lambda_value "$lambda_value"