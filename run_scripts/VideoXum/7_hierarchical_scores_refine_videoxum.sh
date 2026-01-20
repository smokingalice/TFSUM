#!/bin/bash


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1

VideoXum_dir="/data01/TFSUM/datasets/VideoXum"
root_path="${VideoXum_dir}/frames"
annotationfile_path="${VideoXum_dir}/annotations/annotations.txt"

llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
fps=30
T=10
N=10 

context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."


exp_id="_000"
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | cut -c1-243)
dir_name=${dir_name//\//_}
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")



index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
captions_dir="$VideoXum_dir/captions/summary/keyfram/${llm_model_name}/$index_name/"
index_dir="$VideoXum_dir/index/summary/keyfram/${llm_model_name}/$index_name/index_flat_ip/"
scores_dir="$VideoXum_dir/scores/raw/keyfram/${llm_model_name}/${index_name}/${dir_name}/"


K=10
echo "Running with num_neighbors=$K ..."

output_scores_dir="${VideoXum_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${VideoXum_dir}/captions/clean_summary/keyfram/${llm_model_name}/$index_name/"
output_similarity_dir="${VideoXum_dir}/similarity/clean_summary/keyfram/${llm_model_name}/${index_name}/"
output_filenames_dir="${VideoXum_dir}/filenames/clean_summary/keyfram/${llm_model_name}/${index_name}/"

python -m src.models.hierarchical_score_refiner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_scores_dir "$output_scores_dir" \
    --output_summary_dir "$output_summary_dir" \
    --output_similarity_dir "$output_similarity_dir" \
    --output_filenames_dir "$output_filenames_dir" \
    --captions_dir "$captions_dir" \
    --index_dir "$index_dir" \
    --scores_dir "$scores_dir" \
    --fps "$fps" \
    --clip_duration "$T" \
    --num_samples "$N" \
    --num_neighbors "$K"
