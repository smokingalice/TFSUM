#!/bin/bash


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1
SumMe_dir="/data01/TFSUM/datasets/SumMe"

root_path="${SumMe_dir}/frames"
annotationfile_path="${SumMe_dir}/annotations/annotations.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
fps=30
T=10
N=10
num_neighbors=10
#context_prompt="How would you rate the importance of the scene described on a scale from 0 to 1, with 0 representing low importance and 1 representing high importance?"
context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."


exp_id="_000"

dir_name=$(echo "the_goal_is_to_ensure_that_the_summary_length_is_between_5%_and_15%_of_the_total_video_length._as_a_video_summarizer,_your_task_is_to_assess_the_importance_of_each_frame,_considering_the_characters,_environment,_objects,_and_other_key_elements" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

index_name="opt-6.7b-coco"
captions_dir="$SumMe_dir/captions/summary/keyfram/${llm_model_name}/$index_name/"
index_dir="$SumMe_dir/index/summary/keyfram/${llm_model_name}/$index_name/index_flat_ip/"
scores_dir="$SumMe_dir/scores/raw/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
output_scores_dir="${SumMe_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${SumMe_dir}/captions/clean_summary/keyfram/${llm_model_name}/$index_name/"
output_similarity_dir="${SumMe_dir}/similarity/clean_summary/keyfram/${llm_model_name}/${index_name}/"
output_filenames_dir="${SumMe_dir}/filenames/clean_summary/keyfram/${llm_model_name}/${index_name}/"

# Run the Python script with the specified parameters
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
    --num_neighbors "$num_neighbors"
