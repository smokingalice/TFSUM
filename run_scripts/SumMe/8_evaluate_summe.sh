#!/bin/bash
SumMe_dir="/data01/TFSUM/datasets/SumMe"

# Set paths
root_path="${SumMe_dir}/frames"
annotationfile_path="${SumMe_dir}/annotations/annotations.txt"
temporal_annotation_file="${SumMe_dir}/annotations/ydata-summe50-anno.tsv"
datasates="${SumMe_dir}/annotations/eccv16_dataset_summe_google_pool5.h5"
summe_mat_file="./datasets/SumMe/annotations/GT"
llm_model_name="llama-2-13b-chat"
#index_name="opt-6.7b-coco"
index_name="opt-6.7b-coco"
#index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
frame_interval=16
num_neighbors=10
normal_label=0
video_fps=30
context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."

exp_id="_000"
# Convert to lowercase and replace spaces with underscores
#dir_name=$(echo "as_a_video_summarizer,_your_task_is_to_assess_the_importance_of_each_scene_in_a_video._rate_the_scenes_on_a_scale_from_0_to_1,_where_0_represents_the_least_important_scenes_and_1_represents_the_most_important_scenes._the_goal_is_to_select_the_" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
#_000_if_you_were_a_video_summarizer,_how_would_you_rate_the_scenes_described_on_a_scale_from_0_to_1,_where_0_represents_the_least_important_scenes_in_the_video_and_1_represents_the_most_important_scenes_in_the_video?
# Truncate dir_name to the first 243 characters
dir_name=$(echo "the_goal_is_to_ensure_that_the_summary_length_is_between_5%_and_15%_of_the_total_video_length._as_a_video_summarizer,_your_task_is_to_assess_the_importance_of_each_frame,_considering_the_characters,_environment,_objects,_and_other_key_elements" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")
raw_caption_scores_dir="${SumMe_dir}/scores/raw/keyfram/${llm_model_name}/${index_name}/from_clean_caption"


captions_dir="${SumMe_dir}/captions/clean_summary/keyfram/${llm_model_name}/$index_name/"
scores_dir="${SumMe_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
similarity_dir="${SumMe_dir}/similarity/clean_summary/keyfram/${llm_model_name}/${index_name}/"
output_dir="${SumMe_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.evaluate_summe \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --temporal_annotation_file "$temporal_annotation_file" \
    --scores_dir "$scores_dir" \
    --similarity_dir "$similarity_dir" \
    --captions_dir "$captions_dir" \
    --output_dir "$output_dir" \
    --frame_interval "$frame_interval" \
    --normal_label "$normal_label" \
    --num_neighbors "$num_neighbors" \
    --datasates "$datasates"\
    --video_fps "$video_fps"\
    --lambda_value 1\
    --raw_caption_scores_dir "$raw_caption_scores_dir"\
    --summe_mat_file "$summe_mat_file"
