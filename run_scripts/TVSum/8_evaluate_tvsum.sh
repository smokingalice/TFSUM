#!/bin/bash

TVsum_dir="/data01/TFSUM/datasets/TVSum"


root_path="${TVsum_dir}/frames"
annotationfile_path="${TVsum_dir}/annotations/annotations.txt"
temporal_annotation_file="${TVsum_dir}/annotations/ydata-tvsum50-anno.tsv"
datasates="${TVsum_dir}/annotations/eccv16_dataset_tvsum_google_pool5.h5"
tvsum_mat_file="./datasets/TVSum/annotations/ydata-tvsum50.mat"
llm_model_name="llama-2-13b-chat"
#index_name="opt-6.7b-coco"
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
frame_interval=16
num_neighbors=10
normal_label=0
video_fps=30

#context_prompt="As a video summarizer, your task is to assess the importance of each scene in a video. Rate the scenes on a scale from 0 to 1, where 0 represents the least important scenes and 1 represents the most important scenes. The goal is to select the top 15% most important scenes for summarization."
#context_prompt="How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious or potentially criminal activities?"

context_prompt="The goal is to ensure that the summary length is between 5% and 15% of the total video length. As a video summarizer, your task is to assess the importance of each frame, considering the characters, environment, objects, and other key elements within the frame. Rate each frame on a scale from 0 to 1, where 0 represents the least important frame and 1 represents the most important frame. The goal is to select the top 5% to 15% most important frames for summarization, ensuring that the summary highlights the key moments and defining actions of the video."

exp_id="_000"



dir_name=$(echo "the_goal_is_to_ensure_that_the_summary_length_is_between_5%_and_15%_of_the_total_video_length._as_a_video_summarizer,_your_task_is_to_assess_the_importance_of_each_frame,_considering_the_characters,_environment,_objects,_and_other_key_elements" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
#dir_name=$(echo "if_you_were_a_video_summarizer,_how_would_you_rate_the_scenes_described_on_a_scale_from_0_to_1,_where_0_represents_the_least_important_scenes_in_the_video_and_1_represents_the_most_important_scenes_in_the_video?" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}

dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")
raw_caption_scores_dir="${TVsum_dir}/scores/raw/keyfram/${llm_model_name}/${index_name}/from_clean_caption"




captions_dir="${TVsum_dir}/captions/clean_summary/keyfram/${llm_model_name}/$index_name/"
scores_dir="${TVsum_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"
similarity_dir="${TVsum_dir}/similarity/clean_summary/keyfram/${llm_model_name}/${index_name}/"
output_dir="${TVsum_dir}/scores/refined/keyfram/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.evaluate_tvsum \
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
    --datasates "$datasates" \
    --video_fps "$video_fps"\
    --lambda_value 0.7\
    --raw_caption_scores_dir "$raw_caption_scores_dir" \
    --tvsum_mat_file "$tvsum_mat_file"