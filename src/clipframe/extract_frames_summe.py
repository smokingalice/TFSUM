import argparse
import os
from pathlib import Path
import cv2
import subprocess

def convert_video(video_path):
    output_path = video_path.replace(".mpg", "_converted.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def extract_frames(video_path, frames_dir):
    if video_path.endswith(".mpg"):
        video_path = convert_video(video_path)

    # ✅ 替换空格为下划线
    video_name = Path(video_path).stem.replace(" ", "_")
    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, frames_dir, annotations_file):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    with open(annotations_file, "w") as f:
        for video_file in os.listdir(videos_dir):
            if video_file.lower().endswith((".avi", ".mp4", ".mpg", ".mkv")):
                video_path = os.path.join(videos_dir, video_file)
                video_name, num_frames = extract_frames(video_path, frames_dir)
                # ✅ 写入注释时也使用替换后的名称
                f.write(f"{video_name} 0 {num_frames - 1} 0\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory path to the videos.")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory path to the frames.")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the annotations file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.videos_dir, args.frames_dir, args.annotations_file)
