import argparse
import json
from pathlib import Path
import h5py
import hdf5storage
import json
import tables
import numpy as np
import os
import scipy.io as sio
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from typing import Iterable, List
from ortools.sat.python import cp_model
from src.data.video_record import VideoRecord
from src.utils.vis_utils import visualize_video
from scipy.stats import spearmanr, kendalltau


def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()

    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item

        item_dict = {
            'video': video[0, 0],
            'category': category[0, 0],
            'title': title[0, 0],
            'length': length[0, 0],
            'nframes': nframes[0, 0],
            'user_anno': user_anno,
            'gt_score': gt_score
        }

        data_list.append((item_dict))

    return data_list


from scipy.stats import spearmanr, kendalltau, rankdata

from scipy.stats import spearmanr, kendalltau, rankdata


def calculate_rank_correlation(video_scores, user_anno):
    spearman_scores = []
    kendall_scores = []
    avg_user_scores = np.mean(user_anno, axis=0)

    # Normalize the user scores
    avg_user_scores = (avg_user_scores - avg_user_scores.min()) / (avg_user_scores.max() - avg_user_scores.min() + 1e-6)

    # Ensure both video_scores and avg_user_scores are of the same length
    min_len = min(len(video_scores), len(avg_user_scores))
    video_scores = video_scores[:min_len]
    avg_user_scores = avg_user_scores[:min_len]

    video_ranks = rankdata(video_scores)
    user_ranks = rankdata(avg_user_scores)

    # Calculate Spearman and Kendall correlations
    spearman_corr, _ = spearmanr(video_scores, avg_user_scores)
    kendall_corr, _ = kendalltau(-video_ranks, -user_ranks)

    return spearman_corr, kendall_corr



np.set_printoptions(threshold=np.inf)


def f1_score(pred: np.ndarray, test: np.ndarray) -> float:

    min_len = min(pred.shape[0], test.shape[0])

    pred = pred[:min_len]
    test = test[:min_len]
    assert abs(pred.shape[0] - test.shape[0]) <= 1
    pred = np.asarray(pred, dtype=np.bool_)
    test = np.asarray(test, dtype=np.bool_)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def knapsack(values: Iterable[int], weights: Iterable[int], capacity: int) -> List[int]:
    model = cp_model.CpModel()
    n = len(values)
    x = [model.NewBoolVar(f'x{i}') for i in range(n)]

    model.Add(sum(x[i] * weights[i] for i in range(n)) <= capacity)
    model.Maximize(sum(x[i] * values[i] for i in range(n)))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        packed = [i for i in range(n) if solver.Value(x[i])]

        return packed
    else:

        return []



def load_reference_annotations(anno_file):
    annotations = {}

    video_annotations = {}

    with open(anno_file) as f:
        for line in f:
            parts = line.strip().split()
            video_id = parts[0]  # ��ƵID
            scores = list(map(int, parts[2].split(',')))  # ��ȡ��Ҫ�Է���

            if video_id not in video_annotations:
                video_annotations[video_id] = []
            video_annotations[video_id].append(scores)


    for video_id, all_scores in video_annotations.items():

        scores_matrix = np.array(all_scores)

        avg_scores = np.mean(scores_matrix, axis=0)

        normalized_scores = [(score - 1) / 4 for score in avg_scores]
        annotations[video_id] = normalized_scores


    return annotations


def get_video_labels(video_record, annotations, normal_label, video_fps, frame_interval=2):
    video_name = Path(video_record.path).name
    labels = []
    video_annotations = annotations.get(video_name, [])
    total_frames = video_record.num_frames
    num_scores = len(video_annotations)

    frames_per_score = int(frame_interval * video_fps)

    for frame_index in range(total_frames):
        frame_label = normal_label
        score_idx = frame_index // frames_per_score

        if score_idx < num_scores:
            frame_label = video_annotations[score_idx]
        labels.append(frame_label)

    return labels


def calculate_weighted_scores(
        scores_dict,
        similarity_dict,
        num_neighbors,
        frame_interval,
        clean_caption_scores,
        lambda_value=0.5
):
    scores = []
    for frame_idx in scores_dict.keys():
        idx_str = str(frame_idx)
        if isinstance(scores_dict[frame_idx], dict):
            frame_scores = np.array(
                [scores_dict[idx_str][str(nn_idx)] for nn_idx in range(num_neighbors)]
            )
            frame_similarity = np.array(
                [similarity_dict[idx_str][str(nn_idx)] for nn_idx in range(num_neighbors)]
            )
            frame_weights = np.exp(frame_similarity) / np.sum(np.exp(frame_similarity))
            refined_score = np.sum(frame_scores * frame_weights)
        else:
            refined_score = scores_dict[frame_idx]

        segment_idx = int(frame_idx) // frame_interval
        original_score = clean_caption_scores[segment_idx]
        final_score = lambda_value * refined_score + (1 - lambda_value) * original_score
        scores.append(final_score)

    scores = np.repeat(scores, frame_interval)
    return scores


def save_metric(output_dir, metric_name, num_neighbors, metric_value):
    with open(output_dir / f"{metric_name}_nn_{num_neighbors}.txt", "w") as f:
        f.write(f"{metric_value}\n")


def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     proportion: float = 0.15,
                     video_name: str = "") -> np.ndarray:
    seg_scores = np.zeros(len(cps), dtype=np.float32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = pred[first:last + 1]
        if scores.size == 0:
            print(f"[Warning] Empty segment detected in video '{video_name}' at segment index {seg_idx} ({first}-{last}).")
        seg_scores[seg_idx] = int(1000 * scores.mean())

    limits = int(n_frames * proportion)
    packed = knapsack(seg_scores.astype(int), nfps, limits)

    summary = np.zeros(n_frames, dtype=np.bool_)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True

    return summary



def main(
        root_path,
        annotationfile_path,
        temporal_annotation_file,
        datasates,
        scores_dir,
        similarity_dir,
        captions_dir,
        output_dir,
        frame_interval,
        normal_label,
        num_neighbors,
        without_labels,
        visualize,
        video_fps,
        raw_caption_scores_dir,
        lambda_value,
        tvsum_mat_file,
):
    scores_dir = Path(scores_dir)
    similarity_dir = Path(similarity_dir)
    captions_dir = Path(captions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_reference_annotations(temporal_annotation_file)


    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]

    flat_scores = []
    flat_labels = []
    finalF = []
    file_path = datasates
    spearman_all = []
    kendall_all = []
    for video in video_list:
        video_name = Path(video.path).name
        tvsum_data = load_tvsum_mat(tvsum_mat_file)
        user_anno = None
        raw_caption_scores_dir = Path(raw_caption_scores_dir)
        clean_caption_scores_path = raw_caption_scores_dir / f"{video_name}.json"
        with open(clean_caption_scores_path) as f:
            clean_caption_scores_dict = json.load(f)

        max_idx = max([int(k) for k in clean_caption_scores_dict.keys()])
        num_segments = max_idx // frame_interval + 1
        clean_caption_scores = [0.0] * num_segments
        for k, v in clean_caption_scores_dict.items():
            idx = int(k) // frame_interval
            clean_caption_scores[idx] = v
        for d in tvsum_data:
            if d['video'] == video_name:
                user_anno = d['user_anno'].T
                break

        if user_anno is None:
            print(f"Warning: No user_anno found for {video_name}")
            continue
        # Load the scores and similarity
        video_scores_path = scores_dir / f"{video_name}.json"
        video_similarity_path = similarity_dir / f"{video_name}.json"
        video_captions_path = captions_dir / f"{video_name}.json"

        with open(video_scores_path) as f:
            video_scores_dict = json.load(f)
        with open(video_similarity_path) as f:
            video_similarity = json.load(f)
        with open(video_captions_path) as f:
            video_captions = json.load(f)


        video_labels = get_video_labels(video, annotations, normal_label, video_fps)

        video_scores = calculate_weighted_scores(
            video_scores_dict,
            video_similarity,
            num_neighbors,
            frame_interval,
            clean_caption_scores,
            lambda_value
        )
        video_scores = video_scores[: video.num_frames]

        with h5py.File(file_path, 'r') as f:
            for name in f:
                
                sdate = f[name]["video_name"][...]
                sdate = sdate.item().decode("utf-8").replace(" ", "_")

                if sdate == video_name:
                    print("true")
                    cps = f[name]["change_points"][...]
                    nfps = f[name]["n_frame_per_seg"][...]
                    user_sum = f[name]["user_summary"][...]


            summary = get_keyshot_summ(video_scores, cps, video.num_frames, nfps, video_name=video_name)

            if summary.size > video.num_frames:
                pred_summ = pred_summ[:video.num_frames]
            elif summary.size < video.num_frames:
                summary = np.pad(summary, (0, video.num_frames - summary.size))

            f1s = [f1_score(user_summ, summary) for user_summ in user_sum]

            final_f1 = np.mean(f1s)
            finalF.append(final_f1)
            spearman_corr, kendall_corr = calculate_rank_correlation(video_scores * 100, user_anno)
            spearman_all.append(spearman_corr)
            kendall_all.append(kendall_corr)
            print(
                f"Video: {video_name} - F1: {final_f1:.3f}, Spearman: {spearman_corr:.3f}, Kendall: {kendall_corr:.3f}")

            print(final_f1)
            if visualize:
                visualize_video(
                    video_name,
                    [],
                    video_scores,
                    video_captions,
                    video.path,
                    video_fps,
                    output_dir / f"{video_name}.mp4",
                    normal_label,
                    "{:06d}.jpg",
                    None,
                )


    average_f1 = np.mean(finalF)
    average_spearman = np.mean(spearman_all)
    average_kendall = np.mean(kendall_all)

    print(f"Final F1 Score: {average_f1:.3f}")
    print(f"Final Spearman Correlation: {average_spearman:.3f}")
    print(f"Final Kendall Correlation: {average_kendall:.3f}")




def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--temporal_annotation_file", type=str, required=True)
    parser.add_argument("--datasates", type=str, required=True)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--similarity_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Optional arguments with defaults
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int)
    parser.add_argument("--num_neighbors", type=int, default=10)

    parser.add_argument("--without_labels", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--video_fps", type=float)
    parser.add_argument("--raw_caption_scores_dir", type=str, required=True)
    parser.add_argument("--lambda_value", type=float, default=0.5)
    parser.add_argument("--tvsum_mat_file", type=str, required=True,
                        help="Path to the TVSum MATLAB annotation file (.mat)")
    args = parser.parse_args()
    if args.temporal_annotation_file is None and not args.without_labels:
        parser.error("--temporal_annotation_file is required when --without_labels is not used")
    if args.visualize:
        if args.video_fps is None:
            parser.error("--video_fps is required when --visualize is used")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.root_path,
        args.annotationfile_path,
        args.temporal_annotation_file,
        args.datasates,
        args.scores_dir,
        args.similarity_dir,
        args.captions_dir,
        args.output_dir,
        args.frame_interval,
        args.normal_label,
        args.num_neighbors,
        args.without_labels,
        args.visualize,
        args.video_fps,
        args.raw_caption_scores_dir,
        args.lambda_value,
        args.tvsum_mat_file,

    )
