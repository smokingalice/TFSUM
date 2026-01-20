import argparse
import json
import numpy as np
from pathlib import Path
from ortools.sat.python import cp_model
from src.utils.vis_utils import visualize_video

np.set_printoptions(threshold=np.inf, precision=6)

def f1_score(pred, test):
    pred = np.asarray(pred, dtype=np.bool_)
    test = np.asarray(test, dtype=np.bool_)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def knapsack(values, weights, capacity):
    model = cp_model.CpModel()
    n = len(values)
    x = [model.NewBoolVar(f'x{i}') for i in range(n)]
    model.Add(sum(x[i] * weights[i] for i in range(n)) <= capacity)
    model.Maximize(sum(x[i] * values[i] for i in range(n)))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return [i for i in range(n) if solver.Value(x[i])]
    return []

def get_keyshot_summ(pred, n_frames, proportion=0.15):
    limits = int(n_frames * proportion)
    scores = np.array(pred)
    weights = np.ones(n_frames, dtype=int)
    packed = knapsack((scores * 1000).astype(int), weights, limits)
    summary = np.zeros(n_frames, dtype=np.bool_)
    for idx in packed:
        summary[idx] = True
    return summary

def resize_scores(scores, target_length):
    source_length = len(scores)
    if source_length == target_length:
        return scores
    indices = np.linspace(0, source_length - 1, target_length)
    return np.interp(indices, np.arange(source_length), scores)

def calculate_weighted_scores(scores_dict, similarity_dict, num_neighbors, frame_interval, clean_caption_scores, lambda_value):
    scores = []
    sorted_keys = sorted([int(k) for k in scores_dict.keys()])
    
    for frame_idx in sorted_keys:
        idx_str = str(frame_idx)
        if isinstance(scores_dict[idx_str], dict):
            frame_scores = np.array([scores_dict[idx_str][str(i)] for i in range(num_neighbors)])
            frame_sim = np.array([similarity_dict[idx_str][str(i)] for i in range(num_neighbors)])
            frame_sim = frame_sim - np.max(frame_sim)
            weights = np.exp(frame_sim) / np.sum(np.exp(frame_sim))
            refined = np.sum(frame_scores * weights)
        else:
            refined = scores_dict[idx_str]
            
        seg_idx = frame_idx // frame_interval
        orig = clean_caption_scores[seg_idx] if seg_idx < len(clean_caption_scores) else 0.0
        
        scores.append(lambda_value * refined + (1 - lambda_value) * orig)
        
    return np.repeat(scores, frame_interval)

def main(args):
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.annotation_file, 'r') as f:
        data = json.load(f)
        
    gt_dict = {}
    if isinstance(data, list):
        for item in data:
            vid = item.get('video_id', item.get('id'))
            if vid:
                gt_dict[vid] = item
    elif isinstance(data, dict):
        gt_dict = data
        
    final_f1_list = []
    processed_count = 0
    
    for video_id, video_info in gt_dict.items():
        score_file = scores_dir / f"{video_id}.json"
        if not score_file.exists():
            if video_id.startswith("v_"):
                score_file = scores_dir / f"{video_id[2:]}.json"
            else:
                score_file = scores_dir / f"v_{video_id}.json"
                
        if not score_file.exists():
            continue
            
        with open(score_file) as f:
            pred_data = json.load(f)
            
        raw_file = Path(args.raw_caption_scores_dir) / score_file.name
        clean_scores = []
        if raw_file.exists():
            with open(raw_file) as rf:
                rd = json.load(rf)
                max_k = max([int(k) for k in rd.keys()]) if rd else 0
                clean_scores = [0.0] * (max_k // args.frame_interval + 2)
                for k, v in rd.items():
                    clean_scores[int(k)//args.frame_interval] = v
                    
        similarity_file = Path(args.similarity_dir) / score_file.name
        sim_dict = {}
        if similarity_file.exists():
            with open(similarity_file) as sf:
                sim_dict = json.load(sf)
                
        pred_scores = calculate_weighted_scores(
            pred_data, sim_dict, args.num_neighbors, 
            args.frame_interval, clean_scores, args.lambda_value
        )
        
        vsum_onehot = np.array(video_info['vsum_onehot'])
        target_len = vsum_onehot.shape[1]
        
        resized_pred = resize_scores(pred_scores, target_len)
        pred_summary = get_keyshot_summ(resized_pred, target_len)
        
        user_f1s = [f1_score(pred_summary, user_gt) for user_gt in vsum_onehot]
        max_f1 = max(user_f1s)
        
        final_f1_list.append(max_f1)
        processed_count += 1
        print(f"Video {video_id}: F1 = {max_f1:.4f}")
        
    if final_f1_list:
        avg_f1 = np.mean(final_f1_list)
        print("-" * 30)
        print(f"Processed {processed_count} videos.")
        print(f"Average F1 (VideoXum): {avg_f1:.4f}")
        print("-" * 30)
        with open(output_dir / "result.txt", "w") as f:
            f.write(f"F1: {avg_f1:.6f}\n")
    else:
        print("No matched videos found between GT and Scores.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--similarity_dir", required=True)
    parser.add_argument("--captions_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_caption_scores_dir", required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--lambda_value", type=float, default=0.7)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)