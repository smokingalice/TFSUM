import argparse
import json
from pathlib import Path
import h5py
import scipy.io
import numpy as np
from ortools.sat.python import cp_model
from src.utils.vis_utils import visualize_video

np.set_printoptions(threshold=np.inf, precision=6)

def f1_score(pred, test):
    pred = np.asarray(pred, dtype=np.bool_)
    test = np.asarray(test, dtype=np.bool_)
    min_len = min(pred.shape[0], test.shape[0])
    pred = pred[:min_len]
    test = test[:min_len]
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

def get_keyshot_summ(pred, cps, n_frames, nfps, proportion=0.15):
    seg_scores = np.zeros(len(cps), dtype=np.float32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = pred[first:last + 1]
        if scores.size > 0:
            seg_scores[seg_idx] = scores.mean()
    
    limits = int(n_frames * proportion)
    packed = knapsack((seg_scores * 1000).astype(int), nfps, limits)
    
    summary = np.zeros(n_frames, dtype=np.bool_)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True
    return summary

def load_tags_mat(mat_path):
    data = scipy.io.loadmat(mat_path)
    valid_keys = [k for k in data.keys() if not k.startswith('__')]
    if len(valid_keys) == 0:
        raise ValueError("No valid data found in Tags.mat")
    
    struct = data[valid_keys[0]]
    videos_data = []
    
    if struct.shape[0] == 1:
        struct = struct[0]
        
    for i in range(len(struct)):
        vid_entry = {}
        fields = struct[i].dtype.names
        for field in fields:
            val = struct[i][field]
            if val.shape == (1,): 
                val = val[0]
            vid_entry[field] = val
        videos_data.append(vid_entry)
        
    return videos_data

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
    dataset_root = Path(args.dataset_root)
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gt_data = load_tags_mat(args.tags_mat)
    final_f1_list = []
    
    for i in range(4):
        video_id = i + 1
        video_name_prefix = f"V{video_id}"
        
        h5_path = dataset_root / video_name_prefix / f"{video_name_prefix}_resnet_avg.h5"
        if not h5_path.exists():
            h5_path = dataset_root / f"{video_name_prefix}_resnet_avg.h5"
            
        if not h5_path.exists():
            continue
            
        with h5py.File(h5_path, 'r') as f:
            if 'seg_len' in f:
                seg_len = f['seg_len'][()]
                seg_len = seg_len.flatten() 
                n_frames = int(np.sum(seg_len))
                
                cps = []
                curr = 0
                for length in seg_len:
                    cps.append([curr, curr + length - 1])
                    curr += length
                cps = np.array(cps)
                nfps = seg_len
            else:
                continue

        if i >= len(gt_data):
            continue
            
        vid_gt = gt_data[i] 
        labels = None
        
        for k in vid_gt.keys():
            if 'abel' in k or 'ummary' in k:
                labels = vid_gt[k]
                
        if labels is None:
            continue
            
        video_f1s = []
        
        score_file = scores_dir / f"V{video_id}.json"
        if not score_file.exists():
            score_file = scores_dir / f"P{video_id:02d}.json"
                
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
        
        if len(pred_scores) > n_frames:
            pred_scores = pred_scores[:n_frames]
        elif len(pred_scores) < n_frames:
            pred_scores = np.pad(pred_scores, (0, n_frames - len(pred_scores)))
            
        pred_summary = get_keyshot_summ(pred_scores, cps, n_frames, nfps)
        
        if labels.ndim == 2:
            if labels.shape[1] == n_frames:
                user_summaries = labels
            else:
                user_summaries = labels.T
        else:
            user_summaries = labels[np.newaxis, :]
            
        current_f1s = [f1_score(pred_summary, u_sum) for u_sum in user_summaries]
        final_f1_list.append(np.mean(current_f1s))
        print(f"Video V{video_id}: F1 = {np.mean(current_f1s):.4f}")

    if final_f1_list:
        print("-" * 30)
        print(f"Average F1 (QFVS): {np.mean(final_f1_list):.4f}")
        print("-" * 30)
        with open(output_dir / "result.txt", "w") as f:
            f.write(f"F1: {np.mean(final_f1_list):.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--tags_mat", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--similarity_dir", required=True)
    parser.add_argument("--captions_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_caption_scores_dir", required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--lambda_value", type=float, default=0.7)
    main(parser.parse_args())