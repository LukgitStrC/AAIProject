import numpy as np
from typing import List, Dict, Tuple
import cv2

from cropping_panorama import *

def get_angular_distance(angle1: float, angle2: float) -> float:
    """Calculates the shortest distance between two angles on a circle."""
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)

def evaluate_predictions(predicted_yaws: List[float], true_yaws: List[float], tolerance: float = 15.0) -> Dict:
    """
    Compares predicted angles to ground truth angles using greedy matching.
    """
    tp = 0
    fp = 0
    fn = 0
    angular_errors = []

    # Track which ground truth angles have been matched
    matched_gt_indices = set()
    
    # Sort predictions (optional, but good for consistency)
    predicted_yaws = sorted(predicted_yaws)

    for pred_angle in predicted_yaws:
        best_match_idx = -1
        min_dist = float('inf')

        # Find the closest unmatched ground truth angle
        for i, gt_angle in enumerate(true_yaws):
            if i in matched_gt_indices:
                continue
                
            dist = get_angular_distance(pred_angle, gt_angle)
            if dist < min_dist and dist <= tolerance:
                min_dist = dist
                best_match_idx = i

        if best_match_idx != -1:
            # We found a match within the tolerance
            tp += 1
            matched_gt_indices.add(best_match_idx)
            angular_errors.append(min_dist)
        else:
            # Prediction didn't match any GT within tolerance
            fp += 1

    # Any ground truth angles that were never matched are False Negatives
    fn = len(true_yaws) - len(matched_gt_indices)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "errors": angular_errors
    }

def run_benchmark(dataset: List[Dict], depth_model: 'DepthEstimationModel', tolerance: float = 15.0):
    """
    Runs the pipeline over a dataset and calculates aggregate metrics.
    
    Expected dataset format:
    [
        {"image_path": "intersection1.jpg", "true_yaws": [0.0, 90.0, 180.0, 270.0]},
        {"image_path": "t_junction.jpg", "true_yaws": [45.0, 225.0, 315.0]}
    ]
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_errors = []

    print(f"Starting benchmark on {len(dataset)} images with a tolerance of {tolerance}°...")

    for data in dataset:
        img_path = data["image_path"]
        true_yaws = data["true_yaws"]

        # 1. Load image and predict depth
        print(f"Processing {img_path}...")
        image = cv2.imread(img_path)
        if image is None:
            print(f"  -> Error: Could not load {img_path}")
            continue
            
        depth_map = depth_model.predict(img_path)

        # 2. Extract angles using your existing pipeline
        predicted_yaws = find_street_angles(depth_map)

        # 3. Evaluate
        results = evaluate_predictions(predicted_yaws, true_yaws, tolerance)
        
        total_tp += results["TP"]
        total_fp += results["FP"]
        total_fn += results["FN"]
        all_errors.extend(results["errors"])
        
        print(f"  -> Predicted: {predicted_yaws}")
        print(f"  -> True:      {true_yaws}")
        print(f"  -> Image Stats: TP={results['TP']}, FP={results['FP']}, FN={results['FN']}")

    # 4. Calculate Final Metrics
    metric_dict = {}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_error = np.mean(all_errors) if all_errors else 0.0

    metric_dict['accuracy'] = precision
    metric_dict['recall'] = recall
    metric_dict['f1_score'] = f1_score
    metric_dict['mean_error'] = mean_error


    print("\n" + "="*30)
    print("🎯 BENCHMARK RESULTS")
    print("="*30)
    print(f"Total True Positives:  {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print("-" * 30)
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1_score:.2%}")
    print(f"Mean Angular Error: {mean_error:.2f}°")
    print("="*30)

    return metric_dict

    

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Initialize your model
    # depth_model = DepthEstimationModel()
    
    # Create your ground truth dataset manually by looking at a few panoramas
    # and estimating where the streets are in degrees (-180 to 180)
    mock_dataset = [
        {
            "image_path": "path/to/your/test_pano_1.jpg", 
            "true_yaws": [-90.0, 0.0, 85.0] # Example: a 3-way intersection
        },
        {
            "image_path": "path/to/your/test_pano_2.jpg", 
            "true_yaws": [-175.0, -5.0, 90.0, 175.0] # Example: a 4-way intersection
        }
    ]
    
    # Run the benchmark
    # run_benchmark(mock_dataset, depth_model, tolerance=15.0)