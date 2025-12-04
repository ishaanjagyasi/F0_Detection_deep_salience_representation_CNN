"""
Evaluation Script for Deep Salience F0 Estimation
STEPS:
1. Load trained model
2. Processe full audio tracks with sliding windows
3. Extracts F0 contours from salience maps
4. Apply temporal smoothing
5. Evaluate using mir_eval metrics
"""

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from pathlib import Path
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from scipy.signal import medfilt
import mir_eval
import json
from tqdm import tqdm

# Import model and data prep utilities
from train_model import DeepSalienceCNN
from data_set_prep import Config as DataConfig, compute_hcqt


def bin_to_hz(bin_idx, config):
    return config.FMIN * (2 ** (bin_idx / config.BINS_PER_OCTAVE)) #Convert frequency bin index to Hz


class EvalConfig:
    """Configuration for evaluation"""
    
    # Paths
    MODEL_PATH = Path("final_model.pt")  
    EVAL_AUDIO_DIR = Path("Data/Evaluation/vocadito/Audio")  
    EVAL_PITCH_DIR = Path("Data/Evaluation/vocadito/Annotations/F0") 
    RESULTS_DIR = Path("Data/evaluation_results")
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inference parameters
    SEGMENT_LENGTH = 5.0  # seconds - same as training
    OVERLAP = 0.5  # 50% overlap between windows
    
    # F0 extraction parameters
    SALIENCE_THRESHOLD = 0.1  # Minimum salience to consider as voiced
    MIN_F0 = 80.0  # Hz - minimum F0 to extract
    MAX_F0 = 1000.0  # Hz - maximum F0 to extract
    
    # Post-processing parameters
    MEDIAN_FILTER_SIZE = 5  # frames - for temporal smoothing
    GAUSSIAN_SIGMA = 1.0  # frames - additional smoothing
    MIN_VOICED_DURATION = 0.1  # seconds - minimum voiced segment duration
    

def load_model(model_path, device):

    model = DeepSalienceCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    
    model.eval()
    return model


def predict_salience_full_track(model, audio_path, config, data_config):
    """
    Predict salience map for full audio track using sliding windows.
    
    Parameters:
    -----------
    model : DeepSalienceCNN
        Trained model
    audio_path : Path
        Path to audio file
    config : EvalConfig
        Evaluation configuration
    data_config : DataConfig
        Data processing configuration
        
    Returns:
    --------
    salience : np.ndarray
        Full-track salience map, shape (n_bins, n_frames)
    times : np.ndarray
        Time in seconds for each frame
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=data_config.SAMPLE_RATE)
    
    # Calculate segment parameters
    segment_samples = int(config.SEGMENT_LENGTH * data_config.SAMPLE_RATE)
    hop_samples = int(segment_samples * (1 - config.OVERLAP))
    
    # Initialize output accumulator
    total_frames = len(y) // data_config.HOP_LENGTH
    salience_sum = np.zeros((data_config.N_BINS, total_frames), dtype=np.float32)
    salience_count = np.zeros((data_config.N_BINS, total_frames), dtype=np.float32)
    
    # Process in sliding windows
    n_windows = (len(y) - segment_samples) // hop_samples + 1
    
    with torch.no_grad():
        for i in range(n_windows):
            # Extract audio window
            start_sample = i * hop_samples
            end_sample = start_sample + segment_samples
            
            # Handle last window
            if end_sample > len(y):
                end_sample = len(y)
                start_sample = max(0, end_sample - segment_samples)
            
            y_window = y[start_sample:end_sample]
            
            # Compute HCQT for window
            hcqt = compute_hcqt_from_audio(y_window, data_config)
            hcqt_tensor = torch.FloatTensor(hcqt).unsqueeze(0).to(config.DEVICE)
            
            # Predict salience
            salience_window = model(hcqt_tensor)
            salience_window = salience_window.squeeze().cpu().numpy()  # (n_bins, n_frames)
            
            # Calculate frame indices for this window
            start_frame = start_sample // data_config.HOP_LENGTH
            window_frames = salience_window.shape[1]
            end_frame = start_frame + window_frames
            
            # Accumulate predictions (for averaging overlaps)
            salience_sum[:, start_frame:end_frame] += salience_window
            salience_count[:, start_frame:end_frame] += 1
    
    # Average overlapping predictions
    salience_count[salience_count == 0] = 1  # Avoid division by zero
    salience = salience_sum / salience_count
    
    # Generate time array
    times = librosa.frames_to_time(
        np.arange(total_frames),
        sr=data_config.SAMPLE_RATE,
        hop_length=data_config.HOP_LENGTH
    )
    
    return salience, times


def compute_hcqt_from_audio(y, config):
    hcqt_channels = []
    for h in config.HARMONICS:
        cqt = librosa.cqt(
            y,
            sr=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH,
            fmin=config.FMIN * h,
            n_bins=config.N_BINS,
            bins_per_octave=config.BINS_PER_OCTAVE
        )
        cqt_mag = np.abs(cqt)
        hcqt_channels.append(cqt_mag)
    return np.array(hcqt_channels)


def extract_f0_from_salience(salience, times, config, data_config):
    """
    Extract F0 contour from salience map using peak picking.
    
    1. For each time frame, find frequency bin with maximum salience
    2. Apply threshold for voicing detection
    3. Convert bin to Hz
    4. Apply temporal smoothing
    5. Filter by frequency range
    
    Parameters:
    -----------
    salience : np.ndarray
        Salience map, shape (n_bins, n_frames)
    times : np.ndarray
        Time array
    config : EvalConfig
        Evaluation configuration
    data_config : DataConfig
        Data processing configuration
        
    Returns:
    --------
    f0_hz : np.ndarray
        F0 values in Hz (0 = unvoiced)
    voicing : np.ndarray
        Boolean voicing flags
    """
    n_frames = salience.shape[1]
    f0_hz = np.zeros(n_frames)
    voicing = np.zeros(n_frames, dtype=bool)
    
    # Extract raw F0 estimates
    for frame_idx in range(n_frames):
        salience_frame = salience[:, frame_idx]
        
        # Find peak
        max_bin = np.argmax(salience_frame)
        max_salience = salience_frame[max_bin]
        
        # Check if voiced
        if max_salience > config.SALIENCE_THRESHOLD:
            freq = bin_to_hz(max_bin, data_config)
            
            # Check frequency range
            if config.MIN_F0 <= freq <= config.MAX_F0:
                f0_hz[frame_idx] = freq
                voicing[frame_idx] = True
    
    # Apply temporal smoothing to F0 (only on voiced regions)
    f0_smoothed = apply_temporal_smoothing(f0_hz, voicing, config)
    
    # Remove short voiced segments
    voicing_smoothed = remove_short_segments(voicing, times, config.MIN_VOICED_DURATION)
    
    # Zero out unvoiced regions
    f0_final = f0_smoothed * voicing_smoothed
    
    return f0_final, voicing_smoothed


def apply_temporal_smoothing(f0_hz, voicing, config):

    f0_smoothed = f0_hz.copy()
    
    if voicing.sum() == 0:
        return f0_smoothed
    
    # Only smooth voiced regions
    voiced_indices = np.where(voicing)[0]
    
    if len(voiced_indices) < config.MEDIAN_FILTER_SIZE:
        return f0_smoothed
    
    # Apply median filter (removes outliers)
    f0_voiced = f0_hz[voiced_indices]
    f0_median = medfilt(f0_voiced, kernel_size=config.MEDIAN_FILTER_SIZE)
    
    # Apply Gaussian smoothing (temporal continuity)
    f0_gaussian = gaussian_filter1d(f0_median, sigma=config.GAUSSIAN_SIGMA)
    
    # Put back into full array
    f0_smoothed[voiced_indices] = f0_gaussian
    
    return f0_smoothed


def remove_short_segments(voicing, times, min_duration):

    voicing_filtered = voicing.copy()
    
    # Find segment boundaries
    changes = np.diff(voicing.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if voicing[0]:
        starts = np.concatenate([[0], starts])
    if voicing[-1]:
        ends = np.concatenate([ends, [len(voicing)]])
    
    # Filter short segments
    for start, end in zip(starts, ends):
        duration = times[end - 1] - times[start]
        if duration < min_duration:
            voicing_filtered[start:end] = False
    
    return voicing_filtered


def load_ground_truth(csv_path, times):

    df = pd.read_csv(csv_path, names=['time', 'frequency'])
    
    ref_f0 = np.zeros(len(times))
    ref_voicing = np.zeros(len(times), dtype=bool)
    
    # Align annotations to prediction times
    for i, t in enumerate(times):
        # Find closest annotation
        time_diffs = np.abs(df['time'].values - t)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] < 0.05:  # Within 50ms
            freq = df.iloc[closest_idx]['frequency']
            if freq > 0:
                ref_f0[i] = freq
                ref_voicing[i] = True
    
    return ref_f0, ref_voicing


def evaluate_track(model, audio_path, csv_path, config, data_config):
    """
    Evaluate model on a single track.
    
    Parameters:
    -----------
    model : DeepSalienceCNN
        Trained model
    audio_path : Path
        Path to audio file
    csv_path : Path
        Path to annotation CSV
    config : EvalConfig
        Evaluation configuration
    data_config : DataConfig
        Data processing configuration
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    # Predict salience
    salience, times = predict_salience_full_track(model, audio_path, config, data_config)
    
    # Extract F0
    est_f0, est_voicing = extract_f0_from_salience(salience, times, config, data_config)
    
    # Load ground truth
    ref_f0, ref_voicing = load_ground_truth(csv_path, times)
    
    # Convert to mir_eval format
    # mir_eval expects: (ref_time, ref_freq), (est_time, est_freq)
    # where freq=0 means unvoiced
    
    # Evaluate using mir_eval
    scores = mir_eval.melody.evaluate(times, ref_f0, times, est_f0)
    
    return scores, {
        'times': times,
        'ref_f0': ref_f0,
        'est_f0': est_f0,
        'ref_voicing': ref_voicing,
        'est_voicing': est_voicing,
        'salience': salience
    }


def evaluate_dataset(config=EvalConfig(), data_config=DataConfig()):

    print("\n" + "="*60)
    print("DEEP SALIENCE F0 EVALUATION")
    print("="*60)
    print(f"Model: {config.MODEL_PATH}")
    print(f"Device: {config.DEVICE}")
    print(f"Evaluation set: {config.EVAL_AUDIO_DIR}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(config.MODEL_PATH, config.DEVICE)
    
    # Get evaluation files
    audio_files = sorted(config.EVAL_AUDIO_DIR.glob("*.wav"))
    
    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {config.EVAL_AUDIO_DIR}")
        return
    
    print(f"Found {len(audio_files)} audio files\n")
    
    # Evaluate each track
    all_scores = []
    per_track_results = {}
    
    for audio_path in tqdm(audio_files, desc="Evaluating tracks"):
        track_id = audio_path.stem
        
        # Find annotation file for this track (e.g., vocadito_1_f0.csv)
        csv_path = config.EVAL_PITCH_DIR / f"{track_id}_f0.csv"
        
        if not csv_path.exists():
            print(f"Warning: No annotation for {track_id}, skipping")
            continue
        
        try:
            scores, outputs = evaluate_track(model, audio_path, csv_path, config, data_config)
            all_scores.append(scores)
            per_track_results[track_id] = {
                'scores': scores,
                'outputs': outputs
            }
        except Exception as e:
            print(f"Error processing {track_id}: {e}")
            continue
    
    # Aggregate metrics
    metric_names = all_scores[0].keys()
    aggregated_metrics = {}
    
    for metric in metric_names:
        values = [s[metric] for s in all_scores]
        aggregated_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'all_values': values
        }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Tracks evaluated: {len(all_scores)}")
    print("\nMetrics (mean ± std):")
    print("-"*60)
    
    for metric, stats in aggregated_metrics.items():
        print(f"{metric:30s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("="*60 + "\n")
    
    # Save results
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        'aggregated_metrics': {k: {m: v for m, v in stats.items() if m != 'all_values'} 
                               for k, stats in aggregated_metrics.items()},
        'per_track_scores': {track_id: result['scores'] 
                             for track_id, result in per_track_results.items()},
        'config': {
            'salience_threshold': config.SALIENCE_THRESHOLD,
            'min_f0': config.MIN_F0,
            'max_f0': config.MAX_F0,
            'median_filter_size': config.MEDIAN_FILTER_SIZE,
            'gaussian_sigma': config.GAUSSIAN_SIGMA,
            'min_voiced_duration': config.MIN_VOICED_DURATION
        }
    }
    
    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return results, per_track_results


if __name__ == "__main__":
    print("Starting evaluation...")
    print(f"Current directory: {Path.cwd()}")
    
    config = EvalConfig()
    print(f"MODEL_PATH exists: {config.MODEL_PATH.exists()}")
    print(f"EVAL_AUDIO_DIR exists: {config.EVAL_AUDIO_DIR.exists()}")
    print(f"EVAL_PITCH_DIR exists: {config.EVAL_PITCH_DIR.exists()}")
    
    results, per_track = evaluate_dataset()
