"""
1. Compute HCQT (Harmonic Constant-Q Transform) from audio
2. Create target salience representations from F0 annotations
3. Align annotations with HCQT time frames
4. Create random train/validation splits using PyTorch (85/15)
"""

import numpy as np
import librosa
import pandas as pd
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    
    SAMPLE_RATE = 22050
    HOP_LENGTH = 256  # ~11.6ms at 22050 Hz
    
    # HCQT parameters
    N_BINS = 360  # 6 octaves * 60 bins/octave
    BINS_PER_OCTAVE = 60  # 20 cents per bin
    FMIN = 32.7  # C1 (for h=1 fundamental)
    HARMONICS = [0.5, 1, 2, 3, 4, 5]  # Subharmonic, fundamental, 4 harmonics
    
    # Target generation parameters
    GAUSSIAN_BLUR_SIGMA = 3  # ~quarter-tone blur (3 bins at 20 cents/bin)
    
    # Segment parameters (for training)
    SEGMENT_LENGTH = 5.0  # seconds - length of training segments
    # This gives ~430 frames at hop_length=256, sr=22050
    # HCQT size: (6, 360, ~430) = ~50MB (fits in GPU easily)
    
    # Data split ratios (no test set - using separate Evaluation folder)
    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.15
    # Note: Evaluation/vocadito will be used as final test set
    
    # Paths
    DATA_DIR = Path("Data/Training_Validation/MedleyDB-Pitch")
    AUDIO_DIR = DATA_DIR / "audio"
    PITCH_DIR = DATA_DIR / "pitch"
    METADATA_PATH = DATA_DIR / "medleydb_pitch_metadata.json"
    OUTPUT_DIR = Path("Data/processed_data")


# ============================================================================
# 1. TIME-FREQUENCY REPRESENTATION FOR AUDIO (HCQT)
# ============================================================================

def compute_hcqt(audio_path, config=Config):
    """
    Compute Harmonic Constant-Q Transform (HCQT) from audio file.
    
    The HCQT extends the standard CQT by computing multiple CQTs at different
    fundamental frequencies to capture harmonic relationships. Each harmonic
    channel h computes a CQT with fmin scaled by h.
    
    Key insight: In HCQT, harmonics align vertically across channels.
    For example, if bin k in channel h=1 represents 440 Hz (A4), then:
    - Bin k in channel h=2 represents 880 Hz (A5, 2nd harmonic)
    - Bin k in channel h=3 represents 1320 Hz (3rd harmonic)
    
    This alignment makes it easy for CNNs to learn harmonic patterns.
    
    Parameters:
    -----------
    audio_path : Path or str
        Path to audio file (.wav)
    config : Config object
        Configuration parameters
    
    Returns:
    --------
    hcqt : np.ndarray
        HCQT representation with shape (n_harmonics, n_bins, n_frames)
        - n_harmonics = 6 (0.5, 1, 2, 3, 4, 5)
        - n_bins = 360
        - n_frames = varies by audio length
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    
    hcqt_channels = []
    
    for h in config.HARMONICS:
        # Compute CQT for this harmonic
        # fmin is scaled by harmonic number h
        cqt = librosa.cqt(
            y,
            sr=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH,
            fmin=config.FMIN * h,
            n_bins=config.N_BINS,
            bins_per_octave=config.BINS_PER_OCTAVE
        )
        
        # Take magnitude (we don't need phase for pitch tracking)
        cqt_mag = np.abs(cqt)
        hcqt_channels.append(cqt_mag)
    
    # Stack all harmonics into 3D array
    hcqt = np.array(hcqt_channels)  # Shape: (6, 360, T)
    
    return hcqt


# ============================================================================
# 2. TIME-FREQUENCY REPRESENTATION FOR GROUND TRUTH
# ============================================================================

def hz_to_bin(frequency, config=Config):
    """
    Convert frequency in Hz to CQT bin index.
    
    CQT bins are logarithmically spaced:
    bin_k = bins_per_octave * log2(freq / fmin)
    
    """
    if frequency <= 0:
        return -1
    
    bin_idx = config.BINS_PER_OCTAVE * np.log2(frequency / config.FMIN)
    bin_idx = int(np.round(bin_idx))
    
    # Check if in valid range
    if 0 <= bin_idx < config.N_BINS:
        return bin_idx
    else:
        return -1


def create_target_salience(csv_path, n_frames, config=Config):
    """
    salience representation from F0 annotations.
    
    Process:
    1. Load CSV with (time, frequency) annotations
    2. For each HCQT time frame, find the closest annotation
    3. Convert frequency to bin index and mark as 1
    4. Apply Gaussian blur in frequency to soften targets

    """
    # Load annotations
    df = pd.read_csv(csv_path, names=['time', 'frequency'])
    
    # Initialize target matrix
    target = np.zeros((config.N_BINS, n_frames), dtype=np.float32)
    
    # Compute time for each HCQT frame
    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH
    )
    
    # For each frame, find closest annotation and mark target
    for frame_idx, frame_time in enumerate(frame_times):
        # Find annotation closest in time
        time_diffs = np.abs(df['time'].values - frame_time)
        closest_idx = np.argmin(time_diffs)
        
        freq = df.iloc[closest_idx]['frequency']
        
        # Only mark if voiced (frequency > 0)
        if freq > 0:
            bin_idx = hz_to_bin(freq, config)
            if bin_idx >= 0:  # Valid bin
                target[bin_idx, frame_idx] = 1.0
    
    # Apply Gaussian blur along frequency axis
    # This creates a "soft" target where nearby bins also have some energy
    target = gaussian_filter1d(target, sigma=config.GAUSSIAN_BLUR_SIGMA, axis=0)
    
    # Renormalize to [0, 1] after blur
    if target.max() > 0:
        target = target / target.max()
    
    return target


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class MedleyDBPitchDataset(Dataset):
    """
    PyTorch Dataset for MedleyDB-Pitch.
    During training: Returns random segments from tracks (5 seconds)
    During inference: Returns full tracks
    
    """
    
    def __init__(self, track_ids, config=Config, train=True):
        """
        Parameters:
        -----------
        track_ids : list of str
            List of track identifiers to include in this dataset
        config : Config object
            Configuration parameters
        train : bool
            If True, extract random segments. If False, return full tracks.
        """
        self.track_ids = track_ids
        self.config = config
        self.train = train
        
        # Compute segment length in frames
        self.segment_frames = int(
            config.SEGMENT_LENGTH * config.SAMPLE_RATE / config.HOP_LENGTH
        )
        
    def __len__(self):
        return len(self.track_ids)
    
    def __getitem__(self, idx):
        """
        Load and process a track.
        
        Returns:
        --------
        sample : dict
            Dictionary with keys:
            - 'hcqt': Tensor of shape (6, 360, T) where T is segment_frames (train) or full length (val)
            - 'target': Tensor of shape (360, T)
            - 'track_id': str
        """
        track_id = self.track_ids[idx]
        
        # Paths
        audio_path = self.config.AUDIO_DIR / f"{track_id}.wav"
        csv_path = self.config.PITCH_DIR / f"{track_id}.csv"
        
        # Minimum audio length (0.5 seconds)
        min_audio_length = int(0.5 * self.config.SAMPLE_RATE)
        
        # If training, extract audio segment first to save memory
        if self.train:
            # Load audio to check length
            y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
            
            # Skip if audio is too short
            if len(y) < min_audio_length:
                print(f"Warning: {track_id} too short ({len(y)} samples), skipping")
                # Return a dummy sample (will be filtered out)
                return self.__getitem__((idx + 1) % len(self.track_ids))
            
            audio_length_frames = len(y) // self.config.HOP_LENGTH
            
            if audio_length_frames > self.segment_frames:
                # Random starting point (in samples)
                max_start_frame = audio_length_frames - self.segment_frames
                start_frame = np.random.randint(0, max_start_frame)
                start_sample = start_frame * self.config.HOP_LENGTH
                end_sample = start_sample + int(self.segment_frames * self.config.HOP_LENGTH)
                
                # Ensure we don't go out of bounds
                end_sample = min(end_sample, len(y))
                
                # Extract audio segment
                y_segment = y[start_sample:end_sample]
                
                # Pad if too short
                if len(y_segment) < min_audio_length:
                    y_segment = np.pad(y_segment, (0, min_audio_length - len(y_segment)))
                
                # Compute HCQT on segment only
                hcqt = self._compute_hcqt_from_audio(y_segment, self.config)
                
                # Load and extract corresponding annotation segment
                df = pd.read_csv(csv_path, names=['time', 'frequency'])
                # Adjust times to segment
                segment_start_time = start_frame * self.config.HOP_LENGTH / self.config.SAMPLE_RATE
                segment_end_time = segment_start_time + self.config.SEGMENT_LENGTH
                
                # Filter annotations to segment
                df_segment = df[(df['time'] >= segment_start_time) & (df['time'] < segment_end_time)].copy()
                df_segment['time'] = df_segment['time'] - segment_start_time
                
                # Create target for segment
                n_frames = hcqt.shape[-1]
                target = self._create_target_from_df(df_segment, n_frames, self.config)
            else:
                # Track shorter than segment, use full track
                # Pad if necessary
                if len(y) < min_audio_length:
                    y = np.pad(y, (0, min_audio_length - len(y)))
                
                hcqt = self._compute_hcqt_from_audio(y, self.config)
                n_frames = hcqt.shape[-1]
                
                df = pd.read_csv(csv_path, names=['time', 'frequency'])
                target = self._create_target_from_df(df, n_frames, self.config)
        else:
            # Validation: use full track
            hcqt = compute_hcqt(audio_path, self.config)
            n_frames = hcqt.shape[-1]
            target = create_target_salience(csv_path, n_frames, self.config)
        
        # Convert to PyTorch tensors
        sample = {
            'hcqt': torch.FloatTensor(hcqt),
            'target': torch.FloatTensor(target),
            'track_id': track_id
        }
        
        return sample
    
    def _compute_hcqt_from_audio(self, y, config):
        """Compute HCQT from audio array"""
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
    
    def _create_target_from_df(self, df, n_frames, config):
        """Create target from DataFrame of annotations"""
        target = np.zeros((config.N_BINS, n_frames), dtype=np.float32)
        frame_times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH
        )
        
        for frame_idx, frame_time in enumerate(frame_times):
            if len(df) == 0:
                continue
            time_diffs = np.abs(df['time'].values - frame_time)
            closest_idx = np.argmin(time_diffs)
            freq = df.iloc[closest_idx]['frequency']
            
            if freq > 0:
                bin_idx = hz_to_bin(freq, config)
                if 0 <= bin_idx < config.N_BINS:
                    target[bin_idx, frame_idx] = 1.0
        
        # Apply Gaussian blur
        target = gaussian_filter1d(target, sigma=config.GAUSSIAN_BLUR_SIGMA, axis=0)
        if target.max() > 0:
            target = target / target.max()
        
        return target


# ============================================================================
# 4. TRAIN/VALIDATION SPLIT (Random Split with PyTorch)
# ============================================================================

def create_random_splits(metadata_path, config=Config, seed=42):
    """
    Create random train/validation splits using PyTorch.
    
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load metadata to get all track IDs
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_track_ids = list(metadata.keys())
    n_tracks = len(all_track_ids)
    
    # Calculate split sizes
    train_size = int(n_tracks * config.TRAIN_RATIO)
    val_size = n_tracks - train_size
    
    # Split track IDs
    indices = list(range(n_tracks))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_track_ids = [all_track_ids[i] for i in train_indices]
    val_track_ids = [all_track_ids[i] for i in val_indices]
    
    # Create datasets
    # Train: extract random segments (train=True)
    # Val: use full tracks (train=False)
    train_dataset = MedleyDBPitchDataset(train_track_ids, config, train=True)
    val_dataset = MedleyDBPitchDataset(val_track_ids, config, train=False)
    
    # Print split statistics
    print("\n" + "="*60)
    print("DATASET SPLITS (Random Segments for Training)")
    print("="*60)
    print(f"{'TRAIN':>10}: {len(train_dataset):>4} tracks ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"           Random {config.SEGMENT_LENGTH}s segments per track")
    print(f"{'VAL':>10}: {len(val_dataset):>4} tracks ({config.VAL_RATIO*100:.0f}%)")
    print(f"           Full tracks for evaluation")
    print(f"{'Total':>10}: {n_tracks:>4} tracks")
    print("="*60)
    print("Note: Evaluation/vocadito will be used as final test set")
    print("="*60 + "\n")
    
    splits = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    return splits


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def save_dataset_info(splits, config=Config):

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get track IDs from datasets
    train_track_ids = splits['train'].track_ids
    val_track_ids = splits['val'].track_ids
    
    # Save track IDs
    save_path = config.OUTPUT_DIR / "split_info.npz"
    np.savez(
        save_path,
        train_track_ids=train_track_ids,
        val_track_ids=val_track_ids
    )
    print(f"Saved split information to {save_path}")


def create_dataloaders(splits, config=Config, batch_size=1):
    """
    Create PyTorch DataLoaders for train and validation.
    
    Note: batch_size=1 because tracks have variable lengths.
    You'll need custom collate_fn for batch_size > 1.
    
    Parameters:
    -----------
    splits : dict
        Dictionary with 'train', 'val' datasets
    config : Config object
        Configuration parameters
    batch_size : int
        Batch size (keep at 1 for variable-length tracks)
    
    Returns:
    --------
    dataloaders : dict
        Dictionary with 'train', 'val' DataLoaders
    """
    dataloaders = {
        'train': DataLoader(
            splits['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        ),
        'val': DataLoader(
            splits['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    }
    
    return dataloaders


def prepare_data(config=Config):
    """
    Main function to prepare data with PyTorch.
    
    This creates:
    1. Random train/val splits (85/15) using PyTorch
    2. DataLoaders for each split
    3. Saves split information for reproducibility
    """
    print("\n" + "="*60)
    print("PREPARING MEDLEYDB-PITCH DATASET WITH PYTORCH")
    print("="*60 + "\n")
    
    # Verify paths
    assert config.DATA_DIR.exists(), f"Data directory not found: {config.DATA_DIR}"
    assert config.AUDIO_DIR.exists(), f"Audio directory not found: {config.AUDIO_DIR}"
    assert config.PITCH_DIR.exists(), f"Pitch directory not found: {config.PITCH_DIR}"
    assert config.METADATA_PATH.exists(), f"Metadata not found: {config.METADATA_PATH}"
    
    # Create splits
    print("Creating random train/validation splits...")
    splits = create_random_splits(config.METADATA_PATH, config)
    
    # Save split info
    save_dataset_info(splits, config)
    
    # Create DataLoaders
    print("\nCreating PyTorch DataLoaders...")
    dataloaders = create_dataloaders(splits, config)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nDatasets are ready to use with PyTorch!")
    print("Example usage:")
    print("  for batch in dataloaders['train']:")
    print("      hcqt = batch['hcqt']")
    print("      target = batch['target']")
    print("      # ... train model ...")
    print("\nRemember: Use Evaluation/vocadito for final testing!")
    
    return dataloaders, splits


# ============================================================================
# VISUALIZATION UTILITIES (for verification)
# ============================================================================

def visualize_sample(track_id, config=Config):
    # Create a small dataset with just this track
    dataset = MedleyDBPitchDataset([track_id], config)
    sample = dataset[0]
    
    hcqt = sample['hcqt'].numpy()  # Convert tensor to numpy
    target = sample['target'].numpy()
    
    # Create output directory if needed
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Plot HCQT for all harmonics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"HCQT - Track: {track_id}", fontsize=16)
    
    for i, h in enumerate(config.HARMONICS):
        ax = axes.flat[i]
        librosa.display.specshow(
            librosa.amplitude_to_db(hcqt[i], ref=np.max),
            sr=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH,
            x_axis='time',
            y_axis='cqt_hz',
            ax=ax,
            fmin=config.FMIN * h
        )
        ax.set_title(f"Harmonic h={h}")
        ax.set_ylabel("Frequency (Hz)")
    
    plt.tight_layout()
    save_path = config.OUTPUT_DIR / f"hcqt_{track_id}.png"
    plt.savefig(save_path)
    print(f"Saved HCQT visualization to {save_path}")
    plt.close()
    
    # Plot target salience
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        target,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        x_axis='time',
        y_axis='cqt_hz',
        cmap='hot',
        fmin=config.FMIN
    )
    plt.colorbar(label='Salience')
    plt.title(f"Target Salience: {track_id}")
    plt.tight_layout()
    save_path = config.OUTPUT_DIR / f"target_{track_id}.png"
    plt.savefig(save_path)
    print(f"Saved target visualization to {save_path}")
    plt.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    # Initialize config
    config = Config()
    
    # Prepare data
    dataloaders, splits = prepare_data(config)
    
    # Test loading one batch
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    train_loader = dataloaders['train']
    for batch in train_loader:
        print(f"\nLoaded one training sample:")
        print(f"  HCQT shape: {batch['hcqt'].shape}")
        print(f"  Target shape: {batch['target'].shape}")
        print(f"  Track ID: {batch['track_id']}")
        break  # Just test one batch
    
    print("\nData is ready for training!")
