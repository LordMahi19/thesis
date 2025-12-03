"""
Convert MP4 video to DVS event format compatible with DVS Gesture dataset
This script uses v2e to generate events and formats them to match your model's input requirements.
"""

import numpy as np
import subprocess
import os
from pathlib import Path
import tempfile
import h5py

def convert_mp4_to_dvs_events(
    mp4_path,
    output_npy_path,
    target_width=128,
    target_height=128,
    pos_thres=0.2,
    neg_thres=0.2,
    sigma_thres=0.03,
    cutoff_hz=0,
    leak_rate_hz=0.1,
    shot_noise_rate_hz=0.0,
    dvs_emulator_seed=0,
    output_folder=None,
    cleanup=True
):
    """
    Convert MP4 video to DVS event format matching DVS Gesture dataset structure.
    
    Parameters:
    -----------
    mp4_path : str
        Path to input MP4 video file
    output_npy_path : str
        Path where the output .npy file will be saved
    target_width : int
        Width of DVS sensor (128 for DVS Gesture)
    target_height : int
        Height of DVS sensor (128 for DVS Gesture)
    pos_thres : float
        Positive threshold for event generation (0.2 is default)
    neg_thres : float
        Negative threshold for event generation (0.2 is default)
    sigma_thres : float
        Temporal noise parameter
    cutoff_hz : float
        Cutoff frequency for lowpass filter (0 = no filter)
    leak_rate_hz : float
        Leak rate for DVS pixels
    shot_noise_rate_hz : float
        Shot noise rate (events/pixel/sec)
    dvs_emulator_seed : int
        Random seed for reproducibility
    output_folder : str
        Temporary folder for v2e output (auto-created if None)
    cleanup : bool
        Whether to delete temporary v2e files after conversion
    """
    
    # Create temporary directory if not specified
    if output_folder is None:
        output_folder = tempfile.mkdtemp(prefix="v2e_temp_")
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    print(f"Converting {mp4_path} to DVS events...")
    print(f"Temporary v2e output folder: {output_folder}")
    
    # Build v2e command
    v2e_cmd = [
        "C:\\Users\\mahia\\miniconda3\\python.exe", "C:\\Users\\mahia\\miniconda3\\Scripts\\v2e.py",
        "-i", mp4_path,
        "-o", output_folder,
        "--overwrite",
        "--unique_output_folder", "false",
        "--dvs_params", "DVS128",
        "--output_width", str(target_width),
        "--output_height", str(target_height),
        "--pos_thres", str(pos_thres),
        "--neg_thres", str(neg_thres),
        "--sigma_thres", str(sigma_thres),
        "--cutoff_hz", str(cutoff_hz),
        "--leak_rate_hz", str(leak_rate_hz),
        "--shot_noise_rate_hz", str(shot_noise_rate_hz),
        "--dvs_emulator_seed", str(dvs_emulator_seed),
        "--skip_video_output",  # We only need events
        "--disable_slomo", # Disable SuperSloMo due to model checkpoint mismatch
        "--dvs_h5", os.path.join(output_folder, "events.h5"), # Explicitly output events as HDF5
    ]
    
    # Run v2e
    print("\nRunning v2e...")
    result = subprocess.run(v2e_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("v2e STDERR:", result.stderr)
        raise RuntimeError(f"v2e failed with return code {result.returncode}")
    
    print("v2e conversion complete!")
    
    # Find the events file (v2e outputs events.h5, events.txt or events.npy)
    events_h5_path = os.path.join(output_folder, "events.h5")
    events_txt_path = os.path.join(output_folder, "events.txt")
    events_npy_path = os.path.join(output_folder, "events.npy")
    
    if os.path.exists(events_h5_path):
        print(f"\nLoading events from {events_h5_path}")
        with h5py.File(events_h5_path, 'r') as f:
            # Assuming the events are stored under a dataset named 'events' or similar
            # You might need to inspect the HDF5 file structure if this doesn't work
            if 'events' in f:
                raw_events = f['events'][()]
            else:
                raise KeyError(f"No 'events' dataset found in HDF5 file {events_h5_path}. Available datasets: {list(f.keys())}")
    elif os.path.exists(events_npy_path):
        print(f"\nLoading events from {events_npy_path}")
        raw_events = np.load(events_npy_path)
    elif os.path.exists(events_txt_path):
        print(f"\nLoading events from {events_txt_path}")
        raw_events = np.loadtxt(events_txt_path)
    else:
        raise FileNotFoundError(f"No events file found in {output_folder}")
    
    print(f"Raw events shape: {raw_events.shape}")
    print(f"Raw events dtype: {raw_events.dtype}")
    
    # v2e format: [timestamp(s), x, y, polarity]
    # We need: [x, y, polarity, normalized_timestamp]
    
    timestamps = raw_events[:, 0]
    x = raw_events[:, 1].astype(np.int16)
    y = raw_events[:, 2].astype(np.int16)
    polarity = raw_events[:, 3].astype(np.int16)
    
    # Normalize timestamps to [0, 1]
    t_min = timestamps.min()
    t_max = timestamps.max()
    t_normalized = (timestamps - t_min) / (t_max - t_min) if t_max > t_min else timestamps
    
    print(f"\nTimestamp normalization:")
    print(f"  Original range: [{t_min:.6f}, {t_max:.6f}] seconds")
    print(f"  Normalized range: [{t_normalized.min():.6f}, {t_normalized.max():.6f}]")
    print(f"  Duration: {t_max - t_min:.3f} seconds")
    
    # Create output array in DVS Gesture format: [x, y, polarity, normalized_t]
    output_events = np.column_stack([
        x.astype(np.float64),
        y.astype(np.float64),
        polarity.astype(np.float64),
        t_normalized.astype(np.float64)
    ])
    
    print(f"\nOutput events shape: {output_events.shape}")
    print(f"Output events dtype: {output_events.dtype}")
    print(f"First 5 events:\n{output_events[:5]}")
    print(f"\nEvent statistics:")
    print(f"  Total events: {len(output_events)}")
    print(f"  X range: [{x.min()}, {x.max()}]")
    print(f"  Y range: [{y.min()}, {y.max()}]")
    print(f"  Polarity distribution: {np.bincount(polarity)}")
    
    # Compute temporal statistics (like your analysis)
    dt = np.diff(t_normalized)
    print(f"\nTemporal statistics (Δt between events):")
    print(f"  Median Δt: {np.median(dt):.6f}")
    print(f"  Mean Δt: {dt.mean():.6f}")
    print(f"  Min Δt: {dt.min():.6f}")
    print(f"  Max Δt: {dt.max():.6f}")
    print(f"  99th percentile: {np.percentile(dt, 99):.6f}")
    
    # Save to .npy file
    np.save(output_npy_path, output_events)
    print(f"\n✓ Saved DVS events to: {output_npy_path}")
    
    # Cleanup temporary files
    if cleanup and output_folder.startswith(tempfile.gettempdir()):
        import shutil
        shutil.rmtree(output_folder, ignore_errors=True)
        print(f"✓ Cleaned up temporary folder: {output_folder}")
    
    return output_events


def verify_conversion(npy_path):
    """
    Verify that the converted file matches DVS Gesture format.
    """
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    events = np.load(npy_path)
    
    print(f"Shape: {events.shape}")
    print(f"Dtype: {events.dtype}")
    print(f"Expected format: (N, 4) with columns [x, y, polarity, normalized_t]")
    
    # Check format
    assert events.ndim == 2 and events.shape[1] == 4, "Shape must be (N, 4)"
    assert events.dtype == np.float64, "Dtype must be float64"
    
    x = events[:, 0].astype(int)
    y = events[:, 1].astype(int)
    p = events[:, 2].astype(int)
    t_norm = events[:, 3]
    
    print(f"\n✓ Format checks passed!")
    print(f"  X range: [{x.min()}, {x.max()}] (expected: [0, 127])")
    print(f"  Y range: [{y.min()}, {y.max()}] (expected: [0, 127])")
    print(f"  Polarity values: {np.unique(p)} (expected: [0, 1])")
    print(f"  Timestamp range: [{t_norm.min():.6f}, {t_norm.max():.6f}] (expected: [0, 1])")
    
    print(f"\n✓ File is ready for inference!")
    return True


# Example usage
if __name__ == "__main__":
    # Example: Convert your MP4 video
    mp4_video = r"conversion/wave.mp4"  # Replace with your MP4 path
    output_npy = "conversion/converted_gesture.npy"
    
    # Check if v2e is installed and accessible
    try:
        subprocess.run(["C:\\Users\\mahia\\miniconda3\\python.exe", "C:\\Users\\mahia\\miniconda3\\Scripts\\v2e.py", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: v2e.py is not found at C:\\Users\\mahia\\miniconda3\\Scripts\\v2e.py or is not executable.")
        print("Please ensure v2e is correctly installed in your Miniconda environment.")
        exit(1)
    
    if not os.path.exists(mp4_video):
        print(f"Please update 'mp4_video' path. File not found: {mp4_video}")
        print("\nUsage:")
        print("  events = convert_mp4_to_dvs_events('path/to/video.mp4', 'output.npy')")
        exit(1)
    
    # Convert video to DVS events
    events = convert_mp4_to_dvs_events(
        mp4_path=mp4_video,
        output_npy_path=output_npy,
        target_width=128,
        target_height=128,
        pos_thres=0.2,  # Adjust these thresholds based on your video
        neg_thres=0.2,
        sigma_thres=0.03
    )
    
    # Verify the conversion
    verify_conversion(output_npy)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"Use this file in your inference script:")
    print(f'  npy_file = r"{output_npy}"')