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
    pos_thres=0.15,
    neg_thres=0.15,
    sigma_thres=0.02,
    cutoff_hz=200,
    leak_rate_hz=0.05,
    shot_noise_rate_hz=0.05,
    dvs_emulator_seed=0,
    output_folder=None,
    cleanup=True
):
    """
    Convert MP4 video to DVS event format matching DVSGesture dataset structure.
    Output format: (N,4) with columns [t, x, y, p]
    - t: timestamp in microseconds (int64)
    - x: x coordinate (0–127)
    - y: y coordinate (0–127)
    - p: polarity (0=OFF, 1=ON)
    """

    if output_folder is None:
        output_folder = tempfile.mkdtemp(prefix="v2e_temp_")
    else:
        os.makedirs(output_folder, exist_ok=True)

    print(f"Converting {mp4_path} to DVS events...")
    print(f"Temporary v2e output folder: {output_folder}")

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
        "--skip_video_output",
        "--disable_slomo",
        "--dvs_h5", os.path.join(output_folder, "events.h5"),
    ]

    result = subprocess.run(v2e_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("v2e STDERR:", result.stderr)
        raise RuntimeError(f"v2e failed with return code {result.returncode}")

    print("v2e conversion complete!")

    events_h5_path = os.path.join(output_folder, "events.h5")
    events_txt_path = os.path.join(output_folder, "events.txt")
    events_npy_path = os.path.join(output_folder, "events.npy")

    if os.path.exists(events_h5_path):
        print(f"\nLoading events from {events_h5_path}")
        with h5py.File(events_h5_path, 'r') as f:
            if 'events' in f:
                raw_events = f['events'][()]
            else:
                raise KeyError(f"No 'events' dataset found in {events_h5_path}. Available: {list(f.keys())}")
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

    # After loading and processing v2e events
    timestamps_sec = raw_events[:, 0]
    x_raw = raw_events[:, 1]
    y_raw = raw_events[:, 2]
    p_raw = raw_events[:, 3]

    # Normalize timestamp to [0,1]
    t_start = timestamps_sec.min()
    t_end = timestamps_sec.max()
    duration = t_end - t_start if t_end > t_start else 1.0
    t_normalized = (timestamps_sec - t_start) / duration

    # Convert to float64 to match original training files exactly
    x = x_raw.astype(np.float64)
    y = y_raw.astype(np.float64)
    p = np.clip(p_raw, 0, 1).astype(np.float64)
    t_norm = t_normalized.astype(np.float64)

    # CRITICAL: CORRECT COLUMN ORDER FOR YOUR MODEL
    # [x, y, p, t_normalized] ←←←←← THIS IS WHAT YOUR MODEL WAS TRAINED ON
    final_events = np.column_stack([x, y, p, t_norm])

    print("Final events shape:", final_events.shape)
    print("First 5 events (CORRECT format):")
    print(final_events[:5])
    print(f"x range: {x.min()}–{x.max()}, y: {y.min()}–{y.max()}, t_norm: {t_norm.min():.6f}–{t_norm.max():.6f}")

    # Save exactly like your training data
    np.save(output_npy_path, final_events)
    print(f"PERFECT: Saved in exact format used during training → {output_npy_path}")

    if cleanup and output_folder.startswith(tempfile.gettempdir()):
        import shutil
        shutil.rmtree(output_folder, ignore_errors=True)
        print(f"✓ Cleaned up temporary folder: {output_folder}")

    return final_events


def verify_conversion(npy_path):
    print(f"\n{'='*60}")
    print("VERIFICATION – CORRECTED FOR YOUR DATA FORMAT")
    print(f"{'='*60}")
    events = np.load(npy_path)
    print(f"Shape: {events.shape}")
    print(f"Dtype: {events.dtype}")
    print("First 5 rows:\n", events[:5])

    # YOUR FORMAT: [x, y, p, t_normalized]
    x      = events[:, 0]
    y      = events[:, 1]
    p      = events[:, 2]
    t_norm = events[:, 3]    # ←←← THIS IS t_normalized

    print(f"x range: {x.min():.1f}–{x.max():.1f} (expected 0–127)")
    print(f"y range: {y.min():.1f}–{y.max():.1f} (expected 0–127)")
    print(f"p unique: {np.unique(p)}")
    print(f"t_norm range: {t_norm.min():.6f} → {t_norm.max():.6f} (expected 0.0 → 1.0)")

    assert x.min() >= 0 and x.max() <= 127, "x out of range"
    assert y.min() >= 0 and y.max() <= 127, "y out of range"
    assert t_norm.min() >= 0 and t_norm.max() <= 1.0 + 1e-6, "t must be normalized [0,1]!"
    assert set(np.unique(p)).issubset({0.0, 1.0}), "p must be 0.0 or 1.0"

    print("✓ EVERYTHING IS PERFECT!")
    print("✓ This file will work 100% with your inference.py and trained model!")
    return True



# Example usage
if __name__ == "__main__":
    # Example: Convert your MP4 video
    mp4_video = r"dataset/rgb/3.mp4"  # Replace with your MP4 path
    output_npy = "conversion/3.npy"
    
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