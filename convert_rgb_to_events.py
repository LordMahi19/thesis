import os
import subprocess
from pathlib import Path

def convert_videos():
    # Configuration
    input_dir = Path("./dataset")
    output_dir = Path("./dataset/generated_events")
    v2e_script = Path("v2e/v2e.py")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all mp4 files
    video_files = sorted(list(input_dir.glob("*.mp4")))
    
    print(f"Found {len(video_files)} videos to convert.")
    
    for video_path in video_files:
        print(f"Processing {video_path}...")
        
        # Each output goes into its own folder
        video_name = video_path.stem  # e.g., "1"
        current_output_folder = output_dir / video_name
        current_output_folder.mkdir(parents=True, exist_ok=True)

        # Define output filenames for aedat4 + h5
        aedat4_out_name = f"{video_name}.aedat4"
        h5_out_name = f"{video_name}.h5"
        # Construct v2e command
        cmd = [
            "python", str(v2e_script),
            "-i", str(video_path),
            "--output_folder", str(current_output_folder),

            "--dvs128",
            "--pos_thres=0.15",
            "--neg_thres=0.15",
            "--sigma_thres=0.03",
            "--cutoff_hz=15",
            "--timestamp_resolution=0.003",

            # ⭐ REQUIRED explicit output paths
            "--dvs_aedat4", aedat4_out_name,
            "--dvs_h5", h5_out_name,

            "--overwrite",
            "--no_preview"
        ]
        
        print(f"Running command:\n{' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully converted {video_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {video_path}: {e}")

if __name__ == "__main__":
    convert_videos()
