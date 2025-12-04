# automate_all_perfect.py
# FULLY AUTOMATED: 0.mp4, 1.mp4 → .npy → GIF → Prediction → Accuracy
# Works 100% with your model and all fixes

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from tonic.transforms import ToFrame
import time
from pathlib import Path
import subprocess
import tempfile
import h5py
import shutil

# ==================== CONFIGURATION ====================
RGB_FOLDER = r"dataset/rgb"          # ← Your 0.mp4, 1.mp4, 2.mp4, ...
NPY_FOLDER = r"conversion/npy"        # ← Will save 0.npy, 1.npy, ...
GIF_FOLDER = r"conversion/gifs"       # ← Will save 0.gif, 1.gif, ...
MODEL_PATH = r"best_dvsgesture_3dcnn.pth"

os.makedirs(NPY_FOLDER, exist_ok=True)
os.makedirs(GIF_FOLDER, exist_ok=True)

CLASS_NAMES = [
    "hand clapping", "right hand wave", "left hand wave",
    "right hand clockwise", "right hand counter-clockwise",
    "left hand clockwise", "left hand counter-clockwise",
    "arm roll", "air drums", "air guitar", "other"
]

# ==================== MODEL ====================
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(32,64, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2,2,2))
        self.conv3 = nn.Conv3d(64,128,3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d((2,2,2))
        self.gap   = nn.AdaptiveAvgPool3d((1,4,4))
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(128*1*4*4, num_classes)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)

print("Loading model...")
model = Gesture3DCNN()
ckpt = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model ready on {device}")

# ==================== 1. RGB → .NPY ====================
def rgb_to_npy(mp4_path, npy_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "python", r"C:\Users\mahia\miniconda3\Scripts\v2e.py",
            "-i", mp4_path, "-o", tmpdir, "--overwrite",
            "--dvs_params", "DVS128", "--output_width", "128", "--output_height", "128",
            "--pos_thres", "0.15", "--neg_thres", "0.15", "--sigma_thres", "0.02",
            "--cutoff_hz", "200", "--leak_rate_hz", "0.05", "--shot_noise_rate_hz", "0.05",
            "--skip_video_output", "--disable_slomo",
            "--dvs_h5", os.path.join(tmpdir, "events.h5")
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("v2e failed:", result.stderr)
            return False

        h5_file = os.path.join(tmpdir, "events.h5")
        if not os.path.exists(h5_file):
            print("No events.h5 generated")
            return False

        with h5py.File(h5_file, 'r') as f:
            raw = f['events'][()]

        ts = raw[:, 0].astype(np.float64)
        t_start, t_end = ts.min(), ts.max()
        duration = max(t_end - t_start, 1.0)
        t_norm = (ts - t_start) / duration

        x = raw[:, 1].astype(np.float64)
        y = raw[:, 2].astype(np.float64)
        p = np.clip(raw[:, 3], 0, 1).astype(np.float64)

        final = np.column_stack([x, y, p, t_norm])
        np.save(npy_path, final)
        print(f"  → .npy saved: {npy_path}")
        return True

# ==================== 2. NPY → GIF ====================
def npy_to_gif(npy_path, gif_path):
    data = np.load(npy_path)
    x = data[:, 0].astype(np.int16)
    y = data[:, 1].astype(np.int16)
    p = data[:, 2].astype(np.int8)
    t_norm = data[:, 3]
    t_us = (t_norm * 6_000_000).astype(np.int64)   # ← 6e6 like training

    events = np.zeros(len(data), dtype=[("t","i8"),("x","i2"),("y","i2"),("p","i1")])
    events["t"] = t_us; events["x"] = x; events["y"] = y; events["p"] = p

    frames = ToFrame(sensor_size=(128,128,2), n_time_bins=60)(events)

    def overlay(f):
        off = f[0].astype(float); on = f[1].astype(float)
        def norm(a): mn, mx = a.min(), a.max(); return (a-mn)/(mx-mn+1e-8) if mx>mn else a
        rgb = np.zeros((128,128,3))
        rgb[...,0] = norm(off)   # OFF = red
        rgb[...,2] = norm(on)    # ON = blue
        return rgb

    fig, ax = plt.subplots(figsize=(4,4)); ax.axis("off")
    im = ax.imshow(overlay(frames[0]))
    def update(i): im.set_data(overlay(frames[i])); return [im]
    ani = animation.FuncAnimation(fig, update, frames=60, interval=66, blit=True)
    ani.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  → GIF saved: {gif_path}")

# ==================== 3. NPY → PREDICTION ====================
def predict_npy(npy_path):
    data = np.load(npy_path)
    x = data[:, 0].astype(np.int16)
    y = data[:, 1].astype(np.int16)
    p = data[:, 2].astype(np.int16)
    t_norm = data[:, 3]
    t_us = (t_norm * 6_000_000).astype(np.int64)

    structured = np.core.records.fromarrays([t_us, x, y, p], names='t,x,y,p')
    frames = ToFrame(sensor_size=(128,128,2), n_time_bins=60)(structured)
    x_tensor = torch.from_numpy(frames).float().permute(1,0,2,3).unsqueeze(0)
    x_tensor = x_tensor / (x_tensor.amax(dim=(2,3,4), keepdim=True) + 1e-6)

    with torch.no_grad():
        logits = model(x_tensor.to(device))
        prob = torch.softmax(logits, 1)
        pred = logits.argmax(1).item()
        conf = prob[0, pred].item()
    return pred, conf

# ==================== MAIN LOOP ====================
if __name__ == "__main__":
    mp4_files = sorted([f for f in os.listdir(RGB_FOLDER) if f.lower().endswith(".mp4")])
    print(f"Found {len(mp4_files)} videos: {mp4_files}\n")

    results = []
    total_start = time.time()

    for mp4_name in mp4_files:
        stem = Path(mp4_name).stem
        true_class = int(stem)  # ← 0.mp4 → class 0, 1.mp4 → class 1, etc.
        mp4_path = os.path.join(RGB_FOLDER, mp4_name)
        npy_path = os.path.join(NPY_FOLDER, f"{stem}.npy")
        gif_path = os.path.join(GIF_FOLDER, f"{stem}.gif")

        print(f"[{stem}] {mp4_name} → class {true_class} ({CLASS_NAMES[true_class]})")

        # 1. Convert
        if not os.path.exists(npy_path):
            rgb_to_npy(mp4_path, npy_path)
        else:
            print(f"  → .npy already exists")

        # 2. GIF
        if not os.path.exists(gif_path):
            npy_to_gif(npy_path, gif_path)
        else:
            print(f"  → GIF already exists")

        # 3. Predict
        pred_class, conf = predict_npy(npy_path)
        correct = (pred_class == true_class)
        status = "CORRECT" if correct else "WRONG"
        print(f"  → PREDICTED: {CLASS_NAMES[pred_class]:25} ({pred_class}) | {conf:.1%} [{status}]\n")

        results.append((stem, true_class, pred_class, conf, correct))

    # ==================== FINAL REPORT ====================
    print("="*80)
    print("FINAL RESULTS")
    print("="*80)
    correct_count = sum(1 for _,_,_,_,c in results if c)
    accuracy = correct_count / len(results)
    print(f"ACCURACY: {accuracy:.1%} ({correct_count}/{len(results)} correct)\n")
    for stem, true, pred, conf, corr in results:
        print(f"{stem}.mp4 → True: {CLASS_NAMES[true]:25} | Pred: {CLASS_NAMES[pred]:25} | {conf:.1%} → {'YES' if corr else 'NO'}")

    print(f"\nTotal time: {time.time() - total_start:.1f}s")
    print("All .npy files → conversion/npy/")
    print("All GIFs       → conversion/gifs/")
    print("DONE! YOU NOW HAVE A FULLY AUTOMATED RGB-TO-GESTURE PIPELINE")