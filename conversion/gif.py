import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

import tonic
import tonic.transforms as transforms

# -----------------------------
# 1. Sensor size
# -----------------------------
SENSOR_SIZE = (128, 128, 2)   # DVS128
H, W, C = SENSOR_SIZE

# -----------------------------
# 2. Load your CORRECTLY converted .npy
# -----------------------------
data = np.load("conversion/3.npy")
print("Data shape:", data.shape)
print("First 5 rows:\n", data[:5])

# -----------------------------
# 2. CORRECT COLUMN MAPPING FOR YOUR DATA FORMAT
# -----------------------------
# Your format: column 0 = x, 1 = y, 2 = p, 3 = t_normalized [0,1]

x = data[:, 0].astype(np.int16)
y = data[:, 1].astype(np.int16)
p = data[:, 2].astype(np.int8)        # 0.0 or 1.0 → will be converted to 0/1
t_norm = data[:, 3]

# Convert normalized time [0,1] → microseconds (Tonic expects real timestamps)
t_us = (t_norm * 1e6).astype(np.int64)   # 1e6 is standard for ~1 second gestures

# Build proper structured array
events_structured = np.zeros(len(data), dtype=[("t", "i8"), ("x", "i2"), ("y", "i2"), ("p", "i1")])
events_structured["t"] = t_us
events_structured["x"] = x
events_structured["y"] = y
events_structured["p"] = p

print("Structured array created:")
print("t range (µs):", events_structured["t"].min(), "→", events_structured["t"].max())
print("x range:", x.min(), "→", x.max())
print("y range:", y.min(), "→", y.max())
print("p unique:", np.unique(p))

# -----------------------------
# 3. ToFrame transform
# -----------------------------
N_FRAMES = 60
toframe = transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=N_FRAMES)

frames = toframe(events_structured)  # shape: (60, 2, 128, 128)

# -----------------------------
# 4. Color overlay (ON = blue, OFF = red)
# -----------------------------
def make_overlay(frame_2ch):
    off = frame_2ch[0].astype(float)
    on  = frame_2ch[1].astype(float)
    
    def normalize(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-8) if mx > mn else a
    
    off_n = normalize(off)
    on_n  = normalize(on)
    
    rgb = np.zeros((H, W, 3), dtype=float)
    rgb[..., 0] = off_n   # Red channel  → OFF events
    rgb[..., 2] = on_n    # Blue channel → ON events
    rgb[..., 1] = (off_n + on_n) * 0.3  # slight green mix for visibility
    return rgb

# -----------------------------
# 5. Preview static frames
# -----------------------------
def preview():
    num_show = 20
    cols = 10
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3))
    axes = axes.flatten()
    
    for i in range(num_show):
        ax = axes[i]
        rgb = make_overlay(frames[i])
        ax.imshow(rgb)
        ax.set_title(f"Frame {i}", fontsize=8)
        ax.axis("off")
    for j in range(num_show, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Create GIF
# -----------------------------
def save_gif(path="conversion/converted_gesture.gif", fps=12):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    im = ax.imshow(make_overlay(frames[0]))
    
    def update(i):
        im.set_data(make_overlay(frames[i]))
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=True)
    
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF saved → {path}")

# -----------------------------
# RUN
# -----------------------------
preview()
save_gif("conversion/converted_gesture_corrected.gif", fps=15)