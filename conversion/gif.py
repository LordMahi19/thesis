import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

import tonic
import tonic.transforms as transforms

# -----------------------------
# 1. Sensor size setup
# -----------------------------
def ensure_3d_sensor_size(sensor_size):
    if len(sensor_size) == 2:
        return (sensor_size[0], sensor_size[1], 2)
    return tuple(sensor_size)

SENSOR_SIZE = ensure_3d_sensor_size(tuple(tonic.datasets.DVSGesture.sensor_size))
H, W, C = SENSOR_SIZE
print("Sensor size:", SENSOR_SIZE)  # should be (128, 128, 2)

# -----------------------------
# 2. Load your converted gesture
# -----------------------------
data = np.load("conversion/3.npy", allow_pickle=True)
print("Data shape:", data.shape)   # (N,4)
print("First row:", data[0])

# -----------------------------
# 2a. Normalize timestamps
# -----------------------------
t = data[:,0].astype(np.int64)
t_norm = (t - t.min()) / (t.max() - t.min())   # normalize to [0,1]
t_scaled = (t_norm * 1e6).astype(np.int64)     # rescale to microseconds

# -----------------------------
# 2b. Build structured array
# -----------------------------
dtype = np.dtype([("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", np.int8)])
event_stream = np.zeros(data.shape[0], dtype=dtype)
event_stream["t"] = t_scaled
event_stream["x"] = data[:,1].astype(np.int16)
event_stream["y"] = data[:,2].astype(np.int16)
event_stream["p"] = data[:,3].astype(np.int8)

print("Structured dtype:", event_stream.dtype)
print("First structured row:", event_stream[0])
print("t range after normalization:", event_stream["t"].min(), event_stream["t"].max())

# -----------------------------
# 3. Frame conversion
# -----------------------------
N_FRAMES = 60
toframe = transforms.ToFrame(
    sensor_size=SENSOR_SIZE,
    n_time_bins=N_FRAMES
)

def events_to_frames(event_stream):
    """
    Convert event stream to frames of shape (T, 2, H, W).
    """
    frames = toframe(event_stream)
    return frames

# -----------------------------
# 4. Color overlay helper
# -----------------------------
def make_color_overlay(frame_2ch):
    off = frame_2ch[0].astype(np.float32)
    on  = frame_2ch[1].astype(np.float32)

    def norm(a):
        rng = a.max() - a.min()
        return (a - a.min()) / (rng + 1e-6)

    off_n = norm(off)
    on_n  = norm(on)

    rgb = np.zeros((frame_2ch.shape[1], frame_2ch.shape[2], 3), dtype=np.float32)
    rgb[..., 0] = off_n  # Red
    rgb[..., 2] = on_n   # Blue
    return rgb

# -----------------------------
# 5. Preview frames
# -----------------------------
def preview_event(event_stream, mode="overlay", num_cols=10):
    frames = events_to_frames(event_stream)
    T = frames.shape[0]
    num_show = min(20, T)
    num_rows = int(np.ceil(num_show / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.8*num_cols, 1.8*num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols]
        ax.axis("off")
        if i < num_show:
            f = frames[i]
            if mode == "overlay":
                img = make_color_overlay(f)
                ax.imshow(img)
            else:
                ax.imshow(f.sum(axis=0), cmap='gray')
            ax.set_title(f"t={i}")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Animate and save GIF
# -----------------------------
def animate_event(event_stream, mode="overlay", save_gif=True, out_path="./conversion/converted_gesture.gif"):
    frames = events_to_frames(event_stream)
    T = frames.shape[0]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    if mode == "overlay":
        im = ax.imshow(make_color_overlay(frames[0]))
    else:
        im = ax.imshow(frames[0].sum(axis=0), cmap='gray')

    def update(i):
        if mode == "overlay":
            im.set_data(make_color_overlay(frames[i]))
        else:
            im.set_data(frames[i].sum(axis=0))
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=80, blit=True)
    plt.close(fig)

    if save_gif:
        ani.save(out_path, writer=PillowWriter(fps=12))
        print(f"Saved GIF -> {out_path}")
    else:
        from IPython.display import display
        display(ani)
    return ani

# -----------------------------
# Run preview and animation
# -----------------------------
preview_event(event_stream, mode="overlay")
animate_event(event_stream, mode="overlay", save_gif=True, out_path="./conversion/converted_gesture.gif")
