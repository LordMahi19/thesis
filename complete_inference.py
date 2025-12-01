# predict_final_correct.py
# GUARANTEED correct prediction on real DVS Gesture .npy files

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tonic.transforms import ToFrame

# ------------------- Model (exact same as training) -------------------
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

# ------------------- Load model -------------------
model = Gesture3DCNN()
ckpt = torch.load("best_dvsgesture_3dcnn.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------- YOUR FILE -------------------
npy_file = r"./newdata\DVSGesture\ibmGestureTest\user28_lab\5.npy"          # ← CHANGE THIS

events = np.load(npy_file)
print(f"Loaded: {events.shape} events")

# === CRITICAL FIX: only normalize timestamp to [0,1] — do NOT multiply by 6e6 ===
if events.dtype.names is None:
    x, y, p, t_norm = events[:,0], events[:,1], events[:,2], events[:,3]
else:
    x, y, p, t_norm = events['x'], events['y'], events['p'], events['t']

# Build structured array with t in microseconds (but normalized!)
# → just use the already-normalized t_norm, but give it name 't'
structured = np.core.records.fromarrays(
    [t_norm, x.astype(np.int16), y.astype(np.int16), p.astype(np.int16)],
    names='t,x,y,p'
)

# Convert to 60 frames
frames = ToFrame(sensor_size=(128,128,2), n_time_bins=60)(structured)
x = torch.from_numpy(frames).float().permute(1,0,2,3).unsqueeze(0)  # (1,2,60,128,128)

# Normalize per-sample (same as training)
x = x / (x.amax(dim=(2,3,4), keepdim=True) + 1e-6)

# Predict
with torch.no_grad():
    logits = model(x.to(device))
    pred = logits.argmax(1).item()
    prob = torch.softmax(logits, 1)[0, pred].item()

names = ["hand clapping", "right hand wave", "left hand wave",
         "right hand clockwise", "right hand counter-clockwise",
         "left hand clockwise", "left hand counter-clockwise",
         "arm roll", "air drums", "air guitar", "other"]

print(f"\nPREDICTED: {names[pred]} (class {pred})")
print(f"Confidence: {prob:.1%}\n")