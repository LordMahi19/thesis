# README.md – DVS Gesture 3D-CNN (81.8% on IBM DVS128 Gesture)

**Model:** `best_dvsgesture_3dcnn.pth`  
**Accuracy:** 81.82% on the official test set  
**Input:** Event data from a 128×128 DVS sensor (real or synthetic from v2e)  
**Classes:** 11 hand/arm gestures (see list below)

## Files in this package

| File                            | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `best_dvsgesture_3dcnn.pth`     | Trained weights (state_dict + accuracy)                                     |
| `dvs_gesture_3dcnn.py`          | Complete model definition + helper to load the model                        |
| `npy_format_spec.json`          | Exact .npy format specification (must be followed 100%)                    |
| `example_001.npy` (optional)    | Real sample from the original dataset (right-hand wave)                     |
| `README.md`                     | This file                                                                   |

## 11 Gesture Classes (index → name)

| Index | Gesture Name                     |
|-------|----------------------------------|
| 0     | Hand clapping                    |
| 1     | Right hand wave                  |
| 2     | Left hand wave                   |
| 3     | Right hand clockwise circle      |
| 4     | Right hand counter-clockwise     |
| 5     | Left hand clockwise circle       |
| 6     | Left hand counter-clockwise      |
| 7     | Arm roll                         |
| 8     | Air drums                        |
| 9     | Air guitar                       |
| 10    | Other / random                   |

## Step-by-step: Predict on any correctly formatted .npy file

### 1. Install dependencies (one time)

```bash
pip install torch torchvision tonic matplotlib numpy
# Optional (only if you want to convert AEDAT files yourself)
pip install aedat
```

### 2. Put everything in one folder

```
my_gesture_model/
├── best_dvsgesture_3dcnn.pth
├── dvs_gesture_3dcnn.py
├── npy_format_spec.json
└── your_file.npy          ← your own event file (must match the spec!)
```

### 3. Run prediction (single command)

```bash
python -c "
from dvs_gesture_3dcnn import load_best_model
import numpy as np, torch
from tonic.transforms import ToFrame

# Load model
model = load_best_model('best_dvsgesture_3dcnn.pth')
model.eval()

# Load your .npy file (must be (N,4) → x,y,p,t_normalized)
events = np.load('your_file.npy')
print('Loaded events:', events.shape)

# Convert to the exact tensor the model expects
transform = ToFrame(sensor_size=(128,128,2), n_time_bins=60)
frames = transform(events)                                    # (60,2,128,128)
x = torch.tensor(frames).float().permute(1,0,2,3).unsqueeze(0) # (1,2,60,128,128)
x = x / (x.amax(dim=(2,3,4), keepdim=True) + 1e-6)

# Predict
with torch.no_grad():
    logits = model(x.to(model.device))
    pred   = logits.argmax(1).item()
    prob   = torch.softmax(logits,1)[0,pred].item()

gesture_names = ['hand clapping','right hand wave','left hand wave',
                 'right hand clockwise','right hand counter-clockwise',
                 'left hand clockwise','left hand counter-clockwise',
                 'arm roll','air drums','air guitar','other']

print(f'→ Predicted: {gesture_names[pred]} (class {pred})')
print(f'→ Confidence: {prob:.1%}')
"
```

### 4. That’s it!  
You now have the prediction for any correctly formatted `.npy` file.

## Create your own compatible .npy from video (full pipeline)

```bash
# 1. Video → synthetic events
v2e --input=my_video.mp4 --output_file=my_video.aedat4 --davis_output

# 2. AEDAT4 → correct .npy (one-liner script)
python -c "
from aedat import AedatFile
import numpy as np, sys
path = 'my_video.aedat4'
ev = [[e.x, e.y, 1 if e.polarity else 0, e.timestamp] for e in AedatFile(path)['events']]
ev = np.array(ev)
t = ev[:,3]
ev[:,3] = (t-t.min())/(t.max()-t.min()+1e-12)
np.save('my_video.npy', ev)
print('Saved my_video.npy → ready for the model!')
"
```

Your `my_video.npy` can now be dropped into step 3 above and will give a correct gesture prediction.

Enjoy!  
Any questions → feel free to open an issue or contact me. Happy event-based gesture recognition!