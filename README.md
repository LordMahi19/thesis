# DVS Gesture 3D-CNN (93.9% on IBM DVS128 Gesture)

**Model:** `best_dvsgesture_3dcnn.pth`  
**Accuracy:** 93.9% on the official test set  
**Input:** Event data from a 128×128 DVS sensor (real or synthetic from v2e)  
**Classes:** 11 hand/arm gestures (see list below)

## Files in this project

| File                        | Description                                                       |
| --------------------------- | ----------------------------------------------------------------- |
| `best_dvsgesture_3dcnn.pth` | Trained model weights (state_dict + accuracy).                    |
| `train_export_eval.ipynb`   | The complete Python script for training and evaluating the model. |
| `complete_inference.py`     | A script to run inference on a single `.npy` file.                |
| `npy_structure.json`        | Detailed specification of the required `.npy` file format.        |
| `examining_data.ipynb`      | A Jupyter Notebook for exploring and visualizing the dataset.     |
| `newdata/`                  | Directory containing the DVS Gesture dataset.                     |
| `README.md`                 | This file.                                                        |

## 11 Gesture Classes (index → name)

| Index | Gesture Name                 |
| ----- | ---------------------------- |
| 0     | Hand clapping                |
| 1     | Right hand wave              |
| 2     | Left hand wave               |
| 3     | Right hand clockwise circle  |
| 4     | Right hand counter-clockwise |
| 5     | Left hand clockwise circle   |
| 6     | Left hand counter-clockwise  |
| 7     | Arm roll                     |
| 8     | Air drums                    |
| 9     | Air guitar                   |
| 10    | Other / random               |

## Step-by-step: Predict on any correctly formatted .npy file

### 1. Install dependencies

```bash
pip install torch torchvision tonic numpy
```

### 2. Run prediction

The `complete_inference.py` script is set up to run prediction on a sample file.

1.  **Open `complete_inference.py`** in a text editor.
2.  **Change the `npy_file` variable** to the path of your `.npy` file.

    ```python
    # ------------------- YOUR FILE -------------------
    npy_file = r"./path/to/your/file.npy"          # ← CHANGE THIS
    ```

3.  **Run the script:**

    ```bash
    conda env create -f thesis_environment.yml
    conda activate thesis
    python complete_inference.py
    ```

The script will print the predicted gesture and the confidence level.

## Training the model

The `full_working_code.py` script contains the entire pipeline for training and evaluating the model on the DVS Gesture dataset.

### 1. Install dependencies

In addition to the dependencies for inference, you will need `scikit-learn`, `seaborn`, and `matplotlib`.

```bash
pip install torch torchvision tonic numpy scikit-learn seaborn matplotlib
```

### 2. Dataset

The script expects the DVS Gesture dataset to be in the `newdata/DVSGesture` directory, with `ibmGestureTrain` and `ibmGestureTest` subdirectories.

### 3. Run training

To start training, simply run the script:

```bash
python full_working_code.py
```

The script will train the model, save the best performing version to `best_dvsgesture_3dcnn.pth`, and display training progress, final classification report, and a confusion matrix.

## Creating compatible .npy files

To run inference or train the model on your own data, you need to convert it to the `.npy` format specified in `npy_structure.json`.

### From video

You can convert a standard video file to synthetic DVS events using a tool like [v2e](https://github.com/neuromorphs/v2e).

```bash
# 1. Video → synthetic events
v2e --input=my_video.mp4 --output_file=my_video.aedat4 --davis_output
```

### From AEDAT4 to .npy

The `npy_structure.json` file contains a Python snippet to convert `.aedat4` files to the correct `.npy` format. Refer to the `how_to_convert_from_aedat4_v2e` section in that file.

Enjoy!
