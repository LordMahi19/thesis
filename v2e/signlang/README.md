# Sign language character prediction

**Overview**

This is the first repository in a series of repository where I will create an android app that runs a machine learning model locally to predict sign language characters in real time from hand gestures shown in live camera feed. 

In each repository I will cover one fundamental step needed to make the app.

**List of repository in the series and their goal:**
1. This is the first repository.
   - Here I will create the most basic version of the model.
   - For a demo model I have created my own synthetic data for three sign language characters: A, B and L
   - Then I have collected hand landmarks coordinates of each image using [mediapipes](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
   - Then I have created a pickle dataset with hand landmark coordinate of each hand and its label.
   - Then I have used scikit-learn's randomForrestClassifier() to train the model and saved the model as model.p file.
   - run the inference_classifier.py to run and test the model.


2. [second repository](https://github.com/LordMahi19/ASL-detection)
   - Here I have trained similar model.p model but with a very large [dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data)
   - Can detect all english alphabets
   - added some UI with tkinter
4. [Third repository](https://github.com/LordMahi19/ASL-detection-tensorflow-model)
   - Here I have trained the model on the same big [dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data)
   - But created a .npz dataset out of it with same kind of data: hand landmark coordinate and label
   - With this .npz dataset I have trained tensorflow model ".h5" instead of model.p
   - Used googles ydf instead of scikit-learn. But still using a random model classifier.
   - Converted the .h5 model to .tflite model.
6. [Final repository](https://github.com/LordMahi19/ASL-detection-android)
   - Used the .tflite model.
   - created labels.txt that has all the labels with one label per line in the same order as it was during the training.

**Running the model:**
First install all the required dependency libraries
 ```bash
 pip install -r requirements.txt
 ```
Then run the following script in your terminal:
 ```bash
   python inference_classifier.py
   ```
