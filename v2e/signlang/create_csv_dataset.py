import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import csv
from multiprocessing import Pool, cpu_count

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
CSV_FILE = 'hand_landmarks.csv' 

def process_image(args):
    dir_, img_path = args
    data_aux = []

    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

    return data_aux, dir_

if __name__ == '__main__':
    img_paths = [(dir_, img_path) for dir_ in os.listdir(DATA_DIR) for img_path in os.listdir(os.path.join(DATA_DIR, dir_))]

    # Determine the number of CPU cores to use (90% of available cores or 9 cores, whichever is less)
    num_cores = min(int(cpu_count() * 0.9), 9)

    with Pool(num_cores) as pool:
        landmarks_data = pool.map(process_image, img_paths)

    with open(CSV_FILE, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        header = ['landmark_index', 'x_coordinate', 'y_coordinate']
        csvwriter.writerow(header)

        for landmarks, label in landmarks_data:
            if landmarks:
                csvwriter.writerow(landmarks)

    print(f'Hand landmark data saved to {CSV_FILE}')
