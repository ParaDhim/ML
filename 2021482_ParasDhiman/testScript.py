# test_script.py

import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

def read_dict_from_text(filename):
    dict2 = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(':')
            dict2[int(key)] = value.strip()
    return dict2

# Example usage:
dict2_filename = "dict2.txt"
dict2 = read_dict_from_text(dict2_filename)

# Load the trained models and other required parameters
data_folder = "/content/drive/MyDrive/KAGGLE-2/SpeechCommand"
Class_mapping = {'right': 0, 'eight': 1, 'cat': 2, 'tree': 3, 'bed': 4, 'happy': 5, 'go': 6, 'dog': 7, 'no': 8,
                 'wow': 9, 'nine': 10, 'left': 11, 'stop': 12, 'three': 13, 'sheila': 14, 'one': 15, 'bird': 16,
                 'zero': 17, 'seven': 18, 'up': 19, 'marvin': 20, 'two': 21, 'house': 22, 'down': 23, 'six': 24,
                 ' yes ': 25, 'on': 26, 'five': 27, 'off': 28, 'four': 29}

order = 37
fs = 16000
frame_period = 5

import joblib

def load_models():
    models = []
    for i in range(30):
        model_filename = f"model_{i}.pkl"
        gm = joblib.load(model_filename)
        models.append(gm)

    return models


def extract_features(file_path):
    x, fs = librosa.load(file_path)
    x = librosa.effects.preemphasis(x)
    
    # Extract chroma and spectral contrast features
    chroma = librosa.feature.chroma_stft(y=x, sr=fs)
    spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=fs)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=17)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    # Combine all features
    combined_features = np.vstack([mfcc, delta, delta_delta, chroma, spectral_contrast])
    
    return combined_features.T

def predict_class(test_folder_path, models):
    outputs = {'ID': [], 'TARGET': []}
    df_final = pd.DataFrame(outputs)

    testing_file_paths = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path) if file.endswith(".wav")]

    for i, test_file in enumerate(testing_file_paths):
        test_features = extract_features(test_file)
        likelihoods = [model.score(test_features) for model in models]
        best_model_index = np.argmax(likelihoods)
        predicted_word = Class_mapping[dict2[best_model_index]]
        df_final = df_final.append({'ID': i, 'TARGET': predicted_word}, ignore_index=True)

    return df_final

if __name__ == "__main__":
    # Specify the folder path containing test audio files
    test_folder_path = input("Enter the path of the Folder: ")

    # Load the trained models
    models = load_models()

    # Perform predictions
    predictions = predict_class(test_folder_path, models)

    # Save predictions to CSV
    predictions.to_csv("predictions.csv", index=False)

    print("Predictions saved to predictions.csv")
