#done & perfect

import librosa
import numpy as np
import joblib
import pandas as pd

def extract_mfcc(audio_address, n_mfcc=40):
    try:
        audio, sr = librosa.load(audio_address, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error extracting features:", e)
        return None

def predict_emotion(audio_address):
    model = joblib.load(r"c:\Users\Lenovo\Desktop\emotion project\emotion_model.pkl")
    encoder = joblib.load(r"c:\Users\Lenovo\Desktop\emotion project\label_encoder.pkl")

    features = extract_mfcc(audio_address)
    if features is None:
        print("not able to read the file properly.")
        return

    columns = [f"mfcc_{i+1}" for i in range(len(features))]
    input_data = pd.DataFrame([features], columns=columns)

    prediction = model.predict(input_data)[0]
    emotion = encoder.inverse_transform([prediction])[0]

    print("Predicted Emotion:", emotion)

if __name__ == "__main__":
    audio_address = r"c:\Users\Lenovo\Desktop\emotion project\data\Audio_Song_Actors_01-24 (1)\Actor_01\03-02-01-01-01-01-01.wav"
    predict_emotion(audio_address)
