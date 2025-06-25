#done and perfect

import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import tempfile

model = joblib.load(r"c:\Users\Lenovo\Desktop\emotion project\emotion_model.pkl")
label_encoder = joblib.load(r"c:\Users\Lenovo\Desktop\emotion project\label_encoder.pkl")

def extract_mfcc(file, n_mfcc=40):
    try:
        audio, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Could not process this audio file: {e}")
        return None

st.set_page_config(page_title="Speech Emotion Classifier", layout="centered")
st.title("Speech Emotion Classifier")
st.write("Upload a WAV file and get the predicted emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    features = extract_mfcc(tmp_path)

    if features is not None:
        cols = [f"mfcc_{i+1}" for i in range(len(features))]
        df = pd.DataFrame([features], columns=cols)
        prediction = model.predict(df)[0]
        emotion = label_encoder.inverse_transform([prediction])[0]
        st.success(f"Predicted Emotion: {emotion.upper()}")
