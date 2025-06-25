#done and perfect

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_mfcc_features_from_audio(file_address):
    try:
        audio, sr = librosa.load(file_address, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"not able to find {file_address} — {e}")
        return None

def get_emotion_label(filename):
    try:
        emotion_code = int(filename.split("-")[2])
        emotions = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry",
            6: "fearful", 7: "disgust", 8: "surprised"
        }
        return emotions.get(emotion_code, "no idea , new emotion")
    except:
        return "error"

def extract_all_features_from_folder(folder_address):
    features = []
    for root, _, files in os.walk(folder_address):
        for file in files:
            if file.endswith(".wav"):
                full_address = os.path.join(root, file)
                mfcc = get_mfcc_features_from_audio(full_address)
                if mfcc is not None:
                    label = get_emotion_label(file)
                    features.append([*mfcc, label])
    return features

def main():
    speech_address = r"c:\Users\Lenovo\Desktop\emotion project\data\Audio_Speech_Actors_01-24" 
    song_address = r"c:\Users\Lenovo\Desktop\emotion project\data\Audio_Song_Actors_01-24 (1)"

    print("speech features...")
    speech_features = extract_all_features_from_folder(speech_address)

    print("song features...")
    song_features = extract_all_features_from_folder(song_address)

    all_features = speech_features + song_features
    print("Total samples are:", len(all_features))

    columns = [f'mfcc_{i+1}' for i in range(40)] + ['label']
    df = pd.DataFrame(all_features, columns=columns)

    if len(df) == 0:
        print("No features can be found")
        return

    print("\nExample:")
    print(df.head(1))

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)

    print("\n work is done , and csv file is saved ar train_data.csv and val_data.csv")

if __name__ == "__main__":
    main()
