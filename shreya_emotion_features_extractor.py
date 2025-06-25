import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_emotion_label(filename):
    try:
        code = int(filename.split("-")[2])
        labels = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry",
            6: "fearful", 7: "disgust", 8: "surprised"
        }
        return labels.get(code, "unknown")
    except:
        return "unknown"

def extract_features(y, sr):
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    feats.extend(mfcc_mean)

    delta = librosa.feature.delta(mfcc)
    feats.extend(np.mean(delta.T, axis=0))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend(np.mean(contrast.T, axis=0))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma.T, axis=0))

    rms = librosa.feature.rms(y=y)
    feats.append(np.mean(rms))

    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(np.mean(zcr))

    envelope = np.abs(y)
    feats.append(np.std(envelope))

    return feats

def extract_from_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        all_feats = []

        original = extract_features(y, sr)
        all_feats.append(original)

        pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        all_feats.append(extract_features(pitch, sr))

        stretch = librosa.effects.time_stretch(y, rate=1.1)
        all_feats.append(extract_features(stretch, sr))

        noise = y + np.random.normal(0, 0.005, len(y))
        all_feats.append(extract_features(noise, sr))

        return all_feats
    except Exception as e:
        print(f"Failed on {file_path}: {e}")
        return []

def extract_all_features_from_folder(folder_path):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                label = get_emotion_label(file)
                if label != "unknown":
                    variants = extract_from_file(full_path)
                    for features in variants:
                        data.append([*features, label])
    return data

def main():
    speech_path = r"c:\Users\Lenovo\Desktop\emotion project\data\Audio_Speech_Actors_01-24"
    song_path = r"c:\Users\Lenovo\Desktop\emotion project\data\Audio_Song_Actors_01-24 (1)"

    print("Loading speech data...")
    speech_data = extract_all_features_from_folder(speech_path)

    print("Loading song data...")
    song_data = extract_all_features_from_folder(song_path)

    all_data = speech_data + song_data
    print(f"Total samples: {len(all_data)}")

    columns = [f'feat_{i+1}' for i in range(102)] + ['label']
    df = pd.DataFrame(all_data, columns=columns)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)

    print("train_data.csv and val_data.csv saved.")

if __name__ == "__main__":
    main()
