# Speech Emotion Classification from Audio

The project involves emotion identification from audio records of speech and song. The system uses machine learning and can identify various emotional tones such as happy, sad, angry, and so forth using patterns in the voice.

---

## Dataset Used

The dataset employed is RAVDESS, which involves `.wav` recordings read or sung by 24 actors. Every file in the dataset has the emotion it is representing.

The following folders were employed from the dataset:
- Audio_Speech_Actors_01-24
- Audio_Song_Actors_01-24 (1)

Each file name contains emotion information.

---
## Feature Extraction and Pre-processing

A total of 102 features were extracted for each `.wav` file to describe how the voice sounds. They include:

- MFCCs (40 values): Capture the general sound and tone
- Delta MFCCs (40): Record how the tone varies over time
- Spectral contrast: Assists with separating different frequency bands
- Chroma features: Pertain to patterns of musical pitch
- RMS energy: Captures how loud the voice is
- Zero crossing rate: Records the level of noise
- Audio envelope: Provides an idea of how the voice amplitude differs

Apart from this, data augmentation was performed to make the model stronger. Three additional copies of each file were prepared by:
- Pitch change
- Slight speed stretching
- Adding slight background noise

This served to boost the diversity in training data.

---

## Data Splitting

The dataset was divided into:
- 80% for training
- 20% for validation

Stratified split was employed to ensure that each emotion was represented evenly in both sets.

---

## Model Pipeline

1. All audio files were feature-extracted
2. The labels (emotions) were encoded
3. Feature values were scaled
4. A Random Forest Classifier was trained with a parameter tuning method (GridSearchCV)
5. The best model was stored, along with the label encoder and scaler

Hyperparameters were tuned to maximize F1 score and accuracy.

---

## Accuracy and Results

The model was tested with the validation data. Performance was evaluated using common classification metrics.

- F1 Score (weighted): 0.84
- Overall Accuracy: 0.84
- Accuracy for every(6) class: Greater than 75% for every primary class(except fearful)

A complete classification report and confusion matrix are printed upon training.

F1 Score: 0.84
Overall Accuracy: 0.84

Class-wise Accuracy:
angry: 0.95
calm: 0.86
fearful: 0.68
fearful class accuracy is below 75%
happy: 0.84
neutral: 1.00
sad: 0.78

---

## Files Included

- `shreya_emotion_features_extractor.py`: Extracts all features and formats the dataset
- `train_model.py`: Trains the classifier and tests it
- `test_model.py`: Makes a prediction for one audio file (which can be converted into train-model.ipynb if req)
- `app.py`: A web app to test the model in real-time using Streamlit
- `train_data.csv` and `val_data.csv`: Training and validation data
- `emotion_model.pkl`, `label_encoder.pkl`, `scaler.pkl`: Saved model and auxiliary files

----
## Video Demo
https://drive.google.com/file/d/1FiDF8MXcpMMmUDZ6VKO2Wr8aBalClbK_/view?usp=sharing

---
## Testing the Model

To test a particular `.wav` file:
```bash
python test_model.py
```

