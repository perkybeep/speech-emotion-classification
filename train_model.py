import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_val = val_df.drop("label", axis=1)
y_val = val_df["label"]

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

param_grid = {
    'n_estimators': [200, 250],
    'max_depth': [20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train_enc)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_val_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val_enc, y_pred))

print("\nClassification Report:")
unique_labels = sorted(np.unique(y_val_enc))
target_names = label_encoder.inverse_transform(unique_labels)
print(classification_report(y_val_enc, y_pred, target_names=target_names))

f1 = f1_score(y_val_enc, y_pred, average="weighted")
acc = accuracy_score(y_val_enc, y_pred)

print(f"\nF1 Score: {f1:.2f}")
print(f"Overall Accuracy: {acc:.2f}")

print("\nClass-wise Accuracy:")
for i in unique_labels:
    mask = (y_val_enc == i)
    class_acc = accuracy_score(y_val_enc[mask], y_pred[mask])
    class_label = label_encoder.inverse_transform([i])[0]
    print(f"{class_label}: {class_acc:.2f}")
    if class_acc < 0.75:
        print(f"{class_label} class accuracy is below 75%")

joblib.dump(best_model, r"c:\Users\Lenovo\Desktop\emotion project\emotion_model.pkl")
joblib.dump(label_encoder, r"c:\Users\Lenovo\Desktop\emotion project\label_encoder.pkl")
joblib.dump(scaler, r"c:\Users\Lenovo\Desktop\emotion project\scaler.pkl")

print("\nModel, encoder, and scaler saved.")
