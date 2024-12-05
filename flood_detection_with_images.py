import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def extract_features(image_path):
    with Image.open(image_path) as img:
        img = img.resize((100, 100))
        img = img.convert('L')
        img_array = np.array(img)
        features = img_array.flatten()
        features = np.append(features, [np.mean(img_array), np.std(img_array)])
    return features

def load_dataset(data_dir, csv_file, image_col, label_col):
    df = pd.read_csv(csv_file)
    features = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(data_dir, row[image_col])
        if os.path.exists(image_path):
            features.append(extract_features(image_path))
            labels.append(row[label_col])
    return np.array(features), np.array(labels)

def train_model(train_dir, train_csv, test_dir, test_csv, image_col, label_col):
    print("Loading and preparing the dataset...")
    X, y = load_dataset(train_dir, train_csv, image_col, label_col)
    X_test, y_test = load_dataset(test_dir, test_csv, image_col, label_col)

    print("Splitting the training data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train_resampled)

    print("Evaluating the model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Flood', 'Flood'], zero_division=1))

    print("Saving the model...")
    joblib.dump(model, 'flood_prediction_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')

    return model, scaler

def predict_flood(model, scaler, image_path):
    features = extract_features(image_path)
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    return "Flood detected" if prediction == 1 else "No flood detected"

if __name__ == "__main__":
    # Define paths to your dataset
    train_dir = 'train'
    train_csv = 'train.csv'
    test_dir = 'test'
    test_csv = 'test.csv'
    
    # Define column names for image filenames and labels
    image_col = 'Image ID'  # Replace with your actual column name for image filenames
    label_col = 'Flooded'  # Replace with your actual column name for labels
    
    model, scaler = train_model(train_dir, train_csv, test_dir, test_csv, image_col, label_col)

print("Improved flood prediction model created successfully.")