from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data"
ACTIVITY = {1:"WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LAYING"}
CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]

def find_dataset():
    for p in [DATA_ROOT / "UCI HAR Dataset", DATA_ROOT / "UCI HAR Dataset" / "UCI HAR Dataset"] + list(DATA_ROOT.glob("**/UCI HAR Dataset")):
        if (p / "train" / "y_train.txt").exists() and (p / "train" / "Inertial Signals").exists():
            return p
    raise FileNotFoundError("Could not find UCI HAR Dataset. Extract it under data/ so train/y_train.txt and train/Inertial Signals exist.")

def load_split(folder, split):
    signals = []
    for channel in CHANNELS:
        signals.append(np.loadtxt(folder / "Inertial Signals" / f"{channel}_{split}.txt", dtype=np.float32))
    return np.stack(signals, axis=2)  # windows x samples x 9

def extract_one(window):
    feats = []
    for j in range(window.shape[1]):
        x = window[:, j].astype(np.float64)
        dx = np.diff(x)
        feats.extend([np.mean(x), np.std(x), np.min(x), np.max(x), np.sqrt(np.mean(x*x)), np.mean(np.abs(x)), np.mean(dx), np.std(dx), np.sqrt(np.mean(dx*dx))])
    for cols in [(0,1,2), (3,4,5), (6,7,8)]:
        mag = np.sqrt(np.sum(window[:, cols] ** 2, axis=1))
        feats.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag), np.sqrt(np.mean(mag*mag))])
    return np.asarray(feats, dtype=np.float32)

def extract_features(data):
    return np.vstack([extract_one(data[i]) for i in range(data.shape[0])])

def main():
    dataset = find_dataset()
    print(f"Using dataset: {dataset}")
    train = load_split(dataset / "train", "train")
    test = load_split(dataset / "test", "test")
    y_train = np.array([ACTIVITY[int(v)] for v in np.loadtxt(dataset / "train" / "y_train.txt", dtype=int)])
    y_test = np.array([ACTIVITY[int(v)] for v in np.loadtxt(dataset / "test" / "y_test.txt", dtype=int)])
    print("Extracting features from inertial-signal windows...")
    X_train, X_test = extract_features(train), extract_features(test)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced"))
    print("Training model...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"UCI HAR held-out test accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, zero_division=0))
    out = ROOT / "live_har_model.joblib"
    joblib.dump(model, out)
    print(f"Saved model: {out}")

if __name__ == "__main__":
    main()
