[README.md](https://github.com/user-attachments/files/33302115/README.md)
# Human Activity Recognition Using a Smartphone

A Streamlit web application that uses a smartphone's motion sensors to recognize human activities in real time. The app collects motion data through the browser, extracts features from short sensor windows, and uses a trained machine-learning model to predict the activity.

## Features

- Live motion-sensor data collection in a supported browser.
- Human activity prediction using a trained model.
- Prediction confidence percentage when the model supports `predict_proba`.
- Simple Streamlit web interface.

> **Note:** Prediction confidence is not the same as measured model accuracy. Actual accuracy should be calculated separately using labelled test data.

## Project Structure

Keep the following files and folders together in the project root:

```text
human_activity_recognition_/
├── app.py
├── live_har_model.joblib
├── requirements.txt
├── motion_sensor_frontend/
│   └── index.html
├── train_live_model.py
└── demo_samples.csv
```

The `motion_sensor_frontend/` folder may contain additional frontend files. Do not remove them if the app uses them.

## Requirements

- Python 3.10 or a version supported by the packages in `requirements.txt`.
- pip
- A modern browser.
- A smartphone with motion sensors for live detection.
- HTTPS for browser motion-sensor access in deployed environments. Localhost is generally treated as a secure context for development.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/karanamsrujana/human_activity_recognition_.git
cd human_activity_recognition_
```

### 2. Create and activate a virtual environment (recommended)

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check required files

Confirm that `app.py`, `live_har_model.joblib`, `requirements.txt`, and the `motion_sensor_frontend/` folder are present in the project root.

### 5. Start the app

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in your browser.

## Use Live Activity Detection

1. Open the app on a supported smartphone browser when possible.
2. Allow motion-sensor access if the browser asks for permission.
3. Start detection using the controls shown in the app.
4. Hold or move the phone naturally while sensor readings are collected.
5. Wait for a sensor window to be processed; the predicted activity and confidence (if supported by the model) will appear.
6. Stop detection when finished.

Sensor access depends on the device, browser, and permissions. On many mobile browsers, motion sensors require HTTPS. If sensor access is unavailable, try Chrome on Android and make sure the site has permission to access motion sensors.

## Deploy on Streamlit Community Cloud

1. Push the project files and folders to your GitHub repository.
2. Confirm that the main script is named exactly `app.py`.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), create or open the app.
4. Select the correct repository and branch.
5. Set **Main file path** to `app.py` if the file is in the repository root.
6. Deploy or reboot the app.
7. Open the deployed HTTPS URL and grant sensor permission when prompted.

Keep `live_har_model.joblib` and `motion_sensor_frontend/` in the same project root as `app.py`, unless the code has been updated to use different paths.

## Troubleshooting

### `Main module does not exist`

- Check that the file is named exactly `app.py`.
- Check the selected branch and repository in Streamlit settings.
- If `app.py` is inside a subfolder, set the Main file path to that relative path (for example, `main/app.py`).

### Model file is missing

- Ensure `live_har_model.joblib` is committed to GitHub and located where `app.py` expects it.
- Check spelling and capitalization; filenames are case-sensitive on deployment servers.

### Motion sensors are unsupported or permission is denied

- Use a supported smartphone and browser.
- Grant the browser permission to access motion sensors.
- Use the deployed HTTPS URL; sensor APIs may not work on insecure origins.
- Check that the `motion_sensor_frontend/` folder is committed to the repository.

### Missing Python packages

- Confirm that dependencies are listed in `requirements.txt`.
- After changing dependencies, commit the updated file and redeploy.

## Model and Data Notes

- `live_har_model.joblib` is the trained model loaded by the app.
- `train_live_model.py` is included in the repository for model-training workflow.
- `demo_samples.csv` is included in the repository; consult the training script or project code for its exact intended use.
- The feature extraction code expects sensor windows containing nine channels and at least 32 samples. Ensure any replacement model is trained with compatible features and the same feature order.

## License

Add a license here if you intend to distribute this project publicly.
