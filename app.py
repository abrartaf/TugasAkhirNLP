from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import librosa
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = load_model('model.keras')
scaler = joblib.load('scaler.pkl')
labelencoder = joblib.load('labelencoder.pkl')

# Ensure the temp directory exists
os.makedirs('temp', exist_ok=True)

def extract_features(y, sr):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rates = librosa.feature.zero_crossing_rate(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {
        'Chroma': chroma.mean(),
        'Spectral Centroid': sc.mean(),
        'Spectral Bandwidth': spectral_bandwidths.mean(),
        'Spectral Rolloff': spectral_rolloff.mean(),
        'Zero-Crossing Rate': zero_crossing_rates.mean(),
        'Tempo': tempo
    }

    for i in range(1, 129):
        features[f'Mel_Spectrogram_{i}'] = spectrogram[i-1].mean()

    for i in range(1, 14):
        features[f'MFCC_{i}'] = mfccs[i-1].mean()

    return features

def preprocess_features(features, scaler):
    df = pd.DataFrame([features])
    df_scaled = scaler.transform(df)
    return df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a POST request")
    file = request.files.get('file')
    if file:
        print(f"File received: {file.filename}")
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        if not os.path.exists(file_path):
            print("File not saved properly")
            return jsonify({'error': 'File not saved properly'})

        print(f"File saved to {file_path}")

        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return jsonify({'error': str(e)})

        features = extract_features(y, sr)
        print(f"Extracted features: {features}")

        try:
            features_scaled = preprocess_features(features, scaler)
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            return jsonify({'error': str(e)})

        print(f"Scaled features: {features_scaled}")

        features_scaled = features_scaled.reshape(1, -1)
        print(f"Reshaped features: {features_scaled}")

        try:
            data = model.predict(features_scaled)
            predicted_class = np.argmax(data)   
            data_genre = labelencoder.inverse_transform([predicted_class])[0]
            print(f"Predicted genre: {data_genre}")
            return jsonify({'prediction': data_genre})
        except FileNotFoundError:
            return jsonify({'error': 'File not found or saved properly'})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        print("No file uploaded")
        return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
