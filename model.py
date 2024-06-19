"""#MODELING"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import joblib

# Load your data
df = pd.read_csv('C:/TugasAkhirNLP/datamusic.csv')

# Preprocess the data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Normalize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

# Save scaler and labelencoder
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(labelencoder, 'labelencoder.pkl')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Dense(units=1024, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=len(np.unique(y_train)), activation='softmax'))

# Compile the model
adam = Adam(learning_rate=1e-4)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Save the model
model.save('model.keras')


import librosa
import numpy as np

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
    # Convert features dictionary to DataFrame
    df = pd.DataFrame([features])
    # Scale the features
    df_scaled = scaler.transform(df)
    return df_scaled

def classify_genre(model, features, labelencoder, scaler):
    # Preprocess the features
    features_scaled = preprocess_features(features, scaler)
    # Predict the genre
    prediction = model.predict(features_scaled)
    # Decode the prediction to get the genre
    genre = labelencoder.inverse_transform([np.argmax(prediction)])
    return genre[0]

import librosa
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# Load the trained model, scaler, and label encoder
model = load_model('C:/TugasAkhirNLP/model.h5')
scaler = MinMaxScaler().fit(X)  # Fit with training data
labelencoder = LabelEncoder().fit(y)  # Fit with training labels

uploaded_file = 'C:/TugasAkhirNLP/blues.00000.wav'

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)
    features = extract_features(y, sr)
    genre = classify_genre(model, features, labelencoder, scaler)
    print(f"The predicted genre is: {genre}")

