import os
import librosa
import numpy as np
import pandas as pd
import joblib  # Para cargar el modelo entrenado

# Función para extraer características MFCC de un archivo de sonido
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error al procesar {file_name}: {e}")
        return None

# Función para reconocer sonidos en una carpeta
def recognize_sounds(audio_dir, model_path):
    # Cargar el modelo entrenado
    model = joblib.load(model_path)

    recognized_sounds = []

    # Procesar cada archivo en la carpeta de audio
    for file_name in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file_name)
        
        # Solo procesar archivos que sean de sonido (ej. .wav)
        if os.path.isfile(file_path) and file_path.endswith(('.wav', '.mp3')):
            mfccs = extract_features(file_path)
            if mfccs is not None:
                # Predecir la categoría del sonido usando el modelo
                prediction = model.predict([mfccs])[0]
                recognized_sounds.append((file_name, prediction))

    return recognized_sounds

# Ruta a la carpeta de audios y al modelo entrenado
audio_dir = '/home/felipe/Descargas/Acus/ESC-50-master/audio'
model_path = '/home/felipe/Descargas/modelo_sonido.pkl'

# Reconocer sonidos en la carpeta y mostrar resultados
print("Reconociendo sonidos...")
sounds = recognize_sounds(audio_dir, model_path)

if not sounds:
    print("No se reconocieron sonidos en la carpeta.")
else:
    print("Lista de sonidos reconocidos:")
    for file_name, category in sounds:
        print(f"Archivo: {file_name} - Categoría reconocida: {category}")
