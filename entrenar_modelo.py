import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

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

# Cargar datos de audio y sus etiquetas desde el archivo de metadatos
def load_all_data(meta_file, audio_dir):
    meta = pd.read_csv(meta_file)
    features = []
    labels = []

    for _, row in meta.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        if os.path.exists(file_path):
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(row['category'])
        else:
            print(f"Archivo de audio no encontrado: {file_path}")

    return np.array(features), np.array(labels)

# Rutas al archivo de metadatos y carpeta de audios
meta_file = '/home/felipe/Descargas/Acus/ESC-50-master/meta/esc50.csv'
audio_dir = '/home/felipe/Descargas/Acus/ESC-50-master/audio'

# Cargar datos de audio y etiquetas
print("Cargando datos de audio...")
X, y = load_all_data(meta_file, audio_dir)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un clasificador
print("Entrenando el modelo...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
model_path = '/home/felipe/Descargas/modelo_sonido.pkl'
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")
