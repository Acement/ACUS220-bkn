import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image

# Función para extraer características de espectrogramas
def extract_features_from_spectrogram(image_path, target_size=(128, 128)):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        return image_array.flatten()
    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")
        return None

# Reconocer sonidos a partir de espectrogramas
def recognize_sounds(spectrogram_dir, model_path, text_widget):
    model = joblib.load(model_path)
    recognized_sounds = []

    for file_name in os.listdir(spectrogram_dir):
        file_path = os.path.join(spectrogram_dir, file_name)
        
        if os.path.isfile(file_path) and file_path.endswith('.png'):
            features = extract_features_from_spectrogram(file_path)
            if features is not None:
                prediction = model.predict([features])[0]
                recognized_sounds.append((file_name, prediction))
                text_widget.insert(tk.END, f"🎶 Archivo: {file_name} - Categoría reconocida: {prediction}\n")

    if not recognized_sounds:
        text_widget.insert(tk.END, "No se reconocieron sonidos en la carpeta.\n")

# Configurar interfaz gráfica
def setup_ui(root):
    root.title("Reconocimiento de Sonidos 🎧")
    root.geometry("800x650")
    root.minsize(800, 650)
    root.config(bg="#1f1f2e")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(3, weight=1)

    # Encabezado
    tk.Label(root, text="🔊 Reconocimiento Automático de Sonidos", font=("Helvetica", 14, "bold"), bg="#1f1f2e", fg="white").grid(row=0, column=0, pady=(20, 10), sticky="ew")

    # Campos para seleccionar carpeta y modelo
    entry_spectrogram_dir = tk.Entry(root, font=("Helvetica", 10), bg="#2d2d3d", fg="#ffffff", insertbackground="white")
    entry_model_path = tk.Entry(root, font=("Helvetica", 10), bg="#2d2d3d", fg="#ffffff", insertbackground="white")

    # Selección de espectrogramas
    tk.Label(root, text="📂 Carpeta de Sonidos:", font=("Helvetica", 11, "bold"), bg="#1f1f2e", fg="white").grid(row=1, column=0, padx=15, pady=5, sticky="w")
    entry_spectrogram_dir.grid(row=2, column=0, padx=15, pady=5, sticky="ew")
    tk.Button(root, text="Seleccionar", command=lambda: select_folder(entry_spectrogram_dir), font=("Helvetica", 11, "bold"), bg="#3498db", fg="white").grid(row=3, column=0, padx=15, pady=5)

    # Selección del modelo
    tk.Label(root, text="🤖 Modelo:", font=("Helvetica", 11, "bold"), bg="#1f1f2e", fg="white").grid(row=4, column=0, padx=15, pady=5, sticky="w")
    entry_model_path.grid(row=5, column=0, padx=15, pady=5, sticky="ew")
    tk.Button(root, text="Seleccionar", command=lambda: select_file(entry_model_path), font=("Helvetica", 11, "bold"), bg="#5cb85c", fg="white").grid(row=6, column=0, padx=15, pady=5)

    # Botón para iniciar reconocimiento
    tk.Button(root, text="Iniciar Reconocimiento 🎬", command=lambda: start_recognition(entry_spectrogram_dir.get(), entry_model_path.get()), font=("Helvetica", 11, "bold"), bg="#e67e22", fg="white").grid(row=7, column=0, pady=15)

    # Área de texto
    text_results = ScrolledText(root, bg="#333344", fg="white", wrap=tk.WORD)
    text_results.grid(row=8, column=0, padx=15, pady=5, sticky="nsew")
    return text_results

# Seleccionar carpeta
def select_folder(entry):
    folder = filedialog.askdirectory(title="Seleccionar Carpeta")
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

# Seleccionar archivo
def select_file(entry):
    file = filedialog.askopenfilename(title="Seleccionar Archivo", filetypes=[("Model Files", "*.pkl")])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)

# Iniciar reconocimiento
def start_recognition(spectrogram_dir, model_path):
    if not spectrogram_dir or not model_path:
        messagebox.showerror("Error", "Selecciona un directorio y un modelo")
        return
    recognize_sounds(spectrogram_dir, model_path, text_results)

# Configuración de la ventana principal
root = tk.Tk()
text_results = setup_ui(root)
root.mainloop()
