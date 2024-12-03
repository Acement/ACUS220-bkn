import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import pathlib as Path

DURACION_GRABACION = 2  # Duración en segundos
FS = 44100  # Frecuencia de muestreo

def plot_spec(signal, sample_rate, audio_parse):
    print(audio_parse)
    stft = librosa.stft(signal)
    spec = np.abs(stft)
    spec_db = librosa.amplitude_to_db(spec)

    plt.figure(figsize=(20,8))
    
    img = librosa.display.specshow(spec_db, sr = sample_rate, cmap='inferno')
    plt.savefig(audio_parse)
    plt.close()


#
#Tienen que cambiar los directorios
#SE VA A DEMORAR, HAGAN NADA MAS O LES VA A CRASHEAR EL SO FEO
#

def main():
    audio_dir = '/home/pablo/Desktop/ACUS220/ESC-50/audio' #Directorio de audios
    spec_dir = '/home/pablo/Desktop/ACUS220/ACUS220-bkn/spec/' #Directorio de donde se guardaran las imagenes, tienen que crear la carpeta o guardarlo en una ya existente si no pesca
    audio_list = os.listdir(audio_dir) #Lista con el path de los audios
    audio_parse = [] #Iniciando lista con los path donde guardara los espectrograma



    for i in range (0,len(audio_list)):
        print("AUDIO: " + audio_list[i])
        tempName = audio_list[i].replace('.wav','')
        audio_parse.append(spec_dir + tempName + '.png') 
        print("AUDIO PARSE: " + audio_parse[i])

        signal , sample_rate = sf.read(audio_dir + '/' + audio_list[i])
        print(f'Sample rate: {sample_rate}')

        plot_spec(signal, sample_rate, audio_parse[i])


if __name__ == '__main__':
    main()
