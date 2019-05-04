# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:53:03 2019

Define las funciones
- intervalos_ininterrumpidos
- num_positivas_negativas
- signal2img

@author: Bananin
"""
import os
import pandas as pd
import numpy as np
from scipy.signal import stft
import re

F = 400 # Hz
CANALES = np.array(["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9",
                    "ch10","ch11","ch12","ch13","ch14","ch15"])
dataset_root = "dataset/"
subsets_names = ["train","val","test"]

# min_duracion (segundos) de una seccion ininterrumpida
def intervalos_ininterrumpidos (min_duracion=5, canales=CANALES):
    # directorios que contienen observaciones de pacientes
    dirs_pacientes = [dir_pac for dir_pac in os.listdir() if dir_pac.startswith("Pat")]
    for dir_paciente in dirs_pacientes:
        # lista de intervalos ininterrumpidos para este paciente
        intervalos = []
        # archivos csv de los eeg de este paciente
        EEGs = [filename for filename in os.listdir(dir_paciente)]

        for EEG in EEGs:
            # listar todos los intervalos ininterrumpidos en este segmento
            segmento = pd.read_csv(dir_paciente+"/"+EEG)
            # valor de dropout parece ser 0.034. Hay dropout si hay 0 en todos los canales, lo identificamos por la suma de canales
            segmento.loc[:,canales] = segmento.loc[:,canales].replace(0.034,0)
            sum_canales = np.sum(np.abs(segmento.loc[:,canales]), axis=1)

            # buscamos intervalos ininterrumpidos
            inicio_intervalo = 0
            for i in range(len(segmento)):
                if sum_canales[i] == 0 or i == len(segmento)-1:
                    # discontinuidad por dropout o fin del archivo
                    if i-inicio_intervalo >= min_duracion*F:
                        # consolidamos un intervalo (fin no inclusivo)
                        intervalos.append({"inicio":inicio_intervalo,"fin":i,"archivo":dir_paciente+"/"+EEG})
                    inicio_intervalo = i+1
        # almacenamos los intervalos encontrados para el paciente
        intervalos = pd.DataFrame(intervalos)
        intervalos.to_csv("intervalos/"+dir_paciente+".csv", index=False)

# numero de imagenes de entrenamiento que obtendriamos de una lista de intervalos con W y sobreposicion dada
# W := ventana grande, sobre_pos/neg := sobreposicion de las ventanas grandes en intervalos ininterrumpidos
def num_positivas_negativas (data_subfolder="Pat3Train", W=30, S=(0,5.08/6)):
    # numero de observaciones que componen una ventana de duracion W y la sobreposicion
    W_ticks = np.floor(W*F)
    S_ticks = (np.floor(S[0]*W*F), np.floor(S[1]*W*F))
    # intervalos calculados con la funcion intervalos_ininterrumpidos
    intervalos = pd.read_csv("intervalos/"+data_subfolder+".csv")
    positivos = intervalos.iloc[[i for i in range(len(intervalos)) if bool(re.match(".*1.csv",intervalos.archivo[i]))]]
    negativos = intervalos.iloc[[i for i in range(len(intervalos)) if bool(re.match(".*0.csv",intervalos.archivo[i]))]]
    positivasXintervalo = (positivos.fin - positivos.inicio - S_ticks[1])//(W_ticks - S_ticks[1])
    negativasXintervalo = (negativos.fin - negativos.inicio - S_ticks[0])//(W_ticks - S_ticks[0])
    # no tiene sentido que un intervalo reste al total de imagenes
    positivasXintervalo[positivasXintervalo < 0] = 0
    negativasXintervalo[negativasXintervalo < 0] = 0
    return sum(positivasXintervalo) , sum(negativasXintervalo)

"""
senal multicanal -> imagen multicanal
multi_signal : dataframe con observaciones contiguas como filas y canales como columnas
w : ventana deslizante de STFT
s : sobreposicion de la anterior
"""
def signal2image (multi_signal, w=1, s=2/3):
    # length of each segment
    nperseg = np.floor(w*F)
    # number of points to overlap between segments. Defaults to nperseg // 2
    noverlap = np.floor(s*w*F)
    # altura de la imagen. Conversion a entero es trivial
    # t = int((len(multi_signal) - noverlap)//(nperseg - noverlap))
    # short-time Fourier en cada canal : imagen retornada siempre tiene 601 frecuencias
    # frecuencias observadas = floor(nperseg/2)+1. Luego restamos 1 para quitar frecuencia 0
    # imagen = np.zeros((t, int((nperseg/2)+1)-1, len(multi_signal.columns)))
    for i in range(len(multi_signal.columns)):
        # calcular y almacenar imagen
        # resolucion de frecuencia es definida
        values = stft(multi_signal.iloc[:,i], fs=F, nperseg=nperseg, noverlap=noverlap, boundary=None)[2]
        if i == 0:
            imagen = np.zeros((values.shape[1],values.shape[0]-1,len(multi_signal.columns)))
        # descartar frecuencia 0 como Truong
        values = values[1:values.shape[0],]
        imagen[:,:,i] = np.transpose(np.abs(values))
    return imagen

"""
genera imagenes de las observaciones del paciente en el conjunto de train o test

W : ventana grande = rango temporal de cada imagen
w : ventana corta = ventana deslizante en STFT
S : sobreposicion porcentual de las ventanas grandes en los ininterrumpidos (preictales,interictales)
s : sobreposicion porcentual de las ventanas pequenas en STFT
canales : que vamos a tener en cuenta
"""
def generar_imagenes (data_subfolder="Pat3Train", W=30, S=(0,5.08/6), w=1, s=2/3, canales=[i for i in range(16)], subsets_dist=[0.6,0.2,0.2]):
    # la ruta en la cual las colocamos es una concatenacion de los parametros
    ruta = "preprocessed/"+data_subfolder+"/W="+str(W)+"_S=("+('%.2f'%S[0])+","+('%.2f'%S[1])+")_w="+str(w)+"_s="+('%.2f'%s)+"_ch"+str(canales).replace(" ","")+"/"
    assert not os.path.exists(ruta) , "Ya hemos calculado las imagenes para estos parametros"
    num_positivas, num_negativas = num_positivas_negativas(data_subfolder="Pat3Train", W=30, S=(0,5.08/6))
    print("Generando imagenes: "+str(num_positivas)+" preictales y "+str(num_negativas)+" interictales.")
    os.makedirs(ruta)
    # carpetas en las cuales almacenar train, val y test
    os.makedirs(ruta+subsets_names[0])
    os.makedirs(ruta+subsets_names[1])
    os.makedirs(ruta+subsets_names[2])
    # obtener nombres de canales
    canales = CANALES[canales]
    # numero de observaciones que componen una ventana de duracion W y las sobreposiciones
    W_ticks = np.floor(W*F)
    S_ticks = (np.floor(S[0]*W*F), np.floor(S[1]*W*F))
    # archivo de intervalos ininterrumpidos en este conjunto de datos del paciente
    intervalos = pd.read_csv("intervalos/"+data_subfolder.replace("/","")+".csv")
    # descartar intervalos de duracion menor al W solicitado
    intervalos = intervalos.loc[(intervalos.fin - intervalos.inicio) >= W_ticks]
    # reordenar los intervalos aleatoriamente para eliminar la correlacion temporal
    intervalos = intervalos.sample(frac=1).reset_index(drop=True)

    # para cada archivo
    for archivo in set(intervalos.archivo):
        segmento = pd.read_csv(dataset_root+archivo) # hay que arreglar el valor de dropout
        segmento.loc[:,canales] = segmento.loc[:,canales].replace(0.034,0)
        intervalos_archivo = intervalos[intervalos.archivo == archivo].reset_index(drop=True)
        # generar las imagenes posibles de cada intervalo ininterrumpido
        # almacenar en train/ val / o test/ segun la distribucion dada
        subset = np.where(np.random.multinomial(n=1, pvals=subsets_dist)==1)[0][0]
        for i in range(len(intervalos_archivo)):
            inicio = intervalos_archivo.inicio[i]
            fin = intervalos_archivo.fin[i]
            # los archivos se llaman ..._0.csv : este numero indica si es preictal o interictal
            label = int(archivo[-5])
            # generar las imagenes posibles. No aparecen negativos pues filtramos intervalos mas cortos que W
            num_imagenes = (fin - inicio - S_ticks[label])//(W_ticks - S_ticks[label])
            for j in range(int(num_imagenes)):
                inicio_imagen = inicio + j*(W_ticks - S_ticks[label])
                fin_imagen = inicio_imagen + W_ticks - 1
                multi_signal = segmento.loc[inicio_imagen:fin_imagen,canales].reset_index(drop=True)
                # lanzar error si hay interrupcion en el subintervalo
                assert np.sum(np.sum(np.abs(multi_signal), axis=1)==0)==0,("Subintervalo "+str(i)+"_"+str(j)+" de "+archivo+" tiene interrupcion.")
                # convertir a "imagen"
                imagen = signal2image(multi_signal, w=w, s=s)
                np.save(file=ruta+subsets_names[subset]+"/"+archivo.split("/")[1]+"_inter"+str(i)+"_sub"+str(j)+".npy",arr=imagen)
