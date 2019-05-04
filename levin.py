# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:16:10 2019

@author: Bananin
"""

import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
CANALES = ["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15"]
# os.chdir("C:/Users/Bananin/Documents/eeg/ecosystem_data")
import signal2img as s2i
# resolucion de frecuencia de las imagenes generadas con STFT
resolucion_frec = 601

# p_train : proporcion de PatXTrain que conforma el conjunto de entrenamiento. El resto es de validacion
def data_subsets (paciente=1, W=30, SOBRE=(20,0), w=3, sobre=2, canales=CANALES, p_train=0.8):
    
    # resolucion de las imagenes dados los parametros (tiempo, frecuencia, canales)
    resolucion = (int(np.floor((W-sobre)/(w-sobre))), resolucion_frec, len(canales))
    
    # la ruta debe tener las imagenes construidas para estos parametros
    ruta = "imagenes/W="+str(W)+"_SP="+str(SOBRE[0])+"_SN="+str(SOBRE[1])+"_w="+str(w)+"_s="+str(sobre)+"_canales="+str(canales)+"/Pat"+str(paciente)+"Train/"
    imagenes = np.array(os.listdir(ruta))
    labels = np.array([ruta_imagen.split("_")[2][0]=="1" for ruta_imagen in imagenes])
    n_pos = np.sum(labels)
    n_neg = imagenes.shape[0] - n_pos

    # obtener conjuntos de entrenamiento y validacion de los archivos PatXTrain
    positivos_train = imagenes[labels][0:int(n_pos*p_train)]
    negativos_train = imagenes[~labels][0:int(n_neg*p_train)]
    positivos_val = imagenes[labels][int(n_pos*p_train):]
    negativos_val = imagenes[~labels][int(n_neg*p_train):]

    rutas_train = np.concatenate((positivos_train, negativos_train))
    rutas_val = np.concatenate((positivos_val, negativos_val))

    # consolidar matriz de imagenes de cada conjunto
    x_train = np.zeros(tuple((len(rutas_train),))+resolucion)
    for i in range(len(rutas_train)):
        x_train[i] = np.load(ruta+rutas_train[i])
        
    x_val = np.zeros(tuple((len(rutas_val),))+resolucion)
    for i in range(len(rutas_val)):
        x_val[i] = np.load(ruta+rutas_val[i])

    # labels
    y_train = np.concatenate((np.ones(len(positivos_train)), np.zeros(len(negativos_train))))
    y_val = np.concatenate((np.ones(len(positivos_val)), np.zeros(len(negativos_val))))

    return x_train, x_val, y_train, y_val