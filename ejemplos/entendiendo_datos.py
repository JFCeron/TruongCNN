# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:54:38 2019

@author: Bananin
"""
import os
import pandas as pd
import numpy as np
os.chdir("C:/Users/Bananin/Documents/eeg")

# primer indice: numero de observacion
# segundo: 0/1 interictal o preictal
csv1 = pd.read_csv("ecosystem_data/Pat2Train/Pat2Train_1_0.csv")
csv2 = pd.read_csv("ecosystem_data/Pat2Train/Pat2Train_2_0.csv")
csv3 = pd.read_csv("ecosystem_data/Pat2Train/Pat2Train_3_0.csv")
# columnas: ['time', 'id', 'channelGroups.id', 'segments.id', 'ch0', 'ch1', 'ch2',
#       'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12',
#       'ch13', 'ch14', 'ch15']
csv1.columns
# id parece ser del paciente
set(csv1.id)
set(csv2.id)
# este tambien parece ser constante al menos por paciente
set(csv1["channelGroups.id"])
# un id para el segmento de 10 minutos
set(csv1["segments.id"])
# mediciones cada 2.5ms
periodo = csv1.time[1]-csv1.time[0]
# cada archivo contiene 10min
(len(csv1)*0.0025)/60
(len(csv2)*0.0025)/60
(len(csv3)*0.0025)/60
# no son contiguos? la pagina dice que si pero parecen separados por 50 minutos
(csv2.time[0]-csv1.time[len(csv1)-1])/60000
(csv3.time[0]-csv2.time[len(csv1)-1])/60000