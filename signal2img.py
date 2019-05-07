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

"""
number of positive and negative images that can be extracted from the data_subfolder

W : large window = temporal range of each image
O : proportion of large window Overlap in uninterrupted intervals (preictal,interictal) tuple
"""
def num_positives_negatives (data_subfolder="Pat3Train", W=30, O=(0,5.2/6)):
    # number of time observations that compose a duration W window at frequency F
    W_ticks = np.floor(W*F)
    O_ticks = (np.floor(O[0]*W*F), np.floor(O[1]*W*F))
    # intervalos calculados con la funcion intervalos_ininterrumpidos
    intervals = pd.read_csv("intervals/"+data_subfolder+".csv")
    positives = intervals.iloc[[i for i in range(len(intervals)) if bool(re.match(".*1.csv",intervals.file[i]))]]
    negatives = intervals.iloc[[i for i in range(len(intervals)) if bool(re.match(".*0.csv",intervals.file[i]))]]
    positivesXinterval = (positives.end - positives.start - O_ticks[1])//(W_ticks - O_ticks[1])
    negativesXinterval = (negatives.end - negatives.start - O_ticks[0])//(W_ticks - O_ticks[0])
    # no tiene sentido que un intervalo reste al total de imagenes
    positivesXinterval[positivesXinterval < 0] = 0
    negativesXinterval[negativesXinterval < 0] = 0
    return sum(positivesXinterval) , sum(negativesXinterval)

"""
multichannel signal to image translation

multi_signal : dataframe with sequential sensor data as rows and channels as columns
w : STFT sliding window duration
o : overlap proportion
"""
def signal2image (multi_signal, w=1, o=2/3):
    # length of each segment
    nperseg = np.floor(w*F)
    # number of points to overlap between segments. Defaults to nperseg // 2
    noverlap = np.floor(o*w*F)
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
            image = np.zeros((values.shape[1],values.shape[0]-1,len(multi_signal.columns)))
        # descartar frecuencia 0 como Truong
        values = values[1:values.shape[0],]
        image[:,:,i] = np.transpose(np.abs(values))
    return image

"""
generates images out of the sensor data in one of the dataset's subfolders

W : large window = temporal range of each image
w : short sliding window in STFT
O : proportion of large window Overlap in uninterrupted intervals (preictal,interictal) tuple
o : proportion of STFT short window overlap
channels : EEG channels to be taken into account
"""
def generate_images (data_subfolder="Pat3Train", W=30, O=(0,5.2/6), w=1, o=2/3, channels=[i for i in range(16)]):
    # the destination folder name is a concatenation of generation parameters
    folder = "preprocessed/W="+str(W)+"_O=("+('%.2f'%O[0])+","+('%.2f'%O[1])+")_w="+str(w)+"_o="+\
            ('%.2f'%o)+"_ch"+str(channels).replace(" ","")+"/"+data_subfolder
    assert not os.path.exists(folder) , "We've already calculated the images for these parameters"
    # validation and test sets have no image overlap to test in the most general scenario
    if "Val" in data_subfolder or "Test" in data_subfolder:
        O = (0,0)
    num_positives, num_negatives = num_positives_negatives(data_subfolder=data_subfolder, W=W, O=O)
    print("Generating images: "+str(num_positives)+" preictal and "+str(num_negatives)+" interictal")
    os.makedirs(folder)
    # obtain channel names
    channels = CANALES[channels]
    # numero de observaciones que componen una ventana de duracion W y las sobreposiciones
    # number of observations in a length (s) W window and in the given overlap at frequency F
    W_ticks = np.floor(W*F)
    O_ticks = (np.floor(O[0]*W*F), np.floor(O[1]*W*F))
    # uninterrupted intervals file for this data_subfolder
    intervals = pd.read_csv("intervals/"+data_subfolder.replace("/","")+".csv")
    # discard intervals with length lower than the requested W
    intervals = intervals.loc[(intervals.end - intervals.start) >= W_ticks]
    # shuffle intervals to eliminate temporal correlation
    intervals = intervals.sample(frac=1).reset_index(drop=True)

    # for every .csv file in the data_subfolder
    for csv_file in set(intervals.file):
        # files are named ..._0.csv : this is the segment's label. NaN means it is private
        try:
            label = int(csv_file[-5])
        except:
            # this is a private file; we have no use for it
            continue

        # the data segment in this csv_file
        segment = pd.read_csv(dataset_root+csv_file) # hay que arreglar el valor de dropout
        segment.loc[:,channels] = segment.loc[:,channels].replace(0.034,0)

        # generate as many images as we can in each of its uninterrupted intervals
        file_intervals = intervals[intervals.file == csv_file].reset_index(drop=True)
        for i in range(len(file_intervals)):
            start = file_intervals.start[i]
            end = file_intervals.end[i]
            # generate the images
            num_images = (end - start - O_ticks[label])//(W_ticks - O_ticks[label])
            for j in range(int(num_images)):
                image_start = start + j*(W_ticks - O_ticks[label])
                image_end = image_start + W_ticks - 1
                # sensor data for this image
                multi_signal = segment.loc[image_start:image_end,channels].reset_index(drop=True)
                # throw exception if the interval has interruptions
                assert np.sum(np.sum(np.abs(multi_signal), axis=1)==0)==0,("Subinterval "+str(i)+"_"+str(j)+" in "+csv_file\
                    +" has data dropout")
                # image translation by STFT
                image = signal2image(multi_signal, w=w, o=o)
                np.save(file=folder+"/"+csv_file.split("/")[1]+"_inter"+str(i)+"_sub"+str(j)\
                    +".npy",arr=image)
