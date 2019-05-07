# -*- coding: utf-8 -*-
"""
The Epilepsy Ecosystem's data has dropout; intervals in which the sensors failed to pick up signals.
They are identified by all sensors reading 0 (which was numerically mapped to 0.034). This script
extracts uninterrupted segments which we can draw images from safely

@author: Bananin
"""
import numpy as np
import pandas as pd
import os

# an interval's minimum duration in seconds
min_duration = 5
F = 400 # Hz
CHANNELS = np.array(["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9", "ch10","ch11",
                     "ch12","ch13","ch14","ch15"])

# directories containing the observations for each patient
patient_dirs = [dir_pat for dir_pat in os.listdir("dataset") if dir_pat.startswith("Pat")]
try:
    os.makedirs("intervals")
except:
    pass

for patient_dir in patient_dirs:
    print("Processing intervals in directory "+patient_dir)
    # list of uninterrupted intervals for this patient
    intervals = []
    # this patient's .csv eeg files
    EEGs = [filename for filename in os.listdir("dataset/"+patient_dir)]

    for EEG in EEGs:
        # list all uninterrupted intervals for this segment (.csv file)
        segment = pd.read_csv("dataset/"+patient_dir+"/"+EEG)
        # dropout value is actually 0.034
        segment.loc[:,CHANNELS] = segment.loc[:,CHANNELS].replace(0.034,0)
        sum_channels = np.sum(np.abs(segment.loc[:,CHANNELS]), axis=1)

        # look for uninterrupted intervals
        interval_start = 0
        for i in range(len(segment)):
            if sum_channels[i] == 0 or i == len(segment)-1:
                # interruption due to dropout or file end
                if i-interval_start >= min_duration*F:
                    # consolidate an interval (end tick not included)
                    intervals.append({"start":interval_start,"end":i,"file":patient_dir+"/"+EEG})
                interval_start = i+1
    # store all discovered intervals for this patient
    intervals = pd.DataFrame(intervals)
    intervals.to_csv("intervals/"+patient_dir+".csv", index=False)
