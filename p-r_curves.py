# -*- coding: utf-8 -*-
"""
Creates precision-recall curves for all available results

@author: Bananin
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

result_folders = [d[0] for d in os.walk("results/")][1:]

for folder in result_folders:
    try:
        test = pd.read_csv(folder+"/test_responses.csv")
    except:
        print("Test responses not yet calculated at "+folder)
        continue
    
    # precision, recall and F2 values for several thresholds
    precision, recall, _ = precision_recall_curve(test.label, test.p1)
    F2 = (1+2**2)*precision*recall/((2**2)*precision + recall)
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    max_combi = np.nanargmax(F2)
    plt.title('Best F2={:.2f} with precision={:.2f}, recall={:.2f}'.
              format(F2[max_combi], precision[max_combi], recall[max_combi]))
    plt.savefig(folder+"/p-r_curve.png")
    plt.close()
