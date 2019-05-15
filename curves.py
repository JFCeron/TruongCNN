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

def pr_curves():
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
        plt.title('AUC={:.2f}\nBest F2={:.2f} with precision={:.2f}, recall={:.2f}'.
                  format(np.mean(precision), F2[max_combi], precision[max_combi], recall[max_combi]))
        plt.savefig(folder+"/p-r_curve.png")
        plt.close()

def training_curves():
    result_folders = [d[0] for d in os.walk("results/")][1:]
    for folder in result_folders:
        try:
            train = pd.read_csv(folder+"/train_metrics.csv")
            train["epoch"] = range(1,len(train)+1)
        except:
            print("Train metrics not yet calculated at "+folder)
            continue

        # graph train and val F2 over the epochs
        plt.plot("epoch", "train_Fbeta", data=train, color='blue', label="train")
        plt.plot("epoch", "val_Fbeta", data=train, color='orange', label="val")
        plt.ylabel("F2")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Best validation F2={:.2f}".format(max(train.val_Fbeta)))
        plt.savefig(folder+"/train-val_curve.png")
        plt.close()

def main():
    pr_curves()
    training_curves()

if __name__ == "__main__":
    main()
