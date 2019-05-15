"""
Keeps constants for the whole project in one place

"""
import numpy as np
import argparse

# signal frequency (Hz) in the EpilepsyEcosystem dataset
F = 400
# channel names
ch_names = np.array(["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15"])
# column names for evaluation metrics
train_metric_names = ["train_lossXsample","train_tp","train_fp","train_tn","train_fn","train_precision",
                    "train_recall","train_Fbeta","val_lossXsample","val_tp","val_fp","val_tn","val_fn",
                    "val_precision","val_recall","val_Fbeta","epoch_duration","eval_duration"]
test_metric_names = ["test_lossXsample","test_tp","test_fp","test_tn","test_fn","test_precision",
                    "test_recall","test_Fbeta","eval_duration"]

"""
parses command line parameters and sets default values
"""
def parseCommandLineParams():
    # parse command line parameters
    parser = argparse.ArgumentParser()
    # data subset for this analysis
    parser.add_argument("-patient", type=str, default="3")
    # image time range and overlap for negatives and positives
    parser.add_argument("-W", type=int, default=30)
    parser.add_argument("-O0", type=float, default=0)
    parser.add_argument("-O1", type=float, default=5.2/6)
    # STFT small window time range and overlap
    parser.add_argument("-w", type=float, default=1)
    parser.add_argument("-o", type=float, default=2/3)
    # channels encoded in a string
    parser.add_argument("-ch", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    # neural network hyperparameters
    parser.add_argument("-batch", type=int, default=24)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-sch_patience", type=int, default=3)
    parser.add_argument("-es_patience", type=int, default=6)
    # loss weights for positive and negative observations
    parser.add_argument("-weight0", type=float, default=1)
    parser.add_argument("-weight1", type=float, default=2)
    # printing parameters
    parser.add_argument("-miniepochs", type=int, default=200)
    # validation hyperparameters
    parser.add_argument("-beta", type=float, default=2)
    # maximum number of images to generate from each class
    parser.add_argument("-maxPerClass", type=int, default=40000)
    # object that stores all parameters
    return parser.parse_args()
