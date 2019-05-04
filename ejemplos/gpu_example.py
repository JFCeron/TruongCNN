# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:36:33 2019

@author: Bananin
"""

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime


shape = (10000, 1000)
device_name = "/cpu:0"

for device_name in ["/cpu:0","/gpu:0"]:
    sum_device = 0
    for i in range(500):
        startTime = datetime.now()
        with tf.device(device_name):
            random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                result = session.run(sum_operation)
        sum_device += (datetime.now()-startTime).total_seconds()
    print(device_name+" took an average of "+str(sum_device/500))

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)