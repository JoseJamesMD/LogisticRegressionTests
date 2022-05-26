# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:45:29 2022

@author: m211991
"""

from math import pi, cos, sin
import numpy as np
import random
import seaborn
import matplotlib.pyplot as plt

def circle_data(n,r1,r2,e):
    
    # create dataset
    x1_1 = np.zeros(n); x2_1 = np.zeros(n)
    x1_2 = np.zeros(n); x2_2 = np.zeros(n)
    y_1 = np.ones(n); y_2 = np.zeros(n)

    # radius of the circle
    
    for i in np.arange(n):
        theta = random.random() * 2 * pi
        
        x1_1[i],x2_1[i] = cos(theta) * (r1+e*random.random()), sin(theta) * (r1+e*random.random())
        x1_2[i],x2_2[i] = cos(theta) * (r2+e*random.random()), sin(theta) * (r2+e*random.random())
    
    x1 = np.concatenate([x1_1,x1_2])
    x2 = np.concatenate([x2_1,x2_2])
    X = np.column_stack([x1,x2])
    y = np.concatenate([y_1,y_2])

    return X, y

def data_plot(x1,x2,y):
    plt.figure()
    color_dict = dict({0:'green', 1:'red'})
    p = seaborn.scatterplot(x1, x2, hue=y, palette = color_dict)
    p.set_xlabel("X1", fontsize = 16)
    p.set_ylabel("X2", fontsize = 16)
    return p