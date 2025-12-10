#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:56:08 2025

@author: ellasteins
"""

import sys
sys.path.append('..')

from _utils.cusum_utils import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#%% LOAD VIRTUAL FLOWS FOR THE DUAL MODEL: here the KWB 2019 interface dataset


# either data_09_virtual_flows.csv or data_08_virtual_flows_uncorrected.csv

# path_scada_pressure = '../data/data_09_virtual_flows.csv' 
path_scada_pressure = '../data/data_08_virtual_flows_uncorrected.csv' 

n = 105121
timestamps = pd.date_range(start="2019-01-01 00:00:00", periods=n, freq="5T") 
df = pd.read_csv(path_scada_pressure,
                        delimiter=',',
                         decimal='.')

df = df.drop(df.columns[0], axis=1)

df.index = timestamps 
df.name = "Timestamp"  

df = df.sum(axis=1)
df = pd.DataFrame(df)
    

#%% GROUND TRUTH OF THE LEAKAGES

#pipe_id: (leak_start, leak_fix, DMA of the leak)
# the information of the DMA is needed to decide if CUSUM method adn or corr is used
 
ground_truth = {'p641': ('2019-01-19 12:55:00', '2019-02-01 12:45:00', 'DMA_A'),
 'p96': ('2019-02-19 20:15:00', '2019-02-21 20:10:00', 'DMA_A'),
 'p742': ('2019-03-15 13:35:00', '2019-04-24 13:25:00', 'DMA_A'),
 'p360': ('2019-05-10 15:40:00', '2019-06-15 15:30:00', 'DMA_A'),
 'p137': ('2019-07-03 22:30:00', '2019-07-23 22:20:00', 'DMA_A'),
 'p127': ('2019-08-12 17:15:00', '2019-08-14 17:10:00', 'DMA_A'),
 'p426': ('2019-09-01 08:05:00', '2019-09-04 08:00:00', 'DMA_A'),
 'p74': ('2019-09-21 19:40:00', '2019-09-25 19:35:00', 'DMA_B'),
 'p250': ('2019-10-14 11:35:00', '2019-11-13 11:25:00', 'DMA_C'),
 'p884': ('2019-11-30 19:20:00', '2019-12-04 19:15:00', 'DMA_A')}


#%% OPTIONALLY SELECT DIFFERENT THRESHOLD PARAMETERS h_adn_set and h_corr_set

# h_adn_set = 300 # data_09
h_adn_set = 250 # data_08


h_corr_set = 250

#%% RUN CUSUM FOR EACH LEAK IN GROUND TRUTH DICT
all_TTD = {}

for i in range(len(list(ground_truth.keys()))):
    pipe_id = list(ground_truth.keys())[i] #p641, p96, ...
    
    detection, df_C = f_CUSUM(df, pipe_id, ground_truth, h_adn= h_adn_set , h_corr= h_corr_set)
    #dteection either FN, FP, or TTD
    #df_C is the dataframe of the CUSUM statitsic
    
    all_TTD[pipe_id] = str(detection)
    # add detection to result dict
    
    f = f_plot_C(df_C, detection, pipe_id, ground_truth, h_adn = h_adn_set , h_corr = h_corr_set)
    # plot the CUSUM statistic
    
#%% PRINT RESULT

print(all_TTD)




#%%