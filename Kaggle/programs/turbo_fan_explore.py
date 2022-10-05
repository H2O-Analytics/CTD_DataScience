"""
Name: 

Purpose: 

Developer: Taylor Waters

Input:  

Output

Parameters:


Usage:

Resources Used:

History:
Date        User    Ticket #    Description
"""

# import packages
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

# import data
DATA_PATH = "/Users/tawate/My Drive/CDT_Data_Science/data_sets/turbofan_sensors/"
train = pd.read_table(DATA_PATH + "train_FD001.txt", 
                    delimiter=' ', 
                    names=['unit_num','time(cycles)','op_1','op_2','op_3'
                            'meas_1','meas_2','meas_3','meas_4','meas_5','meas_6'
                            ,'meas_7','meas_8','meas_9','meas_10','meas_11', 'meas_12'
                            ,'meas_13','meas_14','meas_15','meas_16','meas_17','meas_18'
                            ,'meas_19','meas_20','meas_21','nan1','nan2'])

uni_1 = train[train['unit_num'] == 1]

sns.lineplot(data = uni_1, x="time(cycles)", y = "op_1")
