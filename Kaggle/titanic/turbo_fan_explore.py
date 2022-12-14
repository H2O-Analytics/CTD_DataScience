"""
Name: 
Purpose: exploratory analysis on nasa turbo fan sensor data
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
from operator import contains
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

# import data
# DATA_PATH = "/Users/tawate/My Drive/CDT_Data_Science/data_sets/turbofan_sensors/"
DATA_PATH = "/Users/tawate/Library/CloudStorage/OneDrive-SAS/08_CDT_DataScience/nasa engine turbofan data/turbofan_sensors/"
# FD001 Settings:
#       Fault Mode = High Pressure Compressor Degradation
#       Conditions = 1. tested at sea level
train_fd001 = pd.read_table(DATA_PATH + "train_FD001.txt", 
                    delimiter=' ', 
                    names=['unit_num','time(cycles)','op_1','op_2','op_3',
                            'meas_1','meas_2','meas_3','meas_4','meas_5','meas_6'
                            ,'meas_7','meas_8','meas_9','meas_10','meas_11', 'meas_12'
                            ,'meas_13','meas_14','meas_15','meas_16','meas_17','meas_18'
                            ,'meas_19','meas_20','meas_21','nan1','nan2'])

# FD002 Settings:
#       Fault Mode = High Pressure Compressor Degradation
#       Conditions = 6
train_fd002 = pd.read_table(DATA_PATH + "train_FD002.txt", 
                    delimiter=' ', 
                    names=['unit_num','time(cycles)','op_1','op_2','op_3',
                            'meas_1','meas_2','meas_3','meas_4','meas_5','meas_6'
                            ,'meas_7','meas_8','meas_9','meas_10','meas_11', 'meas_12'
                            ,'meas_13','meas_14','meas_15','meas_16','meas_17','meas_18'
                            ,'meas_19','meas_20','meas_21','nan1','nan2'])

# FD003 Setting:
#       Fault Mode = High Pressure Compressor Degradation and turbofan Degradation
#       Conditions = 1 (Sea Level)
train_fd003 = pd.read_table(DATA_PATH + "train_FD003.txt", 
                    delimiter=' ', 
                    names=['unit_num','time(cycles)','op_1','op_2','op_3',
                            'meas_1','meas_2','meas_3','meas_4','meas_5','meas_6'
                            ,'meas_7','meas_8','meas_9','meas_10','meas_11', 'meas_12'
                            ,'meas_13','meas_14','meas_15','meas_16','meas_17','meas_18'
                            ,'meas_19','meas_20','meas_21','nan1','nan2'])

# FD004 Setting:
#       Fault Mode = High Pressure Compressor Degradation and Turbofan Degradation
#       Conditions = 6
train_fd004 = pd.read_table(DATA_PATH + "train_FD004.txt", 
                    delimiter=' ', 
                    names=['unit_num','time(cycles)','op_1','op_2','op_3',
                            'meas_1','meas_2','meas_3','meas_4','meas_5','meas_6'
                            ,'meas_7','meas_8','meas_9','meas_10','meas_11', 'meas_12'
                            ,'meas_13','meas_14','meas_15','meas_16','meas_17','meas_18'
                            ,'meas_19','meas_20','meas_21','nan1','nan2'])

# Get number of cycles to failure per engine
cycles_to_failure = train_fd001.groupby(['unit_num']).size().reset_index(name='cycles')
cycles_to_failure['cycles'].hist()

uni_1 = train_fd001[train_fd001['unit_num'] == 1]
sns.set()
fig,axes = plt.subplots(3,1)
from scipy.signal import savgol_filter
sns.lineplot(data = uni_1, x="time(cycles)", y = "op_1", ax = axes[0])
sns.lineplot(data = uni_1, x="time(cycles)", y = "meas_1", ax = axes[1])
sns.lineplot(data = uni_1, x="time(cycles)", y = "meas_2", ax = axes[2])
y_smooth = np.convolve(y, box, mode='same')
for index,var in enumerate(uni_1):
        if index >= 4 and 'meas' in var:
                plt.figure()
                sns.lineplot(data=uni_1, x="time(cycles)", y=savgol_filter(uni_1["op_1"],51,3), color="g")
                # ax2 = plt.twinx()
                # sns.lineplot(data=uni_1, x="time(cycles)", y=var, color = "b", ax=ax2)
                ax2 = plt.twinx()
                sns.lineplot(data=uni_1, x="time(cycles)", y=savgol_filter(uni_1[var],51,3), color = "b", ax=ax2)

sns.lineplot(data=uni_1, x="time(cycles)", y = "op_1", color="g")
ax2 = plt.twinx()
sns.lineplot(data=uni_1, x="time(cycles)", y = "meas_2", color="b", ax=ax2).set(title = '')

# Next Steps:
#       Normalize all data and plot operational settings vs measures
#       Smooth all curves to reduce sensor noise
#       Determine relationships between each measure and operatioal setting
