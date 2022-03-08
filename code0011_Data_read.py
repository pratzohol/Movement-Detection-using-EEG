# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 22:31:29 2021

@author: INFINITE-WORKSTATION
"""
from opcode import hasjabs
import mne
import pandas as pd

#Just display errors and warnings
mne.set_log_level("WARNING")

#Loading EEG data
data = mne.io.read_raw_gdf("S01_MI/motorimagination_subject1_run1.gdf", preload=True) 
# mne.viz.plot_raw(data, start=0, duration=10)
# df=data.to_data_frame()
# df.to_excel("S01_MI/motorimagination_subject1_run1.xlsx")
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
all_channels=data.ch_names
print(all_channels)
#select channels
# eeg_data = data.pick_channels(all_channels[:62])
# print(eeg_data.info())
print("-"*80)
# print(eeg_data.ch_names)
# print("-"*80)

eeg = data.get_data()
info = data.info
event_info = data._annotations.description
event_onset = data._annotations.onset
time_values = data.times
print(eeg.shape)
print("-"*80)
# print(info)
print("-"*80)
print(event_info)
# l=event_info.tolist()
# df_temp= pd.DataFrame(l)
# df_temp.to_csv("S01_MI/event_info_S01_run1.csv")
print("-"*80)
print(event_onset)
print("-"*80)
# print(time_values)

"""
Event description
768 - start of Trial
785 - Beep
786 - Fixation Cross
1536 - class01
1537 - class02
1538 - class03
1539 - class04
1540 - class05
1541 - class06
1542 - Rest

33XXX - end of the trial

sampling frequency = 512Hz(Given in "info")
sample of event cue = event_onset * sampling frequency(take the roundoff integer value)
"""