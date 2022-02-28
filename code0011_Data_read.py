# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 22:31:29 2021

@author: INFINITE-WORKSTATION
"""
import mne

#Just display errors and warnings
mne.set_log_level("WARNING")

#Loading EEG data
data = mne.io.read_raw_gdf("motorimagination_subject1_run1.gdf", preload=True) 

eeg = data.get_data()
info = data.info
event_info = data._annotations.description
event_onset = data._annotations.onset
time_values = data.times

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