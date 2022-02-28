import mne
import numpy as np

event_list = [1536, 1537, 1538, 1539, 1540, 1541, 1542]
f_s = 512
t_s = -2.0 #Trials of 5 seconds or Full length trial
t_f = 3.0
sample_s = int(t_s*f_s)
sample_f = int(t_f*f_s)

x = []
y = []

#have to run code below this for all 10 runs for subject1
data = mne.io.read_raw_gdf('motorimagination_subject1_run1.gdf') 
eeg_data = data.get_data()
info = data.info

event_info = data._annotations.description
event_onset = data._annotations.onset
time_values = data.times
print(eeg_data.shape)

for e in range(len(event_info)):
  channel_data=[]
  if (int(event_info[e]) in event_list):
    event_sample = int(event_onset[e]*f_s)
    for ch in range(eeg_data.shape[0]):
      channel_data.append(eeg_data[ch, event_sample+sample_s:event_sample+sample_f])
    x.append(np.reshape(channel_data,(96,2560)))
    y.append(event_list.index(int(event_info[e]))+1)

print(len(x))
print(len(x[0]))
print(len(y))
