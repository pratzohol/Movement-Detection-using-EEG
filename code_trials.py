import mne
import numpy as np

event_list = [1536, 1537, 1538, 1539, 1540, 1541, 1542]
f_s = 512
t_s = 0 #Trials of 5 seconds or Full length trial/ 0 for only 2 to 5 seconds
t_f = 3.0
sample_s = int(t_s*f_s)
sample_f = int(t_f*f_s)

x = []
y = []

#have to run code below this for all 10 runs for subject1
data = mne.io.read_raw_gdf('S01_MI/motorimagination_subject1_run1.gdf') 
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
    for ch in range(61):
      channel_data.append(eeg_data[ch, event_sample+sample_s:event_sample+sample_f])
    x.append(np.reshape(channel_data,(61,1536)))
    y.append(event_list.index(int(event_info[e]))+1)

print(len(x))
print(len(x[0]))
print(len(y))

spectral_x=[]

# calculate power spectral density for each 61 channels
def calculate_psd(data):
  psd = np.zeros((61,1536))
  for ch in range(61):
    psd[ch,:] = np.abs(np.fft.fft(data[ch,:]))
  return psd

# perform psd calculation for each trial
for i in range(len(x)):
  spectral_x.append(calculate_psd(x[i]))

final_x=[]

# 61 electrode placement matrix
l=[[]]

# random suffle can be done here. 

# randomly splitting the data into train, validation and test
train_x=final_x[:int(len(final_x)*0.7)]
validation_x=final_x[int(len(final_x)*0.7):int(len(final_x)*0.85)]
test_x=final_x[int(len(final_x)*0.85):]
