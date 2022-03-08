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
    psd[ch,:] = np.abs(np.fft.rfft(data[ch,:]))**2
  return psd

# perform psd calculation for each trial
for i in range(len(x)):
  spectral_x.append(calculate_psd(x[i]))

final_x=[]

for i in range(len(spectral_x)):
  sample_rate=0.001953125
  freq_axis = np.fft.rfftfreq(1536, d=sample_rate)
  

# 61 electrode placement matrix
l=[[]]

# random suffle can be done here. 

final_x = [[[[0 for i in range(11)] for j in range(11)] for k in range(4)] for w in range(len(spectral_x))]
# randomly splitting the data into train, validation and test
train_x=final_x[:int(len(final_x)*0.7)]
train_y=y[:int(len(y)*0.7)]
validation_x=final_x[int(len(final_x)*0.7):int(len(final_x)*0.85)]
validation_y=y[int(len(y)*0.7):int(len(y)*0.85)]
test_x=final_x[int(len(final_x)*0.85):]
test_y=y[int(len(y)*0.85):]

#  each having 11x11x4 matrix
# 2D CNN model import
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.utils import to_categorical
# convert y to one hot encoding
train_y = to_categorical(train_y)
validation_y = to_categorical(validation_y)
test_y = to_categorical(test_y)
# define model
model = Sequential()
# add 2D convolution such that output is of size (11,11,64)
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(20,20,4), padding='same', strides=(1,1)))
# add 2D convolution such that output is of size (11,11,128)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1,1)))
# add 2D convolution such that output is of size (11,11,256)
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1,1)))
# add 2D convolution such that output is of size (11,11,64)
model.add(Conv2D(64, (1, 1), activation='relu', padding='same', strides=(1,1)))
# add flatten layer
model.add(Flatten())
# add fully connected layer with 1024 neurons
model.add(Dense(1024, activation='relu'))
# add fully connected layer with 2 neurons
model.add(Dense(2, activation='softmax'))
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model structure
print(model.summary())
# fit model
history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=10, batch_size=32)
# evaluate model
scores = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# plot error and accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()