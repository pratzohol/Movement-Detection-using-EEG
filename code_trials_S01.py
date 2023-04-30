import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import keras
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from typing import final
from utils import *
import mne
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

event_list = [1536, 1537, 1538, 1539, 1540, 1541, 1542]
f_s = 512
t_s = 0  # Trials of 5 seconds or Full length trial/ 0 for only 2 to 5 seconds
t_f = 3.0
sample_s = int(t_s*f_s)
sample_f = int(t_f*f_s)

final_x = []
final_y = []
for ff in range(1, 11):
    x = []
    y = []

    data = mne.io.read_raw_gdf('S01_MI/motorimagination_subject1_run' + str(ff) + '.gdf')
    eeg_data = data.get_data()
    info = data.info

    event_info = data._annotations.description
    event_onset = data._annotations.onset
    time_values = data.times

    for e in range(len(event_info)):
        channel_data = []
        if (int(event_info[e]) in event_list):
            event_sample = int(event_onset[e]*f_s)
            for ch in range(61):
                channel_data.append(
                    eeg_data[ch, event_sample+sample_s:event_sample+sample_f])
            x.append(np.reshape(channel_data, (61, 1536)))
            y.append(event_list.index(int(event_info[e]))+1)

    spectral_x = [[0 for j in range(len(x[0]))] for i in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            spectral_x[i][j] = calc_psd(x[i][j], f_s)

    # 61 electrode placement matrix
    for i in range(len(x)):
        image_4d = []
        for j in range(4):
            tb = temp_band(spectral_x, i, j)
            image_4d.append(tb)
        final_x.append(image_4d)
    final_y.extend(y)


# train = 70%, validation = 15%, test = 15%
train_x, test_x, train_y, test_y = train_test_split(final_x, final_y, test_size=0.3, random_state=42)
test_x, validation_x, test_y, validation_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)


# reshape
train_x = train_x.reshape(train_x.shape[0], 15, 15, 4)
validation_x = validation_x.reshape(validation_x.shape[0], 15, 15, 4)
test_x = test_x.reshape(test_x.shape[0], 15, 15, 4)


# normalize the data with scaler
scaler = preprocessing.StandardScaler()
for i in range(len(train_x)):
    for j in range(4):
        train_x[i][:, :, j] = scaler.fit_transform(train_x[i][:, :, j])
for i in range(len(validation_x)):
    for j in range(4):
        validation_x[i][:, :, j] = scaler.fit_transform(
            validation_x[i][:, :, j])
for i in range(len(test_x)):
    for j in range(4):
        test_x[i][:, :, j] = scaler.fit_transform(test_x[i][:, :, j])

# each having 11x11x4 matrix
train_y = tf.keras.utils.to_categorical(train_y)
validation_y = tf.keras.utils.to_categorical(validation_y)
test_y = tf.keras.utils.to_categorical(test_y)


"""Model :
    - add 2D convolution such that output is of size (11,11,64)
    - add 2D convolution such that output is of size (11,11,128)
    - add 2D convolution such that output is of size (11,11,256)
    - add 2D convolution such that output is of size (11,11,64)
"""
model = Sequential()
model.add(Conv2D(64, (4, 4), activation='relu', input_shape=(15, 15, 4), padding='same', strides=(1, 1))) 
model.add(Conv2D(128, (4, 4), activation='relu', padding='same', strides=(1, 1)))
model.add(Conv2D(256, (4, 4), activation='relu', padding='same', strides=(1, 1)))
model.add(Conv2D(64, (4, 4), activation='relu', padding='same', strides=(1, 1)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.add(Dense(max(y)+1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=30, batch_size=32, callbacks=[early_stopping])

scores = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
