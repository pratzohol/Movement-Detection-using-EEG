import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import keras
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from typing import final
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
for subjects in range(1, 6):
    for fff in range(1, 11):
        x = []
        y = []

        # have to run code below this for all 10 runs for subject1
        if fff < 10:
            data = mne.io.read_raw_gdf(
                'S0'+str(subjects)+'_MI/motorimagination_subject'+str(subjects)+'_run'+str(fff)+'.gdf')
        else:
            data = mne.io.read_raw_gdf(
                'S0'+str(subjects)+'_MI/motorimagination_subject'+str(subjects)+'_run'+str(10)+'.gdf')
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

        def calc_psd(temp_psd, f_s):
            f, Pxx_den = signal.periodogram(temp_psd, f_s)
            temp_4 = []
            temp_4.append(np.average(Pxx_den[0:3*4+1]))
            temp_4.append(np.average(Pxx_den[3*4:3*8+1]))
            temp_4.append(np.average(Pxx_den[3*8:3*12+1]))
            temp_4.append(np.average(Pxx_den[3*12:3*30+1]))
            return temp_4

        for i in range(len(x)):
            for j in range(len(x[0])):
                spectral_x[i][j] = calc_psd(x[i][j], f_s)

        # 61 electrode placement matrix
        for i in range(len(x)):
            image_4d = []
            for j in range(4):
                temp_band = [[0, 0, 0, spectral_x[i][0][j], 0, spectral_x[i][1][j], 0, spectral_x[i][2][j], 0, spectral_x[i][3][j], 0, spectral_x[i][4][j], 0, 0, 0],
                             [0, 0, spectral_x[i][5][j], 0, spectral_x[i][6][j], 0, spectral_x[i][7][j], 0,
                                 spectral_x[i][8][j], 0, spectral_x[i][9][j], 0, spectral_x[i][10][j], 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, spectral_x[i][11][j], 0, spectral_x[i][12][j], 0, spectral_x[i][13][j], 0, spectral_x[i]
                              [14][j], 0, spectral_x[i][15][j], 0, spectral_x[i][16][j], 0, spectral_x[i][17][j], 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [spectral_x[i][18][j], 0, spectral_x[i][19][j], 0, spectral_x[i][20][j], 0, spectral_x[i][21][j], 0,
                              spectral_x[i][22][j], 0, spectral_x[i][23][j], 0, spectral_x[i][24][j], 0, spectral_x[i][25][j]],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, spectral_x[i][26][j], 0, spectral_x[i][27][j], 0, spectral_x[i][28][j], 0, spectral_x[i]
                              [29][j], 0, spectral_x[i][30][j], 0, spectral_x[i][31][j], 0, spectral_x[i][32][j], 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [spectral_x[i][33][j], 0, spectral_x[i][34][j], 0, spectral_x[i][35][j], 0, spectral_x[i][36][j], 0,
                              spectral_x[i][37][j], 0, spectral_x[i][38][j], 0, spectral_x[i][39][j], 0, spectral_x[i][40][j]],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, spectral_x[i][41][j], 0, spectral_x[i][42][j], 0, spectral_x[i][43][j], 0, spectral_x[i]
                              [44][j], 0, spectral_x[i][45][j], 0, spectral_x[i][46][j], 0, spectral_x[i][47][j], 0],
                             [0, 0, spectral_x[i][48][j], 0, spectral_x[i][49][j], 0, spectral_x[i][50][j], 0,
                              spectral_x[i][51][j], 0, spectral_x[i][52][j], 0, spectral_x[i][53][j], 0, 0],
                             [0, 0, 0, spectral_x[i][54][j], 0, spectral_x[i][55][j], 0, spectral_x[i]
                              [56][j], 0, spectral_x[i][57][j], 0, spectral_x[i][58][j], 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, spectral_x[i][59][j], 0,
                              spectral_x[i][60][j], 0, 0, 0, 0, 0, 0]
                             ]
                image_4d.append(temp_band)
            final_x.append(image_4d)
        final_y.extend(y)
# make heatmap of final_x[0][0]
final_x = np.array(final_x)
final_y = np.array(final_y)
print(final_x.shape)
print(final_y.shape)
# plt.imshow(final_x[0][0])
# plt.show()

# random suffle can be done here.

# final_x = [[[[0 for i in range(11)] for j in range(11)] for k in range(4)] for w in range(len(spectral_x))]

# sklearn train test and validation split
train_x, test_x, train_y, test_y = train_test_split(
    final_x, final_y, test_size=0.3, random_state=42)
test_x, validation_x, test_y, validation_y = train_test_split(
    test_x, test_y, test_size=0.5, random_state=42)

# randomly splitting the data into train, validation and test
# train_x = final_x[:int(len(final_x)*0.7)]
# train_y = final_y[:int(len(final_y)*0.7)]
# validation_x = final_x[int(len(final_x)*0.7):int(len(final_x)*0.85)]
# validation_y = final_y[int(len(final_y)*0.7):int(len(final_y)*0.85)]
# test_x = final_x[int(len(final_x)*0.85):]
# test_y = final_y[int(len(final_y)*0.85):]

# reshape
train_x = train_x.reshape(train_x.shape[0], 15, 15, 4)
validation_x = validation_x.reshape(validation_x.shape[0], 15, 15, 4)
test_x = test_x.reshape(test_x.shape[0], 15, 15, 4)
print(train_x.shape)

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
#  each having 11x11x4 matrix
# 2D CNN model import
# from keras.utils import to_categorical
# convert y to one hot encoding
train_y = tf.keras.utils.to_categorical(train_y)
validation_y = tf.keras.utils.to_categorical(validation_y)
test_y = tf.keras.utils.to_categorical(test_y)
# define model
model = Sequential()
# add 2D convolution such that output is of size (11,11,64)
model.add(Conv2D(64, (4, 4), activation='relu', input_shape=(
    15, 15, 4), padding='same', strides=(1, 1)))
# add 2D convolution such that output is of size (11,11,128)
model.add(Conv2D(128, (4, 4), activation='relu',
          padding='same', strides=(1, 1)))
# add 2D convolution such that output is of size (11,11,256)
model.add(Conv2D(256, (4, 4), activation='relu',
          padding='same', strides=(1, 1)))
# add 2D convolution such that output is of size (11,11,64)
model.add(Conv2D(64, (4, 4), activation='relu', padding='same', strides=(1, 1)))
# add flatten layer
model.add(Flatten())
# add fully connected layer with 4096 neurons
# model.add(Dense(4096,activation='relu'))
# add fully connected layer with 1024 neurons
model.add(Dense(256, activation='relu'))
# add dropout layer
model.add(Dropout(0.5))
# add early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# add fully connected layer with max(y) neurons
model.add(Dense(max(y)+1, activation='softmax'))
# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# model structure
print(model.summary())
# fit model
history = model.fit(train_x, train_y, validation_data=(
    validation_x, validation_y), epochs=30, batch_size=32, callbacks=[early_stopping])
# evaluate model
scores = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# plot error and accuracy
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
