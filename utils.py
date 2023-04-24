import numpy as np
from scipy import signal

def calc_psd(temp_psd, f_s):
    f, Pxx_den = signal.periodogram(temp_psd, f_s)

    temp_4 = []
    temp_4.append(np.average(Pxx_den[0 : 3 * 4 + 1]))
    temp_4.append(np.average(Pxx_den[3 * 4 : 3 * 8 + 1]))
    temp_4.append(np.average(Pxx_den[3 * 8 : 3 * 12 + 1]))
    temp_4.append(np.average(Pxx_den[3 * 12 : 3 * 30 + 1]))

    return temp_4