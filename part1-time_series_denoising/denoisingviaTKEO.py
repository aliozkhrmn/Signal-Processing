import scipy.io as sio
import copy
import numpy as np
import matplotlib.pyplot as plt

emgdata = sio.loadmat('emg4TKEO.mat')

# extracting time and data as variables
emgtime = emgdata['emgtime'][0]
emg = emgdata['emg'][0]

# initializing filtered signal
emgf = copy.deepcopy(emg)

emgf[1:-1] = emg[1:-1]**2 - emg[0:-2]*emg[2:]

# Converting both signals to zscore

#finding timepoint zero
time0 = np.argmin(emgtime**2)

# convert original EMG to z-score from time-zero
emgZ = (emg-np.mean(emg[0:time0])) / np.std(emg[0:time0])

# z-scoe for filtered EMG energy
emgZf = (emgf-np.mean(emgf[0:time0])) / np.std(emgf[0:time0])

# plt.plot(emgtime, emg/np.max(emg), 'b', label='EMG')
# plt.plot(emgtime, emgf/np.max(emgf), 'r', label='TKEO Energy')
# plt.xlabel('Time(ms)')
# plt.ylabel('Amplitude or Energy')
# plt.legend()
# plt.show()

plt.plot(emgtime,emgZ,'b',label='EMG')
plt.plot(emgtime,emgZf,'m',label='TKEO energy')
plt.xlabel('Time (ms)')
plt.ylabel('Zscore relative to pre-stimulus')
plt.legend()
plt.show()