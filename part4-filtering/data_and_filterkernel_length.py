import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy
import copy

# parameters
dataN = 10000
filtN = 5001

# generate data
signal1 = np.random.randn(dataN)

# create filter kernel
fkern = signal.firwin( filtN,.01,pass_zero=True )

# apply filter kernel to data
#fdata = signal.filtfilt(fkern,1,signal1)

# reflect the signal
signalRefl = np.concatenate( (signal1[::-1],signal1,signal1[::-1]),axis=0 )

# apply filter kernel to data
fdataR = signal.filtfilt(fkern,1,signalRefl)

# and cut off edges
fdata = fdataR[dataN:-dataN]

plt.figure(0)
plt.plot(range(0,dataN), signal1, label='original signal')

plt.figure(1)
plt.plot(fkern, label='filter kernel')
plt.show()

plt.figure(0)
plt.plot(fdata, label='filtered signal')
plt.show()

plt.figure(2)

hz = np.linspace(0,1,dataN)

plt.plot(hz,np.abs(scipy.fftpack.fft(signal1))**2,'k',label='Original')
plt.plot(hz,np.abs(scipy.fftpack.fft(fdata))**2,'m',label='Filtered')
plt.title('Frequency domain')
plt.xlim([0,.5])
plt.legend()
plt.show()