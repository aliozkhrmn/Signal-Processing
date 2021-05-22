import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy


## low-sampling-rate data to upsample

# in Hz
srate = 10

# just some numbers...
data  = np.array( [1, 4, 3, 6, 2, 19] )

# other parameters
npnts = len(data)
time  = np.arange(0,npnts)/srate

# plot the original data
plt.subplot(211)
plt.plot(time,data,'go-')
plt.title('Time domain')

# amplitude spectrum
plt.subplot(212)
plt.plot(np.linspace(0,1,npnts),np.abs(scipy.fftpack.fft(data/np.array(npnts))),'go-')
plt.xlabel('Frequency (a.u.)')
plt.show()

#########################################


## interpolation

# new time vector for interpolation
N = 47
newTime = np.linspace(time[0], time[-1], N)

# different interpolation options
interpOptions = ['linear', 'nearest', 'cubic']

plt.figure()
for methodi in range(0, len(interpOptions)):
    # interpolate using griddata
    newdata = griddata(time, data, newTime, method=interpOptions[methodi])

    # plotting
    plt.subplot(121)
    plt.plot(newTime, newdata, 'ks-', label='interpolated')
    plt.plot(time, data, 'go', label='Original')
    plt.title('Time domain ' + interpOptions[methodi])
    plt.legend()

    plt.subplot(122)
    plt.plot(np.linspace(0, 1, N), np.abs(scipy.fftpack.fft(newdata / N)), 'k')
    plt.xlim([0, .5])
    plt.title('Freq. domain ' + interpOptions[methodi])
    plt.show()
