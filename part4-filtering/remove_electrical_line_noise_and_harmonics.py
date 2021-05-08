import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy.io as sio
import scipy.io.wavfile
import copy

# load data
linedata = sio.loadmat('C:\\Users\\ali\\Desktop\\signal_processing_python\\Filtering\\lineNoiseData.mat')
data  = np.squeeze(linedata['data'])
srate = linedata['srate'][0]

# time vector
pnts = len(data)
time = np.arange(0,pnts)/srate
time = time.T

# compute power spectrum and frequencies vector
pwr = np.abs(scipy.fftpack.fft(data)/pnts)**2
hz  = np.linspace(0,srate,pnts)


### plotting
# time-domain signal
plt.subplot(121)
plt.plot(time[0:-1:1000],data[0:-1:1000],'k')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time domain')

# plot power spectrum
plt.subplot(122)
plt.plot(hz,pwr,'k')
plt.xlim([0,400])
plt.ylim([0,2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Frequency domain')
plt.show()


## narrowband filter to remove line noise

frex2notch = [50, 150, 250]

# initialize filtered signal
datafilt = data

# loop over frequencies
for fi in range(0, len(frex2notch)):
    # create filter kernel using firwin (fir1 in MATLAB)
    plt.figure()
    frange = [frex2notch[fi] - .5, frex2notch[fi] + .5]
    order = int(250 * (srate / frange[0]))
    order = order + ~order % 2

    # filter kernel
    filtkern = signal.firwin(order, frange, pass_zero=True, fs=srate)

    # visualize the kernel and its spectral response
    plt.subplot(121)
    plt.plot(filtkern)
    plt.title('Time domain')

    plt.subplot(122)
    plt.plot(np.linspace(0, srate, 10000), np.abs(scipy.fftpack.fft(filtkern, 10000)) ** 2)
    plt.xlim([frex2notch[fi] - 30, frex2notch[fi] + 30])
    plt.title('Frequency domain')
    plt.show()

    # recursively apply to data
    datafilt = signal.filtfilt(filtkern, 1, datafilt)

plt.figure()
### plot the signal
plt.subplot(121)
plt.plot(time, data, 'k', label='Original')
plt.plot(time, datafilt, 'r', label='Notched')
plt.xlabel('Time (s)')
plt.legend()

# compute the power spectrum of the filtered signal
pwrfilt = np.abs(scipy.fftpack.fft(datafilt) / pnts) ** 2

# plot power spectrum
plt.subplot(122)
plt.plot(hz, pwr, 'k', label='Original')
plt.plot(hz, pwrfilt, 'r', label='Notched')
plt.xlim([0, 400])
plt.ylim([0, 2])
plt.title('Frequency domain')
plt.show()