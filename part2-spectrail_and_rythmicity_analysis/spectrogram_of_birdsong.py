import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.fftpack
import scipy.signal


## load in birdcall (source: https://www.xeno-canto.org/403881)

fs,bc = scipy.io.wavfile.read('C:\\Users\\ali\\Desktop\\signal_processing_python\\spectral_and_rhytmicity_analyses\\sigprocMXC_spectral (1)\\XC403881.wav')


# create a time vector based on the data sampling rate
n = len(bc)
timevec = np.arange(0,n)/fs

# plot the data from the two channels
# plt.plot(timevec,bc)
# plt.xlabel('Time (sec.)')
# plt.title('Time domain')
# plt.show()

# compute the power spectrum
hz = np.linspace(0,fs/2,int(np.floor(n/2)+1))
bcpow = np.abs(scipy.fftpack.fft( scipy.signal.detrend(bc[:,0]) )/n)**2

# now plot it
# plt.plot(hz,bcpow[0:len(hz)])
# plt.xlabel('Frequency (Hz)')
# plt.title('Frequency domain')
# plt.xlim([0,8000])
# plt.show()

## time-frequency analysis via spectrogram

frex,time,pwr = scipy.signal.spectrogram(bc[:,0],fs)

plt.pcolormesh(time,frex,pwr,vmin=0,vmax=9)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.show()