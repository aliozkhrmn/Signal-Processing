import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack

# define filter parameters
lower_bnd = 10 # Hz
upper_bnd = 18 # Hz

lower_trans = .1
upper_trans = .1

samprate  = 2048 # Hz
filtorder = 14*np.round(samprate/lower_bnd)+1

filter_shape = [ 0,0,1,1,0,0 ]
filter_freqs = [ 0, lower_bnd*(1-lower_trans), lower_bnd, upper_bnd, \
                upper_bnd+upper_bnd*upper_trans,  samprate/2 ]

filterkern = signal.firls(filtorder,filter_freqs,filter_shape,fs=samprate)
hz = np.linspace(0,samprate/2,int(np.floor(len(filterkern)/2)+1))
filterpow = np.abs(scipy.fftpack.fft(filterkern))**2


# let's see it
plt.subplot(121)
plt.plot(filterkern)
plt.xlabel('Time points')
plt.title('Filter kernel (firls)')


# plot amplitude spectrum of the filter kernel
plt.subplot(122)
plt.plot(hz,filterpow[:len(hz)],'ks-')
plt.plot(filter_freqs,filter_shape,'ro-')

# make the plot look nicer
plt.xlim([0,upper_bnd+20])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.title('Frequency response')
plt.show()


## now apply to random noise data

filtnoise = signal.filtfilt(filterkern,1,np.random.randn(samprate*10))
timevec = np.arange(0,len(filtnoise))/samprate

plt.figure()
# plot time series
plt.subplot(121)
plt.plot(timevec,filtnoise)
plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')
plt.title('Filtered noise')


# plot power spectrum
noisepower = np.abs(scipy.fftpack.fft(filtnoise))**2
plt.subplot(122)
plt.plot(np.linspace(0,samprate,len(noisepower)),noisepower,'k')
plt.xlim([0,60])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Spectrum of filtered noise')
plt.show()