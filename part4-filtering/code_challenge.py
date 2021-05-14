import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy.io as sio
import scipy.io.wavfile
import copy

data = sio.loadmat('C:\\Users\\ali\\Desktop\\signal_processing_python\\Filtering\\filtering_codeChallenge.mat')
orig = data['x']
filtered = data['y']
srate = data['fs'][0]

pnts = len(orig)
time = np.arange(0,pnts/srate,1/srate)


hz  = np.linspace(0,srate/2,int(pnts/2))
orig_pwr = np.abs(scipy.fftpack.fft(orig,axis=0))**2
filtered_pwr = np.abs(scipy.fftpack.fft(filtered,axis=0))**2

plt.subplot(121)
plt.plot(time, orig, 'r', label='original signal')
plt.plot(time, filtered, 'k', label='filtered signal')
plt.legend()

plt.subplot(122)
plt.plot(hz[0:len(hz)], orig_pwr[0:len(hz)], 'r', label='original signal')
plt.plot(hz[0:len(hz)], filtered_pwr[0:len(hz)], 'k', label='filtered signal')
plt.legend()
plt.xlim([0,80])
plt.show()


# constructing_and_applying filters
plt.figure()


lower_bnd = 5
upper_bnd = 18
samprate = srate[0]

# first applying high pass filter
forder = int(10*samprate/lower_bnd)+1
filtkern = signal.firwin(forder,lower_bnd,pass_zero=False,fs=samprate)

# spectrum of kernel
hz1 = np.linspace(0,samprate/2,int(np.floor(len(filtkern)/2)+1))
filterpow = np.abs(scipy.fftpack.fft(filtkern))**2
plt.subplot(221)
plt.plot(filtkern)

plt.subplot(222)
plt.plot(hz1,filterpow[:len(hz1)],'k')

plt.xlim([0,upper_bnd+40])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.title('Frequency response')
plt.show()

# zero-phase-shift filter with reflection
filtered1 = signal.filtfilt(filtkern,1,orig.T)


### repeat for low-pass filter
forder = 20*int(samprate/upper_bnd)+1
filtkern = signal.firwin(forder,upper_bnd,fs=samprate,pass_zero=True)

# spectrum of kernel
hz2 = np.linspace(0,samprate/2,int(np.floor(len(filtkern)/2)+1))
filterpow = np.abs(scipy.fftpack.fft(filtkern))**2


plt.subplot(223)
plt.plot(filtkern)

plt.subplot(224)
plt.plot(hz2,filterpow[:len(hz2)],'k')

plt.xlim([0,80])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.title('Frequency response')
plt.show()

filtered1 = signal.filtfilt(filtkern,1,filtered1)

plt.figure()

plt.subplot(121)
plt.plot(time, orig, 'r', label='original signal')
plt.plot(time, filtered, 'k', label='filtered signal')
plt.plot(time, filtered1.T, 'b', label='after filtering')

pwr = np.abs(scipy.fftpack.fft(filtered1.T,axis=0))**2
plt.subplot(122)
plt.plot(hz[0:len(hz)], orig_pwr[0:len(hz)], 'r', label='original signal')
plt.plot(hz[0:len(hz)], filtered_pwr[0:len(hz)], 'k', label='filtered signal')
plt.plot(hz[0:len(hz)], pwr[0:len(hz)], 'b', label='after filtering signal')
plt.xlim([0,80])
plt.legend()


##########################
# second part of filtering

plt.figure()
lower_bnd = 26 # Hz
upper_bnd = 32 # Hz

lower_trans = .1
upper_trans = .1

samprate  = srate[0] # Hz
filtorder = 14*np.round(samprate/lower_bnd)+1

filter_shape = [ 0,0,1,1,0,0 ]
filter_freqs = [ 0, lower_bnd*(1-lower_trans), lower_bnd, upper_bnd, \
                upper_bnd+upper_bnd*upper_trans,  samprate/2 ]

filterkern = signal.firls(filtorder,filter_freqs,filter_shape,fs=samprate)
hz3 = np.linspace(0,samprate/2,int(np.floor(len(filterkern)/2)+1))
filterpow = np.abs(scipy.fftpack.fft(filterkern))**2


# plotting filter kernel
plt.subplot(121)
plt.plot(filterkern)

# plotting kernel spectrum
plt.subplot(122)

plt.plot(hz3[0:len(hz3)],filterpow[0:len(hz3)],'ks-')
plt.plot(filter_freqs,filter_shape,'ro-')
plt.xlim([0,80])
plt.yscale('log')
plt.show()

# applying kernel to data

filtered2 = signal.filtfilt(filterkern,1,orig.T)
print(len(hz))

plt.figure()
plt.subplot(121)
plt.plot(time, orig, 'r', label='original signal')
plt.plot(time, filtered, 'k', label='filtered signal')
plt.plot(time, filtered2.T, 'b', label='after filtering')

pwr = np.abs(scipy.fftpack.fft(filtered2.T,axis=0))**2
plt.subplot(122)
plt.plot(hz[0:len(hz)], orig_pwr[0:len(hz)], 'r', label='original signal')
plt.plot(hz[0:len(hz)], filtered_pwr[0:len(hz)], 'k', label='filtered signal')
plt.plot(hz[0:len(hz)], pwr[0:len(hz)], 'b', label='after filtering signal')
plt.xlim([0,80])
plt.legend()


# both range of filters
filteredd = filtered1 + filtered2

plt.figure()
plt.subplot(121)
plt.plot(time, orig, 'r', label='original signal')
plt.plot(time, filtered, 'k', label='filtered signal')
plt.plot(time, filteredd.T, 'b', label='after filtering')

pwr = np.abs(scipy.fftpack.fft(filteredd.T,axis=0))**2
plt.subplot(122)
plt.plot(hz[0:len(hz)], orig_pwr[0:len(hz)], 'r', label='original signal')
plt.plot(hz[0:len(hz)], filtered_pwr[0:len(hz)], 'k', label='filtered signal')
plt.plot(hz[0:len(hz)], pwr[0:len(hz)], 'b', label='after filtering signal')
plt.xlim([0,80])
plt.legend()