import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.fftpack
import copy

# simulation params
srate = 1000
time  = np.arange(-4,4,1/srate)
pnts  = len(time)

# create sinc function
f = 8
sincfilt = np.sin(2*np.pi*f*time) / time

# adjust NaN and normalize filter to unit-gain
sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
sincfilt = sincfilt/np.sum(sincfilt)

# windowed sinc filter
sincfiltW = sincfilt * np.hanning(pnts)


# plot the sinc filter
plt.subplot(221)
plt.plot(time,sincfilt,'k')
plt.xlabel('Time (s)')
plt.title('Non-windowed sinc function')


# plot the power spectrum
plt.subplot(222)
hz = np.linspace(0,srate/2,int(np.floor(pnts/2)+1))
pw = np.abs(scipy.fftpack.fft(sincfilt))
plt.plot(hz,pw[:len(hz)],'k')
plt.xlim([0,f*3])
plt.yscale('log')
plt.plot([f,f],[0,1],'r--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.show()



# now plot the windowed sinc filter

plt.subplot(223)
plt.plot(time,sincfiltW,'k')
plt.xlabel('Time (s)')
plt.title('Windowed sinc function')

plt.subplot(224)
pw = np.abs(scipy.fftpack.fft(sincfiltW))
plt.plot(hz,pw[:len(hz)])
plt.xlim([0,f*3])
plt.yscale('log')
plt.plot([f,f],[0,1],'r--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.show()


## apply the filter to noise

# generate data as integrated noise
data = np.cumsum( np.random.randn(pnts) )

# reflection
datacat = np.concatenate( (data,data[::-1]) ,axis=0)

# apply filter (zero-phase-shift)
dataf = signal.lfilter(sincfiltW,1,datacat)
dataf = signal.lfilter(sincfiltW,1,dataf[::-1])

# flip forwards and remove reflected points
dataf = dataf[-1:pnts-1:-1]

# compute spectra of original and filtered signals
powOrig = np.abs(scipy.fftpack.fft(data)/pnts)**2
powFilt = np.abs(scipy.fftpack.fft(dataf)/pnts)**2



# plot
plt.figure()
plt.subplot(121)
plt.plot(time,data,label='Original')
plt.plot(time,dataf,label='Windowed-sinc filtred')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# plot original and filtered spectra
plt.subplot(122)
plt.plot(hz,powOrig[:len(hz)],label='Original')
plt.plot(hz,powFilt[:len(hz)],label='Windowed-sinc filtred')
plt.xlim([0,40])
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.legend()
plt.show()

## with different windowing functions

sincfiltW = np.zeros((3, pnts))

tapernames = ['Hann', 'Hamming', 'Gauss']

# with Hann taper
# sincfiltW[0,:] = sincfilt * np.hanning(pnts)
hannw = .5 - np.cos(2 * np.pi * np.linspace(0, 1, pnts)) / 2
sincfiltW[0, :] = sincfilt * hannw

# with Hamming taper
# sincfiltW[1,:] = sincfilt * np.hamming(pnts)
hammingw = .54 - .46 * np.cos(2 * np.pi * np.linspace(0, 1, pnts))
sincfiltW[1, :] = sincfilt * hammingw

# with Gaussian taper
sincfiltW[2, :] = sincfilt * np.exp(-time ** 2)

# plot them
plt.figure()
for filti in range(0, len(sincfiltW)):
    plt.subplot(121)
    plt.plot(time, sincfiltW[filti, :])

    plt.subplot(122)
    pw = np.abs(scipy.fftpack.fft(sincfiltW[filti, :]))
    plt.plot(hz, pw[:len(hz)], label=tapernames[filti])
    plt.xlim([f - 3, f + 10])
    plt.yscale('log')

plt.legend()
plt.show()