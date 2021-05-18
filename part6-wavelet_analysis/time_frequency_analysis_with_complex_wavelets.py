import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy
import scipy.io as sio
import copy

# data from http://www.vibrationdata.com/Solomon_Time_History.zip

equake = np.loadtxt('C:\\Users\\ali\\Desktop\\signal_processing_python\\wavelets\\Solomon_Time_History.txt')

# more convenient
times  = equake[:,0]
equake = equake[:,1]
srate  = np.round( 1/np.mean(np.diff(times)) )


## plot the signal

# time domain
plt.subplot(211)
plt.plot(times/60/60,equake)
plt.xlim([times[0]/60/60,times[-1]/60/60])
plt.xlabel('Time (hours)')

# frequency domain using pwelch
plt.subplot(212)
winsize = srate*60*10 # window size of 10 minutes
f, welchpow = scipy.signal.welch(equake,fs=srate,window=np.hanning(winsize),nperseg=winsize,noverlap=winsize/4)
plt.semilogy(f,welchpow)
plt.xlabel('frequency [Hz]')
plt.ylabel('Power')
plt.ylim([10e-11,10e-6])
plt.show()


########################


## setup time-frequency analysis

# parameters (in Hz)
numFrex = 40
minFreq = 2
maxFreq = srate / 2
npntsTF = 1000  # this one's in points

# frequencies in Hz
frex = np.linspace(minFreq, maxFreq, numFrex)

# wavelet widths (FWHM in seconds)
fwhms = np.linspace(5, 15, numFrex)

# time points to save for plotting
tidx = np.arange(1, len(times), npntsTF)

# setup wavelet and convolution parameters
wavet = np.arange(-10, 10, 1 / srate)
halfw = int(np.floor(len(wavet) / 2))
nConv = len(times) + len(wavet) - 1

# create family of Morlet wavelets
cmw = np.zeros((len(wavet), numFrex), dtype=complex)
# loop over frequencies and create wavelets
for fi in range(0, numFrex):
    cmw[:, fi] = np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-(4 * np.log(2) * wavet ** 2) / fwhms[fi] ** 2)

# plot them
plt.figure()
plt.pcolormesh(wavet, frex, np.abs(cmw).T, vmin=0, vmax=1)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.show()


#######################################


## run convolution

# initialize time-frequency matrix
tf = np.zeros((len(frex), len(tidx)))
tfN = np.zeros((len(frex), len(tidx)))

# baseline time window for normalization
basetidx = [0, 0]
basetidx[0] = np.argmin((times - -1000) ** 2)
basetidx[1] = np.argmin(times ** 2)
basepow = np.zeros(numFrex)

# spectrum of data
dataX = scipy.fftpack.fft(equake, nConv)

# loop over frequencies for convolution
for fi in range(0, numFrex):
    # create wavelet
    waveX = scipy.fftpack.fft(cmw[:, fi], nConv)
    waveX = waveX / np.max(waveX)  # normalize

    # convolve
    as1 = scipy.fftpack.ifft(waveX * dataX)
    # trim
    as1 = as1[halfw:-halfw]

    # power time course at this frequency
    powts = np.abs(as1) ** 2

    # baseline (pre-quake)
    basepow[fi] = np.mean(powts[range(basetidx[0], basetidx[1])])

    tf[fi, :] = 10 * np.log10(powts[tidx])
    tfN[fi, :] = 10 * np.log10(powts[tidx] / basepow[fi])


###############################33


## show time-frequency maps

# "raw" power
plt.figure()
plt.subplot(211)
plt.pcolormesh(times[tidx],frex,tf,vmin=-150,vmax=-70)
plt.xlabel('Time'), plt.ylabel('Frequency (Hz)')
plt.title('"Raw" time-frequency power')

# pre-quake normalized power
plt.subplot(212)
plt.pcolormesh(times[tidx],frex,tfN,vmin=-15,vmax=15)
plt.xlabel('Time'), plt.ylabel('Frequency (Hz)')
plt.title('"Raw" time-frequency power')
plt.show()


## normalized and non-normalized power
plt.figure()
plt.subplot(211)
plt.plot(frex,np.mean(tf,axis=1),'ks-')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (10log_{10})')
plt.title('Raw power')

plt.subplot(212)
plt.plot(frex,np.mean(tfN,axis=1),'ks-')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (norm.)')
plt.title('Pre-quake normalized power')
plt.show()
