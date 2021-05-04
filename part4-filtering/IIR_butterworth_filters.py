import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy
import copy

# filter parameters
srate   = 1024 # hz
nyquist = srate/2
frange  = [20,45]

# create filter coefficients
fkernB,fkernA = signal.butter(4,np.array(frange)/nyquist,btype='bandpass')

# power spectrum of filter coefficients
filtpow = np.abs(scipy.fftpack.fft(fkernB))**2
hz      = np.linspace(0,srate/2,int(np.floor(len(fkernB)/2)+1))

plt.figure(0)
# plotting
plt.subplot(121)
plt.plot(fkernB*1e5,'ks-',label='B')
plt.plot(fkernA,'rs-',label='A')
plt.xlabel('Time points')
plt.ylabel('Filter coeffs.')
plt.title('Time-domain filter coefs')
plt.legend()

plt.subplot(122)
plt.stem(hz,filtpow[0:len(hz)],'ks-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power spectrum filter coeffs.')
plt.show()


plt.figure(1)
## how to evaluate an IIR filter: filter an impulse

# generate the impulse
impres = np.zeros(1001)
impres[501] = 1

# apply the filter
fimp = signal.lfilter(fkernB,fkernA,impres,axis=-1)

# compute power spectrum
fimpX = np.abs(scipy.fftpack.fft(fimp))**2
hz = np.linspace(0,nyquist,int(np.floor(len(impres)/2)+1))


# plot
plt.subplot(131)
plt.plot(impres,'k',label='Impulse')
plt.plot(fimp,'r',label='Filtered')
plt.xlim([1,len(impres)])
plt.ylim([-1.2,1.2])
plt.legend()
plt.xlabel('Time points (a.u.)')
plt.title('Filtering an impulse')
plt.show()

plt.subplot(132)
plt.plot(hz,fimpX[0:len(hz)],'ks-')
plt.plot([0,frange[0],frange[0],frange[1],frange[1],nyquist],[0,0,1,1,0,0],'r')
plt.xlim([0,100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation')
plt.title('Frequency response of filter (Butterworth)')
plt.show()

plt.subplot(133)
plt.plot(hz,10*np.log10(fimpX[0:len(hz)]),'ks-')
plt.xlim([0,100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation')
plt.title('Frequency response of filter (Butterworth)')
plt.show()

## effects of order parameter
plt.figure(2)
orders = range(2, 8)

fkernX = np.zeros((len(orders), 1001))
hz = np.linspace(0, srate, 1001)

# loop over orders
for oi in range(0, len(orders)):
    # create filter kernel
    fkernB, fkernA = signal.butter(orders[oi], np.array(frange) / nyquist, btype='bandpass')
    n = len(fkernB)

    # filter the impulse response and take its power
    fimp = signal.lfilter(fkernB, fkernA, impres, axis=-1)
    fkernX[oi, :] = np.abs(scipy.fftpack.fft(fimp)) ** 2

    # show in plot
    time = np.arange(0, len(fkernB)) / srate
    time = time - np.mean(time)
    plt.subplot(121)
    plt.plot(time, scipy.stats.zscore(fkernB) + oi)
    plt.title('Filter coefficients (B)')

    plt.subplot(122)
    plt.plot(time, scipy.stats.zscore(fkernA) + oi)
    plt.title('Filter coefficients (A)')

#plt.show()

# plot the spectra
plt.figure(3)
plt.subplot(121)
plt.plot(hz, fkernX.T)
plt.plot([0, frange[0], frange[0], frange[1], frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'r')
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation')
plt.title('Frequency response of filter (Butterworth)')
plt.show()

# in log space
plt.subplot(122)
plt.plot(hz, 10 * np.log10(fkernX.T))
plt.xlim([0, 100])
plt.ylim([-80, 2])
plt.title('Frequency response of filter (Butterworth)')
plt.show()