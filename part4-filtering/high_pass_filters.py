import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import scipy.fftpack
import scipy.io.wavfile
import copy

# generate 1/f noise
N   = 8000
fs  = 1000
as1 = np.random.rand(N) * np.exp(-np.arange(0,N)/200)
fc  = as1 * np.exp(1j*2*np.pi*np.random.rand(len(as1)))
noise = np.real(scipy.fftpack.ifft(fc)) * N



### create frequency-domain Gaussian
hz = np.linspace(0,fs,N)
s  = 4*(2*np.pi-1)/(4*np.pi); # normalized width
x  = hz-30                    # shifted frequencies
fg = np.exp(-.5*(x/s)**2)     # gaussian

fc = np.random.rand(N) * np.exp(1j*2*np.pi*np.random.rand(N))
fc = fc * fg

# generate signal from Fourier coefficients, and add noise
signal1 = np.real( scipy.fftpack.ifft(fc) )*N
data = signal1 + noise
time = np.arange(0,N)/fs


### plot the data
plt.subplot(121)
plt.plot(time,data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Data = signal + noise')
plt.show()

plt.subplot(122)
plt.plot(hz,np.abs(scipy.fftpack.fft(signal1)/N)**2,label='signal')
plt.plot(hz,np.abs(scipy.fftpack.fft(noise)/N)**2,label='noise')
plt.legend()
plt.xlim([0,100])
plt.title('Frequency domain representation of signal and noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.show()

plt.figure()

## now for high-pass filter

# specify filter cutoff (in Hz)
filtcut = 25

# generate filter coefficients (Butterworth)
filtb,filta = signal.butter(7,filtcut/(fs/2),btype='high')

# test impulse response function (IRF)
impulse  = np.zeros(1001)
impulse[501] = 1
fimpulse = signal.filtfilt(filtb,filta,impulse)
imptime  = np.arange(0,len(impulse))/fs


# plot impulse and IRF
plt.subplot(221)
plt.plot(imptime,impulse,label='Impulse')
plt.plot(imptime,fimpulse/np.max(fimpulse),label='Impulse response')
plt.xlabel('Time (s)')
plt.legend()
plt.title('Time domain filter characteristics')


# plot spectrum of IRF
plt.subplot(222)
hz = np.linspace(0,fs/2,3000)
imppow = np.abs(scipy.fftpack.fft(fimpulse,2*len(hz)))**2
plt.plot(hz,imppow[:len(hz)],'k')
plt.plot([filtcut,filtcut],[0,1],'r--')
plt.xlim([0,60])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Frequency domain filter characteristics')
plt.show()


# now filter the data and compare against the original
fdata = signal.filtfilt(filtb,filta,data)
plt.subplot(223)
plt.plot(time,signal1,label='Original')
plt.plot(time,fdata,label='Filtered')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time domain')
plt.show()


### power spectra of original and filtered signal
signalX = np.abs(scipy.fftpack.fft(signal1)/N)**2
fdataX  = np.abs(scipy.fftpack.fft(fdata)/N)**2
hz = np.linspace(0,fs,N)

plt.subplot(224)
plt.plot(hz,signalX[:len(hz)],label='original')
plt.plot(hz,fdataX[:len(hz)],label='filtered')
plt.xlim([20,60])
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Frequency domain')
plt.show()