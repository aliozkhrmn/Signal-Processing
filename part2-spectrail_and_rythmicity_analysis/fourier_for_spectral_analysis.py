import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.fftpack
import scipy.signal

## Generate a multispectral noisy signal

# simulation parameters
srate = 1234 # in Hz
npnts = srate*2 # 2 seconds
time  = np.arange(0,npnts)/srate

# frequencies to include
frex  = [ 12,18,30 ]

signal = np.zeros(len(time))

# loop over frequencies to create signal
for fi in range(0,len(frex)):
    signal = signal + (fi+1)*np.sin(2*np.pi*frex[fi]*time)

# add some noise
signal = signal + np.random.randn(len(signal))

# amplitude spectrum via Fourier transform
signalX = scipy.fftpack.fft(signal)
signalAmp = 2*np.abs(signalX)/npnts

# vector of frequencies in Hz
hz = np.linspace(0,srate/2,int(np.floor(npnts/2)+1))

fig, axs = plt.subplots(2)
fig.suptitle('Spectral Analysis')

axs[0].plot(time,signal,label='Original')
axs[0].plot(time,np.real(scipy.fftpack.ifft(signalX)),'ro',label='IFFT reconstructed')
axs[0].legend()

axs[1].stem(hz,signalAmp[0:len(hz)],'k')
axs[1].xlim([0,np.max(frex)*3])

