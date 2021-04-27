import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.fftpack
import scipy.signal

# loading the matfile
matdat  = sio.loadmat('spectral_codeChallenge.mat')
signal = matdat['signal'][0]
srate = matdat['srate'][0][0]
timevec = matdat['time'][0]

N = len(signal)

# plt.plot(timevec, signal)
# plt.show()


winlength = int(srate/2)


# window onset times
winonsets = np.arange(0, int(N - winlength), winlength)

# note: different-length signal needs a different-length Hz vector
hzW = np.linspace(0, srate / 2, int(np.floor(winlength / 2) + 1))

# Hann window
hannw = .5 - np.cos(2 * np.pi * np.linspace(0, 1, int(winlength))) / 2

# initialize the power matrix (windows x frequencies)
signalpwr = np.zeros((len(winonsets), len(hzW)))


# loop over frequencies
for wi in range(0, len(winonsets)):
    # get a chunk of data from this time window
    datachunk = signal[winonsets[wi]:winonsets[wi] + winlength]

    # apply Hann taper to data
    datachunk = datachunk * hannw

    # compute its power
    tmppow = np.abs(scipy.fftpack.fft(datachunk) / winlength) ** 2

    # enter into matrix
    signalpwr[wi,:] = signalpwr[wi,:] + tmppow[0:len(hzW)]


plt.pcolormesh(winonsets*6/5500, hzW, signalpwr.transpose())
plt.xlim([0, 6])
plt.ylim([0, 40])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
# plt.legend()
plt.show()






