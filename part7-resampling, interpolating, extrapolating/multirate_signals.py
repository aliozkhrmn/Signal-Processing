import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy

## create multichannel signal with multiple sampling rates

# initialize signals, time vectors, and sampling rates

# sampling rates in Hz
fs = [10, 40, 83]
timez = np.zeros([3], dtype=object)
signals = np.zeros([3], dtype=object)

# create signals
for si in range(0, 3):
    # create signal
    signals[si] = np.cumsum(np.sign(np.random.randn(fs[si])))

    # create time vector
    timez[si] = np.arange(0, fs[si]) / fs[si]

# # plot all signals
color = 'kbr'
shape = 'os^'

for si in range(0, 3):
    plt.plot(timez[si], signals[si], color[si] + shape[si] + '-')

plt.xlabel('Time (s)')
plt.show()

###################################3

## upsample to fastest frequency

# in Hz
newSrate = np.max(fs)
whichIsFastest = np.argmax(fs)

# need to round in case it's not exact
newNpnts = np.round(len(signals[whichIsFastest]) * (newSrate / fs[whichIsFastest]))

# new time vector after upsampling
newTime = np.arange(0, newNpnts) / newSrate

# ## continue on to interpolation
# # initialize (as matrix!)
newsignals = np.zeros((len(fs), len(newTime)))

for si in range(0, len(fs)):
    # interpolate using griddata
    newsignals[si, :] = griddata(timez[si], signals[si], newTime, method='cubic')

plt.figure()
### plot for comparison
for si in range(0, 3):
    plt.plot(newTime, newsignals[si, :], color[si] + shape[si] + '-')

plt.show()