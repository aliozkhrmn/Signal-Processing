import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy

# number of points in signal
n = 10000

# create signal
origsig = np.cumsum( np.random.randn(n) )
signal1 = copy.deepcopy(origsig)


# remove a specified window
boundaryPnts = [ 5000, 6500 ]
signal1[range(boundaryPnts[0],boundaryPnts[1])] = np.nan


# FFTs of pre- and post-window data
fftPre = scipy.fftpack.fft(signal1[ range(boundaryPnts[0]-int(np.diff(boundaryPnts)),boundaryPnts[0])] )
fftPst = scipy.fftpack.fft(signal1[ range(boundaryPnts[1]+1,boundaryPnts[1]+int(np.diff(boundaryPnts)+1))] )

# interpolated signal is a combination of mixed FFTs and straight line
mixeddata = scipy.signal.detrend( np.real(scipy.fftpack.ifft( ( fftPre+fftPst )/2 )))
linedata  = np.linspace(0,1,int(np.diff(boundaryPnts))) * (signal1[boundaryPnts[1]+1]-signal1[boundaryPnts[0]-1]) + signal1[boundaryPnts[0]-1]

# sum together for final result
linterp = mixeddata + linedata

# put the interpolated piece into the signal
filtsig = copy.deepcopy( signal1 )
filtsig[ range(boundaryPnts[0],boundaryPnts[1]) ] = linterp

plt.plot(np.arange(0,n),origsig,label='Original')
plt.plot(np.arange(0,n),signal1+5,label='with gap')
plt.plot(np.arange(0,n),filtsig+10,label='Interpolated')
plt.legend()

## optional zoom
#plt.xlim([boundaryPnts[0]-150, boundaryPnts[1]+150])

plt.show()