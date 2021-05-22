import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy

# get the landscape
signal1 = np.array( [1, 4, 3, 6, 2, 19] )
timevec = np.arange(0,len(signal1))


## two methods of extrapolation

times2extrap = np.arange(-len(signal1),2*len(signal1))


# get extrapolation classes
Flin = scipy.interpolate.interp1d(timevec,signal1,kind='linear',fill_value='extrapolate')
Fcub = scipy.interpolate.interp1d(timevec,signal1,kind='cubic',fill_value='extrapolate')

# now extra/interpolate
extrapLin = Flin(times2extrap)
extrapCub = Fcub(times2extrap)

# # plot them
plt.plot(timevec,signal1,'ks-',label='Original')
plt.plot(times2extrap,extrapLin,'ro-',label='linear')
plt.plot(times2extrap,extrapCub,'bp-',label='cubic')
plt.legend()

## optional zoom
plt.ylim([-100,100])

plt.show()