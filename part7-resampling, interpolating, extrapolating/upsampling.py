import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import griddata
import copy

## low-sampling-rate data to upsample

# in Hz
srate = 10

# just some numbers...
data  = np.array( [1, 4, 3, 6, 2, 19] )

# other parameters
npnts = len(data)
time  = np.arange(0,npnts)/srate

# plot the original signal
plt.plot(time,data,'ko-')
plt.show()


## option 1: upsample by a factor

upsampleFactor = 4
newNpnts = npnts*upsampleFactor

# new time vector after upsampling
newTime = np.arange(0,newNpnts)/(upsampleFactor*srate)

# ## option 2: upsample to desired frequency, then cut off points if necessary
#
# # in Hz
# newSrate = 37
#
# # need to round in case it's not exact
# newNpnts = np.round(npnts * (newSrate / srate))
#
# # new time vector after upsampling
# newTime = np.arange(0, newNpnts) / newSrate


## continue on to interpolation

# cut out extra time points
newTime = newTime[newTime<time[-1]]

# the new sampling rate actually implemented
newSrateActual = 1/np.mean(np.diff(newTime))



# interpolate using griddata
updataI = griddata(time, data, newTime, method='cubic')

# plot the upsampled signal
plt.figure()
plt.plot(newTime,updataI,'rs-',label='Upsampled')
plt.plot(time,data,'ko',label='Original')
plt.legend()
plt.show()


## using Python's resample function

# new sampling rate in Hz
newSrate = 40


# use resample function
newNpnts = int( len(data)*newSrate/srate )
updataR = signal.resample(data,newNpnts)

# the new time vector
newTimeR = np.arange(0,newNpnts)/newSrate


# cut out extra time points
updataR  = updataR[newTimeR<time[-1]]
newTimeR = newTimeR[newTimeR<time[-1]]


# and plot it
plt.figure()
plt.plot(newTimeR,updataR,'b^-',label='resample')
plt.plot(newTime,updataI,'rs-',label='Upsampled')
plt.plot(time,data,'ko',label='Original')
plt.legend()
plt.show()