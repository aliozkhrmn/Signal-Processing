import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy


## create signals

# different time vectors
tx = np.linspace(0,1.5*np.pi,400)
ty = np.linspace(0,8*np.pi,100)

# different signals
x = np.sin(tx**2) # chirp
y = np.sin(ty);# sine wave


# show them
plt.plot(tx,x,'bs-')
plt.plot(ty,y,'rs-')
plt.xlabel('Time (rad.)')
plt.title('Original')
plt.show()

#############################

## distance matrix

# initialize distance matrix (dm) and set first element to zero
dm = np.zeros((len(x), len(y))) / 0
dm[0, 0] = 0

# distance matrix
for xi in range(1, len(x)):
    for yi in range(1, len(y)):
        cost = np.abs(x[xi] - y[yi])
        dm[xi, yi] = cost + np.nanmin([dm[xi - 1, yi], dm[xi, yi - 1], dm[xi - 1, yi - 1]])

plt.figure()
plt.pcolormesh(ty, tx, dm, vmin=0, vmax=100)
plt.show()

############################


# find minimum for each y
minpath = np.zeros((2,len(x)),'int')

for xi in range(0,len(x)):
    minpath[0,xi] = np.nanmin(dm[xi,:])
    minpath[1,xi] = np.nanargmin(dm[xi,:])


plt.figure()
plt.plot(tx,x,'bs-')
plt.plot(tx,y[minpath[1,:]],'rs-')
plt.xlabel('Time (rad.)')
plt.title('Warped')
plt.show()