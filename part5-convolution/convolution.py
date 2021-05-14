import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import copy
import pylab as pl
import time
from IPython import display

## first example to build intuition

signal1 = np.concatenate( (np.zeros(30),np.ones(2),np.zeros(20),np.ones(30),2*np.ones(10),np.zeros(30),-np.ones(10),np.zeros(40)) ,axis=0)
kernel  = np.exp( -np.linspace(-2,2,20)**2 )
kernel  = kernel/sum(kernel)
N = len(signal1)

plt.subplot(311)
plt.plot(kernel,'k')
plt.xlim([0,N])
plt.title('Kernel')

plt.subplot(312)
plt.plot(signal1,'k')
plt.xlim([0,N])
plt.title('Signal')

plt.subplot(313)
plt.plot( np.convolve(signal1,kernel,'same') ,'k')
plt.xlim([0,N])
plt.title('Convolution result')

plt.show()

#####################
## in a bit more detail

plt.figure()
# signal
signal1 = np.zeros(20)
signal1[8:15] = 1

# convolution kernel
kernel = [1,.8,.6,.4,.2]

# convolution sizes
nSign = len(signal1)
nKern = len(kernel)
nConv = nSign + nKern - 1


# plot the signal
plt.subplot(311)
plt.plot(signal1,'o-')
plt.xlim([0,nSign])
plt.title('Signal')

# plot the kernel
plt.subplot(312)
plt.plot(kernel,'o-')
plt.xlim([0,nSign])
plt.title('Kernel')


# plot the result of convolution
plt.subplot(313)
plt.plot(np.convolve(signal1,kernel,'same'),'o-')
plt.xlim([0,nSign])
plt.title('Result of convolution')
plt.show()


##########################
plt.figure()
## convolution in animation

half_kern = int(np.floor(nKern / 2))

# flipped version of kernel
kflip = kernel[::-1]  # -np.mean(kernel)

# zero-padded data for convolution
dat4conv = np.concatenate((np.zeros(half_kern), signal1, np.zeros(half_kern)), axis=0)

# initialize convolution output
conv_res = np.zeros(nConv)

# run convolution

for ti in range(half_kern, nConv - half_kern):
    # get a chunk of data
    tempdata = dat4conv[ti - half_kern:ti + half_kern + 1]

    # compute dot product (don't forget to flip the kernel backwards!)
    conv_res[ti] = np.sum(tempdata * kflip)

    # draw plot
    pl.cla()  # clear the axis
    plt.plot(signal1)
    plt.plot(np.arange(ti - half_kern, ti + half_kern + 1), kflip)
    plt.plot(np.arange(half_kern + 1, ti), conv_res[half_kern + 1:ti])
    plt.xlim([0, nConv + 1])

    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(.1)

# cut off edges
conv_res = conv_res[half_kern:-half_kern]
## convolution in animation


# compare "manual" and Python convolution result
plt.figure()

py_conv = np.convolve(signal1,kernel,'same')


plt.plot(conv_res,'o-',label='Time-domain convolution')
plt.plot(py_conv,'-',label='np.convolve()')
plt.legend()
plt.show()
