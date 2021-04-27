import numpy as np
import cmath
import math
import matplotlib.pyplot as plt

# creating the complex numbers

z = 4 + 3j
z = 4 + 3*1j
z = 4 + 3*cmath.sqrt(-1)
z = complex(4,3)
print(f'Real part is: {np.real(z)} and imaginary part is: {np.imag(z)}')

# plotting a complex number

plt.plot( np.real(z),np.imag(z),'rs' )
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.plot([-5,5],[0,0],'k')
plt.plot([0,0],[-5,5],'k')
plt.xlabel('real axis')
plt.ylabel('imag axis')
plt.show()

# Addition and subtraction

a = complex(4,5)
b = 3+2j
z1 = a+b
z2 = a-b
print(f'The addition of a and b is: {z1},\nthe subtraction of a and b is: {z2}')

# Multiplication

z3 = a*b
print(f'Multiplication of a and b is: {z3}')

# creating complex conjugate

a = complex(4,-5)
z4 = np.conj(a)
print(f'The complex conjugate of {a} is: {z4}')

# Division

a = complex(4,-5)
b = complex(7,8)
z5 = a/b
print(f'The division of {a}/{b} is: {z5}')

# Magnitude and Phase

z = 4 + 3j
mag = np.abs(z)
ang = np.angle(z)
print(f'The magnitude is: {mag}, and the angle is: {ang}')
plt.polar([0,ang],[0,mag],'r')
plt.show()
