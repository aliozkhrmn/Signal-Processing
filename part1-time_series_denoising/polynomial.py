# This code just for polynomiaş intuition
import numpy as np
import matplotlib.pyplot as plt

order = 3
x = np.linspace(-15,15,100)
y = np.zeros(len(x))

for i in range(order+1):
    y = y + np.random.randn(1)*x**i

plt.plot(x,y)
plt.title(f'Order-{order} polynomial')
plt.show()

