import numpy as np
import matplotlib.pyplot as plt

def plot_func(func, k):
    plt.figure(2)
    x = np.linspace(0, 2*np.pi, num=1000)
    plt.plot(x,func(x, k))
    plt.show()
