import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def mag_plot(t,m,merr=None):
    """
    Given input times, magnitudes and 
    errors (if given), plots the input 
    data. Times are assumed to be in JD.
    """
    if np.min(t)>2450000:
        dt = 2450000.
        plt.xlabel('Time - 2450000 (BJD)')
    else:
        dt = 0.
        plt.xlabel('Time (BJD)')
    if merr is not None:
        plt.errorbar(t-dt,m,yerr=merr,fmt='.')
    else:
        plt.plot(t-dt,m,'.')
    plt.ylabel('Magnitude')
    plt.gca().invert_yaxis()
