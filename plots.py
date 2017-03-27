import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def mag_plot(t,m,merr=None):
    """
    Given input times, magnitudes and 
    errors (if given), plots the input 
    data. Times are assumed to be in JD.
    """
    if merr is not None:
        plt.errorbar(t-2450000,m,yerr=merrr,fmt='.')
    else:
        plt.plot(t,m,'.')
    xlabel('Time - 2450000 (BJD)')
    ylabel('Magnitude')
    plt.gca().invert_yaxis()
