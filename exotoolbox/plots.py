import numpy as np
import matplotlib.pyplot as plt
import corner
from glob import glob
import pickle

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


def corner_plot(folder, planet_only=False):
    """
    This function will generate corner plots of posterios from `juliet` in a given folder.
    
    Parameters:
    -----------
    folder : str
        Path of the folder where the .pkl file is located
    planet_only : bool
        Boolean on whether to make corner plot of only
        planetary parameters
        Default is False
    
    Returns:
    --------
    corner plot : .png file
        stored inside folder directory
    """
    pcl = glob(folder + '/*.pkl')[0]
    post = pickle.load(open(pcl, 'rb'), encoding='latin1')
    p1 = post['posterior_samples']
    lst = []
    if not planet_only:
        for i in p1.keys():
            gg = i.split('_')
            if ('p1' in gg) or ('p2' in gg) or ('p3' in gg) or ('p4' in gg) or ('rho' in gg) or ('mflux' in gg) or ('sigma' in gg) or ('GP' in gg) or ('mdilution' in gg) or ('q1' in gg) or ('q2' in gg) or (gg[0][0:5] == 'theta'):
                lst.append(i)
    else:
        for i in p1.keys():
            gg = i.split('_')
            if 'p1' in gg or 'p2' in gg or 'p3' in gg or 'p4' in gg or 'q1' in gg or 'q2' in gg or 'rho' in gg:
                lst.append(i)
    if 't0' in lst[0].split('_'):
        t01 = np.floor(p1[lst[0]][0])
        cd = p1[lst[0]] - t01
        lst[0] = lst[0] + ' - ' + str(t01)
    elif 'fp' in lst[0].split('_'):
        cd = p1[lst[0]]*1e6
        lst[0] = lst[0] + ' (in ppm)'
    elif (lst[0][0:3] == 'p_p') or (lst[0][0:4] == 'p1_p') or (lst[0][0:4] == 'p2_p'):
        cd = p1[lst[0]]
        if len(lst[0].split('_')) > 3:
            lst[0] = '_'.join(lst[0].split('_')[0:3]) + '_et al.'
        else:
            lst[0] = lst[0]
    elif (lst[0][0:2] == 'q1') or (lst[0][0:2] == 'q2'):
        cd = p1[lst[0]]
        if len(lst[0].split('_')) > 2:
            lst[0] = '_'.join(lst[0].split('_')[0:2]) + '_et al.'
        else:
            lst[0] = lst[0]
    else:
        cd = p1[lst[0]]
    for i in range(len(lst)-1):
        if 't0' in lst[i+1].split('_'):
            t02 = np.floor(p1[lst[i+1]][0])
            cd1 = p1[lst[i+1]] - t02
            cd = np.vstack((cd, cd1))
            lst[i+1] = lst[i+1] + ' - ' + str(t02)
        elif 'fp' in lst[i+1].split('_'):
            cd = np.vstack((cd, p1[lst[i+1]]*1e6))
            lst[i+1] = lst[i+1] + ' (in ppm)'
        elif (lst[i+1][0:3] == 'p_p') or (lst[i+1][0:4] == 'p1_p') or (lst[i+1][0:4] == 'p2_p'):
            cd = np.vstack((cd, p1[lst[i+1]]))
            if len(lst[i+1].split('_')) > 3:
                lst[i+1] = '_'.join(lst[i+1].split('_')[0:3]) + '_et al.'
            else:
                lst[i+1] = lst[i+1]
        elif (lst[i+1][0:2] == 'q1') or (lst[i+1][0:2] == 'q2'):
            cd = np.vstack((cd, p1[lst[i+1]]))
            if len(lst[i+1].split('_')) > 2:
                lst[i+1] = '_'.join(lst[i+1].split('_')[0:2]) + '_et al.'
            else:
                lst[i+1] = lst[i+1]
        else:
            cd = np.vstack((cd, p1[lst[i+1]]))
    data = np.transpose(cd)
    value = np.median(data, axis=0)
    ndim = len(lst)
    fig = corner.corner(data, labels=lst)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        ax = axes[i,i]
        ax.axvline(value[i], color = 'r')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value[xi], color = 'r')
            ax.axhline(value[yi], color = 'r')
            ax.plot(value[xi], value[yi], 'sr')

    fig.savefig(folder + "/corner.png")
    plt.close(fig)