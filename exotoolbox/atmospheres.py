import numpy as np
from .utils import *
constants = constants()
G = constants.G()
kB = constants.kB()
amu = constants.amu()
AU = constants.AU()
Rj = constants.Rj()
Rsun = constants.Rsun()
Msun = constants.Msun()
Mj = constants.Mj()

def get_scaleheight(Rp,Mp,aR,Teff,mu = 2.3, albedo = 0., emissivity = 1.):
    """
    Given the planetary radius (Rp, jupiter units), mass (jupiter units), 
    scaled semi-major axis (aR, in stellar units) and the stellar effective temperature (Teff),
    this function calculates the atmospheric scale-height of an atmosphere in km.
    """
    # Planetary gravity:
    g = G*(Mp*Mj)/(Rp*Rj)**2

    # Calculate equilibrium temperature of planet (assuming zero-albedo)
    Teq = Teff * ((1. - albedo)/emissivity)**(1./4.)*np.sqrt(0.5/aR)

    # Get scale-height:
    H = kB*Teq/(mu*amu*g)

    return H/1e3

def get_transpec_signal(Rp,Rstar,Mp,aR,Teff,mu = 2.3, albedo = 0., emissivity = 1.):
    """
    Given the planetary radius (Rp, jupiter units), stellar radius (Rstar, solar units), mass (jupiter units), 
    scaled semi-major axis (aR, in stellar units) and the stellar effective temperature (Teff),
    this function calculates the minimum atmospheric signal in ppm. This has to be multiplied by a factor 
    between 1-3 to estimate the real signal.
    """    
    H = get_scaleheight(Rp,Mp,aR,Teff,mu,albedo,emissivity)
    return (2.*(Rp*Rj)*(H*1e3)/(Rstar*Rsun)**2)*1e6
