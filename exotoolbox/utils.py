import batman
import numpy as np

def get_MAD_sigma(x,median):
    """
    This function returns the MAD-based standard-deviation.
    """
    mad = np.median(np.abs(x-median))
    return 1.4826*mad

def get_phases(t,P,t0):
    """
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        if phase>=0.5:
            phase = phase - 1.0
    return phase

def init_batman(t,law):
    """
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1,0.3]
    params.limb_dark = law
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law):
    """
    Given input times and transit parameters, returns a transit 
    model. ld_law can be 'quadratic', 'squareroot' or 
    'logarithmic'. If 'linear', it assumes q1 is the linear 
    ld coeff.
    """
    params,m = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.u = [coeff1,coeff2]
    return m.light_curve(params)

def convert_ld_coeffs(ld_law, coeff1, coeff2):
    if ld_law == 'quadratic':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff1/(2.*(coeff1+coeff2))
    elif ld_law=='squareroot':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff2/(2.*(coeff1+coeff2))
    elif ld_law=='logarithmic':
        q1 = (1-coeff2)**2
        q2 = (1.-coeff1)/(1.-coeff2)
    return q1,q2

def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law=='linear':
        return q1,0.
    return coeff1,coeff2

import jdcal
import os
from math import modf
from astropy.time import Time
from calendar import monthrange
def getCalDay(JD):
    year, month, day, hour= jdcal.jd2gcal(JD,0.0)
    hour = hour*24
    minutes = modf(hour)[0]*60.0
    seconds = modf(minutes)[0]*60.0
    hh = int(modf(hour)[1])
    mm = int(modf(minutes)[1])
    ss = seconds
    if(hh<10):
       hh = '0'+str(hh)
    else:
       hh = str(hh)
    if(mm<10):
       mm = '0'+str(mm)
    else:
       mm = str(mm)
    if(ss<10):
       ss = '0'+str(np.round(ss,1))
    else:
       ss = str(np.round(ss,1))
    return year,month,day,hh,mm,ss

def read_tepcat(update=False):
    if not os.path.exists('observables.txt'):
        os.system('wget http://www.astro.keele.ac.uk/jkt/tepcat/observables.txt')
    else:
        if update:
            os.system('wget http://www.astro.keele.ac.uk/jkt/tepcat/observables.txt')
    f = open('observables.txt','r')
    output = {}
    while True:
        line = f.readline()
        if line != '':
            if line[0] != '#':
                System,Type,RAhh,RAmm,RAss,Decdd,Decmm,Decss,Vmag,Kmag,Tlength,Tdepth,T0,T0err,P,Perr,ref = line.split()
                System = '-'.join(System.split('-0'))
                if 'Kepler' not in System:
                 output[System] = {}
                 output[System]['RA'] = RAhh+':'+RAmm+':'+RAss
                 output[System]['DEC'] = Decdd+':'+Decmm+':'+Decss
                 output[System]['V'] = np.float(Vmag)
                 output[System]['TDuration'] = np.float(Tlength)
                 output[System]['TDepth'] = np.float(Tdepth)
                 output[System]['T0'] = np.float(T0)
                 output[System]['Period'] = np.float(P)
        else:
            break
    f.close()
    return output

def transit_predictor(year,month,pname=None,day=None,P=None,Tdur=None,t0=None,exo_source='TEPCAT'):
    """
    This function predicts transits of your favorite planet. If no planet name is given, it is 
    assumed (depending if you feed P, Tdur and t0) you either want to know all transits for the 
    given date(s) or you have your own planet. If no period (P), 
    duration (Tdur) or time of transit center (t0) is feeded, it assumes you want 
    to know all the transits of all known planets on the specified dates. If no 
    day is given, it assumes you want all the transits on a given month.

    Months run from 1 (January) to 12 (December). 
    """
   
    # If no input day, get all days in the month:
    if day is None:
        first_w_day,max_d = monthrange(year, month)
        days = range(1,max_d+1)
    else:
        days = [day]
   
    if P is None or Tdur is None or t0 is None:
        if exo_source == 'TEPCAT':
            exodata = read_tepcat()
        else:
            print '\t Error in transit_predictor: exo_source '+exo_source+' not found.'
            return 0
    else:
        names = ["User's planet"]

    if pname is None:
        planet_list = exodata.keys()
    else:
        corrected_input = ''.join(ch for ch in pname if ch.isalnum())
        for planet in exodata.keys():
            corrected_in_list = ''.join(ch for ch in planet if ch.isalnum())
            if corrected_input.lower() == corrected_in_list.lower() or\
               corrected_input.lower() == corrected_in_list.lower()+'b':
                planet_list = [planet]
                break
        
    for planet in planet_list:
        if P is None or Tdur is None or t0 is None:
            pP = exodata[planet]['Period']
            pTdur = exodata[planet]['TDuration']
            pt0 = exodata[planet]['T0']
        else:
            pP = P
            pTdur = Tdur
            pt0 = t0

        transits_t0 = np.array([]) 
        for cday in days:
            # Check closest transit to given day:
            t = Time(str(int(year))+'-'+str(int(month))+'-'+str(int(cday))+' 00:00:00', \
                     format='iso', scale='utc')
            ntimes = int(np.ceil(1./pP))
            for n in range(ntimes):
                c_t0 = t.jd-pP*get_phases(t.jd,pP,pt0)+n*pP

                # Check if mid-transit, egress or ingress happens whithin the 
                # desired day. If it does, and we have not saved it, save the 
                # JD of the transit event:
                tyear,tmonth,tday,thh,tmm,tss = getCalDay(c_t0) 
                if tday == cday and tmonth == month and tyear == year:
                    if c_t0 not in transits_t0:
                        transits_t0 = np.append(transits_t0,c_t0)
                else:
                    tyear,tmonth,tday,thh,tmm,tss = getCalDay(c_t0+(pTdur/2.))
                    if tday == cday and tmonth == month and tyear == year:
                        if c_t0 not in transits_t0:
                            transits_t0 = np.append(transits_t0,c_t0)
                    else:
                        tyear,tmonth,tday,thh,tmm,tss = getCalDay(c_t0-(pTdur/2.))
                        if tday == cday and tmonth == month and tyear == year:
                            if c_t0 not in transits_t0:
                               transits_t0 = np.append(transits_t0,c_t0)
    
        # Now print the transits we found:
        counter = 0
        if len(transits_t0)>0:
            print 'Transits for '+planet+':'
            print '--------------------------\n'
        for ct0 in transits_t0:
            print '\t Transit number '+str(counter+1)+':'
            print '\t ----------------------------'
            tyear,tmonth,tday,thh,tmm,tss = getCalDay(ct0-(pTdur/2.)) 
            print '\t Ingress     : '+str(tyear)+'-'+str(tmonth)+'-'+str(tday)+' at '+str(thh)+\
                  ':'+str(tmm)+':'+str(tss)+' ('+str(ct0-(pTdur/2.))+' JD)'
            tyear,tmonth,tday,thh,tmm,tss = getCalDay(ct0)
            print '\t Mid-transit : '+str(tyear)+'-'+str(tmonth)+'-'+str(tday)+' at '+str(thh)+\
                  ':'+str(tmm)+':'+str(tss)+' ('+str(ct0)+' JD)'
            tyear,tmonth,tday,thh,tmm,tss = getCalDay(ct0+(pTdur/2.))
            print '\t Egress      : '+str(tyear)+'-'+str(tmonth)+'-'+str(tday)+' at '+str(thh)+\
                  ':'+str(tmm)+':'+str(tss)+' ('+str(ct0+(pTdur/2.))+' JD)'
            counter = counter + 1
    
