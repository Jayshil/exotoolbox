# -*- coding: utf-8 -*-
import batman
import emcee
import sys
import numpy as np

def read_AIJ_tbl(fname):
    fin = open(fname,'r')
    firstime = True
    out_dict = {}
    while True:
        line = fin.readline()
        if line != '':
            vec = line.split()
            if firstime:
                out_dict['index'] = np.array([])
                for i in range(len(vec)):
                    out_dict[vec[i]] = np.array([])
                firstime = False
                parameter_vector = ['index'] + vec
            else:
                for i in range(len(vec)):
                    try:
                        out_dict[parameter_vector[i]] = np.append(out_dict[parameter_vector[i]],np.double(vec[i]))
                    except:
                        out_dict[parameter_vector[i]] = np.append(out_dict[parameter_vector[i]],np.nan)
        else:
            break
    return out_dict

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

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """
    get_quantiles function

    DESCRIPTION

        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.

    OUTPUTS

        Median of the parameter,upper credibility bound, lower credibility bound

    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

def RA_to_deg(coords):
    """
    Given a RA string in hours (e.g., '11:12:12.11'), returns the corresponding 
    coordinate in degrees.
    """
    hh,mm,ss = coords.split(':')

    hours = np.double(hh) + np.double(mm)/60. + np.double(ss)/3600.
    return hours * 360./24.

def DEC_to_deg(coords):
    """
    Given a DEC string in degrees (e.g., '-30:12:12.11'), returns the corresponding 
    coordinate in degrees.
    """
    dd,mm,ss = coords.split(':')
    if dd[0] == '-':
        return np.double(dd) - np.double(mm)/60. - np.double(ss)/3600.
    else:
        return np.double(dd) + np.double(mm)/60. + np.double(ss)/3600.


def get_distance(coords1,coords2):
    """
    Given two sets of coordinate strings, calculates the distance in arcminutes 
    between the objects. 

    INPUT

       coords1      Array, 2 dimensions, string. RA and DEC of first object; 
                    eg., np.array(['10:42:11','-30:11:13.13'])

       coords2      Array, 2 dimensions, string. RA and DEC of second object; 
                    eg., np.array(['10:42:11','-30:11:13.13'])
 
    OUTPUT

       Distance between two coordinates, in arcseconds.
    """

    RA1,DEC1 = RA_to_deg(coords1[0]),DEC_to_deg(coords1[1])
    RA2,DEC2 = RA_to_deg(coords2[0]),DEC_to_deg(coords2[1])

    return np.sqrt((RA1 - RA2)**2 + (DEC1-DEC2)**2)*3600.0

    
def mag_to_flux(m,merr):
    """
    Convert magnitude to relative fluxes. 
    """
    fluxes = np.zeros(len(m))
    fluxes_err = np.zeros(len(m))
    for i in range(len(m)):
        dist = 10**(-np.random.normal(m[i],merr[i],1000)/2.51)
        fluxes[i] = np.mean(dist)
        fluxes_err[i] = np.sqrt(np.var(dist))
    return fluxes,fluxes_err

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
    params.limb_dark = law
    if law != 'linear':
        params.u = [0.1,0.3]
    else:
        params.u = [0.1]
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ecc,omega,ld_law):
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
    params.ecc = ecc
    params.w = omega
    if ld_law != 'linear':
        params.u = [coeff1,coeff2]
    else:
        params.u = [coeff1]
    return m.light_curve(params)

def convert_bp(r1,r2,pl,pu):
    Ar = (pu - pl)/(2. + pl + pu)
    nsamples = len(r1)
    p = np.zeros(nsamples)
    b = np.zeros(nsamples)
    for i in range(nsamples):
        if r1[i] > Ar:
            b[i],p[i] = (1+pl)*(1. + (r1[i]-1.)/(1.-Ar)),\
                        (1-r2[i])*pl + r2[i]*pu
        else:
            b[i],p[i] = (1. + pl) + np.sqrt(r1[i]/Ar)*r2[i]*(pu-pl),\
                        pu + (pl-pu)*np.sqrt(r1[i]/Ar)*(1.-r2[i])
    return b,p

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

def emcee_transit(t,f,X,priors,ferr = None,fixed_parameters=None,ld_law='quadratic',noise_model = 'white',nburnin=500,nsteps=500,nwalkers=100):
    """
    This simple function fits a transit lightcurve + linear model to a given time (t) and 
    flux (f) pair using the emcee sampler. The components of the linear model are assumed 
    to be in the rows of the matrix X, and the priors for each of the parameters are expected to 
    go in the priors dictionary, containing the variable names (see below). The priors for the 
    components in the matrix X are supposed to go as 'x0', 'x1', etc. 

    Optionally, fixed_parameters is a dictionary that is supposed to contain the transit parameters 
    to be held fixed (e.g., fixed_parameters['t0'] = 2450000.0). Valid values for fixed_parameters are 
    'q1', 'q2', 'a/R*', 'p', 't0', 'ecc', 'omega', 'inc', 'sigma', 'x0', 'x1', etc.

    If noise_model = 'white', noise is assumed to be 'white gaussian', i.e., i.i.d. with parameter 'sigma_w', where 
    'sigma_w' is in ppm.

    Setting the priors
    ------------------

    The 'priors' dictionary is supposed to have three internal elements. A 'type', where you specify 
    the type of the prior, a 'parameters', where you specify the parameters of that prior and a 'initial_value',
    where you specify the starting point from which the MCMC will run for that parameter. Currently 
    supported types of priors are 'Gaussian', 'Uniform', 'Beta' and 'Jeffreys'. 

    To set a prior for parameter 't0' as 'Gaussian' with mean mu and standard deviation sigma, you would do:
                                >> priors['t0'] = {}
                                >> priors['type'] = 'Gaussian'
                                >> priors['parameters'] = [mu,sigma]
                                >> priors['initial_value'] = 0.1
    To set a uniform prior for the same parameter with lower limit a and upper limit b, you would do:
                                >> priors['t0'] = {}
                                >> priors['type'] = 'Uniform'
                                >> priors['parameters'] = [a,b]
                                >> priors['initial_value'] = (a+b)/2
    Similarly for a Jeffreys prior:
                                >> priors['t0'] = {}
                                >> priors['type'] = 'Jeffreys'
                                >> priors['parameters'] = [a,b]
                                >> priors['initial_value'] = (a+b)/2
    And for a Beta prior (always between 0 and 1):
                                >> priors['t0'] = {}
                                >> priors['type'] = 'Beta'
                                >> priors['parameters'] = [a,b]
                                >> priors['initial_value'] = (a+b)/2

    """
 
    # Initial setups:
    supported_priors = ['Gaussian','Uniform','Jeffreys','Beta']
    priors_that_need_checking = ['Uniform','Jeffreys','Beta']
    supported_priors_objects = [normal_parameter,uniform_parameter,jeffreys_parameter,beta_parameter]
    if ferr is None:
        ferr = np.zeros(len(t))
    # First, set parameters:
    if noise_model == 'white':
        all_parameters = ['sigma_w']
    parameters_to_check = []
    for i in range(X.shape[0]):
        all_parameters = all_parameters + ['x'+str(i)]
    if ld_law == 'linear':
        transit_parameters = ['p','P','t0','a','inc','ecc','omega','q1']
    else:
        transit_parameters = ['p','P','t0','a','inc','ecc','omega','q1','q2']
    all_parameters = all_parameters + transit_parameters
    parameters = {}
    all_mcmc_params = []
    for par in all_parameters:
        if par in priors.keys():
            good_prior = False
            for i in range(len(supported_priors)):
                if priors[par]['type'] == supported_priors[i]:
                    parameters[par] = supported_priors_objects[i](priors[par]['parameters'])
                    parameters[par].set_value(priors[par]['initial_value'])
                    if priors[par]['type'] in priors_that_need_checking:
                        parameters_to_check.append(par)
                    good_prior = True
                    all_mcmc_params.append(par)
                    break
            if not good_prior:
                print '\t ERROR: prior '+priors[par]['type']+' for parameter '+par+\
                      ' not supported. Supported priors are: ',supported_priors
                sys.exit()
                
        else:
            parameters[par] = constant_parameter(fixed_parameters[par])

    n_params = len(all_mcmc_params)
    n_data = len(t)
    log2pi = np.log(2.*np.pi)

    # Initialize ta-na-na-na-na-na-na-na-na-na...batman!
    params,m = init_batman(t,ld_law)

    def lnlike():
        # First, compute transit model:
        if ld_law != 'linear':
            coeff1,coeff2 = reverse_ld_coeffs(ld_law, parameters['q1'].value,parameters['q2'].value)
            params.u = [coeff1,coeff2]
        else: 
            params.u = [parameters['q1'].value]
        params.t0 = parameters['t0'].value
        params.per = parameters['P'].value
        params.rp = parameters['p'].value
        params.a = parameters['a'].value
        params.inc = parameters['inc'].value
        params.ecc = parameters['ecc'].value
        params.w = parameters['omega'].value
        transit_model = m.light_curve(params)
        # Compute linear model:
        linear_model = 0.
        for i in range(X.shape[0]):
            linear_model = linear_model + X[i,:]*parameters['x'+str(i)].value
        # Generate full model and residuals:
        residuals = (np.log10(f) - linear_model - np.log10(transit_model))
        if noise_model == 'white':
            taus = 1.0/(parameters['sigma_w'].value*1e-6/(f*np.log(10.)))**2
            log_like = -0.5*(n_data*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
        else:
            print 'Noise model '+noise_model+' not supported.'
            sys.exit()
        return log_like

    def lnprior(theta):
        # Read in the values of the parameter vector and update values of the objects.
        # For each one, if everything is ok, get the total prior, which is the sum 
        # of the independant priors for each parameter:
        total_prior = 0.0
        for i in range(n_params):
            c_param = all_mcmc_params[i]
            parameters[c_param].set_value(theta[i])
            if c_param in parameters_to_check:
                if not parameters[c_param].check_value(theta[i]):
                    return -np.inf
            total_prior += parameters[c_param].get_ln_prior()
        return total_prior

    def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
                return -np.inf
        return lp + lnlike()

    # Start the MCMC around the points given by the user:
    pos = []

    for j in range(nwalkers):
        while True:
            good_values = True
            theta_vector = np.array([])
            for i in range(n_params):
                par = all_mcmc_params[i]
                # Put the walkers around a small gaussian sphere centered on the input user value. 
                # Walkers will run away from sphere eventually:
                theta_vector = np.append(theta_vector,parameters[par].value + \
                                         (parameters[par].value-parameters[par].sample())*1e-3)
                if par in parameters_to_check:
                    if not parameters[par].check_value(theta_vector[-1]):
                        good_values = False
                        break
            if good_values == True:
                break
        pos.append(theta_vector) 
    print '\t >> Starting MCMC...'
    sampler = emcee.EnsembleSampler(nwalkers, n_params, lnprob)
    sampler.run_mcmc(pos, nsteps+nburnin)
    print '\t >> Done! Saving...'
    for i in range(n_params):
        c_param = all_mcmc_params[i]
        c_p_chain = np.array([])
        for walker in range(nwalkers):
            c_p_chain = np.append(c_p_chain,sampler.chain[walker,nburnin:,i])
        parameters[c_param].set_posterior(np.copy(c_p_chain))
    return parameters

# Here I define some useful classes for the emcee transit MCMC procedures:
class normal_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a normal prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """   
      def __init__(self,prior_hypp):
          self.value = prior_hypp[0]
          self.init_value = prior_hypp[0]
          self.value_u = 0.0
          self.value_l = 0.0
          self.has_guess = False
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return np.log(1./np.sqrt(2.*np.pi*(self.prior_hypp[1]**2)))-\
                 0.5*(((self.prior_hypp[0]-self.value)**2/(self.prior_hypp[1]**2)))

      def set_value(self,new_val):
          self.value = new_val

      def set_init_value(self,new_val):
          self.init_value = new_val    
          self.has_guess = True

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l
      def sample(self):
          return np.random.normal(self.prior_hypp[0],self.prior_hypp[1])

class uniform_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a uniform prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,prior_hypp):
          self.value = (prior_hypp[0]+prior_hypp[1])/2.
          self.init_value = (prior_hypp[0]+prior_hypp[1])/2.
          self.value_u = 0.0
          self.value_l = 0.0
          self.has_guess = False
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return np.log(1./(self.prior_hypp[1]-self.prior_hypp[0]))

      def check_value(self,x):
          if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
              return True
          else:
              return False  
 
      def set_value(self,new_val):
          self.value = new_val

      def set_init_value(self,new_val):
          self.init_value = new_val
          self.has_guess = True

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l
      def sample(self):
          return np.random.uniform(self.prior_hypp[0],self.prior_hypp[1])

log1 = np.log(1)
class jeffreys_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a Jeffreys prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,prior_hypp):
          self.value = np.sqrt(prior_hypp[0]*prior_hypp[1])
          self.init_value = np.sqrt(prior_hypp[0]*prior_hypp[1])
          self.value_u = 0.0
          self.value_l = 0.0
          self.has_guess = False
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return log1 - np.log(self.value*np.log(self.prior_hypp[1]/self.prior_hypp[0]))

      def check_value(self,x):
          if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
              return True
          else:
              return False

      def set_value(self,new_val):
          self.value = new_val

      def set_init_value(self,new_val):
          self.init_value = new_val
          self.has_guess = True

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l
      def sample(self):
          return np.exp(np.random.uniform(np.log(self.prior_hypp[0]),np.log(self.prior_hypp[1])))

from scipy.special import gamma
class beta_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a Beta prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,prior_hypp):
          self.value = 0.5
          self.init_value = 0.5
          self.value_u = 0.0
          self.value_l = 0.0
          self.has_guess = False
          self.prior_hypp = prior_hypp
          self.gamma_alpha = gamma(prior_hypp[0])
          self.gamma_beta = gamma(prior_hypp[1])
          self.gamma_sum = gamma(prior_hypp[0]+prior_hypp[1])
          self.posterior = []

      def get_ln_prior(self):
          return np.log(self.gamma_sum) + (self.prior_hypp[0]-1.)*np.log(self.value) + \
                 (self.prior_hypp[1]-1.)*np.log(1.-self.value) - np.log(self.gamma_alpha) - \
                 np.log(self.gamma_beta)

      def check_value(self,x):
          if x > 0. and  x < 1.:
              return True
          else:
              return False

      def set_value(self,new_val):
          self.value = new_val

      def set_init_value(self,new_val):
          self.init_value = new_val
          self.has_guess = True

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l
      def sample(self):
          return np.random.beta(self.prior_hypp[0],self.prior_hypp[1])

class constant_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a constant value. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,val):
          self.value = val

def read_epdlc():
    """
    This function reads in lightcurves in the epdlc format. It assumes 
    the directory contains the targetlist.txt file and the epdlc lightcurves of 
    the corresponding stars in the target list.
    """
    lc_names = []
    fobj = open('targetlist.txt','r')
    lc_name,ra,dec,mag,mag_err = fobj.readline().split()
    t,m,merr,x,y = np.loadtxt(lc_name,unpack=True,usecols=(1,2,3,17,18))
    counter = 0
    while True:
        line = fobj.readline()
        if line == '':
            break
        elif line[0]!='#':
            lc_name,ra,dec,mag,mag_err = line.split()
            tc,mc,merrc = np.loadtxt(lc_name,unpack=True,usecols=(1,2,3))
            lc_names.append(lc_name)
            if counter == 0:
                X = mc
                counter = counter + 1
            else:
                X = np.vstack((X,mc))
    out = {}
    out['times'] = t 
    out['names'] = lc_names
    out['lcs'] = X
    out['target_lc'] = m
    out['target_lc_err'] = merr
    out['target_x'] = x
    out['target_y'] = y
    return out

from sklearn import linear_model
def fit_linear_model(X,y,idx=None):
    """
    Given a matrix X containing predictors on its rows, a target "y" and optionally a 
    list of indexes to use for fitting ("idx"), fits the predictors assuming a linear model
    to the target. This returns the coefficients of the fit and the prediction using all 
    the predictors.
    """

    regr = linear_model.LinearRegression(fit_intercept=False)
    if idx is None:
        idx = np.arange(X.shape[1])
    regr.fit(X[:,idx].transpose(),y[idx])
    coeffs = regr.coef_
    model = regr.predict(X.transpose())
    return coeffs,model

import lmfit
def fit_transit_model(t,f,pdata,ld_law,white_light=True):
    """
    This function fits a quick transit model given time, (normalized) flux and pdata, which is a dictionary 
    that contains prior transit parameters
    """
    def residuals(params, x, y):
        # Generate lightcurve:
        if white_light:
            transit_model = get_transit_model(x-pdata['TT'],params['t0'].value,pdata['PER'],params['p'].value,\
                            params['a'].value,params['i'].value,params['q1'].value,params['q2'].value,\
                            pdata['ECC'],pdata['OM'],ld_law=ld_law)
        else:
            transit_model = get_transit_model(x,pdata['TT'],pdata['PER'],params['p'].value,\
                            pdata['AR'],pdata['I'],params['q1'].value,params['q2'].value,\
                            pdata['ECC'],pdata['OM'],ld_law=ld_law)
        # Get residuals:
        return ((y - transit_model)*1e6)**2    

    # Initialize parameters of lmfit:
    prms = lmfit.Parameters()
    prms.add('p', value = np.sqrt(pdata['DEPTH']), min = np.sqrt(np.min([0,pdata['DEPTH']-\
                          10.*pdata['DEPTHLOWER']])),max = np.sqrt(pdata['DEPTH']+10.*pdata['DEPTHUPPER']), \
                          vary = True)
    prms.add('q1', value = 0.5, min = 0, max = 1, vary = True)
    if ld_law != 'linear':
       prms.add('q2', value = 0.5, min = 0, max = 1, vary = True)
    else:
       prms.add('q2', value = 0.0, vary = False)
    if white_light:
        prms.add('i', value = pdata['I'], min = np.min([0,pdata['I']-10.*pdata['ILOWER']]), \
                                          max = np.min([90.,pdata['I']+10.*pdata['IUPPER']]), vary = True)
        prms.add('a', value = pdata['AR'], min = np.min([0,pdata['AR']-10.*pdata['ARLOWER']]),\
                                           max = pdata['AR']+10.*pdata['ARUPPER'], vary = True)
        prms.add('t0', value=0.0,min=-(5./24.),max=(5./24.))
    result = lmfit.minimize(residuals, prms, args=(t,f))
    if white_light:
        if ld_law != 'linear':
            out_params = ['p','t0','q1','q2','i','a']
        else:
            out_params = ['p','t0','q1','i','a']
        transit_model = get_transit_model(t,result.params['t0'].value+pdata['TT'],pdata['PER'],result.params['p'].value,\
                            result.params['a'].value,result.params['i'].value,result.params['q1'].value,result.params['q2'].value,\
                            pdata['ECC'],pdata['OM'],ld_law=ld_law)
    else:
        if ld_law != 'linear':
            out_params = ['p','q1','q2']
        else:
            out_params = ['p','q1']
        transit_model = get_transit_model(t,pdata['TT'],pdata['PER'],result.params['p'].value,\
                            pdata['AR'],pdata['I'],result.params['q1'].value,result.params['q2'].value,\
                            pdata['ECC'],pdata['OM'],ld_law=ld_law)

    out = {}
    for par in out_params:
        out[par] = result.params[par].value
    return out,transit_model

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
        if P is None or Tdur is None or t0 is None:
            planet_list = exodata.keys()
        else:
            planet_list = names
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

# Define class that stores constants:
class constants:
    def __init__(self):
        # Boltzmann constant:
        self.kB_val = 1.38064852e-23 # J/K
        self.kB_val_cgs = 1.38064852e-16 # erg/K
        # Gravitational constant:
        self.G_val = 6.67408e-11 # m^3/(kg x s^2)
        self.G_val_cgs = 6.67408e-8 # cm^3/(g x s^2)
        # Atomic mass unit:
        self.amu_val = 1.660539040e-27 # kg
        self.amu_val_cgs = 1.660539040e-24 # kg
        # Astronomical unit:
        self.AU_val = 149597870700. # m
        self.AU_val_cgs = 149597870700.*1e2 # cm
        # Solar radius:
        self.Rsun_val = 6.957e8 # m
        self.Rsun_val_cgs = 6.957e8*1e2 # cm
        # Solar mass:
        self.Msun_val = 1.98855e30 # kg
        self.Msun_val_cgs = 1.98855e30*1e3 # g
        # Jupiter radius:
        self.Rj_val = 7.1492e7 # m
        self.Rj_val_cgs = 7.1492e7*1e2 # cm
        # Jupiter mass:
        self.Mj_val = 1.89813e27 # kg
        self.Mj_val_cgs = 1.89813e30 # g
    def kB(self):
        return self.kB_val
    def kB_cgs(self):
        return self.kB_val_cgs
    def G(self):
        return self.G_val
    def G_cgs(self):
        return self.G_val_cgs
    def amu(self):
        return self.amu_val
    def amu_cgs(self):
        return self.amu_val_cgs
    def AU(self):
        return self.AU_val
    def AU_cgs(self):
        return self.AU_val_cgs
    def Rj(self):
        return self.Rj_val
    def Rj_cgs(self):
        return self.Rj_val_cgs
    def Rsun(self):
        return self.Rsun_val
    def Rsun_cgs(self):
        return self.Rsun_val_cgs
    def Msun(self):
        return self.Msun_val
    def Msun_cgs(self):
        return self.Msun_val_cgs
    def Mj(self):
        return self.Mj_val
    def Mj_cgs(self):
        return self.Mj_val_cgs
