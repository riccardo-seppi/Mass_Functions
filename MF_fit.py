#Riccardo Seppi - MPE - HEG (2019) - 25 October
#This code reads halo masses from DM simulations (GLAM)
#builds HMF and fits them to models with fixed cosmological parameters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dynesty
from dynesty import plotting as dyplot

catalog='/data26s/comparat/simulations/GLAM/1Gpc2000x4000/CATALOGS/CatshortV.0092.0001.DAT'

#read here the cosmological parameters
a = np.loadtxt(catalog, usecols=[2], skiprows=1, max_rows=1, dtype=float)
z = 1/a - 1
print('z = ',z)
Omega0 = np.loadtxt(catalog, usecols=[1], skiprows=3, max_rows=1, dtype=float)
print('Omega0 = ', Omega0)
Omega0DE = np.loadtxt(catalog, usecols=[3], skiprows=3, max_rows=1, dtype=float)
print('Omega0DE = ', Omega0DE)
hubble = np.loadtxt(catalog, usecols=[6], skiprows=3, max_rows=1, dtype=float)
print('h = ', hubble)
#params = {'flat': False, 'H0': hubble*100, 'Om0': Omega0, 'Ode0': Omega0DE}
params = {'flat': False, 'H0': hubble*100, 'Om0': Omega0, 'Ode0': Omega0DE, 'Ob0': 0.049, 'sigma8': 0.828, 'ns': 0.96}

#read masses of the halos identified by Rockstar
print('reading masses from the catalog...')
#masses = pd.read_table(catalog, usecols=[7], skiprows=7, sep=' ', dtype=float)
masses = np.loadtxt(catalog, usecols=[7], skiprows=8, dtype=float)
masses = masses/hubble #get reale masses instead of Msol/h
masses.sort()
print('Lowest mass: %.3g' %np.min(masses))
print('Maximun mass: %.3g' %np.max(masses))

#create mass bins and count halos in different bins
nbins=100
mass_number, mass_bins = np.histogram(masses,bins=np.logspace(12.2,14.1,nbins))
plt.hist(masses,mass_bins)
plt.loglog()
plt.show()
    
diff = np.diff(np.log(mass_bins))
print(diff)
diff=diff[0]


print('mass_number length = ', len(mass_number), 'mass_bins length = ', len(mass_bins))
print('number of halos in each bin: ',mass_number)   
print('total number of halos used: ',np.sum(mass_number))
print('total number of halos in the simulation: ', len(masses))
             
#compute Poisson error in each bin
#the simulation is a cube of 1 Gpc/h...
#divide the number by 10^9 to get the number per (Mpc/h)^3
counts_error = np.sqrt(mass_number)/10**9
mass_number = mass_number/(10**9)/diff

   
#now I can build the mass function
#plot each data in the middle of its respective mass bin
mass_bins_pl = np.zeros(len(mass_number))
for i in range(len(mass_number)):
	mass_bins_pl[i] = (mass_bins[i+1] + mass_bins[i])/2

plt.figure()
plt.errorbar(mass_bins_pl,mass_number,yerr=counts_error, fmt='.', label='data')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\odot}$', fontsize=18)
plt.ylabel(r'dn/dM $[(Mpc/h)^{-3}]$',fontsize=18)
plt.grid(True)
plt.tight_layout()

#Now I want to fit it
#consider the model (comparat17, tinker08...)
#NB: z will have to be the same of the simulation analyzed!!!
from colossus.lss import mass_function as mf
from colossus.cosmology import cosmology
from colossus.lss import peaks
#cosmology.setCosmology('planck18')
cosmology.addCosmology('myCosmo', params) #params was defined at line 24
cosmo=cosmology.setCosmology('myCosmo')
#print(cosmo.rho_m(0.0))

#fitting with Bhattacharya 2011
def mass_function_rseppi(Mass):
	cosmo=cosmology.getCurrent()	
	delta_c = peaks.collapseOverdensity(z=z)
	R = peaks.lagrangianR(Mass)
	sigma = cosmo.sigma(R=R,z=z)
	nu = delta_c / sigma
	nu2 = nu**2
	zp1 = 1.0+z
	A = 0.333 * zp1**-0.11
	a = 0.788 * zp1**-0.01
	p = 0.807
	q = 1.795
	f = A * np.sqrt(2 / np.pi) * np.exp(-a * nu2 * 0.5) * (1.0 + (nu2*a)**-p) * (nu * np.sqrt(a))**q

	d_ln_sigma_d_ln_R = cosmo.sigma(R, z, derivative = True)
	rho_Mpc = cosmo.rho_m(0.0) * 1E9
	mass_func_model = -1/3*f*rho_Mpc/Mass*d_ln_sigma_d_ln_R
	return mass_func_model


mass_func_model = mass_function_rseppi(mass_bins_pl)
print('model values = ', mass_func_model)
#mass_func_model=mf.massFunction(mass_bins_pl,z=z,mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')
plt.plot(mass_bins_pl, mass_func_model, label='model')
plt.show()

#define residual function
import lmfit
parameters = lmfit.Parameters()
parameters.add_many(('A', 0.5,True,0.01,1), ('a', 0.5,True,0.01,1), ('p', 0.5,True,0.01,1), ('q', 0.5,True,0.01,1))

def residual(parameters):
    v = parameters.valuesdict() 	
    res = (mass_number - mass_func_model(v['A'],v['a'],v['p'],v['p']))/counts_error
    return res	

# Define the parameters dimensionality: 4 for comparat17
ndim = 4

def loglike(parameters):
	resid = residual(parameters)
	resid *= resid
	resid += np.log(2*np.pi*counts_error**2)
	lnp = -0.5*np.sum(resid)
	if(np.isnan(lnp)):
		lnp=-1e10
	return lnp

# Define our uniform prior via the prior transform.
def ptform(u):
    return 1.0*u

#dynamic nested sampling
dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim)
dsampler.run_nested()
dresults = dsampler.results

#plot the results
# initialize figure
fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))  # reshape axes

# add white space
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]

# plot initial run (res1; left)
fg, ax = dyplot.cornerplot(dresults, color='blue', truths=np.zeros(ndim),
                           truth_color='black', show_titles=True,
                           max_n_ticks=3, quantiles=None,
                           fig=(fig, axes[:, :4]))



