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
masses.sort()
print('Lowest mass: %.3g' %np.min(masses))
print('Maximun mass: %.3g' %np.max(masses))

#create mass bins
nbins=100
mass_bins=np.logspace(12.0,14.1,nbins)
#create vectors that will contain the number of halos at a given mass
mass_number = np.zeros(len(mass_bins))
mass_average = np.zeros(len(mass_bins))

#fill these vectors counting how many halos belong to different bins
'''
count=0
counter=0
for i in range(len(mass_bins)):
    for j in range(count,len(masses)):
        if(i==0):
            if(masses[j]<=mass_bins[i]):
                mass_number[i]=mass_number[i]+1
                mass_average[i]=mass_average[i]+masses[j]
                counter = counter+1
        else:   
            if(masses[j] <= mass_bins[i] and masses[j] > mass_bins[i-1]):
                mass_number[i]=mass_number[i]+1
                mass_average[i]=mass_average[i]+masses[j]
                counter = counter+1
        count = count+counter
        counter = 0
    print('cicle ',i,' of ', len(mass_bins))
'''

j=0 
for i in range(len(masses)):
    if(masses[i]<=mass_bins[j]):
        mass_number[j]=mass_number[j]+1
    else:
        j=j+1
        if(j==nbins):
            break
        i=i-1
        print('cicle ',j,' of ', len(mass_bins))
    
print('number of halos in each bin: ',mass_number)   
print('total number of halos used: ',np.sum(mass_number))
print('total number of halos in the simulation: ', len(masses))
             
#compute Poisson error in each bin
#the simulation is a cube of 1 Gpc/h...
#divide the number by 10^9 to get the number per (Mpc/h)^3
counts_error = np.sqrt(mass_number)/10**9
mass_number = mass_number/(10**9)

   
#now I can build the mass function
plt.figure()
plt.errorbar(mass_bins,mass_number,yerr=counts_error, fmt='.', label='data')
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
#cosmology.setCosmology('planck18')
cosmology.addCosmology('myCosmo', params) #params was defined at line 24
cosmology.setCosmology('myCosmo')
mass_func_model=mf.massFunction(mass_bins,z=z,mdef = 'vir', model = 'comparat17', q_out = 'dndlnM')
plt.plot(mass_bins, mass_func_model, label='model')
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



