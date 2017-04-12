import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spacepy import pycdf

# Read CDF file and create nurd dataframe
def read_cdf(filename, \
	pathname='/Users/mystical/Work/Science/RBSP/DataCdf/rbsp-a/NURD/'):

	fullpath_filename = pathname + filename
	if not os.path.isfile( fullpath_filename ):
		print("file not found")
		return -1
		
	data = pycdf.CDF(fullpath_filename)

	nurd = pd.DataFrame( data['Epoch'][:], columns=['UT'])
	nurd['L'] = pd.Series( data['L'][:] )
	nurd['MLT'] = pd.Series( data['MLT'][:] )
	nurd['MLAT'] = pd.Series( data['magLat'][:] )
	nurd['R'] = pd.Series(np.sqrt(data['x_sm'][:]**2 + data['y_sm'][:]**2 + \
		data['z_sm'][:]**2) )
	nurd['ne'] = pd.Series( data['density'][:] )

	# Store metadata as attributes
	nurd.spacecraft = 'RBSP-a'

	return nurd

# Returns a boolean array of whether the density is inside 
# or outside the plasmapause defined as 100 cm^-3 
# at L=4 and scaled as L^-4
def find_psphere(ne_eq, L):

	ne_boundary = 100*(4/L)**4
	psphere = np.where( ne_eq >= ne_boundary, True, False )

	return psphere

# Estimates the equatorial density, ne_eq, based on L and R 
# using the Denton et al., 2002 formulation
# Returns an array, ne_eq
def denton_ne_eq( ne, L, R ):

	alpha_ne = 6.0 - 3.0*np.log10( ne ) +0.28*(np.log10( ne ))**2.0
	alpha_L = 2.0-0.43*L

	ne_eq = ne/np.power((L/R),(alpha_ne+alpha_L))

	# Denton forumulation for ne_eq only value over certain range 
	# of L, R and ne otherwise stick with ne
	ne_eq = np.where( np.logical_or(R < 2.0, L < 2.5), ne, ne_eq)
	ne_eq = np.where( np.logical_or(ne > 1500.0, ne_eq > ne ), ne, ne_eq)
	ne_eq = ne
	
	return ne_eq

# Plot Density as a function of L, color-coded by inside/outside plasmasphere
def plot_density_psphere( nurd ):

	fig = plt.figure()
	ax = plt.gca()
	ax.scatter( nurd[nurd.psphere].L, nurd[nurd.psphere].ne_eq, \
		color='b', s=2, label='Plasmasphere')
	ax.scatter( nurd[nurd.psphere==False].L, nurd[nurd.psphere==False].ne_eq, \
		color='r', s=2, label='Plasmatrough')
	ax.set_yscale('log')
	ax.set_xlabel('L');
	ax.set_ylabel('Equatorial Electron Density, cm^-3')
	ax.set_title(nurd.spacecraft + ' ' + \
		nurd.UT.iloc[0].strftime('%Y-%m-%d %H:%M:%S') + ' - ' + \
		nurd.UT.iloc[-1].strftime('%Y-%m-%d %H:%M:%S') + ' UT')
	ax.legend()
	ax.set_ylim( min(nurd.ne_eq), max(nurd.ne_eq) )

	plt.show(block=False)
	plt.ion()

	return

filename = 'rbsp-a_orbit_2624_v1_3.cdf';

nurd = read_cdf( filename )
nurd['ne_eq'] = denton_ne_eq(nurd['ne'], nurd['L'], nurd['R'] );
nurd['psphere'] = find_psphere( nurd['ne_eq'], nurd['L'] );
plot_density_psphere(nurd)


