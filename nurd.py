#import cdf
#import pandas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spacepy import pycdf

# Read CDF file and create NURD dataframe
def read_cdf(fullpath_filename):
	if fullpath_filename[0] != '/':
		fullpath_filename = '/Users/mystical/Work/Science/RBSP/' + \
			'DataCdf/rbsp-a/NURD/' + filename
	if not os.path.isfile( fullpath_filename ):
		return -1
		
	data = pycdf.CDF(fullpath_filename)
	nurd_df = pd.DataFrame( data['Epoch'][:], columns=['UT'])
	nurd_df['L'] = pd.Series( data['L'][:] )
	nurd_df['mlt'] = pd.Series( data['MLT'][:] )
	nurd_df['mlat'] = pd.Series( data['magLat'][:] )
	nurd_df['R'] = pd.Series(np.sqrt(data['x_sm'][:]**2 + data['y_sm'][:]**2 + \
		data['z_sm'][:]**2) )
	nurd_df['ne'] = pd.Series( data['density'][:] )

	return nurd_df

# Returns a boolean array of whether the density is inside 
# or outside the plasmapause defined as 100 cm^-3 
# at L=4 and scaled as L^-4
def find_psphere(ne_eq, L):

	ne_boundary = 100*(4/L)**4
	psphere = np.where( ne_eq >= ne_boundary, True, False )

	return psphere


# Estimates the equatorial density, ne_eq, based on L and R 
# using the Denton et al., 2002 formulation
def denton_ne_eq( ne, L, R ):

	alpha_ne = 6.0 - 3.0*np.log10( ne ) +0.28*(np.log10( ne ))**2.0
	alpha_L = 2.0-0.43*L

	ne_eq = ne/np.power((L/R),(alpha_ne+alpha_L))

	ne_eq = np.where( np.logical_or(R < 2.0, L < 2.5), ne, ne_eq)
	ne_eq = np.where( np.logical_or(ne > 1500.0, ne_eq > ne ), ne, ne_eq)
	
	return ne_eq

# Plot Density as a function of L, color-coded by inside/outside plasmasphere
def plot_density_psphere( nurd_df )

	nurd_df[nurd_df.psphere].plot(x='L', y='ne', logy=True, \
		linestyle='None', marker='o', color='r')
	plt.hold=True
	nurd_df[nurd_df.psphere==False].plot(x='L', y='ne', logy=True, \
		linestyle='None', marker='o', color='b')

	plt.show(block=False)

	return

filename = 'rbsp-a_orbit_2620_v1_3.cdf';

#one_nurd_orbit = nurd_read_mat( fullpath + filename )
nurd_df = nurd_read_cdf( filename )
nurd_df['ne_eq'] = denton_ne_eq(nurd_df['ne'], nurd_df['L'], nurd_df['R'] );
nurd_df['psphere'] = nurd_find_psphere( nurd_df['ne_eq'], nurd_df['L'] );

