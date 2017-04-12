import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from spacepy import pycdf

# Read CDF file and create WFR dataframe
def read_cdf( filename, \
	pathname = '/Users/mystical/data/spacecraft/rbsp/rbsp-a/' + \
	'wfr_spectral_matrix_diagonal_L2/'):

	fullpath_filename = pathname + filename
	if not os.path.isfile( fullpath_filename ):
		print("File Not Found")
		return -1

	data = pycdf.CDF(fullpath_filename)

	UT = data['Epoch'][:]
	freq = data['WFR_frequencies'][0][:]
	freq_bw = data['WFR_bandwidth'][0][:]

	# The total magnetic power spectral density, BB, 
	# is the sum of the three components
	BB = data['BuBu'][:][:] + data['BvBv'][:][:] + data['BwBw'][:][:]
	# Transpose for plotting with pcolormesh
	BB = BB.transpose()

	freq_units = 'Hz'
	BB_units = 'nT$^2$/Hz'
	description = 'RBSP-A EMFISIS WFR'

	wfr = {'UT': UT, 'freq': freq, 'freq_bw': freq_bw, 'BB': BB, \
		'freq_units': freq_units, 'BB_units': BB_units, \
		'description': description}

	return wfr

# Create plot of BB as a function of freq and UT
def plot_BB( wfr ):

	fig, ax = plt.subplots()

	cax = ax.pcolormesh(wfr['UT'], wfr['freq'], np.log10( wfr['BB'] ) )
	cax.set_clim(-9, -5)

	hours = mdates.HourLocator(interval=4)
	hoursFmt = mdates.DateFormatter('%H')
	ax.xaxis.set_major_locator(hours)
	ax.xaxis.set_major_formatter(hoursFmt)
	ax.set_xlabel('UT hour on ' + wfr['UT'][0].strftime('%Y-%m-%d') )

	ax.set_yscale('log')
	ax.set_ylim(10**1.5, 11000)
	ax.set_ylabel('Frequency, ' + wfr['freq_units'] )

	ax.set_title(wfr['description'])

	fig.colorbar(cax, label='Magnetic Power Spectral Density, ' + \
		wfr['BB_units'])
	plt.show(block=False)


# exec(open('wfrpy.py').read())

filename = 'rbsp-a_WFR-spectral-matrix-diagonal_emfisis-L2_20160710_v1.6.4.cdf'
wfr = read_cdf(filename)
plot_BB(wfr)

