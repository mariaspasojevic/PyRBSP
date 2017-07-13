"""
wfrpy.py

This module contains functionality for handling WFR data from the
RBSP EMFISIS intrument

http://emfisis.physics.uiowa.edu/
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from spacepy import pycdf

def read_cdf( filename, \
	pathname = '/Users/mystical/Work/Science/RBSP/DataCdf/rbsp-a/' + \
	'wfr_spectral_matrix_diagonal_L2/'):
	"""	
	read_cdf( filename, pathname = [default location])

	Reads WFR diagonal elements of the cross spectral matrix CDF as downloaded
	from: http://emfisis.physics.uiowa.edu/data/index

	Input: filename, pathname strings
	Output: wfr dictionary with keys UT [datetime], timestamp, freq, freq_bw,
					BB=BxBx + ByBy + BzBz, freq_units, BB_units, description
	"""

	fullpath_filename = pathname + filename
	if not os.path.isfile( fullpath_filename ):
		print("File Not Found")
		return -1

	data = pycdf.CDF(fullpath_filename)

	UT = data['Epoch'][:]
	timestamp = pd.Series(UT).apply( lambda x: time.mktime(x.timetuple())  )
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

	wfr = {'UT': UT, 'timestamp': timestamp, 'freq': freq, 'freq_bw': freq_bw, 'BB': BB, \
		'freq_units': freq_units, 'BB_units': BB_units, \
		'description': description}

	data.close()

	return wfr

def plot_BB( wfr ):
	""" plot_BB( wfr ) Creates a plot of BB as function of freq and UT """

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

def load_day( datestr ):
	"""
	load_day( datestr ):

	Give a datestr as 'YYYYMMDD' finds the corresponding WFR file
	loads it and returns the wfr dictionary using read_cdf() otherwise
	returns -1 if no file is available
	"""

	pathname = '/Users/mystical/Work/Science/RBSP/DataCdf/rbsp-a/' + \
	'wfr_spectral_matrix_diagonal_L2/'
	filename_start = 'rbsp-a_WFR-spectral-matrix-diagonal_emfisis-L2_'
	filename_end = '.cdf'

	filename_wild = filename_start + datestr + '*' + filename_end
	filename = glob.glob( pathname + filename_wild )
	if not filename:
		print('WFR file not found: ' + pathname + filename_wild)
		return -1
	else:
		# Remove pathname and read cdf file
		filename = filename[0].replace(pathname, '')
		wfr = read_cdf( filename )

	return wfr

	


# exec(open('wfrpy.py').read())

#filename = 'rbsp-a_WFR-spectral-matrix-diagonal_emfisis-L2_20160710_v1.6.4.cdf'
#datestr = '20160710'
#wfr = load_day(datestr)
#plot_BB(wfr)

