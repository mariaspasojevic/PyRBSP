import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from spacepy import pycdf

# Read CDF file and create MAG dataframe
def read_cdf( filename, \
	pathname = '/Users/mystical/Work/Science/RBSP/DataCdf/rbsp-a/' + \
	'mag_4sec_sm_L3/'):

	fullpath_filename = pathname + filename
	if not os.path.isfile( fullpath_filename ):
		print("**** File Not Found: " + fullpath_filename)
		return -1

	data = pycdf.CDF(fullpath_filename)

	mag = pd.DataFrame( data['Epoch'][:], columns=['UT'])
	mag['timestamp'] = mag['UT'].apply( lambda x: time.mktime(x.timetuple()) )
	mag['Bt'] = data['Magnitude'][:] 

	# EPHEMERIS Data convert from kms to Earth Radii
	Re = 6371.0
	# Some files have error where coordinates are not the right size
	if data['coordinates'].shape[1] != 3:
		raise Exception( fullpath_filename + ": HAS INVALID COORDINATES.")

	mag['x_sm'] = data['coordinates'][:][:,0]/Re
	mag['y_sm'] = data['coordinates'][:][:,1]/Re
	mag['z_sm'] = data['coordinates'][:][:,2]/Re

	# Compute L (Re), mlat (in degrees) and mlt (0-24)
	r_xy = np.sqrt( mag.x_sm**2 + mag.y_sm**2 );
	r = np.sqrt( mag.x_sm**2 + mag.y_sm**2 + mag.z_sm**2 );
	mlat = np.arctan2( mag.z_sm, r_xy )

	mag['L'] = r/(np.cos(mlat)**2)
	mag['mlat'] = mlat*180/np.pi
	mag['mlt'] = np.arctan2(mag.y_sm, mag.x_sm)*12/np.pi + 12

	# Storing metadata as attributes doesn't really work
  # Many pandas functions will return and dataframe without these values
	mag.Bt_units = 'nT'
	mag.description = 'RBSP-A EMFISIS MAG'

	data.close()

	return mag

# Clean 2 types of corrupted data. Bt < 0 and single point positive spikes.
def clean( mag ):
	
	# Remove rows where Bt < 0
	# Same as mag = mag[mag.Bt>0]
	mag.query( 'Bt > 0', inplace=True)

	# Remove positive spikes greater than 500 nT
	deltaB = np.insert(np.diff(mag.Bt), 0, 0)
	mag = mag[deltaB<500]
	
	return mag

# Calculate the local (fce) and 
# equatorial (fce_eq) electron cyclotron frequencies 
def calc_fce( mag ):

	q = 1.602e-19;
	me = 9.109e-31;

	# Local fce
	mag['fce'] = q*mag.Bt*1e-9/(me*2*np.pi);

	# Scale the fce to the equatorial plane assuming dipolar variation
	# See Walt's book
	mlat_rads = mag.mlat*np.pi/180
	mag['fce_eq'] = mag.fce*( np.cos( mlat_rads )**6 ) \
		/ np.sqrt( 1 + 3*( np.sin( mlat_rads )**2 ) );

	return mag
	
# Create plot of BB as a function of freq and UT
def plot_mag( mag, fignum=1 ):

	plt.ion()
	fig = plt.figure(fignum)
	fig.clf()
	ax = plt.subplot()

	ax.plot(mag['UT'], mag['Bt'])

	hours = mdates.HourLocator(interval=4)
	hoursFmt = mdates.DateFormatter('%H')
	ax.xaxis.set_major_locator(hours)
	ax.xaxis.set_major_formatter(hoursFmt)
	ax.set_xlabel('UT hour on ' + mag['UT'][0].strftime('%Y-%m-%d') )

	#ax.set_yscale('log')
	ax.set_ylabel('Bt, nT')

	ax.set_title('RBSP-a EMFISIS MAG')

# Create plot of ephemeris data
def plot_orbit( mag, fignum=2 ):
	plt.ion()
	fig = plt.figure(fignum)
	fig.clf()

	ax = plt.subplot(4,1,1)
	ax.plot( mag.UT, mag.x_sm, 'r' )
	ax.plot( mag.UT, mag.y_sm, 'g' )
	ax.plot( mag.UT, mag.z_sm, 'b' )
	ax.set_ylabel('X,Y,Z');

	ax = plt.subplot(4,1,2)
	ax.plot( mag.UT, mag.L, 'k' )
	ax.set_ylabel('L')

	ax = plt.subplot(4,1,3)
	ax.plot( mag.UT, mag.mlat, 'k' )
	ax.set_ylabel('$\lambda$')

	ax = plt.subplot(4,1,4)
	ax.plot( mag.UT, mag.mlt, 'k' )
	ax.set_ylabel('MLT')

	ax_list = fig.axes
	hours = mdates.HourLocator(interval=4)
	hoursFmt = mdates.DateFormatter('%H')
	for ii, ax in enumerate(ax_list):
		ax.xaxis.set_major_locator(hours)
		ax.xaxis.set_major_formatter(hoursFmt)
		if ii < 3:
			plt.setp(ax.get_xticklabels(),visible=False)
		else:
			ax.set_xlabel('UT hour on ' + mag['UT'][0].strftime('%Y-%m-%d') )

	ax_list[0].set_title('RBSP-a EMFISIS MAG')

# Given a datestring in the form 'YYYYMMDD', read the mag cdf file,
# clean, calculate fce, return mag dataframe
def load_day( datestr ):
	pathname = '/Users/mystical/Work/Science/RBSP/DataCdf/rbsp-a/mag_4sec_sm_L3/'
	filename_start = 'rbsp-a_magnetometer_4sec-sm_emfisis-L3_'
	filename_end = '.cdf'

	filename_wild = filename_start + datestr + '*' + filename_end
	filename = glob.glob( pathname + filename_wild )
	if not filename:
		print('MAG file not found: ' + pathname + filename_wild)
		return -1
	else:
		#	Remove pathname and read cdf file
		filename = filename[0].replace(pathname, '')
		mag = read_cdf( filename )
		mag = clean( mag )
		mag = calc_fce( mag )

	return mag

# exec(open('magpy.py').read())

# THIS IS SO BIZARRE. pycdf.CDF CRAPS OUT IN THIS LOOP
#
#pathname = '/Users/mystical/data/spacecraft/rbsp/rbsp-a/mag_4sec_sm_L3/'
#pathname = '/Users/mystical/Work/Science/RBSP/DataCdf/rbsp-a/' + \
#	'mag_4sec_sm_L3/'
#for item in sorted(os.listdir( pathname )):
#	if item.startswith('rbsp-a') and item.endswith('.cdf'):
#		print(item)
#		type(item)
#		mag = calc_fce(clean(read_cdf(item, pathname)))
#		plot_mag(mag)
#		typed = input()
#		if( typed == 'q' ):
#			break


# This file has a common type of error. Negative values and single spike values
#filename = 'rbsp-a_magnetometer_4sec-sm_emfisis-L3_20121011_v1.3.2.cdf'
#mag = calc_fce(read_cdf(filename))
#plot_mag(mag,1)
#mag = clean(mag)
#plot_mag(mag,2)
#mag = load_day('20140613')
#plot_mag(mag,1)
#plot_orbit(mag,1)

