import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from spacepy import pycdf

# Read CDF file and create MAG dataframe
def read_cdf( filename, \
	pathname = '/Users/mystical/data/spacecraft/rbsp/rbsp-a/' + \
	'mag_4sec_sm_L3/'):

	fullpath_filename = pathname + filename
	print(fullpath_filename)
	type(fullpath_filename)
	if not os.path.isfile( fullpath_filename ):
		print("File Not Found")
		return -1

	data = pycdf.CDF(fullpath_filename)

	mag = pd.DataFrame( data['Epoch'][:], columns=['UT'])
	mag['Bt'] = data['Magnitude'][:] 

	# EPHEMERIS Data convert from kms to Earth Radii
	Re = 6371.0
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

	mag.Bt_units = 'nT'
	mag.description = 'RBSP-A EMFISIS MAG'

	data.close()

	return mag

# Create plot of BB as a function of freq and UT
def plot_mag( mag ):

	plt.ion()
	fig = plt.figure(1)
	fig.clf()
	ax = plt.subplot()

	ax.plot(mag['UT'], mag['Bt'])

	hours = mdates.HourLocator(interval=4)
	hoursFmt = mdates.DateFormatter('%H')
	ax.xaxis.set_major_locator(hours)
	ax.xaxis.set_major_formatter(hoursFmt)
	ax.set_xlabel('UT hour on ' + mag['UT'][0].strftime('%Y-%m-%d') )

	ax.set_yscale('log')
	ax.set_ylabel('Bt, ' + mag.Bt_units )

	ax.set_title(mag.description)

# Create plot of ephemeris data
def plot_orbit( mag ):
	plt.ion()
	fig = plt.figure(2)
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

	ax.set_title(mag.description)



# exec(open('magpy.py').read())

#pathname = '/Users/mystical/data/spacecraft/rbsp/rbsp-a/mag_4sec_sm_L3/'
#for item in sorted(os.listdir( pathname )):
#	if item.startswith('rbsp-a') and item.endswith('.cdf'):
#		print(item)
#		type(item)
#		mag = read_cdf(item, pathname)
#		plot_mag(mag)
#		typed = input()
#		if( typed == 'q' ):
#			break



filename = 'rbsp-a_magnetometer_4sec-sm_emfisis-L3_20121001_v1.3.3.cdf'
mag = read_cdf(filename)
plot_mag(mag)
plot_orbit(mag)

