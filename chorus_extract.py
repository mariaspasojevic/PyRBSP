import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import wfrpy
import magpy
import nurdpy


def load_day( datestr ):

	wfr = wfrpy.load_day(datestr)
	nurd = nurdpy.load_day(datestr)
	mag = magpy.load_day(datestr)

	# Check that all data is available
	if not(isinstance(mag, pd.DataFrame) & \
		 	isinstance( nurd, pd.DataFrame ) & \
		 	isinstance( wfr, dict) ):
	
		print('*** Missing Data on: ' + datestr )
		return -1

	# Interpolate onto WFR time grid
	wfr['psphere'] = interpolate.interp1d( nurd.timestamp, nurd.psphere, kind='nearest', \
													bounds_error = False, fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['fce_eq'] = interpolate.interp1d( mag.timestamp, mag.fce_eq, kind='nearest', \
													bounds_error = False, fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['L'] = interpolate.interp1d( mag.timestamp, mag.L, kind='nearest', \
													bounds_error = False, fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['mlat'] = interpolate.interp1d( mag.timestamp, mag.mlat, kind='nearest', \
													bounds_error = False, fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['mlt'] = interpolate.interp1d( mag.timestamp, mag.mlt, kind='nearest', \
													bounds_error = False, fill_value = 'extrapolate')(wfr['timestamp'])

	return wfr


def extract_day( wfr, f_lb_lower = 0.05, f_lb_upper = 0.5, f_ub_lower = 0.5, f_ub_upper = 0.8 ):
	
	lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
	upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )

	lb_kk = 0
	ub_kk = 0
	for ii in np.where( np.logical_and.reduce( \
										( wfr['psphere']==0, wfr['L'] >=3.5, wfr['L'] <= 6) ) )[0]:
		# LOWER BAND CHORUS
		jj = np.where( np.logical_and( wfr['freq'] >= f_lb_lower*wfr['fce_eq'][ii], \
																 wfr['freq'] <= f_lb_upper*wfr['fce_eq'][ii] ) )[0]
	
		# Multiple PSD by bandwidth and integrate across the band
		# Bw^2 in units of pT^2
		BB = np.sum(wfr['BB'][jj,ii]*wfr['freq_bw'][jj])*1e6

		if( BB > 0 ):
			lower.loc[lb_kk] = [ wfr['timestamp'][ii], wfr['L'][ii], wfr['mlat'][ii], \
											 wfr['mlt'][ii], BB]
			lb_kk += 1

		# UPPER BAND CHORUS
		jj = np.where( np.logical_and( wfr['freq'] >= f_ub_lower*wfr['fce_eq'][ii], \
																 wfr['freq'] <= f_ub_upper*wfr['fce_eq'][ii] ) )[0]
	
		# Multiple PSD by bandwidth and integrate across the band
		# Bw^2 in units of pT^2
		BB = np.sum(wfr['BB'][jj,ii]*wfr['freq_bw'][jj])*1e6

		if( BB > 0 ):
			upper.loc[ub_kk] = [ wfr['timestamp'][ii], wfr['L'][ii], wfr['mlat'][ii], \
											 wfr['mlt'][ii], BB]
			ub_kk += 1

	return lower, upper

def extract_range( start_date, end_date ):
	lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
	upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )

	while( start_date <= end_date ):

		tic = time.time()
		datestr = start_date.strftime('%Y%m%d')
		print(datestr)

		wfr = load_day(datestr)

		if isinstance( wfr, dict ):

			day_lower, day_upper = extract_day( wfr )

			lower = pd.concat( [lower, day_lower], ignore_index=True )
			upper = pd.concat( [upper, day_upper], ignore_index=True )

		start_date = start_date + datetime.timedelta(days=1)

		toc = time.time()
		print( '\t' + str(round(toc-tic)) + ' sec Elapsed' )

	return lower, upper

# exec(open('extract_chorus.py').read())

##### 2013
#lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
#upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
#
#start_date = datetime.date(2013, 1, 1)
#end_date = datetime.date(2013, 12, 31)
#
#lower, upper = extract_range( start_date, end_date )
#
#lower.to_pickle('chorus_lower_2013.pickle')
#upper.to_pickle('chorus_upper_2013.pickle')
#
###### 2014
#lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
#upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
#
#start_date = datetime.date(2014, 1, 1)
#end_date = datetime.date(2014, 12, 31)
#
#lower, upper = extract_range( start_date, end_date )
#
#lower.to_pickle('chorus_lower_2014.pickle')
#upper.to_pickle('chorus_upper_2014.pickle')
#
###### 2015
#lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
#upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
#
#start_date = datetime.date(2015, 1, 1)
#end_date = datetime.date(2015, 12, 31)
#
#lower, upper = extract_range( start_date, end_date )
#
#lower.to_pickle('chorus_lower_2015.pickle')
#upper.to_pickle('chorus_upper_2015.pickle')
#
##### 2016
lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )

start_date = datetime.date(2016, 1, 1)
end_date = datetime.date(2016, 6, 22)

lower, upper = extract_range( start_date, end_date )

lower.to_pickle('chorus_lower_2016.pickle')
upper.to_pickle('chorus_upper_2016.pickle')

