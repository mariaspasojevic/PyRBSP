"""
chorus_extract.py

This module loads RSBP data cdf files and integrates wave amplitude
outside the plasmapause for upper and lower band chorus. The output
chorus dataframes contain ['timestamp', 'L', 'mlat', 'mlt', 'BB'] where
BB is the instanteous chorus wave amplitude squared in units of pT^2. 
The dataframes are written as pickle files divided on per year.
"""

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
	""" 
	load_day( datestr )

	Load WFR, MAG and NURD data daily cdf files. Interpolates values
	of nurd.sphere, mag.fce_eq, mag.L, mag.mlat and mag.mlt onto the
	wfr data frame

	Input: datestr = 'YYYYMMDD'
	Output: wfr dataframe with added columns as above or -1 if one of
					the data files is not available.
	"""

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
	wfr['psphere'] = interpolate.interp1d( nurd.timestamp, nurd.psphere, \
										 kind='nearest', bounds_error = False, \
										 fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['fce_eq'] = interpolate.interp1d( mag.timestamp, mag.fce_eq, \
										 kind='nearest', bounds_error = False, \
										 fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['L'] = interpolate.interp1d( mag.timestamp, mag.L, kind='nearest', \
							 			 bounds_error = False, \
										 fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['mlat'] = interpolate.interp1d( mag.timestamp, mag.mlat, kind='nearest', \
										 bounds_error = False, \
										 fill_value = 'extrapolate')(wfr['timestamp'])
	wfr['mlt'] = interpolate.interp1d( mag.timestamp, mag.mlt, kind='nearest', \
										 bounds_error = False, \
										 fill_value = 'extrapolate')(wfr['timestamp'])

	return wfr


def extract( wfr, f_lb_lower = 0.05, f_lb_upper = 0.5, \
									f_ub_lower = 0.50, f_ub_upper = 0.8 ):
	"""
	extract( wfr, f_lb_lower = 0.05, f_lb_upper = 0.5, 
	 							f_ub_lower = 0.50, f_ub_upper = 0.8 ):
	Performs the PSD integration across the chorus frequency band seperately
	for lower and upper band chorus to determine BB, chorus amplitude squared

	Inputs: wfr dataframe as returned by load_day(), f_* frequency of 
					lower band (lb) or upper band (ub) cutoff (lower and upper)
	Outputs: lower, upper dataframes with columns ['timestamp', 'L', 'mlat',
					 'mlt', 'BB']	
	"""

	lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
	upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )

	lb_kk = 0
	ub_kk = 0
	for ii in np.where( np.logical_and.reduce( \
										( wfr['psphere']==0, wfr['L'] >=3.5, wfr['L'] <= 6) ) )[0]:
		# LOWER BAND CHORUS
		jj = np.where( np.logical_and(wfr['freq'] >= f_lb_lower*wfr['fce_eq'][ii], \
									 wfr['freq'] <= f_lb_upper*wfr['fce_eq'][ii] ) )[0]
	
		# Multiple PSD by bandwidth and integrate across the band
		# Bw^2 in units of pT^2
		BB = np.sum(wfr['BB'][jj,ii]*wfr['freq_bw'][jj])*1e6

		if( BB > 0 ):
			lower.loc[lb_kk] = [wfr['timestamp'][ii], wfr['L'][ii], wfr['mlat'][ii], \
											 wfr['mlt'][ii], BB]
			lb_kk += 1

		# UPPER BAND CHORUS
		jj = np.where( np.logical_and(wfr['freq'] >= f_ub_lower*wfr['fce_eq'][ii], \
									 wfr['freq'] <= f_ub_upper*wfr['fce_eq'][ii] ) )[0]
	
		# Multiple PSD by bandwidth and integrate across the band
		# Bw^2 in units of pT^2
		BB = np.sum(wfr['BB'][jj,ii]*wfr['freq_bw'][jj])*1e6

		if( BB > 0 ):
			upper.loc[ub_kk] = [ wfr['timestamp'][ii], wfr['L'][ii], \
												wfr['mlat'][ii], wfr['mlt'][ii], BB]
			ub_kk += 1

	return lower, upper


def extract_range( start_date, end_date ):
	"""
	extract_range( start_date, end_date ):

	Loops through a range of dates, extracting chorus amplitudes and
	concatenating results

	Input: start_date, end_date of type time.time(YYYY, MM, DD)
	Output: lower, upper dataframes with columns ['timestamp', 'L', 'mlat',
					 'mlt', 'BB']	
	"""

	lower = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )
	upper = pd.DataFrame( columns=['timestamp', 'L', 'mlat', 'mlt', 'BB'] )

	while( start_date <= end_date ):

		tic = time.time()
		datestr = start_date.strftime('%Y%m%d')
		print(datestr)

		wfr = load_day(datestr)

		if isinstance( wfr, dict ):

			day_lower, day_upper = extract( wfr )

			lower = pd.concat( [lower, day_lower], ignore_index=True )
			upper = pd.concat( [upper, day_upper], ignore_index=True )

		start_date = start_date + datetime.timedelta(days=1)

		toc = time.time()
		print( '\t' + str(round(toc-tic)) + ' sec Elapsed' )

	return lower, upper

def run(year):
	"""
	run(year)

	Extract chorus for indicated year and save pickled dataframes
	Inputs: year integer from [2012, 2013, 2014, 2015, 2016]
	"""

	if year == 2012:
		start_date = datetime.date(year, 10, 1)
	else:
		start_date = datetime.date(year, 1, 1)

	if year == 2016:
		end_date = datetime.date(year, 6, 22)
	else:
		end_date = datetime.date(year, 12, 31)

	lower, upper = extract_range( start_date, end_date )
	lower.to_pickle('chorus_lower_' + str(year) + ' pickle')
	upper.to_pickle('chorus_upper_' + str(year) + '.pickle')
	
	return
		


# exec(open('extract_chorus.py').read())

