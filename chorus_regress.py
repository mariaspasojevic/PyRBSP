"""
chorus_regress

This module provides functionality for creating regession models of 
upper and lower band chorus average wave amplitude squared from RBSP data

Two models are created:
	f(mlt, L, mlat) in units of pT^2 and 
	g(Kp) is a dimensionless scaling factor (units pT^2/pT^2) for scaling f()

Plots can be created and the model coeficients are stored as csv files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import pickle
from scipy import interpolate
from scipy import integrate
import kp_load
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression


def load_pickles( band = 'Lower' ):
	"""
	load_pickles( band = 'Lower' ):
	
	Loads the pickle files that were create by chorus_extract.py and
	contain the integrated wave amplitude squared and ephmeris data.
	Concat into a single dataframe

	Inputs: band = 'Lower' | 'Upper'
	Returns: chorus dataframe
	"""

	wildfile = 'chorus_' + str.lower(band) + '*.pickle'
	files = glob.glob( wildfile )

	for f in files:
		next = pickle.load( open( f, 'rb' ) )
		try:
			chorus = pd.concat( [chorus, next], ignore_index=True )
		except NameError:
			chorus = next

	return chorus


def interp_kp( timestamp ):
	"""
	interp_kp( timestamp )

	Loads the Kp values for the RBSP mission. Interpolates Kp onto input array

	Inputs: timestamp is an array of timestamp to interpolate onto
	Returns: array of Kp values corresponding to input
	"""

	kp = kp_load.rbsp()
	chorus_kp = interpolate.interp1d( kp.timestamp, kp.kp, \
																		kind='linear' )(timestamp)

	return chorus_kp


def avg_kp( chorus, num_bins = 25 ):
	"""
	avg_kp( chorus, num_bins = 25 )

	Averages Bw^2 (chorus.BB) over num_bins-1 linearly spaced bins

	Inputs: chorus dataframe as created by extract_chorus.py
	Returns: kp dataframe with columns 'kp' (median value in bin) 
					 													 'BB' (mean value in bin)
	"""
	# Linearly Spaced Bins
	kp_bins = np.linspace( min( chorus.kp ), max( chorus.kp ), num_bins )
	# Make sure max(kp) goes into last bin 
	kp_bins[-1] = kp_bins[-1] + 0.01

	kp = pd.DataFrame( columns=['kp', 'BB'] )
	print('Averaging Across Kp')
	for k in  np.arange(0,len(kp_bins)-1):

		print(k)
	 
		ii = chorus[ (chorus.kp >= kp_bins[k]) & \
								 (chorus.kp < kp_bins[k+1]) ].index.tolist()
		kp.loc[k] = [np.median(chorus.kp.loc[ii]), np.mean(chorus.BB.loc[ii])]

	return kp


def regress_kp( all_kp, max_kp=6, order=4, band='Lower' ):
	"""
  regress_kp( all_kp, max_kp=6, order=4, band='Lower' ):

	Creates a regression model of BB vs Kp up to max_kp of order
	Scales the model to an average value of 1

	Inputs: all_kp dataframe as created by avg_kp
					max_kp maximum value of Kp for model
					order of polynomial in regression
					band for plotting purposes
	Returns: w: coefficients of unscaled polynomial
				   G_0: inverse of scaling constant
					 feat_names: list of strings corresponding to coefficients
	"""

	kp = all_kp[all_kp.kp<max_kp]

	X = kp.kp.values.reshape(-1,1)
	y = kp.BB
	
	model = make_pipeline( PolynomialFeatures(order), RidgeCV() )
	model.fit( X, y )

	w = model.named_steps['ridgecv'].coef_
	feat_names = model.named_steps['polynomialfeatures'].get_feature_names()
	
	G_0 = integrate.quad( lambda x: \
		np.polyval( w[::-1], x), 0, max_kp)[0]/(max_kp-0)
	g = lambda x: np.polyval( w[::-1], x)/G_0 

	# Should Integrate to max_kp
	h = integrate.quad( g, 0, max_kp)[0]

	print(G_0)
	print(h)
	print(w)

 	# Plot Kp Model
	fig1 = plt.figure(num=1 if band=='Lower' else 2, figsize=(8,4) )
	fig1.clf()
	ax1 = fig1.add_subplot(131)
	ax1.plot( all_kp.kp, np.log10(all_kp.BB), 'bo-')
	ax1.set_ylabel('log(Bw2, pT2)')
	ax1.set_title( band + ' Band Chorus' )

	ax2 = fig1.add_subplot(132)
	ax2.plot( X, np.log10(y), 'bo')
	ax2.plot( X, np.log10(model.predict(X)), 'k')
	ax2.set_title( 'g_0(Kp)')
	ax2.set_xlabel('Kp')

	ax3 = fig1.add_subplot(133)
	ax3.plot( X, g(X), 'k')
	ax3.set_ylabel('Unitless Scaling Factor')
	ax3.set_title( 'g(Kp)')

	fig1.tight_layout()
	plt.show(block=False)
	plt.draw()

	plt.savefig( 'regress_kp_' + str.lower(band) + '.pdf' )

	return w, G_0, feat_names

def avg_bb( chorus ):
	"""
	avg_bb( chorus )

	Computes the average amplitude squared in a grid of mlt, L, mlat

	Inputs: chorus dataframe
	Output: X matrix with columns cos(mlt), sin(mlt), L, mlat, meanBB, numPts
	"""
	chorus.mlat = abs(chorus.mlat)
	L_bin = chorus.L.quantile( \
					np.linspace(1/12, 1, 12) ).reset_index(drop=True)
	mlat_bin = chorus.mlat.quantile( \
						 np.linspace(1/12, 1, 12) ).reset_index(drop=True)
	mlt_bin = chorus.mlt.quantile( \
						np.linspace(1/12, 1, 12) ).reset_index(drop=True)

  # Make sure last bin contains largest value
	L_bin.iloc[-1] = L_bin.iloc[-1] + 0.01
	mlat_bin.iloc[-1] = mlat_bin.iloc[-1] + 0.01
	mlt_bin.iloc[-1] = mlt_bin.iloc[-1] + 0.01

	L_mid = pd.Series()
	mlat_mid = pd.Series()
	mlt_mid = pd.Series()

	X = pd.DataFrame( columns=['cos_mlt', 'sin_mlt', 'L', 'mlat', \
													   'meanBB', 'numPts'] )
	
	print('Averaging Across MLT, L, MLAT')
	for k in np.arange(0,len(mlt_bin)-1):

		print(k)
		# Get bin center
		ii = chorus[ ( chorus.mlt >= mlt_bin[k]) & \
								 (chorus.mlt < mlt_bin[k+1] )].index.tolist()
		mlt_mid.loc[k] = np.median( chorus.mlt.loc[ii] )

		for m in np.arange(0,len(mlat_bin)-1):
			if k == 0:
				# Get bin center
				ii = chorus[ ( chorus.mlat >= mlat_bin[m]) & \
									  	(chorus.mlat < mlat_bin[m+1] )].index.tolist()
				mlat_mid.loc[m]= np.median(chorus.mlat.loc[ii] )

			for p in np.arange(0,len(L_bin)-1):
				if k == 0 and m == 0:
					# Get bin center
					ii = chorus[ ( chorus.L >= L_bin[p]) & \
											 (chorus.L < L_bin[p+1] )].index.tolist()
					L_mid.loc[p]= np.median(chorus.L.loc[ii])
				   
				ii = chorus[ (chorus.mlt >= mlt_bin[k])   &  \
										(chorus.mlt  < mlt_bin[k+1])  & \
								 	  (chorus.mlat >= mlat_bin[m])  & \
										(chorus.mlat < mlat_bin[m+1]) & \
									 	(chorus.L    >= L_bin[p])     & \
										(chorus.L    < L_bin[p+1])   ].index.tolist()

				if len(ii) != 0:
					X.loc[len(X)] = [ np.cos(mlt_mid[k]/12*np.pi), \
													np.sin(mlt_mid[k]/12*np.pi), \
													L_mid[p], mlat_mid[m], \
									 				np.mean( chorus.BB.loc[ii] ), len(ii) ]
			
	return X


def regress_mlt_L_mlat( X, band='Lower'):
	"""
	Create a regression model (first order in cos(MLT), sin(MLT), 
	and second order in L and MLAT

	Input: X as created by avg_BB
	Ouput: model: fit from LinearRegression, poly_feat from PolynomialFeatures
				 feat_names names of polynomial features 

	Note: When applying poly_feat.transform the columns corresponding to x0^2
				and x1^2 must be removed
	"""

	y = X['meanBB']

	X_new = X.drop( 'meanBB', axis=1, inplace=False)
	X_new.drop( 'numPts', axis=1, inplace=True)
	
	poly_feat = PolynomialFeatures( degree = 2 )
	poly_feat.fit(X_new)

	X_new = poly_feat.transform(X_new)
	# Want a model that is second order in L and MLAT, but
	# only first order in cos(MLT) and sin(MLT)
	ii = [poly_feat.get_feature_names().index('x0^2'), \
				poly_feat.get_feature_names().index('x1^2')]
	X_new = np.delete(X_new, ii, axis=1 )

	# Polynomial Features already includes a constant so set fit_intercept=False
	model = LinearRegression(fit_intercept=False)
	model.fit( X_new, y )

	plot_model_mlt_L_mlat(fig_num=10, band=band, model=model, poly_feat=poly_feat)

	feat_names = poly_feat.get_feature_names()
	feat_names.remove('x0^2')
	feat_names.remove('x1^2')

	return model, feat_names, poly_feat


def write_kp_coef( kp_coef, kp_G0, kp_feat_names, filename ):
	""" Write coef to g(Kp) to a csv file along with feature names """

	import csv
	with open(filename, 'w') as csvfile:

		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['G0'] + [x.replace('x0', 'Kp') for x in kp_feat_names] )
		csvwriter.writerow( [str(kp_G0)] + [str(x) for x in kp_coef] )
			

def write_mlt_L_mlat_coef( coef, feat_names, filename ):
	""" Write coef to f(mlt, L, mlat) to a csv file along with feature names """

	with open(filename, 'w') as csvfile:

		csvwriter = csv.writer(csvfile)

		for i, x in enumerate( feat_names ):
			feat_names[i] = feat_names[i].replace('x0', 'cos(MLT)')
			feat_names[i] = feat_names[i].replace('x1', 'sin(MLT)')
			feat_names[i] = feat_names[i].replace('x2', 'L')
			feat_names[i] = feat_names[i].replace('x3', 'MLAT')

		csvwriter.writerow(feat_names)
		csvwriter.writerow( [str(x) for x in coef] )


def makeX_vary_one_of_mlt_L_mlat( which, minWhich, maxWhich, \
																	other1, other2, poly_feat=[] ):
	"""	
 	makeX_vary_one_of_mlt_L_mlat( which, minWhich, maxWhich, 
																other1, other2, poly_feat=[] ):
	Helper function for making 2-D plots of model output
	Inputs: which = 'mlt' | 'L' | 'mlat'
					minWhich, maxWhich: min and max value of which
					other1, other2: fixed value for other two variables
					if poly_feat is specified, it is used to transform variables
						otherwise X is constructed manually
	Returns: x_vary (corresponds to which), X (full matrix of regresion inputs)
	"""

	if which=='mlt':
		mlt = np.linspace( minWhich, maxWhich, 50 )
		L = np.ones(len(mlt))*other1
		mlat = np.ones(len(mlt))*other2
		a = np.cos(mlt/12*np.pi)
		b = np.sin(mlt/12*np.pi)
		x_vary = mlt

	elif which=='L': 
		L = np.linspace( minWhich, maxWhich, 50 )
		a = np.ones(len(L))*np.cos(other1/12*np.pi)
		b = np.ones(len(L))*np.sin(other1/12*np.pi)
		mlat = np.ones(len(L))*other2
		x_vary = L

	elif which=='mlat': 
		mlat = np.linspace( minWhich, maxWhich, 50 )
		a = np.ones(len(mlat))*np.cos(other1/12*np.pi)
		b = np.ones(len(mlat))*np.sin(other1/12*np.pi)
		L = np.ones(len(mlat))*other2
		x_vary = mlat

	# if poly_feat has been loaded from csv file
	if not poly_feat:
		intercept = np.ones(len(L))
		ab = a*b
		aL = a*L
		amlat = a*mlat
		bL = b*L
		bmlat = b*mlat
		LL = L**2
		Lmlat = L*mlat
		mlatmlat = mlat**2

		x_new = np.vstack( (intercept,a,b,L,mlat, ab, aL, amlat, \
											bL, bmlat, LL, Lmlat, mlatmlat) ).transpose()
	else:
		x = np.vstack( (a,b,L,mlat) ).transpose()
		x_new = poly_feat.transform(x)
		ii = [poly_feat.get_feature_names().index('x0^2'), \
					poly_feat.get_feature_names().index('x1^2')]
		x_new = np.delete(x_new, ii, axis=1 )

	return x_vary, x_new


def plot_model_mlt_L_mlat(fig_num = 10, band='Lower', coef_values=[], \
													model=[], poly_feat=[] ):
	"""
	def plot_model_mlt_L_mlat(fig_num = 10, band='Lower', coef_values=[], \
														model=[], poly_feat=[] ):
	Plots a series of 2D plots of f(mlt, L, mlat)
	Two options are available. If coef_values are given then input matrix
	X is computed by manually and Y=X.dot(coef_values). If model and poly_feat
	are give then X = poly_feat.transform() and Y = model.predict(X)

	Saves plot as pdf file
	"""

	fig1 = plt.figure(num=fig_num if band=='Lower' else fig_num+1, \
										figsize=(10,10) )
	fig1.clf()
	q = 1

	color = 'rgb'
	# Vary MLT on x-axis, L by column
	for k, L in enumerate([4,5,6]):
		ax = fig1.add_subplot(3,3,q)
		q = q+1
		for m, mlat in enumerate([1,8,15]):
			# If model not passed, calculate x_new, y_new from loaded csv file
			if not model:
				mlt, x_new = makeX_vary_one_of_mlt_L_mlat( 'mlt', 0, 24, L, mlat)
				y_new = x_new.dot(coef_values)
			# Otherwise use polynomial features and fitted model
			else:
				mlt, x_new = makeX_vary_one_of_mlt_L_mlat( 'mlt', 0, 24, L, mlat, \
																										poly_feat )
				y_new = model.predict( x_new )

			ax.plot(mlt, np.log10(y_new), color[m], label='$\lambda$='+str(mlat) )

		ax.legend()
		ax.set_xlabel('MLT for L = ' + str(L))
		if k==0:
			ax.set_ylabel('log(Bw2, pT2)')
		if k==1:
			ax.set_title( band + ' Band Chorus' )
		ax.set_xlim(0,24)
		if band=='Lower':
			ax.set_ylim(1, 3.4)
		else:
			ax.set_ylim(0, 2.75)

	# Vary MLT on x-axis, MLAT by column
	for k, mlat in enumerate([1,8,15]):
		ax = fig1.add_subplot(3,3,q)
		q = q+1
		for m, L in enumerate([4,5,6]):
			# If model not passed, calculate x_new, y_new from loaded csv file
			if not model:
				mlt, x_new = makeX_vary_one_of_mlt_L_mlat( 'mlt', 0, 24, L, mlat )
				y_new = x_new.dot(coef_values)
			# Otherwise use polynomial features and fitted model
			else:
				mlt, x_new = makeX_vary_one_of_mlt_L_mlat( 'mlt', 0, 24, L, mlat, \
																										poly_feat )
				y_new = model.predict( x_new )

			ax.plot(mlt, np.log10(y_new), color[m], label='L='+str(L) )
	
		ax.legend()
		ax.set_xlabel('MLT for $\lambda$ = ' + str(mlat))
		if k==0:
			ax.set_ylabel('log(Bw2, pT2)')
		ax.set_xlim(0,24)
		if band=='Lower':
			ax.set_ylim(1, 3.4)
		else:
			ax.set_ylim(0, 2.75)
	
	# Vary MLAT on x-axis, MLT by column
	for k, mlt in enumerate([1,6,12]):
		ax = fig1.add_subplot(3,3,q)
		q = q+1
		for m, L in enumerate([4,5,6]):
			# If model not passed, calculate x_new, y_new from loaded csv file
			if not model:
				mlat, x_new = makeX_vary_one_of_mlt_L_mlat( 'mlat', 0, 17, mlt, L )
				y_new = x_new.dot(coef_values)
			# Otherwise use polynomial features and fitted model
			else:
				mlat, x_new = makeX_vary_one_of_mlt_L_mlat( 'mlat', 0, 17, mlt, L, \
																										poly_feat )
				y_new = model.predict( x_new )

			ax.plot(mlat, np.log10(y_new), color[m], label='L='+str(L) )
	
		ax.legend()
		ax.set_xlabel('$\lambda$ for MLT = ' + str(mlt))
		if k==0:
			ax.set_ylabel('log(Bw2, pT2)')
		ax.set_xlim(0,17)
		if band=='Lower':
			ax.set_ylim(1, 3.4)
		else:
			ax.set_ylim(0, 2.75)
	
	plt.show(block=False)
	plt.draw()

	if not model:
		plt.savefig( 'regress_mlt_L_mlat_' + str.lower(band) + '_verify.pdf' )
	else:
		plt.savefig( 'regress_mlt_L_mlat_' + str.lower(band) + '.pdf' )

	return

def run():
	"""
	run()
	
	Main Function of Module
	Loads data, performs averaging, creates models, plots models, saves coefs
	"""

	# UPPER BAND CHORUS MODEL
	upper = load_pickles( band = 'Upper' )
	upper['kp'] = interp_kp( upper['timestamp'] )

	upper_kp = avg_kp( upper )
	upper_kp_coef, upper_kp_G0, upper_kp_feat_names = regress_kp( \
		upper_kp, max_kp=6, band='Upper' )
	write_kp_coef( upper_kp_coef, upper_kp_G0, upper_kp_feat_names, \
		'upper_g_coef.csv' )
	
	X_upper  = avg_bb( upper )
	upper_model, upper_feat_names, poly_feat = regress_mlt_L_mlat( \
		X_upper, band='Upper')
	write_mlt_L_mlat_coef( upper_model.coef_, upper_feat_names, \
		'upper_f_coef.csv' )

	
	# LOWER BAND CHORUS MODEL
	lower = load_pickles( band = 'Lower' )
	lower['kp'] = interp_kp( lower['timestamp'] )
	
	lower_kp = avg_kp( lower )
	lower_kp_coef, lower_kp_G0, lower_kp_feat_names = regress_kp( \
		lower_kp, max_kp=6, band='Lower' )
	write_kp_coef( lower_kp_coef, lower_kp_G0, lower_kp_feat_names, \
		'lower_g_coef.csv' )
	
	X_lower = avg_bb( lower )
	lower_model, lower_feat_names, poly_feat = regress_mlt_L_mlat( \
		X_lower, band='Lower')
	write_mlt_L_mlat_coef( lower_model.coef_, lower_feat_names, \
		'lower_f_coef.csv' )

def verify():
	"""
	verify()

	Produces plots of model by loading csv files. Used to verify output
	is correctly written
	"""

	for band in ['Upper', 'Lower']:

		filename = str.lower(band) + '_f_coef.csv'
		print('Loading ' + filename)

		with open(filename, 'rt') as csvfile:
			csvreader = csv.reader( csvfile )
			coef_names = next(csvreader)
			coef_str = next(csvreader)
			coef_values = np.array(coef_str).astype(np.float)
			
		plot_model_mlt_L_mlat( fig_num=20, band=band, coef_values=coef_values )
	
	return
	
# exec( open('chorus_regress.py').read() )
