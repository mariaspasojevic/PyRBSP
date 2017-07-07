import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import pickle
from scipy import interpolate
from scipy import integrate
import kp_load


# LOAD EXTRACTED CHORUS DATA
def load_pickles( type = 'lower' ):

	wildfile = 'chorus_' + type + '*.pickle'
	files = glob.glob( wildfile )

	for f in files:
		next = pickle.load( open( f, 'rb' ) )
		try:
			chorus = pd.concat( [chorus, next], ignore_index=True )
		except NameError:
			chorus = next

	return chorus

# LOAD AND INTERPOLATE KP
def interp_kp( timestamp ):

	kp = kp_load.rbsp()
	chorus_kp = interpolate.interp1d( kp.timestamp, kp.kp, kind='linear' )(timestamp)

	return chorus_kp


# AVERAGE KP
def avg_kp( chorus, num_bins = 25 ):
	# Linearly Spaced Bins
	kp_bins = np.linspace( min( chorus.kp ), max( chorus.kp ), num_bins )
	# Make sure max(kp) goes into last bin 
	kp_bins[-1] = kp_bins[-1] + 0.01

	kp = pd.DataFrame( columns=['kp', 'BB'] )
	print('Averaging Across Kp')
	for k in  np.arange(0,len(kp_bins)-1):

		print(k)
	 
		ii = chorus[ (chorus.kp >= kp_bins[k]) & (chorus.kp < kp_bins[k+1]) ].index.tolist()
		kp.loc[k] = [ np.median( chorus.kp.loc[ii] ), np.mean( chorus.BB.loc[ii] ) ]

	return kp

# 
def regress_kp( all_kp, max_kp=6, order=4, band='Lower' ):
	from sklearn.linear_model import RidgeCV
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.pipeline import make_pipeline

	kp = all_kp[all_kp.kp<max_kp]

	X = kp.kp.values.reshape(-1,1)
	y = kp.BB
	
	model = make_pipeline( PolynomialFeatures(order), RidgeCV() )
	model.fit( X, y )

	w = model.named_steps['ridgecv'].coef_
	feat_names = model.named_steps['polynomialfeatures'].get_feature_names()
	G_0 = integrate.quad( lambda x: np.polyval( w[::-1], x), 0, max_kp)[0]/(max_kp-0)
	g = lambda x: np.polyval( w[::-1], x)/G_0 
	# Should Integrate to max_kp
	h = integrate.quad( g, 0, max_kp)[0]
	print(G_0)
	print(h)
	print(w)

	if band=='Upper':
		plt_num = 1
	else:
		plt_num = 2

	fig1 = plt.figure(1 if band=='Lower' else 2)
	fig1.clf()
	ax1 = fig1.add_subplot(131)
	ax1.plot( all_kp.kp, all_kp.BB, 'bo-')
	ax1.set_ylabel('log(Bw2, pT2)')

	ax2 = fig1.add_subplot(132)
	ax2.plot( X, y, 'bo')
	ax2.plot( X, model.predict(X), 'k')
	ax2.set_title( band + ' Band Chorus' )
	ax2.set_xlabel('Kp')

	ax3 = fig1.add_subplot(133)
	ax3.plot( X, g(X), 'k')
	ax3.set_ylabel('Unitless Scaling Factor')

	fig1.tight_layout()
	plt.show(block=False)
	plt.draw()

	plt.savefig( 'regress_kp' + str.lower(band) + '.pdf' )

	return w, G_0, feat_names

# MAKE GRID OF L, MLAT, MLT FOR AVERAGING BB
def avg_bb( chorus ):
	chorus.mlat = abs(chorus.mlat)
	L_bin = chorus.L.quantile( np.linspace(1/12, 1, 12) ).reset_index(drop=True)
	mlat_bin = chorus.mlat.quantile( np.linspace(1/12, 1, 12) ).reset_index(drop=True)
	mlt_bin = chorus.mlt.quantile( np.linspace(1/12, 1, 12) ).reset_index(drop=True)

	L_bin.iloc[-1] = L_bin.iloc[-1] + 0.01
	mlat_bin.iloc[-1] = mlat_bin.iloc[-1] + 0.01
	mlt_bin.iloc[-1] = mlt_bin.iloc[-1] + 0.01

	L_mid = pd.Series()
	mlat_mid = pd.Series()
	mlt_mid = pd.Series()

	X = pd.DataFrame( columns=['cos_mlt', 'sin_mlt', 'L', 'mlat', 'meanBB', 'numPts'] )
	
	print('Averaging Across MLT, L, MLAT')
	for k in np.arange(0,len(mlt_bin)-1):

		print(k)
		ii = chorus[ ( chorus.mlt >= mlt_bin[k]) & (chorus.mlt < mlt_bin[k+1] )].index.tolist()
		mlt_mid.loc[k] = np.median( chorus.mlt.loc[ii] )

		for m in np.arange(0,len(mlat_bin)-1):
			if k == 0:
				ii = chorus[ ( chorus.mlat >= mlat_bin[m]) & (chorus.mlat < mlat_bin[m+1] )].index.tolist()
				mlat_mid.loc[m]= np.median(chorus.mlat.loc[ii] )

			for p in np.arange(0,len(L_bin)-1):
				if k == 0 and m == 0:
					ii = chorus[ ( chorus.L >= L_bin[p]) & (chorus.L < L_bin[p+1] )].index.tolist()
					L_mid.loc[p]= np.median(chorus.L.loc[ii])
				   
				ii = chorus[ (chorus.mlt  >= mlt_bin[k])  & (chorus.mlt  < mlt_bin[k+1])  & \
								 	 (chorus.mlat >= mlat_bin[m]) & (chorus.mlat < mlat_bin[m+1]) & \
									 (chorus.L    >= L_bin[p])    & (chorus.L    < L_bin[p+1])   ].index.tolist()

				if len(ii) != 0:
					X.loc[len(X)] = [ np.cos(mlt_mid[k]/12*np.pi), np.sin(mlt_mid[k]/12*np.pi), \
													L_mid[p], mlat_mid[m], \
									 				np.mean( chorus.BB.loc[ii] ), len(ii) ]
			
	return X

def makeX_vary_L( minL, maxL, mlat, mlt, poly_feat ):

	L = np.linspace( minL, maxL, 24 )
	mlat = np.ones(len(L))*mlat
	a = np.ones(len(L))*np.cos(mlt/12*np.pi)
	b = np.ones(len(L))*np.sin(mlt/12*np.pi)

	x = np.vstack( (a,b,L,mlat) ).transpose()
	x_new = poly_feat.transform(x)
	x_new = np.delete(x_new, poly_feat.get_feature_names().index('x0^2'), axis=1 )
	x_new = np.delete(x_new, poly_feat.get_feature_names().index('x1^2'), axis=1 )

	return L, x_new

def makeX_vary_mlat( minmlat, maxmlat, L, mlt, poly_feat ):

	mlat = np.linspace( minmlat, maxmlat, 24 )
	L = np.ones(len(mlat))*L
	a = np.ones(len(mlat))*np.cos(mlt/12*np.pi)
	b = np.ones(len(mlat))*np.sin(mlt/12*np.pi)

	x = np.vstack( (a,b,L,mlat) ).transpose()
	x_new = poly_feat.transform(x)
	x_new = np.delete(x_new, poly_feat.get_feature_names().index('x0^2'), axis=1 )
	x_new = np.delete(x_new, poly_feat.get_feature_names().index('x1^2'), axis=1 )

	return mlat, x_new

def makeX_vary_mlt( minmlt, maxmlt, L, mlat, poly_feat ):

	mlt = np.linspace( minmlt, maxmlt, 24 )
	L = np.ones(len(mlt))*L
	mlat = np.ones(len(mlt))*mlat
	a = np.cos(mlt/12*np.pi)
	b = np.sin(mlt/12*np.pi)

	x = np.vstack( (a,b,L,mlat) ).transpose()
	x_new = poly_feat.transform(x)
	x_new = np.delete(x_new, poly_feat.get_feature_names().index('x0^2'), axis=1 )
	x_new = np.delete(x_new, poly_feat.get_feature_names().index('x1^2'), axis=1 )

	return mlt, x_new

def regress( X, band='Lower'):
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.linear_model import LinearRegression

	y = X['meanBB']

	X_new = X.drop( 'meanBB', axis=1, inplace=False)
	X_new.drop( 'numPts', axis=1, inplace=True)
	
	poly_feat = PolynomialFeatures( degree = 2 )
	poly_feat.fit(X_new)

	X_new = poly_feat.transform(X_new)
	X_new = np.delete(X_new, poly_feat.get_feature_names().index('x0^2'), axis=1 )
	X_new = np.delete(X_new, poly_feat.get_feature_names().index('x1^2'), axis=1 )

	# Polynomial Features already includes a constant so set fit_intercept=False
	model = LinearRegression(fit_intercept=False)
	model.fit( X_new, y )

	fig1 = plt.figure(3 if band=='Lower' else 4)
	fig1.clf()
	q = 1

	color = 'rgb'
	for k, L in enumerate([4,5,6]):
		ax = fig1.add_subplot(3,3,q)
		q = q+1
		for m, mlat in enumerate([1,8,15]):
			mlt, x_new = makeX_vary_mlt( 0, 24, L, mlat, poly_feat )
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
	
	for k, mlat in enumerate([1,8,15]):
		ax = fig1.add_subplot(3,3,q)
		q = q+1
		for m, L in enumerate([4,5,6]):
			mlt, x_new = makeX_vary_mlt( 0, 24, L, mlat, poly_feat )
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
	
	for k, mlt in enumerate([1,6,12]):
		ax = fig1.add_subplot(3,3,q)
		q = q+1
		for m, L in enumerate([4,5,6]):
			mlat, x_new = makeX_vary_mlat( 0, 17, L, mlt, poly_feat )
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
	plt.savefig( 'regress_' + str.lower(band) + '.pdf' )

	feat_names = poly_feat.get_feature_names()
	feat_names.remove('x0^2')
	feat_names.remove('x1^2')


	return model, feat_names


# exec(open('chorus_regress.py').read())

#upper = load_pickles( type = 'upper' )
#upper['kp'] = interp_kp( upper['timestamp'] )
#upper_kp = avg_kp( upper )
#upper_kp_coef, upper_kp_G0, upper_kp_feat_names = regress_kp( upper_kp, max_kp=6, band='Upper' )

X_upper  = avg_bb( upper )
upper_model, upper_feat_names = regress( X_upper, 'Upper')

#lower = load_pickles( type = 'lower' )
#lower['kp'] = interp_kp( lower['timestamp'] )
#lower_kp = avg_kp( lower )
#lower_kp_coef, lower_kp_G0, lower_kp_feat_names = regress_kp( lower_kp, max_kp=6, band='Lower' )

X_lower = avg_bb( lower )
lower_model, lower_feat_names = regress( X_lower, band='Lower')





















