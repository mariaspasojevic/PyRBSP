"""
kp_load.py

This module reads txt files as downloaded from 
http://wdc.kugi.kyoto-u.ac.jp/kp/index.html
containing values of the Kp index
"""

import os.path
import pandas as pd
import numpy as np
import datetime
import time
import pickle


def read_txt( filename ):
	"""
	read_txt( filename )

	Reads and returns Kp data
	
	Input: filename: string containing full path of Kp text file
				 					 Kp text file must contain a 2 line header before
									 main data set
	Output: kp: dataframe with columns ['timestamp', 'kp']
	"""

	k = 0
	kp = pd.DataFrame( columns=['timestamp', 'kp'] )

	f = open( filename )
 
	f.readline()
	f.readline()
	for line in f:

		if line == '\n':
			continue

		kdate = datetime.datetime( int(line[0:4]), int(line[4:6]), \
															 int(line[6:8]), 1, 30 )

		for m in np.arange(9,24,2):
			kvalue = int(line[m])
			if line[m+1] == '+':
				kvalue = kvalue + 1/3
			elif line[m+1] == '-':
				kvalue = kvalue - 1/3
		
			kp.loc[k] = [time.mktime( kdate.timetuple() ), kvalue]
		
			k += 1
			kdate = kdate + datetime.timedelta( hours = 3 )
	
	return kp

def rbsp():
	"""
	rbsp()
	
	Reads the Kp data for the interval of the RBSP mission. If a pickle
	file already exists, it is loaded. Otherwise the txt file is read.

	Output: kp: dataframe with columns ['timestamp', 'kp']
	"""

	filename = '/Users/mystical/Work/Science/MagneticIndex/kp_rbsp'
	if os.path.isfile( filename + '.pickle' ):
		kp = pickle.load( open( filename + '.pickle', 'rb') )
	else:
		kp = read_txt( filename + '.txt' )
		kp.to_pickle(filename + '.pickle' )

	return kp

		
# exec(open('kp_load.py').read())

