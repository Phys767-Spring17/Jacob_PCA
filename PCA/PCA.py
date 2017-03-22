import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from operator import itemgetter

def read_data(file, dropnans=True):
	'''Reads in the data using pandas.read_table.

	Arguments:
	file - filename with path
	dropnans - if True will remove rows with nan values. Default: True
	
	Note:Is not yet smart enough to read in messy data, assumes it is such that read_table can handle it.

	'''	
	data=pd.read_table(file, sep='\s+', header=2, index_col=0, skip_footer=9, na_values=('----', '-----'),engine='python')
	if dropnans==True:
		data=data.dropna()
	return data


def normalize_data(data):
	'''Normalizes the data such that the mean is 0 and variance is 1.

	Arguments:
	data - pandas table that is output by read_data
	'''
	for col in data.columns:
		data.loc[:,col]=(data.loc[:,col]-data[col].mean())/data[col].var()
	return data
	#test is to check each column average is 0

def eigenpairs(data):
	'''Returns pairs of eigenvalues and eigenvectors for the covariance matrix.
	
	Arguments:
	data - normalized data passed by above function

	Note: Normalizes eigenvalues such that they sum to 1, and sorts the pairs by decreasing magnitude.
	'''
	covariance_matrix=data.cov()
	eigenvalues, eigenvectors=np.linalg.eig(covariance_matrix)
	eigenvalues=eigenvalues/np.sum(eigenvalues)
	pairs=[(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
	pairs=sorted(pairs, key=itemgetter(0))
	return pairs

def select_pairs(pairs, keep=False, threshold=False):
	'''Selects eigenpairs by keeping a certain number or by a threshold on eigenvalue size.

	Arguments:
	pairs - pairs passed by eigenpairs function
	keep - an integer number of eigenpairs to keep, or False when using threshold. Default: False.
	threshold - a value between 0 and 1, eigenpairs with eigenvalues below this value will be dropped, or False when using keep. Default: False.
	'''
	if keep!=False:
		pairs=pairs[:keep]
	if threshold!=False:
		keep=[]
		for i in range(len(pairs)):
			if pairs[i][0]>threshold:
				keep.append(pairs[i])
			else:
				break
		pairs=keep
	return pairs

def transform_data(normed_data, pairs):
	'''Returns the original data tranformed to have the principal components as its basis.

	Arguments:
	normed_data - data passed from normalized_data
	pairs - pairs that have passed through select_pairs
	'''
	proj_mat=np.hstack([(pairs[i][1].reshape(len(pairs[i][1]),1)) for i in range(len(pairs))])
	transformed=normed_data.dot(proj_mat)
	return transformed



