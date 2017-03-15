import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from operator import itemgetter

def read_data(file, dropnans=True):
	data=pd.read_table(file)
	if dropnans==True:
		data=data.dropna()
	return data
	#to test, generate test data, check it comes out with expected shape


def normalize_data(data):
	for col in data.columns:
		data.loc[:,col]=(data.loc[:,col]-data[col].mean())/data[col].var()
	return data
	#test is to check each column average is 0

def eigenpairs(data):
	covariance_matrix=data.cov()
	eigenvalues, eigenvectors=np.linalg.eig(covariance_matrix)
	eigenvalues=eigenvalues/np.sum(eigenvalues)
	pairs=[(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
	pairs=sorted(pairs, key=itemgetter(0))
	return pairs

def select_pairs(pairs, keep=False, threshold=False):
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
	proj_mat=np.hstack([(pairs[i][1].reshape(len(pairs[i][1]),1)) for i in range(len(pairs))])
	transformed=normed_data.dot(proj_mat)
	return transformed

def 


