from pca.pca import *

def test_read_data1():
	file='data/4point3redone.dat'
	tb=read_data(file)
	assert tb.shape==(18,13)

def test_normalize_data():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	for c in normed.columns:
		assert c.mean()==0

def test_eigenpairs1():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	assert np.sum(pairs[:,0])==1

def test_eigenpairs2():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	for i in np.arange(len(pairs)-1):
		assert pairs[i+1][0] > pairs[i][0]

def test_eigenpairs3():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	for i in np.arange(len(pairs)):
		assert np.linalg.norm(pairs[i][1])==1

def test_select_pairs1():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	selected_pairs=select_pairs(pairs, keep=2)
	assert len(selected_pairs)==2

def test_select_pairs2():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	selected_pairs=select_pairs(pairs, threshold=.2)
	assert np.min(selected_pairs[:,0])>.2

def test_transform_data():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	selected_pairs=select_pairs(pairs, keep=2)
	trans_data=transform_data(normed, selected_pairs)
	assert trans_data.shape==(2,13)
