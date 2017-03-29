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
		assert np.isclose(normed[c].mean(),0)

def test_eigenpairs1():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	total_var=0
	for pair in pairs:
		total_var+=pair[0]
	assert np.isclose(total_var,1)

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
		assert np.isclose(np.linalg.norm(pairs[i][1]),1)

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
	for i in range(len(selected_pairs)):
		assert selected_pairs[i,0]>.2

def test_transform_data():
	file='data/4point3redone.dat'
	tb=read_data(file)
	normed=normalize_data(tb)
	pairs=eigenpairs(normed)
	selected_pairs=select_pairs(pairs, keep=2)
	trans_data=transform_data(normed, selected_pairs)
	assert trans_data.shape==(18,2)
