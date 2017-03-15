import PCA 
import pytest

def test_read_data():
	file='/Users/jacobhamer/Desktop/College\ Work/Spring\ 2017/PHYS767/Jacob-Practical-Stats-Exercises/4point3redone.dat'
	tb=PCA.read_data(file)
	assert tb.shape==(22,13)