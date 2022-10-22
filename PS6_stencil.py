import csv
from math import log
from collections import defaultdict, Counter
import random
import numpy as np

# set random seed so that random draws are the same each time
random.seed(12409)


# compute ridge estimates given X, y, and Lambda
def ridge(X, y, fLambda):
	
	# TODO: compute afBeta
	
	return afBeta

loans = []

f = open('loans_PS6.csv', 'r')
reader = csv.reader(f)
header = reader.next()

X = []
Y = []
testing_X = []
testing_Y = []

for i, row in enumerate(reader):
	# TODO: read in file, split sample into train and test

f.close()

X_matrix  = np.matrix(X)
Y_matrix  = np.array(Y)
testing_X = np.matrix(testing_X)
testing_Y = np.array(testing_Y)

X_matrix_demeaned  = X_matrix  - np.mean(X_matrix, 0)
Y_matrix_demeaned  = Y_matrix  - np.mean(Y_matrix)
testing_X_demeaned = testing_X - np.mean(testing_X, 0)
testing_Y_demeaned = testing_Y - np.mean(testing_Y)

beta = ridge(X_matrix_demeaned, Y_matrix_demeaned, 0) # (Modify the last parameter to change lambda)
print beta


# TODO: compute accuracy of your estimates
