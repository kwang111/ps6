import csv
from math import log
from collections import defaultdict, Counter
import random
from re import A
import numpy as np
import string
import matplotlib.pyplot as plt

# set random seed so that random draws are the same each time
random.seed(12409)


# compute ridge estimates given X, y, and Lambda
def ridge(X, y, lamb):
	
	# TODO: compute afBeta
	a, b = np.shape(X)
	I = np.identity(b)
	Xt = np.transpose(X)
	beta = (np.linalg.inv((Xt*X + lamb*I))*Xt)*y
	
	return beta

loans = []

def read_data():

	f = open('../loans_ridge.csv', 'r')
	reader = csv.reader(f)
	header = next(reader)

	X = []
	Y = []
	testing_X = []
	testing_Y = []

	n_headings = len(header)
	num_obs = 0

	# Use dictionary representation such that its easier to generate variables later
	integers = ['id','loan_amount', 'repayment_term']
	floats = ['terms_disbursal_amount','funded_amount']
	exclude = set(string.punctuation)
	for i, row in enumerate(reader):
		# TODO: read in file, split sample into train and test
		dictionary = {}
		for heading in range(n_headings-1):
			if header[heading] == 'paid_amount':
				if row[heading] == 'NA':
					dictionary.update({header[heading]:0})
				else:
					dictionary.update({header[heading]:float(row[heading])})
			elif header[heading] in floats :
				dictionary.update({header[heading]:float(row[heading])})
			elif header[heading] in integers:
				dictionary.update({header[heading]:int(row[heading])})
			elif header[heading] == 'description':
				st = ''.join(ch for ch in row[heading] if ch not in exclude).lower()
				dictionary.update({header[heading]:set(st.split())})
			else:
				dictionary.update({header[heading]:row[heading]})
			
		if random.randint(0,1):
			X.append(dictionary)
			Y.append(int(row[n_headings-1]))
		else:
			testing_X.append(dictionary)
			testing_Y.append(int(row[n_headings-1]))
		num_obs += 1

	f.close()

	return X, Y, testing_X, testing_Y

# generate our 30 variables train_x and text_x
def generate_variables(train_x, test_x):
	new_train_x = []
	new_test_x = []
	for row in train_x:
		new_train_x.append(make_new_x(row))
	for row in test_x:
		new_test_x.append(make_new_x(row))
	return new_train_x, new_test_x

# Helper function to make new data x
# x is a dictionary of attributes
def make_new_x(x):
	new_x = []
	new_x.append(x['loan_amount'])
	new_x.append(x['funded_amount'])
	return new_x

# TODO: compute accuracy of your estimates
def calculate_mse(data, label, beta):
	sqr_err = 0
	a, b = data.shape
	for i in range(len(data)):
		Xi = data[i]
		yi = label[i]
		estimate = Xi*beta
		sqr_err += (yi - estimate)**2
	mse = sqr_err/a
	return mse

def make_graph(x, y, title, x_lab, y_lab):
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	plt.show()

def main():
	train_x, train_y, test_x, test_y = read_data()
	print(train_x[5])
	print(train_y[5])
	print(test_x[100])
	print(test_y[100])
	train_x, test_x = generate_variables(train_x, test_x)
	train_x  = np.matrix(train_x)
	train_y  = np.array(train_y)
	test_x = np.matrix(test_x)
	test_y = np.array(test_y)

	train_x_demeaned  = train_x  - np.mean(train_x, 0)
	train_y_demeaned  = train_y  - np.mean(train_y)
	test_x_demeaned = test_x - np.mean(test_x, 0)
	test_y_demeaned = test_y - np.mean(test_y)
	y1 = len(train_y_demeaned)
	y2 = len(test_y_demeaned)
	train_y_demeaned = np.reshape(train_y_demeaned, (y1, 1))
	test_y_demeaned = np.reshape(test_y_demeaned, (y2, 1))

	beta = ridge(train_x_demeaned, train_y_demeaned, 0) # (Modify the last parameter to change lambda)
	print(beta)
	mse = calculate_mse(train_x_demeaned, train_y_demeaned, beta)
	mse = mse.tolist()[0][0]
	print(mse)

	betas = []
	mses_train = []
	mses_test = []
	lambdas = []

	for i in range(10):
		beta = ridge(train_x_demeaned, train_y_demeaned, i)
		mse_train = calculate_mse(train_x_demeaned, train_y_demeaned, beta).tolist()[0][0]
		mse_test = calculate_mse(test_x_demeaned, test_y_demeaned, beta).tolist()[0][0]
		betas.append(beta)
		mses_train.append(mse_train)
		mses_test.append(mse_test)
		lambdas.append(i)
	
	# make_graph(lambdas, betas, 'lambda values vs. beta values', 'lambda', 'beta')
	make_graph(lambdas, mses_train, 'lambda values vs. mse of training data', 'lambda', 'mse')
	make_graph(lambdas, mses_test, 'lambda values vs. mse of testing data', 'lambda', 'mse')

main()