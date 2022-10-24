import csv
from math import log
from collections import defaultdict, Counter
import random
from re import A, X
import numpy as np
import string
import matplotlib.pyplot as plt
import pycountry_convert as pc


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
			elif header[heading] == 'description_texts_en':
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
# x is a dictionary of attributes
def generate_variables(train_x, test_x):
	new_train_x = []
	new_test_x = []
	for row in train_x:
		new_train_x.append(make_new_x(row))
	for row in test_x:
		new_test_x.append(make_new_x(row))
	return new_train_x, new_test_x

# helper function to convert country into continents
def country_to_continent(country_name):
	country_continent_code = ''
	if(country_name == 'Congo (Dem. Rep.)' or country_name == 'Congo (Rep.)' or country_name == 'The Democratic Republic of the Congo'):
		country_continent_code == 'AF'
	elif (country_name == 'Myanmar (Burma)' or country_name == 'Cote D\'Ivoire' or country_name == 'Lao PDR' or country_name == 'Timor-Leste'):
		country_continent_code == 'AS'
	elif (country_name == 'Kosovo'):
		country_continent_code == 'EU'
	else:
		country_alpha2 = pc.country_name_to_country_alpha2(country_name)
		country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
	
	return country_continent_code

# helper function to create description dummies for word presence
def word_count(x):
	positive_words = set(['good','support', 'expertise', 'happy','responsible', 'trustful', 'honor', 'polite', 'leader', 'president', 'represent', 'nice', 'budget', 'active',
									'grow', 'improve', 'improving', 'great', 'help', 'motivate', 'first', 'grow', 'enjoy', 'photo', 'skill', 'dream', 'new', 'bless' ])
	negative_words = set(['quit','unreliable', 'four','five', 'six','seven','eight', 'hard', 'without', 'maybe', 'unstable', 'party', 
									'already', 'previous', 'supplement', 'another', 'additional', 'enough', 'increase', 'increasing', 'neither', 'nor',
									'little', 'continue', 'far', 'but', 'struggle', 'risk', 'lack'])
	supportive_personal_background = set(['married', 'husband', 'wife', 'children', 'help', 'house', 'town', 'job', 'income'])
	unsupportive_personal_background = set(['children', 'single mother', 'single', 'divorce', 'widow', 'rural', 'village', 'volatile', 'challenge', 'unemploy', 'young'])
	
	positive_word_presence = {i for i in x if any(j in i for j in positive_words)}
	positive_word_count = len(positive_word_presence)
	negative_word_presence = {i for i in x if any(j in i for j in negative_words)}
	negative_word_count = len(negative_word_presence)
	supportive_background_word_presence = {i for i in x if any(j in i for j in supportive_personal_background)}
	supportive_background_word_count = len(supportive_background_word_presence)
	unsupportive_background_word_presence = {i for i in x if any(j in i for j in unsupportive_personal_background)}
	unsupportive_background_word_count = len(unsupportive_background_word_presence)

	return positive_word_count, negative_word_count, supportive_background_word_count, unsupportive_background_word_count

# Helper function to make new data x
# x is a dictionary of attributes
def make_new_x(x):
	new_x = []
	# from description
	new_x.append(len(x['description_texts_en']))
	positive_word_count, negative_word_count, supportive_background_word_count, unsupportive_background_word_count = word_count(x['description_texts_en'])
	new_x.append(positive_word_count)
	new_x.append(negative_word_count)
	new_x.append(supportive_background_word_count)
	new_x.append(unsupportive_background_word_count)

	# factor variable
	new_x.append(x['borrowers_borrower_gender'] =='F')
	new_x.append(x['status'] == 'NA')
	new_x.append(x['status'] == 'paid')
	new_x.append(x['status'] == 'defaulted')
	new_x.append(x['activity'] == 'Farming')
	new_x.append(x['activity'] == 'Food')
	new_x.append(x['activity'] == 'Retail')
	new_x.append(x['activity'] == 'Electrical Goods' or x['activity'] == 'Movie Tapes & DVDs' or x['activity'] == 'Music Discs & Tapes' or x['activity'] =='Phone Accessories' or x['activity'] =='Computers'
				 or x['activity'] == 'Mobile Phones' or x['activity'] == 'Internet Cafe' or x['activity'] == 'Electronics Repair')
	new_x.append(x['activity'] == 'Animal Sales' or x['activity'] == 'Bicycle Sales' or x['activity'] == 'Charcoal Sales' or x['activity'] == 'Clothing Sales'
				 or x['activity'] == 'Cosmetics Sales' or x['activity'] == 'Decorations Sales' or x['activity'] == 'Electronics Sales' 
				 or x['activity'] == 'Food Production/Sales' or x['activity'] == 'Home Products Sales' or x['activity'] == 'Milk Sales' or x['activity'] == 'Paper Sales'
				 or x['activity'] == 'Personal Products Sales' or x['activity'] == 'Phone Use Sales' or x['activity'] == 'Plastics Sales' or x['activity'] == 'Shoe Sales'
				 or x['activity'] == 'Souvenir Sales' or x['activity'] == 'Sporting Good Sales' or x['activity'] == 'Timber Sales' or x['activity'] == 'Traveling Sales' or x['activity'] == 'Veterinary Sales')
	new_x.append(x['activity'] == 'Personal Medical Expenses' or x['activity'] == 'Wedding Expenses' or x['activity'] == 'Funeral Expenses' or x['activity'] == 'Personal Housing Expenses'
				 or x['activity'] == 'Higher education costs' or x['activity'] == 'Primary/secondary school costs')
	new_x.append(x['sector'] == 'Agriculture')
	new_x.append(x['sector'] == 'Food')
	new_x.append(x['sector'] == 'Retail')
	new_x.append(country_to_continent(x['location_country']) == 'AF')
	new_x.append(country_to_continent(x['location_country']) == 'AS')
	new_x.append(country_to_continent(x['location_country']) == 'NA')
	new_x.append(country_to_continent(x['location_country']) == 'SA')
	new_x.append(country_to_continent(x['location_country']) == 'EU' or country_to_continent(x['location_country']) == 'OC')
	# Top 5 frequently appearing counrties get their own variable!! 
	new_x.append(x['location_country'] == 'Philippines') #3605
	new_x.append(x['location_country'] == 'Kenya') #2254
	new_x.append(x['location_country'] == 'Peru') #1594
	new_x.append(x['location_country'] == 'Uganda') #827
	new_x.append(x['location_country'] == 'Cambodia') #770


	# continuous variable
	new_x.append(x['terms_disbursal_amount'])
	new_x.append(x['loan_amount'])
	new_x.append(x['repayment_term'])
	
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

def make_beta_graph(x, y, title, x_lab, y_lab):
	n, m = y.shape
	for i in range(m):
		bi = y[:,i]
		label = "beta" + str(i)
		plt.plot(x, bi, label = label)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(title)
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	plt.show()

def main():
	train_x, train_y, test_x, test_y = read_data()
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

	for i in range(1000):
		if i % 50 == 0:
			print("calculating " + str(i) + " as lambda")
		beta = ridge(train_x_demeaned, train_y_demeaned, i)
		mse_train = calculate_mse(train_x_demeaned, train_y_demeaned, beta).tolist()[0][0]
		mse_test = calculate_mse(test_x_demeaned, test_y_demeaned, beta).tolist()[0][0]
		betas.append(beta)
		mses_train.append(mse_train)
		mses_test.append(mse_test)
		lambdas.append(i)
	make_beta_graph(lambdas, np.squeeze(np.array(betas)), 'lambda values vs. beta values', 'lambda', 'beta')
	make_graph(lambdas, mses_train, 'lambda values vs. mse of training data', 'lambda', 'mse')
	make_graph(lambdas, mses_test, 'lambda values vs. mse of testing data', 'lambda', 'mse')

main()