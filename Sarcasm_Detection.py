import csv
import random
import sklearn
import numpy as np
from os import listdir
from textblob import TextBlob
from os.path import isfile, join
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import train_test_split

'''global variables'''
testbody = []
body = []
author = []
score = []
id = []
subreddit = []
sarcasm_tag = []

''' function:: train_model

purpose of this function is to train the Gaussian Naive Bayesian Model using the 
training features and training labels so that it can assign a score to the upcoming
sentences tweeted by users

it is also used to test and predict the accuracy of the system'''
def train_model(training_features, training_labels, model, path):
	line = 0
	test = [] 
	testing_features = []
	with open('reddit_test.csv', 'rt') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			line = line + 1
			features = []
			
			score_2p = 0
			score_2s = 0
			word = 0
			for z in range(len(row['body'])-1):
				word = word + 1
				n_gram = row['body'][z:z+2]
				gram_2 = TextBlob(n_gram)
				try:
					score_2p = score_2p + gram_2.sentiment.polarity
					score_2s = score_2s + gram_2.sentiment.subjectivity
				except:
					word = word - 1
			
			score_2p = score_2p/(float(word)+0.0000001)
			score_2s = score_2s/(float(word)+0.0000001)
		
			score_3p = 0
			score_3s = 0
			word = 0
			for z in range(len(row['body'])-2):
				word = word + 1
				n_gram = row['body'][z:z+3]
				gram_3 = TextBlob(n_gram)
				try:
					score_3p = score_3p + gram_3.sentiment.polarity
					score_3s = score_3s + gram_3.sentiment.subjectivity
				except:
					word = word - 1
			
			score_3p = score_3p/(float(word)+0.0000001)
			score_3s = score_3s/(float(word)+0.0000001)
			
			testbody.append(row['body'])
			#print(line, row['body'])
			sentence = TextBlob(row['body'])
			#features.append(row['body'])
			#features.append(row['author'])
			#features.append(int(row['score']))
			#features.append(row['subreddit'])
			features.append(sentence.sentiment.polarity)
			features.append(sentence.sentiment.subjectivity)
			#features.append(sentence.ngrams(n=2))
			features.append(score_2p)
			features.append(score_2s)
			#features.append(sentence.ngrams(n=3))
			features.append(score_3p)
			features.append(score_3s)
			test.append(features)
			#print(features)
			
	model.fit(training_features, training_labels)
	#print(model.predict(testing_features))
	#print(model.predict(test))

	testing_labels = model.predict(test)
	return [testing_labels, model.predict_proba(test)]

'''function:: predict_sentence

Given a sentence, predict the sarcasm score for a particular sentence/tweet'''
def predict_sentence(model, sentence):
	feature = []
	feature.append(random.randint(-100, 100))
	feature.append(TextBlob(sentence).sentiment.polarity)
	feature.append(TextBlob(sentence).sentiment.subjectivity)
	print(model.predict(feature))
	print(model.predict_proba(feature))

'''function:: print_test-

prints all the passed test cases'''
def print_test(model, testing_labels, predicted_probability):
	for i in range(len(testing_labels)):
		if ( testing_labels[i] == 'yes' ):
			print(testbody[i])
			print('----------')
			
'''function:: print_test_passed

prints all the passed test cases with their score'''
def print_test_passed(model, testing_labels, predicted_probability):
	for i in range(len(testing_labels)):
		if ( testing_labels[i] == 'yes' ):
			print(testbody[i], predicted_probability[i][1]*100)
	
'''function:: fetch_data

fetches the data from the csv file into training_labels and training_features'''
def fetch_data(path):
	line = 0
	with open('reddit_training.csv', 'rt') as csvfile:
		training_features = []
		training_labels = []
		
		reader = csv.DictReader(csvfile)
		
		writtenfile = open('training_features.csv', 'w')
		'''features'''
		fieldnames = ('id', 'body', 'author', 'score', 'subreddit', 'sarcasm_tag', 'polarity', 'subjectivity', 'n_gram2', 'n_gram2_polarity', 'n_gram2_subjectivity', 'n_gram3', 'n_gram3_polarity', 'n_gram3_subjectivity')
		writer = csv.DictWriter(writtenfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in reader:
			body.append(row['body'])
			author.append(row['author'])
			score.append(row['score'])
			id.append(row['id'])
			subreddit.append(row['subreddit'])
			sarcasm_tag.append(row['sarcasm_tag'])
			
			'''n_gram (n=2) score assignment'''
			score_2p = 0
			score_2s = 0
			word = 0
			for z in range(len(row['body'])-1):
				word = word + 1
				n_gram = row['body'][z:z+2]
				gram_2 = TextBlob(n_gram)
				try:
					score_2p = score_2p + gram_2.sentiment.polarity
					score_2s = score_2s + gram_2.sentiment.subjectivity
				except:
					word = word - 1
			
			score_2p = score_2p/(float(word)+0.00000001)
			score_2s = score_2s/(float(word)+0.00000001)
		
			'''n_gram (n=3) score assignment'''
			score_3p = 0
			score_3s = 0
			word = 0
			for z in range(len(row['body'])-2):
				word = word + 1
				n_gram = row['body'][z:z+3]
				gram_3 = TextBlob(n_gram)
				try:
					score_3p = score_3p + gram_3.sentiment.polarity
					score_3s = score_3s + gram_3.sentiment.subjectivity
				except:
					word = word - 1
		
			score_3p = score_3p/(float(word)+0.0000000001)
			score_3s = score_3s/(float(word)+0.0000000001)
			
			'''writing to csv file'''
			line = line + 1
			features = []
			sentence = TextBlob(row['body'])
			# print(line)
			writer.writerow({ 'id': row['id'],
			                  'body': row['body'],
							  'author': row['author'],
							  'score': row['score'],
							  'subreddit': row['subreddit'],
							  'sarcasm_tag': row['sarcasm_tag'],
							  'polarity': sentence.sentiment.polarity,
							  'subjectivity': sentence.sentiment.subjectivity,
							  'n_gram2': sentence.ngrams(n=2),
							  'n_gram2_polarity': score_2p,
							  'n_gram2_subjectivity': score_2s,
							  'n_gram3': sentence.ngrams(n=3),
							  'n_gram3_polarity': score_3p,
							  'n_gram3_subjectivity': score_3s
			})
			
			'''appending training features and labels'''
			training_labels.append(row['sarcasm_tag'])
			
			#features.append(row['body'])
			#features.append(row['author'])
			#features.append(int(row['score'])

			#features.append(row['subreddit'])
			features.append(sentence.sentiment.polarity)
			features.append(sentence.sentiment.subjectivity)
			#features.append(sentence.ngrams(n=2))
			features.append(score_2p)
			features.append(score_2s)
			#features.append(sentence.ngrams(n=3))
			features.append(score_3p)
			features.append(score_3s)
			training_features.append(features)
			
		return [training_features, training_labels]

'''main:: training model'''
#training_labels = []
#training_features = []
#model = sklearn.svm.SVC()
model = GaussianNB()
[training_features, training_labels] = fetch_data('')

'''for i in range(1):
	X_test = []
	y_test = []
	X_train = []
	y_train = []
	X_test = sarcasm_tag
	X_train = training_features
	# X_train, X_test, y_train, y_test = train_test_split(training_features, sarcasm_tag, test_size=0.3, random_state=0)
	while (len(y_train) < len(body)*0.3):
		length = len(X_train)
		rand = random.randint(0, length-1)
		y_train.append(X_train[rand])
		y_test.append(X_test[rand])
		del X_train[rand]
		del X_test[rand]
	
	model.fit(X_train, X_test)
	y_labels = model.predict(y_train)
	
	invalid = 0
	correct = 0
	for j in range(len(y_train)):
		if ( y_train[j]==y_labels[j] ):
			correct = correct + 1
		else:
			invalid = invalid + 1
		
	print("Accuracy: ", (correct/(correct+invalid)))'''
	
'''k-fold training k=10'''
baseline = 0
num_folds = 10
subset_size = len(training_features)/num_folds
for i in range(num_folds):
	testing_this_round = training_features[int(i*subset_size):][:int(subset_size)]
	testing_labels_this_round = training_labels[int(i*subset_size):][:int(subset_size)]
	training_this_round = training_features[:int(i*subset_size)] + training_features[int((i+1)*subset_size):]
	training_labels_this_round = training_labels[:int(i*subset_size)] + training_labels[int((i+1)*subset_size):]
	
	model.fit(training_this_round, training_labels_this_round)
	predicted_labels_this_round = model.predict(testing_this_round)
	
	invalid = 0
	correct = 0
	for j in range(len(predicted_labels_this_round)):
		if ( predicted_labels_this_round[j]==testing_labels_this_round[j] ):
			correct = correct + 1
		else:
			invalid = invalid + 1
		
	print("Accuracy: ", (correct/float(correct+invalid)))
	baseline = baseline + (correct/float(correct+invalid))

baseline = baseline/float(10)
print("Baseline Accuracy: ", baseline)
	
'''testing model'''
[testing_labels, predicted_probability] = train_model(training_features, training_labels, model, '')
print_test_passed(model, testing_labels, predicted_probability)
print_test(model, testing_labels, predicted_probability)