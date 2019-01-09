# Load libraries
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
import scipy
import scipy.stats
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets.samples_generator import make_blobs
import csv
import importlib
import importlib.util
import numpy
import pickle
import sklearn.neighbors.typedefs
import matplotlib.pyplot as plt


def myml():
	pathy = input("Enter the dataset directory path: ")
	pa1=pathy+"/"+"MLENGLISH.csv"
	pa2=pathy+"/"+"stats.csv"
	pa3=pathy+"/"+"datacorrP.csv"
	pa4=pathy+"/"+"datanewchi.csv"
	pa5=pathy+"/"+"datanewchi33.csv"
	pa6=pathy+"/"+"datanewchi33.csv"
	pa7=pathy+"/"+"datanewchi44.csv"
	ns=['average_syll_pause_duration','No._long_pause','speaking_time','ave_No._of_words_in_minutes','articulation_rate','No._words_in_minutes','formants_index','f0_index','f0_quantile_25_index',
								  'f0_quantile_50_index','f0_quantile_75_index','f0_std','f0_max','f0_min','No._detected_vowel','perc%._correct_vowel','(f2/f1)_mean','(f2/f1)_std',
									'no._of_words','no._of_pauses','intonation_index',
						'(voiced_syll_count)/(no_of_pause)','TOEFL_Scale_Score','Prosody_CFER_Score','Score_Shannon_index','CEFR_Scale_Score','speaking_rate']

	dataset = pandas.read_csv(pa1, names=ns)    
		 
	#shape
	print(dataset.shape)

	# descriptions
	stat=dataset.describe()

	stat.to_csv(pa2)
	print(dataset.describe())

	names= ['avepauseduratin','avelongpause','speakingtot','avenumberofwords','articulationrate','inpro','f1norm','mr','q25',
								  'q50','q75','std','fmax','fmin','vowelinx1','vowelinx2','formantmean','formantstd','nuofwrds','npause','ins',
								  'fillerratio','xx','xxx','totsco','xxban','speakingrate']

	dataset = pandas.read_csv(pa1, names=names)    

	# class distribution
	print(dataset.groupby('speakingrate').size())
	print(dataset.groupby('articulationrate').size())
	print(dataset.groupby('vowelinx2').size())
	print(dataset.groupby('formantmean').size())
	print(dataset.groupby('formantstd').size())
	print(dataset.groupby('xxban').size())

	# box and whisker plots
	dataset.plot(kind='box', subplots=True, layout=(8,4), sharex=False, sharey=False)
	plt.show()

	# histograms
	dataset.hist()
	plt.show()

	# Correlation
	df = pandas.read_csv(pa1,
						 names = names)
	corMx=df.drop(['avenumberofwords','f1norm','inpro','q25','q75','vowelinx1','nuofwrds','npause','xx','xxx','totsco','xxban'], axis=1).corr(method='pearson')
	print(corMx)
	s = corMx.unstack()
	so = s.sort_values(kind="quicksort")
	corMx.to_csv(pa3, index = False)
	print(so)

	# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
	newMLdataset=dataset.drop(['avenumberofwords','f1norm','inpro','q25','q75','vowelinx1','nuofwrds','npause','xx','totsco','xxban','speakingrate','fillerratio'], axis=1)
	newMLdataset.to_csv(pa4, header=False,index = False)
	namess=nms = ['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
								  'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins',
								  'xxx']
	df1 = pandas.read_csv(pa4,
							names = namess)

	scatter_matrix(df1,alpha=0.2) # scatter plot matrix
	plt.show()
	print(df1.shape)

	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	array=numpy.absolute(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)

	# feature extraction
	test = SelectKBest(score_func=chi2, k=10)
	fit = test.fit(X, Y)

	# summarize scores
	numpy.set_printoptions(precision=3)
	print(fit.scores_)
	features = fit.transform(X)

	# summarize selected features
	print(features[0:11,:])

	# Feature Extraction with RFE
	namess = nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)


	# feature extraction
	model = LogisticRegression(solver='lbfgs',max_iter=40000, multi_class='auto')
	rfe = RFE(model, 10)
	fit = rfe.fit(X, Y)
	print("Num Features: %d" % fit.n_features_)
	print("Selected Features: %s" % fit.support_)
	print("Feature Ranking: %s" % fit.ranking_)

	# Feature Extraction with PCA
	namess =nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)

	# feature extraction
	pca = PCA(n_components=3)
	fit = pca.fit(X)

	# summarize components
	print("Explained Variance: %s" % fit.explained_variance_ratio_)
	print(fit.components_)

	# Feature Importance with Extra Trees Classifier
	namess = nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)

	# feature extraction
	model = ExtraTreesClassifier()
	model.fit(X, Y)
	print(model.feature_importances_)

	# Split-out validation dataset
	namess = nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression(solver='lbfgs',max_iter=10000, multi_class='auto'))) 
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	# Compare Algorithms
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

	# Make predictions on validation dataset
	knn = DecisionTreeClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))

	# Feature Extraction with RFE
	namess = nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)

	# feature extraction
	model = LogisticRegression(solver='lbfgs',max_iter=40000, multi_class='auto')
	rfe = RFE(model, 10)
	fit = rfe.fit(X, Y)
	# save the model to disk
	filename = 'RFE_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Feature Extraction with PCA
	namess =nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)

	# feature extraction
	pca = PCA(n_components=3)
	fit = pca.fit(X)
	# save the model to disk
	filename = 'PCA_model.sav'
	pickle.dump(model, open(filename, 'wb'))


	# Feature Importance with Extra Trees Classifier
	namess = nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)

	# feature extraction
	model = ExtraTreesClassifier()
	model.fit(X, Y)
	# save the model to disk
	filename = 'ETC_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Split-out validation dataset
	namess =nms
	df1 = pandas.read_csv(pa4,
							names = namess)
										   
	arrayy= df1.values
	array = df1.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	X = array[:,0:13]
	Y = arrayy[:,13]
	Y=Y.astype(str)
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression(solver='lbfgs',max_iter=10000, multi_class='auto'))) 
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	model = LinearDiscriminantAnalysis()
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
		
	plt.scatter(Y_validation, predictions)
	plt.xlabel('Y_validation')
	plt.ylabel('Predictions')	
	plt.show()
		
	# Fit the model on 33%
	model = LogisticRegression(solver='lbfgs',max_iter=10000, multi_class='auto')
	model.fit(X_train, Y_train)
	# save the model to disk
	filename = 'LR_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Fit the model on 33%
	model = LinearDiscriminantAnalysis()
	model.fit(X_train, Y_train)
	# save the model to disk
	filename = 'LDA_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Fit the model on 33%
	model = KNeighborsClassifier()
	model.fit(X_train, Y_train)
	# save the model to disk
	filename = 'KNN_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Fit the model on 33%
	model = DecisionTreeClassifier()
	model.fit(X_train, Y_train)
	# save the model to disk
	filename = 'CART_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Fit the model on 33%
	model = GaussianNB()
	model.fit(X_train, Y_train)
	# save the model to disk
	filename = 'NB_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# Fit the model on 33%
	model = SVC()
	model.fit(X_train, Y_train)
	# save the model to disk
	filename = 'SVN_model.sav'
	pickle.dump(model, open(filename, 'wb'))


	namess =nms
	df2= pandas.read_csv(pa5,
							names = namess)

	array = df2.drop(['xxx'],axis=1).values
	array=numpy.log(array)
	XT = array[:,0:13]
	YT = model.predict(XT)

	filename = 'LR_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	result = loaded_model.score(XT, YT)
	print(result)


	filename = 'LR_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	namess = nms

	print("#########################################################################")
	print("predicting fluency level of spoken languages:")

	df3 = pandas.read_csv(pa6,
							names = namess)
	df33=df3.drop(['xxx'], axis=1)

	array = df33.values
	array=numpy.log(array)
	x = array[:,0:13]
	model = LogisticRegression(solver='lbfgs',max_iter=10000, multi_class='auto')
	model.fit(X_train, Y_train)
	predictions = model.predict(x)
	print("60% accuracy    ",predictions)

	model = DecisionTreeClassifier()
	model.fit(X_train, Y_train)
	model.fit(X_train, Y_train)
	predictions = model.predict(x)
	print("55% accuracy    ",predictions)

	model = GaussianNB()
	model.fit(X_train, Y_train)
	predictions = model.predict(x)
	print("65% accuracy    ",predictions)

	model = LinearDiscriminantAnalysis()
	model.fit(X_train, Y_train)
	predictions = model.predict(x)
	print("70% accuracy    ",predictions)
	#y = model.predict(x)
	#print(y)

	nsns=['average_syll_pause_duration','No._long_pause','speaking_time','ave_No._of_words_in_minutes','articulation_rate','No._words_in_minutes','formants_index','f0_index','f0_quantile_25_index',
								  'f0_quantile_50_index','f0_quantile_75_index','f0_std','f0_max','f0_min','No._detected_vowel','perc%._correct_vowel','(f2/f1)_mean','(f2/f1)_std',
									'no._of_words','no._of_pauses','intonation_index',
						'(voiced_syll_count)/(no_of_pause)','TOEFL_Scale_Score','Score_Shannon_index','speaking_rate']
	dataframe = pandas.read_csv(pa2) 
	df55 = pandas.read_csv(pa7,names=nsns)
	dataframe=dataframe.values
	array = df55.values
	print("#########################################################################")
	print("")
	print("Compared to native speech, here are the prosodic features of your speech:")
	print("")
	for i in range(25):
		sl0=dataframe[4:7:1,i+1]
		score = array[0,i]
		he=scipy.stats.percentileofscore(sl0, score, kind='strict')
		if he==0:
			he=25
			dfout = "%s:\t %f (%s)" %  (nsns[i],he,"% percentile ")
			print(dfout) 
		elif he>=25 and he<=75:
			dfout = "%s:\t %f (%s)" % (nsns[i],he,"% percentile ")
			print(dfout) 
		else:
		   dfout = "%s:\t (%s)" % (nsns[i],":Out of Range")
		   print(dfout) 

if __name__ =="__main__":
    myml()
    input("press any key to terminate the program")
