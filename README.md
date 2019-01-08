# myspskills
to predict spoken languages proficiency skills - a Machine Learning algorithm 

This python package proposes a comprehensive method for predicting second language proficiency based on; an acoustic model, a language model, and linguistic cognitive ability in a high-entropy free speech. Our method is based on an idea introduced by Kroll, J. F., Michael, E., Tokowicz, N., & Dufour, R. "The development of lexical fluency in a second language" (Juornal of Second Language Research 2002). Our proposed method for predicting second language proficiency uses as input learner’s acoustic features and linguistic cognition aptitude data. The method produced promising results with the predictive power as high as 70 %. Acoustic features and linguistic cognitive ability are measured through speaking tasks, which include: reading-loud tasks, introducing what being seen task, listening and summarizing tasks, and free speech tasks. Each type of the tasks is related to a different linguistic function in the brain and delivers in the spoken mode. After measuring the learner’s acoustic features and linguistic factors, the result is fed as input for a machine learning model, which makes predictions for the corresponding language proficiency level. 

In training the linguistic proficiency classifier, we used Logistic Regression (LR), Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), Support Vector Machines (SVM). This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.

For input data set in our model, we had 697 samples, audio files recorded from non-native speakers. Our classifier showed an accuracy around 70 % in predicting spoken language proficiency level. Among the models, Gaussian Naive Bayes and Linear Discriminant Analysis models produced the best predictive power.

                                                    ----------------------
                                                    How to use the package
                                                    ----------------------
Please download the folder called "database". The file includes the acoustic and linguistic data. Three files of four include data from a trainee. You will need to develop those files for your trainees.
                                                ---------------------------------------
                                    please contact me at sabahi.s@mysol-gc.jp if you need me to help out               
                                                ---------------------------------------
You need to install Python 3 and the following libraries:  
1- pandas
2- scipy
3- sklearn
  3-1 sklearn.learn.metrics 
  3-2 sklearn.linear_model
  3-3 skearn.tree 
  3-4 sklearn.neighbors 
  3-5 sklearn.discriminant_analysis 
  3-6 sklearn.naive_bayes
  3-7 sklearn.svm import SVC
  3-8 sklearn.feature_selection 
  3-9 sklearn.decomposition
  3-10 sklearn.ensemble
  3-11 sklearn.datasets.samples_generator 
4- importlib
5- numpy
6- pickle
7- matplotlib.pyplot 
