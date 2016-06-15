# INF 552 Homework 1
# Change directory to wherever it is you downloaded the data
# cd C:\Users\hovo_\Google Drive\Classes\PhD\INF 552\Homework\Car Evaluation

# Invoke pandas library as pd
import pandas as pd

# Import CSV file and get some info on it
df = pd.read_csv('car.csv',header=0)
df.info()

# convert from pandas to numpy
car = df.values

# split the data to X,y which is attributes and output class (small y)
X,y = car[:,:6], car[:,6]
# This selects all rows then X holds first 6 column attributes and y the last column as class (i.e. 1 dimension array)

# make sure that all the values in numpy are int to avoid potential numpy problems
X,y = X.astype(int), y.astype(int)

# import scikit learn
import sklearn

# import the cross validation module from scikit learn
from sklearn import cross_validation

# split the data for 70% training and 30% for testing in scikit ML
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

# import ensemble from scikit
from sklearn import ensemble

# set up the Random Forest Classifier with 10 trees
clf = ensemble.RandomForestClassifier(n_estimators=10, verbose=1)

# fit the classifier with the training data
clf.fit(X_train,y_train)

# check the accuracy of the classification on the test data
clf.score(X_test,y_test) # got ~97% <-- sweeeet!

# see which how many features were used and which are the most important
clf.n_features_ # 6
clf.feature_importances_
# buying - 0.15459505
# maintenance - 0.14216624
# doors - 0.06176783
# persons - 0.24950191
# lug_boot - 0.09016715
# safety - 0.30180183
# looks like all 6 were used, but safety/persons are the most important and doors/lug_boot least

# plot the values to see what is going on
y_pred = clf.predict(X_test)

# import matplot so we can plot it
import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.show()

