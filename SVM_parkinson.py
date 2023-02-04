# -*- coding: utf-8 -*-
"""
@author: rosha
"""
#importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#data collection and analysis
#loading the data from csv file to a pandas dataframe
parkinsons_data=pd.read_csv("C:/Users/rosha/Parkinson_svm/parkinsons_dataset.csv")
#print first 5 rows of dataframe
parkinsons_data.head()
#number of rows and columns in the dataframe
parkinsons_data.shape #195 rows and 24 columns
#getting more info about datasets
parkinsons_data.info() #dtypes: float64(22), int64(1), object(1) memory usage: 36.7+ KB and no null values
#checking for missing values in each columns
parkinsons_data.isnull().sum()
#getting some statistical measures about the data
parkinsons_data.describe()
#distribution of target variable i.e. status
parkinsons_data['status'].value_counts()#out of 195 voice recordings 147 people have PD and 48 don't have PD
''' 0 means patients without PD and 1 means patients with PD'''
#grouping the data based on the target variable
parkinsons_data.groupby('status').mean() #if value is more than healthy else positive
parkinsons_data.groupby('status').median()
parkinsons_data.groupby('status').std()
#Data Preprocessing
'''here we will seperate features and status variable'''
X=parkinsons_data.drop(columns=['name','status'],axis=1)
Y=parkinsons_data['status']
#splitting our data in training and testing data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
print(X.shape,x_train.shape,x_test.shape)
#data standardization
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(x_train)
#training our model SVM
model_svm=svm.SVC(kernel='linear')
model_lgbm=LGBMClassifier()
model_xgb=XGBClassifier()
model_rfc= RandomForestClassifier()
#training the different models with training data
model_svm.fit(x_train,y_train)
model_lgbm.fit(x_train,y_train)
model_xgb.fit(x_train,y_train)
model_rfc.fit(x_train,y_train)
#model Evaluation svm
'''accuracy score on training data'''
x_train_prediction=model_svm.predict(x_train)
training_data_accuracy=accuracy_score(y_train, x_train_prediction)
print("Accuracy score of training data SVM:",training_data_accuracy)
'''accuracy score on test data'''
x_test_prediction_svm=model_svm.predict(x_test)
test_data_accuracy=accuracy_score(y_test, x_test_prediction_svm)
print("Accuracy score of training data SVM:",test_data_accuracy)
#model Evaluation of RandomForestClassifier
x_train_prediction_rfc=model_rfc.predict(x_train)
training_data_accuracy_rfc=accuracy_score(y_train, x_train_prediction_rfc)
print("Accuracy score of training data RFC:",training_data_accuracy_rfc)
'''accuracy score on test data'''
x_test_prediction_rfc=model_rfc.predict(x_test)
test_data_accuracy_rfc=accuracy_score(y_test, x_test_prediction_rfc)
print("Accuracy score of training data RFC:",test_data_accuracy_rfc)
#model Evaluation of XGboost
x_train_prediction_xgb=model_xgb.predict(x_train)
training_data_accuracy_xgb=accuracy_score(y_train, x_train_prediction_xgb)
print("Accuracy score of training data XGB:",training_data_accuracy_xgb)
'''accuracy score on test data'''
x_test_prediction_xgb=model_xgb.predict(x_test)
test_data_accuracy_xgb=accuracy_score(y_test, x_test_prediction_xgb)
print("Accuracy score of training data XGB:",test_data_accuracy_xgb)
#model Evalutaion LGBM
x_train_prediction_lgbm=model_lgbm.predict(x_train)
training_data_accuracy_lgbm=accuracy_score(y_train, x_train_prediction_lgbm)
print("Accuracy score of training data LGBM:",training_data_accuracy_lgbm)
'''accuracy score on test data'''
x_test_prediction_lgbm=model_lgbm.predict(x_test)
test_data_accuracy_lgbm=accuracy_score(y_test, x_test_prediction_lgbm)
print("Accuracy score of training data LGBM:",test_data_accuracy_lgbm)

#Building a Predictive System on basis of LGBM
input_data=(162.56800,198.34600,77.63000,0.00502,0.00003,0.00280,0.00253,0.00841,0.01791,0.16800,0.00793,0.01057,0.01799,0.02380,0.01170,25.67800,0.427785,0.723797,-6.635729,0.209866,1.957961,0.135242)
#changing input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the numpy array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#standardize the data
std_data=scaler.transform(input_data_reshaped)
prediction=model_lgbm.predict(std_data)
print(prediction)
if prediction[0]==0:
    print("The Person does not have parkinsons disease")
else:
    print("The person has parkinsons")
