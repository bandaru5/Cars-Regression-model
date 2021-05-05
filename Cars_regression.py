# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:00:02 2021

@author: DELL-INSPIRION
"""
#==============================
# Predicting price for pre-owned cars
#===================================

#importing required libraries

import pandas as pd

import numpy as np

import seaborn as sns

# Setting dimensions for plot

sns.set(rc={'figure.figsize':(11.7,8.27)})

# import the data set

cars_data = pd.read_csv('cars_sampled.csv')

# copy of the data

cars = cars_data.copy()

# Structure of the data
cars.describe()

#to see all the columns in describe function, and round the values to 3 places

pd.set_option('display.float_format', lambda x: '%.3f' % x)
cars.describe()

pd.set_option('display.max_columns', 500)
cars.describe()

#drop the unwanted columns
col = ['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
cars= cars.drop(columns=col,axis=1) 

# Removing the duplicate records
cars.drop_duplicates(keep='first', inplace=True)
#470 duplicate records have been removed

#=============================
# Data Cleaning
#==============================

# Missing values in each column

cars.isnull().sum()

# Variable yearofregistration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)

sum(cars['yearOfRegistration']<1950)

sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)

# we have chose between 1950 and 2018

# Variable price

price_count = cars['price'].value_counts().sort_index()

sns.distplot(cars['price'])
cars.describe()
sns.boxplot(cars['price'])

sum(cars['price']>150000)

sum(cars['price']<100)

# we work woth 100-150000 range



#variable powerps

power_count = cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#range 10-500

#============================
# Working range of data
#============================

cars = cars[(cars.yearOfRegistration <= 2018)
        & (cars.yearOfRegistration >=1950)
        & (cars.price<=150000)
        & (cars.price>=100)
        & (cars.powerPS>=10)
        & (cars.powerPS<=500)
        ]

#6700 records dropped

#combine the years and month of regestration

cars['monthOfRegistration']/=12


# create new variable age and add month and year of registration to it

cars['Age'] = (2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']= round(cars['Age'],2)
cars['Age'].describe()

#drop year and month of the registraoin as we have added them to age variable

cars = cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)


# Visualizing after some cleaning

# Age

sns.distplot(cars['Age'])
sns.boxplot(cars['Age'])

# price

sns.distplot(cars['price'])
sns.boxplot(cars['price'])

#powerPS

sns.distplot(cars['powerPS'])
sns.boxplot(cars['powerPS'])

# Visualizing after narrowing range

#Age Price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)

#PowerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)

# Categorial variables

# Seller variable

cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count', normalize=True)
sns.countplot(cars['seller'])

# only one commercial seller type =  insignificant

# offerType variable
cars['offerType'].value_counts()

sns.countplot(cars['offerType'])
 
# all have offer  - insignificant

# abtest variable

cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count', normalize=True)
sns.countplot(cars['abtest'])
# equally distributed
# 50-50 distribution - insignificant

# vehicletype variable

cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count', normalize=True)
sns.countplot(cars['vehicleType'])
sns.boxplot(x='vehicleType',y='price',data=cars)

# each ahs different price range, so it affects the price

# gearbox variable

cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count', normalize=True)
sns.countplot(cars['gearbox'])
sns.boxplot(x='gearbox',y='price',data=cars)
 # gearbox affects the price
 
 # model variable
 
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count', normalize=True)
sns.countplot(cars['model'])
sns.boxplot(x='model',y='price',data=cars)
# so many models and it affects the price

#kilometer variable
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count', normalize=True)
sns.countplot(cars['kilometer'])
sns.boxplot(x='kilometer',y='price',data=cars)
#affects the price

#fueltype variable
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count', normalize=True)
sns.countplot(cars['fuelType'])
sns.boxplot(x='fuelType',y='price',data=cars)
#affects price

#brand variable
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count', normalize=True)
sns.countplot(cars['brand'])
sns.boxplot(x='brand',y='price',data=cars)
#affects the price

#notRepairedDamage
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count', normalize=True)
sns.countplot(cars['notRepairedDamage'])
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
#affects the price

#==============================================
# Removing insignificant variables
#==============================================

col = ['seller','offerType','abtest']
cars = cars.drop(columns=col,axis=1)
cars_copy=cars.copy()
 


#================
#Correlation
#=================

cars_select1 = cars.select_dtypes(exclude=[object])
correlation = cars_select1.corr()
print(correlation)



#==================================
# Omitting the missing values
#==================================

cars_omit = cars.dropna(axis=0)

# converting the categorial variables to dummy variables

cars_omit = pd.get_dummies(cars_omit,drop_first=True)

#====================================

#Importing the necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#====================================
# Build the model with omitted data
#=========================

#Separating the x and y variable

x1 = cars_omit.drop(['price'],axis='columns',inplace=False)
y1 =  cars_omit['price']
y1 = np.log(y1)  #transforming into log to get bell shaped curve
y1.hist()

#splitting the data

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)


#Baseline model for omitted data

base_pred = np.mean(y_test)
print(base_pred)

#repeat the same value till the length of the test data
base_pred = np.repeat(base_pred,len(y_test))

# finding the RMSE

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)


#===============================
# Linear Regression with omitted data
#==============================

#setting the intercept as true
lgr = LinearRegression(fit_intercept=True)

#Model
lin_model1 = lgr.fit(x_train,y_train)

#Prediction
cars_prediction_lin = lgr.predict(x_test)

#RMSE from the predicted value
lin_rmse1 = np.sqrt(mean_squared_error(y_test,cars_prediction_lin))
print(lin_rmse1)

#R squared values
r2_lin_test = lin_model1.score(x_test,y_test)
r2_lin_train = lin_model1.score(x_train,y_train)

print(r2_lin_test)
print(r2_lin_train)

# Residuals and the plots

residuals_lin = y_test-cars_prediction_lin
sns.regplot(x=cars_prediction_lin,y=residuals_lin,scatter=True,fit_reg=False)
residuals_lin.describe()



#=======================================
# Random Forest Method with omitted data
#======================================

rf = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10
                          ,min_samples_leaf=4, random_state=1)

#Fit the model
rf_model = rf.fit(x_train,y_train)

#prediction
cars_prediction_rf = rf.predict(x_test)

#RMSE computing
rf_rmse = np.sqrt(mean_squared_error(y_test,cars_prediction_rf))
print(rf_rmse)

# R squared values of test and train data
r2_rf_test = rf_model.score(x_test,y_test)
r2_rf_train= rf_model.score(x_train,y_train)
print(r2_rf_test,r2_rf_train)



#=============================================================================================
# Model Building with Imputed data
#==============================================================================================

cars_imputed = cars.apply(lambda x: x.fillna(x.median()) if x.dtype =='float' else x.fillna(x.value_counts().index[0]))

cars_imputed.isnull().sum()

# converting the categorial variable into dummies

cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)



#====================================
# Build the model with imputed data
#=========================

#Separating the x and y variable

x2 = cars_imputed.drop(['price'],axis='columns',inplace=False)
y2 =  cars_imputed['price']
y2 = np.log(y2)  #transforming into log to get bell shaped curve
y2.hist()

#splitting the data

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=3)


#Baseline model for imputed data

base_pred = np.mean(y_test2)
print(base_pred)

#repeat the same value till the length of the test data
base_pred = np.repeat(base_pred,len(y_test2))

# finding the RMSE

base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test2,base_pred))
print(base_root_mean_square_error_imputed)


#===============================
# Linear Regression with imputed data
#==============================

#setting the intercept as true
lgr2 = LinearRegression(fit_intercept=True)

#Model
lin_model2 = lgr2.fit(x_train2,y_train2)

#Prediction
cars_prediction_lin_imputed = lgr2.predict(x_test2)

#RMSE from the predicted value
lin_rmse2 = np.sqrt(mean_squared_error(y_test2,cars_prediction_lin_imputed))
print(lin_rmse2)

#R squared values
r2_lin_test2 = lin_model2.score(x_test2,y_test2)
r2_lin_train2 = lin_model2.score(x_train2,y_train2)

print(r2_lin_test2)
print(r2_lin_train2)

# Residuals and the plots

#===============================================
#Random Forest Method with omitted data
#===============================================

rf2 = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10
                          ,min_samples_leaf=4, random_state=1)

#Fit the model
rf_model2 = rf2.fit(x_train2,y_train2)

#prediction
cars_prediction_rf2 = rf2.predict(x_test2)

#RMSE computing
rf_rmse2 = np.sqrt(mean_squared_error(y_test2,cars_prediction_rf2))
print(rf_rmse2)

# R squared values of test and train data
r2_rf_test2 = rf_model2.score(x_test2,y_test2)
r2_rf_train2= rf_model2.score(x_train2,y_train2)
print(r2_rf_test2,r2_rf_train2)








































































































































































