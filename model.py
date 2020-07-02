# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:03:17 2020

@author: dibya
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#read dataset
df = pd.read_csv(("D:\Data Science\Projects\Kaggle- Car price prediction of cardekho\car data.csv"))
df.head()
df.shape

#printing categorial features
print(df['Seller_Type'].unique()) #['Dealer' 'Individual']
print(df['Owner'].unique())   #[0 1 3]
print(df['Transmission'].unique())   #['Manual' 'Automatic']

#checking  missing or null values in dataset
df.isnull().sum()  # no null values

df.describe()
df.columns
#feature engineering year value
#subtracting from 2020 the current year value

#1. removing car name
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()

#2. creating new feature- adding current_year=2020
final_dataset['current_year']=2020
final_dataset.head()
final_dataset['no_of_years']=final_dataset['current_year']-final_dataset['Year']
final_dataset.head()

#dropping year and current year from dataset
final_dataset.drop(['Year','current_year'],axis=1,inplace=True)

#converting categorical features to dummies
final_dataset = pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()

#finding co-relation among inputs
final_dataset.corr()
sns.heatmap(final_dataset.corr(),annot=True)
sns.pairplot(final_dataset)

#seperating independent and dependent features
x = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]

x.head()
y.head()

#Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)

#plotting graph for feature importance
fet_imp = pd.Series(model.feature_importances_,index=x.columns)
fet_imp.plot(kind='barh')
plt.show()

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#model building
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

#hyperparametrs
#1. no of trees
n_estimators= [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

from sklearn.model_selection import RandomizedSearchCV
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

#performing randomised search
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train,y_train)

rf_random.best_params_ # checking best parameters

#validation prediction
predictions=rf_random.predict(x_test)
sns.distplot(y_test-predictions)  #y test -predictions
plt.scatter(y_test,predictions)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)












