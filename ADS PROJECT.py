#!/usr/bin/env python
# coding: utf-8

# ### Capstone Project for ADS Data Science Program on the Boston Housing Dataset from Github

# #### Importing the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# #### Importing the Dataset from an online source

# In[5]:


url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print(df.head())


# In[7]:


df.columns ## 


# In[9]:


X = df.drop(columns=['medv'])  # features (indepdendent variable)
y = df['medv']  # response / outcome / target (depdendent variable)


# In[11]:


X


# In[13]:


X.shape


# In[14]:


y.shape


# In[15]:


X.head()


# In[16]:


y.head()


# #### Conducting a Simple EDA to understand the Data

# In[17]:


df.info()


# In[18]:


df.isnull().sum()  # checking for missing values in the data


# In[19]:


corr_values = df.corr().round(2)
corr_values


# In[20]:


corr_values[abs(corr_values) > 0.6]


# #### Plotting a few graphs

# In[21]:


sns.set(style='whitegrid')
cols = ['lstat', 'indus', 'nox', 'rm', 'medv']
sns.pairplot(df[cols], size=2)


# In[22]:


df[cols].corr().values


# In[23]:


sns.heatmap(df[cols].corr().values, cbar=True, annot=True, yticklabels=cols, xticklabels=cols, cmap='viridis')


# In[ ]:


## Modelling Steps


# In[24]:


X.shape, y.shape


# In[25]:


# Split your data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)
X_train.shape, X_test.shape


# In[26]:


y_train.shape, y_test.shape


# In[27]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()   # creating an instance of the LinearRegression Class
LR.fit(X_train, y_train)   # training should be done on the training set.


# In[28]:


type(LR)


# In[29]:


LR.intercept_   # fetching the vlaue of intercept_ attribute  (w0 ot beta0)


# In[30]:


LR.coef_    # coef_ is also an attrbibute... (w1, w2 .... wj)


# In[34]:


coef_df = pd.DataFrame(LR.coef_, index=X.columns, columns=['LR_Coef'])
coef_df


# In[35]:


# fetch the R^2 of the model >> Evaluating the model
LR.score(X_test, y_test)  # validating our model's performance on the TEST SET.


# #### This means/conveys that our LR model has captured 71% of the variation (pattern) in the data... i.e. the model accounts for 70% variation in the data.

# In[37]:


y_pred = LR.predict(X_test)
y_pred


# In[38]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[39]:


# RMSE
np.sqrt(mean_squared_error(y_test, y_pred) )


# In[40]:


## Getting the predictions from the model??? 
# This single datapoint has to be reshaped as a ROW VECTOR
newX = np.array([7e-03, 15, 5, 1, 0.2, 7, 50, 7.5, 1, 310, 16, 400, 5]).reshape(1,-1)
newX.shape


# In[41]:


newX


# In[42]:


LR.predict(newX)


# #### Polynomial Regression

# In[44]:


from sklearn.preprocessing import PolynomialFeatures


# In[47]:


X1 = X_train.iloc[:, 5].values.reshape(-1,1)  # single input feature


# In[48]:


poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X1)   # get/create the polynomial features


# In[49]:


X1[:5]


# In[50]:


X_poly2[:5]


# In[54]:


poly2.get_feature_names_out()


# In[55]:


poly2.powers_


# In[56]:


LR = LinearRegression()
LR.fit(X1, y_train)  # train a Linear Reg model

PR = LinearRegression()
PR.fit(X_poly2, y_train)  # Feed the polynomial features, and do a Linear Reg ... train a Poly Reg model


# In[57]:


LR.score(X1, y_train)  # score on the traning set


# In[60]:


LR.score(X_test.iloc[:, 5].values.reshape(-1,1), y_test) # score on the test set


# In[61]:


PR.score(X_poly2, y_train)   # score on the traning set


# In[63]:


x_test1 = poly2.transform(X_test.iloc[:, 5].values.reshape(-1,1)) # you are transforming the test set as well
# basically means that we have to generate the polynomial features even for the test set!!!

x_test1


# In[64]:


PR.score(x_test1, y_test)  # definitely the Polynial REg has a better R^2


# In[66]:


poly3 = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly3.fit_transform(X_train)
X_poly.shape


# #### Comparison to other Regression Models

# In[68]:


### Let's try out a few more models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet

from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures


# In[69]:


### Create 6 baseline models!!
models = [] 
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso(random_state=100)))
models.append(('EN', ElasticNet(random_state=100)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor(random_state=200)))
models.append(('SVR', SVR(kernel='linear')))

models # we have created a list of models which will be tried!!


# In[70]:


for name, model in models:
    print(model)


# In[71]:


scores = []
names = []
for name, model in models:
    model.fit(X_train, y_train)  # training the estimator ...using the TRAIN sET
    names.append(name)
    sc = model.score(X_test, y_test)
    scores.append(sc) # R^2 for the estimator on the TEST SET
    
    print(name, '\t:train_score: ', np.round(model.score(X_train, y_train), 3) )
    print(name, '\t:test_score: ', np.round(sc, 3) )


# #### Model Selection and Evaluation

# In[73]:


## evaluate each model in turn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

scoring = 'r2' #OR 'neg_mean_squared_error'
results = []
names = []
n_splits = 5  # this is 5-fold splitting (K=5)

for name, model in models:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=105)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='r2') 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#     print(sorted(cv_results, reverse=True))


# In[74]:


results


# In[75]:


results_df = pd.DataFrame(results, index=names, \
                          columns='CV1 CV2 CV3 CV4 CV5'.split())
results_df['CV Mean'] = results_df.iloc[:,0:n_splits].mean(axis=1)
results_df['CV Std Dev'] = results_df.iloc[:,0:n_splits].std(axis=1)
results_df.sort_values(by='CV Mean', ascending=False)*100


# #### Standardizing the Data

# In[79]:


## Standardize the dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipelines = []  # i am creating a list of pipelines!!

# pipelines.append(('ScaledLR', make_pipleline(StandardScaler(), LinearRegression()  )   )   ) 

# we are appending this tuple ('name of the pipeline', Pipeline())

# Within the pipeline, each step has a name, tranformer/estimator class.... again this pair will be a tuple..
# and each of the steps (tuples) will be assembled as a list
# this list of "steps" is your final inout to the Pipeline class

# Pipeline()

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])   ))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())]))) 
pipelines


# In[80]:


scoring = 'r2' #'neg_mean_squared_error'
results = []
names = []
n_splits = 5

for name, model in pipelines:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=105)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring) 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
results_df = pd.DataFrame(results, index=names, \
                          columns='CV1 CV2 CV3 CV4 CV5'.split())
results_df['CV Mean'] = results_df.iloc[:,0:n_splits].mean(axis=1)
results_df['CV Std Dev'] = results_df.iloc[:,0:n_splits].std(axis=1)
results_df.sort_values(by='CV Mean', ascending=False)*100   


# #### Trying Out Hyper Parameter Tuning

# In[81]:


from sklearn import set_config
set_config(display="text")
knn_model = pipelines[3][1]
knn_model


# In[82]:


type(knn_model)


# In[83]:


set_config(display="diagram")
knn_model  # click on the diagram below to see the details of each step


# In[84]:


# HYPRER PARAM tuning the pipiline
from sklearn.model_selection import GridSearchCV   # or RandomizedSearchCV

k_values = np.array([1,3,5,7,9,11,13,15,16,17,18,19,21])
param_grid = {"KNN__n_neighbors": k_values}

kfold = KFold(n_splits=5, shuffle=True, random_state=105)
scoring = 'r2'

grid = GridSearchCV(estimator = knn_model, param_grid=param_grid, \
                    scoring=scoring, cv=kfold)  
grid_result = grid.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_,\
                             grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[85]:


grid_result.best_estimator_


# #### Final Model and Prediction

# In[86]:


final_knn_model = grid_result.best_estimator_


# In[87]:


final_knn_model.fit(X,y)  # training the model ONE LAST TIME one the ENTIRE Data


# In[88]:


final_knn_model.predict(newX)  # getting the prediction form the Final model


# In[ ]:




