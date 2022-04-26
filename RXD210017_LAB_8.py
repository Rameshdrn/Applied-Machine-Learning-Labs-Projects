#!/usr/bin/env python
# coding: utf-8

# # Lab-8 Template
# 
# Change title and notebook name to reflect your work.
# 
# Answer questions in the designated cells
# 
# Resources: 
# - https://github.com/slundberg/shap
# - H2O Explainability best practices: https://github.com/h2oai/h2o-tutorials/blob/master/best-practices/model-interpretability/interpreting_models.ipynb
# 

# ## Preparation
# 
# Use dataset provided in the eLearning

# In[1]:


#Extend cell width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd 

#Install shap package as needed:
#!pip uninstall numpy
#!pip uninstall numba
get_ipython().system('pip install shap==0.40.0')

import shap

import h2o
from h2o.estimators import H2OTargetEncoderEstimator

try:
    h2o.cluster().shutdown()
except:
    pass 


# In[2]:


#Limit to 3 threads and 8GB memory - modify as needed
h2o.init(nthreads=3, max_mem_size=8)


# ### Load data

# In[3]:


train = h2o.import_file('SBA_loans_train.csv')
test = h2o.import_file('SBA_loans_test.csv')


# In[4]:


print("Train shape:", train.shape)
print("Test shape:", test.shape)


# # Prepare Dataset
# 
# Prepare dataframe by encoding some columns as categorical.

# In[5]:


# Choose which columns to encode
cat_columns = ["City","State","Bank","BankState", "UrbanRural", "FranchiseCode",
               "NewExist", "RevLineCr","LowDoc", "Zip"]
response = "Defaulted"

train[cat_columns+[response]] = train[cat_columns+[response]].asfactor()
test[cat_columns+[response]] = test[cat_columns+[response]].asfactor()


# ## Question 1
# 
# Train H2O `H2OGradientBoostingEstimator` with parameters:
# ```
# nfolds=5,
# ntrees=500,
# stopping_rounds=5,
# stopping_metric='AUCPR',
# seed=1234,
# keep_cross_validation_predictions = False
# ```
# Display model performance on `test` dataset using `model_performance` function.

# In[6]:


test


# In[7]:


import h2o
from h2o.estimators import H2OGradientBoostingEstimator
pros_gbm = H2OGradientBoostingEstimator(nfolds=5,ntrees=500,stopping_rounds=5,stopping_metric='AUCPR',
                                        seed=1234,keep_cross_validation_predictions = False)
pros_gbm.train(x=cat_columns, y=response, training_frame=train)
pros_gbm.model_performance(test)


# ## Question 2
# 
# Use model from Question 1 to answer Q#2.
# 
# - Calculate and display permutation feature importance for the model using **test** dataset
# - What is most important feature?
# - Can you tell how feature is impacting (direction) probability? 

# In[8]:


pros_gbm.varimp(test)


# - "BankState" is the most important feature.
# - The element significance is determined by seeing the increment or decline in mistake when we permute the upsides of a component. In the event that permuting the qualities causes an enormous change in the blunder, it implies the element is significant for our model. The element "BankState" is influencing with the pace of 25.7% on the likelihood.

# ## Question 3
# 
# Calculate and plot PDP plot for `"UrbanRural","SBA_Appv","DisbursementGross"]`
# 
# You might find using following parameters useful: `nbins=52,figsize=(10, 10)` 
# 
# What is your conclusion for each of the variables?

# In[9]:


pros_gbm.partial_plot(train, cols=["UrbanRural","SBA_Appv","DisbursementGross"], nbins=52, figsize=(10,10))


# - The graph delineates that the urbanrural has a direct relationship with the mean reaction and is expanding in lockstep with it. The plot shows that sbappv and disbursementgross don't essentially affect the mean reaction.

# ## Question 4
# 
# Calculate and display summary plot of Shapley values. Use `test` dataset to calculate Shapley values.
# 
# - What is the most important feature based on Shapley values? 
# - Why "Zip" feature is not rated at the top when many of the obesrvations have significant high/low Shapley values? 

# In[10]:


pros_gbm.shap_summary_plot(train)


# - The most important feature based on Shapley values is "UrbanRural"
# - "Zip" feature is not rated at the top when many of the obesrvations have significant high/low Shapley values because it has various unique values that cannot decide the response variable.

# ## Question 5
# 
# Plot individual Shapley values plots for records 0,1 and 4 in test dataset, for the total of 3 plots.
# Explain each plot in terms of what are most influential  features and how they impact model prediction.

# In[11]:


pros_gbm.shap_explain_row_plot(train,row_index = 0)
pros_gbm.shap_explain_row_plot(train,row_index = 1)
pros_gbm.shap_explain_row_plot(train,row_index = 4)


# - Bankstate=FL, Bank=Business Loan center,LLC, UrbanRural=1 are the most influential factors for record 0 and will have a positive impact on model prediction.
# - FranchiseCode=0 will positively influence model prediction, Bank= Wells Fargo Bank NATL ASSOC, and UrbanRural=0 will negatively impact model prediction for record 1.
# - Bank= BIZCAPITAL BIDCO ll,LLC will positively affect model prediction, while Bankstate=LA will negatively impact model prediction for record 4.

# ## Question 6 - optional (no points)
# 
# Train decision tree surrogate model with 4 levels and plot the result.

# In[ ]:




