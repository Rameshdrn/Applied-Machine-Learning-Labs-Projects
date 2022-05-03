#!/usr/bin/env python
# coding: utf-8

# # Lab-9 Template
# 
# Answer questions in the designated cells.
# 
# Rename notebook to include your name and NET-ID in the file name.
# Submit two versions of your notebook:
# - Original notebook
# - HTML version. Make sure to produce Shapley plots using `matplotlib=True` option so that your plots are displayed in the HTML version
# - Validate that all plots are displayed in the HTML version
# - You can use `matplotlib=False` in the original version to get nicer plots
# 
# No need to encode missing values or categorical variables, other than make sure H2O recognizes them properly.
# Use `asfactor` if need to correct encoding.
# 
# Use H2O to train models.
# 
# Resources: 
# - https://github.com/slundberg/shap
# - H2O Explainability best practices: https://github.com/h2oai/h2o-tutorials/blob/master/best-practices/model-interpretability/interpreting_models.ipynb
# 
# Materials by Patrick Hall:
# https://github.com/jphall663/interpretable_machine_learning_with_python
# https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/debugging_resid_analysis_redux.ipynb   
# Decision tree plotting: https://github.com/h2oai/h2o-tutorials/blob/master/best-practices/model-interpretability/interpreting_models.ipynb
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
#!pip install shap==0.40.0

import shap

import h2o
from h2o.estimators import H2OTargetEncoderEstimator

try:
    h2o.cluster().shutdown()
except:
    pass 


# In[2]:


#Limit to 3 threads and 8GB memory
h2o.init(nthreads=3, max_mem_size=8)


# ### Load data

# In[3]:


train = h2o.import_file('Car_Prices_Poland_train.csv')
test = h2o.import_file('Car_Prices_Poland_test.csv')


# In[4]:


train.head(5)


# In[5]:


print("Train shape:", train.shape)
print("Test shape:", test.shape)


# In[6]:


train.describe()


# ## Question 1
# 
# Train H2O `H2OGradientBoostingEstimator` with parameters:
# ```
# nfolds=5,
# ntrees=500,
# stopping_rounds=5,
# stopping_metric='MAE',
# seed=1234,
# keep_cross_validation_predictions = False
# ```
# Display model performance on `test` dataset using `model_performance` function.

# In[7]:


from h2o.estimators.gbm import H2OGradientBoostingEstimator


# In[8]:


response = "price"      
predictors = train.columns


# In[9]:


train.head(2)


# In[10]:


gbm = H2OGradientBoostingEstimator(nfolds=5,
                                   ntrees=500,
                                   stopping_rounds=5,
                                   stopping_metric='MAE',
                                   seed=1234,
                                   keep_cross_validation_predictions = False)
gbm.train(x=predictors, y=response, training_frame=train)


# In[11]:


perf = gbm.model_performance(test)


# In[12]:


print(perf)


# In[13]:


gbm.varimp_plot()


# ## Question 2
# 
# Use model from Question 1 to answer Q#2.
# 
# - Calculate and display permutation feature importance for the model using **test** dataset
# - What is most important feature?
# - Can you tell how feature is impacting (direction) probability? 

# In[14]:


permutation_varimp = gbm.permutation_importance(test, use_pandas=True)


# In[15]:


gbm.permutation_importance_plot(test)


# In[16]:


gbm.permutation_importance_plot(test, n_repeats=15)


# ## Question 3
# 
# Calculate absolute error percentage of the price on test dataset. Add absolute error column to the dataset and call it `abs_error`.   
# 
# Formula: `abs(price-predict_price)`

# In[17]:


pred = gbm.predict(test)


# In[18]:


test.head(2)


# In[19]:


test['abs_error'] = abs((test['price'] - pred['predict']) / test['price'])


# In[20]:


test.head(2)


# ## Question 4
# 
# Calculate and display summary plot of Shapley values. Use `test` dataset to calculate Shapley values.
# 
# - What is the most important feature based on Shapley values? 
# - Based on the Summary plot, how features `year` and `mileage` affect model precition of the car price?  

# In[21]:


test.shape


# In[22]:


# calculate SHAP values using function predict_contributions
contributions = gbm.predict_contributions(test)


# In[23]:


# convert the H2O Frame to use with shap's visualization functions
contributions_matrix = contributions.as_data_frame().to_numpy()


# In[24]:


len(contributions_matrix)


# In[25]:


# shap values are calculated for all features
shap_values = contributions_matrix[:,0:11]


# In[26]:


# expected values is the last returned column
expected_value = contributions_matrix[:,9].min()


# In[27]:


test.head(2)


# In[28]:


X = ['mark','model','generation_name','year','mileage','vol_engine','fuel','city','province','abs_error']


# In[29]:


shap.summary_plot(shap_values[:,0:9], X)


# In[30]:


shap.summary_plot(shap_values[:,0:9], X, plot_type="bar")


# ## Question 5
# 
# Plot individual Shapley values plots for records top 3 and bottom 3 records based on the residuals, for the total of 6 plots.  
# 
# Explain each plot in terms of what are most influential features and how they impact model prediction, and why you think model was correct for the top 3 records (smallest residuals) and significantly incorrect for the bottom 3 records (largest residuals).
# 

# In[31]:


shap_val = pd.DataFrame(shap_values,columns = X)


# In[32]:


shap_val = shap_val.sort_values(by = ['abs_error'],ascending=False)


# In[33]:


shap_val['index1'] = shap_val.index


# In[34]:


shap_val.head(2)


# In[35]:


top_3 = shap_val.iloc[0:3,:]


# In[36]:


bottom_3 = shap_val.iloc[-3:,:]


# In[37]:


cols = [x for x in shap_val.columns if x not in ['index1']]
for i in cols:
    print(top_3.plot.scatter(x = 'index1',y=i,figsize=(16,3)));


# In[38]:


cols = [x for x in shap_val.columns if x not in ['index1']]
for i in cols:
    print(bottom_3.plot.scatter(x = 'index1',y=i,figsize=(16,3)));


# ## Question 6 
# 
# Train new H2O GBM model on the test dataset trying to predict residuals. This is your surrogate modedel that you will use to understand what features are driving high residuals.   
# 
# Use same parameters as in question #1, except:
#   - predictor is now "abs_error"
#   - dataset now is "test" dataset
#   - make sure not to include original "price" column in the model
# 
# Plot Shapley summary plot for the top 100 records with highest residuals.
# Answer following question:
#   - What is the most important feature of the surrogate model that impacts high residuals?

# In[39]:


test_new = test.drop(['price'])


# In[40]:


response = "abs_error"      
predictors = test_new.columns


# In[41]:


gbm_test = H2OGradientBoostingEstimator(nfolds=5,
                                   ntrees=500,
                                   stopping_rounds=5,
                                   stopping_metric='MAE',
                                   seed=1234,
                                   keep_cross_validation_predictions = False)
gbm_test.train(x=predictors, y=response, training_frame=test_new)


# In[42]:


test.head(2)


# In[43]:


test_new.head(2)


# In[44]:


test_sort = test_new.sort(9, ascending=False)


# In[45]:


test_sort = test_sort[0:100,:]


# In[46]:


test_sort.shape


# In[47]:


# calculate SHAP values using function predict_contributions
contributions = gbm_test.predict_contributions(test_sort)


# In[48]:


# convert the H2O Frame to use with shap's visualization functions
contributions_matrix = contributions.as_data_frame().to_numpy()


# In[49]:


len(contributions_matrix)


# In[50]:


test_sort.shape


# In[51]:


test_sort.head(2)


# In[52]:


# shap values are calculated for all features
shap_values = contributions_matrix[:,0:10]


# In[53]:


# expected values is the last returned column
expected_value = contributions_matrix[:,9].min()


# In[54]:


shap.summary_plot(shap_values[:,0:9], X)

