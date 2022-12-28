#!/usr/bin/env python
# coding: utf-8

# # Pesticide: Summary of the data
# 
# > Summary plots and tables 
# 
# 

# ## Imports
# 
# This bit imports any libraries included in the following including those from `Pesticide` to handle the data.
# 
# The first cell import general libraries and sets the folder to allow importing Pesticide which is done in the second cell

# In[1]:


import pandas as pd
from pandasql import sqldf
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from pathlib import Path
import sys

# this sets the path for Pesticide so we can import the functions
# N.B. PesticideDocs and Pesticide need to be in same folder
# tried to load from github using https://pypi.org/project/httpimport/ but 
# couldn't get to work due to modular structure i.e. how to access src/plotfuncts/plot1.py greatplot

module_path = Path( os.getcwd() )
module_path = module_path.parent.__str__() + '\\Pesticide'

sys.path.insert(0, module_path)




# In[2]:


from src.data_loading.loads_from_url import *
from src.data_cleaning.modify_dfs import *
from src.data_visualisation.plot_funcs import *
from src.utils.utils import *

cwd = module_path

folder_path = os.path.join(cwd,'data')

create_csvs=False
if create_csvs:
    getAllFilesThenSave(folder_path)


# ## Summaries of the data
# 
# The data is loaded then rows near the beginning and end are shown below
# 
# After this `df.describe()` and `df.info()` are used to get more info on the dataframe.

# In[3]:


df2 = pd.read_csv(os.path.join(folder_path,'combined_data.csv') ,index_col=0 )
# change data type of columns
df2['date_of_sampling'] = pd.to_datetime(df2['date_of_sampling'])
df2


# In[4]:


df2.to_csv('./_data/df2')


# In[5]:


df2.describe(include='all', datetime_is_numeric=True)


# ## Modify functions

# In[6]:


df2_grouped, df2_grouped_sample = groupby_id_and_q(df2)
df2_grouped


# In[7]:


df2_grouped_sample.dtypes


# 

# ## Plot functions

# In[8]:


range_plots(df2);
range_plots(df2,plot_type='hist');



# In[9]:


pie_plot(df2_grouped,col_groupby='country_of_origin' , col_plot='amount_detected');

pie_plot(df2_grouped,col_groupby='country_of_origin' , col_plot='number_of_tests');

pie_plot(df2_grouped,col_groupby='country_of_origin' , col_plot='amount_pc');


# ## Changes with time
# 
# 

# In[10]:


df2


# In[11]:


# Return a named tuple object with three components: year, week and weekday
# https://docs.python.org/3/library/datetime.html#datetime.date.isocalendar
def get_week(x):
    
    return x.isocalendar()[1]


xx=df2_grouped_sample.copy()
xx['week'] = df2_grouped_sample['date_of_sampling'].apply(get_week)
xx['day'] = pd.DatetimeIndex(xx['date_of_sampling']).day
xx['month'] = pd.DatetimeIndex(xx['date_of_sampling']).month
xx['quarter'] = pd.DatetimeIndex(xx['date_of_sampling']).quarter
xx['year'] = pd.DatetimeIndex(xx['date_of_sampling']).year


xx2 = xx.groupby(['year','month','week'], as_index=False).mean(numeric_only=True)


xx2['date_grouped']=pd.to_datetime(
    {"year": xx2['year'], "month": xx2['month'], "day": xx2['day']}   
)
xx2

plt.plot(xx2['date_grouped'], xx2['amount_pc'],'ok')


xx2 = xx.groupby(['year','quarter'], as_index=False).mean(numeric_only=True)
xx2['month'] = xx2['month'].astype('int')
xx2['day'] = 1
xx2['date_grouped']=pd.to_datetime(
    {"year": xx2['year'], "month": xx2['month'], "day": xx2['day']}   
)
plt.plot(xx2['date_grouped'], xx2['amount_pc'],'b--');




# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:




