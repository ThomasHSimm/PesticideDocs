#!/usr/bin/env python
# coding: utf-8

# # Pesticide: Summary of the data
# 
# > Summary plots and tables 
# 
# 

# ## Imports

# In[1]:


import pandas as pd
from pandasql import sqldf
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from pathlib import Path
import sys


module_path = Path( os.getcwd() )
module_path = module_path.parent.__str__() + '\\Pesticide'

sys.path.insert(0, module_path)

module_path


# In[2]:


from src.data_loading.loads_from_url import *
from src.data_cleaning.modify_dfs import *
from src.data_visualisation.plot_funcs import *


cwd = module_path

folder_path = os.path.join(cwd,'data')
file_path = []
for x in os.listdir(folder_path):
    file_path.append(os.path.join(folder_path,x) )

# load data and clean    
df = import_ods(file_path[2])

df2 = modify_df(df)

# needs to be added to modify
df2['amount_pc']=df2['amount_detected']/df2['mrl']

df2.loc[df2['amount_pc'].isna(),['amount_pc']]=0


# In[3]:


df2.head()


# In[4]:


df2.describe(include='all', datetime_is_numeric=True)


# ## Modify functions

# In[5]:


def groupby_id_and_q(df2: pd.DataFrame,
                     col_groupby: str = 'country_of_origin') -> pd.DataFrame:
    """
    Groups a Pandas DataFrame based on the sample_id and country_of_origin
        the new dataframe has a new column number_of_tests
        - this is the number of unique sample_ids
        the other 2 numerical columns are means of previous values

    Args:
        df2 (pd.DataFrame): DataFrame of Pesticide data after 1st clean
        col_groupby (str): Column to do 1st groupby

    Raises:
        ValueError: ??

    Returns:
        df2_grouped (pd.DataFrame): Pandas DataFrame grouped by id and country- note the mean is taken twice
                                        1. When grouping by id
                                        2. When grouping by col_groupby
                                    the new dataframe has a new column number_of_tests
                                        - this is the number of unique sample_ids
                                    in addition to the previous numeric columns
        df2_grouped_sample (pd.DataFrame): Pandas DataFrame grouped by id only
    """
    
    
    # group by id
    df2_grouped_sample = df2.groupby(['sample_id','date_of_sampling', col_groupby],as_index=False).mean(numeric_only =True).sort_values('amount_detected', ascending=False)
    
    # group by col_groupby-> mean
    df2_grouped = df2_grouped_sample.groupby(col_groupby,as_index=False).mean(numeric_only =True)
    
    # group by col_groupby-> count
    df2_grouped_b = df2_grouped_sample.groupby(col_groupby, as_index=False).count().iloc[:,0:2]
    
    # merge the 2 new dfs and rename count column
    df2_grouped= df2_grouped.merge(df2_grouped_b, left_on=col_groupby, right_on=col_groupby)
    df2_grouped.rename(columns ={'sample_id':'number_of_tests'},inplace=True)
    
    # sort dataframe by counts
    df2_grouped= df2_grouped.sort_values('number_of_tests', ascending=False)

    # reset index
    df2_grouped.reset_index(inplace=True, drop=True)
    df2_grouped_sample.reset_index(inplace=True, drop=True)

    return df2_grouped, df2_grouped_sample

df2_grouped, df2_grouped_sample = groupby_id_and_q(df2)
df2_grouped


# 

# ## Plot functions

# In[6]:


column_names_dict={'amount_detected':  'Pesticide residues \nfound (mg/kg)' ,
                    'mrl': 'Maximum reporting \nlimit MRL (mg/kg)', 
                    'amount_pc' : 'Pesticide residues \nfraction of MRL'}

def range_plots(df2: pd.DataFrame, 
                plot_type: str ='boxplot', column_to_plot: str = 'amount_detected',                
                cols_numeric: list=[], max_bin: list=[], bin_no: int = 30,
                column_names_dict: dict =column_names_dict ) -> plt.figure:
    """
    Produces plots to show ranges in data
    N.B. zeros are removed
    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot
    change for violinplot
    
    Args:
        df2 (pd.DataFrame): Pandas DataFrame. Pesticide data after grouping
        plot_type (str): string of type of plot to do 'boxplot' or 'hist'
        column_to_plot (str): which column to plot
        cols_numeric (list[str]): list of column strings to plot (no input finds numeric columns)
        max_bin (list[float]): list of floats (or ints) to gave max range for histogram (min = 0). 
        bin_no (int): number of bins
        column_names_dict (dict): dict of what to change column names to and from                        
        
    Raises:
        ValueError: ??

    Returns:
        pyplot figures object of a boxplot or histogram
    """
    
    # if no input of cols_numeric finds from dtype=int or float
    if not cols_numeric:
        cols_numeric = [column for column in df2 if df2[column].dtype=='float' or df2[column].dtype=='int' ]
    
    # get just the cols_numeric columns
    numeric_df = df2.loc[:,cols_numeric].copy()
    numeric_df = numeric_df.loc[ numeric_df[column_to_plot]!=0]
    
    # rename columns
    numeric_df = numeric_df.rename(columns = column_names_dict)

    # different plot types
    
    if plot_type=='boxplot':
    
        fig = plt.figure(figsize=(8,8))
        sns.boxplot( data = numeric_df )
        plt.yscale('log')
    
    elif plot_type=='hist':
        
        n_cols = 3
        n_rows = len(numeric_df.columns) // n_cols 

        fig, ax = plt.subplots(nrows= n_rows, ncols= n_cols, sharex= False,
                                    figsize= (12, 6))

        if not max_bin:
            max_bin = [5, 20, 1]
            
        for i,cols in enumerate(numeric_df):
            data_col = numeric_df[cols]
            counts, bins = np.histogram(data_col,
                    range = [0, max_bin[i]], bins= bin_no)
            ax[i].hist(bins[:-1], bins, weights=counts)
            
            # ax[i].set_yscale('log')
            ax[i].set_title(cols)
            if i==0:
                ax[i].set_ylabel('Count')
    
    return fig

range_plots(df2);
range_plots(df2,plot_type='hist');


# In[7]:


def pie_plot(df_grouped_temp: pd.DataFrame, 
             col_groupby : str = 'country_of_origin', col_plot: str ='number_of_tests',
             NUM_LABELS : int = 15, MIN_PCT : float = 2.) -> plt.figure:
    """
    Produces a pie plot from grouped data see groupby_id_country_chemical which does the grouping
    

    Args:
        df_grouped_temp (pd.DataFrame): Pandas DataFrame. Pesticide data after grouping
        col_groupby (str): which column has the names of the pie slices
        col_plot (str): which column has the data
        NUM_LABELS (int): max number of labels shown on the plot
        MIN_PCT (float): min % to display text has to be more than
        
    Raises:
        ValueError: ??

    Returns:
        pyplot figures object of a pie chart
    """
    
    maxed_out = False
    
    # what multiple of MIN_PCT have no decimal point
    go_to_1dp = 2.
    
    df_grouped_temp = df_grouped_temp[[col_groupby, col_plot]].copy()
    
    df_grouped_temp = df_grouped_temp.sort_values(col_plot, ascending=False)
    
    if len(df_grouped_temp) > NUM_LABELS:
        df_grouped_temp[col_plot] = 100*df_grouped_temp[col_plot]/sum(df_grouped_temp[col_plot])

        df_grouped_temp = df_grouped_temp.loc[df_grouped_temp[col_plot]>MIN_PCT]

        new_row = pd.Series({col_groupby: 'Other', 
                             col_plot: 100-sum( df_grouped_temp[col_plot] )})

        df_grouped_temp = pd.concat([df_grouped_temp, new_row.to_frame().T], ignore_index=True)
        
        maxed_out = True
     
    
    labels=df_grouped_temp[col_groupby].copy()
    colors = sns.color_palette('pastel')
    len_colors = len(colors)
    
    # create colors of length of labels
    for i in range(0, len(labels)//len_colors + 1):
        colors= colors + colors
    colors = colors[:len(labels)]
    
    # if there is an 'other' make the other this color
    if maxed_out:
        colors[-1]=(.9,.9,.9)
    
    
    fig=plt.figure(figsize=(7,7))
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        if pct> MIN_PCT * go_to_1dp:
            text_out = "{:.1f}%".format(pct)
        else:
            text_out = "{:.0f}%".format(pct)
        return text_out

    wedges, texts, autotexts =plt.pie(df_grouped_temp[col_plot],  
            colors = colors, labels=labels,
            autopct=lambda pct: func(pct, df_grouped_temp[col_plot]));
    
    col_plot = col_plot.replace('_',' ')
    col_groupby = col_groupby.replace('_',' ')
    plt.title(f'The {col_plot} by {col_groupby}')

    return fig

pie_plot(df2_grouped,col_groupby='country_of_origin' , col_plot='amount_detected');

pie_plot(df2_grouped,col_groupby='country_of_origin' , col_plot='number_of_tests');

pie_plot(df2_grouped,col_groupby='country_of_origin' , col_plot='amount_pc');


# ## Changes with time
# 
# 

# In[8]:


# Return a named tuple object with three components: year, week and weekday
# https://docs.python.org/3/library/datetime.html#datetime.date.isocalendar
def get_week(x):
    
    return x.isocalendar()[1]


xx=df2_grouped_sample.copy()
xx['week'] = df2_grouped_sample['date_of_sampling'].apply(get_week)
xx['month'] = pd.DatetimeIndex(xx['date_of_sampling']).month
xx2 = xx.groupby(['month','week'], as_index=False).sum()


plt.plot(xx2['week'], xx2['amount_pc'],'ok')

# xx2

xx3 = xx.groupby('month', as_index=False).sum()
# xx3['week'] = 
xx3 = xx2.loc[:,['month','week','amount_pc']].groupby('month', as_index=False).mean()
plt.plot(xx3['week'], xx3['amount_pc'],'-o')
xx3
# xx3


# In[9]:


# xx2 = xx.groupby('month', as_index=False).mean()
# plt.plot(xx2['week'], xx2['amount_pc'],'-o')

# xx2


# In[ ]:





# ## Other Stuff
# 
# Didn't make the grade ideas

# In[10]:


labels=df2_grouped['country_of_origin']
colors = sns.color_palette('pastel')[0:len(labels)]
fig=plt.figure(figsize=(7,7))
def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    if pct>2.5:
        text_out = "{:.1f}%".format(pct)
    else:
        text_out=''
    return text_out

wedges, texts, autotexts =plt.pie(df2_grouped['number_of_tests'],  colors = colors,
    autopct=lambda pct: func(pct, df2_grouped['number_of_tests']));

# plt.legend(wedges[0:10], labels[0:10],
#           title="Ingredients",
#           loc="center left",
#           bbox_to_anchor=(1, 0, 0.5, 1))

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    if i<10:
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        plt.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)


# In[11]:


def pie_plot_old(df_grouped_temp: pd.DataFrame, 
             col_groupby : str = 'country_of_origin', col_plot : str ='number_of_tests',
             NUM_LABELS : int = 10, MIN_PCT : float = 2.5) -> plt.figure:
    """
    Produces a pie plot from grouped data see groupby_id_country_chemical which does the grouping
    
    Ideas: 
        - below a certain % values go to 'other'

    Args:
        df_grouped_temp (pd.DataFrame): Pandas DataFrame. Pesticide data after grouping
        col_groupby (str): which column has the names of the pie slices
        col_plot (str): which column has the data
        NUM_LABELS (int): max number of labels shown on the plot
        MIN_PCT (float): min % to display text has to be more than
        
    Raises:
        ValueError: ??

    Returns:
        pyplot figures object of a pie chart
    """
    
    df_grouped_temp = df_grouped_temp.copy()
    
    df_grouped_temp = df_grouped_temp.sort_values(col_plot, ascending=False)
    
    
    
    labels=df_grouped_temp[col_groupby].copy()

    if len(labels)>NUM_LABELS:
        labels[NUM_LABELS:]=''
        
    colors = sns.color_palette('pastel')[0:len(labels)]
    fig=plt.figure(figsize=(7,7))
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        if pct> MIN_PCT:
            text_out = "{:.1f}%".format(pct)
        else:
            text_out=''
        return text_out

    wedges, texts, autotexts =plt.pie(df_grouped_temp[col_plot],  
            colors = colors, labels=labels,
            autopct=lambda pct: func(pct, df_grouped_temp[col_plot]));
    
    col_plot = col_plot.replace('_',' ')
    col_groupby = col_groupby.replace('_',' ')
    plt.title(f'The {col_plot} by {col_groupby}')

    return fig

pie_plot_old(df2_grouped,col_groupby='country_of_origin' , col_plot='amount_detected');



# In[ ]:




