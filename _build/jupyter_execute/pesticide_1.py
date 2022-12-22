#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import ezodf
from modify_dfs import *
import os


# In[2]:


s_fname = "C:\\Users\simmt\Code1\Pesticide\data\August_2022_rolling_reports.ods"
# doc = ezodf.opendoc(s_fname)


# In[27]:


folder_path = 'C:\\Users\simmt\Code1\Pesticide\Pesticide-main\data_files'
file_path = []
for x in os.listdir(folder_path):
    file_path.append(os.path.join(folder_path,x) )

    
doc = ezodf.opendoc(file_path[0])
# get last sheet
sheet = doc.sheets[-4]


# In[37]:


df_dict = {}
print(sheet.name)
for i, row in enumerate(sheet.rows()):
#     row is a list of cells
# #     assume the header is on the first row
    if i == 2:
    # columns as lists in a dictionary
        df_dict = {cell.value:[] for cell in row}
df_dict


# In[48]:


data_sheet = []
for i,row in enumerate(sheet.rows()):
    if i==0:
        column_names = [cell.value for cell in row]
        print(column_names)
    else:
        data_sheet.append( [cell.value for cell in row] )
ddf = pd.DataFrame(data_sheet)

ddf.columns = column_names

ddf


# In[ ]:





# In[4]:


def import_ods(fname):
    doc = ezodf.opendoc(s_fname)
    for i, sheet in enumerate(doc.sheets):
        if i!=0: #ignore 1st sheet
            product = sheet.name
            print(product)
            
            df_new =import_ods_inner(sheet)
            df_new.columns = df_new.columns.str.strip()
            try:
                df = pd.concat([df,df_new])
            except:
                df = df_new
                
    return df
            
def import_ods_inner(sheet):
    df_dict = {}
    for i, row in enumerate(sheet.rows()):
    #     row is a list of cells
# #     assume the header is on the first row
        if i == 1:
        # columns as lists in a dictionary
            df_dict = {cell.value:[] for cell in row}

    # create index for the column headers
    for i,row in enumerate(sheet.rows()):
        if i == 0:
            continue
        elif i == 1:
            col_index = [cell.value for cell in row]
            continue
        for j,cell in enumerate(row):
            df_dict[col_index[j]].append(cell.value)
            
    # delete none column
    try:
        del df_dict[None]
    except:
        pass
    # and convert to a DataFrame
    df = pd.DataFrame(df_dict)
    
    # fill based on previous values    
    df.fillna(method='ffill', inplace=True)

    return df


# In[7]:


df = import_ods(s_fname)

modify_df(df)
# import re
# def extract_pcode(x):  
#     regexp = r'([A-Za-z]+[0-9]+\s[0-9]+[A-Za-z]+$)'
#     try:
#         return re.findall(regexp,x)[0]
#     except:
#         return 0
# eg_pc = 'hdwhck ss1 2wg'
# regexp = r'([A-Za-z]+[0-9]+\s[0-9]+[A-Za-z]+$)'
                      
                      
# # df = extract_pcode(df,'Address')
# df['Address'].apply(extract_pcode)


# In[98]:


# convert the first sheet to a pandas.DataFrame
sheet = doc.sheets[2]
product = sheet.name
df_dict = {}
for i, row in enumerate(sheet.rows()):
#     row is a list of cells
# #     assume the header is on the first row
    if i == 1:
        # columns as lists in a dictionary
#         print(cell.value)
#         df_dict.update({cell.value:[]})
        df_dict = {cell.value:[] for cell in row}
# df_dict

#         # create index for the column headers
for i,row in enumerate(sheet.rows()):
    if i == 0:
        continue
    elif i == 1:
        col_index = [cell.value for cell in row]
#         print([cell.value for cell in row])
        continue
    for j,cell in enumerate(row):
#         print (j,cell.value)
        df_dict[col_index[j]].append(cell.value)
# # # and convert to a DataFrame
try:
    del df_dict[None]
except:
    pass

df = pd.DataFrame(df_dict)


df.head()


# In[90]:


[ (x, len(df_dict[x])) for x in df_dict]
# 


# In[58]:


del df[None]
df.head()


# In[45]:


df.describe()


# In[11]:


df.info()


# In[57]:


list(df)


# In[59]:


def modify_df(df):
    
    # remove null column at end
    del df[None]
    # fill na values
    df.fillna(method='ffill', inplace=True)
    
     
def extract_pcode(df,col):  
    regexp = r'([A-Za-z]*[0-9]*\s[0-9]*[A-Za-z]*$)'
    
    df[col+'_post_code']=df[col].str.extract(regexp)
    return df
    
def extract_pesticide(df):
    
    df.replace({'None were detected above the set RL': 'n/a'}, regex=True, inplace=True)
    
    df2=df['pesticide_residues_found_in_mg/kg_(mrl)'].str.extract(r'(.*)\s(\d[\d.]*)\s+\(MRL\s*=\s*(\d[\d.]*)\)')

    df2.fillna(0, inplace=True)
    
    df2.rename(columns={0:'chem_name',1:'amount_detected',2:'mrl'},inplace=True)
   
    
    return df2
    
    
def rename_cols(df):
    renaming_dict = {
    old_name : new_name 
        for old_name,new_name 
        in zip(list(df),[new_name.lower().replace(' ','_') 
                         for new_name in list(df)])
    }

    df = df.rename(columns=renaming_dict)
    return df


# In[113]:





# In[47]:


# # df['Address'][0].str.extract(r')
# test_add = df['Address'][0]
# print(test_add)
# import re

# regexp = r'([A-Za-z]*[0-9]*\s[0-9]*[A-Za-z]*$)'
# re.search(regexp, test_add)

# print(df.Address.head(10))

# dfpcode=df.Address.str.extract(regexp)

# len(df), len(dfpcode)


# In[34]:


# 

# df2


# In[32]:


from modify_dfs import *

df2 = rename_cols(df)
df2 = extract_pesticide(df2)
df2


# In[62]:


df2 = extract_pcode(df,'Address')
df2.head()


# In[ ]:




