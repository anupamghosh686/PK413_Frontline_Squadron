#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import seaborn as sns
from pandas import DataFrame


# In[3]:


df = pd.read_csv("mango.csv")
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


X = df.iloc[:, :-1]
y = df.iloc[:, 8]


# In[6]:


X =X.fillna(0)


# In[7]:


X.isnull().sum()


# In[8]:


#SPLITTING DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)


# In[9]:


from sklearn import preprocessing
columns_to_scale = df.columns.tolist()
columns_to_scale = [x for x in columns_to_scale if x != "Yield"]
print(columns_to_scale)

std_scaler = preprocessing.StandardScaler().fit(X_train[columns_to_scale])
minmax_scaler = preprocessing.MinMaxScaler().fit(X_train[columns_to_scale])

X_train[columns_to_scale] = std_scaler.transform(X_train[columns_to_scale])




# In[10]:


#APPLY SCALER ON TEST SET
X_test[columns_to_scale] = std_scaler.transform(X_test[columns_to_scale])


# In[11]:


from sklearn.model_selection import cross_val_score
cv_k = 5
cv_scoring = 'neg_mean_squared_error'
cv_scoring = 'r2'


# In[12]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=cv_k, shuffle=True)


# In[13]:


from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR


# In[14]:


import time

#RANDOM FOREST REGRESSION

now = time.time()
est_best = RandomForestRegressor(n_estimators=10, n_jobs=-1)
est_best.fit(X_train, y_train)
scores = cross_val_score(est_best, X_train, y_train, cv=kf, scoring=cv_scoring)
print("ACCURACY: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
after = time.time()
print("Exec. time: {:5.2} s".format(after-now))


# In[15]:


#PREDICTING THE TEST SET RESULTS

y_pred = est_best.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)


# In[16]:


score


# In[17]:


y_test[100]


# In[18]:


y_pred[100]


# In[19]:


y_pred.shape


# In[20]:


y_test.dtypes


# In[21]:


import pickle
import joblib

joblib.dump(est_best, "best_regressor_model.pkl")


# In[ ]:


import joblib


#TAKING THE USER INPUT





import tkinter as tk

fields = ('Latitude', 'Longitude', 'ATMAX', 'ATMIN', 'humidity', 'pressure', 'tempmax', 'tempmin', 'Output')

def outputv(entries):
   
    a = float(entries['Latitude'].get())
    print("a", a)
   
    b = float(entries['Longitude'].get())
    c =  float(entries['ATMAX'].get())
    d = float(entries['ATMIN'].get())
    e = float(entries['humidity'].get())
    f = float(entries['pressure'].get())
    g = float(entries['tempmax'].get())
    h = float(entries['tempmin'].get())
    i = float(entries['Output'].get())
   
    
    pred_args = [Latitude, Longitude, ATMAX, ATMIN, humidity, pressure, tempmax, tempmin]
    pred_args_arr = np.array(pred_args)
    pred_args_arr = pred_args_arr.reshape(1, -1)
    mul_reg = open("multiple_regressor_model.pkl", "rb")
    ml_model = joblib.load(mul_reg)
    model_prediction = ml_model.predict(pred_args_arr)
    print("The Predicted Yield in Metric Tonnes per Hectare [t/ha] is")

    outputval = round(float(model_prediction), 2)
   
    entries['Output'].delete(0, tk.END)
    entries['Output'].insert(0, outputval )

    print("Output: %f" % float(outputval))



def makeform(root, fields):
    entries = {}
    for field in fields:
        print(field)
        row = tk.Frame(root)
        lab = tk.Label(row, width=22, text=field+": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, "0")
        row.pack(side=tk.TOP,
                 fill=tk.X,
                 padx=5,
                 pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT,
                 expand=tk.YES,
                 fill=tk.X)
        entries[field] = ent
    return entries

if __name__ == '__main__':
    root = tk.Tk()
    ents = makeform(root, fields)
    b1 = tk.Button(root, text='Output',
           command=(lambda e=ents: outputv(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
 
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()




pred_args = [Latitude, Longitude, ATMAX, ATMIN, humidity, pressure, tempmax, tempmin]
pred_args_arr = np.array(pred_args)
pred_args_arr = pred_args_arr.reshape(1, -1)
mul_reg = open("multiple_regressor_model.pkl", "rb")
ml_model = joblib.load(mul_reg)
model_prediction = ml_model.predict(pred_args_arr)
print("The Predicted Yield in Metric Tonnes per Hectare [t/ha] is")

round(float(model_prediction), 2)


# In[23]:


#KERNEL DENSITY ESTIMATION PLOT FOR ACTUAL YIELD
sns.distplot(df["Yield"], bins=10, kde = True)


# In[24]:


#JOINT PLOT FOR ACTUAL YIELD
sns.jointplot("pressure", "Yield", data=df, kind="kde")


# In[26]:


#COUNT PLOT FOR ACTUAL VALUE
sns.countplot(x = "Yield", data=df)


# In[27]:


y_test.shape


# In[28]:


y_pred.shape


# In[29]:


pred_df = pd.DataFrame(y_pred)
pred_df["mak"] = pred_df
pred_df


# In[30]:


#DISTRIBUTION PLOT FOR PREDICTED VALUE
sns.distplot(pred_df[0], bins=10, kde = True, rug=True)


# In[31]:


#COUNT PLOT FOR PREDICTED DATA

sns.countplot(x = "mak", data=pred_df)


# In[ ]:




