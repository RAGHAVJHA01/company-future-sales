#!/usr/bin/env python
# coding: utf-8

# In[348]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
#import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[229]:


train=pd.read_csv(r'C:\Users\Raghav Jha\Downloads\TRAIN_DATA_FILE.csv')
test=pd.read_csv(r'C:\Users\Raghav Jha\Downloads\TEST_file.csv')


# In[230]:


train.head(5)


# In[231]:


test.head(5)


# In[232]:


print(train.columns)


# In[233]:


test.columns


# In[234]:


print(train.shape)


# In[235]:


print(test.shape)


# In[236]:


train.isnull().sum()


# In[237]:


test.isnull().sum()


# In[238]:


test.isnull().sum()


# In[239]:


train.describe().T


# In[240]:


test.describe().T


# In[241]:


train.head(1)


# In[242]:


sns.pairplot(train)
plt.show()


# In[243]:


print(train.Holiday.value_counts())
print(train.Discount.value_counts())
print(train.Store_Type.value_counts())
print(train.Location_Type.value_counts())
print(train.Region_Code.value_counts())


# In[244]:


print(test.Holiday.value_counts())
print(test.Discount.value_counts())
print(test.Store_Type.value_counts())
print(test.Location_Type.value_counts())
print(test.Region_Code.value_counts())


# EDA

# In[245]:


values=train['Store_Type'].value_counts().values
label=['S1','S2','S3','S4']
fig,ax1=plt.subplots()
ax1.pie(values,labels=label,shadow=True,startangle=90,autopct='%1.1f%%')
plt.show()

sns.barplot(x='Store_Type',y='Sales',data=train)
label=['S1','S2','S3','S4']
plt.show()


# In[246]:


values=train['Location_Type'].value_counts().values
fig,ax1=plt.subplots()
label=['L1','L2','L3','L4','L5']
ax1.pie(values,labels=label,shadow=True,startangle=90,autopct='%1.1f%%')
plt.show()

sns.barplot(x='Location_Type',y='Sales',data=train)
label=['L1','L2','L3','L4','L5']
plt.show()


# In[247]:


ts=train.groupby(["Date"])["Sales"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);


# In[248]:


train.head(2)


# In[249]:


sns.boxplot(x='Holiday',y='Sales',data=train)


# In[250]:


sns.boxplot(x='Discount',y='Sales',data=train)


# In[251]:


sns.relplot(x='#Order',y='Sales',data=train,color='r')


# In[253]:


#convert date field from string to datetime
train['Date'] = pd.to_datetime(train['Date'],errors='coerce')


# In[254]:


#represent month in date field as its first day
train['Date'] = train['Date'].dt.year.astype('str') + '-' + train['Date'].dt.month.astype('str') + '-01'
train['Date'] = pd.to_datetime(train['Date'])
#groupby date and sum the sales
train_data= train.groupby('Date').Sales.sum().reset_index()


# In[255]:


#plot monthly sales
plot_data = [
    go.Scatter(
        x=train_data['Date'],
        y=train_data['Sales'],
    )
]
plot_layout = go.Layout(
        title='Monthly Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[316]:


#label Encoding
train['Store_Type']=label_encoder.fit_transform(train['Store_Type'])
train['Location_Type']=label_encoder.fit_transform(train['Location_Type'])
train['Region_Code']=label_encoder.fit_transform(train['Region_Code'])
train['Discount']=label_encoder.fit_transform(train['Discount'])
train['ID']=label_encoder.fit_transform(train['ID'])


# In[317]:


test['Store_Type']=label_encoder.fit_transform(test['Store_Type'])
test['Location_Type']=label_encoder.fit_transform(test['Location_Type'])
test['Region_Code']=label_encoder.fit_transform(test['Region_Code'])
test['Discount']=label_encoder.fit_transform(test['Discount'])
test['ID']=label_encoder.fit_transform(test['ID'])


# In[ ]:


train=train.drop(columns =['Date'] ,axis=1)


# In[329]:


train.head(1)


# In[319]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),cmap='RdYlGn',annot=True)
plt.show()


# PREDICTION

# In[321]:


X= train.drop(columns = ['Sales'] ,axis=1)
Y=train['Sales']


# In[351]:


test['Date']=test.drop(columns=['Date'] ,axis=1)


# In[353]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.4)


# In[354]:


# Model Building
LR = LinearRegression(normalize=True)
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)


# In[355]:


MSE= metrics.mean_squared_error(y_test,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# In[356]:


submission = pd.read_csv(r'C:\Users\Raghav Jha\Downloads\SAMPLE_submission_file.csv')
final_predictions = LR.predict(test)
submission['Sales'] = final_predictions
#only positive predictions for the target variable
submission['Sales'] = submission['Sales'].apply(lambda x: 0 if x<0 else x)
submission.to_csv('my_submission.csv', index=False)


# In[ ]:




