# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start
2.Data Preparation
3.Hypothesis Definition
4.Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model Evaluation
8.End
```
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: YUVARAJ V
RegisterNumber:  212223230252
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from  sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

data=fetch_california_housing()
print(data)
```
![Screenshot 2024-09-11 091055](https://github.com/user-attachments/assets/2c4030d2-8e93-44ee-b413-f5edda8d35a9)
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/16f33b6f-1220-472b-8568-efe77b4f57c3)
```
df.info()
```
![image](https://github.com/user-attachments/assets/dd9ac11c-b136-4314-9744-78c345ade378)
```
df.describe()
```
![image](https://github.com/user-attachments/assets/069d1eab-fa2c-406b-9339-06e86a4bd5eb)
```
x=df.drop(columns=['AveOccup','target'])
x.info()
```
![image](https://github.com/user-attachments/assets/590df28c-a0da-4eb5-b10a-21b7c205fce9)
```
y=df[['AveOccup','target']]
y.info()
```
![image](https://github.com/user-attachments/assets/b9a8574a-05ab-4bcb-8c06-16d70265a0c6)
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x.head()
```
![image](https://github.com/user-attachments/assets/53841f57-359d-41ba-9957-f59e62bc1269)
```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)
```
![image](https://github.com/user-attachments/assets/70cf1e51-2ad6-43d7-86c0-e07523caa895)
```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/565caaf0-1bc0-423a-bace-a29036f3da63)
```
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
```
![image](https://github.com/user-attachments/assets/7b7270fb-81a4-47ed-a1b4-8166e1f3e7ce)
```
print("\nPredictions:\n", y_pred[:5])
```
![image](https://github.com/user-attachments/assets/4cb3e4ff-8b72-4ff0-8660-6daa35c99a13)
```
## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
