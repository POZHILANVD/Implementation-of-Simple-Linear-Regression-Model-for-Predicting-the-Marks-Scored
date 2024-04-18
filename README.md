# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
 1.Import the standard Libraries.
 2.Set variables for assigning dataset values.
 3.Import linear regression from sklearn.
 4.Assign the points for representing in the graph.
 5.Predict the regression for marks by using the representation of the graph.
 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: POZHILAN V D
RegisterNumber: 212223240118 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores (1).csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
## DATASET:
![Screenshot 2024-02-23 112841](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/c6e96746-0d1c-4afb-8c1d-451c79ed128d)
## HEAD VALUES:
![Screenshot 2024-02-23 112849](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/b4108f71-8506-4d3b-84d1-e26e7b22f2a6)
## TAIL VALUES:
![Screenshot 2024-02-23 112855](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/2bd452c3-faf2-4514-8160-b9b80a2c3ee8)
## X and Y VALUES:
![Screenshot 2024-02-23 112928](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/084d401a-7eb4-40fb-b0cd-48b1740f3e26)
## Predication values of X and Y:
![Screenshot 2024-02-23 112950](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/1cd36544-ba2d-41e2-9ef7-62d0ba0c5509)
## MSE,MAE and RMSE:
![Screenshot 2024-02-23 113030](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/c2cf9776-577a-4555-9980-3818ff9770c8)
## Training Set:
![Screenshot 2024-02-23 113018](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/9179af8e-6b01-4a56-aa6f-b7ddc9904480)
##  Testing Set:
![Screenshot 2024-02-23 113026](https://github.com/POZHILANVD/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870498/672ca21c-3a2c-4363-96c4-1f1a801f493a)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
