#Импорт библиотек
import pandas as pd
import numpy as np
import sklearn as sl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import metrics
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
#Загрузка БД
data=pd.read_excel('3.xlsx')
#Добавление новых дескрипторов и их расчет
data.insert(1, 'Volume', 0, False)
data.insert(1, 'SSA', 0, False)
for i in range(227):
    a=data['length']
    b=data['width']
    c=data['depth']
    data['Volume']=a*b*c
    data['SSA']=(2*a*b+2*a*c+2*b*c)/a*b*c
    i=i+1
#Матрица корреляции
data_corr=data.corr('pearson')
plt.figure(figsize= (10,10))
sns.set(font_scale=0.7)
sns.heatmap(data_corr, annot=True)
plt.show()
#Feature importance
model= RandomForestRegressor()
x=list(set(data.columns)-set(['formula','Km','Kcat','Subtype']))
model.fit(data[x],data['Km'])
importance = model.feature_importances_
plt.bar(x,importance)
plt.show()
model.fit(data[x],data['Kcat'])
importance = model.feature_importances_
plt.bar(x,importance)
plt.show()
#Нормализация
x1=list(set(data.columns)-set(['formula','Subtype']))
t=data[x1].values
cols=data[x1].columns
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(t)
data=pd.DataFrame(x_scaled, columns=cols)
#Рандомный лес
x_train, x_test, y_train, y_test = train_test_split(data[x], data['Kcat'], test_size=0.3)
model.fit(x_train,y_train)
predicted=model.predict(x_test)
print(predicted)
k=model.score(x_test,y_test)
print(k)
#Линейная регрессия
model1=LinearRegression()
model1.fit(x_train,y_train)
predicted=model1.predict(x_test)
print(predicted)
k1=model1.score(x_test,y_test)
print(k1)
#Байесова регрессия
model2=BayesianRidge()
model2.fit(x_train,y_train)
predicted=model2.predict(x_test)
print(predicted)
k2=model2.score(x_test,y_test)
print(k2)
#Дерево решений регрессия
model3=DecisionTreeRegressor()
model3.fit(x_train,y_train)
predicted=model3.predict(x_test)
print(predicted)
k3=model3.score(x_test,y_test)
print(k3)
