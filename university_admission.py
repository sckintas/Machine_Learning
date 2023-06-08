#University Admission,

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
university_df = pd.read_csv("university_admission.csv")
#university_df.head(6)
#university_df.tail(6)
print(university_df.columns)
print(university_df.shape)
print(university_df.isnull().sum())
#university_df.dtypes
print(university_df.describe())
print(university_df['TOEFL_Score'].max())
print(university_df['TOEFL_Score'].min())
print(university_df['TOEFL_Score'].mean())

sns.heatmap(university_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()

university_df.hist(bins = 30, figsize = (20,20), color = 'r');
plt.show()

sns.pairplot(university_df)
plt.show()

for i in university_df.columns:
    plt.figure(figsize=(13, 7))
    sns.scatterplot(x=i, y='Chance_of_Admission', hue="University_Rating", hue_norm=(1, 5), data=university_df)
    plt.show()

#Correlation Matrix

corr_matrix = university_df.corr()
plt.figure(figsize = (12, 12))
sns.heatmap(corr_matrix, annot = True)
plt.show()

print(university_df.columns)
X = university_df.drop(columns = ['Chance_of_Admission'])
y = university_df['Chance_of_Admission']

print(X.shape)
print(y.shape)

X = np.array(X)
y = np.array(y)
y = y.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

import xgboost as xgb


model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 30, n_estimators = 100)

model.fit(X_train, y_train)
# predict the score of the trained model using the testing dataset

result = model.score(X_test, y_test)
print("Accuracy : {}".format(result))

y_predict = model.predict(X_test)


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)