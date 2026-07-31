# Encoding,feature scaling, Data Splitting,Model Trainig, Evaluation
#print(os.getcwd())

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

import os
import pandas as pd
print(os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR,"Dataset","adult.csv")
df = pd.read_csv(csv_path)

df = pd.DataFrame(df)
x = df[["age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country"]]
y = df["income"]


from sklearn.preprocessing import LabelEncoder
df = pd.get_dummies(df,columns=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"],dtype=int)
x = df.drop("income",axis=1)
y = df["income"]

print("Training Columns:")
print(x.columns.tolist())

print("Training Values:")
print(x.iloc[100].tolist())

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size = 0.2 , random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LogisticRegression()
lr.fit(x_train,y_train)

prediction = lr.predict(x_test)
print(prediction)
print(type(prediction))

print(df.iloc[100])

print("Accuracy:",accuracy_score(y_test,prediction))

import joblib

model_path = os.path.join(BASE_DIR,"models/model.pkl")
scaler_path = os.path.join(BASE_DIR,"models/scaler.pkl")
feature_path = os.path.join(BASE_DIR,"models/feature.pkl")



joblib.dump(lr,model_path)
joblib.dump(scaler,scaler_path)
joblib.dump(x.columns,feature_path)




