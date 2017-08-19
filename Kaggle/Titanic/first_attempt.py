# first model : implementing logistic regression using features Pclass, Sex, Age, SibSp, Parch

import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("/home/sirajsandhu/Desktop/ACA-CML/Kaggle/Titanic")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#feature modification
y = train["Survived"]
train = train.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Survived', 'PassengerId', 'Embarked'], 1)
test_PId = test["PassengerId"]
test = test.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'PassengerId', 'Embarked'], 1)

#modifying Age
mean_age = train["Age"].mean()
round_mean_age = round(mean_age)
train["Age"] = train["Age"].fillna(round_mean_age)
test["Age"] = test["Age"].fillna(round_mean_age)
std_age = train["Age"].std()
train["Age"] = (train["Age"]-mean_age)/std_age
test["Age"] = (test["Age"]-mean_age)/std_age

# modifying Sex
import numpy as np
train["Sex"] = np.where(train["Sex"]=='male', 1, -1)
test["Sex"] = np.where(test["Sex"]=='male', 1, -1)

columns = train.columns.tolist()

# logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train[columns], y)

# predictions on test data
test_columns = test.columns.tolist()
test_predictions = model.predict(test[test_columns])

import csv
with open('submission1.csv', 'wb') as outfile :
    writer = csv.writer(outfile)
    writer.writerow(["PassengerId", "Survived"])
    for x, y in zip(test_PId, test_predictions) :
        writer.writerow([x, y])
