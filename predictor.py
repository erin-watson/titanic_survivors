# This is a python script that predicts the probability of survival of titanic
# passengers.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train.csv")

print(df.columns)

for column in df.columns:
    print(df[column])

print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")

# some columns are not numbers and therefore are difficult to use.
# I will now analyse these columns and figure out which are necessary to keep.

for column in df.columns:
    if df[column].dtype == "object":
        print(df[column])
        del df[column]

print(df.columns)

        # whatever code is here is only done to object columns

df = df.dropna()

answers = df['Survived']

features = df.drop(['Survived'], axis=1)

print(answers)
print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
print(features)

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

rf.fit(features, answers)

print(rf)

importances = rf.feature_importances_
print(importances)

df = pd.read_csv("test.csv")

print(df.columns)

for column in df.columns:
    print(df[column])

print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")

# some columns are not numbers and therefore are difficult to use.
# I will now analyse these columns and figure out which are necessary to keep.

for column in df.columns:
    if df[column].dtype == "object":
        print(df[column])
        del df[column]

print(df.columns)

        # whatever code is here is only done to object columns

df = df.dropna()

test_features = df

predict = rf.predict(test_features)
print(predict)
