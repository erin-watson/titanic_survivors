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
# I will now identify these columns and remove them from the dataset.

for column in df.columns:
    if df[column].dtype == "object":
        print("THE FOLLOWING DATA IS OBJECT DATA TYPES ONLY")
        print(df[column])
        del df[column]

print("THE FOLLOWING DATA HAS HAD THE OBJECT COLUMN TYPES REMOVED")
print(df.columns)
# all non numeric datatypes should now be removed.



df = df.dropna()

answers = df['Survived']

features = df.drop(['Survived'], axis=1)

print(answers)
print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
print(features)



rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(features, answers)
print(rf)

print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
print("FEATURE IMPORTANCES FEATURE IMPORTANCES FEATURE IMPORTANCES")
importances = rf.feature_importances_
print(importances)
#Important features were identified as 'PassengerId', 'Pclass', 'Name', 'Sex',
#'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked'.
#Of those, SibSp seems unusual. it is likely this is linked to wealth such as
# having many siblings/ be maried making it more or less likely for you to be
# from a wealthy family.

print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
print("NOW WE PREPARE THE TEST DATA")
df = pd.read_csv("test.csv")

print(df.columns)

for column in df.columns:
    print(df[column])


for column in df.columns:
    if df[column].dtype == "object":
        print(df[column])
        del df[column]

print(df.columns)


df = df.dropna()

test_features = df

predict = rf.predict(test_features)
print(predict)
