# This is a python script that predicts the probability of survival of titanic
# passengers.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#####################################################
# DATA PREP
#####################################################
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

training_data, val_data = train_test_split(df, test_size=0.2, random_state=25)


###################################################

answers = training_data['Survived']

features = training_data.drop(['Survived'], axis=1)

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
# 6/7 of the remaining features were chosen as important.
# how do i find out which ones they are.

print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
print("preparation of Validation data")


val_answers = val_data['Survived']

val_features = val_data.drop(['Survived'], axis=1)


predict = rf.predict(val_features)
print(predict)

print(val_answers.values)

from sklearn.metrics import accuracy_score
print("accuracy_score: ", accuracy_score(val_answers, predict))
