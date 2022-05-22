# This is a python script that predicts the probability of survival of titanic
# passengers.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
###################################################################
# DATA PREP
###################################################################
df = pd.read_csv('train.csv')


for column in df.columns:
    print(df[column])




# Change Sex to binary data
df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)




#Created a new column with the combined Parch and SibSp data
df["Traveled with"] = df["Parch"] + df["SibSp"]

del df['Parch']
del df['SibSp']



# The remaining Object columns have been deemed not important and will now be
# removed.

for column in df.columns:
    if df[column].dtype == 'object':
        print(df[column])
        del df[column]

del df['PassengerId']


print(df.columns)

print('THE FOLLOWING DATA HAS HAD THE OBJECT COLUMN TYPES REMOVED')
# all non numeric datatypes (and Passenger ID) should now be removed.




# Check for NaN values and replace them with the mode of the data for each column.
for column in df.columns:
    print(df[column].isna().sum())
    df[column].fillna(df[column].mode()[0], inplace=True)
    print(df[column])
    print(df[column].isna().sum())


df = df.dropna()





# Creating a training and validation dataset
training_data, val_data = train_test_split(df, test_size=0.2, random_state=25)

#############################################################################






answers = training_data['Survived']

features = training_data.drop(['Survived'], axis=1)

print(answers)
print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
print(features)



rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(features, answers)
print(rf)

print('''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''')
print('FEATURE IMPORTANCES FEATURE IMPORTANCES FEATURE IMPORTANCES')
importances = rf.feature_importances_
print(importances)

# Class, Sex, Age, Fare and Traveled with have been used to train the model.
# Of those 5, fare was found to have been most important by the model followed
# by sex and age.


print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
print('preparation of Validation data')


val_answers = val_data['Survived']

val_features = val_data.drop(['Survived'], axis=1)


predict = rf.predict(val_features)
print(predict)

print(val_answers.values)

print('accuracy_score: ', accuracy_score(val_answers, predict))
# Model predicts survival at 79% accracy.
