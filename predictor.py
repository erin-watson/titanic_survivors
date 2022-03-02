# This is a python script that predicts the probability of survival of titanic
# passengers.

import pandas as pd

df = pd.read_csv("train.csv")

print(df.columns)
print(df["SibSp"])
