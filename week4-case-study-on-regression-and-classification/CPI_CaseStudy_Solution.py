# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data_income = pd.read_csv("income.csv")
data = data_income.copy()

# To check variables datatypes
print(data.info())

# Check for missing values
data.isnull()

print("Data columns with null values:\n", data.isnull().sum())

# Summary of numerical variables
summary_num = data.describe()
print(summary_num)

# Summary of categorical variables
summary_cate = data.describe(include="O")
print(summary_cate)

# Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

# Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

data = pd.read_csv('income.csv', na_values=[" ?"])

data.isnull().sum()

missing = data[data.isnull().any(axis=1)]

data2 = data.dropna(axis=0)

correlation = data2.corr()

data2.columns

gender_salstat = pd.crosstab(index=data2["gender"],
                     columns=data2['SalStat'],
                     margins=True,
                     normalize='index')
print(gender_salstat)

SalStat = sns.countplot(data2['SalStat'])
sns.distplot(data2['age'], bins=10, kde=False)
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()