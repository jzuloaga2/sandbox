import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

zzz = []
for ii in range(df.shape[0]):
    if df['species'][ii] == 'setosa':
        zzz.append(1)
    else:
        zzz.append(0)

df['zzz'] = zzz


# **************************************************
# Do a Logistic regression to predict:
# **************************************************

y = df['zzz']
X = df.drop(['species','zzz'],axis=1)

model0 = LogisticRegression()
model0.fit(X,y)
y_pred = model0.predict(X)

accuracy = metrics.accuracy_score(y,y_pred)
print('The accuracy was %f' % (accuracy))



