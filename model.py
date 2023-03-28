
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("dataset2.csv")
print("dataset : \n",data)

data = np.array(data)

X = data[1:, :-1]
y = data[1:, -1]

print("X : \n",X)
print("y : \n",y)

y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

input=[int(x) for x in "38 62000 700".split(' ')]
final=[np.array(input)]

predict = log_reg.predict_proba(final)
print("predicted value : ",predict )
joblib.dump(log_reg,'model2.joblib')



