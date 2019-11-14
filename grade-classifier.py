import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
data = pd.read_csv("/users/programming/downloads/student/student-por.csv", sep=";")


data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
# print(data.head())
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
prediction = linear.predict([[7, 17, 2, 0, 2]])
print(acc)
# print(x_test)
print(x_train)
print(prediction)
