import tkinter as tk
import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
data = pd.read_csv("/users/programming/downloads/student/student-por.csv", sep=";")


data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
# prediction = linear.predict([[7, 17, 2, 0, 2]])
def prediction_func(g1, g2, studytime, failures, absences):
    prediction = linear.predict([[int(g1), int(g2), int(studytime), int(failures), int(absences)]])
    print(prediction)
    submit = tk.Button(frame, text=prediction, command=lambda: prediction_func(entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry5.get()))
    submit.place(relx=0.5, rely=0.60, relwidth=0.75, relheight=0.15, anchor='n')
win = tk.Tk()
win.title("Grade Classifier")


HEIGHT = 500
WIDTH = 500


canvas = tk.Canvas(win, height=HEIGHT, width=WIDTH)
canvas.pack()

frame = tk.Frame(win, bg='#808080', bd=3)
frame.place(relx = 0.5, rely=0.1, relwidth=0.80, relheight=0.80, anchor='n')

credit_label = tk.Label(frame, text="P.Cortez and A. Silva. Using Data Mining to Predict Secondary \n School Student Performance. In A. Brito and J. Teixeira Eds., \n Proceedings of 5th FUture BUsiness TEChnology Conference \n (FUBUTEC 2008) pp. 5-12, Porto, Portugal, \n April, 2008, EUROSIS, ISBN 978-9077381-39-7.")
credit_label.place(relx=0.5, rely=0.80, relwidth=0.99, relheight=0.20, anchor='n')

instruction_label = tk.Label(frame, text="Input G1, G2, studytime, failures, absences")
instruction_label.place(relx=0.5, rely=0.03, relwidth=0.75, relheight=0.05, anchor='n')

entry1 = tk.Entry(frame)
entry1.place(relx=0.5, rely=0.1, relwidth=0.50, relheight=0.08, anchor='n')

entry2 = tk.Entry(frame)
entry2.place(relx=0.5, rely=0.20, relwidth=0.50, relheight=0.08, anchor='n')

entry3 = tk.Entry(frame)
entry3.place(relx=0.5, rely=0.30, relwidth=0.50, relheight=0.08, anchor='n')

entry4 = tk.Entry(frame)
entry4.place(relx=0.5, rely=0.40, relwidth=0.50, relheight=0.08, anchor='n')

entry5 = tk.Entry(frame)
entry5.place(relx=0.5, rely=0.50, relwidth=0.50, relheight=0.08, anchor='n')


submit = tk.Button(frame, text="Submit!", command=lambda:prediction_func(entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry5.get()))
submit.place(relx=0.5, rely=0.60, relwidth=0.75, relheight=0.15, anchor='n')

win.mainloop()
