print(f"======= TASK 1 Reproduce and Identify Leakage=========")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.30,random_state=42)

model=LogisticRegression()

model.fit(X_train,y_train)

train_accuracy=model.score(X_train,y_train)
Test_accuracy=model.score(X_test,y_test)

print("Train accuracy:",train_accuracy)
print("Test accuracy:",test_accuracy)                                        (#while testing an accuracy, we could not use trained data, always use test data. some underfitting occurs)


output:

======= TASK 1 Reproduce and Identify Leakage=========
Train accuracy: 0.8728571428571429
Test accuracy: 0.8666666666666667



print(f"==========Task 2 — Fix the Workflow Using a Pipeline=============")

pipe=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())])

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.30,random_state=42)

scores=cross_val_score(pipe,X_scaled,y,cv=5)

print("Mean:",scores.mean().round(2))
print("standardDeviation:",scores.std().round(2))



output:

==========Task 2 — Fix the Workflow Using a Pipeline=============
Mean: 0.86
standardDeviation: 0.02





X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.30,random_state=42)

for depth in [1,5,20]:
    model=DecisionTreeClassifier(max_depth=depth,random_state=42)
    model.fit(X_train,y_train)

    training_accuracy=model.score(X_train,y_train)
    testing_accuracy=model.score(X_test,y_test)

    print(f"Depth:{depth:2d}|Train:{training_accuracy:.2f}|Test:{testing_accuracy:.2f}")


Output:

Depth: 1|Train:0.88|Test:0.85
Depth: 5|Train:0.94|Test:0.89
Depth:20|Train:1.00|Test:0.86                                  (#depth 5 is the balanced data and also a good fit because it was passed in training and also in real world)
