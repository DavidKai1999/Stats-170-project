
X_train, X_test, y_train, y_test = model_selection.train_test_split() # data, label, test_size

print("-------------Run logistic regression---------------")
logReg = LogisticRegression(n_jobs=3)
logReg.fit(X_train, y_train)
y_predicted_lr = logReg.predict(X_test)
print("logistic regression accuracy_score ", accuracy_score(y_test, y_predicted_lr))
print("logistic regression balanced_accuracy_score ", balanced_accuracy_score(y_test, y_predicted_lr))
print("logistic regression confusion_matrix_score ", confusion_matrix(y_test, y_predicted_lr))
print('logistic linear regression','Mean squared error:',mean_squared_error(y_test, y_predicted_lr))
print('logistic linear regression','mean_absolute_error:',mean_absolute_error(y_test, y_predicted_lr))
print('logistic linear regression r2:',r2_score(y_test, y_predicted_lr))

print("-------------Run RandomForestClassifier---------------")
forest = RandomForestClassifier(n_estimators=100, n_jobs=3)
forest.fit(X_train, y_train)
y_predicted = forest.predict(X_test)

print("-------------Run SVCClassifier---------------")
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc.fit(X_train, y_train)
y_predicted_sv = svc.predict(X_test)

print("-------------Run Naive Bayes Classifier---------------")
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
y_predicted_nb = nb.predict(X_test)
