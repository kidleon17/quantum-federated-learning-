from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts

#import our data
BCW = datasets.load_breast_cancer()
X = BCW.data
y = BCW.target

#split the data to  3:1
X_train,X_test,y_train,y_test = ts(X,y,test_size=0.25)

# select different type of kernel function and compare the score

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
# print("The score of rbf is : %f"%score_rbf)
with open("SVM.txt","a+") as file:
    file.write("The score of rbf is : %f\n"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
# print("The score of linear is : %f"%score_linear)
with open("SVM.txt","a+") as file:
    file.write("The score of linear is : %f\n"%score_linear)
# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
# print("The score of poly is : %f"%score_poly)
with open("SVM.txt","a+") as file:
    file.write("The score of poly is : %f\n"%score_poly)