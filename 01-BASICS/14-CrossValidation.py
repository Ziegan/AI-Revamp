from sklearn import datasets
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

##KFOLD
k_folds = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

##STRATIFIED KFOLD
sk_folds = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=sk_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

##LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

##LeavePOut
lpo = LeavePOut(p=2)
scores = cross_val_score(clf, X, y, cv=lpo)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

##SHUFFLE SPLIT
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores = cross_val_score(clf, X, y, cv=ss)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
