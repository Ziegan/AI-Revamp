from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
# print(iris)

X = iris["data"]
y = iris["target"]
# print(X, y)

logit = LogisticRegression(max_iter=1000)

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

scores = []

for choice in C:
    logit.set_params(C=choice)
    logit.fit(X, y)
    scores.append(logit.score(X, y))

print(logit.fit(X, y))
print(logit.score(X, y))
print(C, scores)
