import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

raw_data = pd.read_csv("DecisionTree.csv")
print(raw_data)

dict = {"UK": 0, "USA": 1, "N": 2}
raw_data["Nationality"] = raw_data["Nationality"].map(dict)

dict = {"YES": 1, "NO": 0}
raw_data["Go"] = raw_data["Go"].map(dict)
print(raw_data)

x = raw_data[["Age", "Experience", "Rank", "Nationality"]]
y = raw_data["Go"]
print(x, y)

decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(x, y)
tree.plot_tree(
    decision_tree, feature_names=["Age", "Experience", "Rank", "Nationality"]
)
plt.show()
print(decision_tree.predict([[50, 20, 9, 1]]))
