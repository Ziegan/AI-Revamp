import pandas as pd
from sklearn import linear_model

raw_data = pd.read_csv("data.csv")
print(raw_data)

x = raw_data[["Weight", "Volume"]]
y = raw_data["CO2"]

print(x, y)

regression = linear_model.LogisticRegression(max_iter=500)
regression.fit(x, y)

prediction = regression.predict([[2300, 1300]])
print(prediction)
