import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scaling = StandardScaler()

data_csv = pandas.read_csv("6.01-VISUALIZING/data.csv")
print(data_csv)

x = data_csv[["Weight", "Volume"]]
print(x)

scaled_x = scaling.fit_transform(x)
print(scaled_x)

regression = linear_model.LinearRegression()
regression.fit(scaled_x, data_csv["CO2"])
scale_to = scaling.transform([[2300, 1.3]])
print(scale_to)

predicted_CO2 = regression.predict([scale_to[0]])
print(predicted_CO2)
