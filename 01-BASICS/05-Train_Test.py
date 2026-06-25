import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x
plt.scatter(x, y)
plt.show()

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

model = np.poly1d(np.polyfit(train_x, train_y, 4))

linespace = np.linspace(0, 6, 100)
plt.scatter(train_x, train_y, c="Black")
plt.scatter(test_x, test_y, c="Red")
plt.plot(linespace, model(linespace))
plt.show()

r2_score_train = r2_score(train_y, model(train_x))
r2_score_test = r2_score(test_y, model(test_x))
print(r2_score_train, r2_score_test)

print(model(5))
