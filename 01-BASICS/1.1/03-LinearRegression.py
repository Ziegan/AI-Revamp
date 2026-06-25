import matplotlib.pyplot as plt
import scipy.stats as stats

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
plt.scatter(x, y)
plt.show()

slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)


def slope_func(x):
    return slope * x + intercept


model = list(map(slope_func, x))
plt.scatter(x, y)
plt.plot(x, model)
plt.show()

data = slope_func(99)
print(data)
