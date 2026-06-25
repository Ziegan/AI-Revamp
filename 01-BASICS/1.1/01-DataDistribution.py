import matplotlib.pyplot as plt
import numpy as np

uniform_dist = np.random.uniform(0.0, 0.5, 100000)
plt.hist(uniform_dist, 100)
plt.show()

normal_dist = np.random.normal(5.0, 1.0, 100000)
plt.hist(normal_dist, 100)
plt.show()
