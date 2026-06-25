import numpy as np

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

deviation = np.std(speed)
varience = np.var(speed)
print(deviation, varience)
