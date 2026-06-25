import numpy as np
import scipy.stats as stat

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

mean = np.mean(speed)
median = np.median(speed)
mode = stat.mode(speed)

print(mean, median, mode)
