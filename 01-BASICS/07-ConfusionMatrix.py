import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, 0.9, size=1000)
predic = np.random.binomial(1, 0.9, size=1000)

confusion_matrix = metrics.confusion_matrix(actual, predic)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix, display_labels=[0, 1]
)
confusion_matrix_display.plot()
plt.show()

accuracy = metrics.accuracy_score(actual, predic)
precision = metrics.precision_score(actual, predic)
sensitivity_recall = metrics.recall_score(actual, predic)
specificity = metrics.recall_score(actual, predic, pos_label=0)
f1_score = metrics.f1_score(actual, predic)
print(accuracy, precision, sensitivity_recall, specificity, f1_score)
