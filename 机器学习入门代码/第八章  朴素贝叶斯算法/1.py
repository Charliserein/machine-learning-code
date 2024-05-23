import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

# Generate synthetic data
np.random.seed(42)
X_pos = np.random.randn(500, 10) + 2
X_neg = np.random.randn(500, 10) - 2
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(500), np.zeros(500)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = X[:700], X[700:], y[:700], y[700:]

# Train a Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Get predicted probabilities for positive class (class 1)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
