from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris # A classic multiclass classification dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load a classification dataset
data = load_iris()
X, y = data.data, data.target # X: features (sepal/petal length/width), y: target (iris species)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (often beneficial for SVMs, especially with RBF kernel)
# SVMs are sensitive to feature scales.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize the SVM classifier
# kernel: 'linear' for linear separation, 'rbf' (Radial Basis Function) for non-linear.
# C: Regularization parameter. Smaller C means stronger regularization (more misclassifications allowed).
# gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Affects the "reach" of a single training example.
model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) # 'scale' uses 1 / (n_features * X.var())

# 5. Train the model
model.fit(X_train_scaled, y_train)

# 6. Make predictions
y_pred = model.predict(X_test_scaled)

# 7. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Interview points:
# - Explain hyperplane, margin, and support vectors.
# - Crucially, explain the "kernel trick" and common kernel types (linear, RBF).
# - Discuss regularization parameter `C` and its effect on bias-variance trade-off.
# - Mention the importance of feature scaling for SVM.