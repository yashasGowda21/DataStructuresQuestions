# Linear Regression using sklearn
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Example dataset for Linear Regression
X, y = np.random.rand(100, 1) * 10, np.random.rand(100) * 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr_model = SklearnLinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])





# Logistic Regression using sklearn
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_model = SklearnLogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])







# K-Nearest Neighbors using sklearn
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])




# Decision Tree using sklearn
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])







# Naive Bayes using sklearn
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])





# KMeans Clustering using sklearn
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
kmeans_model = SklearnKMeans(n_clusters=3)
kmeans_model.fit(X)
predicted_labels = kmeans_model.predict(X)
print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)
print("Sample Cluster Predictions:", predicted_labels[:5])






# PCA using sklearn
from sklearn.decomposition import PCA

pca_model = PCA(n_components=2)
X_reduced = pca_model.fit_transform(X)
print("PCA Reduced Shape:", X_reduced.shape)
print("Sample Reduced Data:", X_reduced[:5])

# SVM using sklearn
from sklearn.svm import SVC

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])

# Random Forest using sklearn
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])






# XGBoost using xgboost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])








# Hierarchical Clustering using scipy
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linked = linkage(X, 'single')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# DBSCAN using sklearn
from sklearn.cluster import DBSCAN

db_model = DBSCAN(eps=1.5, min_samples=5)
db_model.fit(X)
print("DBSCAN Labels:", np.unique(db_model.labels_))
print("Sample Cluster Assignments:", db_model.labels_[:5])



# Gaussian Mixture Model using sklearn
from sklearn.mixture import GaussianMixture

gmm_model = GaussianMixture(n_components=3)
gmm_model.fit(X)
labels = gmm_model.predict(X)
print("GMM Labels:", np.unique(labels))
print("Sample GMM Predictions:", labels[:5])




# Neural Network using sklearn (MLP)
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
mlp_model.fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)
print("Neural Network (MLP) Accuracy:", accuracy_score(y_test, y_pred))
print("Sample Prediction:", X_test[:5], y_pred[:5])
