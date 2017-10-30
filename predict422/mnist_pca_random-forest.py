# Evaluating MNIST with Random Forest Classifiers
# Principal Component Analysis

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time

RSEED = 85

mnist = fetch_mldata('MNIST original')
print(mnist)

# To plot classification reports
def plot_cr(cr):
    lines = cr.split('\n')

    classes = []
    plot_mat = []
    for line in lines[2:(len(lines) - 3)]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1:len(t)-1]]
        print(v)
        plot_mat.append(v)

    plt.figure(figsize=(8,8))
    plt.imshow(plot_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Classification report')
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, 
      ['precision', 'recall', 'f1-score'], 
      rotation=45
    )
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')

# Split the data into train, test sets
split = 60000
X_train, X_test = mnist["data"][:split], mnist["data"][split:]
y_train, y_test = mnist["target"][:split], mnist["target"][split:]

# Shuffle the training indices for cross validation
shuffle = np.random.permutation(split)
X_train, y_train = X_train[shuffle], y_train[shuffle]

### RANDOM FOREST on full set ###
start = time.clock()
forest_clf = RandomForestClassifier(
  bootstrap = True,
  n_estimators=20,
  max_features='sqrt', 
  random_state=RSEED
)
forest_clf.fit(X_train, y_train)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test.astype(np.float64))
print(cross_val_score(
  forest_clf, 
  X_scaled, y_test, 
  cv=10, 
  scoring="accuracy"
))
y_pred = cross_val_predict(
  forest_clf,
  X_scaled, y_test,
  cv=10
)

print(classification_report(y_test, y_pred))
plot_cr(classification_report(y_test, y_pred))

f1score_a = f1_score(y_test, y_pred, average='macro')
stop = time.clock()
time1 = stop - start

cm = confusion_matrix(y_test, y_pred)

plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(10, 10))
plt.show()

rws = cm.sum(axis=1, keepdims=True)
norm = cm / rws

np.fill_diagonal(norm, 0)
plt.matshow(norm, cmap=plt.cm.gray)
plt.show()

### PRINCIPAL COMPONENT ANALYSIS ###
start = time.clock()
pca_start = time.clock()
pca = PCA(n_components = 0.95) 
X_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
pca_stop = time.clock()
pca_time = pca_stop - pca_start

forest_clf2 = RandomForestClassifier(
  bootstrap = True,
  n_estimators=20,
  max_features='sqrt', 
  random_state=RSEED
)
forest_clf2.fit(X_pca, y_train)

X_scaled = scaler.fit_transform(X_test_pca.astype(np.float64))
print(cross_val_score(
  forest_clf2, 
  X_scaled, y_test, 
  cv=10, 
  scoring="accuracy"
))
y_pred = cross_val_predict(
  forest_clf2,
  X_scaled, y_test,
  cv=10
)

print(classification_report(y_test, y_pred))
plot_cr(classification_report(y_test, y_pred))

f1score_b = f1_score(y_test, y_pred, average='macro')
stop = time.clock()
time2 = stop - start

print("RF took:", round(time1, 2), "secs")
print("RF with PCA took:", round(time2, 2), "secs")
print("PCA component identification took:", round(pca_time, 2), "secs")
print("PCA increased time:", round((time2 - time1) / time1 * 100, 1), "%")
print("Main set has", len(X_train[0]), 
      "variables and F1 score of:", round(f1score_a * 100, 1))
print("PCA set has", len(X_pca[0]), 
      "variables and F1 score of:", round(f1score_b * 100, 1))

cm = confusion_matrix(y_test, y_pred)

plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(10, 10))
plt.show()

rws = cm.sum(axis=1, keepdims=True)
norm = cm / rws

np.fill_diagonal(norm, 0)
plt.matshow(norm, cmap=plt.cm.gray)
plt.show()
