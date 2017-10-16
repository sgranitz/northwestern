# Bank Marketing Study
# as described in Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python (Miller 2015)

# Set seed value for random number generators to obtain reproducible results
RANDOM_SEED = 85

# import packages 
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn import svm

# Import data set
path = 'C:/Users/sgran/Desktop/northwestern/predict_422/assignment_4/'
data = pd.read_csv(path + 'bank.csv', sep = ';')

# Examine the shape of original input data
print('--Observations, Variables--\n', data.shape)

# drop observations with missing data, if any
data = data.dropna()

# See if any observations were dropped
print('--Observations--\n', data.shape[0])
print(data.info())

# look at the beginning of the DataFrame and quick stats
print('\n--Head of data set--\n', data.head())
print('\n--Descriptive Statistics--\n', data.describe())

# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}

# define binary variable for having credit in default
credit_default = data['default'].map(convert_to_binary)
print('\nTotal defaults:', credit_default.sum())
print(round(100 * credit_default.sum() / data.shape[0]), 
      "% have credit in default")

# define balance variable
balance = data['balance']
print('\nAverage balance:', round(balance.mean(), 2))
balance.describe()

# define binary variable for having a mortgage
mortgage = data['housing'].map(convert_to_binary)
print('\nTotal with mortgage:', mortgage.sum())
print(round(100 * mortgage.sum() / data.shape[0]), 
      "% have a mortgage")

# define binary variable for having a personal loan
loan = data['loan'].map(convert_to_binary)
print('\nTotal with personal loan:', loan.sum())
print(round(100 * loan.sum() / data.shape[0]), 
      "% have a personal loan")

# define response variable to use in the model
response = data['response'].map(convert_to_binary)
print('\nResponse counts:\n', data.response.value_counts())
print(round(100 * response.sum() / data.shape[0]), 
      "% subscribed to a term deposit")

# gather three explanatory variables and response
model_data = np.array([
  np.array(credit_default),
  np.array(balance), 
  np.array(mortgage), 
  np.array(loan), 
  np.array(response)
]).transpose()

# examine the shape of model_data, which we will use in subsequent modeling
print('--Observations, Variables--\n', model_data.shape)

### LOGISTIC REGRESSION ###
print('\n\n### LOGISTIC REGRESSION ###\n')
X = model_data[:, 0:4]
y = model_data[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

log_reg = LogisticRegression() 
log_reg.fit(X_train, y_train)
y_prob = log_reg.predict_proba(X_test)
print('\nProbabilities:\n', y_prob)

yhat = log_reg.predict(X_test)
print('\nNumber of yes predictions:', yhat.sum())
print('\nAccuracy of predictions:', metrics.accuracy_score(y_test, yhat))
print('\nConfusion Matrix:\n', metrics.confusion_matrix(y_test, yhat))
print('\nClassification Report:\n', metrics.classification_report(y_test, yhat))
print('\nAUROC:', roc_auc_score(y_test, y_prob[:, 1]))

# Plot confusion matrix
# From sklearn 
def plot_confusion_matrix(cm, classes, model,
                          title='Confusion matrix: ',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + model)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Plot non-normalized confusion matrix
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(
  cm=metrics.confusion_matrix(y_test, yhat),
  classes=['no','yes'],
  model='Logistic Regression')
plt.show()

print('\n----Softmax----')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Try various methods of logistic regression
softmax_reg = LogisticRegression(
  multi_class="multinomial", 
  solver="lbfgs", 
  C=10) 
softmax_reg.fit(X_train, y_train)
y_prob2 = softmax_reg.predict_proba(X_test)
print('\nProbabilities:\n', y_prob2)

yhat2 = softmax_reg.predict(X_test)
print('\nNumber of yes predictions:', yhat2.sum())
print('\nAccuracy of predictions:', metrics.accuracy_score(y_test, yhat2))

print('\n----Cross Validation----')
yhat_cross = cross_validation.cross_val_predict(
  LogisticRegression(), 
  X, y, cv=10
)
print('\nNumber of yes predictions:', yhat_cross.sum())
print('\nAccuracy of predictions:', metrics.accuracy_score(y, yhat_cross))
print('\nAUROC:', roc_auc_score(y_test, y_prob2[:, 1]))

### SUPPORT VECTOR MACHINES ###
print('\n\n### SVM ###\n')

print('\n----Linear SVM----')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svm_clf = Pipeline((
  ("scaler", StandardScaler()), 
  ("linear_svc", svm.SVC(C=1, kernel='linear', probability=True)),
))

svm_clf.fit(X_train, y_train)
yhat = svm_clf.predict(X_test)
y_prob = svm_clf.predict_proba(X_test)
print('\nNumber of yes predictions:', yhat.sum())
print('\nAccuracy of predictions:', metrics.accuracy_score(y_test, yhat))
print('\nConfusion Matrix:\n', metrics.confusion_matrix(y_test, yhat))
print('\nClassification Report:\n', metrics.classification_report(y_test, yhat))
print('\nAUROC:', roc_auc_score(y_test, y_prob[:, 1]))

print('\n----RBF SVM----')
svm_clf = Pipeline((
  ("scaler", StandardScaler()), 
  ("rbf", svm.SVC(C=1, kernel='rbf', probability=True)),
))

svm_clf.fit(X_train, y_train)
yhat = svm_clf.predict(X_test)
y_prob = svm_clf.predict_proba(X_test)
print('\nNumber of yes predictions:', yhat.sum())
print('\nAccuracy of predictions:', metrics.accuracy_score(y_test, yhat))
print('\nConfusion Matrix:\n', metrics.confusion_matrix(y_test, yhat))
print('\nClassification Report:\n', metrics.classification_report(y_test, yhat))
print('\nAUROC:', roc_auc_score(y_test, y_prob[:, 1]))

print('\n----Polynomial SVM----')
svm_poly_clf = Pipeline((
  ("poly_features", PolynomialFeatures(degree=3)),
  ("scaler", StandardScaler()), 
  ("poly", svm.SVC(C=1, kernel='linear', probability=True)),
))

svm_poly_clf.fit(X_train, y_train)
yhat = svm_poly_clf.predict(X_test)
y_prob = svm_poly_clf.predict_proba(X_test)
print('\nNumber of yes predictions:', yhat.sum())
print('\nAccuracy of predictions:', metrics.accuracy_score(y_test, yhat))
print('\nConfusion Matrix:\n', metrics.confusion_matrix(y_test, yhat))
print('\nClassification Report:\n', metrics.classification_report(y_test, yhat))
print('\nAUROC:', roc_auc_score(y_test, y_prob[:, 1]))

### CROSS VALIDATION ###
print('\n\n### CROSS VALIDATION ###\n')
# Adapted from Scikit Learn documentation
names = ["LinearSVC", "RBF", "Polynomial", "Logistic_Regression"]
models = [
  Pipeline((
    ("scaler", StandardScaler()), 
    ("linear_svc", svm.SVC(
      C=1, kernel='linear', 
      probability=True,
      random_state=9999
    )),
  )),
  Pipeline((
    ("scaler", StandardScaler()), 
    ("rbf_svc", svm.SVC(
      C=1, kernel='rbf', 
      probability=True,
      random_state=9999
    )),
  )),
  Pipeline((
    ("scaler", StandardScaler()), 
    ("poly_svc", svm.SVC(
      C=3, kernel='poly',
      degree=3, coef0=1.5,
      probability=True
    )),
  )),
  LogisticRegression()
]

# Shuffle the rows
np.random.seed(RANDOM_SEED)
np.random.shuffle(model_data)
num_folds = 10

# set up numpy array for storing results
results = np.zeros((num_folds, len(names)))

kf = KFold(
  n_splits=num_folds, 
  shuffle=False, 
  random_state=RANDOM_SEED
)

i = 0
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', i, '-----')
  
    X_train = model_data[train_index, 0:X.shape[1]]
    y_train = model_data[train_index, X.shape[1]]
    
    X_test = model_data[test_index, 0:X.shape[1]]
    y_test = model_data[test_index, X.shape[1]]   
    
    print('\nShape of input data for this fold:\n')
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('\nX_test:', X_test.shape)
    print('y_test:', y_test.shape)

    j = 0
    for name, model in zip(names, models):
        print('\nClassifier evaluation for:', name)
        print('\nScikit Learn method:', model)
        model.fit(X_train, y_train)
        
        # evaluate on the test set for this fold
        y_prob = model.predict_proba(X_test)
        print('\nNumber of yes predictions:', model.predict(X_test).sum())
        auroc = roc_auc_score(y_test, y_prob[:, 1]) 
        print('Area under ROC curve:', auroc)
        results[i, j] = auroc
        
        j += 1
    i += 1

df = pd.DataFrame(results)
df.columns = names

print('\n----------------------------------------------')
print('Average results from ', 
      num_folds, 
      '-fold cross-validation\n',
      '\nMethod                 Area under ROC Curve', sep = '')     
print(df.mean())   

#Get recommendations using Logistic
log_reg = LogisticRegression() 
log_reg.fit(X, y)
y_prob = log_reg.predict_proba(X)

cols = ['default', 'balance', 'housing', 'loan', 'response']
recs = data[cols]

recs.default = recs['default'].map(convert_to_binary)
recs.housing = recs['housing'].map(convert_to_binary)
recs.loan = recs['loan'].map(convert_to_binary)
recs.response = recs['response'].map(convert_to_binary)

recs['prob'] = y_prob[:, 1]
recs = recs.sort_values(['prob'], ascending=False)
cutoff = recs.prob.mean()
rec_yes = recs[recs.prob >= cutoff]
rec_no = recs[recs.prob < cutoff]

print(rec_yes.describe())
print("% yes TP:", sum(rec_yes.response == 1)/len(rec_yes))
print(rec_no.describe())
print("% no FN:", sum(rec_no.response == 1)/len(rec_no))

recs_piv = recs.set_index(['response', 'default', 'housing', 'loan']).sort_index()

# Test on set of possible customer combinations
give_recs = {
  'default': [1,0,0,0,1,1,1,0,0,0,1,1,1,0,1],
  'housing': [0,1,0,0,1,0,0,1,1,0,1,1,0,1,1],
  'loan': [0,0,1,0,0,1,0,1,0,1,1,0,1,1,1],
  'balance': [0,0,0,1000,0,0,1000,0,1000,1000,0,1000,1000,1000,1000]
}
give_recs = pd.DataFrame(give_recs)
give_recs = give_recs[['default', 'balance', 'housing', 'loan']]
y_prob = log_reg.predict_proba(give_recs)
give_recs['probability'] = y_prob[:, 1]
give_recs.probability = round(give_recs.probability * 100, 2)
give_recs = give_recs.sort_values(['probability'], ascending=False)

# Print table of recs
print(tabulate(give_recs.reset_index(drop=True), headers=give_recs.columns, tablefmt='psql'))

#Test on known responses
rec_yes = recs[recs.response == 1]
rec_no = recs[recs.response == 0]

print(rec_yes.describe())
print("% yes TP:", sum(rec_yes.response == 1)/len(rec_yes))
print(rec_no.describe())
print("% no FN:", sum(rec_no.response == 1)/len(rec_no))

#Get recommendations using Polynomial
svm_poly_clf = Pipeline((
  ("poly_features", PolynomialFeatures(degree=3)),
  ("scaler", StandardScaler()), 
  ("linear_svc", svm.SVC(C=1, kernel='linear', probability=True)),
))

svm_poly_clf.fit(X, y)
y_prob = svm_poly_clf.predict_proba(X)

recs['prob'] = y_prob[:, 1]
recs = recs.sort_values(['prob'], ascending=False)
cutoff = recs.prob.mean()
rec_yes = recs[recs.prob >= cutoff]
rec_no = recs[recs.prob < cutoff]

print(rec_yes.describe())
print("% yes TP:", sum(rec_yes.response == 1)/len(rec_yes))
print(rec_no.describe())
print("% no FN:", sum(rec_no.response == 1)/len(rec_no))
