# For this assignment you are asked to fit classification models to data from the Bank Marketing Study
# Focus is on Logistic Regression and Naive Bayes

# Set seed value for random number generators to obtain reproducible results
RANDOM_SEED = 85

# import packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import itertools
import matplotlib.pyplot as plt

# Import data set
path = 'C:/Users/sgran/Desktop/northwestern/predict_422/assignment_2/jump-start-bank-v001/'
data = pd.read_csv(path + 'bank.csv', sep = ';')

# Examine the shape of original input data
print('--Observations, Variables--\n', data.shape)

# look at the list of column names
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

# define binary variable for having a mortgage or housing loan
mortgage = data['housing'].map(convert_to_binary)
print('\nTotal with mortgage:', mortgage.sum())
print(round(100 * mortgage.sum() / data.shape[0]), 
      "% have a mortgage or housing loan")

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
  np.array(mortgage), 
  np.array(loan), 
  np.array(response)
]).transpose()

# examine the shape of model_data
print('--Observations, Variables--\n', model_data.shape)

### LOGISTIC REGRESSION ###
print('\n\n### LOGISTIC REGRESSION ###\n')
X = model_data[:, 0:3]
y = model_data[:, 3]

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
# Based on method from sklearn 

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


### NAIVE BAYES ###
print('\n\n### NAIVE BAYES ###\n')

print('\n----Bernoulli----')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
prior = response.sum() / data.shape[0]

clf = BernoulliNB(
  alpha=1.0, 
  binarize=0.5, 
  class_prior=[1 - prior, prior], 
  fit_prior=False
)
clf.fit(X_train, y_train)

df = pd.DataFrame(X_test)
df.columns = ['credit_default', 'mortgage', 'loan']
df['response'] = y_test

# add predicted probabilities to the training sample
df['prob_NO'] = clf.predict_proba(X_test)[:,0]
df['prob_YES'] = clf.predict_proba(X_test)[:,1]
df['prediction'] = clf.predict(X_test)
print(df.head(10))
print('\nNumber of yes predictions:', df.prediction.sum())
print('\nOverall training set accuracy:', clf.score(X_test, y_test))

print('\n----Gaussian----')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
df['prob_NO'] = clf2.predict_proba(X_test)[:,0]
df['prob_YES'] = clf2.predict_proba(X_test)[:,1]
df['prediction'] = clf2.predict(X_test)

print(df.head(10))
print('\nNumber of yes predictions:', df.prediction.sum())
print('\nOverall training set accuracy:', clf2.score(X_test, y_test))

### CROSS VALIDATION ###
print('\n\n### CROSS VALIDATION ###\n')
# Adapted from Scikit Learn documentation
names = ["Naive_Bayes", "Logistic_Regression"]
models = [
  BernoulliNB(
    alpha=1.0, 
    binarize=0.5, 
    class_prior = [0.5, 0.5], 
    fit_prior=False
  ), 
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
        plt.figure()
        plot_confusion_matrix(
          cm=metrics.confusion_matrix(y_test, model.predict(X_test)), 
          classes=['no','yes'],
          model=name + str(i)
        )
        plt.show()
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
