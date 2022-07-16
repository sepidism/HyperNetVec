from __future__ import print_function
import sys
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
import random 
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statistics
# import the embeddings
newX = []
filename = sys.argv[1]
f = open(filename,"r")
for line in f:
      newl = line.split()
      newX.append(newl)
X_train = np.array(newX)
#hedgenum = 1154 #137404 
scaler = preprocessing.StandardScaler().fit(X_train)
#embeddings = scaler.transform(X_train)
embedding_train = scaler.transform(X_train)
#print(embedding_train)
#print(embedding)
#embedding_train = []
#for a in embedding:
#  embedding_train.append(a)

#positive and negative samplesi for test:

examples_test = []
labels_test = []
examples_model_selection = []
labels_model_selection = []
pos = open("pos.txt","r")
#all_test = 175689#205712#47887#175689 #177234 #1242
#examp = all_test * 0.2
cc = 0
for line in pos:
      newl = line.split()
      if len(newl) == 3:
        #if (cc < examp/2):
        #  examples_model_selection.append(newl)
        #  labels_model_selection.append(1)
        #  cc +=1
        #else:
          examples_test.append(newl)  
          labels_test.append(1)
      else:
          print("not 3 in pos test")
cc = 0
neg = open("neg.txt","r")
for line in neg:
      newl = line.split()
      if len(newl) == 3:
        #if cc < examp/2:
        #  examples_model_selection.append(newl)
         # labels_model_selection.append(0)
         # cc +=1
        #else:
          examples_test.append(newl)
          labels_test.append(0)
      else:
          print("not 3 in neg test")


#positive and negative samplesi for train:
examples_train = []
labels_train = []
cc = 0
trainpos = open("trainpos.txt","r")
for line in trainpos:
      newl = line.split()
      if len(newl) == 3:
        if cc < 20:
          examples_model_selection.append(newl)
          labels_model_selection.append(1)
          cc +=1
        else:
          examples_train.append(newl)
          labels_train.append(1)
      else:
          print("not 3 in trainpos")
cc = 0
trainneg = open("trainneg.txt","r")
for line in trainneg:
      newl = line.split()
      if len(newl) == 3:
        if cc < 20:
          examples_model_selection.append(newl)
          labels_model_selection.append(0)
          cc +=1
        else:
          examples_train.append(newl)
          labels_train.append(0)
      else:
          print("not 3 in trainNeg")


def get_embedding(u):
    return embedding_train[u]

def link_examples_to_features(link_examples, transform_node, binary_operator):
    ret = []
    for src, mid, dst in link_examples:
      #print(embedding_train[int(mid)])
      res = binary_operator(embedding_train[int(src)], embedding_train[int(mid)], embedding_train[int(dst)])
      ret.append(res)
    return ret

def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf

def link_prediction_classifier(max_iter=7000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

def operator_hadamard(u, c, v):
    return u * v * c


def operator_l1(u, c, v):
    return np.abs(u - c - v)


def operator_l2(u, c, v):
    return (u - c - v) ** 2


def operator_avg(u, c, v):
    return (u + c + v) / 3.0

def operator_med(u, c, v):
    arr = []
    for i in range(len(u)):
        arr.append(np.median([u[i],c[i],v[i]]))
    return arr

def operator_max(u, c, v):
    arr = []
    for i in range(len(u)):
        arr.append(max(u[i],c[i],v[i]))
    return arr

def operator_min(u, c, v):
    arr = []
    for i in range(len(u)):
        arr.append(min(u[i],c[i],v[i]))
    return  arr 

def operator_var(u, c, v):
    arr = []
    for i in range(len(u)):
        tmp = []
        tmp.append(u[i])
        tmp.append(c[i])
        tmp.append(v[i])
        arr.append(np.var(tmp, dtype=np.float64))
    return  arr 

def run_link_prediction(binary_operator):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


binary_operators = [operator_var]
#binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
#binary_operators = [operator_med, operator_max, operator_min]
#binary_operators = [operator_avg]
results = [run_link_prediction(op) for op in binary_operators]
best_result = max(results, key=lambda result: result["score"])

print(f"Best result from '{best_result['binary_operator'].__name__}'")



test_score = evaluate_link_prediction_model(
    best_result["classifier"],
    examples_test,
    labels_test,
    embedding_train,
    best_result["binary_operator"],
)
print(
    f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
)


