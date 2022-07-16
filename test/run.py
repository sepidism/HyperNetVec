from __future__ import print_function
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
import random 
from sklearn import preprocessing

print("%%%%%% Starting Evaluation %%%%%%")
print("Loading data...")

total = 1434
max = 0.0
min = 100.0
sum = 0.0
std = []
filename = sys.argv[1]
newX = []
f = open(filename,"r")
for line in f:
        newl = line.split()
        newX.append(newl)
X_train = np.array(newX)

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

for it in range(100) :
  crossval = open("train"+str(it)+".txt", "r")
  line = crossval.readline()
  sp = line.split()
  tests = int(sp[0])
  train = int(sp[1])

  ftest = {}
  ftrain = {}

  for i in range(0,total):
    line = crossval.readline()
    sp = line.split()
    if i < tests:
      ftest[int(sp[0])] = 1
    else :
      ftrain[int(sp[0])] = 1
  j = 0
  train_labels = []
  test_labels = []
  mytrainlabel = []
  mytestlabel = []
  train_embeds = []
  test_embeds = []


  for line in X_scaled:
      if j in ftest:
        test_embeds.append(line)
      else:
        train_embeds.append(line)
      j = j + 1
  j = 0
  lf = open("labels.txt","r")
  for line in lf:
      if j in ftest:
        test_labels.append(int(line))
      else:
        train_labels.append(int(line))
      j = j + 1


  train_ids    =  [i for i in range(0,len(train_embeds))]
  test_ids    =  [i for i in range(0,len(test_embeds))]
  log         = LogisticRegression(solver='lbfgs', multi_class='auto', dual=False, max_iter=7600)
  log.fit(train_embeds, train_labels)
  pred_labels = (log.predict(test_embeds)).tolist()
  acc         = accuracy_score(test_labels, pred_labels)
  if acc > max:
      max = acc
  if acc < min:
      min = acc
  sum += acc
  std.append(acc)
  print("Test Accuracy: ", acc)
fstd = np.std(std)
print("Max accuracy: ", str(max))
print("Min accuracy: ", str(min))
ss = sum / 100;
print("Avg accuracy: ", ss)
print("standard dev: ", str(fstd))

