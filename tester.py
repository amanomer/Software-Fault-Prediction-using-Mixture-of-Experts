import os
import time
import statistics
import numpy as np
import helpers as hlp  # helpers.py requires
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

print("we are in tester")

# cwd = os.getcwd()
# Reading arguments passed from wrapper.py
dataset_name = sys.argv[1]
expert_model = sys.argv[2]
agg_type = sys.argv[3]
LP_threshold = float(sys.argv[4])
DS_threshold = float(sys.argv[5])
print(dataset_name, expert_model, agg_type, LP_threshold, DS_threshold)

cwd = os.getcwd()

# 'Results' folder should be in same directory
# Output will be stored in 'Results' folder in file like 'CM1_ME_DT.csv'
outfile = open(cwd + "/Results/" + dataset_name + "_" + expert_model + "_" + agg_type + ".csv", "a+")
# outfile = open(cwd + dataset_name + "_" + expert_model + "_" + agg_type +".csv" , "a+")
# print(outfile)
# Sampling required
sr = "yes"
# Standardization
st = "yes"
# Feature selection using PCA
pr = "no"
# normailzation
nr = "no"

X, Y, color = hlp.loaddata(dataset_name=dataset_name,
                           sampling_required=sr,
                           standardisation=st,
                           pca_reduced=pr,
                           normalization=nr)

print("Data loaded with:", end=" ")
if sr == "yes":
    print("Sampling", end=" ")
if st == "yes":
    print("Standardisation", end=" ")
if pr == "yes":
    print("Reduced features", end=" ")
if nr == "yes":
    print("Normalization", end=" ")

# outfile.write("DS_threshold,LP_threshold,Accuracy,f1-score,Precision,Recall,AUC,Time(secs),TP,FP,FN,TN\n")

# Number of base learners = 10
for n in range(10, 30, 2):
    # for n in range(10):
    start_time = time.time()
    kf = KFold(n_splits=5, random_state=None, shuffle=True)  # 5-fold CV
    acc = []
    prec = []
    rec = []
    f1 = []
    AUC = []
    # TP= []
    # TN= []
    # FP= []
    # FN= []
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if agg_type == "ME":
            Y_pred = hlp.ME_model(X_train, Y_train, X_test,
                                  n_estimator=n,
                                  expert_model=expert_model,
                                  _DS=DS_threshold,
                                  _LP=LP_threshold)

        elif agg_type == "bag":
            Y_pred = hlp.bag_model(X_train, Y_train, X_test,
                                   n_estimator=n,
                                   expert_model=expert_model)

        elif agg_type == "indi":
            Y_pred = hlp.indi_model(X_train, Y_train, X_test,
                                    expert_model=expert_model)

        elif agg_type == "boost":
            Y_pred = hlp.boost_model(X_train, Y_train, X_test,
                                     n_estimator=n,
                                     expert_model=expert_model)
        elif agg_type == "stack":
            Y_pred = hlp.stack_model(X_train, Y_train, X_test,
                                     n_estimator=n,
                                     expert_model=expert_model)

        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_test[i] == 1:
                TP += 1
            if Y_pred[i] == 1 and Y_test[i] != Y_pred[i]:
                FP += 1
            if Y_test[i] == Y_pred[i] == 0:
                TN += 1
            if Y_pred[i] == 0 and Y_test[i] != Y_pred[i]:
                FN += 1
        acc.append(accuracy_score(Y_test, Y_pred))
        prec.append(precision_score(Y_test, Y_pred, pos_label=1))
        rec.append(recall_score(Y_test, Y_pred, pos_label=1))
        f1.append(f1_score(Y_test, Y_pred, pos_label=1))
        AUC.append(roc_auc_score(Y_test, Y_pred))
        # TP.append(TP)
        # TN.append(TN)
        # FP.append(FP)
        # FN.append(FN)
    outfile.write(str(DS_threshold) +
                  "," + str(LP_threshold) +
                  "," + str(round(statistics.mean(acc), 4)) +
                  "," + str(round(statistics.mean(f1), 4)) +
                  "," + str(round(statistics.mean(prec), 4)) +
                  "," + str(round(statistics.mean(rec), 4)) +
                  "," + str(round(statistics.mean(AUC), 4)) +
                  "," + str(round(time.time() - start_time, 3)) +
                  "," + str(TP) +
                  "," + str(FP) +
                  "," + str(FN) +
                  "," + str(TN) + "\n")
outfile.close()