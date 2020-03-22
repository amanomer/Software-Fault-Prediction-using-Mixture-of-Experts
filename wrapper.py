import os

# import tester as test
import time

cwd = os.getcwd()

# These files with .csv extension should be in folder 'Datasets' in same directory.
# Dataset should not contain attribute names.
# Last column should have class labels.
# Label should numeric 0/1
# 0 shows non faulty
dataset_name = ["Ant-1.7"]
# "Camel-1.2","camel-1.4","camel-1.6","CM1", "eclipse-2.0","eclipse-2.1","eclipse-3.0", "Equinox_Framework","ivy-2.0","JDT_Core","jedit-4.3","JM1",
#               "KC1","KC2","KC3","Lucene","MC1","MC2","MW1","mylyn","PC1", "PC2","PC3","PC4","PC5","PDE_UI","poi-3.0","prop-1","prop-2", "prop-2","prop-3","prop-4","prop-5","prop-6",
#             "synapse-1.2","velocity-1.6","xalan-2.4","xalan-2.5","xalan-2.6","xalan-2.7","xerces-1.4"]
print(dataset_name)

# list_DS = [0, 0.2, 0.4, 0.6, 0.8, 1]    #Data Selection threshold
# list_LP = [0, 0.2, 0.4, 0.6, 0.8, 1]    #Label Prediction threshold

list_DS = [0.2]  # Data Selection threshold
list_LP = [0.6]  # Label Prediction threshold

# DT = Decision Tree
# MLP = Multi-layer Perceeptron
expert_model = ["MLP", "DT", "NB", "svm"]  # Base model

# indi = Individual/single model
# bag = Bagging
# ME = Mixture of Experts
agg_type = ["indi", "bag", "ME", "boost", "stack"]  # Aggregation/ensemble type
log_file_name = time.strftime("%Y%m%d-%H%M%S")
log_file = "logs\\" + log_file_name + ".log"

for d in dataset_name:
    for e in expert_model:
        for a in agg_type:
            for t1 in list_LP:
                for t2 in list_DS:
                    cmd = 'python ' + cwd + '\\tester.py ' + d + ' ' + e + ' ' + a + ' ' + str(t1) + ' ' + str(t2) + " > " + log_file
                    print(cmd)
                    os.system(cmd)
