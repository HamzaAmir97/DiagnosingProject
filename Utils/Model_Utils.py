import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from Utils.Constants import Constants

def get_classes_Names(classes, classes_cat, labels):
    classes_names = {}
    for key in range(int(classes[0]), int(classes[-1] + 1)):
        key_t = str(classes_cat[0][key - 1]).replace('[\'', '').replace('\']', '')
        value = np.count_nonzero(labels == key)
        classes_names[key_t] = value
    return classes_names


def calc_ROC_data(y_test, pred1, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(pred1))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def save_ROC_curve_plot(plt, randomline=True):
    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    filename = str(Constants.OUTPUT_PATH) + str(Constants.SESSION_NAME) + "/ROC_CURVE_" + str(Constants.MODEL_NAME) + ".png"
    plt.savefig(filename)


def prepare_Train_Dirs():
    os.makedirs(Constants.OUTPUT_PATH+Constants.SESSION_NAME  + "\\", exist_ok=True)


def increaseExecutionNums():
    if  os.path.exists(Constants.EXEC_FILE) == False:
        open(Constants.EXEC_FILE, "w").write("1")
    else:
        open(Constants.EXEC_FILE, "w").write(str(Constants.EXEC_NUMS + 1))
    EXEC_NUMS = int(open(Constants.EXEC_FILE, "r").readline())
    SESSION_NAME = "Execution_"+str(Constants.EXEC_NUMS)

