from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from Utils.Model_Utils import get_classes_Names
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from Utils.Constants import Constants, MODELS
from sklearn.pipeline import Pipeline
from scipy.io import loadmat
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



def  loadData():
    data = loadmat(Constants.FEATURES_PATH_MAT)
    classes_cat = data['TisCatsName']
    # classes_cat = np.array([['ADI' , 'BACK' , "LYM"]])
    data = pd.DataFrame(data['source_and_target'])
    # data = pd.read_csv(Constants.FEATURES_PATH_CSV)

    Train_DS, Test_DS = train_test_split(data, train_size=Constants.TRAIN_SIZE,
                                                      shuffle=True, random_state=0)

    data = dataClean(data)
    Train_DS = dataClean(Train_DS)
    Test_DS = dataClean(Test_DS)


    classes = np.unique(data.iloc[:, -1].values)
    Train_Features, Train_labels = Train_DS.iloc[:, :-1].values, Train_DS.iloc[:, -1].values
    Test_Features, Test_labels = Test_DS.iloc[:, :-1].values, Test_DS.iloc[:, -1].values
    Train_classes_names = get_classes_Names(classes, classes_cat, Train_labels)
    Test_classes_names = get_classes_Names(classes, classes_cat, Test_labels)
    Dataset_classes_names = get_classes_Names(classes, classes_cat, data.iloc[:, -1].values)


    return {"DataSet" : data ,
                    "Train_DS" : Train_DS ,
                    "Test_DS" : Test_DS ,
                    "classes" : classes ,
                    "Train_classes_names" : Train_classes_names,
                    "Test_classes_names" : Test_classes_names ,
                    "Dataset_classes_names" : Dataset_classes_names}


def dataClean(df):
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df)

    return pd.DataFrame(imputer.transform(df))



def getCurrentModel():
    if Constants.MODEL_NAME == MODELS['SVM']:
        return ('SVM', svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-5,
                                     C=1.0, multi_class='ovr', random_state=0))
    elif Constants.MODEL_NAME == MODELS['RBFSVM']:
        return ('RBFSVM', svm.SVC(kernel="rbf", gamma='auto', C=100))
    elif Constants.MODEL_NAME == MODELS['DT']:
        return ('DT', DecisionTreeClassifier(criterion='entropy', random_state=0))
    elif Constants.MODEL_NAME == MODELS['RF']:
        return ('RF', RandomForestClassifier(n_estimators=322,max_features='sqrt', criterion='entropy', random_state=0 , n_jobs=-1))
    elif Constants.MODEL_NAME == MODELS['ETC']:
        return ('ETC', ExtraTreesClassifier(n_estimators=322,max_features=1.0, random_state=0 , criterion='entropy', n_jobs=-1))
    elif Constants.MODEL_NAME == MODELS['KNN']:
        return ('KNN', KNeighborsClassifier(n_neighbors=5 , n_jobs=-1,weights='uniform' ,
                                            algorithm='ball_tree' , p=1 , metric='minkowski'))
    else:
        return None



def Model_Initlizer():

    model = getCurrentModel()
    pca = PCA(n_components=74)
    if Constants.MODEL_NAME == MODELS['RBFSVM']:
        pca = PCA(n_components=37)

    steps = [('ss', StandardScaler(copy=True, with_mean=True, with_std=True)), ('pca', pca), model]
    # steps = [('pca', pca), model]

    return Pipeline(steps=steps) , RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)


def Data_Splitting(Train_DS, Test_DS):

    Train_Features, Train_labels = Train_DS.iloc[:,:-1].values,Train_DS.iloc[:,-1].values
    Test_Features, Test_labels = Test_DS.iloc[:,:-1].values,Test_DS.iloc[:,-1].values

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler2 = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(Train_Features)
    scaler2.fit(Test_Features)

    Train_Features = scaler.transform(Train_Features)

    # Scalling Test Data Alone
    Test_Features = scaler2.transform(Test_Features)


    return {"Train_Features" : Train_Features , "Test_Features" : Test_Features ,
                   "Train_labels" : Train_labels , "Test_labels": Test_labels }
