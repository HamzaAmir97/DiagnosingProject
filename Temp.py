# # # import pandas as pd
# # # import numpy as np
# # # from sklearn import svm
# # # from sklearn.preprocessing import StandardScaler
# # # from scipy.io import loadmat
# # # from sklearn.model_selection import cross_val_score
# # # from sklearn.model_selection import RepeatedStratifiedKFold
# # # from sklearn.pipeline import Pipeline
# # # from sklearn.decomposition import PCA
# # #
# # # data = loadmat('Features/BreastCancer_glcmRotInv/glcmRotInv_numFeatures20_last_output_rand_73400.mat')
# # # data = pd.DataFrame(data['source_and_target'])
# # # Features, labels = data.iloc[:, :-1].values, data.iloc[:, -1].values
# # # print(data.describe())
# # # print(data.head(10))
# # # print(data.info())
# # # print(np.unique(labels))
# # #
# # # validations_acc = []
# # # for n_components in range(Features.shape[1], 0, -1):
# # #     # ---------------------------------- Data Pre-processing ------------------------------------
# # #
# # #     # define the pipeline
# # #     steps = [('ss', StandardScaler(copy=True, with_mean=True, with_std=True)),
# # #              ('pca', PCA(n_components=n_components)), ('m', svm.SVC(kernel='rbf'))]
# # #     model = Pipeline(steps=steps)
# # #
# # #     # evaluate model
# # #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # #     validation_accuracy = cross_val_score(model, Features, labels, scoring='accuracy', cv=cv, n_jobs=-1,
# # #                                           error_score='raise')
# # #
# # #     cc = 'n_components = %1.0f , Validation Accuracy: %1.1f%% (+/- %1.1f%%)' % (n_components,
# # #                                                                                 np.mean(validation_accuracy) * 100,
# # #                                                                                 np.std(validation_accuracy) * 100)
# # #     print(cc)
# # #     open('validation_accuracy_glcm.txt', 'a').write(cc + '\n')
# #
# #
# # import pandas as pd
# # import numpy as np
# # from skimage import io
# # from skimage.io import imsave
# # from sklearn import metrics, svm
# # from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# # from sklearn.externals._pilutil import imresize
# # from sklearn.feature_selection import SelectFromModel
# # from sklearn.impute import SimpleImputer
# # from sklearn.model_selection import train_test_split
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.preprocessing import StandardScaler
# # from scipy.io import loadmat
# # from sklearn.model_selection import cross_val_score
# # from sklearn.model_selection import RepeatedStratifiedKFold
# # from sklearn.pipeline import Pipeline
# # from sklearn.decomposition import PCA
# #
# # # ------------------------------------------------------------------------------
# #
# #
# # features = ['colon_lung_best5/best5_numFeatures74_last_output_rand_37555.mat', ]
# # train_size = 0.7
# # test_size = 0.2
# #
# # for n_estimators in range(74, 75):
# #     # ---------------------------------- Data Definition ------------------------------------
# #
# #     # data = loadmat(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\best5_numFeatures74_last_output_rand_37555.mat')
# #     # data = pd.DataFrame(data['source_and_target'])
# #
# #
# #     data = pd.read_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\csv_colon_with_lung_colon_224_768.csv')
# #     print('Before: \n',data['74'].value_counts())
# #     data.dropna(axis=0,inplace=True)
# #     print('After: \n', data['74'].value_counts())
# #     Features, labels = data.iloc[:,1:-1].values, data.iloc[:, -1].values
# #     # -----------------------------------------------------
# #
# #     # ---------------------------------- Data Pre-processing ------------------------------------
# #
# #     steps = [('ss', StandardScaler(copy=True, with_mean=True, with_std=True)),
# #              ('pca', PCA(n_components=74)),
# #              ('et', ExtraTreesClassifier(n_estimators=322, random_state=42))]
# #
# #     model = Pipeline(steps=steps)
# #
# #     # evaluate model
# #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
# #
# #     # ---------------------------------- Data Splitting ------------------------------------
# #
# #     X_train, X_rem, y_train, y_rem = train_test_split(Features, labels, train_size=train_size,
# #                                                       shuffle=True, random_state=42)
# #     X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=test_size,
# #                                                         shuffle=True, random_state=42)
# #
# #     print('X_train.shape = ',X_train.shape)
# #     print('X_rem.shape = ', X_rem.shape)
# #     print('X_valid.shape = ', X_valid.shape)
# #     print('X_test.shape = ', X_test.shape)
# #
# #     scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# #     scaler.fit(X_train)
# #
# #     X_train = scaler.transform(X_train)
# #     X_valid = scaler.transform(X_valid)
# #     X_test = scaler.transform(X_test)
# #
# #     classifier = model
# #
# #     # Performing training
# #     validation_accuracy = cross_val_score(classifier, X_valid, y_valid,
# #                                           scoring='accuracy', cv=cv, n_jobs=-1,
# #                                           error_score='raise')
# #
# #     classifier.fit(X_train, y_train)
# #
# #
# #
# #     y_pred = classifier.predict(X_test)
# #
# #     training_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
# #
# #     # ------------------------------------------------------------------------
# #
# #     cc = 'Trainig Accuracy: %1.1f%% , Validation Accuracy: %1.1f%% (+/- %1.1f%%) ' % (
# #         training_accuracy,
# #         np.mean(validation_accuracy) * 100,
# #         np.std(validation_accuracy) * 100)
# #     print(cc)
# #     open('validation_accuracy_best5_ExtraTree_colon_with_lung_colon_224_768.txt', 'a').write(cc + '\n')
# #
# #
# #
# #
# #
# # #
# # # data = loadmat('Features/best5_numFeatures74_colon_45k_224_224/best5_numFeatures74_colon_45k_224_224.mat')
# # # # data = loadmat('Features/best5_numFeatures74_colon_45k_224_224/best5_numFeatures74_last_output_rand_74481.mat')
# # #
# # #
# # # data = pd.DataFrame(data['source_and_target'])
# # #
# # # imputer = SimpleImputer(strategy="mean")
# # # imputer.fit(data)
# # # data =  pd.DataFrame(imputer.transform(data))
# # #
# # # classes = np.unique(data.iloc[:,-1].values)
# # #
# # #
# # # indexes = [j for j,i in enumerate(data.iloc[:, -1].values) if i not in (1,2,4) ]
# # #
# # # # data.dropna(axis=0,inplace=True)
# # # data = data.drop(indexes,axis=0)
# # #
# # # # data[str(data.shape[1]-2)] = data[str(data.shape[1]-2)] + max(classes2)
# # #
# # #
# # # for i in range(data.shape[0]):
# # #     if data.iloc[i,data.keys()[-1]] == 4:
# # #         data.iloc[i,data.keys()[-1]] = 3
# # #
# # # print(data[74].value_counts())
# # # # data.to_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\csv_lung_colon_224.csv')
# # # data.to_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\best5_colon_(1,2,3)_224_224.csv')
# # #
# #
# #
# #
# #
# # #
# # # data1 = pd.DataFrame(pd.read_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\best5_lung_colon_224_224.csv'))
# # # data2 = pd.DataFrame(pd.read_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\best5_colon_(1,2,3)_224_224.csv'))
# # # # print(data1.shape)
# # # # indexes = []
# # # # cls = np.unique(data1.iloc[:,-1].values)
# # # # classes = dict()
# # # # for key , value in zip(cls,np.full(len(cls),5000-625)):
# # # #     classes[key] = value
# # # #
# # # # for i in range(data1.shape[0]):
# # # #     id = data1.iloc[i,-1]
# # # #     cc = classes[id]
# # # #     if classes[id] <= 0: continue
# # # #     classes[id] -= 1
# # # #     indexes.append(i)
# # #
# # # # data1 = data1.drop(indexes,axis=0)
# # # # print('data1\n',data1[str(data1.shape[1]-2)].value_counts())
# # # # print('data2\n',data2[str(data1.shape[1]-2)].value_counts())
# # #
# # #
# # # classes1 = np.unique(data1.iloc[:,-1].values)
# # # classes2 = np.unique(data2.iloc[:,-1].values)
# # #
# # # print('Befor:\n')
# # # print('classes1 : ',classes1)
# # # print('classes2 : ',classes2)
# # #
# # # data1[str(data1.shape[1]-2)] = data1[str(data1.shape[1]-2)] + max(classes2)
# # #
# # # classes1 = np.unique(data1.iloc[:,-1].values)
# # # classes2 = np.unique(data2.iloc[:,-1].values)
# # #
# # # print('After:\n')
# # # print('classes1 : ',classes1)
# # # print('classes2 : ',classes2)
# # #
# # # data2 = data2.append(data1)
# # # data2.to_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\csv_colon_with_lung_colon_224_224.csv')
# # #
# from scipy.io import loadmat
# from sklearn import svm
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, roc_curve, auc
# from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, RepeatedStratifiedKFold
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# def plot_learning_curves(model, X, y):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
#     train_errors, val_errors = [], []
#     print(len(X_train))
#     for m in range(2, len(X_train)):
#         model.fit(X_train[:m], y_train[:m])
#         print(m)
#         y_train_predict = model.predict(X_train[:m])
#         y_val_predict = model.predict(X_val)
#         train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
#         val_errors.append(mean_squared_error(y_val, y_val_predict))
#     plt.plot(np.sqrt(train_errors), "r--", linewidth=1, label="train loss")
#     plt.plot(np.sqrt(val_errors), "b-", linewidth=1, label="val loss")
#     plt.show()
#
# def dataClean(df):
#     imputer = SimpleImputer(strategy="mean")
#     imputer.fit(df)
#
#     return pd.DataFrame(imputer.transform(df))
#
#
# def calc_ROC_data(y_test, pred1, n_classes):
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(np.pd.get_dummies(y_test))[:, i], np.pd.get_dummies(pred1))[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#     return fpr, tpr, roc_auc
#
#
# clsf = svm.LinearSVC(dual=False , random_state=0 , C=.5)
# # data = loadmat(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\Lung_colon_and_colon_224_4k_images.mat')
# # data = pd.DataFrame(data['source_and_target'])
# #
# # data = dataClean(data)
# # classes = np.unique(data.iloc[:, -1].values)
# # X_train, X_val, y_train, y_val = train_test_split(data.iloc[:,:-1].values, data.iloc[:,-1].values, test_size=0.3)
# # clsf.fit(X_train,y_train)
#
# # Prediction_labels = clsf.predict(X_val)
# #
# # fpr, tpr, roc_auc = calc_ROC_data(y_val, Prediction_labels, len(classes))
# #
# # from sklearn.metrics import precision_recall_curve
# #
# #
# # def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
# #     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
# #     plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# #
# # plot_precision_recall_vs_threshold(fpr, tpr, roc_auc)
# # plt.show()
#
# d = pd.read_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\best5_colon_(1,2,3)_224_224.csv')
# d = dataClean(d)
#
# X_train, X_val, y_train, y_val = train_test_split(d.iloc[:,:-1].values, d.iloc[:,-1].values, test_size=0.1 , shuffle=True )
# clsf.fit(X_train,y_train)
#
# cv =  RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
# validation_accuracy = cross_val_score(clsf, X_train, y_train,
#                                                                     scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# print('validation_accuracy = ',validation_accuracy)
#
# loss = cross_val_score(clsf, X_train, y_train,
#                                                                     scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
# print('loss = ',loss)
#
# hyp = cross_val_score(clsf, X_train, y_train,
#                                                                     scoring=['accuracy' , 'neg_mean_squared_error'], cv=cv, n_jobs=-1, error_score='raise')
# print('hyp = ',hyp)
# exit()

# plot_learning_curves(clsf, X_val,y_val)




#
#
# print(__doc__)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV
#
#
# # Utility function to move the midpoint of a colormap to be around
# # the values of interest.
#
# class MidpointNormalize(Normalize):
#
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         Normalize.__init__(self, vmin, vmax, clip)
#
#     def __call__(self, value, clip=None):
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_np.interp(value, x, y))
#
# # #############################################################################
# # Load and prepare data set
# #
# # dataset for grid search
#
# iris = load_iris()
# X = iris.data
# y = iris.target
# print(y)
#
# # Dataset for decision function visualization: we only keep the first two
# # features in X and sub-sample the dataset to keep only 2 classes and
# # make it a binary classification problem.
#
# X_2d = X[:, :2]
# X_2d = X_2d[y > 0]
# y_2d = y[y > 0]
# y_2d -= 1
#
# # It is usually a good idea to scale the data for SVM training.
# # We are cheating a bit in this example in scaling all of the data,
# # instead of fitting the transformation on the training set and
# # just applying it on the test set.
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_2d = scaler.fit_transform(X_2d)
#
# # #############################################################################
# # Train classifiers
# #
# # For an initial search, a logarithmic grid with basis
# # 10 is often helpful. Using a basis of 2, a finer
# # tuning can be achieved but at a much higher cost.
#
# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# grid.fit(X, y)
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
#
# # Now we need to fit a classifier for all parameters in the 2d version
# # (we use a smaller set of parameters here because it takes a while to train)
#
# C_2d_range = [1e-2, 1, 1e2]
# gamma_2d_range = [1e-1, 1, 1e1]
# classifiers = []
# for C in C_2d_range:
#     for gamma in gamma_2d_range:
#         clf = SVC(C=C, gamma=gamma)
#         clf.fit(X_2d, y_2d)
#         classifiers.append((C, gamma, clf))
#
# # #############################################################################
# # Visualization
# #
# # draw visualization of parameter effects
#
# plt.figure(figsize=(8, 6))
# xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
# for (k, (C, gamma, clf)) in enumerate(classifiers):
#     # evaluate decision function in a grid
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     # visualize decision function for these parameters
#     plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
#     plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
#               size='medium')
#
#     # visualize parameter's effect on decision function
#     plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
#     plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
#                 edgecolors='k')
#     plt.xticks(())
#     plt.yticks(())
#     plt.axis('tight')
#
# scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
#                                                      len(gamma_range))
#
# # Draw heatmap of the validation accuracy as a function of gamma and C
# #
# # The score are encoded as colors with the hot colormap which varies from dark
# # red to bright yellow. As the most interesting scores are all located in the
# # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# # as to make it easier to visualize the small variations of score values in the
# # interesting range while not brutally collapsing all the low score values to
# # the same color.
#
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
#            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
# plt.xlabel('gamma')
# plt.ylabel('C')
# plt.colorbar()
# plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
# plt.yticks(np.arange(len(C_range)), C_range)
# plt.title('Validation accuracy')
# plt.show()

#
#
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# iris = load_iris()
# X = iris.data[:, 2:] # petal length and width
# y = iris.target
# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(X, y)
#
#
# from sklearn.tree import export_graphviz
# export_graphviz(
# tree_clf,
# out_file="iris_tree.dot",
# feature_names=iris.feature_names[2:],
# class_names=iris.target_names,
# rounded=True,
# filled=True
# )
# from scipy.io import loadmat
# from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# import numpy as np
# import pandas as pd
#

# param_grid = {
#    'n_neighbors': np.arange(1,30),
#    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'metric' : ['euclidean','manhattan','chebyshev','minkowski']
# }
# grid = GridSearchCV(model, param_grid = param_grid, cv=4)
# data = loadmat(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\colon_lung_best5_768_768.mat')
# data = pd.DataFrame(data['source_and_target'])
# X = data.iloc[:,:-1].values
# y = data.iloc[:,-1].values
# grid.fit(X, y)
# best_estimator = grid.best_estimator_
# print(best_estimator)

import numpy as np
import pandas as pd

rnd_clf = RandomForestClassifier(n_estimators=200,random_state=300,max_features=1.0,n_jobs=-1)
data = loadmat(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\Lung_colon_and_colon_224_4k_images.mat')
data = pd.DataFrame(data['source_and_target'])
data.dropna(axis=0,inplace=True)
# data = pd.read_csv(r'C:\Users\Dell\Spider Projects\Graduate_Project\Inputs\Features\best5_colon_(1,2,3)_224_224.csv')
rnd_clf.fit(data.iloc[:,:-1].values, data.iloc[:,-1].values)
n , max = data.keys()[0] , rnd_clf.feature_importances_[0]*100
for name, score in zip(data.keys(), rnd_clf.feature_importances_*100):
    print('%s ====> %1.4f%%'%(name, score))
    if score>max:
        max = score
        n = name

print('The most important Feature is ',n ,' ==> ' ,max)

def save_ROC_curve_plot(plt, randomline=True):
    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    filename = "ROC_CURVE_All_Classifiers" + ".png"
    plt.savefig(filename)

import matplotlib.pyplot as plt


MODELS = ["RBFSVM",
          "Decision Tree " ,
          "Random Forest" ,
          "Extra Tree " ,
          "KNN" ]

tpr = {
        0: [0., 0.2350209, 0.40105218, 0.49287881, 0.76130522, 0.81210282, 0.81228674, 0.94371679, 0.97422204, 1.],
        1: [0.        , 0.42530768, 0.516016  , 0.52800093, 0.77645497, 0.79491769, 0.82837068, 0.90428797, 0.9180876 , 1.        ],
        2: [0.        , 0.3455361 , 0.47914003, 0.50858018, 0.6604958 , 0.84557301, 0.87969961, 0.95369586, 0.97662152, 1.        ],
        3: [0.        , 0.30145317, 0.4201006 , 0.52391767, 0.7515172 ,0.85429131, 0.87934818, 0.94657546, 0.98296068, 1.        ],
        4: [0.        , 0.26160289, 0.39906862, 0.45640316, 0.60509198, 0.88637543, 0.91364579, 0.95406532, 0.95623227, 1.        ]
        }

fpr = {
       0: [0.00000000e+00, 1.90186383e-04, 4.76099791e-04, 7.63140322e-04, 2.47548320e-03, 2.94929122e-03, 2.95209980e-03, 8.44963448e-03, 1.14821548e-02, 1.00000000e+00],
       1: [0.        , 0.00323317, 0.00419727, 0.00438012, 0.01094092, 0.01161573, 0.01352252, 0.02183613, 0.02526074, 1.],
       2: [0.00000000e+00, 4.75465957e-04, 7.61759665e-04, 8.58532863e-04, 1.80935149e-03, 3.80843569e-03, 4.37636761e-03, 6.69792364e-03, 8.25975506e-03, 1.00000000e+00],
       3: [0.00000000e+00, 2.85279574e-04, 4.76099791e-04, 7.63140322e-04, 1.90458052e-03, 2.66590498e-03, 2.94929122e-03, 4.40149268e-03, 6.26602108e-03, 1.00000000e+00],
       4: [0.00000000e+00, 3.81570161e-04, 7.60745531e-04, 1.04741954e-03, 2.85687077e-03, 9.51384264e-03, 1.04433685e-02, 1.26630487e-02, 1.29174242e-02, 1.00000000e+00]
       }


roc_auc =  {
            0: 0.9852021991423813,
            1: 0.9527341531663875,
            2: 0.9865822097334298,
            3: 0.9902245031984772,
            4: 0.9748562786375231}



def plot_ROC_curve(fpr, tpr, roc_auc, classes, model_name, lw=1.5):
    n_classes = len(classes)
    plt.figure(figsize=(8, 5))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, linestyle='-', linewidth=1.3,
                 label='{0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='#726d6d')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + "_ROC_Curve")
    plt.legend(loc="lower right")
    save_ROC_curve_plot(plt)
    plt.show()


plot_ROC_curve(fpr,tpr,roc_auc,MODELS,"All Classifiers")