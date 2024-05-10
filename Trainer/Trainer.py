from Utils.Model_Utils import calc_ROC_data, prepare_Train_Dirs, increaseExecutionNums
from PreProcess.Prprocess import loadData, Model_Initlizer, Data_Splitting
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from Utils.Constants import Constants
from Plotter.Plotter import Plotter
from sklearn import metrics
from wx import DateTime
import numpy as np
import os



class Trainer:

    def __init__(self):
        T = loadData()
        self.DataSet = T['DataSet']
        self.Train_DS = T['Train_DS']
        self.Test_DS = T['Test_DS']
        self.classes = T['classes']
        self.Train_classes_names = T['Train_classes_names']
        self.Test_classes_names = T['Test_classes_names']
        self.Dataset_classes_names = T['Dataset_classes_names']


        DS = Data_Splitting(self.Train_DS,self.Test_DS)
        self.Train_Features = DS["Train_Features"]
        self.Test_Features = DS["Test_Features"]
        self.Train_labels = DS["Train_labels"]
        self.Test_labels = DS["Test_labels"]

        self.Prediction_labels = ""
        self.training_accuracy = 0.0
        self.validation_accuracy = 0.0
        self.training_loss = 0.0
        self.validation_loss = 0.0
        self.confu_matrix = ""
        self.classifier ,  self.cv = Model_Initlizer()
        self.plotter = Plotter()
        self.Train_Start_Time = ""
        self.Train_Finish_Time = ""



    def Train(self):
        # Performing training
        prepare_Train_Dirs()
        H = DateTime.GetHour(DateTime().Now())
        M = DateTime.GetMinute(DateTime().Now())
        S = DateTime.GetSecond(DateTime().Now())
        self.Train_Start_Time = "Start Training ...  Time : %s : %s : %s" % (H,M,S)
        print(self.Train_Start_Time)
        self.classifier.fit(self.Train_Features, self.Train_labels)

        self.validation_accuracy = cross_val_score(self.classifier, self.DataSet.iloc[:,:-1].values, self.DataSet.iloc[:,-1].values,
                                                                                   scoring='accuracy', cv=self.cv, n_jobs=-1, error_score='raise')
        self.validation_loss = cross_val_score(self.classifier, self.DataSet.iloc[:, :-1].values,
                                                   self.DataSet.iloc[:, -1].values,
                                                   scoring='neg_mean_squared_error', cv=self.cv, n_jobs=-1, error_score='raise')
        self.Prediction_labels = self.classifier.predict(self.Test_Features)

        self.training_accuracy = metrics.accuracy_score(self.Test_labels, self.Prediction_labels) * 100

        # self.training_accuracy = cross_val_score(self.classifier, self.Train_Features, self.Train_labels,
        #                 scoring='accuracy', cv=self.cv, n_jobs=-1, error_score='raise')

        self.training_loss = metrics.hamming_loss(self.Test_labels, self.Prediction_labels) * 100

        # self.training_loss = cross_val_score(self.classifier, self.Train_Features, self.Train_labels,
        #                 scoring='neg_mean_squared_error', cv=self.cv, n_jobs=-1, error_score='raise')

        # self.testing_accuracy = cross_val_score(self.classifier, self.Test_Features, self.Test_labels,
        #                 scoring='accuracy', cv=self.cv, n_jobs=-1, error_score='raise')

        # self.testing_loss = cross_val_score(self.classifier, self.Test_Features, self.Test_labels,
        #                 scoring='neg_mean_squared_error', cv=self.cv, n_jobs=-1, error_score='raise')

        self.confu_matrix = confusion_matrix(self.Test_labels, self.Prediction_labels)
        H = DateTime.GetHour(DateTime().Now())
        M = DateTime.GetMinute(DateTime().Now())
        S = DateTime.GetSecond(DateTime().Now())
        self.Train_Finish_Time = "Finish Training ...  Time : %s : %s : %s" % (H, M, S)
        print(self.Train_Finish_Time)
        #
        # plot_tree(self.classifier.steps[2][1])
        # plt.show()


    def VisualizeData(self):
        # ---------------------------------- Data Visualization ------------------------------------

        self.plotter.plotData(self.Train_Features[:, 0:2], self.Train_labels, self.classes, self.Train_classes_names, 10, ' Train Data', 0.5)
        self.plotter.plotData(self.Test_Features[:, 0:2], self.Test_labels, self.classes, self.Test_classes_names, 10, ' Test Data ', 0.5)
        self.plotter.show_pie(self.Train_classes_names)
        self.plotter.show_pie(self.Test_classes_names, ' Test Data ')
        self.plotter.plot_conf_matriex(self.confu_matrix, self.training_accuracy, self.Train_classes_names)
        fpr, tpr, roc_auc = calc_ROC_data(self.Test_labels, self. Prediction_labels, len(self.classes))
        print("TPR : ",tpr)
        print("fpr : ", fpr)
        print("roc_auc : ", roc_auc)
        self.plotter.plot_ROC_curve(fpr, tpr, roc_auc, list(self.Train_classes_names.keys()), Constants.MODEL_NAME)

    def SaveReport(self):
        # ------------------------------------------------------------------------

        file = open(Constants.OUTPUT_PATH+Constants.SESSION_NAME+"\\"+Constants.MODEL_NAME+Constants.REPORT_NAME,"w")
        file.writelines(" ----------- Classification Report --------------\n")
        file.writelines("\n%s \n%s \n\n" % ( str(self.Train_Start_Time) , str( self.Train_Finish_Time)) )
        file.writelines("\nTraining Accuracy: %1.1f%%" % self.training_accuracy)
        file.writelines("\nTraining loss: %1.1f%%" % self.training_loss)
        file.writelines("\nValidation Accuracy: %1.1f%% (+/- %1.1f%%)" % ( np.mean(self.validation_accuracy) * 100,
                                                                           np.std(self.validation_accuracy) * 100))
        self.validation_loss = np.mean(-self.validation_loss) * 100
        file.writelines("\nValidation loss: %1.1f%%" % self.validation_loss)

        file.writelines('\nconfusion_matrix : \n'+ str(self.confu_matrix))
        file.writelines('\nClassification Report : \n' + str(classification_report(self.Test_labels, self.Prediction_labels)))
        file.writelines('\n\n-------------------------------------------------\n')
        file.close()

        print(' Report Saved at:  '+os.path.abspath(Constants.OUTPUT_PATH+Constants.SESSION_NAME))

        increaseExecutionNums()


