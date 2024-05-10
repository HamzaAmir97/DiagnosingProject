import numpy as np
import matplotlib.pyplot as plt
from Utils import Model_Utils
from Utils.Constants import Constants


class Plotter:

    def plotData(self, Features, labels, classes,
                 classes_cat, point_size, separation_name  ,
                 alpha=1.0):

        classes_cat = list(classes_cat.keys())
        classes_cat.reverse()
        for class_ in classes:
            pos = (labels == class_).ravel()
            for feature in range(0, int(Features.shape[1] / 2), 2):
                plt.scatter(Features[pos, feature],
                            Features[pos, feature + 1],
                            s=point_size, linewidths=1,
                            alpha=alpha,
                            label=str(classes_cat.pop()))
        plt.legend()
        plt.title("( "+separation_name + ") Data_Features")
        filename = str(Constants.OUTPUT_PATH) + str(Constants.SESSION_NAME) + \
                   "/DataFaetures_" + str(Constants.MODEL_NAME) + \
                   separation_name + ".png"
        plt.savefig(filename)
        plt.show()

    def plot_conf_matriex(self, data, training_accuracy, classes):
        classes = list(classes.keys())
        classes.insert(0, 0)
        plt.imshow(data, cmap='hot', aspect='auto')
        plt.axes().set(xlabel='Predicted Class', ylabel='True Class')
        plt.axes().set_xticklabels(classes)
        plt.axes().set_yticklabels(classes)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.colorbar()
        plt.title("%s\nTraining Accuracy = %1.2f%%" %  (Constants.MODEL_NAME + "_Confusion_Materiex" , training_accuracy))
        filename = str(Constants.OUTPUT_PATH) + str(Constants.SESSION_NAME) + "/Confusion_Materix_" + str(
            Constants.MODEL_NAME) + ".png"
        plt.savefig(filename)
        plt.show()

    def show_pie(self, classes_names , separation_name = ' Train Data'):
        total = 0
        for i in classes_names.values(): total+=i
        plt.pie(classes_names.values(), labels=classes_names.keys(), autopct='%1.1f%%', shadow=True,
                     explode=np.full(len(classes_names.keys()), fill_value=0.1))
        plt.title("("+ separation_name+") Data_Categories" + "(" +str(total) + ")")
        filename = str(Constants.OUTPUT_PATH) + str(Constants.SESSION_NAME) + "/Data_Description_" + str(
            Constants.MODEL_NAME) + separation_name + "( " +str(total) + " )"+  ".png"
        plt.savefig(filename)
        plt.show()


    def plot_ROC_curve(self , fpr, tpr, roc_auc, classes , model_name , lw=1.5):
        n_classes = len(classes)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr["macro"], tpr["macro"], lw=lw,
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='gray', linestyle=':')

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(classes[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='#726d6d')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(model_name+"_ROC_Curve")
        plt.legend(loc="lower right")
        Model_Utils.save_ROC_curve_plot(plt)
        plt.show()
