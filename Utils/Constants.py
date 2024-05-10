
"""
This File Includes All  Constants Of The Project
"""
import os
from shutil import rmtree

MODELS = {"SVM" : "Linear SVM" , "RBFSVM" : "RBF_SVM" ,
                        "DT" : "Decision Tree Classifier" , "RF" : "Random Forest" ,
                        "ETC" : "Extra Tree Classifier" , "KNN" : "K-Nearest Neighbor (KNN)"}
FEATURES = {"Colon": "best5_colon_(1,2,3)_224_224.csv" ,
                            "Lung_Colon": "best5_numFeatures74_lung_colon_224_224.mat" ,
                            "Colon_Lung_Colon": "Lung_colon_and_colon_224_4k_images.mat"}
class Constants:
    TRAIN_SIZE = 0.70
    INPUT_PATH = r"Inputs\\"
    FEATURES_NAME_MAT = FEATURES["Colon_Lung_Colon"]
    FEATURES_NAME_CSV = FEATURES["Colon"]
    MODEL_NAME = MODELS['RF']
    DATASET_PATH = str(INPUT_PATH ) + r"DataSet\\"
    FEATURES_PATH_MAT = str(INPUT_PATH) + r"Features\\" + str(FEATURES_NAME_MAT)
    FEATURES_PATH_CSV = str(INPUT_PATH) + r"Features\\" + str(FEATURES_NAME_CSV)
    OUTPUT_PATH = r"OutPut\\"
    EXEC_FILE = "config.cnfg"
    if os.path.exists(EXEC_FILE) == False:
        rmtree(OUTPUT_PATH,True)
        open(EXEC_FILE, "w").write("1")
    EXEC_NUMS =  int(open(EXEC_FILE, "r").readline())
    SESSION_NAME = MODEL_NAME + " Execution_"+str(EXEC_NUMS)
    REPORT_NAME = "Training_Report.txt"
