

from Trainer import Trainer as main_model

from Utils.Constants import Constants , MODELS
if __name__ == '__main__' :

    # for model in MODELS:
    #     print(model)
    #     Constants.MODEL_NAME = MODELS[model]
        model = main_model()
        model.Train()
        model.VisualizeData()
        model.SaveReport()