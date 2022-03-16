import os
import random
import glob
import itertools
import sys
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import matplotlib.animation as animation

# in case it is called from a different location
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import data_loader
import models

from termcolor import colored

import commentjson
from pydoc import locate





    # is there a config in the current directory?

def get_config(config_path="config.json"):
    with open('ModelData/' + config_path) as f:
        config = commentjson.load(f)
    return config


class ModelData:
    def __init__(self, model, hero_feature_indicies,label_indicies ):
        self.model = model
        self.hero_feature_indicies = hero_feature_indicies
        self.label_indicies = label_indicies


def load_pytorch_model(modelPath,config,data):


    get_feature_indicies_fn = locate(config["feature_set"])
    get_label_indicies_fn = locate(config["lable_set"])

    
    hero_feature_indicies = get_feature_indicies_fn(data)
    label_indicies = get_label_indicies_fn(data,config["label_set_arg"])


    model_type = locate(config["model"])
    inputFeatureSize = len(hero_feature_indicies[0])
    outputFeatureSize = len(label_indicies)
    model = model_type(inputFeatureSize,outputFeatureSize,**config["model_params"])

    if config["optimizer"] == "Adam":
        OptimizerType = torch.optim.Adam
    elif config["optimizer"] == "SGD":
        OptimizerType = torch.optim.SGD

    optimizer = OptimizerType

    model.load_state_dict(torch.load(modelPath))
    print('loaded model')


    return ModelData(model,hero_feature_indicies,label_indicies) 



heroStuff = []
heroStuffWindow = []
labelStuff = []

def modelPred(model,X):
    predX = model(X)
    predX = torch.sigmoid(predX)
    predX = predX.cpu().detach().numpy()
    return predX

def averagePred(models,X):
    vals =[]
    for m in models:
        vals = modelPred(m,X)+ vals

    return (vals / len(models))


def make_predictions(testingFiles):
    #glob.glob("/scratch/staff/ak1774/shared_folder/data/train/*.h5")
    data = data_loader.load_data_from_file(testingFiles[0])

    models = []
    for i in range(1,2):
        print(i)
        models.append( load_pytorch_model('ModelData/' +str(i) +'/' +'model.model',
                            get_config('/' +str(i) +'/config.json'), data) )
    currentTruePosAccuracy = 0
    currentFalsePosAccuracy=0
    currentFalseNegAccuracy = 0
    currentTrueNegAccuracy =0


    accuracy_valuesTotal =0
    averageCounter =0
    certaintyTruePos =0
    certaintyFalsePos=0
    certaintyFalseNeg=0

    clampValue = 0.5


    for test in testingFiles:
        data = data_loader.load_data_from_file(test)


        gameData,fullGameLabels = data_loader.getSequencialNaive(data,models[0].hero_feature_indicies,models[0].label_indicies)
        print(len(gameData[0]))
        print('Loaded game')

        for i in range(0,len(gameData[0])):
            predX = 0
            for m in models:
                y = fullGameLabels[i]
                y = np.array(y)
                y = np.expand_dims(y,0)

                X = [torch.from_numpy(hero_X[i:(i+1),:]) for hero_X in gameData]
            
                predX = modelPred(m.model,X) +predX


            predX = predX/len(models)

            accuracy_values = ((predX > 0.5) == (y > 0.5)).astype(np.float32)
            accuracy_valuesTotal += accuracy_values.mean()

            for playerPred in range(0,len(np.squeeze(predX,0))):
                if predX.reshape(-1)[playerPred] >= clampValue and y.reshape(-1)[playerPred] > clampValue:
                    certaintyTruePos += predX.reshape(-1)[playerPred] 
                    currentTruePosAccuracy +=1
                elif predX.reshape(-1)[playerPred] > clampValue and y.reshape(-1)[playerPred] < clampValue:
                    certaintyFalsePos += predX.reshape(-1)[playerPred] 
                    currentFalsePosAccuracy +=1
                elif predX.reshape(-1)[playerPred] < clampValue and y.reshape(-1)[playerPred] > clampValue:
                    certaintyFalseNeg += predX.reshape(-1)[playerPred] 
                    currentFalseNegAccuracy +=1
                elif predX.reshape(-1)[playerPred] < clampValue and y.reshape(-1)[playerPred] < clampValue:
                    currentTrueNegAccuracy+=1

        print('Processed game')
        averageCounter += len(gameData[0])
        print()
        print('True Pos = ' + str(currentTruePosAccuracy))
        print('False pos = ' + str(currentFalsePosAccuracy))
        print('False neg = ' + str(currentFalseNegAccuracy))
        print('True neg = ' + str(currentTrueNegAccuracy))
        print()
        print('Precison = '  + str (currentTruePosAccuracy/ (currentTruePosAccuracy+currentFalsePosAccuracy) ))
        print('Recall = ' +  str ( currentTruePosAccuracy/ (currentTruePosAccuracy+currentFalseNegAccuracy)  ))
        print('----------------')
        print('Accuracy ' + str(accuracy_valuesTotal/averageCounter))
        print('Combiend game length ' + str(averageCounter))
        print()
        


    print()
    print('True Pos = ' + str(currentTruePosAccuracy))
    print('True pos certainty= ' + str(np.sum(certaintyTruePos)))
    print()
    print('False pos = ' + str(currentFalsePosAccuracy))
    print('False pos certainty = ' + str(np.sum(certaintyFalsePos)))
    print()
    print('False neg = ' + str(currentFalseNegAccuracy))
    print('False neg certainty = ' + str(np.sum(certaintyFalseNeg)))
    print()
    print('True neg = ' + str(currentTrueNegAccuracy))
    print()
    print('Precison = '  + str (currentTruePosAccuracy/ (currentTruePosAccuracy+currentFalsePosAccuracy) ))
    print('Recall = ' +  str ( currentTruePosAccuracy/ (currentTruePosAccuracy+currentFalseNegAccuracy)  ))
    print('Accuracy ' + str(accuracy_valuesTotal/averageCounter))
    print('Combiend game length ' + str(averageCounter))


if __name__ == "__main__":

    matches = glob.glob("Matches/*.h5")
    print(matches)
    make_predictions(matches)









