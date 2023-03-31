# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:25:23 2023

@author: Agnieszka Florkowska
"""
import os
import pandas as pd
import numpy as np
import random
from enum import Enum
from sklearn.model_selection import train_test_split

def swap_target(x):
    if x == 0:
        return 1
    else:
        return 0
    
class SetType(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class HemorrageDataset:
    def __init__(self, csvPath, folderPath):
        self.CSV = pd.read_csv(csvPath)
        self.pathToWholeFolder = folderPath
        self.healthyCases = []
        self.sickCases = []
        self.trainCasesNumbers = []
        self.testCasesNumbers = []
        self.valCasesNumbers = []
    
    def removeRecordFromDataset(self, patientNum, sliceNumber):
        # Remove corrupted images or with missing brain/bone windowed images
        index_to_drop = self.CSV[(self.CSV['PatientNumber'] == patientNum) & (self.CSV['SliceNumber'] == sliceNumber)].index
        index_to_drop = index_to_drop[0]
        self.CSV = self.CSV.drop(index_to_drop, axis=0)  
        
    def invertBinaryValues(self, colToChangeAndRemove, newCol):
        self.CSV[newCol] = self.CSV[colToChangeAndRemove].apply(swap_target)
        self.CSV = self.CSV.drop(colToChangeAndRemove, axis=1)
     
    def distinquishHealthyAndSickCases(self):
        for patientNum in np.unique(self.CSV['PatientNumber']):
            isSick = self.CSV[(self.CSV['PatientNumber'] == patientNum)].Has_Hemorrhage.sum()
            if isSick > 0:
                self.sickCases.append(patientNum)
            else:
                self.healthyCases.append(patientNum)
        
    def splitDatasetBasedOnPatientsCases(self):
        #add parameter whether split to 2 or 3 sets and kfold
        distinquishHealthyAndSickCases()
        
        trainHealthyPatientsNumbers, testHealthyPatientsNumbers = train_test_split(self.healthyCases, test_size=0.3,random_state=25, shuffle=True)
        trainSickPatientsNumbers, testSickPatientsNumbers= train_test_split(self.sickCases, test_size=0.3,random_state=25, shuffle=True)
        
        trainCases = trainHealthyPatientsNumbers + trainSickPatientsNumbers
        self.trainCasesNumbers = random.sample(trainCases, len(trainCases))
        testCases = testHealthyPatientsNumbers + testSickPatientsNumbers
        self.testCasesNumbers = random.sample(testCases, len(testCases))
        # self.valCasesNumbers
    
    # add method for k-fold cross validation - list of lists
     # __ means private
    def prepareDataForLoading(self, setType):
        if setType == SetType.TRAIN:
            chosenSet = self.trainCasesNumbers
        elif setType == SetType.TEST:
            chosenSet = self.testCasesNumbers
        elif setType == SetType.VAL:
            chosenSet = self.valCasesNumbers
        
            #return list of lists for k-fold
        data = []
        labels = []
        for patientNum in chosenSet:
            for sliceNum in np.unique(self.CSV.loc[(self.CSV['PatientNumber'] == patientNum)]['SliceNumber']):
                diagnose = self.CSV.loc[(self.CSV['PatientNumber'] == patientNum) & (self.CSV['SliceNumber'] == sliceNum)]['Has_Hemorrhage'].values[0]
                data.append((patientNum, sliceNum))
                labels.append(diagnose)
                
        return data, labels

        
# Create class for ml method - load images, preprocess, choose method and fit model 
# generator for all
# for imbalanced dataset: https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/

 #########################################       
# Prepare csv file    
basePath  = r'D:\Brain_JPG_Hemmorage\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0'
csvPath = basePath + '\hemorrhage_diagnosis.csv'

dataset = HemorrageDataset(csvPath, basePath) 
dataset.removeRecordFromDataset(84, 36)
dataset.invertBinaryValues('No_Hemorrhage', 'Has_Hemorrhage')




    