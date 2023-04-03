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

TEST_SET_SIZE = 0.3 # 1 - TEST_SET_SIZE (70%) - training set, TEST_SET_SIZE (30%) testing set
VAL_SET_SIZE = 0.5  # 50% of testing set, it means     
                    # TEST_SET_SIZE / 2 (15%) - testing set, TEST_SET_SIZE / 2 (15%) validation set

def swap_target(x):
    if x == 0:
        return 1
    else:
        return 0
    
class DatasetSplittingType(Enum):
    kFOLD= 0
    TRAIN_TEST = 1
    TRAIN_VAL_TEST = 2

class HemorrageDataset:
    def __init__(self, diagnoseCsvPath, folderPath):
        self.__diagnoseCSV = pd.read_csv(diagnoseCsvPath)
        self.__pathToWholeFolder = folderPath
        self.__trainDataForLoading = []
        self.__trainLabelsForLoading = []
        self.__testDataForLoading = []
        self.__testLabelsForLoading = []
        self.__valDataForLoading = []
        self.__valLabelsForLoading = []
        self.__kFoldDataForLoading = []
        self.__kFoldLabelsForLoading = []
      
    def __kFoldSplitting(self, k):
        pass
    
    def __subsetSplitting(self, sickCases, healthyCases):
        trainHealthyPatientsNumbers, testHealthyPatientsNumbers = train_test_split(healthyCases, test_size=TEST_SET_SIZE,random_state=25, shuffle=True)
        testHealthyPatientsNumbers, valHealthyPatientsNumbers = train_test_split(testHealthyPatientsNumbers, test_size=VAL_SET_SIZE,random_state=25, shuffle=True)
        trainSickPatientsNumbers, testSickPatientsNumbers = train_test_split(sickCases, test_size=TEST_SET_SIZE,random_state=25, shuffle=True)
        testSickPatientsNumbers, valSickPatientsNumbers = train_test_split(testSickPatientsNumbers, test_size=VAL_SET_SIZE,random_state=25, shuffle=True)
        
        trainCases = trainHealthyPatientsNumbers + trainSickPatientsNumbers 
        trainCases = random.sample(trainCases, len(trainCases))
        testCases = testHealthyPatientsNumbers + testSickPatientsNumbers
        testCases = random.sample(testCases, len(testCases))
        valCases = valHealthyPatientsNumbers + valSickPatientsNumbers
        valCases = random.sample(valCases, len(valCases))
        return trainCases, testCases, valCases
    
    def __distinquishHealthyAndSickCases(self):
        sickCases = []
        healthyCases = []
        for patientNum in np.unique(self.__diagnoseCSV['PatientNumber']):
            isSick = self.__diagnoseCSV[(self.__diagnoseCSV['PatientNumber'] == patientNum)].Has_Hemorrhage.sum()
            if isSick > 0:
                sickCases.append(patientNum)
            else:
                healthyCases.append(patientNum)
                
        return healthyCases, sickCases

    def __prepareDataSavingPatienNumberAndSlice(self, chosenSet):
        data = []
        labels = []
        for patientNum in chosenSet:
            for sliceNum in np.unique(self.__diagnoseCSV.loc[(self.__diagnoseCSV['PatientNumber'] == patientNum)]['SliceNumber']):
                diagnose = self.__diagnoseCSV.loc[(self.__diagnoseCSV['PatientNumber'] == patientNum) 
                                        & (self.__diagnoseCSV['SliceNumber'] == sliceNum)]['Has_Hemorrhage'].values[0]
                data.append((patientNum, sliceNum))
                labels.append(diagnose)
                
        return data, labels
    
    def splitDatasetBasedOnPatientsCases(self, splittingType, kFold = 0):
         healthyPatientsNumbers, sickPatientsNumbers = self.__distinquishHealthyAndSickCases()
         
         if(splittingType == DatasetSplittingType.kFOLD):
             self.__kFoldSplitting(kFold)
         elif ((splittingType == DatasetSplittingType.TRAIN_VAL_TEST) or (splittingType == DatasetSplittingType.TRAIN_TEST)):
             trainSubset, testSubset, valSubset = self.__subsetSplitting(sickPatientsNumbers, healthyPatientsNumbers) 
             self.__trainDataForLoading, self.__trainLabelsForLoading = self.__prepareDataSavingPatienNumberAndSlice(trainSubset)
             self.__testDataForLoading, self.__testLabelsForLoading = self.__prepareDataSavingPatienNumberAndSlice(testSubset)
             self.__valDataForLoading, self.__valLabelsForLoading = self.__prepareDataSavingPatienNumberAndSlice(valSubset)
             
             if(splittingType == DatasetSplittingType.TRAIN_TEST):
                 self.__testDataForLoading += self.__valDataForLoading
                 self.__testLabelsForLoading += self.__valLabelsForLoading
                 self.__valDataForLoading = []
                 self.__valLabelsForLoading= []
                         
    def removeRecordFromDataset(self, patientNum, sliceNumber):
        # Remove corrupted images or with missing brain/bone windowed images
        index_to_drop = self.__diagnoseCSV[(self.__diagnoseCSV['PatientNumber'] == patientNum) & (self.__diagnoseCSV['SliceNumber'] == sliceNumber)].index
        index_to_drop = index_to_drop[0]
        self.__diagnoseCSV = self.__diagnoseCSV.drop(index_to_drop, axis=0)  
        
        
    def invertBinaryValues(self, colToChangeAndRemove, newCol):
        self.__diagnoseCSV[newCol] = self.__diagnoseCSV[colToChangeAndRemove].apply(swap_target)
        self.__diagnoseCSV = self.__diagnoseCSV.drop(colToChangeAndRemove, axis=1)    
    
    def get_trainDataWithLabels(self):
        return self.__trainDataForLoading, self.__trainLabelsForLoading

    def get_testDataWithLabels(self):
        return self.__testDataForLoading, self.__testLabelsForLoading
    
    def get_valDataWithLabels(self):
        return self.__valDataForLoading, self.__valLabelsForLoading
    
    def get_kFoldDataWithLabels(self):
        return self.__kFoldDataForLoading, self.__kFoldLabelsForLoading
             
# Create class for ml method - load images, preprocess, choose method and fit model 
# another class for statistic and visualization -> acces private property: self._Parent__private(), self.__demographyCSV = pd.read_csv(demographyCsvPath)
# generator for all
# for imbalanced dataset: https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/

#########################################         
basePath  = r'D:\Brain_JPG_Hemmorage\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0'
csvPath = basePath + '\hemorrhage_diagnosis.csv'
dataset = HemorrageDataset(csvPath, basePath) 
# Prepare csv file  
dataset.removeRecordFromDataset(84, 36)
dataset.invertBinaryValues('No_Hemorrhage', 'Has_Hemorrhage')
# Split dataset using chosen method
dataset.splitDatasetBasedOnPatientsCases(DatasetSplittingType.TRAIN_VAL_TEST)
trainData, trainLabels = dataset.get_trainDataWithLabels()
testData, testLabels = dataset.get_testDataWithLabels()
valData, valLabels = dataset.get_valDataWithLabels()



    