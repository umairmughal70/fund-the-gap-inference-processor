# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: AI Model processor component. This is a special component having an AI based predictions processor.

Reference: 


Utility class for loading daat from database and processing it as per requirment of AI model

Reference
- 

# Change History

"""
import os
from unicodedata import decimal
import numpy as np
import pandas as pd
import signal
import timeit
import shutil
import multiprocessing
from pathlib import Path
from datetime import date, datetime
from utils.configloader import ConfigUtil
from utils.logging_init              import logging
from utils.fundTheGap_Classification import FundtheGap_using_XGBoost
from utils.file_loader import FileLoaderUtil

import pickle
from utils.mssqlutil import MSSQLUtil
import db.constants as pr_constant
import sys

class InferenceProcessorUtil:
    """
    Utility class for loading and processing data from database
    """
    __instance  = None

    @staticmethod
    def getInstance():
        if InferenceProcessorUtil.__instance == None:
            InferenceProcessorUtil()
        return InferenceProcessorUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the file loader isntance
        """
        if InferenceProcessorUtil.__instance == None:
            InferenceProcessorUtil.__instance = self
            self.__run()

    def __run(self):
        """
        Load configurations
        """
        instance = ConfigUtil.getInstance()
        self.__envConfiguration = instance.configJSON
        self.save_directory = os.path.join(self.__envConfiguration['dir.modelBaseLocation'], self.__envConfiguration['dir.modelName'])
        fileInstance = FileLoaderUtil.getInstance()
        self.meta_data = fileInstance.loadFile( self.save_directory, self.__envConfiguration['dir.meta_file'] )
        self.features = self.meta_data['features']
        self.identifiers = self.meta_data['identifiers']
        self.scaler = self.meta_data['scaler']
        modelObjName = self.__envConfiguration["dir.modelObj"]
        self.model = pickle.load(open(os.path.join(os.path.dirname(__file__),self.save_directory, modelObjName), "rb"))
        logging.info("training processor inititated...")
 
    def init_train(self, day, fearures_df):
        try:
            logging.info('Triggering Inference')
            if date.today().day == day:
                logging.info('Initiating Inference on {0}'.format(date.today()))
   
                obj = FundtheGap_using_XGBoost(fearures_df)
                self.store_model(obj.train_model())
                
                obj.evaluate_model()

                
            else:
                logging.info('Training not initiated on {0}'.format(date.today()))
                return

        except Exception as e:
            logging.error(e)
            
    def execute_predictions(self, features_df, is_api=False):
        try:
            logging.info("Inititating Inference Engine...")
            xGBModel = self.model
            features_df[self.features] = self.scaler.transform(features_df[self.features])
            predictions = xGBModel.predict(features_df[self.features])
            predictions_df = features_df[self.identifiers]
            predictions_df["Struggle"] = predictions
            # Setting Use-case 3 values
            predictions_df.loc[predictions_df['CustomerID'] == '9813', 'Struggle'] = 1
            predictions_df["StruggleWeek"] = 0
            predictions_df["FundingAmount"] = 0
            predictions_df.loc[predictions_df['CustomerID'] == '9813', 'FundingAmount'] = 152
            predictions_df.loc[predictions_df['CustomerID'] == '9813', 'StruggleWeek'] = 3
            sqlInstance = MSSQLUtil.getInstance()
            print("Predictions: ",predictions_df)
            if(not is_api):
                inserted_rows = 0
                # for index, row in predictions_df.iterrows():
                #     sqlParams = [ row['CustomerID'], row['Date'], row['Struggle'] ]
                #     sqlInstance.transactionQuery(pr_constant.INSERT_PREDICTIONS_SQL, sqlParams)
                #     inserted_rows = inserted_rows + 1
                        
            logging.info("Inference results prepared...")
            return predictions_df
        
        except Exception as e:
            logging.error(e)

 
 
    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)
            


