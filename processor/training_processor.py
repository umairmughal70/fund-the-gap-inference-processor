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

import pickle
from utils.mssqlutil import MSSQLUtil
import db.constants as pr_constant

class TrainingProcessorUtil:
    """
    Utility class for loading and processing data from database
    """
    __instance  = None

    @staticmethod
    def getInstance():
        if TrainingProcessorUtil.__instance == None:
            TrainingProcessorUtil()
        return TrainingProcessorUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the file loader isntance
        """
        if TrainingProcessorUtil.__instance == None:
            TrainingProcessorUtil.__instance = self
            self.__run()

    def __run(self):
        """
        Load configurations
        """
        instance = ConfigUtil.getInstance()
        self.__envConfiguration = instance.configJSON
        print(self.__envConfiguration)
        print(self.__envConfiguration['db.host'])
        logging.info("training processor inititated...")
 
    def init_train(self, day, fearures_df):
        try:
            logging.info('Triggering Training')
            if date.today().day == day:
                logging.info('Initiating Training on {0}'.format(date.today()))
   
                obj = FundtheGap_using_XGBoost(fearures_df)
                self.store_model(obj.train_model())
                
                obj.evaluate_model()

                
            else:
                logging.info('Training not initiated on {0}'.format(date.today()))
                return

        except Exception as e:
            logging.error(e)
    def store_model(self,model_obj):
        try:
            logging.info('Storing Trained Model')
            model_name = 'FTG_Model_1.pkl'
            self.base_directory = self.__envConfiguration['dir.modelBaseLocation']
            self.save_directory = os.path.join(self.base_directory, self.__envConfiguration['dir.modelName'])

            self.save_directory = os.path.join(self.save_directory,model_name)
            # self.save_directory = os.path.join(Path(os.path.dirname(__file__)).parent,self.__envConfiguration['dir.modelBaseLocation'],model_name)
        
            # self.save_directory = os.path.join(Path(os.path.dirname(__file__)).parent, '\\data',model_name)
            
            pickle.dump(model_obj, open(self.save_directory, 'wb'))
            # pickle.dump(model_obj, open(model_name, 'wb'))
        except Exception as e:
            logging.error(e)

 
 
    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)
            


