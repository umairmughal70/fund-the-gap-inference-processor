## All Necessary imports
import pandas as pd
import numpy as np
import os

# sklearn to perform ml modelling
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from utils.logging_init              import logging

from utils.configloader import ConfigUtil
from utils.file_loader import FileLoaderUtil




import warnings
warnings.filterwarnings('ignore')

class FundtheGap_using_XGBoost():

    def __init__(self, features_df):
        try:
            instance = ConfigUtil.getInstance()
            self.__envConfiguration = instance.configJSON

            """ The default constructor of the class where all class attributes are properly initiallized """
            
            self.base_directory = self.__envConfiguration['dir.modelBaseLocation']
            self.save_directory = os.path.join(self.base_directory, self.__envConfiguration['dir.modelName'])

            self.features = ["SegmentID","Credit","Debit","Balance","Prev_Mean_Bal","Age","Week_1_Avg_Credit","Week_2_Avg_Credit","Week_3_Avg_Credit","Week_4_Avg_Credit","Week_1_Avg_Debit","Week_2_Avg_Debit","Week_3_Avg_Debit","Week_4_Avg_Debit","Monthly_Count_Median_Credit","Monthly_Count_Median_Debit ","Monthly_Count_Variance_Credit","Monthly_Count_Variance_Debit"]
            self.target = ["Struggle_shifted"]
            self.identifiers = ["CustomerID", "ProductCode", "Date"]

            # self.dataset = self.retrieve_data_from_db()
            self.dataset = features_df
            self.normalize_data()
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset()
            self.meta_data = {}
            self.meta_data['features'] = self.features
            self.meta_data['target'] = self.target
            self.meta_data['identifiers'] = self.identifiers
            self.meta_data['scaler'] = self.scaler

                        
            fileObj = FileLoaderUtil()
            fileObj.writeFile( self.meta_data, self.save_directory, 'meta_data' )

        except Exception as e:
            logging.error(e)
        
        
    def retrieve_data_from_db(self):
        logging.info("Retrieved Data from Database")
        # # features = pd.read_csv(r"C:\Users\NoumanNusrallah\Desktop\latest_features.csv")
        # # dataset  = pd.read_csv(r"C:\Users\NoumanNusrallah\Desktop\final_df.csv")
        # dataset['Struggle_shifted'] = dataset['Struggle'].shift(-1).where(dataset.CustomerID.eq(dataset.CustomerID.shift(-1)))
        # dataset["Struggle_shifted"] = dataset["Struggle_shifted"].fillna(0).astype(int)
        # dataset = pd.merge(dataset, features[["CustomerID","Age","Week_1_Avg_Credit","Week_2_Avg_Credit","Week_3_Avg_Credit","Week_4_Avg_Credit","Week_1_Avg_Debit","Week_2_Avg_Debit","Week_3_Avg_Debit","Week_4_Avg_Debit","Monthly_Count_Median_Credit","Monthly_Count_Median_Debit","Monthly_Count_Variance_Credit","Monthly_Count_Variance_Debit"]], on = "CustomerID", how = "left")

        return 1
    
    def normalize_data(self):
        try:
            x = self.dataset[self.features].values #returns a numpy array
            # min_max_scaler = preprocessing.MinMaxScaler()
            # x_scaled = min_max_scaler.fit_transform(x)
            # x_scaled = np.log(x)
            self.scaler = preprocessing.RobustScaler().fit(x)
            x_scaled = self.scaler.transform(x)
            df_scaled = pd.DataFrame(x_scaled, columns=self.features )
            
            df_scaled[df_scaled.isnull()] = -1
            df_scaled.replace([np.inf, -np.inf], -1, inplace=True)
            df_scaled = pd.concat((self.dataset[self.identifiers],df_scaled),1)
            df_scaled = pd.concat((df_scaled,self.dataset[self.target]),1)
            
            logging.info("Normalized the data")
            return df_scaled
        except Exception as e:
            logging.error(e)
        
    def split_dataset(self):
        try:
            seed = 42
            test_size = 0.2
            X_train, X_test, y_train, y_test = train_test_split(self.dataset[self.features], self.dataset[self.target], stratify = self.dataset[self.target], test_size=test_size, random_state=seed)
            logging.info("Shape of Train Data : {}".format( X_train.shape))
            logging.info("Shape of Test Data : {}".format(X_test.shape))

            logging.info("Splitted the data into test-train splits.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)

    def train_model(self):
        try:
            logging.info("Model Training Started")
            param_test = {
            'max_depth':[4,5],
            # 'min_child_weight':[2,4,5,6],
            # 'learning_rate':[0.1,0.01,0.05],
            # 'n_estimators':[100,10,1000],
            # 'objective':['binary:logistic','binary:hinge']
            }
            gsearch = GridSearchCV(estimator = xgb.XGBClassifier(
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread=4,
                seed=27),
                                param_grid = param_test, scoring='roc_auc', cv=5)

            self.train_model = gsearch.fit(self.X_train, self.y_train)
            logging.info("Training Completed")
            return self.train_model
        except Exception as e:
            logging.error(e)
        
    def evaluate_model(self):
        try:
            logging.info("Model Evaluation Started")
            y_pred = self.train_model.predict(self.X_test)
            logging.info("Accuracy for Fully Tuned Model: {}".format((accuracy_score(self.y_test, y_pred) * 100)))
            logging.info("F1-Score for Fully Tuned Model: {}".format((f1_score(self.y_test,  y_pred))))
        except Exception as e:
            logging.error(e)



    